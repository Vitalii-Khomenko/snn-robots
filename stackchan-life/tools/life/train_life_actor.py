"""The actor of platform B (stage 3): the decisions the collector made at random, learned from the robot's own diary
and weighted by how they turned out (advantage-weighted regression, the critic of train_critic.py).

The collector's decisions are events, not levels: "start a move now, to there, this fast" and "call now". A network
that regressed the head target every tick would learn the mean of random targets - straight ahead - and the head
would freeze. So the actor gives, every tick:

  move    logit of starting a move in this tick
  yaw     mean and log-std of the target (-1..1 of the executor's window)
  pitch   mean and log-std of the target (-1..1)
  speed   0..1
  call    logit of starting a call in this tick

and the robot samples from it (main/behavior/collector.h with an actor: the rest - faces, voice colour, LEDs, every
touch answered, quiet nights - stays the collector's). Trained only on the collector's ticks, where every decision
was random; each tick weighted by exp(A / beta). Exported like the policy (dense 64 -> GRU 64 -> dense 48 -> 7, INT8):
main/hybrid/actor_model_data.h and the weights blob (actor.*).

  ./py ../life/train_life_actor.py [days ...] [--epochs 15] [--beta 0.5]       (from tools/hybrid)
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lifedata as L  # noqa: E402
import train_actor as A  # noqa: E402 - advantages()
import train_critic as C  # noqa: E402
import train_world as W  # noqa: E402
from models import PolicyNet  # noqa: E402

OUT = ["move", "yaw_mu", "yaw_logsig", "pitch_mu", "pitch_logsig", "speed", "call", "value"]
# expression (2026-10-17): which face when the face changes, which voice style when a call starts - the collector chose
# them at random, so the diary shows what each one led to (main/behavior/collector.cpp kDayFaces, styles)
FACES = ["normal", "happy", "curious", "thinking", "cool", "wink", "confused", "surprised", "bored", "very_happy", "shy",
         "love"]
STYLES = ["calm", "ask", "play", "grumble"]
OUT = OUT + ["face_" + f for f in FACES] + ["style_" + x for x in STYLES]
F0, S0 = 8, 8 + len(FACES)
SEQ = 400


def labels(v):
    hold = v[:, L.col("act_hold")] > 0.6
    move = np.zeros(len(v), np.float32)
    move[1:] = (~hold[1:] & hold[:-1]).astype(np.float32)
    snd = v[:, L.col("act_sound_act")] > 0.5
    call = np.zeros(len(v), np.float32)
    call[1:] = (snd[1:] & ~snd[:-1]).astype(np.float32)
    # A call in the 2 s after a touch is the collector's rule (every touch is answered), not a decision - and it falls
    # on the touch's big reward, so the actor took the credit and learned to call 10 times a minute (2026-10-15).
    # Those ticks do not teach the call at all (label -1 = masked).
    touch = np.maximum(v[:, L.col("touch_head")], v[:, L.col("touch_screen")]) > 0.5
    after = np.convolve(touch.astype(float), np.ones(40))[:len(v)] > 0
    call[after] = -1.0
    # the face chosen when it changed (not the touch answer: happy/love for 4 s after a touch), -1 = no choice here
    emo = np.stack([v[:, L.col("act_emo_" + f)] for f in FACES], 1)
    face_now = np.where(emo.max(1) > 0.5, emo.argmax(1), -1)
    face = np.full(len(v), -1.0, np.float32)
    changed = np.r_[False, (face_now[1:] != face_now[:-1]) & (face_now[1:] >= 0)]
    after4 = np.convolve(touch.astype(float), np.ones(80))[:len(v)] > 0
    face[changed & ~after4] = face_now[changed & ~after4]
    # the style of a call that the actor (or the collector's timer) started, not a touch answer
    sty = np.stack([v[:, L.col("act_style_" + x)] for x in STYLES], 1).argmax(1)
    style = np.full(len(v), -1.0, np.float32)
    style[call > 0] = sty[call > 0]
    return np.stack([move, v[:, L.col("act_yaw")], v[:, L.col("act_pitch")], v[:, L.col("act_speed")], call, face,
                     style], 1)


def windows(eps, advs, split, scale):
    import temperament as T
    tr, va = [], []
    for (v, flags), adv in zip(eps, advs):
        x = v[:, C.STATE].astype(np.float32)
        g = (C.discounted(T.reward(v)) / scale).astype(np.float32)
        lab = labels(v)
        y = np.concatenate([lab[:, :5], g[:, None], lab[:, 5:]], 1).astype(np.float32)  # ..., call, value, face, style
        m = ((flags & 3) != 0).astype(np.float32)  # the collector's random decisions and the actor's sampled ones
        cut = int(len(x) * split)
        for (a, b), dest in (((0, cut), tr), ((cut, len(x)), va)):
            for s in range(a, b - SEQ, SEQ):
                dest.append((x[s:s + SEQ], y[s:s + SEQ], adv[s:s + SEQ], m[s:s + SEQ], v[s:s + SEQ]))
    return tr, va


def loss_fn(out, y, w):
    move, call = y[..., 0], y[..., 4]
    ev = move * w  # the target and the speed only where a move started
    wsum = w.sum().clamp(min=1.0)
    l_move = (F.binary_cross_entropy_with_logits(out[..., 0], move, reduction="none") * w).sum() / wsum
    cm = (call >= 0).float() * w
    l_call = (F.binary_cross_entropy_with_logits(out[..., 6], call.clamp(min=0), reduction="none") * cm).sum() / \
        cm.sum().clamp(min=1.0)
    nll = 0.0
    for j_mu, j_ls, k in ((1, 2, 1), (3, 4, 2)):
        mu = torch.tanh(out[..., j_mu])
        ls = out[..., j_ls].clamp(-3.0, 1.0)
        nll = nll + (((y[..., k] - mu) ** 2) / (2 * torch.exp(2 * ls)) + ls) * ev
    l_speed = (torch.sigmoid(out[..., 5]) - y[..., 3]) ** 2 * ev
    esum = ev.sum().clamp(min=1.0)
    # the value of the situation (the robot's TD learning starts from it): every tick after the first 5 s of a window
    # (GRU warm-up) and before its last 10 s (returns truncated there)
    l_value = ((out[..., 7] - y[..., 5])[:, 100:-200] ** 2).mean()
    # expression: cross-entropy where a face / a style was chosen, weighted like the rest
    def choice(lo, n, col):
        lab = y[..., col]
        m = (lab >= 0).float() * w
        ce = F.cross_entropy(out[..., lo:lo + n].reshape(-1, n), lab.clamp(min=0).long().reshape(-1), reduction="none")
        return (ce * m.reshape(-1)).sum() / m.sum().clamp(min=1.0)
    return l_move + l_call + (nll.sum() + l_speed.sum()) / esum + 0.5 * l_value + choice(F0, len(FACES), 6) + \
        choice(S0, len(STYLES), 7)


def train(tr, weighted, beta, epochs, dev, quant_epochs=3):
    torch.manual_seed(0)
    model = PolicyNet(len(C.STATE), len(OUT)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X = torch.tensor(np.stack([d[0] for d in tr]))
    Y = torch.tensor(np.stack([d[1] for d in tr]))
    Av = torch.tensor(np.stack([d[2] for d in tr]))
    M = torch.tensor(np.stack([d[3] for d in tr]))
    Wt = torch.clamp(torch.exp(Av / beta), max=20.0) * M if weighted else M
    for ep in range(epochs):
        model.quant = ep >= epochs - quant_epochs
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            b = perm[i:i + 64]
            out, _ = model(X[b].to(dev))
            loss = loss_fn(out, Y[b].to(dev), Wt[b].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    model.quant = True
    return model


def probe(model, va, dev):
    outs = []
    with torch.no_grad():  # in batches: several days do not fit the GPU at once
        for i in range(0, len(va), 256):
            out, _ = model(torch.tensor(np.stack([d[0] for d in va[i:i + 256]]), device=dev))
            outs.append(out.cpu().numpy())
    out = np.concatenate(outs)
    v = np.stack([d[4] for d in va])
    pm = 1 / (1 + np.exp(-out[..., 0]))
    pc = 1 / (1 + np.exp(-out[..., 6]))
    touch = np.maximum(v[..., L.col("touch_head")], v[..., L.col("touch_screen")]) > 0.5
    person = (v[..., L.col("person")] > 0.5) & (v[..., L.col("target_conf")] > 0.5)
    yaw_t = np.tanh(out[..., 1]) * 100.0
    where = v[..., L.col("yaw")] * 100.0 + v[..., L.col("target_x")] * 30.0
    night = v[..., L.col("night")] > 0.5
    fp = np.exp(out[..., F0:F0 + len(FACES)])
    fp = (fp / fp.sum(-1, keepdims=True)).reshape(-1, len(FACES)).mean(0)
    sp = np.exp(out[..., S0:S0 + len(STYLES)])
    sp = (sp / sp.sum(-1, keepdims=True)).reshape(-1, len(STYLES)).mean(0)
    print("  faces: " + " ".join(f"{f} {p:.2f}" for f, p in sorted(zip(FACES, fp), key=lambda z: -z[1])[:6]) +
          " | styles: " + " ".join(f"{x} {p:.2f}" for x, p in zip(STYLES, sp)))
    g = np.stack([d[1][:, 5] for d in va])
    ev = 1 - ((out[..., 7] - g)[:, 100:-200] ** 2).mean() / g[:, 100:-200].var()
    return {
        "value: explained variance": ev,
        "moves / min (day)": pm[~night].mean() * 1200,
        "moves / min, a person in view": pm[person & ~night].mean() * 1200 if (person & ~night).any() else float("nan"),
        "calls / min (day)": pc[~night].mean() * 1200,
        "calls / min, touched": pc[touch].mean() * 1200 if touch.any() else float("nan"),
        "target off the person (deg)": np.abs(yaw_t - where)[person].mean() if person.any() else float("nan"),
        "target spread (yaw std, deg)": (np.exp(out[..., 2].clip(-3, 1)) * 100).mean(),
    }


def features(model, x, dev):
    """The actor's hidden layer (after fc1, ReLU) - what the robot's own value head and its online learning see."""
    with torch.no_grad():
        xt = torch.tensor(x, device=dev)
        d = F.relu(F.linear(xt, model.qw(model.inp.weight), model.inp.bias))
        out, _ = torch._VF.gru(d, d.new_zeros(1, d.shape[0], 64), [model.qw(model.gru.weight_ih_l0),
                               model.qw(model.gru.weight_hh_l0), model.gru.bias_ih_l0, model.gru.bias_hh_l0],
                               True, 1, 0.0, False, False, True)
        return F.relu(F.linear(out, model.qw(model.fc1.weight), model.fc1.bias)).cpu().numpy()


def value_head(model, windows_, scale, dev):
    """The robot's value of a situation, V = w . f + b on the actor's hidden layer, fitted (ridge) to the discounted
    return (scaled by the critic's scale): the starting point of the robot's online TD learning (stage 4)."""
    import temperament as T
    fs, gs = [], []
    for d in windows_[: 3000]:
        g = C.discounted(T.reward(d[4])) / scale
        fs.append(features(model, d[0][None], dev)[0][100:-200])  # GRU warmed up; returns not truncated
        gs.append(g[100:-200])
    f = np.concatenate(fs)
    g = np.concatenate(gs)
    fa = np.concatenate([f, np.ones((len(f), 1), f.dtype)], 1)
    w = np.linalg.solve(fa.T @ fa + 1.0 * np.eye(fa.shape[1]), fa.T @ g)
    ev = 1 - ((fa @ w - g) ** 2).mean() / g.var()
    return w[:-1].astype(np.float32), float(w[-1]), float(ev)


def export(model, path):
    from nn import Header
    h = Header("actor_model", True)
    h.layer("inp", model.inp.weight, model.inp.bias)
    h.layer("gru_i", model.gru.weight_ih_l0, model.gru.bias_ih_l0)
    h.layer("gru_h", model.gru.weight_hh_l0, model.gru.bias_hh_l0)
    h.layer("fc1", model.fc1.weight, model.fc1.bias)
    h.layer("fc2", model.fc2.weight, model.fc2.bias)
    h.write(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("days", nargs="*")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--out-header", default=os.path.join(L.ROOT, "main", "hybrid", "actor_model_data.h"))
    ap.add_argument("--out-pt", default=os.path.join(L.ROOT, "tools", "hybrid", "data", "life_actor.pt"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(L.ROOT, "tools", "hybrid", "data", "critic_net.pt"), map_location=dev)
    critic = C.Critic(len(C.STATE)).to(dev)
    critic.load_state_dict(ck["state"])
    critic.eval()
    eps = C.runs(a.days or W.all_days())
    advs = [A.advantages(critic, v, ck["scale"], dev) for v, _ in eps]
    allv = np.concatenate(advs)
    advs = [(x - allv.mean()) / (allv.std() + 1e-8) for x in advs]
    tr, va = windows(eps, advs, 0.8, ck["scale"])
    print(f"{sum(len(v) for v, _ in eps) / 72000:.1f} h, {len(tr)} / {len(va)} windows")
    plain = train(tr, False, a.beta, a.epochs, dev)
    awr = train(tr, True, a.beta, a.epochs, dev)
    p0, p1 = probe(plain, va, dev), probe(awr, va, dev)
    print(f"{'held out':34s} {'imitation':>10s} {'reward (AWR)':>13s}")
    for k in p0:
        print(f"{k:34s} {p0[k]:10.2f} {p1[k]:13.2f}")
    import json
    print("PROBE " + json.dumps({k: float(v) for k, v in p1.items()}))
    torch.save(awr.state_dict(), a.out_pt)
    export(awr.cpu(), a.out_header)
    print(f"exported {a.out_header}")


if __name__ == "__main__":
    main()
