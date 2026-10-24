"""The first actor learned from the robot's own experience (platform B stage 3, offline, on the PC).

Advantage-weighted regression (Peng et al. 2019): imitate the actions in the diary, each weighted by exp(A / beta),
where A is the critic's advantage (train_critic.py) - do more of what turned out better than expected, less of what
turned out worse. The collector's babbling (random actions) is ideal data for it: every action was tried in many
situations.

To see what the reward taught, the same network is also trained with all weights 1 (plain imitation of the babbling):
the difference between the two is the learning. Checked on held-out time:
  - after a touch: does it call (sound_act)?
  - a person in view: does its head target point at them?
  - how often does it start a move / hold still?

  ./py ../life/train_actor.py [days ...] [--epochs 20] [--beta 1.0]       (from tools/hybrid)
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lifedata as L  # noqa: E402
import temperament as T  # noqa: E402
import train_critic as C  # noqa: E402
import train_world as W  # noqa: E402

# the decisions learned here: where the head goes, how fast, whether it holds still, whether it calls
ACTS = ["act_yaw", "act_pitch", "act_speed", "act_hold", "act_sound_act"]
BINARY = {"act_hold", "act_sound_act"}
A_IDX = [L.col(a) for a in ACTS]
SEQ = C.SEQ


class Actor(nn.Module):
    def __init__(self, n_in, n_out, hidden=64):
        super().__init__()
        self.inp = nn.Linear(n_in, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_out)

    def forward(self, x, h=None):
        y, h = self.gru(torch.relu(self.inp(x)), h)
        return self.out(y), h


def advantages(critic, v, scale, dev):
    """GAE over one run (scaled reward units), normalised later."""
    vals, h = [], None
    with torch.no_grad():  # in chunks: a run can be many hours long (cuDNN refuses one call that long)
        for i in range(0, len(v), 20000):
            x = torch.tensor(v[i:i + 20000, C.STATE].astype(np.float32), device=dev)[None]
            out, h = critic(x, h)
            vals.append(out[0].cpu().numpy())
    val = np.concatenate(vals)
    r = T.reward(v) / scale
    adv = np.zeros(len(v), np.float32)
    last = 0.0
    for t in range(len(v) - 2, -1, -1):
        delta = r[t + 1] + C.GAMMA * val[t + 1] - val[t]  # reward of the step t -> t+1 is recorded at t+1
        last = delta + C.GAMMA * C.LAMBDA * last
        adv[t] = last
    return adv


def windows(eps, advs, split):
    tr, va = [], []
    for (v, flags), adv in zip(eps, advs):
        x = v[:, C.STATE].astype(np.float32)
        y = v[:, A_IDX].astype(np.float32)
        coll = (flags & 1).astype(np.float32)  # only the collector's ticks teach (random, every action tried)
        cut = int(len(x) * split)
        for (a, b), dest in (((0, cut), tr), ((cut, len(x)), va)):
            for s in range(a, b - SEQ, SEQ):
                dest.append((x[s:s + SEQ], y[s:s + SEQ], adv[s:s + SEQ], coll[s:s + SEQ], v[s:s + SEQ]))
    return tr, va


def loss_fn(out, y, w):
    loss = 0.0
    for j, name in enumerate(ACTS):
        if name in BINARY:
            l = F.binary_cross_entropy_with_logits(out[..., j], (y[..., j] > 0.5).float(), reduction="none")
        else:
            l = (torch.tanh(out[..., j]) - y[..., j]) ** 2 if name in ("act_yaw", "act_pitch") else \
                (torch.sigmoid(out[..., j]) - y[..., j]) ** 2
        loss = loss + (l * w).sum() / w.sum().clamp(min=1.0)
    return loss


def train(tr, weighted, beta, epochs, dev):
    torch.manual_seed(0)
    model = Actor(len(C.STATE), len(ACTS)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X = torch.tensor(np.stack([d[0] for d in tr]))
    Y = torch.tensor(np.stack([d[1] for d in tr]))
    A = torch.tensor(np.stack([d[2] for d in tr]))
    M = torch.tensor(np.stack([d[3] for d in tr]))
    Wt = torch.clamp(torch.exp(A / beta), max=20.0) * M if weighted else M
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            b = perm[i:i + 64]
            out, _ = model(X[b].to(dev))
            loss = loss_fn(out, Y[b].to(dev), Wt[b].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def probe(model, va, dev):
    """What the policy does in held-out situations."""
    x = torch.tensor(np.stack([d[0] for d in va]), device=dev)
    with torch.no_grad():
        out, _ = model(x)
    out = out.cpu().numpy()
    v = np.stack([d[4] for d in va])
    call = 1 / (1 + np.exp(-out[..., ACTS.index("act_sound_act")]))
    hold = 1 / (1 + np.exp(-out[..., ACTS.index("act_hold")]))
    yaw_t = np.tanh(out[..., ACTS.index("act_yaw")]) * 100.0
    touch = np.maximum(v[..., L.col("touch_head")], v[..., L.col("touch_screen")]) > 0.5
    onset = touch & ~np.concatenate([np.zeros_like(touch[:, :1]), touch[:, :-1]], axis=1)
    after = np.zeros_like(onset)
    for k in range(1, 11):
        after[:, k:] |= onset[:, :-k]
    person = (v[..., L.col("person")] > 0.5) & (v[..., L.col("target_conf")] > 0.5)
    where = v[..., L.col("yaw")] * 100.0 + v[..., L.col("target_x")] * 30.0
    err_person = np.abs(yaw_t - where)[person].mean() if person.any() else float("nan")
    err_random = np.abs(yaw_t - (v[..., L.col("yaw")] * 100.0 + np.random.default_rng(0).uniform(-30, 30, yaw_t.shape))
                        )[person].mean() if person.any() else float("nan")
    return {
        "call after touch": call[after].mean() if after.any() else float("nan"),
        "call otherwise": call[~after].mean(),
        "head target off the person (deg)": err_person,
        "  (a random target would be)": err_random,
        "holds still": hold.mean(),
        "touches seen": int(onset.sum()),
        "person ticks": int(person.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("days", nargs="*")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--beta", type=float, default=1.0)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(L.ROOT, "tools", "hybrid", "data", "critic_net.pt"), map_location=dev)
    critic = C.Critic(len(C.STATE)).to(dev)
    critic.load_state_dict(ck["state"])
    critic.eval()
    eps = C.runs(a.days or W.all_days())
    advs = [advantages(critic, v, ck["scale"], dev) for v, _ in eps]
    allv = np.concatenate(advs)
    advs = [(x - allv.mean()) / (allv.std() + 1e-8) for x in advs]
    tr, va = windows(eps, advs, 0.8)
    print(f"{sum(len(v) for v, _ in eps) / 72000:.1f} h, {len(tr)} / {len(va)} windows")
    plain = train(tr, False, a.beta, a.epochs, dev)
    awr = train(tr, True, a.beta, a.epochs, dev)
    p0, p1 = probe(plain, va, dev), probe(awr, va, dev)
    print(f"{'held out':36s} {'imitation':>10s} {'reward (AWR)':>13s}")
    for k in p0:
        print(f"{k:36s} {p0[k]:10.3f} {p1[k]:13.3f}")
    torch.save(awr.state_dict(), os.path.join(L.ROOT, "tools", "hybrid", "data", "actor_awr.pt"))


if __name__ == "__main__":
    main()
