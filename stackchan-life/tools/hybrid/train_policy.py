"""Trains the policy GRU on rollouts (data/policy/it*/ep*/bus.bin): the teacher
labels every logged policy input (imitation; DAgger when the rollouts were
driven by the learner).   ./py train_policy.py [--epochs 30]"""
import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import teacher_policy as tp
from models import PolicyNet
from spec import ACTION, ACTION_SIGNED, SNN

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
from spec import POLICY_INPUT  # noqa: E402

N_IN, N_OUT = len(POLICY_INPUT), len(ACTION)
REC = N_IN + 3 + N_OUT  # policy input, efference copy (SNN only), executed action
SIGNED = torch.tensor([n in ACTION_SIGNED for n in ACTION])

# how much each output matters
W = {n: 1.0 for n in ACTION}
W.update({n: 1.5 for n in ACTION if n.startswith('emo_')})
W.update(yaw=3.0, pitch=3.0, speed=1.0, sound_act=3.0, startle=3.0, invite=8.0, hold=1.5, lids=1.5,
         face_valence=1.5, face_arousal=1.5, explore=1.0)
VOICE = {"voice_register", "voice_slope", "voice_expr", "voice_length", "voice_rate", "style_calm", "style_ask",
         "style_play", "style_grumble", "sound_arousal", "sound_valence"}


def face_boost(ys):
    """rare faces weigh more where they are the label (Love, Shy, Cool were 0.02-0.2 % of the ticks and never won)"""
    emo = [i for i, n in enumerate(ACTION) if n.startswith("emo_")]
    freq = np.concatenate([y[:, emo] for y in ys]).mean(0) + 1e-5
    return dict(zip(emo, np.clip(np.median(freq) / freq, 1.0, 20.0)))


def load_episodes(pattern):
    xs, ys = [], []
    for f in sorted(glob.glob(pattern)):
        d = np.fromfile(f, np.float32)
        d = d[: d.size // REC * REC].reshape(-1, REC)
        if len(d) < 400:
            continue
        x = d[:, :N_IN]
        xs.append(x)
        ys.append(tp.label(x))
    return xs, ys


def batchify(xs, ys, chunk):
    """Equal-length sequences (truncate to a common multiple of chunk)."""
    T = min(len(x) for x in xs) // chunk * chunk
    return np.stack([x[:T] for x in xs]), np.stack([y[:T] for y in ys])


# Steadiness is learned, not imposed by the executor: a change of these outputs from one tick to the next costs, so the
# network holds a face, a gaze and a head target until the situation really changes (its GRU memory is what makes that
# possible; the teacher itself is stateless). Replaces the executor's face hysteresis and most of its hold times.
# gaze / glow: only flicker beyond the teacher's own step counts; faces: every change counts (the teacher itself flips
# between two nearly equal faces - Laughing / VeryHappy at arousal 0.65 - which the executor used to hide with a hysteresis)
# (not the head target: a smoothed target glided and the head crept after it; the executor's saccade generator moves the
# head once per settled target)
STEADY = torch.tensor([n in ("gaze_x", "gaze_y", "led_r", "led_g", "led_b") for n in ACTION])
FACES = torch.tensor([n.startswith("emo_") for n in ACTION])
STEADY_WEIGHT, FACE_WEIGHT = 4.0, 0.5
BOOST = None  # per-output extra weight of a positive label (set from the data: rare faces)


def loss_fn(logits, y, w, vmask):
    sig = SIGNED.to(logits.device)
    out = torch.where(sig, torch.tanh(logits), torch.sigmoid(logits))
    mse = (out - y) ** 2
    # probabilities: cross entropy for the unit-range outputs (sharper pulses)
    bce = F.binary_cross_entropy_with_logits(logits, y.clamp(0, 1), reduction="none")
    base = torch.where(sig, mse, bce * 0.5 + mse)
    if BOOST is not None:  # positive labels of rare faces count more
        base = base * (1.0 + y * BOOST.to(logits.device))
    # flicker that the teacher does not ask for: the step of the output beyond the step of the target
    d_out = (out[:, 1:] - out[:, :-1]).abs()
    d_y = (y[:, 1:] - y[:, :-1]).abs()
    flicker = (torch.relu(d_out - d_y) ** 2 * STEADY.to(logits.device)).mean()
    face_flip = (d_out ** 2 * FACES.to(logits.device)).mean()
    return (base * w * vmask).mean() + STEADY_WEIGHT * flicker + FACE_WEIGHT * face_flip


@torch.no_grad()
def evaluate(model, X, Y, dev, chunk):
    model.eval()
    x, y = torch.as_tensor(X).to(dev), torch.as_tensor(Y).to(dev)
    h = None
    outs = []
    for c in range(0, x.shape[1], chunk):
        o, h = model(x[:, c:c + chunk], h)
        outs.append(o)
    o = torch.cat(outs, 1)
    sig = SIGNED.to(dev)
    p = torch.where(sig, torch.tanh(o), torch.sigmoid(o))
    err = ((p - y) ** 2).mean((0, 1)).sqrt().cpu().numpy()
    return err, p.cpu().numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--qat", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--glob", default=os.path.join(HERE, "data", "policy", "it*", "ep*", "bus.bin"))
    ap.add_argument("--init", default=None)
    a = ap.parse_args()
    xs, ys = load_episodes(a.glob)
    print(len(xs), "episodes", sum(len(x) for x in xs), "ticks")
    boost = face_boost(ys)
    BOOST = torch.zeros(N_OUT)
    for i, b in boost.items():
        BOOST[i] = float(b) - 1.0
    print("face boost:", {ACTION[i][4:]: round(float(b), 1) for i, b in boost.items() if b > 1.5})
    n_val = max(2, len(xs) // 10)
    vx, vy = batchify(xs[:n_val], ys[:n_val], a.chunk)
    X, Y = batchify(xs[n_val:], ys[n_val:], a.chunk)
    dev = torch.device("cuda")
    model = PolicyNet().to(dev)
    if a.init:
        model.load_state_dict(torch.load(a.init))
    w = torch.tensor([W[n] for n in ACTION], device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    n_groups = max(1, len(X) // a.batch)
    chunks = X.shape[1] // a.chunk
    total = (a.epochs + a.qat) * n_groups * chunks
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=total, pct_start=0.1)
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    vidx = [ACTION.index(n) for n in VOICE]
    for ep in range(a.epochs + a.qat):
        model.quant = ep >= a.epochs
        model.train()
        perm = np.random.permutation(len(X))
        agg, cnt, t0 = 0.0, 0, time.time()
        for g in range(n_groups):
            idx = perm[g * a.batch:(g + 1) * a.batch]
            xb, yb = Xt[idx].to(dev), Yt[idx].to(dev)
            xb = xb + torch.randn_like(xb) * 0.01
            h = None
            for c in range(chunks):
                sl = slice(c * a.chunk, (c + 1) * a.chunk)
                logits, h = model(xb[:, sl], h)
                h = h.detach()
                # voice parameters only matter while a call is wanted
                call = (yb[:, sl, ACTION.index("sound_act")] > 0.3).float()[..., None]
                vmask = torch.ones_like(logits)
                vmask[..., vidx] = 0.15 + 0.85 * call
                loss = loss_fn(logits, yb[:, sl], w, vmask)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                sched.step()
                agg += loss.item()
                cnt += 1
        print(f"epoch {ep + 1}/{a.epochs + a.qat} loss {agg / cnt:.4f} quant={model.quant} {time.time() - t0:.0f}s", flush=True)
    err, _ = evaluate(model, vx, vy, dev, a.chunk)
    for n, e in zip(ACTION, err):
        print(f"  {n:15s} rmse {e:.3f}")
    torch.save(model.state_dict(), os.path.join(HERE, "data", "policy_net.pt"))
    model.cpu().export(os.path.join(ROOT, "main", "hybrid", "policy_model_data.h"))
    print("exported")
