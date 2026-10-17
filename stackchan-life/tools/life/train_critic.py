"""The critic (platform B stage 3, on the PC): from the diary, how good is a situation - the discounted reward that
followed it (temperament.py: the reward recomputed from the needs in the diary), gamma 0.995 per 50 ms tick (~10 s
ahead).

It evaluates the behaviour that produced the diary (the collector's babbling, the v3 policy): V(s) of that behaviour.
Its advantages A = r + gamma V(s') - V(s) (GAE, lambda 0.95) are what the actor learns from (train_actor.py:
advantage-weighted regression - do more of what turned out better than expected).

The state is what the robot senses and feels, not the old spiking core's readouts (copies of a hand-written teacher):
vision, hearing, body, time of day, prediction errors, learned expectations, the five needs, the world model's errors.

  ./py ../life/train_critic.py [days ...] [--epochs 30]        (from tools/hybrid)

Reports how much of the variance of the real discounted return on held-out time the critic explains (1 = all, 0 =
no better than the mean). Checkpoint tools/hybrid/data/critic_net.pt.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lifedata as L  # noqa: E402
import temperament as T  # noqa: E402
import train_world as W  # noqa: E402
from spec import AUDIO, BODY, CTX, LEARN, PE, VISION  # noqa: E402

GAMMA, LAMBDA = 0.995, 0.95
STATE_NAMES = list(VISION) + list(AUDIO) + list(BODY) + list(CTX) + list(PE) + list(LEARN) + \
    [n for n in L.MOTIVATION if n.startswith("need_")] + L.WORLD + ["expect_talk"]
STATE = [L.col(n) for n in STATE_NAMES]
SEQ = 400  # 20 s windows


def runs(days, need_motivation=True):
    """Runs of consecutive ticks with the motivation columns: (all values [T, C], flags [T])."""
    out = []
    for day in days:
        ticks, _ = L.load(day)
        if ticks is None:
            continue
        for b in np.unique(ticks["boot"]):
            sel = np.nonzero(ticks["boot"] == b)[0]
            sel = sel[np.argsort(ticks["seq"][sel])]
            cuts = np.nonzero(np.diff(ticks["seq"][sel].astype(np.int64)) != 1)[0] + 1
            for part in np.split(sel, cuts):
                v = ticks["values"][part]
                if need_motivation:
                    ok = ~np.isnan(v[:, L.col("reward")]) & ~np.isnan(v[:, L.col("wm_rms_vision")])
                    part, v = part[ok], v[ok]
                if len(part) > SEQ + 1:
                    out.append((np.nan_to_num(v), ticks["flags"][part]))
    return out


def discounted(r):
    g = np.zeros_like(r)
    acc = 0.0
    for t in range(len(r) - 1, -1, -1):
        acc = r[t] + GAMMA * acc
        g[t] = acc
    return g


class Critic(nn.Module):
    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.inp = nn.Linear(n_in, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, h=None):
        y, h = self.gru(torch.relu(self.inp(x)), h)
        return self.out(y)[..., 0], h


def scale_of(eps):
    """Rewards are small (drive reduction per 50 ms); returns are scaled to unit variance for training."""
    g = np.concatenate([discounted(T.reward(v)) for v, _ in eps])
    return float(g.std() + 1e-8)


def windows(eps, split, scale):
    tr, va = [], []
    for v, _ in eps:
        x = v[:, STATE].astype(np.float32)
        g = (discounted(T.reward(v)) / scale).astype(np.float32)
        cut = int(len(x) * split)
        for (a, b), dest in (((0, cut), tr), ((cut, len(x)), va)):
            # the last 10 s of each part have truncated returns: not used as targets
            for s in range(a, b - SEQ - 200, SEQ):
                dest.append((x[s:s + SEQ], g[s:s + SEQ]))
    return tr, va


def explained(model, data, dev, batch=256):
    """Held-out explained variance of the return (in batches: five days do not fit a 6 GB GPU at once)."""
    vs, gs = [], []
    with torch.no_grad():
        for i in range(0, len(data), batch):
            x = torch.tensor(np.stack([d[0] for d in data[i:i + batch]]), device=dev)
            v, _ = model(x)
            vs.append(v.cpu().numpy())
            gs.append(np.stack([d[1] for d in data[i:i + batch]]))
    v, g = np.concatenate(vs), np.concatenate(gs)
    # the first 5 s of a window: the GRU has not seen enough yet
    return 1 - ((v[:, 100:] - g[:, 100:]) ** 2).mean() / g[:, 100:].var()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("days", nargs="*")
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    days = a.days or W.all_days()
    eps = runs(days)
    scale = scale_of(eps)
    tr, va = windows(eps, 0.8, scale)
    hours = sum(len(v) for v, _ in eps) / 72000
    print(f"{hours:.1f} h with motivation; {len(tr)} / {len(va)} windows; return scale {scale:.4f}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = Critic(len(STATE)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X = torch.tensor(np.stack([d[0] for d in tr]))
    G = torch.tensor(np.stack([d[1] for d in tr]))
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            b = perm[i:i + 64]
            v, _ = model(X[b].to(dev))
            loss = ((v[:, 100:] - G[b].to(dev)[:, 100:]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 5 == 4:
            model.eval()
            print(f"  epoch {ep + 1}: explained variance of the return, held out: {explained(model, va, dev):+.3f}",
                  flush=True)
    torch.save({"state": model.state_dict(), "scale": scale, "names": STATE_NAMES},
               os.path.join(L.ROOT, "tools", "hybrid", "data", "critic_net.pt"))


if __name__ == "__main__":
    main()
