"""World model, first version (platform B stage 1, on the PC): from the diary, learn to predict the next tick of
what the robot senses - vision, hearing, body - from what it senses now and what it does now.

The gate of stage 1 is to predict better than "as it was a tick ago" (persistence). This script trains a small GRU
on the diary days and reports, per group and on held-out time, the skill against persistence:
skill = 1 - error(model) / error(persistence)  (0 = no better, 1 = perfect, negative = worse).

  ./py ../life/train_world.py [days ...] [--epochs 20]       (from tools/hybrid; default: every day in the diary)

The network has the policy's shape (dense 64 -> GRU 64 -> dense 48 -> out, INT8 on the robot), so the robot runs it
with the same kernels: main/hybrid/world_model.* computes, every tick, the error of the prediction per group and
its fall (learning progress = curiosity) for the diary. The last epochs train with the INT8 rounding (fake quant).
Checkpoint tools/hybrid/data/world_net.pt, header main/hybrid/world_model_data.h (and the weights blob, pack_weights).
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lifedata as L  # noqa: E402
from models import PolicyNet  # noqa: E402
from spec import ACTION, AUDIO, BODY, EFF, POLICY_INPUT, VISION  # noqa: E402

GROUPS = {"vision": VISION, "audio": AUDIO, "body": BODY}
TARGET = [L.col(n) for g in GROUPS.values() for n in g]
INPUT = [L.col(n) for n in POLICY_INPUT] + [L.col("act_" + a) for a in ACTION] + [L.col(e) for e in EFF]
# the v3 spiking core's section is off in mode life (2026-10-17): zeroed in every day, old ones too, so the world model
# does not lean on it (its 25 inputs stay in the vector: the robot's world model reads the diary's first 154 values)
from spec import SNN as _SNN  # noqa: E402
ZERO = [i for i, c in enumerate(INPUT) if L.COLUMNS[c] in set(_SNN)]
SEQ = 200  # ticks per training window (10 s)


def episodes(days):
    """Runs of consecutive ticks (same boot, no gap) of at least SEQ + 1 ticks: (inputs [T, I], targets [T, O])."""
    out = []
    for day in days:
        ticks, _ = L.load(day)
        if ticks is None:
            continue
        for b in np.unique(ticks["boot"]):
            sel = np.nonzero(ticks["boot"] == b)[0]
            sel = sel[np.argsort(ticks["seq"][sel])]
            seq = ticks["seq"][sel].astype(np.int64)
            cuts = np.nonzero(np.diff(seq) != 1)[0] + 1
            for part in np.split(sel, cuts):
                if len(part) > SEQ + 1:
                    v = np.nan_to_num(ticks["values"][part])
                    x = v[:, INPUT].copy()
                    x[:, ZERO] = 0.0
                    out.append((x, v[:, TARGET]))
    return out


def WorldNet(n_in, n_out):
    return PolicyNet(n_in, n_out)


def export(model, path):
    from nn import Header
    h = Header("world_model", True)
    h.layer("inp", model.inp.weight, model.inp.bias)
    h.layer("gru_i", model.gru.weight_ih_l0, model.gru.bias_ih_l0)
    h.layer("gru_h", model.gru.weight_hh_l0, model.gru.bias_hh_l0)
    h.layer("fc1", model.fc1.weight, model.fc1.bias)
    h.layer("fc2", model.fc2.weight, model.fc2.bias)
    h.write(path)


def windows(eps, split):
    """Cuts every episode at `split` of its length: training windows from the first part, validation from the rest
    (held-out time, not shuffled ticks)."""
    tr, va = [], []
    for x, y in eps:
        cut = int(len(x) * split)
        for part, dest in (((x[:cut], y[:cut]), tr), ((x[cut:], y[cut:]), va)):
            px, py = part
            for s in range(0, len(px) - SEQ - 1, SEQ):
                # input at t, target = change from t to t + 1
                dest.append((px[s:s + SEQ], py[s + 1:s + SEQ + 1] - py[s:s + SEQ]))
    return tr, va


def skill(model, data, dev, batch=256):
    """Per group: 1 - MSE(model) / MSE(persistence), over all held-out windows (in batches: a week does not fit the GPU
    at once)."""
    if not data:
        return {}
    err_p = np.zeros(len(GROUPS))
    err_m = np.zeros(len(GROUPS))
    model.eval()
    for i in range(0, len(data), batch):
        x = torch.tensor(np.stack([d[0] for d in data[i:i + batch]]), device=dev)
        dy = torch.tensor(np.stack([d[1] for d in data[i:i + batch]]), device=dev)
        with torch.no_grad():
            pred, _ = model(x)
        k = 0
        for j, names in enumerate(GROUPS.values()):
            sl = slice(k, k + len(names))
            k += len(names)
            err_p[j] += (dy[..., sl] ** 2).sum().item()
            err_m[j] += ((pred[..., sl] - dy[..., sl]) ** 2).sum().item()
    return {g: 1 - err_m[j] / max(err_p[j], 1e-12) for j, g in enumerate(GROUPS)}


def fit(model, tr, va, epochs, quant_epochs, dev, lr=2e-3, log=print):
    """Trains on the training windows (the last quant_epochs with INT8 rounding); prints held-out skill now and then."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(np.stack([d[0] for d in tr]))
    Y = torch.tensor(np.stack([d[1] for d in tr]))
    for ep in range(epochs):
        model.quant = ep >= epochs - quant_epochs
        model.train()
        perm = torch.randperm(len(X))
        total = 0.0
        for i in range(0, len(X), 64):
            b = perm[i:i + 64]
            pred, _ = model(X[b].to(dev))
            loss = ((pred - Y[b].to(dev)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        if log and (ep % 5 == 4 or ep == epochs - 1):
            s = skill(model, va, dev)
            log(f"  epoch {ep + 1}: train loss {total / len(X):.6f}; held-out skill vs persistence "
                + ", ".join(f"{g} {v:+.3f}" for g, v in s.items()))
    model.eval()
    return model


def all_days():
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(L.ROOT, "data", "life", "20*")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("days", nargs="*")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--quant-epochs", type=int, default=10, help="the last epochs with INT8 rounding")
    ap.add_argument("--split", type=float, default=0.8)
    a = ap.parse_args()
    days = a.days or all_days()
    eps = episodes(days)
    tr, va = windows(eps, a.split)
    print(f"{len(days)} day(s), {len(eps)} runs, {sum(len(e[0]) for e in eps) / 72000:.2f} h; "
          f"{len(tr)} training / {len(va)} held-out windows of {SEQ} ticks")
    if not tr or not va:
        sys.exit("not enough diary yet")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = WorldNet(len(INPUT), len(TARGET)).to(dev)
    fit(model, tr, va, a.epochs, a.quant_epochs, dev, log=lambda m: print(m, flush=True))
    path = os.path.join(L.ROOT, "tools", "hybrid", "data", "world_net.pt")
    torch.save(model.state_dict(), path)
    header = os.path.join(L.ROOT, "main", "hybrid", "world_model_data.h")
    export(model.cpu(), header)
    print(f"saved {path}, {header}")


if __name__ == "__main__":
    main()
