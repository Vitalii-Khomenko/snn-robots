"""Trains the spiking core to embody teacher_drives.py on synthetic sequences.
   ./py train_snn.py [--epochs N]        -> data/snn.pt, main/hybrid/snn_model_data.h"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import torch

import snn_model as M
import synth
import teacher_drives
from spec import SNN

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(HERE, "data", "snn_seq.npz")


def one(seed):
    rng = np.random.default_rng(seed)
    X = synth.make_sequence(rng, 12000)
    return X, teacher_drives.run(X)


REAL = os.path.join(HERE, "data", "snn_real.npz")
T_REAL = 9500  # rollout episodes are 9514 ticks (9600 skipped every one of them: --real trained on nothing)


def _real_one(path):
    from spec import POL_IN, SNN_INPUT, pin, sin
    d = np.fromfile(path, np.float32)
    from spec import ACTION, POLICY_INPUT
    rec = len(POLICY_INPUT) + 3 + len(ACTION)
    d = d[: d.size // rec * rec].reshape(-1, rec)
    if len(d) < T_REAL:
        return None
    d = d[:T_REAL]
    bus = np.zeros((T_REAL, len(SNN_INPUT)), np.float32)
    effs = ["eff_yaw", "eff_pitch", "eff_sound"]
    for n in SNN_INPUT:
        bus[:, sin(n)] = d[:, len(POLICY_INPUT) + effs.index(n)] if n in effs else d[:, pin(n)]
    return bus, teacher_drives.run(bus)


def build_real(limit=240):
    """Bus sequences logged from closed-loop rollouts in the simulator (what the
    core really sees), labelled by the drive teacher."""
    import glob
    files = sorted(glob.glob(os.path.join(HERE, "data", "policy", "it*", "ep*", "bus.bin")))[:limit]
    with Pool(os.cpu_count()) as p:
        out = [o for o in p.map(_real_one, files) if o is not None]
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])


def build_dataset(n, first_seed):
    with Pool(os.cpu_count()) as p:
        out = p.map(one, range(first_seed, first_seed + n))
    X = np.stack([o[0] for o in out])
    Y = np.stack([o[1] for o in out])
    return X, Y


W = None


def loss_fn(core, y_raw, rates, target, std):
    pred = core.decode(y_raw)
    err = ((pred - target) / std) ** 2
    loss = (err * W).mean()
    # keep the core alive but not saturated: mean spike rate per neuron
    r = rates.mean((0, 1))
    reg = (torch.relu(r - 0.10) ** 2).mean() * 30 + (torch.relu(0.01 - r) ** 2).mean() * 100
    return loss + reg, loss.item(), err.mean((0, 1)).detach(), r.mean().item()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--qat", type=int, default=6)
    ap.add_argument("--seqs", type=int, default=192)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--real", action="store_true", help="mix in rollout buses; fine-tune from data/snn.pt")
    ap.add_argument("--lr", type=float, default=3e-3)
    a = ap.parse_args()
    if a.rebuild or not os.path.exists(DATA):
        t0 = time.time()
        X, Y = build_dataset(a.seqs, 100)
        np.savez(DATA, X=X, Y=Y)
        print(f"dataset {X.shape} in {time.time() - t0:.0f}s")
    d = np.load(DATA)
    X, Y = d["X"], d["Y"]
    if a.real:
        if a.rebuild or not os.path.exists(REAL):
            rx, ry = build_real()
            np.savez(REAL, X=rx, Y=ry)
        r = np.load(REAL)
        X = np.concatenate([X[:, :T_REAL], r["X"]])
        Y = np.concatenate([Y[:, :T_REAL], r["Y"]])
        perm0 = np.random.default_rng(0).permutation(len(X))
        X, Y = X[perm0], Y[perm0]
        print("mixed dataset", X.shape)
    n_val = max(a.batch // 2, 8)
    Xv, Yv, X, Y = X[:n_val], Y[:n_val], X[n_val:], Y[n_val:]
    dev = torch.device("cuda")
    std = torch.as_tensor(Y.reshape(-1, Y.shape[-1]).std(0) + 0.05, device=dev)
    core = M.SpikingCore().to(dev)
    if a.real:
        core.load_state_dict(torch.load(os.path.join(HERE, 'data', 'snn.pt')))
        core.quant = False
    W = torch.ones(len(SNN), device=dev)
    for n in ("nov_touch", "tr_touch", "hab_touch", "annoyance", "hab_sound", "nov_aud", "surprise", "boredom", "social"):
        W[SNN.index(n)] = 3.0
    opt = torch.optim.AdamW(core.parameters(), lr=a.lr, weight_decay=1e-5)
    n_groups = len(X) // a.batch
    chunks = X.shape[1] // a.chunk
    total = (a.epochs + a.qat) * n_groups * chunks
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.08)
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    for ep in range(a.epochs + a.qat):
        core.quant = ep >= a.epochs
        perm = np.random.permutation(len(X))
        t0 = time.time()
        agg, rate_agg, cnt = 0.0, 0.0, 0
        for g in range(n_groups):
            idx = perm[g * a.batch:(g + 1) * a.batch]
            xb, yb = Xt[idx].to(dev), Yt[idx].to(dev)
            xb = xb + torch.randn_like(xb) * 0.01
            state = core.init_state(len(idx), dev)
            for c in range(chunks):
                sl = slice(c * a.chunk, (c + 1) * a.chunk)
                y_raw, rates, state = core(xb[:, sl], state)
                loss, l, per, rate = loss_fn(core, y_raw, rates, yb[:, sl], std)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(core.parameters(), 2.0)
                opt.step()
                sched.step()
                state = {k: v.detach() for k, v in state.items()}
                agg += l
                rate_agg += rate
                cnt += 1
        print(f"epoch {ep + 1}/{a.epochs + a.qat} loss {agg / cnt:.4f} rate {rate_agg / cnt:.3f} quant={core.quant} {time.time() - t0:.0f}s", flush=True)
        if ep % 5 == 4 or ep == a.epochs + a.qat - 1:
            torch.save(core.state_dict(), os.path.join(HERE, "data", "snn.pt"))
    # validation
    core.eval()
    with torch.no_grad():
        state = core.init_state(len(Xv), dev)
        xb, yb = torch.as_tensor(Xv).to(dev), torch.as_tensor(Yv).to(dev)
        preds = []
        for c in range(chunks):
            sl = slice(c * a.chunk, (c + 1) * a.chunk)
            y_raw, rates, state = core(xb[:, sl], state)
            preds.append(core.decode(y_raw))
        pred = torch.cat(preds, 1)
        yb = yb[:, : pred.shape[1]]  # whole chunks only (9500-tick rollout sequences are not a multiple of the chunk)
        err = ((pred - yb) ** 2).mean((0, 1)).sqrt().cpu().numpy()
        corr = [np.corrcoef(pred[..., k].flatten().cpu().numpy(), yb[..., k].flatten().cpu().numpy())[0, 1] for k in range(len(SNN))]
    for n, e, c in zip(SNN, err, corr):
        print(f"  {n:13s} rmse {e:.3f} corr {c:.3f}")
    torch.save(core.state_dict(), os.path.join(HERE, 'data', 'snn.pt'))
