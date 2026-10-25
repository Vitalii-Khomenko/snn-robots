"""Trains brain v4's spiking analyzer to embody the innate evaluation (teacher_snn.py). Surrogate gradients,
truncated backpropagation through time over 5-minute lives, int8 quantization-aware in the last epochs.

  ./py train_snn_v4.py data            # build the dataset: synthetic lives + real diary stretches -> out/snn_data.npz
  ./py train_snn_v4.py train [epochs]  # -> data/snn_v4.pt (best on held-out data), report out/snn_report.txt
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
from snn_v4 import SnnV4, detach_state  # noqa: E402
from spec_v4 import CHANNELS, SNN_IN, SNN_OUT  # noqa: E402
from synth_v4 import make_channels  # noqa: E402
from teacher_snn import teach  # noqa: E402

OUT = os.path.join(HERE, "out")
DATA = os.path.join(HERE, "data")
SEQ = 6000          # 5 minutes per life
CHUNK = 100         # 5 s per backprop window
WARM = 100          # the first 5 s of a life only settle the state
N_SYN, N_REAL = 1400, 400


def real_sequences(n, rng):
    """Stretches of the robot's real life, channels computed after the fact from the diary (calibrate.raw_hour). On
    platform B's days light, proximity and motion were not logged: a dark room (the robot's usual), nobody at the
    face, the Vision NN's activity for motion."""
    import diary as D
    from calibrate import raw_hour
    from channels import encode, new_state
    days = sorted(d for d in os.listdir(D.LIFE) if d[:2] == "20")[-9:-1]
    pool = []
    for day in days:
        st = new_state()
        X = []
        for h in range(24):
            raw, _ = raw_hour(day, h)
            if raw is None:
                continue
            T = len(raw["energy"])
            if "als" not in raw:  # platform B's days
                raw["als"] = np.full(T, rng.uniform(1, 40))
                raw["als_ok"] = np.ones(T)
            X.append(encode(raw, st))
        if X:
            pool.append(np.concatenate(X))
        print("real", day, flush=True)
    out = []
    for _ in range(n):
        d = pool[rng.integers(len(pool))]
        s = rng.integers(0, len(d) - SEQ - 1300)
        out.append(d[s + 1300:s + 1300 + SEQ])  # skip a warm-up (tilt baseline) inside the day stream
    return out


def build():
    rng = np.random.default_rng(7)
    X = [make_channels(rng, SEQ) for _ in range(N_SYN)]
    X += real_sequences(N_REAL, rng)
    X = np.stack(X).astype(np.float32)
    Y = np.stack([teach(x) for x in X]).astype(np.float32)
    real = np.zeros(len(X), bool)
    real[N_SYN:] = True
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "snn_data.npz"), X=X, Y=Y, real=real)
    print("dataset", X.shape, Y.shape)


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def evaluate(model, X, Y, dev, bs=200):
    model.eval()
    P = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i:i + bs], device=dev)
            st = model.init_state(len(xb), dev)
            out, _, _ = model(xb, st)
            P.append(out.cpu().numpy())
    model.train()
    P = np.concatenate(P)[:, WARM:]
    T = Y[:, WARM:]
    return P, T


def train(epochs):
    d = np.load(os.path.join(OUT, "snn_data.npz"))
    X, Y, real = d["X"], d["Y"], d["real"]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    n_val = 120
    val, tr = idx[:n_val], idx[n_val:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.tensor(np.maximum(Y[tr, WARM:].reshape(-1, Y.shape[2]).std(0), 0.03), device=dev)
    # rare, important numbers count more where they happen
    event_w = torch.tensor([3.0 if n in ("threat", "surprise", "over") or n.startswith("sal_") else 1.0
                            for n in SNN_OUT], device=dev)
    model = SnnV4().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    B = 96
    best = 1e9
    os.makedirs(DATA, exist_ok=True)
    log = []
    for ep in range(epochs):
        model.quant = ep >= int(epochs * 0.7)
        t0 = time.time()
        perm = rng.permutation(tr)
        tot, nb = 0.0, 0
        for b in range(0, len(perm) - B + 1, B):
            xb = torch.tensor(X[perm[b:b + B]], device=dev)
            yb = torch.tensor(Y[perm[b:b + B]], device=dev)
            st = model.init_state(B, dev)
            for c in range(0, SEQ, CHUNK):
                out, layer, _ = model(xb[:, c:c + CHUNK], st)
                st = detach_state(st)
                if c + CHUNK <= WARM:
                    continue
                tgt = yb[:, c:c + CHUNK]
                w = 1.0 + (event_w - 1.0) * tgt.clamp(0, 1) * 3.0
                loss = (((out - tgt) / sd) ** 2 * w).mean()
                mix_rate = layer[2]
                loss = loss + 20.0 * (torch.relu(mix_rate - 0.12) ** 2 + torch.relu(0.01 - mix_rate) ** 2) * 100
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(loss)
                nb += 1
        sched.step()
        P, T = evaluate(model, X[val], Y[val], dev)
        vloss = float(np.mean(((P - T) / sd.cpu().numpy()) ** 2))
        cs = [corr(P[..., i].ravel(), T[..., i].ravel()) for i in range(len(SNN_OUT))]
        msg = (f"epoch {ep + 1}/{epochs} {'q ' if model.quant else ''}train {tot / max(nb, 1):.3f} val {vloss:.3f} "
               f"corr mean {np.mean(cs):.3f} min {np.min(cs):.3f} ({SNN_OUT[int(np.argmin(cs))]}) "
               f"rates {[round(float(r), 3) for r in layer]} {time.time() - t0:.0f}s")
        print(msg, flush=True)
        log.append(msg)
        if model.quant and vloss < best or (ep == int(epochs * 0.7) - 1 and not model.quant):
            if model.quant:
                best = vloss
            torch.save({"state": model.state_dict(), "epoch": ep + 1, "val": vloss, "corr": cs},
                       os.path.join(DATA, "snn_v4.pt"))
    # final report on the saved network
    ck = torch.load(os.path.join(DATA, "snn_v4.pt"), weights_only=False)
    model.load_state_dict(ck["state"])
    model.quant = True
    lines = [f"brain v4 spiking analyzer: epoch {ck['epoch']}, held-out loss {ck['val']:.3f}", ""]
    for name, sel in (("synthetic", ~real[val]), ("real diary", real[val])):
        P, T = evaluate(model, X[val][sel], Y[val][sel], dev)
        lines.append(f"{name} ({sel.sum()} lives): corr / MAE per output")
        for i, n in enumerate(SNN_OUT):
            lines.append(f"  {n:16s} corr {corr(P[..., i].ravel(), T[..., i].ravel()):+.3f}  "
                         f"mae {np.abs(P[..., i] - T[..., i]).mean():.3f}  teacher sd {T[..., i].std():.3f}")
    text = "\n".join(log + [""] + lines) + "\n"
    open(os.path.join(OUT, "snn_report.txt"), "w").write(text)
    print("\n".join(lines))


if __name__ == "__main__":
    assert len(SNN_IN) == len(CHANNELS) + 5
    if sys.argv[1] == "data":
        build()
    else:
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
