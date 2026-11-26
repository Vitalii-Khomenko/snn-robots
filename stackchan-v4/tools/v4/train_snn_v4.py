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
N_SYN, N_REAL = 1600, 800
N_VAL_SYN, N_VAL_REAL = 100, 100  # held out: synthetic lives at random, real ones all from the newest day (split())


def real_sequences(n, rng, n_held=N_VAL_REAL):
    """Stretches of the robot's real brain-v4 life (2026-10-25 onward: real als, ps, motion, voice, touch).
    Half of the training stretches are centred on active motion/touch/light/voice events so attention (att_x/att_y)
    and rare salience channels (sal_touch_head, sal_light) get rich real-world signal. n_held stretches come from
    the newest full day and are never trained on."""
    import diary as D
    from calibrate import raw_hour
    from channels import encode, new_state
    all_v4 = sorted(d for d in os.listdir(D.LIFE) if d[:2] == "20" and d >= "2026-10-25")
    # Pick up to 14 evenly-spaced v4 days (excluding today's partial day) so we cover diverse rooms/lighting/habits
    cand_days = all_v4[:-1] if len(all_v4) > 2 else all_v4
    if len(cand_days) > 14:
        pick = np.linspace(0, len(cand_days) - 1, 14).round().astype(int)
        days = [cand_days[i] for i in sorted(set(pick))]
    else:
        days = cand_days
    pool = []
    for day in days:
        st = new_state()
        X = []
        for h in range(24):
            raw, _ = raw_hour(day, h)
            if raw is None:
                continue
            T = len(raw["energy"])
            if "als" not in raw:
                raw["als"] = np.full(T, rng.uniform(1, 40))
                raw["als_ok"] = np.ones(T)
            X.append(encode(raw, st))
        if X:
            arr = np.concatenate(X)
            if len(arr) > SEQ + 1500:
                pool.append(arr)
                print(f"real {day} ({len(arr) / 72000:.1f} h)", flush=True)
    # Precompute active-event indices per day (motion, touch_head, light_spot, voice)
    ch_motion = CHANNELS.index("motion")
    ch_touch_h = CHANNELS.index("touch_head")
    ch_voice = CHANNELS.index("voice")
    wh_light_spot = len(CHANNELS) + 4
    active_idx = []
    for d in pool:
        lo, hi = 1300, len(d) - SEQ
        sub = d[lo:hi]
        act = np.flatnonzero((sub[:, ch_motion] > 0.06) | (sub[:, ch_touch_h] > 0.04) |
                             (sub[:, wh_light_spot] > 0.15) | (sub[:, ch_voice] > 0.25)) + lo
        active_idx.append(act)
    out, day = [], []
    for i in range(n):
        k = len(pool) - 1 if i < n_held else int(rng.integers(len(pool) - 1))
        d = pool[k]
        act = active_idx[k]
        if i >= n_held and len(act) > 0 and rng.random() < 0.55:
            center = int(act[rng.integers(len(act))])
            s = int(np.clip(center - int(rng.integers(SEQ // 4, 3 * SEQ // 4)), 1300, len(d) - SEQ))
        else:
            s = int(rng.integers(1300, len(d) - SEQ))
        out.append(d[s:s + SEQ])
        day.append(k)
    return out, day


def build_real():
    """Extract only real v4 diary sequences into compact float16 out/snn_real.npz (for fast upload to Colab T4)."""
    rng = np.random.default_rng(7)
    seqs, days = real_sequences(N_REAL, rng)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "snn_real.npz")
    np.savez_compressed(path, X_real=np.stack(seqs).astype(np.float16), day_real=np.array(days, np.int16))
    print(f"wrote {path}: {len(seqs)} real sequences ({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)


def build(real_npz=None):
    rng = np.random.default_rng(7)
    X_syn = [make_channels(rng, SEQ) for _ in range(N_SYN)]
    if real_npz is not None and os.path.exists(real_npz):
        z = np.load(real_npz)
        X_real = z["X_real"].astype(np.float32)
        days = z["day_real"].astype(np.int16)
        X = np.concatenate([np.stack(X_syn).astype(np.float32), X_real], axis=0)
    else:
        seqs, days = real_sequences(N_REAL, rng)
        X = np.stack(X_syn + seqs).astype(np.float32)
        days = np.array(days, np.int16)
    Y = np.stack([teach(x) for x in X]).astype(np.float32)
    real = np.zeros(len(X), bool)
    real[N_SYN:] = True
    day = np.full(len(X), -1, np.int16)
    day[N_SYN:] = days
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "snn_data.npz"), X=X, Y=Y, real=real, day=day)
    print("dataset", X.shape, Y.shape, flush=True)


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


def split(real, day, rng, n_syn=N_VAL_SYN):
    """-> (held-out, training) indices: synthetic lives at random (they are independent), the real stretches by day -
    every one of the newest day held out, none of it trained on. A dataset built before 2026-10-28 has no days: a
    random split, as it was."""
    if day is None:
        idx = rng.permutation(len(real))
        return idx[:120], idx[120:]
    syn = rng.permutation(np.flatnonzero(~real))
    held_day = day[real].max()
    val = np.concatenate([syn[:n_syn], np.flatnonzero(real & (day == held_day))])
    tr = np.concatenate([syn[n_syn:], np.flatnonzero(real & (day != held_day))])
    return val, rng.permutation(tr)


def train(epochs, init_ckpt=None, batch_size=256, lr=None):
    d = np.load(os.path.join(OUT, "snn_data.npz"))
    X, Y, real = d["X"], d["Y"], d["real"]
    rng = np.random.default_rng(0)
    val, tr = split(real, d["day"] if "day" in d else None, rng)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.tensor(np.maximum(Y[tr, WARM:].reshape(-1, Y.shape[2]).std(0), 0.03), device=dev)
    # rare, important numbers count more where they happen (including attention coordinates)
    event_w = torch.tensor([3.0 if n in ("threat", "surprise", "over", "att_x", "att_y", "att_strength") or n.startswith("sal_") else 1.0
                            for n in SNN_OUT], device=dev)
    model = SnnV4().to(dev)
    if init_ckpt and os.path.exists(init_ckpt):
        model.load_state_dict(torch.load(init_ckpt, weights_only=False, map_location=dev)["state"])
        print(f"Warm-started SnnV4 from {init_ckpt}", flush=True)
    base_lr = lr if lr is not None else (8e-4 if init_ckpt else 2e-3)
    opt = torch.optim.Adam(model.parameters(), lr=base_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    B = batch_size
    qat_at = max(1, int(epochs * 0.6))
    best = 1e9
    os.makedirs(DATA, exist_ok=True)
    log = []
    for ep in range(epochs):
        model.quant = ep >= qat_at
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
                tot += float(loss.detach())
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
        if model.quant and vloss < best or (ep == qat_at - 1 and not model.quant):
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
    if sys.argv[1] == "real":
        build_real()
    elif sys.argv[1] == "data":
        build(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 35)
