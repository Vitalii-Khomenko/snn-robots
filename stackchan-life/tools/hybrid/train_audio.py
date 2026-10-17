"""Trains the Auditory NN on data/audio/*.bin and exports
main/hybrid/auditory_model_data.h.   ./py train_audio.py"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

from models import AUDIO_IN, AUDIO_OUT, AuditoryNet

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REC = AUDIO_IN + 9  # recorded files: input + 8 classes + a direction label that is no longer used


def floored(d):
    """The same per-band baseline subtraction AuditoryNet::run does, over a time-ordered file."""
    d = d.copy()
    band = d[:, :16].copy()
    fl = band[0].copy()
    lvl = d[:, 97].copy()
    lf = 0.5
    for t in range(len(d)):
        fl = np.minimum(band[t], fl + 0.00003) if t else band[0]
        lf = min(lvl[t], min(lf + 0.00003, 0.5))
        d[t, :96] = np.maximum(0.0, d[t, :96] - np.tile(fl, 6))
        d[t, 97] = max(0.0, lvl[t] - lf)
    return d


def impulse_rows(raw, jump=2.0):
    e1 = raw[:, :16].sum(1)
    base = np.array([np.median(e1[max(0, i - 40):i + 1]) for i in range(len(e1))])
    spike = (e1 - base > jump).astype(float)
    return np.convolve(spike, np.ones(5))[: len(e1)] > 0


def load(files):
    """A recording session carries one class label for all of its frames, but a clap is a few frames in a second of
    quiet (the head also turns towards it, so the level alone tells nothing): in the sessions of impulsive sounds
    (clap, sharp) only the frames around a clear energy peak keep the class label, the rest are silence."""
    parts = []
    for f in files:
        d = np.fromfile(f, dtype=np.float32)
        raw_d = d[: d.size // REC * REC].reshape(-1, REC)
        d = floored(raw_d)
        name = os.path.basename(f)
        if name.startswith(("real_clap", "real_sharp")):
            # parked-head sessions: an impulse is a jump of the newest frame's energy above the room's own background (a rolling
            # median, rooms differ); it stays inside the net's ~5-row history window afterwards
            near = impulse_rows(raw_d)
            lab = d[:, AUDIO_IN:].copy()
            d[~near, AUDIO_IN:] = 0.0
            d[~near, AUDIO_IN] = 1.0
            d[near, AUDIO_IN:] = lab[near]
            # a clap and a knock on the table are the same to the robot - something banged: both labels
            d[near, AUDIO_IN + 2] = d[near, AUDIO_IN + 3] = 1.0
        parts.append(d)
    d = np.concatenate(parts)
    return d[:, :AUDIO_IN], d[:, AUDIO_IN:AUDIO_IN + len(AUDIO_OUT)]


def to_rms(v):
    return 8.0 * (10.0 ** (3.0 * v) - 1.0)


def from_rms(r):
    return np.log10(1.0 + r / 8.0) / 3.0


def clap_over_servo(x, y, noise_x, n_per_row, rng):
    """The robot is never quiet: its servos whine while it moves and the whine is as loud as a clap. Parked-head recordings of
    claps are laid over real recordings of the servo noise (energies add per band and history slot), flagged "own motion"."""
    pos = np.where(y[:, 2] + y[:, 3] > 0.5)[0]
    xs, ys = [], []
    for _ in range(n_per_row):
        nz = noise_x[rng.integers(0, len(noise_x), len(pos))]
        m = x[pos].copy()
        gain = rng.uniform(0.6, 1.3, (len(pos), 1))  # the noise is a little louder or quieter than in the recording
        m[:, :96] = from_rms(np.sqrt(to_rms(x[pos, :96]) ** 2 + (gain * to_rms(nz[:, :96])) ** 2))
        m[:, 97] = from_rms(np.sqrt(to_rms(x[pos, 97]) ** 2 + (gain[:, 0] * to_rms(nz[:, 97])) ** 2))
        m[:, 98] = 0.0
        m[:, 99] = 1.0
        xs.append(m)
        ys.append(y[pos])
    return np.concatenate(xs), np.concatenate(ys)


def augment(x, y):
    """Mic gain and colour differ between rooms and robots; noise floors too."""
    B = x.shape[0]
    dev = x.device
    x = x.clone()
    bands = x[:, :96].reshape(B, 6, 16)
    gain = torch.empty(B, 1, 1, device=dev).uniform_(-0.08, 0.10)  # log-level offset
    tilt = torch.empty(B, 1, 16, device=dev).uniform_(-0.04, 0.04)
    bands = (bands + gain + tilt).clamp_min(0.0) + torch.randn_like(bands) * 0.01
    x[:, :96] = bands.reshape(B, 96)
    x[:, 97] = (x[:, 97] + gain[:, 0, 0]).clamp_min(0.0)
    # left/right swap (mirror world)
    flip = torch.rand(B, device=dev) < 0.5
    x[:, 96] = torch.where(flip, -x[:, 96], x[:, 96])
    return x, y


def loss_fn(logits, y):
    cls = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    w = 1.0 + 2.0 * y
    return (cls * w).mean()


@torch.no_grad()
def evaluate(model, x, y, dev):
    model.eval()
    o = model(torch.from_numpy(x).to(dev)).cpu()
    p = torch.sigmoid(o).numpy()
    rep = {}
    for i, n in enumerate(AUDIO_OUT):
        t = y[:, i] > 0.5
        tp = ((p[:, i] > 0.5) & t).sum()
        rep[n] = f"rec {tp / max(1, t.sum()):.2f} prec {tp / max(1, (p[:, i] > 0.5).sum()):.2f} pos {t.mean():.3f}"
    return rep


@torch.no_grad()
def report_real(model, files, dev):
    """Per held-out session: the net + the runtime gate, what each class says over the loud rows / how many impulses are caught."""
    model.eval()
    for f in files:
        raw = np.fromfile(f, dtype=np.float32)
        raw = raw[: raw.size // REC * REC].reshape(-1, REC)
        d = floored(raw)
        p = torch.sigmoid(model(torch.from_numpy(d[:, :AUDIO_IN]).to(dev))[:, :8]).cpu().numpy()
        p = p * np.clip((d[:, 97] - 0.03) / 0.15, 0, 1)[:, None]
        loud = impulse_rows(raw)
        # impulses = groups of loud rows
        idx = np.where(loud)[0]
        groups = np.split(idx, np.where(np.diff(idx) > 6)[0] + 1) if len(idx) else []
        hit = {n: sum(p[g, i].max() > 0.5 for g in groups) for i, n in ((2, "clap"), (3, "sharp"), (1, "speech"))}
        hit["BANG(clap|sharp)"] = sum(max(p[g, 2].max(), p[g, 3].max()) > 0.5 for g in groups)
        print(f"  {os.path.basename(f)}: {len(groups)} loud events; caught as {hit}; rows>0.5: clap {(p[:, 2] > .5).sum()} "
              f"sharp {(p[:, 3] > .5).sum()} speech {(p[:, 1] > .5).sum()} of {len(d)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--qat", type=int, default=8)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--rw", type=int, default=15, help="repeat factor of the real recordings")
    ap.add_argument("--mix", type=int, default=0, help="clap-over-servo variants per impulse row")
    a = ap.parse_args()
    allf = sorted(glob.glob(os.path.join(HERE, "data", "audio", "*.bin")))
    real = [f for f in allf if os.path.basename(f).startswith("real_")]
    files = [f for f in allf if f not in real]
    val_files = files[::10]
    # held out: the last silence session, speech from the right, rhythmic claps; 'mixed' sessions (music, TV, home
    # life) carry no label and are only looked at
    held = ("real_silence_7", "real_speech_2", "real_clap_4", "real_sharp_3")
    real_val = [f for f in real if os.path.basename(f).startswith(held)]
    real_train = [f for f in real if f not in real_val and "real_mixed" not in f]
    # the real microphone is the one that matters: its recordings count 15x
    x, y = load([f for f in files if f not in val_files] + real_train * a.rw)
    rng = np.random.default_rng(0)
    a_mix = a.mix
    nx, _ = load([f for f in real_train if "real_silence" in f])
    nx = nx[(nx[:, 99] > 0.5) & (nx[:, 98] < 0.5) & (nx[:, 97] > 0.15)]  # rows with the servos running
    cx, cy = clap_over_servo(*load([f for f in real_train if "real_clap" in f or "real_sharp" in f]), nx, a.mix, rng) if a.mix else (None, None)
    print("servo-noise rows", len(nx), "| clap-over-servo rows added (x15)", len(cx) if a.mix else 0)
    if a.mix:
        x, y = np.concatenate([x] + [cx] * a.rw), np.concatenate([y] + [cy] * a.rw)
    vx, vy = load(val_files)
    rvx, rvy = load(real_val) if real_val else (None, None)
    print("records", len(x), len(vx))
    dev = torch.device("cuda")
    model = AuditoryNet().to(dev)
    X, Y = torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    steps = len(X) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=(a.epochs + a.qat) * steps, pct_start=0.1)
    for ep in range(a.epochs + a.qat):
        model.quant = ep >= a.epochs
        model.train()
        perm = torch.randperm(len(X), device=dev)
        agg = 0.0
        for i in range(steps):
            idx = perm[i * a.bs:(i + 1) * a.bs]
            xb, yb = augment(X[idx], Y[idx])
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            agg += loss.item()
        if ep % 5 == 4:
            print(f"epoch {ep + 1} loss {agg / steps:.4f} quant={model.quant}")
    for k, v in evaluate(model, vx, vy, dev).items():
        print(f"  {k:11s} {v}")
    if rvx is not None:
        report_real(model, real_val, dev)
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(rvx).to(dev))[:, :8]).mean(0).cpu().numpy()
        print("real held-out ambient, mean class probabilities", dict(zip(["silence", "speech", "clap", "sharp", "cont", "low", "high", "tonal"], p.round(2))))
    torch.save(model.state_dict(), os.path.join(HERE, "data", "auditory_net.pt"))
    model.cpu().export(os.path.join(ROOT, "main", "hybrid", "auditory_model_data.h"))
    print("exported")
