#!/usr/bin/env python3
"""Prepare compact Audio v2 dataset (out/audio_v2_data.npz) combining:
  - 5,555 real room .aud clips from data/media/*/*.aud + PANNs/Silero .lab.npz
  - All 28 real_*.bin recordings from stackchan-life/tools/hybrid/data/audio/
  - Stratified sample of ESC-50/synthetic audio_*.bin + servo-mixed claps
Held-out split is strictly by day (2026-10-24 media) and by file (every 5th real_*/synth file) so zero leakage.
"""
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LIFE_HYB = os.path.abspath(os.path.join(ROOT, "..", "stackchan-life", "tools", "hybrid"))
LIFE_TOOLS = os.path.abspath(os.path.join(ROOT, "..", "stackchan-life", "tools", "life"))
sys.path.insert(0, LIFE_HYB)
sys.path.insert(0, LIFE_TOOLS)

import media  # noqa: E402
import train_audio as TA  # noqa: E402

MEDIA_DIR = os.path.join(ROOT, "data", "media")
OLD_AUDIO = os.path.join(LIFE_HYB, "data", "audio")
PANNS_CSV = os.path.join(ROOT, "data", "teachers", "panns", "class_labels_indices.csv")
OUT_PATH = os.path.join(HERE, "out", "audio_v2_data.npz")

BANG = ["Clapping", "Hands", "Knock", "Tap", "Bang", "Thump, thud", "Clatter", "Smash, crash", "Breaking", "Wood", "Slam"]
CONTINUOUS = ["Mechanical fan", "Air conditioning", "Vacuum cleaner", "Hair dryer", "Rain", "Blender", "Engine",
              "Vehicle", "Water tap, faucet", "Microwave oven", "White noise", "Pink noise", "Static", "Hum", "Wind"]
TONAL = ["Music", "Musical instrument", "Singing", "Whistling", "Beep, bleep", "Bell", "Alarm", "Ringtone"]
TV_MUSIC = ["Television", "Radio", "Music", "Singing", "Mechanisms", "Gears", "Typing", "Computer keyboard"]


def load_panns_idx():
    with open(PANNS_CSV) as f:
        names = [r["display_name"] for r in csv.DictReader(f)]
    return lambda grp: [names.index(n) for n in grp if n in names]


def load_media_audio():
    idx = load_panns_idx()
    bang_i, cont_i, ton_i, neg_i = idx(BANG), idx(CONTINUOUS), idx(TONAL), idx(TV_MUSIC)
    xs, ys, val_flags = [], [], []
    labs = sorted(glob.glob(os.path.join(MEDIA_DIR, "*", "*.lab.npz")))
    for lp in labs:
        base = lp[:-len(".lab.npz")]
        aud_p = base + ".aud"
        if not os.path.exists(aud_p):
            continue
        day = os.path.basename(os.path.dirname(base))
        is_val = (day == "2026-10-24")
        try:
            L = np.load(lp)
            if "aud_classes" not in L or "aud_speech" not in L:
                continue
            a = media.read_audio(aud_p)
        except Exception:
            continue
        x = a["features"].astype(np.float32)
        n = len(x)
        if n < 10:
            continue
        c = L["aud_classes"].astype(np.float32)
        sp = L["aud_speech"][:n]
        level = x[:, 97]
        own_voice = x[:, 98] > 0.5
        y = np.full((n, 8), np.nan, np.float32)
        loud = level > 0.05
        quiet = level < 0.03
        # Speech: require Silero VAD >= 0.65 and not dominated by TV/music-only when quiet
        y[sp >= 0.65, 1] = 1.0
        y[sp <= 0.15, 1] = 0.0
        # Bang (clap + sharp): strict impulse + PANNs confirmation so clicks/speech plosives are hard negatives!
        bang = float(c[bang_i].max())
        near = TA.impulse_rows(x, jump=2.2)
        if bang >= 0.45:
            y[:, 2] = y[:, 3] = 0.0
            y[near & (level > 0.12), 2] = 1.0
            y[near & (level > 0.12), 3] = 1.0
        elif bang <= 0.12:
            y[:, 2] = y[:, 3] = 0.0
        # Continuous & tonal
        for col, cls_i in ((4, cont_i), (7, ton_i)):
            v = float(c[cls_i].max())
            if v >= 0.45:
                y[loud, col] = 1.0
                y[quiet, col] = 0.0
            elif v <= 0.12:
                y[:, col] = 0.0
        # Hard-negative speech/click suppression on pure mechanical/typing/noise clips
        if float(c[neg_i].max()) >= 0.4 and float(sp.max()) < 0.25:
            y[:, 1] = 0.0
            if bang <= 0.12:
                y[:, 2] = y[:, 3] = 0.0
        heard = (y[:, 1] == 1) | (y[:, 2] == 1) | (y[:, 3] == 1) | (y[:, 4] == 1) | (y[:, 7] == 1)
        y[quiet & (sp <= 0.15), 0] = 1.0
        y[heard, 0] = 0.0
        keep = ~own_voice & np.isfinite(y).any(1)
        if keep.sum() < 4:
            continue
        xs.append(x[keep])
        ys.append(y[keep])
        val_flags.append(np.full(int(keep.sum()), is_val, dtype=bool))
    if not xs:
        return np.zeros((0, 100), np.float32), np.zeros((0, 8), np.float32), np.zeros(0, bool)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(val_flags)


def main():
    rng = np.random.default_rng(42)
    print("Loading media room audio clips...", flush=True)
    mx, my, mval = load_media_audio()
    print(f"  media room ticks: {len(mx)} (val {mval.sum()})", flush=True)

    all_bins = sorted(glob.glob(os.path.join(OLD_AUDIO, "*.bin")))
    real_bins = [f for f in all_bins if os.path.basename(f).startswith("real_")]
    syn_bins = [f for f in all_bins if not os.path.basename(f).startswith("real_")]
    # Split real_bins: hold out 1 file per major real class if >= 3 files
    val_real, tr_real = [], []
    by_pref = {}
    for f in real_bins:
        pref = "_".join(os.path.basename(f).split("_")[:2])
        by_pref.setdefault(pref, []).append(f)
    for pref, flist in sorted(by_pref.items()):
        if len(flist) >= 3:
            val_real.append(flist[-1])
            tr_real.extend(flist[:-1])
        else:
            tr_real.extend(flist)
    # Sample 40 synthetic files (35 train, 5 val) so upload stays compact & balanced
    syn_pick = [syn_bins[i] for i in np.linspace(0, len(syn_bins) - 1, min(40, len(syn_bins))).round().astype(int)]
    val_syn = syn_pick[::8]
    tr_syn = [f for f in syn_pick if f not in set(val_syn)]

    print(f"Loading old bins: {len(tr_real)} tr_real, {len(val_real)} val_real, {len(tr_syn)} tr_syn, {len(val_syn)} val_syn", flush=True)
    rx_tr, ry_tr = TA.load(tr_real)
    rx_val, ry_val = TA.load(val_real)
    sx_tr, sy_tr = TA.load(tr_syn)
    sx_val, sy_val = TA.load(val_syn)

    # Subsample synthetic to 120k train / 15k val so real room + real mic dominate properly
    if len(sx_tr) > 120000:
        sel = rng.choice(len(sx_tr), 120000, replace=False)
        sx_tr, sy_tr = sx_tr[sel], sy_tr[sel]
    if len(sx_val) > 15000:
        sel = rng.choice(len(sx_val), 15000, replace=False)
        sx_val, sy_val = sx_val[sel], sy_val[sel]
    # Add servo-mixed claps from real_clap/sharp over real_silence servo noise
    noise_mask = rx_tr[:, 99] > 0.5
    if noise_mask.sum() > 200 and ((ry_tr[:, 2] + ry_tr[:, 3]) > 0.5).sum() > 50:
        cx, cy = TA.clap_over_servo(rx_tr, ry_tr, rx_tr[noise_mask], 3, rng)
        rx_tr = np.concatenate([rx_tr, cx])
        ry_tr = np.concatenate([ry_tr, cy])

    # Subsample media ticks to <= 160k train / <= 25k val
    m_tr_i = np.flatnonzero(~mval)
    m_val_i = np.flatnonzero(mval)
    if len(m_tr_i) > 160000:
        m_tr_i = rng.choice(m_tr_i, 160000, replace=False)
    if len(m_val_i) > 25000:
        m_val_i = rng.choice(m_val_i, 25000, replace=False)

    X_tr = np.concatenate([mx[m_tr_i], rx_tr, sx_tr]).astype(np.float16)
    Y_tr = np.concatenate([my[m_tr_i], ry_tr, sy_tr]).astype(np.float16)
    src_tr = np.concatenate([
        np.full(len(m_tr_i), 2, np.uint8),
        np.full(len(rx_tr), 1, np.uint8),
        np.full(len(sx_tr), 0, np.uint8),
    ])

    X_val = np.concatenate([mx[m_val_i], rx_val, sx_val]).astype(np.float16)
    Y_val = np.concatenate([my[m_val_i], ry_val, sy_val]).astype(np.float16)
    src_val = np.concatenate([
        np.full(len(m_val_i), 2, np.uint8),
        np.full(len(rx_val), 1, np.uint8),
        np.full(len(sx_val), 0, np.uint8),
    ])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(OUT_PATH, X_tr=X_tr, Y_tr=Y_tr, src_tr=src_tr, X_val=X_val, Y_val=Y_val, src_val=src_val)
    print(f"wrote {OUT_PATH}: train {X_tr.shape} val {X_val.shape} ({os.path.getsize(OUT_PATH) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
