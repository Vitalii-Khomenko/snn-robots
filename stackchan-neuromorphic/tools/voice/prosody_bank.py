"""Real intonation from real speech: pitch contours of LibriSpeech phrases.

The babble teacher does not invent melodies: it replays the shape of how people
really raise and drop their voice. This script tracks F0 over ~700 utterances,
cuts them into phrases at pauses and stores each phrase's contour (semitones
around its own median, 32 points) with a few descriptors.

  ../vocal_decoder/.venv/bin/python prosody_bank.py   -> data/prosody_bank.npz
"""
import glob
import os
import random

import numpy as np
import soundfile as sf
from scipy.signal import medfilt, resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.abspath(os.path.join(HERE, "..", "..", "..", "datasets"))


def f0_track(x, sr=16000, hop=160, win=640, fmin=70, fmax=400):
    n = (len(x) - win) // hop
    f0 = np.zeros(n)
    conf = np.zeros(n)
    lo, hi = int(sr / fmax), int(sr / fmin)
    for i in range(n):
        seg = x[i * hop:i * hop + win]
        seg = seg - seg.mean()
        e = np.dot(seg, seg)
        if e < 1e-3:
            continue
        ac = np.correlate(seg, seg, "full")[win - 1:win + hi + 1]
        ac = ac / (ac[0] + 1e-9)
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.45:
            f0[i] = sr / k
            conf[i] = ac[k]
    return f0, conf


def main():
    rng = random.Random(3)
    files = sorted(glob.glob(os.path.join(DS, "ls", "LibriSpeech", "dev-clean", "*", "*", "*.flac")))
    rng.shuffle(files)
    contours, desc = [], []
    for f in files[:700]:
        x, sr = sf.read(f)
        x = resample_poly(x, 16000, sr).astype(np.float32) if sr != 16000 else x.astype(np.float32)
        f0, conf = f0_track(x)
        f0 = medfilt(f0, 5)
        voiced = f0 > 0
        # phrases: voiced stretches with gaps < 250 ms bridged
        idx = np.where(voiced)[0]
        if len(idx) < 20:
            continue
        starts = [idx[0]]
        ends = []
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a > 25:
                ends.append(a)
                starts.append(b)
        ends.append(idx[-1])
        for s, e in zip(starts, ends):
            dur = (e - s) * 0.01
            if dur < 0.6 or dur > 2.8:
                continue
            seg = f0[s:e + 1].copy()
            v = seg > 0
            if v.mean() < 0.6:
                continue
            t = np.arange(len(seg))
            seg[~v] = np.interp(t[~v], t[v], seg[v])
            st = 12 * np.log2(seg / np.median(seg[v]))
            c = np.interp(np.linspace(0, len(st) - 1, 32), np.arange(len(st)), st)
            contours.append(c.astype(np.float32))
            desc.append([dur, c[-6:].mean() - c[-14:-6].mean(), np.ptp(c), np.polyfit(np.linspace(0, 1, 32), c, 1)[0]])
    contours = np.array(contours)
    desc = np.array(desc, np.float32)
    np.savez(os.path.join(HERE, "data", "prosody_bank.npz"), contours=contours, desc=desc)
    print(len(contours), "phrase contours; median range", np.median(desc[:, 2]).round(1), "st; final move p10/p50/p90",
          np.percentile(desc[:, 1], [10, 50, 90]).round(1), "; declination p50", np.median(desc[:, 3]).round(1), "st/phrase")


main()
