"""The Muse (docs/CREATIVITY.md): brain v4's source of ideas, a small chaotic reservoir next to the main network.

64 leaky tanh neurons in three groups by time scale (16 x 0.5 s, 24 x 3 s, 24 x 60 s), recurrently connected a little
past the edge of chaos and gently pushed by the analyzer's mood numbers (DRIVES). It replaces the three blind
pacemakers: its first three outputs ARE the pacemakers (pace_fast ~3 s, pace_slow ~60 s, pace_jitter ~0.5 s, the same
range and spread), so every random choice of the teacher now depends a little on the room and the mood, and hangs
together for longer. Five more outputs are IDEAS the main network sees (PREV idea_*):

  play      a sound game: short calls in a row, a lively face and lights
  show      try a face it rarely makes, a flicker of the LEDs
  still     be still and watch: no babble, the head holds, the lights breathe low
  social    call to whoever is around: an asking voice, a look to where someone was last seen
  surprise  a short odd act: a nod dance, a burst of light

Each idea is a readout of the reservoir plus a direct push from the mood (play: lively and content; still: sleepy,
calm; social: lonely; surprise: bored, not threatened). The Muse gets bored of its own ideas: an idea it has been
having fades (habituation over 90 s), so the reservoir's other ideas come up. What the night sleep learns is one bias
per idea (learned blob "muse.bias"): ideas the owner answered come more often (sleep_v4.py learn_muse).

  ./py muse.py make        generate the fixed weights (seeded), calibrate, write data/muse_v4.npz + main/v4/muse_v4_data.h
  ./py muse.py stats       how the outputs behave on the analyzer's dataset

The C++ twin is main/v4/muse_v4.cpp (parity_muse.py). The noise comes from outside (BrainV4's generator on the robot,
an array in the parity test), so both twins compute the same numbers.
"""
import os
import sys

import numpy as np

from spec_v4 import SNN_OUT

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(HERE, "data", "muse_v4.npz")
HEADER = os.path.join(ROOT, "main", "v4", "muse_v4_data.h")

DT = 0.1
GROUPS = [(16, 0.5), (24, 3.0), (24, 60.0)]  # (neurons, time constant s): jitter, fast, slow
N = sum(n for n, _ in GROUPS)
TAU = np.concatenate([np.full(n, t) for n, t in GROUPS]).astype(np.float32)
A = (DT / TAU).astype(np.float32)          # leak per step
NOISE = (np.sqrt(2.0 * A) * 0.6).astype(np.float32)  # Ornstein-Uhlenbeck scale: the same spread in every group
DRIVES = ["valence", "arousal", "under", "over", "threat", "att_strength", "sleepiness", "loneliness"]
DRIVE_IDX = [SNN_OUT.index(d) for d in DRIVES]
OUTS = ["pace_fast", "pace_slow", "pace_jitter", "play", "show", "still", "social", "surprise"]
IDEAS = OUTS[3:]
PACE_GROUP = {"pace_fast": 1, "pace_slow": 2, "pace_jitter": 0}
RHO = 1.1          # spectral radius of the recurrent weights: a little past the edge of chaos
TAU_HAB = 90.0     # s: how fast the Muse gets bored of an idea it keeps having
HAB_CUT = 0.7
IDEA_ON = 0.5      # an idea counts as had above this (teacher_nn.py, sleep_v4.py)
IDEA_SHARE = 0.10  # calibrated: each idea above IDEA_ON ~10 % of the day before habituation and learning
PACE_STD = 0.6     # the old pacemakers' spread (tanh(0.9 N(0, 1)))
BIAS_MAX = 0.6     # what the night may move an idea's bias, either way
# the mood's direct push on each idea (rows: OUTS, columns: DRIVES, applied to drive - 0.5)
PUSH = {
    "play":     {"valence": 0.8, "arousal": 1.0, "threat": -1.0},
    "show":     {"valence": 0.8, "arousal": 0.4},
    "still":    {"sleepiness": 1.2, "arousal": -0.6},
    "social":   {"loneliness": 0.8, "under": 0.3},
    "surprise": {"under": 1.2, "threat": -1.0},
}


def make(seed=20261026):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (N, N)) * (rng.random((N, N)) < 0.15)
    W *= RHO / np.max(np.abs(np.linalg.eigvals(W)))
    Win = rng.normal(0, 0.5, (N, len(DRIVES)))
    b = rng.normal(0, 0.1, N)
    Wout = np.zeros((len(OUTS), N))
    Wu = np.zeros((len(OUTS), len(DRIVES)))
    starts = np.cumsum([0] + [n for n, _ in GROUPS])
    for k, name in enumerate(OUTS):
        if name in PACE_GROUP:
            g = PACE_GROUP[name]
            v = np.zeros(N)
            v[starts[g]:starts[g + 1]] = rng.normal(0, 1, starts[g + 1] - starts[g])
        else:  # ideas live for seconds to minutes: the fast and slow groups
            v = np.zeros(N)
            v[starts[1]:] = rng.normal(0, 1, N - starts[1])
            for d, w in PUSH[name].items():
                Wu[k, DRIVES.index(d)] = w
        Wout[k] = v / np.linalg.norm(v)
    p = dict(W=W, Win=Win, b=b, Wout=Wout, Wu=Wu, bout=np.zeros(len(OUTS)))
    return {k: np.asarray(v, np.float32) for k, v in p.items()}


class Muse:
    """B Muses at once (numpy, float32). step(snn [B, 32], noise [B, N]) -> [B, 8]."""

    def __init__(self, p, B=1, bias=None):
        self.p = p
        self.x = np.zeros((B, N), np.float32)
        self.h = np.zeros((B, len(IDEAS)), np.float32)
        self.bias = np.zeros(len(OUTS), np.float32) if bias is None else np.asarray(bias, np.float32)

    def raw(self, snn, noise):
        p = self.p
        u = (np.asarray(snn, np.float32)[:, DRIVE_IDX] - np.float32(0.5)).astype(np.float32)
        pre = self.x @ p["W"].T + u @ p["Win"].T + p["b"]
        self.x = np.clip((1 - A) * self.x + A * np.tanh(pre) + NOISE * noise, -3.0, 3.0).astype(np.float32)
        return np.tanh(self.x @ p["Wout"].T + u @ p["Wu"].T + p["bout"] + self.bias).astype(np.float32)

    def step(self, snn, noise):
        y = self.raw(snn, noise)
        idea = y[:, 3:]
        on = np.maximum(idea, 0)
        self.h += (on - self.h) * np.float32(DT / TAU_HAB)
        y[:, 3:] = np.where(idea > 0, idea * (1 - HAB_CUT * self.h), idea)
        return y


def load(bias=None, B=1):
    return Muse({k: v for k, v in np.load(DATA).items()}, B, bias)


def run(S, rng, bias=None):
    """The Muse over SNN output sequences S [B, T, 32] at 10 Hz -> [B, T, 8] (training: one noise per life)."""
    B, T, _ = S.shape
    m = load(bias, B)
    out = np.zeros((B, T, len(OUTS)), np.float32)
    for t in range(T):
        out[:, t] = m.step(S[:, t], rng.standard_normal((B, N)).astype(np.float32))
    return out


def calibrate(p, S, rng):
    """Scale the pace readouts to the pacemakers' spread and set each idea's resting bias for IDEA_SHARE."""
    m = Muse(p, S.shape[0])
    raws = []
    for t in range(S.shape[1]):
        m.raw(S[:, t], rng.standard_normal((S.shape[0], N)).astype(np.float32))
        raws.append(m.x @ p["Wout"].T)
    R = np.stack(raws, 1)[:, 300:]  # after the slow group's warm-up
    u = (S[:, 300:, DRIVE_IDX] - 0.5) @ p["Wu"].T
    for k, name in enumerate(OUTS):
        r = R[..., k]
        if name in PACE_GROUP:
            r = r - np.median(r)  # centred like the pacemakers: the teacher's picks stay even
            lo, hi = 0.05, 20.0
            for _ in range(40):  # gain so that std(tanh(g r)) = PACE_STD
                g = (lo * hi) ** 0.5
                lo, hi = (g, hi) if np.tanh(g * r).std() < PACE_STD else (lo, g)
            p["bout"][k] = -g * np.median(R[..., k])
            p["Wout"][k] *= g
        else:
            g = 1.0 / max(r.std(), 1e-6)  # the reservoir's part ~ as strong as the mood's push
            p["Wout"][k] *= g
            z = g * r + u[..., k]
            p["bout"][k] = np.arctanh(IDEA_ON) - np.quantile(z, 1 - IDEA_SHARE)
    return p


def header(p):
    def arr(name, a):
        flat = ", ".join(f"{float(v):.8e}f" for v in np.asarray(a, np.float32).ravel())
        return f"inline constexpr float {name}[{a.size}] = {{{flat}}};"
    lines = ["#pragma once", "// Generated by tools/v4/muse.py make - do not edit. The Muse's fixed weights (docs/CREATIVITY.md).",
             "namespace muse_data {",
             f"inline constexpr int kN = {N}, kDrives = {len(DRIVES)}, kOuts = {len(OUTS)};",
             "inline constexpr int kDriveIdx[kDrives] = {" + ", ".join(str(i) for i in DRIVE_IDX) + "};",
             arr("kLeak", A), arr("kNoise", NOISE)]
    for k in ("W", "Win", "b", "Wout", "Wu", "bout"):
        lines.append(arr("k" + k[0].upper() + k[1:], p[k]))
    lines.append(f"inline constexpr float kHabRate = {DT / TAU_HAB:.8g}f, kHabCut = {HAB_CUT}f, kBiasMax = {BIAS_MAX}f;")
    lines.append("}  // namespace muse_data")
    return "\n".join(lines) + "\n"


def dataset(n=40):
    d = np.load(os.path.join(HERE, "out", "nn_snn.npz"))
    return d["S"][:n].astype(np.float32)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    rng = np.random.default_rng(7)
    if cmd == "make":
        p = calibrate(make(), dataset(), rng)
        np.savez(DATA, **p)
        open(HEADER, "w").write(header(p))
        print(f"wrote {DATA} and {HEADER}")
        cmd = "stats"
    if cmd == "stats":
        S = dataset(20)
        Y = run(S, rng)[:, 300:]
        for k, name in enumerate(OUTS):
            y = Y[..., k]
            ac = [np.mean([np.corrcoef(y[i, :-lag], y[i, lag:])[0, 1] for i in range(len(y))]) for lag in (5, 30, 300)]
            print(f"  {name:12s} mean {y.mean():+.2f} std {y.std():.2f}  > {IDEA_ON}: {np.mean(y > IDEA_ON):.3f}  "
                  f"autocorr 0.5 s {ac[0]:.2f} 3 s {ac[1]:.2f} 30 s {ac[2]:.2f}")
