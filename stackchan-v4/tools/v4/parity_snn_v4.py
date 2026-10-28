"""C++ SnnV4 (main/v4/snn_v4.cpp, exported weights) against the PyTorch network (quantized) on held-out lives: three
lives of the training data (out/snn_data.npz, synthetic and real) or, without it, three fresh synthetic lives.

A spiking network's two implementations drift apart here and there (a membrane a hair from its threshold spikes in one
and not the other), so the mean alone hides them (2026-10-28, an audit: mean 0.004, yet 0.109 at one tick of att_y).
Judged: the mean error, the share of ticks off by more than TAIL_ERR and the longest such stretch, and what matters
downstream - how often the executor's thresholds on these numbers decide differently (THRESHOLDS)."""
import os
import subprocess
import sys

import numpy as np
import torch

import snn_v4 as S
from spec_v4 import SNN_OUT

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
MAIN = os.path.join(HERE, "..", "..", "main", "v4")
TAIL_ERR = 0.05
# the executor's and BrainLoop's thresholds on the analyzer's numbers (v4_executor: kAttLook, the voice's reaction)
THRESHOLDS = [("att_strength", 0.12), ("surprise", 0.35), ("threat", 0.45)]
MAX_MEAN, MAX_TAIL, MAX_FLIP = 0.01, 0.01, 0.005


def main():
    exe = os.path.join(OUT, "parity_snn_v4")
    subprocess.run(["g++", "-std=c++17", "-O2", "-I", MAIN, os.path.join(HERE, "parity_snn_v4.cpp"),
                    os.path.join(MAIN, "snn_v4.cpp"), "-o", exe], check=True)
    data = os.path.join(OUT, "snn_data.npz")
    if os.path.exists(data):
        d = np.load(data)
        lives = {k: d["X"][k][:3000] for k in (5, 1500, 1700)}  # synthetic and real lives
    else:
        from synth_v4 import make_channels
        rng = np.random.default_rng(7)
        lives = {f"synthetic {i}": make_channels(rng, 3000) for i in range(3)}
    worst = 0.0
    for k, X in lives.items():
        X = X.astype(np.float32)
        X.tofile(os.path.join(OUT, "snn_in.bin"))
        subprocess.run([exe, os.path.join(OUT, "snn_in.bin"), os.path.join(OUT, "snn_out.bin")], check=True)
        cpp = np.fromfile(os.path.join(OUT, "snn_out.bin"), np.float32).reshape(len(X), -1)
        m = S.SnnV4()
        m.load_state_dict(torch.load(os.path.join(HERE, "data", "snn_v4.pt"), weights_only=False, map_location="cpu")["state"])
        m.quant = True
        m.eval()
        with torch.no_grad():
            py, _, _ = m(torch.tensor(X)[None], m.init_state(1, "cpu"))
        py = py[0].numpy()
        err = np.abs(cpp - py)
        off = (err > TAIL_ERR).any(1)
        run, longest = 0, 0
        for o in off:
            run = run + 1 if o else 0
            longest = max(longest, run)
        flips = {n: float(((cpp[:, SNN_OUT.index(n)] > th) != (py[:, SNN_OUT.index(n)] > th)).mean()) for n, th in THRESHOLDS}
        print(f"life {k}: mean |C++ - torch| {err.mean():.4f}, p99 {np.quantile(err, 0.99):.4f}, max {err.max():.3f} "
              f"({SNN_OUT[int(err.max(0).argmax())]}); ticks off by > {TAIL_ERR} {100 * off.mean():.2f} % (longest "
              f"{longest} ticks); thresholds decided otherwise: " + ", ".join(f"{n} {100 * f:.2f} %" for n, f in flips.items()))
        worst = max(worst, err.mean())
        ok_life = err.mean() < MAX_MEAN and off.mean() < MAX_TAIL and max(flips.values()) < MAX_FLIP
        all_ok = ok_life if k == next(iter(lives)) else all_ok and ok_life
    ok = all_ok
    print(f"(gates: mean < {MAX_MEAN}, ticks off < {100 * MAX_TAIL:g} %, threshold decisions differing < {100 * MAX_FLIP:g} %)")
    print("PARITY", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
