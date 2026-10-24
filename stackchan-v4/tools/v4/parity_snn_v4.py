"""C++ SnnV4 (main/v4/snn_v4.cpp, exported weights) against the PyTorch network (quantized) on held-out lives: three
lives of the training data (out/snn_data.npz, synthetic and real) or, without it, three fresh synthetic lives."""
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
        print(f"life {k}: mean |C++ - torch| {err.mean():.4f}, max {err.max():.3f} ({SNN_OUT[int(err.max(0).argmax())]})")
        worst = max(worst, err.mean())
    ok = worst < 0.01
    print("PARITY", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
