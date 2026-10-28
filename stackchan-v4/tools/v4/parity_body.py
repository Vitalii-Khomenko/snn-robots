"""C++ BodyV4 (main/v4/body_v4.cpp) against its twin tools/v4/body.py on real diary days.

  ./py parity_body.py [DAY ...]      default: 2026-10-26 (a whole day and night)
"""
import os
import subprocess
import sys

import numpy as np

import body as B
import diary as D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
MAIN = os.path.join(HERE, "..", "..", "main")

if __name__ == "__main__":
    days = sys.argv[1:] or ["2026-10-26"]
    exe = os.path.join(OUT, "parity_body")
    subprocess.run(["g++", "-std=c++17", "-O2", "-I", os.path.join(MAIN, "v4"), os.path.join(HERE, "parity_body.cpp"),
                    os.path.join(MAIN, "v4", "body_v4.cpp"), "-o", exe], check=True)
    ok = True
    for day in days:
        t, lines = D.load(day)
        X, boot = B.inputs(t, lines)
        starts = np.flatnonzero(np.concatenate([[True], boot[1:] != boot[:-1]]))
        stream = np.zeros((len(X) + len(starts), X.shape[1]), np.float32)
        pos = np.arange(len(X)) + np.searchsorted(starts, np.arange(len(X)), side="right")
        stream[:, 0] = -1.0  # boot markers where no tick goes
        stream[pos] = X
        stream.tofile(os.path.join(OUT, "body_in.bin"))
        subprocess.run([exe, os.path.join(OUT, "body_in.bin"), os.path.join(OUT, "body_out.bin")], check=True)
        cpp = np.fromfile(os.path.join(OUT, "body_out.bin"), np.float32).reshape(-1, 3)
        py = np.stack(B.run(X, boot), 1)
        err = np.abs(cpp - py).max(0)
        print(f"{day}: {len(X)} ticks, {int(X[:, B.COLUMNS.index("hurt")].sum())} power cycles, max |C++ - twin| pressure {err[0]:.2e}, "
              f"load {err[1]:.2e}, budget {err[2]:.2e}")
        ok = ok and err[0] < 1e-4 and err[1] < 1e-3 and err[2] < 1e-4
    print("PARITY PASS" if ok else "PARITY FAIL")
    sys.exit(0 if ok else 1)
