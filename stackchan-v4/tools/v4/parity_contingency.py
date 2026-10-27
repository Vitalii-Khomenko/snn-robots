"""C++ ContingencyV4 (main/v4/contingency_v4.cpp) against its twin tools/v4/contingency.py on real diary days.

  ./py parity_contingency.py [DAY ...]      default: 2026-10-26 (a day with petting, talk and someone near)
"""
import os
import subprocess
import sys

import numpy as np

import contingency as C
import diary as D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
MAIN = os.path.join(HERE, "..", "..", "main")

if __name__ == "__main__":
    days = sys.argv[1:] or ["2026-10-26"]
    exe = os.path.join(OUT, "parity_contingency")
    subprocess.run(["g++", "-std=c++17", "-O2", "-I", os.path.join(MAIN, "v4"), os.path.join(HERE, "parity_contingency.cpp"),
                    os.path.join(MAIN, "v4", "contingency_v4.cpp"), "-o", exe], check=True)
    ok = True
    for day in days:
        t, _ = D.load(day)
        ms, rows, boot = C.inputs(t)
        rec = np.dtype([("ms", "<u4"), ("r", "<f4", (12,))])
        starts = np.flatnonzero(np.concatenate([[True], boot[1:] != boot[:-1]]))
        stream = np.zeros(len(ms) + len(starts), rec)
        pos = np.arange(len(ms)) + np.searchsorted(starts, np.arange(len(ms)), side="right")
        stream["r"][:, 0] = -1.0  # boot markers where no tick goes
        stream["ms"][pos] = ms
        stream["r"][pos] = rows
        stream.tofile(os.path.join(OUT, "contingency_in.bin"))
        subprocess.run([exe, os.path.join(OUT, "contingency_in.bin"), os.path.join(OUT, "contingency_out.bin")], check=True)
        res = np.fromfile(os.path.join(OUT, "contingency_out.bin"), np.dtype([("o", "<f4", (6,)), ("c", "u1")]))
        py, credits = C.run(ms, rows, boot)
        err = np.abs(res["o"] - py).max()
        n_cpp, n_py = int(res["c"].sum()), len(credits)
        print(f"{day}: {len(ms)} ticks, credits C++ {n_cpp} / numpy {n_py}, max |C++ - twin| {err:.2e}")
        ok = ok and err < 1e-3 and n_cpp == n_py
    print("PARITY PASS" if ok else "PARITY FAIL")
    sys.exit(0 if ok else 1)
