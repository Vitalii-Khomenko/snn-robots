"""C++ MuseV4 (main/v4/muse_v4.cpp) against its twin tools/v4/muse.py on the analyzer's dataset, with a learned bias."""
import os
import subprocess
import sys

import numpy as np

import muse as M

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

if __name__ == "__main__":
    rng = np.random.default_rng(3)
    S = M.dataset(1)[0, :6000]
    noise = rng.standard_normal((len(S), M.N)).astype(np.float32)
    bias = np.array([0, 0, 0, 0.3, -0.2, 0.1, -0.4, 0.5], np.float32)
    np.concatenate([S, noise], 1).astype(np.float32).tofile(os.path.join(OUT, "muse_in.bin"))
    bias.tofile(os.path.join(OUT, "muse_bias.bin"))
    exe = os.path.join(OUT, "parity_muse")
    subprocess.run(["g++", "-std=c++17", "-O2", "-I", os.path.join(HERE, "..", "..", "main", "v4"),
                    os.path.join(HERE, "parity_muse.cpp"), os.path.join(HERE, "..", "..", "main", "v4", "muse_v4.cpp"),
                    "-o", exe], check=True)
    subprocess.run([exe, os.path.join(OUT, "muse_in.bin"), os.path.join(OUT, "muse_out.bin"),
                    os.path.join(OUT, "muse_bias.bin")], check=True)
    c = np.fromfile(os.path.join(OUT, "muse_out.bin"), np.float32).reshape(-1, len(M.OUTS))
    m = M.load(bias)
    py = np.stack([m.step(S[t:t + 1], noise[t:t + 1])[0] for t in range(len(S))])
    err = np.abs(c - py).max()
    print(f"max |C++ - numpy| {err:.2e} over {len(S)} steps")
    print("PARITY PASS" if err < 1e-3 else "PARITY FAIL")
    sys.exit(0 if err < 1e-3 else 1)
