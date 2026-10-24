"""C++ NnV4 (main/v4/nn_v4.cpp, exported weights) against the PyTorch network on random-but-plausible inputs."""
import os
import subprocess
import sys

import numpy as np
import torch

from nn_v4 import NnV4, activate
from spec_v4 import NN_IN

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
MAIN = os.path.join(HERE, "..", "..", "main", "v4")


def main():
    exe = os.path.join(OUT, "parity_nn_v4")
    subprocess.run(["g++", "-std=c++17", "-O2", "-I", MAIN, os.path.join(HERE, "parity_nn_v4.cpp"),
                    os.path.join(MAIN, "nn_v4.cpp"), "-o", exe], check=True)
    rng = np.random.default_rng(0)
    T = 2000
    X = np.clip(rng.random((T, len(NN_IN))) * 0.6 + 0.2 * np.sin(np.arange(T)[:, None] / 50), -1, 1).astype(np.float32)
    X.tofile(os.path.join(OUT, "nn_in.bin"))
    subprocess.run([exe, os.path.join(OUT, "nn_in.bin"), os.path.join(OUT, "nn_out.bin")], check=True)
    cpp = np.fromfile(os.path.join(OUT, "nn_out.bin"), np.float32).reshape(T, -1)
    m = NnV4()
    m.load_state_dict(torch.load(os.path.join(HERE, "data", "nn_v4.pt"), weights_only=False, map_location="cpu")["state"])
    with torch.no_grad():
        raw, _ = m(torch.tensor(X)[None])
        py = activate(raw)[0].numpy()
    err = np.abs(cpp - py).max()
    print(f"max |C++ - torch| {err:.2e}")
    print("PARITY", "PASS" if err < 1e-3 else "FAIL")
    sys.exit(0 if err < 1e-3 else 1)


if __name__ == "__main__":
    main()
