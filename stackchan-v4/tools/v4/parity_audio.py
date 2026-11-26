#!/usr/bin/env python3
"""Parity test: C++ AuditoryNet (2D ConvNet v2) vs PyTorch AuditoryNetV2 (tools/v4/data/auditory_v2.pt)."""
import os
import subprocess
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BIN = os.path.join(HERE, "out", "parity_audio")


def q_w(w):
    flat = w.reshape(w.shape[0], -1)
    s = flat.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    s = s.reshape(-1, *([1] * (w.dim() - 1)))
    return torch.round(w / s).clamp(-127, 127) * s


class AuditoryNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, 3, padding=1)
        self.fc1 = nn.Linear(12 * 3 * 8 + 4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 8)

    def forward(self, x):
        spec = x[:, :96].reshape(-1, 1, 6, 16)
        side = x[:, 96:]
        h = F.relu(F.conv2d(spec, q_w(self.conv1.weight), self.conv1.bias, padding=1))
        h = F.max_pool2d(h, 2).flatten(1)
        h = F.relu(F.linear(torch.cat([h, side], 1), q_w(self.fc1.weight), self.fc1.bias))
        h = F.relu(F.linear(h, q_w(self.fc2.weight), self.fc2.bias))
        return F.linear(h, q_w(self.fc3.weight), self.fc3.bias)


def main():
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    cmd = ["g++", "-O2", "-std=c++17", f"-I{ROOT}/main/perception", f"-I{ROOT}/simulator/shims",
           os.path.join(HERE, "parity_audio.cpp"),
           os.path.join(ROOT, "main/perception/auditory_net.cpp"),
           os.path.join(ROOT, "main/perception/audio_frontend.cpp"),
           "-o", BIN]
    subprocess.run(cmd, check=True)

    ck = torch.load(os.path.join(HERE, "data", "auditory_v2.pt"), weights_only=False, map_location="cpu")
    m = AuditoryNetV2()
    m.load_state_dict(ck["state"])
    m.eval()

    rng = np.random.default_rng(123)
    X = rng.uniform(0.0, 1.0, (256, 100)).astype(np.float32)
    X[:, 96] = rng.uniform(-1.0, 1.0, 256).astype(np.float32)
    with torch.no_grad():
        py_out = torch.sigmoid(m(torch.from_numpy(X))).numpy()

    p = subprocess.run([BIN], input=X.tobytes(), capture_output=True, check=True)
    cpp_out = np.frombuffer(p.stdout, np.float32).reshape(len(X), 8)
    err = float(np.max(np.abs(py_out - cpp_out)))
    print(f"parity AuditoryNetV2: max |py - cpp| = {err:.2e} ({'PASS' if err < 1e-4 else 'FAIL'})")
    if err >= 1e-4:
        sys.exit(1)


if __name__ == "__main__":
    main()
