#!/usr/bin/env python3
"""Parity test: C++ VisionNet (Core 1 int8 CNN) vs PyTorch VisionNetV2 (tools/v4/data/vision_v2.pt)."""
import os
import subprocess
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BIN = os.path.join(HERE, "out", "parity_vision")


def q_w(w):
    flat = w.reshape(w.shape[0], -1)
    s = flat.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    s = s.reshape(-1, *([1] * (w.dim() - 1)))
    return torch.round(w / s).clamp(-127, 127) * s


class VisionNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 24, 3, padding=1)
        self.fc1 = nn.Linear(24 * 3 * 4 + 4, 64)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, img, side):
        x = F.relu(F.conv2d(img, q_w(self.conv1.weight), self.conv1.bias, padding=1))
        x = F.max_pool2d(x, 2)
        x = F.relu(F.conv2d(x, q_w(self.conv2.weight), self.conv2.bias, padding=1))
        x = F.max_pool2d(x, 2)
        x = F.relu(F.conv2d(x, q_w(self.conv3.weight), self.conv3.bias, padding=1))
        x = F.max_pool2d(x, 2).flatten(1)
        x = F.relu(F.linear(torch.cat([x, side], 1), q_w(self.fc1.weight), self.fc1.bias))
        return F.linear(x, q_w(self.fc2.weight), self.fc2.bias)


def main():
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    cmd = ["g++", "-O2", "-std=c++17", f"-I{ROOT}/main/perception",
           os.path.join(HERE, "parity_vision.cpp"),
           os.path.join(ROOT, "main/perception/vision_net.cpp"),
           "-o", BIN]
    subprocess.run(cmd, check=True)

    ck = torch.load(os.path.join(HERE, "data", "vision_v2.pt"), weights_only=False, map_location="cpu")
    m = VisionNetV2()
    m.load_state_dict(ck["state"])
    m.eval()

    rng = np.random.default_rng(123)
    img = rng.uniform(-1.5, 1.5, (64, 3, 24, 32)).astype(np.float32)
    side = rng.uniform(-1.0, 1.0, (64, 4)).astype(np.float32)
    with torch.no_grad():
        raw = m(torch.from_numpy(img), torch.from_numpy(side)).numpy()
    py_out = np.empty_like(raw)
    for c in (0, 1, 4):
        py_out[:, c] = 1.0 / (1.0 + np.exp(-raw[:, c]))
    for c in (2, 3):
        py_out[:, c] = np.tanh(raw[:, c])

    buf = b"".join(img[i].tobytes() + side[i].tobytes() for i in range(len(img)))
    p = subprocess.run([BIN], input=buf, capture_output=True, check=True)
    cpp_out = np.frombuffer(p.stdout, np.float32).reshape(len(img), 5)
    err = float(np.max(np.abs(py_out - cpp_out)))
    print(f"parity VisionNetV2: max |py - cpp| = {err:.2e} ({'PASS' if err < 1e-4 else 'FAIL'})")
    if err >= 1e-4:
        sys.exit(1)


if __name__ == "__main__":
    main()
