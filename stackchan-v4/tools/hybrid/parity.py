"""Writes data/parity.bin (inputs + PyTorch outputs of the quantized nets) and
runs the C++ comparison:  ./py parity.py"""
import os
import subprocess

import numpy as np
import torch

from models import AUDIO_IN, AuditoryNet, VisionNet, VISION_OUT, VISION_SIGNED

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1)
vis, aud = VisionNet(), AuditoryNet()
vis.load_state_dict(torch.load(os.path.join(HERE, "data", "vision_net.pt")))
aud.load_state_dict(torch.load(os.path.join(HERE, "data", "auditory_net.pt")))
vis.quant = aud.quant = True
vis.eval(), aud.eval()
# real data
v = np.fromfile(os.path.join(HERE, "data", "vision", sorted(os.listdir(os.path.join(HERE, "data", "vision")))[3]), np.uint8)
rec = 2304 + 16 + 52 + 8
a = v[: v.size // rec * rec].reshape(-1, rec)[100:132]
img = torch.from_numpy(a[:, :2304].copy().view(np.int8).astype(np.float32) / 60).reshape(-1, 3, 24, 32)
side = torch.from_numpy(a[:, 2304:2320].copy().view(np.float32))
with torch.no_grad():
    o = vis(img, side)
    o = torch.where(torch.tensor([n in VISION_SIGNED for n in VISION_OUT]), torch.tanh(o), torch.sigmoid(o))
    live = torch.clamp((o - 0.15) / 0.85, 0, 1)  # the C++ dead zone
    keep = torch.tensor([n in VISION_SIGNED for n in VISION_OUT])
    o = torch.where(keep, o, live)
    for n in ("person", "face"):  # a person / face is believed only above 0.30 (VisionNet::run)
        i = VISION_OUT.index(n)
        o[:, i] = torch.clamp((torch.sigmoid(vis(img, side))[:, i] - 0.30) / 0.70, 0, 1)
d = np.fromfile(os.path.join(HERE, "data", "audio", sorted(os.listdir(os.path.join(HERE, "data", "audio")))[3]), np.float32)
d = d[: d.size // (AUDIO_IN + 9) * (AUDIO_IN + 9)].reshape(-1, AUDIO_IN + 9)[200:232]
with torch.no_grad():
    ao = aud(torch.from_numpy(d[:, :AUDIO_IN].copy()))
    ao = torch.sigmoid(ao)
with open(os.path.join(HERE, "data", "parity.bin"), "wb") as f:
    f.write(np.array([32, 32], np.int32).tobytes())
    for i in range(32):
        f.write(img[i].numpy().tobytes() + side[i].numpy().tobytes() + o[i].numpy().astype(np.float32).tobytes())
    for i in range(32):
        f.write(d[i, :AUDIO_IN].tobytes() + ao[i].numpy().astype(np.float32).tobytes())
subprocess.run(["g++", "-std=c++17", "-O2", "-I../../simulator/shims", "-o", "data/parity_nets", "parity_nets.cpp",
                "../../main/hybrid/vision_net.cpp", "../../main/hybrid/vision_frontend.cpp",
                "../../main/hybrid/auditory_net.cpp", "../../main/hybrid/audio_frontend.cpp"], check=True, cwd=HERE)
subprocess.run(["data/parity_nets", "data/parity.bin"], check=True, cwd=HERE)
