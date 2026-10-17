"""Exports data/snn.pt to the C++ header, runs the PyTorch core (noise off, QAT weights)
and the C++ core on the same sequence and compares.   ./py parity_snn.py"""
import os
import subprocess

import numpy as np
import torch

import snn_model as M
import synth
from spec import SNN

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
core = M.SpikingCore()
core.load_state_dict(torch.load(os.path.join(HERE, "data", "snn.pt"), map_location="cpu"))
M.export_header(core, os.path.join(ROOT, "main", "hybrid", "snn_model_data.h"))
core = M.SpikingCore()
core.load_state_dict(torch.load(os.path.join(HERE, "data", "snn.pt"), map_location="cpu"))
core.quant, core.noise = True, 0.0
core.eval()
rng = np.random.default_rng(99)
X = synth.make_sequence(rng, 3000, dense=1.0)
with torch.no_grad():
    st = core.init_state(1, "cpu")
    y, _, _ = core(torch.as_tensor(X)[None], st, noise=False)
    pred = core.decode(y)[0].numpy()
with open(os.path.join(HERE, "data", "parity_snn.bin"), "wb") as f:
    f.write(np.int32(len(X)).tobytes())
    f.write(X.astype(np.float32).tobytes())
    f.write(pred.astype(np.float32).tobytes())
subprocess.run(["g++", "-std=c++17", "-O2", "-I../../simulator/shims", "-o", "data/parity_snn", "parity_snn.cpp",
                "../../main/hybrid/spiking_core.cpp"], check=True, cwd=HERE)
subprocess.run(["data/parity_snn", "data/parity_snn.bin"], cwd=HERE)
