"""Policy parity: the compiled GRU (simulator, INT8 weights, float activations) against the quantized PyTorch
model on the very same inputs. Runs a 3 minute scenario with --bus-dump (which records the inputs the policy saw,
`HybridBrain::lastPolicyInput`) and replays them through torch.   ./py parity_policy.py"""
import os
import subprocess
import tempfile

import numpy as np
import torch

from models import PolicyNet
from spec import ACTION, ACTION_SIGNED, POLICY_INPUT

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SIM = os.path.join(ROOT, "simulator", "build", "stackchan_sim")
with tempfile.TemporaryDirectory() as tmp:
    dump = os.path.join(tmp, "bus.bin")
    subprocess.run([SIM, os.path.join(ROOT, "simulator", "scenarios", "play_session.scn"), "--minutes", "3", "--bus-dump", dump,
                    "--out", os.path.join(tmp, "out")], check=True, capture_output=True)
    d = np.fromfile(dump, np.float32).reshape(-1, len(POLICY_INPUT) + 3 + len(ACTION))
NIN = len(POLICY_INPUT)
x, a = d[:, :NIN], d[:, NIN + 3:]
m = PolicyNet()
m.load_state_dict(torch.load(os.path.join(HERE, "data", "policy_net.pt"), map_location="cpu"))
m.eval()
m.quant = True
with torch.no_grad():
    o, _ = m(torch.from_numpy(x).view(1, -1, NIN))
signed = torch.tensor([n in ACTION_SIGNED for n in ACTION])
p = torch.where(signed, torch.tanh(o[0]), torch.sigmoid(o[0])).numpy()
err = np.abs(p - a)
print(f"policy parity over {len(x)} ticks: max |diff| {err.max():.2e}, mean {err.mean():.2e}")
worst = np.argsort(-err.max(0))[:3]
print("worst outputs:", [(ACTION[i], float(err[:, i].max())) for i in worst])
