"""Colab T4 GPU job (Stage 5b): run the freshly trained SnnV4 (/content/out/snn_v4.pt) over 1400 synth + 800 real
v4 lives on Tesla T4 and retrain NnV4 (imitation + 3 DAgger rounds) so the main network matches the upgraded SnnV4.
"""
import os
import shutil
import sys
import tarfile
import time
import numpy as np
import torch

os.environ["STACKCHAN_WORKERS"] = "1"  # avoid fork() OOM in Colab's 12 GB Jupyter container
t0 = time.time()
os.makedirs("/content/work", exist_ok=True)
if os.path.exists("/content/bundle.tar.gz") and not os.path.exists("/content/work/train_nn_v4.py"):
    with tarfile.open("/content/bundle.tar.gz", "r:gz") as tf:
        tf.extractall("/content/work")
os.chdir("/content/work")
if "/content/work" not in sys.path:
    sys.path.insert(0, "/content/work")

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("NnV4 retrain on", dev, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

from snn_v4 import SnnV4
from synth_v4 import make_channels
import train_nn_v4 as TN

TN.OUT = "/content/work/out"
TN.DATA = "/content/work/data"
os.makedirs(TN.OUT, exist_ok=True)
os.makedirs(TN.DATA, exist_ok=True)

# 1. Build SNN output trajectories on T4 directly from synthetic + snn_real.npz
rng = np.random.default_rng(7)
N_SYN, SEQ = 1200, 6000
X_syn = np.stack([make_channels(rng, SEQ) for _ in range(N_SYN)]).astype(np.float32)
z = np.load("/content/work/snn_real.npz")
X_real = z["X_real"].astype(np.float32)
days = z["day_real"].astype(np.int16)
X = np.concatenate([X_syn, X_real], axis=0)
real = np.zeros(len(X), bool)
real[N_SYN:] = True
day = np.full(len(X), -1, np.int16)
day[N_SYN:] = days

snn_ckpt = "/content/out/snn_v4.pt" if os.path.exists("/content/out/snn_v4.pt") else "/content/work/data/snn_v4.pt"
snn = SnnV4().to(dev)
snn.load_state_dict(torch.load(snn_ckpt, weights_only=False, map_location=dev)["state"])
snn.quant = True
snn.eval()
outs = []
with torch.no_grad():
    for i in range(0, len(X), 64):
        xb = torch.from_numpy(X[i:i + 64]).to(dev)
        o, _, _ = snn(xb, snn.init_state(len(xb), dev))
        outs.append(o[:, ::2].cpu().numpy())
S = np.concatenate(outs).astype(np.float32)
np.savez_compressed(os.path.join(TN.OUT, "nn_snn.npz"), S=S, real=real, day=day)
del X_syn, X_real, X, snn, outs, S
torch.cuda.empty_cache()
print(f"Built nn_snn.npz on {dev} in {time.time() - t0:.1f}s", flush=True)

# 2. Train NnV4 (imitation + 3 DAgger rounds)
TN.train(3)

os.makedirs("/content/out", exist_ok=True)
shutil.copy2(os.path.join(TN.DATA, "nn_v4.pt"), "/content/out/nn_v4.pt")
shutil.copy2(os.path.join(TN.OUT, "nn_report.txt"), "/content/out/nn_report.txt")
print(f"=== NnV4 Colab job done in {(time.time() - t0) / 60:.1f} min ===", flush=True)
