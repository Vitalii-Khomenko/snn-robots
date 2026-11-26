"""Colab T4 GPU job: build full SnnV4 dataset (1200 synth + 800 real v4 lives) and fine-tune from
pretrained snn_v4.pt with batch_size=256 for 10 epochs (6 fp32 + 4 QAT int8) -> ~25 min total on T4.
"""
import os
import shutil
import subprocess
import sys
import tarfile
import time

t0 = time.time()
os.makedirs("/content/work", exist_ok=True)
with tarfile.open("/content/bundle.tar.gz", "r:gz") as tf:
    tf.extractall("/content/work")

os.chdir("/content/work")
if "/content/work" not in sys.path:
    sys.path.insert(0, "/content/work")

import torch
print("GPU:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)
try:
    import numba  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numba"], check=True)

import train_snn_v4 as T
T.OUT = "/content/work/out"
T.DATA = "/content/work/data"
T.N_SYN = 1200  # balanced with 800 real v4 lives -> 2000 total, fits cleanly inside Colab free window
os.makedirs(T.OUT, exist_ok=True)
os.makedirs(T.DATA, exist_ok=True)

print("=== Building synthetic + teacher labels on Colab ===", flush=True)
T.build("/content/work/snn_real.npz")
print(f"Dataset ready in {time.time() - t0:.1f}s; warm-start T4 training (10 epochs, B=256)...", flush=True)
init_ckpt = "/content/work/data/snn_v4.pt" if os.path.exists("/content/work/data/snn_v4.pt") else None
T.train(10, init_ckpt=init_ckpt, batch_size=256)

os.makedirs("/content/out", exist_ok=True)
shutil.copy2(os.path.join(T.DATA, "snn_v4.pt"), "/content/out/snn_v4.pt")
shutil.copy2(os.path.join(T.OUT, "snn_report.txt"), "/content/out/snn_report.txt")
print(f"=== SnnV4 Colab job done in {(time.time() - t0) / 60:.1f} min ===", flush=True)
