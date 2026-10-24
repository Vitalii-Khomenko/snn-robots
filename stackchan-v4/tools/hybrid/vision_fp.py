"""False 'person' rate of data/vision_net.pt on the robot's own empty-room recordings (and hit rate on person files).
   ./py vision_fp.py"""
import glob
import os

import numpy as np
import torch

import train_vision as tv
from models import VisionNet

HERE = os.path.dirname(os.path.abspath(__file__))
m = VisionNet()
m.load_state_dict(torch.load(os.path.join(HERE, "data", "vision_net.pt")))
m.eval()
for f in sorted(glob.glob(os.path.join(HERE, "data", "vision", "real_*.bin"))):
    img, side, lab = tv.load([f])
    with torch.no_grad():
        o = m(torch.from_numpy(img).float() / 60.0, torch.from_numpy(side)).numpy()
    p, fc = 1 / (1 + np.exp(-o[:, 5])), 1 / (1 + np.exp(-o[:, 6]))
    print(f"{os.path.basename(f):18s} {len(img):4d} frames  person>0.5 {100 * (p > .5).mean():3.0f}%  >0.3 {100 * (p > .3).mean():3.0f}%  mean {p.mean():.2f}  face>0.5 {100 * (fc > .5).mean():3.0f}%")
