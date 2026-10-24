"""Per-frame labels for a parked-head session in which a person stepped in and out of the picture.
   ./py relabel_by_background.py data/vision/real_front_1.bin
The head does not move, so the median frame is the empty room (the person is away for more than half of the session).
Frames far from it (mean luminance deviation > HI) hold a person: person = 1, target_conf = 1; frames close to it (< LO) are
empty room: person, face, loom, recede, target_conf = 0; frames in between stay unknown (NaN, masked in training)."""
import sys

import numpy as np

import train_vision as tv

HI, LO = 8.0, 4.0
path = sys.argv[1]
raw = np.fromfile(path, dtype=np.uint8).reshape(-1, tv.REC)
img, _, lab = tv.load([path])
lum = img[:, 0].astype(np.float32)
dev = np.abs(lum - np.median(lum, 0)).mean((1, 2))
lab = lab.copy()
for col in (5, 6, 7, 8, 11):
    lab[:, col] = np.nan
person, empty = dev > HI, dev < LO
lab[person, 5] = lab[person, 11] = 1.0
lab[empty, 5] = lab[empty, 6] = lab[empty, 7] = lab[empty, 8] = lab[empty, 11] = 0.0
raw[:, 2320:2372] = lab.astype(np.float32).view(np.uint8)
raw.tofile(path)
print(f"{path}: person {int(person.sum())}, empty room {int(empty.sum())}, unknown {int((~person & ~empty).sum())} of {len(dev)} frames")
