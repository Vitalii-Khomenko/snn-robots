"""Records what the robot's camera really sees as Vision NN training data, in labelled parts.

  ./py record_vision.py LABEL SECONDS      # LABEL: empty | front | side
Reads 'vframe:' lines from the robot's serial port (firmware built with kVisionRec != Off in
main/core/debug_flags.h) and writes data/vision/real_<label>_<n>.bin in the training format.
  empty  nobody in the picture (use the Free build: the robot scans the room by itself)  -> person/face/target 0
  front  a person stands 0.7-2 m in front of the parked head, face to the camera          -> person 1, face 1
  side   a person in view, turned sideways / back / walking about                         -> person 1
Labels that cannot be known (where exactly the person is, motion) are NaN and masked in training.
"""
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NAN = float("nan")
#            ml  mc  mr  mu  md  person face loom rec  tx   ty   conf act
LABELS = {
    "empty": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "front": [NAN] * 5 + [1, 1, NAN, NAN, NAN, NAN, 1, NAN],
    "side": [NAN] * 5 + [1, NAN, NAN, NAN, NAN, NAN, 1, NAN],
}
line = re.compile(r"vframe: ([0-9a-f]{%d})" % (2 * (2304 + 10)))

label, seconds = sys.argv[1], int(sys.argv[2])
rows = []
t_end = time.time() + seconds
while time.time() < t_end:
    s = min(55, int(t_end - time.time()) + 1)
    out = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "capture_serial.py"), "--seconds", str(s)],
                         capture_output=True, text=True, errors="ignore").stdout
    for m in line.finditer(out):
        rows.append(bytes.fromhex(m.group(1)))
rec = bytearray()
kept = 0
for r in rows:
    img = r[:2304]
    v = np.frombuffer(r[2304:], ">i2").astype(np.float32) / 1000.0  # side[4], contrast/100
    side, contrast = v[:4], v[4] * 100.0
    lab = np.array(LABELS[label], np.float32)
    if contrast < 3.5:  # a dark frame: nothing can be seen, whatever the label says
        lab[[5, 6, 7, 8, 9, 10, 11]] = 0
    rec += img + side.astype(np.float32).tobytes() + lab.tobytes() + np.zeros(2, np.float32).tobytes()
    kept += 1
d = os.path.join(HERE, "data", "vision")
os.makedirs(d, exist_ok=True)
n = 0
while os.path.exists(os.path.join(d, f"real_{label}_{n}.bin")):
    n += 1  # never overwrite a session
path = os.path.join(d, f"real_{label}_{n}.bin")
open(path, "wb").write(bytes(rec))
print(kept, "frames ->", path)
