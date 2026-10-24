"""Records the real microphone as Auditory NN training data, in labelled parts.

  ./py record_audio.py LABEL SECONDS      # LABEL: silence | speech | clap | sharp | continuous | tonal | mixed
Reads 'audioin:' lines from the robot's serial port (firmware with kAudioRecording = true) for SECONDS
(split into <= 60 s captures), and writes data/audio/real_<label>_<n>.bin in the training format
(input[100], label[9]). Everything captured gets the label's classes; direction label is 0 except
for 'speech'/'clap' where the balance feature itself is the best guess (left unlabelled = masked below).
"""
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CLASSES = ["silence", "speech", "clap", "sharp", "continuous", "low", "high", "tonal"]
LABELS = {"silence": [0], "speech": [1], "clap": [2], "sharp": [3], "continuous": [4], "tonal": [7], "mixed": []}
line = re.compile(r" audio: (\d+) ([0-9a-f]{400})")

label, seconds = sys.argv[1], int(sys.argv[2])
rows = []
t_end = time.time() + seconds
while time.time() < t_end:
    s = min(55, int(t_end - time.time()) + 1)
    out = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "capture_serial.py"), "--seconds", str(s)],
                         capture_output=True, text=True, errors="ignore").stdout
    for m in line.finditer(out):
        h = m.group(2)
        v = np.array([int(h[i:i + 4], 16) for i in range(0, 400, 4)], np.int32)
        v = np.where(v >= 0x8000, v - 0x10000, v) / 1000.0
        rows.append(v.astype(np.float32))
x = np.asarray(rows, np.float32)
y = np.zeros((len(x), 9), np.float32)
for c in LABELS[label]:
    y[:, c] = 1.0
if label != "silence":
    y[:, 0] = 0.0
os.makedirs(os.path.join(HERE, "data", "audio"), exist_ok=True)
n = 0
while os.path.exists(os.path.join(HERE, "data", "audio", f"real_{label}_{n}.bin")):
    n += 1  # never overwrite a session
path = os.path.join(HERE, "data", "audio", f"real_{label}_{n}.bin")
np.concatenate([x, y], 1).tofile(path)
print(len(x), "frames ->", path, "| own_motion mean", x[:, 99].mean().round(2), "self mean", x[:, 98].mean().round(2))
