"""Converts 'dataset:' lines captured from the robot's serial console
(tools/capture_serial.py, with kDatasetLogging = true in robot_runtime.cpp)
into a rollout file the policy trainer reads (data/policy/real/ep*/bus.bin).

  ./py serial_dataset.py capture.log data/policy/real/ep1
"""
import os
import re
import sys

import numpy as np

line = re.compile(r"dataset: (\d+) ([0-9a-f]+)")
src, out = sys.argv[1], sys.argv[2]
rows = []
for text in open(src, errors="ignore"):
    m = line.search(text)
    if not m:
        continue
    hexs = m.group(2)
    if len(hexs) != 126 * 4:
        continue
    v = np.array([int(hexs[i:i + 4], 16) for i in range(0, len(hexs), 4)], np.int32)
    v = np.where(v >= 0x8000, v - 0x10000, v) / 1000.0
    rows.append(v)
os.makedirs(out, exist_ok=True)
np.asarray(rows, np.float32).tofile(os.path.join(out, "bus.bin"))
print(len(rows), "records ->", out)
