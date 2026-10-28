#!/usr/bin/env python3
"""Records the robot's microphone on the home PC (main/net/sound_tap.h, GET /listen; docs/CONTINGENCY.md step 2).

  python3 tools/robot/listen.py [SECONDS]         default 60; at most 600 per request

Writes data/sound/<day>/<HHMMSS>.wav (24 kHz mono, as the robot hears it) and <HHMMSS>.npz: per 128-sample block the
robot's ms since boot (the diary's clock: tools/v4/sound_check.py lines it up with the ear's speech, the calls and the
head), the flags (1 speaker mute, 2 own-motion threshold) and the boot's PC time of block 0. The sound stays on this
PC (data/ is not in git).
"""
import os
import struct
import sys
import time
import urllib.request
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ota  # noqa: E402

BLOCK = struct.Struct("<IB3x128h")


def main():
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    req = urllib.request.Request(f"http://{ota.address()}/listen?s={seconds}", headers={"X-Token": ota.token()})
    t0 = time.time()
    raw = bytearray()
    with urllib.request.urlopen(req, timeout=seconds + 30) as r:
        head = r.read(16)
        magic, rate, frames, start_ms = struct.unpack("<4sIII", head)
        if magic != b"SND1" or frames != 128:
            sys.exit(f"unexpected stream: {magic} {rate} {frames}")
        while True:
            chunk = r.read(65536)
            if not chunk:
                break
            raw += chunk
    n = len(raw) // BLOCK.size
    rec = np.frombuffer(bytes(raw[:n * BLOCK.size]), np.dtype([("ms", "<u4"), ("flags", "u1"), ("pad", "u1", 3),
                                                                ("s", "<i2", (128,))]))
    day = time.strftime("%Y-%m-%d", time.localtime(t0))
    folder = os.path.join(ota.ROOT, "data", "sound", day)
    os.makedirs(folder, exist_ok=True)
    stem = os.path.join(folder, time.strftime("%H%M%S", time.localtime(t0)))
    with wave.open(stem + ".wav", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(rec["s"].tobytes())
    np.savez(stem + ".npz", ms=rec["ms"], flags=rec["flags"], pc_start=t0, robot_start_ms=start_ms, rate=rate)
    # the blocks come in DMA bursts (their ms jitter), so loss = the time covered minus the sound received
    span_s = (int(rec["ms"][-1]) - int(rec["ms"][0])) / 1000.0 if n > 1 else 0.0
    lost = max(int(round((span_s - (n - 1) * 128 / rate) * rate / 128)), 0)
    print(f"{stem}.wav: {n} blocks, {n * 128 / rate:.1f} s, ~{lost} blocks lost; muted {np.mean(rec['flags'] & 1):.2f}, "
          f"own motion {np.mean((rec['flags'] & 2) > 0):.2f}; peak {np.abs(rec['s']).max()}")


if __name__ == "__main__":
    main()
