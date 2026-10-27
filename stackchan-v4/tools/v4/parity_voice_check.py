"""C++ VoiceCheck (main/perception/voice_check.cpp) against its twin tools/v4/voice_check.py on a recording.

  ./py parity_voice_check.py data/sound/2026-10-27/063836
"""
import os
import subprocess
import sys

import numpy as np

import sound_check as SC
import voice_check as V

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
MAIN = os.path.join(HERE, "..", "..", "main")

if __name__ == "__main__":
    stem, s, meta = SC.load_sound(sys.argv[1])
    mono = np.round(s * 32768).astype(np.int16)
    mono.tofile(os.path.join(OUT, "voice_in.bin"))
    exe = os.path.join(OUT, "parity_voice_check")
    subprocess.run(["g++", "-std=c++17", "-O2", "-I", os.path.join(MAIN, "perception"),
                    os.path.join(HERE, "parity_voice_check.cpp"), os.path.join(MAIN, "perception", "voice_check.cpp"),
                    "-o", exe], check=True)
    subprocess.run([exe, os.path.join(OUT, "voice_in.bin"), os.path.join(OUT, "voice_out.bin")], check=True)
    c = np.fromfile(os.path.join(OUT, "voice_out.bin"), np.float32).reshape(-1, 3)
    vc = V.VoiceCheck()
    rows = []
    for _, v in vc.feed(mono):
        rows.append(v)
    py = np.array(rows, np.float32)
    n = min(len(py), len(c))
    err = np.abs(c[:n, 0] - py[:n]).max() if n else 1.0
    print(f"{os.path.basename(stem)}: {n} frames (C++ {len(c)}, twin {len(py)}), max |voice C++ - twin| {err:.2e}")
    ok = len(c) == len(py) and err < 0.02
    print("PARITY PASS" if ok else "PARITY FAIL")
    sys.exit(0 if ok else 1)
