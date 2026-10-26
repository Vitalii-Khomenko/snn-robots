"""C++ ChannelEncoder (main/v4/channel_encoder.cpp) against its twin tools/v4/channels.py on a random raw stream."""
import os
import subprocess
import sys

import numpy as np

from channels import RAW, encode

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def random_raw(T, rng):
    r = {n: np.zeros(T) for n in RAW}
    r["energy"] = np.clip(rng.normal(0.1, 0.1, T), 0, 1)
    r["speech"] = (rng.random(T) < 0.2) * rng.random(T)
    r["sharp_ev"] = (rng.random(T) < 0.003) * rng.uniform(0.5, 1.5, T)
    r["head_ev"] = (rng.random(T) < 0.01) * 1.0
    r["head_hold"] = (np.sin(np.arange(T) / 300) > 0.95) * 1.0
    r["screen_ev"] = (rng.random(T) < 0.004) * 1.0
    r["accel_sd"] = np.abs(rng.normal(0, 0.05, T)) + (np.arange(T) % 4000 < 60) * 0.6
    tiltseg = ((np.arange(T) // 3000) % 5 == 3) * 0.4
    r["imu_pitch"] = rng.normal(0, 0.003, T) + tiltseg
    r["imu_roll"] = 0.4 + rng.normal(0, 0.02, T)
    r["imu_ok"] = np.ones(T)
    r["near_raw"] = np.clip(rng.normal(0, 50, T), 0, None) + ((np.arange(T) // 2000) % 3 == 1) * 400
    r["near_ok"] = np.ones(T)
    r["als"] = 300 + 200 * np.sin(np.arange(T) / 5000)
    r["als_ok"] = np.ones(T)
    r["frame_ok"] = (np.arange(T) % 3 == 0) * np.where((np.arange(T) // 500) % 2 == 0, 1.0, 0.5)  # still / turning
    r["motion_frac"] = np.clip(rng.normal(0.05, 0.05, T), 0, 1)
    r["motion_x"], r["motion_y"] = rng.uniform(-1, 1, T), rng.uniform(-1, 1, T)
    # a bright spot that comes and goes (the head turning to a lamp) and moves
    seen = ((np.arange(T) // 700) % 3 != 0) * 1.0
    r["light_str"] = seen * np.clip(0.6 + 0.3 * np.sin(np.arange(T) / 900), 0, 1)
    r["light_x"], r["light_y"] = seen * np.sin(np.arange(T) / 400), seen * 0.3
    r["head_yaw"] = 50 * np.sin(np.arange(T) / 1300)  # the head looking round: the spot's place in the room moves
    r["battery"] = np.linspace(1, 0.3, T)
    r["stamina"] = np.ones(T) * 0.8
    r["hour"] = (20.5 + np.arange(T) * 0.05 / 3600 * 30) % 24
    r["clock_ok"] = np.ones(T)
    return r


def main():
    os.makedirs(OUT, exist_ok=True)
    exe = os.path.join(OUT, "parity_channels")
    subprocess.run(["g++", "-std=c++17", "-O2", "-I", os.path.join(HERE, "..", "..", "main", "v4"),
                    os.path.join(HERE, "parity_channels.cpp"),
                    os.path.join(HERE, "..", "..", "main", "v4", "channel_encoder.cpp"), "-o", exe], check=True)
    T = 40000
    raw = random_raw(T, np.random.default_rng(1))
    R = np.stack([raw[n] for n in RAW], 1).astype(np.float32)
    R.tofile(os.path.join(OUT, "raw.bin"))
    subprocess.run([exe, os.path.join(OUT, "raw.bin"), os.path.join(OUT, "ch.bin")], check=True)
    cpp = np.fromfile(os.path.join(OUT, "ch.bin"), np.float32).reshape(T, -1)
    py = encode({n: R[:, i] for i, n in enumerate(RAW)})
    err = np.abs(cpp - py).max(0)
    print("max |C++ - Python| per channel:", np.round(err, 5))
    print("channel ranges:", np.round(py.min(0), 2), np.round(py.max(0), 2))
    ok = err.max() < 2e-3
    print("PARITY", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
