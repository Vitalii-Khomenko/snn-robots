"""Stage 0: the 13 channels of the robot's real days and how much time each spends per zone.

  ./py calibrate.py [days...]          default: the last 7 full days; report -> out/calibration.txt

Brain v4's own diary (diary.py, from 2026-10-24) logs the raw readings the channels are made from. Older days hold
platform B's bus (legacy_diary.py): there some channels are approximations (see APPROX).
"""
import os
import sys

import numpy as np

import diary as D
import legacy_diary as LD
from channels import encode, new_state
from spec_v4 import CHANNELS, ZONES, zones_of

HERE = os.path.dirname(os.path.abspath(__file__))
APPROX = {"shake": "old days: the old shake event (2 s pulse) as 0.4 g", "motion": "old days: Vision NN activity x 0.3",
          "near": "old days: not logged", "light": "old days: not logged"}


def raw_hour(day, h):
    """channels.encode's input for one hour of a day and the hour of day per tick, from whichever diary layout the
    hour has; (None, None) for an hour without ticks."""
    ticks, _ = D.load(day, hours=[h])
    if ticks is not None:
        return D.raw_for_channels(ticks)
    v, unix = LD.load(day, hours=[h])
    if v is None:
        return None, None
    return LD.raw_for_channels(v, unix)


def main():
    days = sys.argv[1:] or sorted(d for d in os.listdir(D.LIFE) if d[:2] == "20")[-8:-1]
    chans, hrs = [], []
    for day in days:
        state = new_state()
        for h in range(24):
            raw, hours = raw_hour(day, h)
            if raw is None:
                continue
            chans.append(encode(raw, state))
            hrs.append(hours)
        print(day, "done", flush=True)
    X = np.concatenate(chans)
    H = np.concatenate(hrs)
    day_mask = (H >= 7) & (H < 21)
    night_mask = (H >= 22) | (H < 6)
    lines = [f"STAGE 0 CALIBRATION  days {days[0]} .. {days[-1]}  ({len(X) / 72000:.1f} h of ticks)", "",
             "channel        p50   p90   p99   max   | zone share, day 07-21 (night 22-06)"]
    for i, ch in enumerate(CHANNELS):
        x = X[:, i]
        p = np.percentile(x, [50, 90, 99])
        zs = zones_of(ch, x)
        share = []
        for z in ZONES:
            if z in zs:
                d = zs[z][day_mask].mean() * 100
                n = zs[z][night_mask].mean() * 100
                share.append(f"{z} {d:4.1f}% ({n:4.1f}%)")
        note = f"   [{APPROX[ch]}]" if ch in APPROX else ""
        lines.append(f"{ch:13s} {p[0]:.2f}  {p[1]:.2f}  {p[2]:.2f}  {x.max():.2f}  | " + ", ".join(share) + note)
    out = os.path.join(HERE, "out")
    os.makedirs(out, exist_ok=True)
    text = "\n".join(lines) + "\n"
    open(os.path.join(out, "calibration.txt"), "w").write(text)
    np.save(os.path.join(out, "calibration_channels.npy"), X[::10])
    np.save(os.path.join(out, "calibration_hours.npy"), H[::10])
    print(text)


if __name__ == "__main__":
    main()
