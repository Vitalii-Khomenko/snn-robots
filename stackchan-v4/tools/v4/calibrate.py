"""Stage 0: the 13 channels computed after the fact from the robot's diary, and how much time each spends per zone.

  ./py calibrate.py [days...]          default: the last 7 full days; report -> out/calibration.txt

The diary holds the v3 bus, not the raw sensors, so some channels are approximations here (see APPROX): touch,
sound, voice, tilt, hunger, tiredness and loneliness are close to what the firmware will compute; shake is the old
shake event, motion the Vision NN's activity; proximity and light are not in the diary at all (stage 1 logs them).
"""
import datetime
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "life"))
import lifedata as L  # noqa: E402
from channels import encode, new_state  # noqa: E402
from spec_v4 import CHANNELS, ZONE_TABLE, ZONES, zones_of  # noqa: E402

APPROX = {"shake": "old shake event (2 s pulse) as 0.4 g", "motion": "Vision NN activity x 0.3",
          "near": "not in the diary", "light": "not in the diary"}


def raw_from_diary(v, unix):
    c = lambda n: np.nan_to_num(v[:, L.col(n)])
    T = len(v)

    def edges(x, thr=0.5):
        on = x > thr
        e = np.zeros(T, bool)
        e[1:] = on[1:] & ~on[:-1]
        return e

    # the microphone's onset detector (and its loud events): 2-4 an hour in a quiet room, where the Auditory NN's
    # clap/sharp classes fire 20-40 times an hour on nothing (2026-10-21)
    onset_ev = edges(c("onset"))
    loud_ev = edges(c("loud"), 0.6)
    # (loud events: intensity 0.6 .. 1.0 -> sharp strength 0.5 .. 1.5, as BrainLoop::tick)
    sharp_ev = np.where(loud_ev, 0.5 + (c("loud") - 0.6) / 0.4, np.where(onset_ev, np.maximum(c("onset"), 0.5), 0.0))
    hours = np.array([np.nan if not np.isfinite(u) else
                      (lambda d: d.hour + d.minute / 60 + d.second / 3600)(datetime.datetime.fromtimestamp(u))
                      for u in unix[::200]])
    hours = np.repeat(hours, 200)[:T]
    moving = (np.abs(c("yaw_vel")) + np.abs(c("pitch_vel"))) > 0.01
    return dict(energy=c("energy"), speech=c("speech"), sharp_ev=sharp_ev,
                head_ev=edges(c("touch_head")).astype(np.float32), screen_ev=edges(c("touch_screen")).astype(np.float32),
                accel_sd=np.where(c("shake") > 0.5, 0.4, 0.0), imu_pitch=c("imu_pitch"), imu_roll=c("imu_roll"),
                head_pitch_deg=c("pitch") * 35.0, imu_ok=np.ones(T), motion_frac=c("activity") * 0.3,
                frame_ok=(~moving).astype(np.float32), battery=c("battery"), stamina=c("stamina"),
                hour=np.nan_to_num(hours, nan=12.0), clock_ok=np.isfinite(hours).astype(np.float32)), hours


def main():
    days = sys.argv[1:] or sorted(d for d in os.listdir(os.path.join(L.ROOT, "data", "life")) if d[:2] == "20")[-8:-1]
    chans, hrs = [], []
    for day in days:
        state = new_state()
        for h in range(24):
            ticks, _ = L.load(day, hours=[h])
            if ticks is None:
                continue
            raw, hours = raw_from_diary(ticks["values"], ticks["unix"])
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
