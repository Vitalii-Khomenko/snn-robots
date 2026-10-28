"""The head's own jerks and hangs, from the robot's 5 s stats lines (docs/HARDWARE.md, servos).

  ./py jerks.py 2026-10-27 [--since 17:15] [...]

A jerk: a 5 s window where the gyro saw a peak over 300 deg/s and 40 deg more real rotation than was commanded (the
diary's head angles are the commanded model's: a servo that runs off on its own is seen only by the gyro). A hang:
the servo watchdog power-cycled the rail (commanded motion, the gyro saw none). Both per servo rail switch-on, by hour.

  ./py jerks.py --pose 2026-10-28    where the head really points (2026-10-28, an audit A13: "no observed pose")

The servos' own position cannot be read on this board, but the IMU sits in the head: still, gravity tells the head's
real pitch (the diary's raw_imu_roll, x 180 deg). --pose fits it against the commanded pitch and shows how far apart
the two are, overall and around the hangs. The yaw has no such reference (gravity does not change with it).
"""
import collections
import datetime
import re
import sys

import diary as D

STATS = re.compile(r"I \((\d+)\) runtime: stats:.*?cmd_travel=(\d+)deg real_rotation=(\d+)deg .*?gyro_max=(\d+)")
POWER = re.compile(r"I \((\d+)\) runtime: power:.*servo_rail=(\d+)/(\d+)ms")
GYRO_PEAK, EXTRA_DEG = 300, 40


def count(day, since=None):
    _, lines = D.load(day)
    t0 = None
    if since:
        h, m = map(int, since.split(":"))
        t0 = datetime.datetime.strptime(day, "%Y-%m-%d").replace(hour=h, minute=m).timestamp()
    by_hour = collections.defaultdict(collections.Counter)
    for t, _, _, text in lines:
        if t0 is not None and t < t0:
            continue
        h = datetime.datetime.fromtimestamp(t).hour
        if (m := STATS.search(text)):
            cmd, real, peak = (int(x) for x in m.groups()[1:])
            by_hour[h]["windows"] += 1
            by_hour[h]["jerks"] += peak > GYRO_PEAK and real > cmd + EXTRA_DEG
        elif (m := POWER.search(text)):
            by_hour[h]["rail_ons"] += int(m.group(2))
        elif "servo rail power cycle" in text:
            by_hour[h]["hangs"] += 1
        elif "servos: a jerk of" in text:  # the firmware's own count after a switch-on (from 2026-10-27 evening)
            by_hour[h]["logged"] += 1
    return by_hour


def pose(day):
    """The head's real pitch from gravity against the commanded one: the fit, the error overall and before / after the
    servo rail's power cycles."""
    import numpy as np
    t, lines = D.load(day)
    v, boot, ms = t["v"], t["boot"], t["ms"].astype(np.int64)
    cmd = v[:, D.col("head_pitch_deg")]
    real = v[:, D.col("raw_imu_roll")] * 180.0
    ok = (v[:, D.col("raw_imu_ok")] > 0.5) & np.isfinite(real) & np.isfinite(cmd)
    k, c = np.linalg.lstsq(np.vstack([cmd[ok], np.ones(ok.sum())]).T, real[ok], rcond=None)[0]
    err = (real - c) / k - cmd  # the real pitch in the command's degrees, minus the command
    base = np.median(err[ok])
    e = np.abs(err - base)
    print(f"== {day}: gravity {k:+.3f} x commanded pitch {c:+.1f} deg; |real - commanded| median {np.median(e[ok]):.1f} deg, "
          f"p90 {np.percentile(e[ok], 90):.1f}, > 5 deg {100 * np.mean(e[ok] > 5):.1f} % of {ok.sum()} ticks")
    before, after = [], []
    for _, b, m, text in lines:
        if "servo rail power cycle" in text:
            i = np.flatnonzero((boot == b) & (ms >= m - 6000) & (ms < m) & ok)
            j = np.flatnonzero((boot == b) & (ms > m + 4000) & (ms < m + 8000) & ok)
            if len(i):
                before.append(np.median(e[i]))
            if len(j):
                after.append(np.median(e[j]))
    if before:
        print(f"   {len(before)} hangs: > 5 deg off in the 6 s before {100 * np.mean(np.array(before) > 5):.0f} %, "
              f"4-8 s after {100 * np.mean(np.array(after) > 5):.0f} % (most hangs are the yaw servo: not seen here)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--pose":
        for day in args[1:]:
            pose(day)
        sys.exit(0)
    since = None
    if "--since" in args:
        i = args.index("--since")
        since = args[i + 1]
        del args[i:i + 2]
    for day in args:
        hours = count(day, since)
        total = sum(hours.values(), collections.Counter())
        ons = max(total["rail_ons"], 1)
        print(f"== {day}{' since ' + since if since else ''}: {total['jerks']} jerks, {total['hangs']} hangs, "
              f"{total['rail_ons']} rail switch-ons ({total['jerks'] / ons * 100:.1f} % / {total['hangs'] / ons * 100:.1f} %)")
        for h in sorted(hours):
            c = hours[h]
            print(f"  {h:02d}: jerks {c['jerks']:3d} (logged at a switch-on {c['logged']})  hangs {c['hangs']:3d}  "
                  f"rail ons {c['rail_ons']:4d}  windows {c['windows']}")
