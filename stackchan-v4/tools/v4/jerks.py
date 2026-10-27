"""The head's own jerks and hangs, from the robot's 5 s stats lines (docs/HARDWARE.md, servos).

  ./py jerks.py 2026-10-27 [--since 17:15] [...]

A jerk: a 5 s window where the gyro saw a peak over 300 deg/s and 40 deg more real rotation than was commanded (the
diary's head angles are the commanded model's: a servo that runs off on its own is seen only by the gyro). A hang:
the servo watchdog power-cycled the rail (commanded motion, the gyro saw none). Both per servo rail switch-on, by hour.
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


if __name__ == "__main__":
    args = sys.argv[1:]
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
