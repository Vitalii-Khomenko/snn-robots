"""How independent is the robot? Runs a scenario with nobody around (default: idle_endurance, one hour) and prints
the measures of its inner life: faces, calls, glances, invites, drives.   ./py autonomy_report.py [scenario] [minutes]
A robot that is alone should still do something every few minutes: look around, babble, change its face, ask for
company; and it should sleep at night."""
import collections
import csv
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SIM = os.path.join(ROOT, "simulator", "build", "stackchan_sim")
scn = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "simulator", "scenarios", "idle_endurance.scn")
with tempfile.TemporaryDirectory() as out:
    cmd = [SIM, scn, "--out", out] + (["--minutes", sys.argv[2]] if len(sys.argv) > 2 else [])
    subprocess.run(cmd, check=True, capture_output=True)
    rows = [r for r in csv.DictReader(open(os.path.join(out, "trace.csv"))) if None not in r.values()]
    log = open(os.path.join(out, "log.txt"), errors="ignore").read()
f = lambda k: np.array([float(r[k]) for r in rows])
t = f("t_s")
hrs = (t[-1] - t[0]) / 3600.0
emo = [r["emotion"] for r in rows]
cnt = collections.Counter(emo)
p = np.array(list(cnt.values()), float) / len(emo)
print(f"{hrs:.2f} h simulated")
print("faces:", {k: round(v / len(emo), 2) for k, v in cnt.most_common(8)}, f"| face entropy {-(p * np.log2(p)).sum():.2f} bits")
print("face changes per minute: %.1f" % (sum(a != b for a, b in zip(emo, emo[1:])) / ((t[-1] - t[0]) / 60)))
for k in ("curiosity", "boredom", "arousal", "annoyance", "social", "sleep"):
    v = f(k)
    print(f"  {k:10s} mean {v.mean():.2f}  p5 {np.percentile(v, 5):.2f}  p95 {np.percentile(v, 95):.2f}")
for name, pat in (("calls", "call: start"), ("glances (moves >= 12 deg)", "executor: move"), ("invites (nod series)", "invite: nodding"),
                  ("startles", "startle: flinch")):
    n = len(re.findall(pat, log))
    print(f"  {name:28s} {n:5d}  ({n / hrs:6.1f} per hour)")
print("  asleep (sleep > 0.55): %.0f %% of the time" % (100 * (f("sleep") > 0.55).mean()))
