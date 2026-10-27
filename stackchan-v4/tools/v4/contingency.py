"""Contingency in the diary (docs/CONTINGENCY.md, step 0): does the world answer the robot's acts?

For every spontaneous act (a call, a gesture, a face change; nothing answered in the 10 s before, so the robot's own
answers to touches do not count), with someone around (an answer in the last 2 minutes), by day:
P(an answer 0.5-4 s after the act) against P(the same) after moments with no act in the 5 s before. Answers: pet,
tap, someone near, speech (not in the 5 s after the robot's own call), motion onset while the head is still.

  ./py contingency.py 2026-10-25 2026-10-26
"""
import sys
import numpy as np
import diary as D
from reward_v4 import onsets, DT

def conv_ahead(x, a, b):  # any x in (i+a, i+b]
    T = len(x); c = np.concatenate([[0], np.cumsum(x)])
    i = np.arange(T); lo = np.minimum(i + a, T); hi = np.minimum(i + b, T)
    return (c[hi] - c[lo]) > 0
def conv_back(x, n):
    T = len(x); c = np.concatenate([[0], np.cumsum(x)]); i = np.arange(T)
    return (c[i] - c[np.maximum(i - n, 0)]) > 0

tot = {}
for day in sys.argv[1:]:
    t, _ = D.load(day)
    v = t["v"]; c = lambda n: np.nan_to_num(v[:, D.col(n)])
    valid = ((t["flags"] & D.FLAG_DRIVES) > 0) & ((t["flags"] & D.FLAG_NIGHT) == 0)
    sp = c("raw_speech") > 0.5
    own = conv_back((c("said") > 0), int(5 / DT))
    still = np.abs(np.diff(c("head_yaw_deg"), prepend=0)) < 0.3
    ans = {
        "pet": (c("raw_head_ev") > 0) & (c("ch_touch_head") < 0.7),
        "tap": (c("raw_screen_ev") > 0) & (c("ch_touch_screen") < 0.4),
        "near": onsets(c("raw_near_raw") > 150, int(3 / DT)),
        "talk": onsets(sp & ~own, int(2 / DT)),
        "motion": onsets((c("ch_motion") > 0.15) & still, int(3 / DT)),
    }
    anyans = np.zeros(len(v), bool)
    for a in ans.values(): anyans |= a
    face = c("face")
    acts = {
        "call": onsets(c("said") > 0, int(1 / DT)),
        "gesture": onsets(c("gesture") > 0, int(1 / DT)),
        "face": np.concatenate([[False], face[1:] != face[:-1]]),
    }
    allact = np.zeros(len(v), bool)
    for a in acts.values(): allact |= a
    around = conv_back(anyans, int(120 / DT)) & valid      # someone answered something in the last 2 min
    base_t = around & ~conv_back(allact, int(5 / DT)) & ~allact
    a0, a1 = int(0.5 / DT), int(4 / DT)
    for an_name, an in list(ans.items()) + [("any", anyans)]:
        hit = conv_ahead(an, a0, a1)
        pb = hit[base_t].mean() if base_t.any() else np.nan
        for ac_name, ac in acts.items():
            m = ac & around & ~conv_back(anyans, int(10 / DT))   # spontaneous: nothing answered in the 10 s before
            if m.sum() < 5: continue
            p = hit[m].mean()
            k = (ac_name, an_name); s = tot.setdefault(k, [0, 0, 0, 0])
            s[0] += hit[m].sum(); s[1] += m.sum(); s[2] += hit[base_t].sum(); s[3] += base_t.sum()
    print(day, f"talk onsets/h {ans["talk"][valid].sum()/(valid.sum()*DT/3600):.0f}, pet/h {ans["pet"][valid].sum()/(valid.sum()*DT/3600):.1f}, hours around {around.sum()*DT/3600:.1f}, acts " + ", ".join(f"{k} {int((a & around).sum())}" for k, a in acts.items()))
print("act      answer   P(ans|act)  P(ans|no act)  dP    n")
for (ac, an), (h, n, hb, nb) in sorted(tot.items()):
    print(f"{ac:8s} {an:7s}  {h/n:.3f}       {hb/nb:.3f}        {h/n-hb/nb:+.3f}  {n}")
