"""Episodic memory, step 0 (docs/EPISODES.md): what vivid moments the robot's days hold, and whether yesterday's
moments tell anything about today's.

  ./py episodes.py 2026-10-25 2026-10-26 ...     moments per day and hour, their kinds, and the day-to-day check

A moment: an outcome the night already scores (reward_v4: pet, tap, talk, near, shake, tilt, poke, left alone), or the
analyzer startled (surprise or threat onset). Its context: the hour, the room lit or dark, someone around (near, a
voice or motion in the last 2 minutes), the robot's own act in the 4 s before. The question for a memory: does the
chance of a good (or bad) moment in a context on one day predict it on the next day better than the plain daily rate?
"""
import sys

import numpy as np

import diary as D
from reward_v4 import REWARDS, reward

DT = 0.05
GOOD = {"pet", "tap", "talk", "near"}


def moments(day):
    """-> list of (unix hour of the day, kind, context tuple) and the context's time per (hour, lit, around) in s."""
    t, _ = D.load(day)
    if t is None:
        return [], {}
    v = t["v"]
    c = lambda n: np.nan_to_num(v[:, D.col(n)])
    r, _ = reward(t)
    valid = ((t["flags"] & D.FLAG_DRIVES) > 0) & ((t["flags"] & D.FLAG_NIGHT) == 0)
    hour = np.floor(c("raw_hour")).astype(int)
    lit = c("ch_light") >= 0.04
    contact = (c("raw_near_raw") > 50) | (c("ch_voice") > 0.05) | (c("ch_motion") > 0.05) | (c("raw_head_ev") > 0)
    around = np.convolve(contact.astype(np.int32), np.ones(int(120 / DT), np.int32))[:len(v)] > 0
    acted = np.convolve(((c("said") > 0) | (c("gesture") > 0)).astype(np.int32), np.ones(int(4 / DT), np.int32))[:len(v)] > 0
    sur = c("snn_surprise") > 0.5
    thr = c("snn_threat") > 0.45
    startle = (sur & ~np.r_[False, sur[:-1]]) | (thr & ~np.r_[False, thr[:-1]])
    out = []
    # the reward's kinds by their weight (pet 1, tap / talk 0.5, near 0.3, shake -1, tilt / poke -0.5, left alone -0.3)
    by_weight = {}
    for k, w in REWARDS.items():
        by_weight.setdefault(w, []).append(k)
    for i in np.flatnonzero(valid & ((r != 0) | startle)):
        kind = "startle" if r[i] == 0 else "/".join(by_weight.get(float(r[i]), ["mixed"]))
        out.append((int(hour[i]), kind, (bool(lit[i]), bool(around[i]), bool(acted[i]))))
    ctx = {}
    for h in range(24):
        for l in (False, True):
            for a in (False, True):
                n = int((valid & (hour == h) & (lit == l) & (around == a)).sum())
                if n:
                    ctx[(h, l, a)] = n * DT
    return out, ctx


def rates(ms, ctx, good):
    """Per context (hour, lit, around): good (or bad) moments per hour of that context."""
    cnt = {}
    for h, kind, (l, a, _) in ms:
        if (any(k in GOOD for k in kind.split("/")) if good else (kind != "startle" and not any(k in GOOD for k in kind.split("/")))):
            cnt[(h, l, a)] = cnt.get((h, l, a), 0) + 1
    return {k: cnt.get(k, 0) / (s / 3600) for k, s in ctx.items() if s >= 300}


if __name__ == "__main__":
    days = sys.argv[1:]
    per = {}
    for day in days:
        ms, ctx = moments(day)
        per[day] = (ms, ctx)
        kinds = {}
        for _, k, _ in ms:
            kinds[k] = kinds.get(k, 0) + 1
        hours = sum(ctx.values()) / 3600
        print(f"== {day}: {len(ms)} moments in {hours:.1f} h by day ({len(ms) / max(hours, 1e-6):.0f} an hour): "
              + ", ".join(f"{k} {n}" for k, n in sorted(kinds.items(), key=lambda kv: -kv[1])))
        good = [m for m in ms if any(k in GOOD for k in m[1].split("/"))]
        if good:
            print(f"   good moments: someone around {100 * np.mean([m[2][1] for m in good]):.0f} %, lit "
                  f"{100 * np.mean([m[2][0] for m in good]):.0f} %, right after the robot's own act "
                  f"{100 * np.mean([m[2][2] for m in good]):.0f} %")
    # does a day's rate per context predict the next day's better than the next day's overall rate?
    for good in (True, False):
        errs_ctx, errs_flat = [], []
        for d0, d1 in zip(days, days[1:]):
            r0, r1 = rates(*per[d0], good), rates(*per[d1], good)
            ms1, ctx1 = per[d1]
            common = [k for k in r1 if k in r0]
            if not common:
                continue
            ms0, ctx0 = per[d0]
            flat = sum(r0[k] * ctx0[k] for k in r0) / sum(ctx0[k] for k in r0)  # yesterday's plain rate
            w = np.array([ctx1[k] for k in common])
            errs_ctx.append(np.average([abs(r0[k] - r1[k]) for k in common], weights=w))
            errs_flat.append(np.average([abs(flat - r1[k]) for k in common], weights=w))
        if errs_ctx:
            print(f"== {'good' if good else 'bad'} moments per hour of a context, the next day: yesterday's rate in that "
                  f"context off by {np.mean(errs_ctx):.2f}, yesterday's plain rate off by {np.mean(errs_flat):.2f} "
                  f"({'memory helps' if np.mean(errs_ctx) < np.mean(errs_flat) else 'memory does not help'})")
