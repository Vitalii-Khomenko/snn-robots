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
from reward_v4 import REWARDS, events, reward

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


# ---- the remembered moments (step 2; main/v4/memory_v4.h): what the night gives the robot to remember
KINDS = ["pet", "tap", "talk", "near", "shake", "tilt", "poke", "fright"]  # memory_v4.cpp kindName
GOOD_W = {"pet": 1.0, "tap": 0.6, "talk": 0.6, "near": 0.4}
BAD_W = {"shake": 1.0, "poke": 0.7, "tilt": 0.6, "fright": 0.6}
N_GOOD, N_BAD = 16, 8   # learned.MEM_ROWS = 24
OLDER = 0.8             # a day older weighs this much less
SAME_MIN = 10.0         # moments of one kind closer than this are one moment


def vivid(day, days_ago=0):
    """A day's vivid moments as memory rows: [minute, valence, strength, yaw, around, lit, kind, date]. Strength: the
    kind's weight x (0.6 + 0.4 x the analyzer's arousal then: emotion picks what is kept, McGaugh) x OLDER^days_ago;
    around and lit as the firmware sees them (BrainLoop: somebody heard, moving or near in the last 2 minutes; the
    light channel >= 0.04)."""
    t, _ = D.load(day)
    if t is None:
        return []
    v = t["v"]
    c = lambda n: np.nan_to_num(v[:, D.col(n)])
    ev, valid = events(t)
    thr = c("snn_threat") > 0.6
    ev["fright"] = thr & ~np.r_[False, thr[:-1]]
    speech = np.where(np.isfinite(v[:, D.col("ear_speech")]), c("ear_speech"), c("raw_speech"))
    somebody = (speech > 0.4) | (c("ch_motion") > 0.1) | (c("ch_near") > 0.3)
    around = np.convolve(somebody.astype(np.int32), np.ones(int(120 / DT), np.int32))[:len(v)] > 0
    minute, yaw = c("raw_hour") * 60.0, c("head_yaw_deg")
    lit, arousal = c("ch_light") >= 0.04, np.clip(c("snn_arousal"), 0, 1)
    date = int(day[2:4] + day[5:7] + day[8:10])
    rows = []
    for kind, w in list(GOOD_W.items()) + list(BAD_W.items()):
        good = kind in GOOD_W
        cand = []
        for i in np.flatnonzero(ev[kind] & valid & (minute > 0)):
            strength = w * (0.6 + 0.4 * arousal[i]) * OLDER ** days_ago
            cand.append([minute[i], 1.0 if good else -1.0, strength, yaw[i], float(around[i]), float(lit[i]),
                         float(KINDS.index(kind)), float(date)])
        cand.sort(key=lambda r: -r[2])
        kept = []
        for r in cand:
            if all(abs(r[0] - k[0]) >= SAME_MIN for k in kept):
                kept.append(r)
        rows += kept
    return rows


def table(days):
    """The moments to remember from these days (the last is the newest): the N_GOOD strongest good ones and the N_BAD
    strongest bad ones."""
    rows = []
    for k, day in enumerate(days):
        rows += vivid(day, len(days) - 1 - k)
    good = sorted([r for r in rows if r[1] > 0], key=lambda r: -r[2])[:N_GOOD]
    bad = sorted([r for r in rows if r[1] < 0], key=lambda r: -r[2])[:N_BAD]
    return np.array(good + bad, np.float32).reshape(-1, 8)


def describe(row):
    m, d = int(row[0]), int(row[7])
    return (f"{(d // 100) % 100:02d}-{d % 100:02d} {m // 60:02d}:{m % 60:02d} {KINDS[int(row[6])]} "
            f"({'good' if row[1] > 0 else 'bad'} {row[2]:.2f}, yaw {row[3]:.0f}, {'someone around' if row[4] else 'alone'}, "
            f"{'lit' if row[5] else 'dark'})")


def rates(ms, ctx, good):
    """Per context (hour, lit, around): good (or bad) moments per hour of that context."""
    cnt = {}
    for h, kind, (l, a, _) in ms:
        if (any(k in GOOD for k in kind.split("/")) if good else (kind != "startle" and not any(k in GOOD for k in kind.split("/")))):
            cnt[(h, l, a)] = cnt.get((h, l, a), 0) + 1
    return {k: cnt.get(k, 0) / (s / 3600) for k, s in ctx.items() if s >= 300}


if __name__ == "__main__":
    if sys.argv[1:2] == ["--table"]:
        for r in table(sys.argv[2:]):
            print(describe(r))
        sys.exit(0)
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
