"""The reward of brain v4's life (stage 6, BRAIN_V4_PLAN.txt 8.4): how the people around answer what the robot does,
read from the diary (diary.py), tick by tick (20 Hz).

Good: a stroke on the head (+1), a gentle tap on the face (+0.5), being talked to (+0.5: speech of >= 1 s, not the
tail of the robot's own voice), someone coming close
(+0.3). Bad: being shaken (-1), tilted (-0.5), poked hard (head or face pad in its anger zone, -0.5), and calling or
gesturing at someone who then leaves the robot alone for a minute (-0.3). Night ticks (the night rules) and ticks
when the networks did not drive the body count nothing.

  r, events = reward(ticks)            r [T] per tick, events: {name: count}
  ./py reward_v4.py 2026-10-25         a day's reward events
"""
import sys

import numpy as np

import diary as D

DT = 0.05
REWARDS = {"pet": 1.0, "tap": 0.5, "talk": 0.5, "near": 0.3, "shake": -1.0, "tilt": -0.5, "poke": -0.5,
           "left_alone": -0.3}


def onsets(on, quiet_ticks):
    """Ticks where `on` becomes true after at least `quiet_ticks` ticks of false."""
    on = np.asarray(on, bool)
    out = np.zeros(len(on), bool)
    last_on = -10 ** 9
    for i in np.flatnonzero(on):
        if i - last_on > quiet_ticks and (i == 0 or not on[i - 1]):
            out[i] = True
        last_on = i
    return out


def reward(ticks):
    """-> r [T] per tick, {event: count}."""
    ev, valid = events(ticks)
    r = np.zeros(len(valid), np.float32)
    counts = {}
    for k, w in REWARDS.items():
        e = ev[k] & valid
        r[e] += w
        counts[k] = int(e.sum())
    return r, counts


def events(ticks):
    """-> {event: bool [T]} (the ticks where each rewarded event happens) and valid [T] (by day, networks driving)."""
    v = ticks["v"]
    c = lambda n: np.nan_to_num(v[:, D.col(n)])
    T = len(v)
    ev = {}
    ev["pet"] = (c("raw_head_ev") > 0) & (c("ch_touch_head") < 0.7)
    ev["tap"] = (c("raw_screen_ev") > 0) & (c("ch_touch_screen") < 0.4)
    # talk: speech that lasts (>= 1 s of the next 2 s), not in the 5 s after the robot's own call (its voice's tail
    # reads as speech: a third of the raw speech onsets on 2026-10-24)
    sp = c("raw_speech") > 0.5
    sustained = np.convolve(sp[::-1].astype(np.int32), np.ones(int(2 / DT), np.int32))[:T][::-1] >= int(1 / DT)
    own_voice = np.convolve((c("said") > 0).astype(np.int32), np.ones(int(5 / DT), np.int32))[:T] > 0
    # since 2026-10-27 morning the firmware holds its own voice's tail and voiceless sounds out of raw_speech (voice
    # check, kOwnVoiceTailMs; the ear_voice column exists from then): a reply 1 s after a call is a reply (2026-10-28,
    # an audit: the 5 s mask threw away exactly the quick answers the contingency is about)
    own_voice &= ~np.isfinite(v[:, D.col("ear_voice")])
    ev["talk"] = onsets(sp, int(2 / DT)) & sustained & ~own_voice
    ev["near"] = onsets(c("raw_near_raw") > 150, int(3 / DT))
    ev["shake"] = onsets(c("raw_accel_sd") > 0.3, int(2 / DT))
    ev["tilt"] = onsets(c("ch_tilt") > 0.3, int(5 / DT))
    ev["poke"] = onsets((c("ch_touch_head") >= 0.85) | (c("ch_touch_screen") >= 0.65), int(5 / DT))
    # contact: anything that shows somebody is with the robot
    contact = ev["pet"] | ev["tap"] | (sp & ~own_voice) | (c("raw_near_raw") > 50) | (c("ch_motion") > 0.05)
    recent = np.convolve(contact.astype(np.int32), np.ones(int(15 / DT), np.int32))[:T] > 0   # within the last 15 s
    ahead = np.convolve(contact[::-1].astype(np.int32), np.ones(int(60 / DT), np.int32))[:T][::-1] > 0
    # (ahead[i]: contact in [i, i + 60 s]; the act itself must be answered after it, so look from the next tick)
    ahead_next = np.concatenate([ahead[1:], [False]])
    acted = (c("said") > 0) | (c("gesture") > 0)
    # ... and only when that minute was seen: the same boot, no gap (2026-10-28, an audit: a call in the last minute of
    # a recording, before a reboot or a gap in the diary counted as unanswered)
    n_ahead = int(60 / DT)
    later = np.minimum(np.arange(T) + n_ahead, T - 1)
    seen = (later - np.arange(T) == n_ahead) & (ticks["boot"][later] == ticks["boot"]) & \
        (np.abs((ticks["ms"][later].astype(np.int64) - ticks["ms"].astype(np.int64)) - 60000) < 5000)
    ev["left_alone"] = acted & recent & ~ahead_next & seen
    valid = ((ticks["flags"] & D.FLAG_DRIVES) > 0) & ((ticks["flags"] & D.FLAG_NIGHT) == 0)
    return ev, valid


if __name__ == "__main__":
    for day in sys.argv[1:]:
        t, _ = D.load(day)
        if t is None:
            print(day, "no brain v4 ticks")
            continue
        r, n = reward(t)
        print(f"{day}: reward {r.sum():+.1f} over {len(r) / 72000:.1f} h; " + ", ".join(f"{k} {v}" for k, v in n.items()))
