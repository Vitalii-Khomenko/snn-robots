"""The body and its needs (docs/BODY.md): what the robot's own body tells it.

  ./py body.py --measure 2026-10-26 [...]   step 0: what the body channels did over whole days, hour by hour
  ./py body.py 2026-10-27 [...]             the sleep pressure and the body budget over the day: the twin replayed on
                                            the diary, and the robot's own (raw_stamina = 1 - pressure, raw_budget)

Sleep pressure (step 1; main/v4/body_v4.cpp is the robot's, parity_body.py): Borbely's two-process model. Process S
builds up while awake, towards 1 with a time constant of TAU_WAKE_H, faster the more the robot does (head travel,
calls, arousal over ~10 minutes: the load, 1 on a usual day of 2026-10-26); it drains while asleep (the owner's night,
or a nap: the sleepy face shown) with TAU_SLEEP_H. Process C, the clock, holds a usual day's pressure off (the alerting
signal of Edgar, Dement & Fuller 1993): the tired channel is 0.75 * night_ramp + channels.TIRED_FELT * the pressure
above usual(hour). A busy day makes the robot tired and a dark room then puts it to sleep; a usual day is as before;
a nap or the night clears it.

The body budget (step 2, BodyV4 too): energy 0..1, spent by the load, fear and too much, the servo rail's power cycles;
refilled by sleep and a gentle touch; costs halve with someone near (Coan). The hunger channel is the larger of
1 - battery and 1 - budget.
"""
import math
import sys

import numpy as np

import diary as D
from spec_v4 import DIARY, EMOTIONS

TAU_WAKE_H, TAU_SLEEP_H = 12.0, 3.0  # adults ~18 h and ~4 h (Borbely & Achermann); a child's run faster
EFFORT_TAU_S = 600.0                 # the load is the last ~10 minutes'
REF_DEG_S, REF_SAY_S, REF_AROUSAL = 2.3, 5.5 / 60.0, 0.16  # a usual day (diary 2026-10-27 08-16 h, after contingency step 3)
W_DEG, W_SAY, W_AROUSAL = 0.4, 0.3, 0.3
TIRED_FELT = 2.0                     # channels.TIRED_FELT (channels imports this file)
LOAD_GAIN, LOAD_LO, LOAD_HI = 2.0, 0.25, 3.0  # load = 1 + LOAD_GAIN * (effort - 1): play counts, idle rests
DAY_START_H, DAY_END_H = 6.0, 21.75  # the owner's day (the night is the robot's sleep)
DEG_TICK_MAX = 20.0                  # a jump of the logged head angle (a reboot, a stall) is not travel
# the body budget (step 2): spent by the load (empty in SPEND_H at load 1; NEAR_SHARE with someone near or talking,
# Coan's social baseline), by fear and too much (threat / over above STRESS_FROM: STRESS_S at full) and by a servo
# rail power cycle (HURT; ~20 an hour by day on 2026-10-27, kept small); refilled asleep (REFILL_SLEEP_H) and by a
# gentle touch: each pet or tap (contingency.py's: not a poke) fills PET_SHARE of what is missing
SPEND_H, NEAR_SHARE, STRESS_FROM, STRESS_S, HURT = 50.0, 0.5, 0.3, 600.0, 0.0007
REFILL_SLEEP_H, PET_SHARE, BUDGET_START = 2.0, 0.06, 0.85
BUDGET_FELT = 0.7  # channels: hunger = 1 - budget / BUDGET_FELT (a usual day stays above it)
SLEEPY = EMOTIONS.index("sleepy")
F = np.float32


def usual(hour):
    """The pressure of a usual day (load 1) at this hour: the boot's start (a reboot keeps no state)."""
    a = math.exp(-(24.0 - DAY_END_H + DAY_START_H) / TAU_SLEEP_H)
    b = math.exp(-(DAY_END_H - DAY_START_H) / TAU_WAKE_H)
    evening = (1.0 - b) / (1.0 - a * b)
    morning = evening * a
    if DAY_START_H <= hour < DAY_END_H:
        return 1.0 - (1.0 - morning) * math.exp(-(hour - DAY_START_H) / TAU_WAKE_H)
    return evening * math.exp(-((hour - DAY_END_H) % 24.0) / TAU_SLEEP_H)


class Body:
    """The twin of BodyV4, tick by tick: step(dt, clock_ok, hour, moved_deg, said, arousal, asleep, threat, over,
    pet, nearby, hurt) -> pressure; .b is the budget."""

    def __init__(self):
        self.s = 0.3
        self.b = BUDGET_START
        self.clock = False
        self.deg, self.say = REF_DEG_S, REF_SAY_S  # a usual load until it knows better
        self.arousal = REF_AROUSAL

    def load(self):
        e = W_DEG * self.deg / REF_DEG_S + W_SAY * self.say / REF_SAY_S + W_AROUSAL * self.arousal / REF_AROUSAL
        return min(LOAD_HI, max(LOAD_LO, 1.0 + LOAD_GAIN * (e - 1.0)))

    def step(self, dt, clock_ok, hour, moved_deg, said, arousal, asleep, threat=0.0, over=0.0, pet=False,
             nearby=False, hurt=False):
        if clock_ok and not self.clock:
            self.s = usual(F(hour))
        self.clock = self.clock or clock_ok
        if dt <= 0:
            return self.s
        k = 1.0 - math.exp(-dt / EFFORT_TAU_S)
        self.deg += (min(moved_deg, DEG_TICK_MAX) / dt - self.deg) * k
        self.say += ((1.0 / dt if said else 0.0) - self.say) * k
        self.arousal += (arousal - self.arousal) * k
        if asleep:
            self.s *= math.exp(-dt / (TAU_SLEEP_H * 3600.0))
        else:
            self.s = 1.0 - (1.0 - self.s) * math.exp(-dt * self.load() / (TAU_WAKE_H * 3600.0))
        # the body budget
        if hurt:
            self.b -= HURT
        if asleep:
            self.b += (1.0 - self.b) * (1.0 - math.exp(-dt / (REFILL_SLEEP_H * 3600.0)))
        else:
            stress = max(0.0, max(threat, over) - STRESS_FROM) / (1.0 - STRESS_FROM)
            self.b -= (self.load() / (SPEND_H * 3600.0) * (NEAR_SHARE if nearby else 1.0) + stress / STRESS_S) * dt
        if pet:
            self.b += (1.0 - self.b) * PET_SHARE
        self.b = min(1.0, max(0.0, self.b))
        return self.s


COLUMNS = ["dt", "clock_ok", "hour", "moved_deg", "said", "arousal", "asleep", "threat", "over", "pet", "nearby",
           "hurt"]


def inputs(t, lines=()):
    """A diary (and its log lines: the servo rail's power cycles) -> per tick the COLUMNS [T, 12] and the boot of each
    tick: what BodyV4 got at the end of that tick (BrainLoop::tick; the diary row is written right after it)."""
    v = t["v"]
    I = {n: i for i, n in enumerate(DIARY)}
    col = lambda n: np.nan_to_num(v[:, I[n]])
    boot, ms = t["boot"], t["ms"].astype(np.int64)
    new = np.concatenate([[True], boot[1:] != boot[:-1]])
    dt = np.where(new, 0.0, np.clip(np.diff(ms, prepend=ms[0]) / 1000.0, 0.0, 1.0))
    yaw, pitch = col("head_yaw_deg"), col("head_pitch_deg")
    moved = np.where(new, 0.0, np.abs(np.diff(yaw, prepend=yaw[0])) + np.abs(np.diff(pitch, prepend=pitch[0])))
    driving, night = (t["flags"] & 1) > 0, (t["flags"] & 2) > 0
    asleep = night | (driving & (col("face").astype(int) == SLEEPY))
    pet = ((col("raw_head_ev") > 0) & (col("ch_touch_head") < 0.7)) | ((col("raw_screen_ev") > 0) & (col("ch_touch_screen") < 0.4))
    nearby = (col("ch_near") > 0.1) | (col("ch_voice") > 0.05)
    hurt = np.zeros(len(v), bool)
    for _, b, m, text in lines:  # the next tick of that boot after the power cycle
        if "servo rail power cycle" in text:
            idx = np.flatnonzero((boot == b) & (ms >= m))
            if len(idx):
                hurt[idx[np.argmin(ms[idx])]] = True
    X = np.stack([dt, col("raw_clock_ok"), col("raw_hour"), moved, col("said") > 0.5, col("snn_arousal"), asleep,
                  col("snn_threat"), col("snn_over"), pet, nearby, hurt], 1)
    return X.astype(F), boot


def run(X, boot):
    """The twin over inputs(): -> pressure [T], load [T], budget [T] after each tick."""
    out, loads, budget = np.zeros(len(X), F), np.zeros(len(X), F), np.zeros(len(X), F)
    body = None
    for i in range(len(X)):
        if i == 0 or boot[i] != boot[i - 1]:
            body = Body()
        dt, clock, hour, moved, said, arousal, asleep, threat, over, pet, nearby, hurt = (float(x) for x in X[i])
        out[i] = body.step(dt, clock > 0, hour, moved, said > 0, arousal, asleep > 0, threat, over, pet > 0,
                           nearby > 0, hurt > 0)
        loads[i] = body.load()
        budget[i] = body.b
    return out, loads, budget


def replay(t, lines=()):
    return run(*inputs(t, lines))


def measure(days):
    """Step 0: the body channels and what fills them, hour by hour."""
    I = {n: i for i, n in enumerate(DIARY)}
    for day in days:
        t, lines = D.load(day)
        if t is None:
            continue
        v = t["v"]
        hour = v[:, I["raw_hour"]]
        print(f"== {day}: {len(v)} ticks")
        for c in ("raw_battery", "raw_stamina", "ch_hunger", "ch_tired", "ch_lonely", "snn_sleepiness"):
            x = v[:, I[c]]
            x = x[np.isfinite(x)]
            print(f"  {c:15s} min {x.min():.2f} med {np.median(x):.2f} p90 {np.percentile(x, 90):.2f} max {x.max():.2f}")
        p, load, budget = replay(t, lines)
        emo = np.argmax(np.nan_to_num(v[:, [I["nn_emo_" + e] for e in EMOTIONS]], nan=-1), 1)
        print("  hour  load  pressure  usual  tired_now  tired_new  sleepy%  lonely  budget")
        for h in range(24):
            m = np.floor(hour) == h
            if m.sum() < 1000:
                continue
            tired_now = np.nanmean(v[m, I["ch_tired"]])
            felt = np.mean(np.maximum(0.0, p[m] - np.array([usual(x) for x in hour[m][::200]]).mean()))
            print(f"  {h:02d}   {load[m].mean():.2f}   {p[m].mean():.2f}      {usual(h + 0.5):.2f}   {tired_now:.2f}"
                  f"       {min(1.0, tired_now + TIRED_FELT * felt):.2f}       {np.mean(emo[m] == SLEEPY) * 100:3.0f}"
                  f"     {np.nanmean(v[m, I['ch_lonely']]):.2f}    {budget[m].mean():.2f}")


def report(days):
    I = {n: i for i, n in enumerate(DIARY)}
    for day in days:
        t, lines = D.load(day)
        if t is None:
            continue
        v = t["v"]
        p, _, bud = replay(t, lines)
        robot = 1.0 - v[:, I["raw_stamina"]]
        rb = v[:, I["raw_budget"]]  # NaN before step 2
        bud = np.concatenate([[np.nan], bud[:-1]])
        hour = v[:, I["raw_hour"]]
        p = np.concatenate([[np.nan], p[:-1]])  # the robot's channels read the last tick's pressure
        new = np.concatenate([[True], t["boot"][1:] != t["boot"][:-1]])
        p[new] = bud[new] = np.nan
        ok = np.isfinite(robot) & np.isfinite(p) & (robot > 0.01)  # ~0: before this step's firmware (old stamina 0.9999)
        if ok.any():
            print(f"== {day}: pressure, max |twin - robot| {np.abs(p[ok] - robot[ok]).max():.4f} over {ok.sum()} ticks")
        okb = np.isfinite(rb) & np.isfinite(bud)
        if okb.any():
            print(f"== {day}: budget, max |twin - robot| {np.abs(bud[okb] - rb[okb]).max():.4f} over {okb.sum()} ticks")
        print(f"== {day}  hour: pressure twin / robot's (1 - raw_stamina), budget twin / robot's")
        for h in range(24):
            m = np.floor(hour) == h
            if m.sum() >= 1000:
                print(f"  {h:02d}: {np.nanmean(p[m]):.3f} / {np.nanmean(robot[m]):.3f}   {np.nanmean(bud[m]):.3f} / "
                      f"{np.nanmean(rb[m]) if np.isfinite(rb[m]).any() else float('nan'):.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--measure":
        measure(sys.argv[2:])
    elif len(sys.argv) > 1:
        report(sys.argv[1:])
    else:
        print(__doc__)
