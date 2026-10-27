"""The sense of "I did that" (docs/CONTINGENCY.md): does the world answer the robot's acts?

  ./py contingency.py 2026-10-27 [...]           the detector over the day (the twin), and the robot's own ct_* columns
  ./py contingency.py --measure 2026-10-25 [...] step 0: the plain statistics of acts and answers over whole days

Detector (step 1; main/v4/contingency_v4.cpp is the robot's, parity_contingency.py): every tick it takes what the
robot did (a call, a gesture, a face change) and what came from outside (a gentle pet or tap, a new speech onset that is
not its own voice, someone coming near, motion seen while the head is still). An answer 0.5-4 s after an act is
credited to the latest act; how often each kind of act is answered (leaky counts, ~5 min) is set against the chance of
an answer landing in such a window anyway (the background rate at moments with nobody acting, ~30 min). Acts within
3 s after an answer are reactions and do not count; nothing counts at night or when the networks do not drive.
Outputs (diary ct_*): agency, answered (a pulse, ~3 s), ignored (acts unanswered with someone around, fades ~10 min),
and per act kind (call, gesture, face) how much more often than chance it is answered (answers weighted).

Step 0 (--measure): for every spontaneous act (nothing answered in the 10 s before) with someone around,
P(an answer 0.5-4 s after) against the same after moments with no act in the 5 s before.
"""
import math
import sys

import numpy as np

import diary as D
from reward_v4 import onsets, DT
from spec_v4 import CONTINGENCY

ACTS = ["call", "gesture", "face"]
ANSWERS = ["pet", "tap", "speech", "near", "motion"]
WEIGHT = [1.0, 0.8, 0.6, 0.5, 0.3]
QUIET_MS = [2000, 2000, 2000, 3000, 3000]
WIN_LO_MS, WIN_HI_MS = 500, 4000
REACT_MS, NO_ACT_MS, AROUND_MS, OWN_VOICE_MS, ACT_QUIET_MS = 3000, 5000, 120000, 5000, 1000
TAU_NOW, TAU_BG, TAU_PULSE, TAU_IGNORED = 300.0, 1800.0, 3.0, 600.0
PRIOR, IGNORE_STEP = 3.0, 0.15
WINDOW_S = (WIN_HI_MS - WIN_LO_MS) / 1000.0
F = np.float32
INPUTS = ["active", "said", "gesture", "face", "raw_head_ev", "raw_screen_ev", "ch_touch_head", "ch_touch_screen",
          "raw_near_raw", "raw_speech", "ch_motion", "raw_head_yaw"]


class Detector:
    """The twin of ContingencyV4, tick by tick: step(ms, row) with row in the order of INPUTS -> (out [6], credit)."""

    def __init__(self):
        self.pending = []                # [ms, act, around, answered set]
        self.n = [0.0] * 3
        self.h = [[0.0] * 5 for _ in range(3)]
        self.bg_n = [0.0] * 5
        self.bg_s = 0.0
        self.last_true = [None] * 5
        self.pulse = self.ignored = 0.0
        self.last_ms = None
        self.last_said = self.last_gesture = self.last_act = self.last_answer = self.last_around = None
        self.face = -1
        self.yaw = 0.0

    @staticmethod
    def _within(then, now, ms):
        return then is not None and now - then <= ms

    def _finish(self, p):
        ms, k, around, answered = p
        if not around:
            return
        self.n[k] += 1.0
        for a in range(5):
            self.h[k][a] += 1.0 if a in answered else 0.0
        if not answered:
            self.ignored += (1.0 - self.ignored) * IGNORE_STEP

    def step(self, now, row):
        active, said, gesture, face, head_ev, screen_ev, touch_head, touch_screen, near, speech, motion, yaw = row
        dt = 0.0 if self.last_ms is None else min(max((now - self.last_ms) / 1000.0, 0.0), 1.0)
        self.last_ms = now
        kn, kb = math.exp(-dt / TAU_NOW), math.exp(-dt / TAU_BG)
        self.n = [v * kn for v in self.n]
        self.h = [[v * kn for v in r] for r in self.h]
        self.bg_n = [v * kb for v in self.bg_n]
        self.bg_s *= kb
        self.pulse *= math.exp(-dt / TAU_PULSE)
        self.ignored *= math.exp(-dt / TAU_IGNORED)
        credit = None

        act = [False, False, False]
        if said:
            act[0] = not self._within(self.last_said, now, ACT_QUIET_MS)
            self.last_said = now
        if gesture:
            act[1] = not self._within(self.last_gesture, now, ACT_QUIET_MS)
            self.last_gesture = now
        act[2] = self.face >= 0 and face != self.face
        self.face = face

        cond = [head_ev > 0 and touch_head < F(0.7), screen_ev > 0 and touch_screen < F(0.4),
                speech > F(0.5) and not self._within(self.last_said, now, OWN_VOICE_MS), near > F(150),
                motion > F(0.15) and abs(F(yaw) - F(self.yaw)) < F(0.3)]
        self.yaw = yaw
        onset = [False] * 5
        for a in range(5):
            onset[a] = cond[a] and not self._within(self.last_true[a], now, QUIET_MS[a])
            if cond[a]:
                self.last_true[a] = now
        if onset[0] or onset[1] or onset[3] or near > F(50):
            self.last_around = now
        around = self._within(self.last_around, now, AROUND_MS)

        if not active:
            self.pending = []
        else:
            no_act = not self._within(self.last_act, now, NO_ACT_MS)
            if around and no_act:
                self.bg_s += dt
            for a in range(5):
                if not onset[a]:
                    continue
                best = None
                for p in self.pending:
                    if WIN_LO_MS <= now - p[0] <= WIN_HI_MS and (best is None or (p[0], p[1]) > (best[0], best[1])):
                        best = p
                if best is not None:
                    best[3].add(a)
                    self.pulse = max(self.pulse, WEIGHT[a])
                    self.ignored = 0.0
                    if credit is None:
                        credit = (best[1], a, now - best[0])
                elif around and no_act:
                    self.bg_n[a] += 1.0
            keep = []
            for p in self.pending:
                if now - p[0] > WIN_HI_MS:
                    self._finish(p)
                else:
                    keep.append(p)
            self.pending = keep
            reaction = self._within(self.last_answer, now, REACT_MS)
            for k in range(3):
                if not act[k] or reaction:
                    continue
                if len(self.pending) >= 8:
                    oldest = min(self.pending, key=lambda p: p[0])
                    self.pending.remove(oldest)
                    self._finish(oldest)
                self.pending.append([now, k, around, set()])
        if any(onset):
            self.last_answer = now
        if any(act):
            self.last_act = now

        p0 = [1.0 - math.exp(-self.bg_n[a] / (self.bg_s + 60.0) * WINDOW_S) for a in range(5)]
        c = []
        for k in range(3):
            s = sum(WEIGHT[a] * ((self.h[k][a] + p0[a] * PRIOR) / (self.n[k] + PRIOR) - p0[a]) for a in range(5))
            c.append(min(max(s, -1.0), 1.0))
        out = [min(max(max(c), 0.0), 1.0), self.pulse, self.ignored] + c
        return out, credit


def inputs(ticks):
    """The detector's inputs from the diary: ms [T], rows [T, len(INPUTS)] (float32), boot [T]."""
    v = ticks["v"]
    c = lambda n: np.nan_to_num(v[:, D.col(n)]).astype(np.float32)
    active = (((ticks["flags"] & D.FLAG_DRIVES) > 0) & ((ticks["flags"] & D.FLAG_NIGHT) == 0)).astype(np.float32)
    rows = np.stack([active, (c("said") > 0).astype(np.float32), (c("gesture") > 0).astype(np.float32),
                     np.round(c("face"))] + [c(n) for n in INPUTS[4:]], 1)
    return ticks["ms"].astype(np.uint32), rows.astype(np.float32), ticks["boot"]


def run(ms, rows, boot):
    """The detector over a stream (a fresh one at every boot) -> out [T, 6], credits [(i, act, answer, latency ms)]."""
    out = np.zeros((len(ms), len(CONTINGENCY)), np.float32)
    credits = []
    det, b = None, None
    for i in range(len(ms)):
        if boot[i] != b:
            det, b = Detector(), boot[i]
        r = rows[i]
        o, cr = det.step(int(ms[i]), (r[0] > 0, r[1] > 0, r[2] > 0, int(r[3])) + tuple(r[4:]))
        out[i] = o
        if cr:
            credits.append((i,) + cr)
    return out, credits


def report(day):
    t, _ = D.load(day)
    if t is None:
        print(day, "no brain v4 ticks")
        return
    ms, rows, boot = inputs(t)
    out, credits = run(ms, rows, boot)
    day_t = rows[:, 0] > 0
    hours = day_t.sum() * DT / 3600
    print(f"{day}: {hours:.1f} h by day, {len(credits)} answered acts ({len(credits) / max(hours, 1e-9):.1f}/h)")
    for k, name in enumerate(ACTS):
        by = [sum(1 for c in credits if c[1] == k and c[2] == a) for a in range(len(ANSWERS))]
        print(f"  {name:8s} answered by " + ", ".join(f"{ANSWERS[a]} {n}" for a, n in enumerate(by)) +
              f"; {name} contingency mean {out[day_t, 3 + k].mean():+.3f}, p95 {np.percentile(out[day_t, 3 + k], 95):+.3f}")
    print(f"  agency > 0.3 {np.mean(out[day_t, 0] > 0.3):.3f} of the day, ignored > 0.5 {np.mean(out[day_t, 2] > 0.5):.3f}")
    live = ~np.isnan(t["v"][:, D.col("ct_agency")])
    if live.any():
        fw = t["v"][live][:, [D.col("ct_" + n) for n in CONTINGENCY]]
        err = np.abs(fw - out[live])
        print(f"  the robot's own ct_* ({live.sum()} ticks) vs the twin: mean |diff| {err.mean():.4f}, "
              f"max {err.max():.3f} (float16 diary inputs; small differences are expected)")


def measure(days):
    def ahead(x, a, b):  # any x in (i+a, i+b]
        T = len(x)
        cs = np.concatenate([[0], np.cumsum(x)])
        i = np.arange(T)
        return (cs[np.minimum(i + b, T)] - cs[np.minimum(i + a, T)]) > 0

    def back(x, n):
        cs = np.concatenate([[0], np.cumsum(x)])
        i = np.arange(len(x))
        return (cs[i] - cs[np.maximum(i - n, 0)]) > 0

    tot = {}
    for day in days:
        t, _ = D.load(day)
        v = t["v"]
        c = lambda n: np.nan_to_num(v[:, D.col(n)])
        valid = ((t["flags"] & D.FLAG_DRIVES) > 0) & ((t["flags"] & D.FLAG_NIGHT) == 0)
        sp = c("raw_speech") > 0.5
        own = back(c("said") > 0, int(5 / DT))
        still = np.abs(np.diff(c("head_yaw_deg"), prepend=0)) < 0.3
        ans = {"pet": (c("raw_head_ev") > 0) & (c("ch_touch_head") < 0.7),
               "tap": (c("raw_screen_ev") > 0) & (c("ch_touch_screen") < 0.4),
               "near": onsets(c("raw_near_raw") > 150, int(3 / DT)),
               "talk": onsets(sp & ~own, int(2 / DT)),
               "motion": onsets((c("ch_motion") > 0.15) & still, int(3 / DT))}
        anyans = np.zeros(len(v), bool)
        for a in ans.values():
            anyans |= a
        face = c("face")
        acts = {"call": onsets(c("said") > 0, int(1 / DT)), "gesture": onsets(c("gesture") > 0, int(1 / DT)),
                "face": np.concatenate([[False], face[1:] != face[:-1]])}
        allact = np.zeros(len(v), bool)
        for a in acts.values():
            allact |= a
        around = back(anyans, int(120 / DT)) & valid
        base_t = around & ~back(allact, int(5 / DT)) & ~allact
        for an_name, an in list(ans.items()) + [("any", anyans)]:
            hit = ahead(an, int(0.5 / DT), int(4 / DT))
            for ac_name, ac in acts.items():
                m = ac & around & ~back(anyans, int(10 / DT))
                if m.sum() < 5:
                    continue
                s = tot.setdefault((ac_name, an_name), [0, 0, 0, 0])
                s[0] += hit[m].sum(); s[1] += m.sum(); s[2] += hit[base_t].sum(); s[3] += base_t.sum()
        per_h = lambda x: x[valid].sum() / (valid.sum() * DT / 3600)
        print(day, f"talk onsets/h {per_h(ans['talk']):.0f}, pet/h {per_h(ans['pet']):.1f}, hours around "
                   f"{around.sum() * DT / 3600:.1f}, acts " + ", ".join(f"{k} {int((a & around).sum())}" for k, a in acts.items()))
    print("act      answer   P(ans|act)  P(ans|no act)  dP    n")
    for (ac, an), (h, n, hb, nb) in sorted(tot.items()):
        print(f"{ac:8s} {an:7s}  {h / n:.3f}       {hb / nb:.3f}        {h / n - hb / nb:+.3f}  {n}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--measure"]:
        measure(sys.argv[2:])
    else:
        for d in sys.argv[1:]:
            report(d)
