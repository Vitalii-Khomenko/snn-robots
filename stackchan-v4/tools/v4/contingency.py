"""The sense of "I did that" (docs/CONTINGENCY.md): does the world answer the robot's acts?

  ./py contingency.py 2026-10-27 [...]           the detector over the day (the twin), and the robot's own ct_* columns
  ./py contingency.py --measure 2026-10-25 [...] step 0: the plain statistics of acts and answers over whole days
  ./py contingency.py --self 2026-10-27 [...]    me or not me: which sense events follow the robot's own acts at once

Detector (step 1; main/v4/contingency_v4.cpp is the robot's, parity_contingency.py): every tick it takes what the
robot did (a call, a gesture, a face change) and what came from outside (a gentle pet or tap, a new speech onset that is
not its own voice, someone coming near, motion seen while the head is still). An answer 0.5-4 s after an act is
credited to the latest act; how often each kind of act is answered (leaky counts, ~5 min) is set against the chance of
an answer landing in such a window anyway (the background rate at moments with nobody acting, ~30 min). Acts within
3 s after an answer are reactions and do not count; nothing counts at night or when the networks do not drive.
Outputs (diary ct_*): agency, answered (a pulse, ~3 s), ignored (+0.1 per unanswered call or gesture with someone
around, an answer resets it, fades ~5 min),
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
REACT_MS, NO_ACT_MS, AROUND_MS, ACT_QUIET_MS = 3000, 5000, 120000, 1000
TAU_NOW, TAU_BG, TAU_PULSE, TAU_IGNORED = 300.0, 1800.0, 3.0, 300.0
PRIOR, IGNORE_STEP = 3.0, 0.1
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
        if not answered and k != 2:  # a face is not a bid
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
                speech > F(0.5), near > F(150),
                motion > F(0.15) and abs(F(yaw) - F(self.yaw)) < F(0.3)]
        self.yaw = yaw
        onset = [False] * 5
        for a in range(5):
            onset[a] = cond[a] and not self._within(self.last_true[a], now, QUIET_MS[a])
            if cond[a]:
                self.last_true[a] = now
        if onset[0] or onset[1] or onset[3] or near > F(50) or motion > F(0.1):
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
                if a in (0, 1, 2):  # petted, tapped, talked to: not ignored
                    self.ignored = 0.0
                if best is not None:
                    best[3].add(a)
                    self.pulse = max(self.pulse, WEIGHT[a])
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


class AnswerWorld:
    """For training the main network (teacher_nn.rollout, train_nn_v4.label; CONTINGENCY step 3): people who answer the
    robot's acts or not, in stretches of 1-5 minutes, seen through the Detector at 10 Hz -> the 6 inputs it gets.
      away        nobody there
      random      someone there, answering now and then whatever the robot does (the control)
      responsive  someone there who answers some kinds of act (a call / a gesture / a face) with p 0.5-0.9 after 0.6-3 s,
                  at most once in ANSWER_GAP_S (a person's pace: not every call of a ping-pong of calls and pets)
      still       someone there who never answers (the still face)
    The answers are a pet, speech or a tap. The lives are recorded, so an answer is also written into the analyzer's
    numbers that follow (inject: the mean trace of a touch / a voice onset in the analyzer's dataset) - on the robot an
    answer is felt as well as counted, and the network must learn to keep its game going through the touch."""
    KINDS, SHARE = ["away", "random", "responsive", "still"], [0.35, 0.15, 0.3, 0.2]
    ANSWER_GAP_S = 6.0

    def __init__(self, rng):
        self.rng = rng
        self.det = Detector()
        self.ms = 0
        self.pending = []
        self.prev_say, self.prev_gesture, self.prev_emo = False, False, -1
        self.out = [0.0] * len(CONTINGENCY)
        self.answers = (False, False, False)
        self.next_answer_ms = 0
        self._stretch()

    def _stretch(self):
        r = self.rng
        self.kind = self.KINDS[int(r.choice(len(self.KINDS), p=self.SHARE))]
        self.left = r.uniform(60.0, 300.0)
        self.p = r.uniform(0.5, 0.9)
        self.acts = set(r.choice(3, size=int(r.integers(1, 4)), replace=False).tolist())
        self.lat = (r.uniform(0.6, 1.2), r.uniform(1.5, 3.0))
        self.rate = 1.0 / r.uniform(15.0, 60.0)
        self.answer_p = r.dirichlet([2.0, 2.0, 1.0])  # pet, speech, tap

    def step(self, said, gesture, emo):
        """What the robot did this 10 Hz step (a call starts, a gesture starts, the shown emotion) -> the 6 numbers."""
        r = self.rng
        self.left -= 0.1
        if self.left <= 0:
            self._stretch()
            self.pending = []
        call = said and not self.prev_say
        gest = gesture and not self.prev_gesture
        face = self.prev_emo >= 0 and emo != self.prev_emo
        self.prev_say, self.prev_gesture, self.prev_emo = said, gesture, emo
        if self.kind == "responsive":
            for k, did in enumerate((call, gest, face)):
                if did and k in self.acts and self.ms >= self.next_answer_ms and not self.pending and r.random() < self.p:
                    at = self.ms + int(1000 * r.uniform(*self.lat))
                    self.pending.append((at, int(r.choice(3, p=self.answer_p))))
                    self.next_answer_ms = at + int(1000 * self.ANSWER_GAP_S)
        due = [a for t, a in self.pending if t <= self.ms]
        self.pending = [(t, a) for t, a in self.pending if t > self.ms]
        if self.kind == "random" and r.random() < self.rate * 0.1:
            due.append(int(r.choice(3, p=self.answer_p)))
        pet, speech, tap = (0 in due), (1 in due), (2 in due)
        self.answers = (pet, speech, tap)
        near = 80.0 if self.kind != "away" else 0.0
        row = (True, call, gest, int(emo), 1.0 if pet else 0.0, 1.0 if tap else 0.0, 0.3 if pet else 0.0,
               0.2 if tap else 0.0, near, 0.9 if speech else 0.0, 0.0, 0.0)
        self.out, _ = self.det.step(self.ms, row)
        self.ms += 100
        return self.out


# what an answer looks like to the analyzer, 10 Hz from the onset (the mean of 2900 head / 700 screen touch onsets in
# out/nn_snn.npz, 2026-10-27, rounded and made to fade; a voice onset by analogy): {output: trace}, applied as a maximum
_FADE = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3]
TRACES = {
    "pet": {"sal_touch_head": [0.35 * f for f in _FADE], "hed_touch_head": [0.68] * 15,
            "surprise": [0.55, 0.5, 0.42, 0.35, 0.3]},
    "tap": {"sal_touch_screen": [0.37 * f for f in _FADE], "hed_touch_screen": [0.6] * 15,
            "surprise": [0.5, 0.45, 0.38, 0.32, 0.28]},
    "speech": {"sal_voice": [0.3 * f for f in _FADE], "hed_voice": [0.6] * 15},
}


def inject(S, t, answers):
    """Write the answers of AnswerWorld.step (pet, speech, tap) into one life's analyzer numbers S [T, 32] from t on."""
    from spec_v4 import SNN_OUT
    for did, kind in zip(answers, ("pet", "speech", "tap")):
        if not did:
            continue
        for name, trace in TRACES[kind].items():
            j = SNN_OUT.index(name)
            n = min(len(trace), len(S) - t)
            if n > 0:
                S[t:t + n, j] = np.maximum(S[t:t + n, j], np.asarray(trace[:n], S.dtype))


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


def self_links(days):
    """Me or not me (CONTINGENCY 3.6): for every kind of sense event, how often it comes in the 0.5 s after the robot's
    own act (a call, a gesture, a head move starting) against any moment. A link far above chance and that fast is the
    robot itself (its voice, its servos), not the world. Strictly after: an event in the act's own tick came first
    (the ear runs before the executor) and the act was the answer to it (2026-10-27: "a sharp sound after a call", x15,
    were sharp sounds the robot answered with a call in the same tick)."""
    def ahead(x, a, b):
        cs = np.concatenate([[0], np.cumsum(x)])
        i = np.arange(len(x))
        return (cs[np.minimum(i + b, len(x))] - cs[np.minimum(i + a, len(x))]) > 0

    for day in days:
        t, _ = D.load(day)
        v = t["v"]
        c = lambda n: np.nan_to_num(v[:, D.col(n)])
        valid = ((t["flags"] & D.FLAG_DRIVES) > 0) & ((t["flags"] & D.FLAG_NIGHT) == 0)
        yaw = c("head_yaw_deg") + c("head_pitch_deg")
        moving = np.abs(np.diff(yaw, prepend=yaw[0])) > 0.05
        acts = {"call": onsets(c("said") > 0, int(1 / DT)), "gesture": onsets(c("gesture") > 0, int(1 / DT)),
                "head move": onsets(moving, int(1 / DT))}
        ev = {"ear speech": onsets(c("ear_speech") > 0.5, int(2 / DT)) if not np.isnan(v[:, D.col("ear_speech")]).all()
              else None,
              "speech": onsets(c("raw_speech") > 0.5, int(2 / DT)),
              "sharp": onsets(c("raw_sharp_ev") > 0, int(1 / DT)),
              "pet": onsets(c("raw_head_ev") > 0, int(2 / DT)), "tap": onsets(c("raw_screen_ev") > 0, int(2 / DT)),
              "near": onsets(c("raw_near_raw") > 150, int(3 / DT)), "motion": onsets(c("ch_motion") > 0.15, int(3 / DT))}
        print(f"{day}: P(event in the 0.5 s after the act) / P(at any moment); >> 1 and fast = the robot itself")
        print("  event        " + "".join(f"{k:>14s}" for k in acts))
        for name, e in ev.items():
            if e is None:
                continue
            e = e & valid
            fast = ahead(e, 1, int(0.5 / DT) + 1)  # ticks i+1 .. i+10
            base = fast[valid].mean()
            cells = []
            for a in acts.values():
                m = a & valid
                p = fast[m].mean() if m.any() else np.nan
                cells.append(f"{p:6.3f} x{p / base if base > 0 else 0:5.1f}")
            print(f"  {name:12s} " + "".join(f"{s:>14s}" for s in cells) + f"   ({int(e.sum())} events)")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--measure"]:
        measure(sys.argv[2:])
    elif sys.argv[1:2] == ["--self"]:
        self_links(sys.argv[2:])
    else:
        for d in sys.argv[1:]:
            report(d)
