"""Behaviour auditor: a day of the robot's life in the digital twin, a census of everything it does, and the rules every
behaviour must keep - so a behaviour that never happens, happens in the wrong situation, or a drive that does not do what it
should, is found by a tool and not by the owner after a week.

  ./py audit.py                      # the simulated day (7 two-hour stretches in parallel, ~15 min), report + exit code
  ./py audit.py --minutes 30         # shorter stretches (a quick look; some rules need the full length)
  ./py audit.py --robot-log FILE     # the same census and the log-based rules on a serial capture of the real robot

Three parts:
  1. census   - how often each behaviour happened and in which stretch: the 21 faces, invitations, startles, grumbles, the
                voice styles, glances, calls, LED colour families, sleep, learned links; a behaviour that never shows up in a
                whole day, or far too often, is a finding;
  2. rules    - invariants checked on every tick of the bus dump / every log event (an invitation needs somebody there, anger
                needs a reason, no songs, the slow needs really grow, ...): every bug the owner finds becomes one more rule;
  3. agreement- the spiking core's inner state against the drive teacher on the very same inputs, closed loop (a module that
                only works with the teacher's clean values - the sleep readout that drained the slow needs - shows up here).
Report: simulator/out/audit/report.txt
"""
import argparse
import collections
import glob
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import reward as R
from spec import ACTION, EFF, POLICY_INPUT, SNN, SNN_INPUT, pin, sin

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SIM = os.path.join(ROOT, "simulator", "build", "stackchan_sim")
OUT = os.path.join(ROOT, "simulator", "out", "audit")
N_IN, N_EFF, N_OUT = len(POLICY_INPUT), len(EFF), len(ACTION)
TICK = 0.05

FACES = ["Normal", "Happy", "VeryHappy", "Laughing", "Sad", "Crying", "Sleepy", "Surprised", "Angry", "Annoyed", "Love", "Shy",
         "Wink", "Curious", "Confused", "Thinking", "Bored", "Scared", "Dizzy", "Cool", "Sick"]
# every face has a situation of the day (2026-10-04: Love when stroked, Shy at a tap on the screen, Cool content after play,
# Confused when a sound has no source, Crying after a long loneliness, Sick with an empty battery); none may be missing
RARE_FACES = set()


# ---------------------------------------------------------------------------------------------------------- the day
def t(sec):
    return f"{int(sec // 3600)}:{int(sec % 3600 // 60):02d}:{sec % 60:05.2f}"


def stretches(minutes):
    """(name, clock, scenario text) for each stretch of the day. Every stretch starts with a freshly booted robot."""
    dur = minutes * 60
    head = lambda clock, seed, light="on", owner="home": (
        f"duration {t(dur)}\nclock {clock}\nseed {seed}\nroom_light {light}\nowner {owner}\nsunrise 07:00\nsunset 19:30\n"
        f"window 170 20 40 30\nlamp desk 100 12 on\n")
    out = []
    # night: the owner is home and asleep, the room dark and quiet
    out.append(("night", "00:30", head("00:30", 101, "off")))
    # morning: the owner gets up, talks to it, strokes it two or three times, makes coffee (noise), leaves for work
    L = [head("06:45", 102, "on")]
    L.append(f"at 2:00 person owner enter -30 1.5")
    for k in range(6):
        s0 = 240 + k * 300
        L.append(f"at {t(s0)} talk -30 3 0.8")
        L.append(f"at {t(s0 + 4)} touch_head 400")
        L.append(f"at {t(s0 + 7)} touch_head 400")
    L.append(f"at {t(min(dur - 600, 2400))} noise 400 0.9")
    L.append(f"at {t(min(dur - 300, 3000))} noise 0 0")
    L.append(f"at {t(min(dur - 120, 3600))} person owner leave")
    L.append(f"at {t(min(dur - 110, 3610))} owner away")
    out.append(("morning", "06:45", "\n".join(L)))
    # alone in the day: nobody, now and then a sound from outside
    L = [head("10:00", 103, "on", "away")]
    for k in range(int(dur // 900)):
        L.append(f"at {t(400 + k * 900)} bang {(-1) ** k * 100}")
    out.append(("alone_day", "10:00", "\n".join(L)))
    # the owner at the PC: in view, busy with something else, hardly any attention for long stretches
    L = [head("16:00", 104, "on")]
    L.append("at 0:05 person owner enter -60 1.2")
    L.append(f"at {t(dur * 0.6)} talk -60 2 0.7")
    L.append(f"at {t(dur * 0.6 + 3)} touch_head 400")
    out.append(("ignored_at_pc", "16:00", "\n".join(L)))
    # evening play: talks then strokes (a habit to learn), taps, claps, rare shakes and a lift, a burst of touches once
    L = [head("19:00", 105, "on")]
    L.append("at 1:00 person owner enter 0 1.0")
    s = 120
    k = 0
    while s < dur - 120:
        L.append(f"at {t(s)} talk 0 2.5 0.8")
        L.append(f"at {t(s + 3.5)} touch_head 600")
        if k % 4 == 1:
            L.append(f"at {t(s + 30)} tap_screen")
        if k % 5 == 2:
            L.append(f"at {t(s + 45)} clap 20 0.8")
        if k % 9 == 4:
            L.append(f"at {t(s + 60)} person owner come_close 0.5")
            L.append(f"at {t(s + 62)} shake 1.5")
        if k % 13 == 6:
            L.append(f"at {t(s + 90)} pick_up 45")
            L.append(f"at {t(s + 95)} put_down")
        if k == 3:
            L.append(f"every 0:02 from {t(s + 100)} until {t(s + 120)} touch_head 300")
        s += 150
        k += 1
    out.append(("evening_play", "19:00", "\n".join(L)))
    # quiet evening: the TV talks (speech and noise from one side), the owner sits there, goes to bed at 22:30
    L = [head("21:15", 106, "on")]
    L.append("at 0:30 person owner enter 40 2.0")
    L.append("at 1:00 noise 250 0.6")
    for k in range(int(dur // 40)):
        L.append(f"at {t(60 + k * 40)} talk 80 6 0.5")
    L.append(f"at {t(min(dur - 300, 4500))} person owner leave")
    L.append(f"at {t(min(dur - 290, 4510))} room_light off")
    L.append(f"at {t(min(dur - 280, 4520))} noise 0 0")
    out.append(("tv_evening", "21:15", "\n".join(L)))
    # the owner away for the night: alone in the dark, should sleep; at dawn the room gets light
    out.append(("alone_dawn", "05:30", head("05:30", 107, "off", "away")))
    # a nearly empty battery, unplugged, the owner around: it should look sick
    out.append(("low_battery", "14:00", head("14:00", 108, "on").replace("owner home\n", "owner home\nbattery 6\ncharging off\n")
                + "at 0:30 person owner enter 0 1.5\n"))
    return out


def run_stretch(job):
    name, text, minutes = job
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    scn = os.path.join(d, "scenario.scn")
    with open(scn, "w") as f:
        f.write(text + "\n")
    subprocess.run([SIM, scn, "--out", d, "--bus-dump", os.path.join(d, "bus.bin"), "--minutes", str(minutes)], check=True,
                   stdout=subprocess.DEVNULL)
    return name


# ---------------------------------------------------------------------------------------------------------- reading a run
class Run:
    def __init__(self, name, folder=None, log_text=None):
        self.name = name
        self.log = log_text if log_text is not None else open(os.path.join(folder, "log.txt"), errors="ignore").read()
        self.bus = None
        if folder and os.path.exists(os.path.join(folder, "bus.bin")):
            self.bus = R.load(os.path.join(folder, "bus.bin"))
        ready = re.search(r"I \((\d+)\) runtime: RobotRuntime ready", self.log)
        self.t0 = int(ready.group(1)) if ready else 0
        self.hours = (len(self.bus) * TICK / 3600.0) if self.bus is not None else self._log_hours()

    def _log_hours(self):
        ts = [int(x) for x in re.findall(r"^I \((\d+)\)", self.log, re.M)]
        return max(1e-6, (ts[-1] - ts[0]) / 3.6e6) if ts else 1e-6

    def events(self, pattern):
        """[(ms, match)] of log lines."""
        return [(int(m.group(1)), m) for m in re.finditer(r"^I \((\d+)\) " + pattern, self.log, re.M)]

    def tick(self, ms):
        return int(np.clip((ms - self.t0) / 50.0, 0, len(self.bus) - 1)) if self.bus is not None else None

    def c(self, n):
        return self.bus[:, pin(n)]

    def act(self, n):
        return self.bus[:, N_IN + N_EFF + ACTION.index(n)]


# ---------------------------------------------------------------------------------------------------------- census
def census(runs):
    rows = []
    face_total = collections.Counter()
    for r in runs:
        faces = collections.Counter(m.group(2) for _, m in r.events(r"executor: emotion=\d+ (\w+)"))
        face_total.update(faces)
        styles = collections.Counter(m.group(2) for _, m in r.events(r"executor: voice: .*style=(\w+)"))
        row = dict(stretch=r.name, hours=r.hours,
                   calls=len(r.events(r"executor: voice:")), moves=len(r.events(r"executor: move d=")),
                   invites=len(r.events(r"executor: invite")), startles=len(r.events(r"executor: startle")),
                   grumbles=styles.get("grumble", 0), face_changes=sum(faces.values()),
                   learned=len(r.events(r"runtime: learn: .*\(learned\)")), faces=faces, styles=styles)
        if r.bus is not None:
            asleep = r.c("sleep") > 0.55
            row["asleep_frac"] = float(asleep.mean())
            rgb = np.stack([r.act("led_r"), r.act("led_g"), r.act("led_b")], 1)
            fam = np.where(rgb.max(1) < 0.12, "dim", np.array(["red", "green", "blue"])[rgb.argmax(1)])
            warm = (rgb[:, 0] > 0.6) & (rgb[:, 1] > 0.3)
            fam = np.where(warm, "warm", fam)
            row["led"] = collections.Counter(fam.tolist())
            row["led_activity_mean"] = float(r.act("led_activity").mean())
            row["boredom_max"] = float(r.c("slow_boredom").max())
            row["social_max"] = float(r.c("slow_social").max())
            row["annoyance_max"] = float(r.c("annoyance").max())
            row["expect_reward_max"] = float(r.c("expect_reward").max())
            row["expect_threat_max"] = float(r.c("expect_threat").max())
            yaw = r.c("yaw") * 100.0
            pit = r.c("pitch") * 35.0
            row["travel_deg"] = float(np.hypot(np.diff(yaw), np.diff(pit)).sum())
        rows.append(row)
    return rows, face_total


# ---------------------------------------------------------------------------------------------------------- rules
class Findings:
    def __init__(self):
        self.items = []

    def check(self, ok, rule, detail):
        self.items.append(("PASS" if ok else "FAIL", rule, detail))

    def warn(self, rule, detail):
        self.items.append(("WARN", rule, detail))


def window(x, i, before, after=0):
    return x[max(0, i - before): i + after + 1]


def rules(runs, rows, face_total, f, full_day):
    by = {r.name: r for r in runs}
    rowby = {row["stretch"]: row for row in rows}
    for r in runs:
        row = rowby[r.name]
        tag = f"[{r.name}]"
        f.check(not re.search(r"^E \(", r.log, re.M), f"{tag} no firmware errors", f"{len(re.findall(r'^E \(', r.log, re.M))} error lines")
        last = re.findall(r"snn_resets=(\d+) policy_faults=(\d+)", r.log)
        if last:
            f.check(last[-1] == ("0", "0"), f"{tag} no spiking-core resets / policy faults", f"resets {last[-1][0]}, faults {last[-1][1]}")
        f.check(not re.search(r"sing: |style=sing", r.log), f"{tag} never sings", "")
        h = max(r.hours, 1e-6)
        if r.bus is None:
            continue
        touch = np.maximum(r.c("touch_head"), r.c("touch_screen")) > 0.5
        touch_on = touch & ~np.concatenate([[False], touch[:-1]])
        shake = (r.c("shake") > 0.5) | (r.c("tilt") > 0.5)
        bang = np.maximum.reduce([r.c("clap"), r.c("sharp"), r.c("loud")]) > 0.5
        # an invitation goes to somebody
        bad = []
        for ms, _ in r.events(r"executor: invite"):
            i = r.tick(ms)
            someone = (window(r.c("tr_person"), i, 40).max() > 0.2) or r.c("owner_near")[i] > 0.5
            if not someone:
                bad.append(round(ms / 1000))
        f.check(not bad, f"{tag} every invitation goes to somebody (a person around or the owner home)", f"to nobody at {bad} s" if bad else
                f"{row['invites']} invitations")
        # anger needs a reason: a run of touches (>= 4 in 30 s), a shake / lift, or repeated bangs (>= 3 in 30 s)
        bad = []
        for ms, m in r.events(r"executor: emotion=\d+ (Angry|Annoyed)"):
            i = r.tick(ms)
            reason = (window(touch_on, i, 600).sum() >= 4 or window(shake, i, 600).any() or
                      (window(bang, i, 600) & ~np.concatenate([[False], window(bang, i, 600)[:-1]])).sum() >= 3)
            if not reason:
                bad.append(round(ms / 1000))
        f.check(not bad, f"{tag} anger only with a reason (a run of touches, a shake, repeated bangs)",
                f"angry without a reason at {bad} s" if bad else "")
        # two or three strokes are petting
        petting_anger = []
        for ms, m in r.events(r"executor: emotion=\d+ (Angry|Annoyed)"):
            i = r.tick(ms)
            n = window(touch_on, i, 600).sum()
            if 1 <= n <= 3 and not window(shake, i, 600).any():
                petting_anger.append(round(ms / 1000))
        f.check(not petting_anger, f"{tag} two or three strokes never anger it", f"at {petting_anger} s" if petting_anger else "")
        # every isolated touch is answered within 2.5 s
        calls = [r.tick(ms) for ms, _ in r.events(r"executor: call: start")]
        ti = np.nonzero(touch_on)[0]
        iso = [i for k, i in enumerate(ti) if (k == 0 or i - ti[k - 1] > 200) and (k + 1 >= len(ti) or ti[k + 1] - i > 200)]
        asleep = r.c("sleep") > 0.55
        iso = [i for i in iso if not asleep[i]]
        if iso:
            answered = sum(any(0 <= c - i <= 50 for c in calls) for i in iso)
            f.check(answered / len(iso) >= 0.8, f"{tag} >= 80 % of isolated touches answered within 2.5 s", f"{answered}/{len(iso)}")
        # glances / calls / face changes in a plausible range while awake
        awake = 1.0 - row.get("asleep_frac", 0.0)
        if awake > 0.5 and r.name not in ("night",):
            low = 10 if r.name in ("ignored_at_pc", "tv_evening") else 30  # watching somebody who sits still: a glance away now and then
            f.check(low <= row["moves"] / h <= 450, f"{tag} head moves {low}-450 an hour awake", f"{row['moves'] / h:.0f}/h")
            f.check(row["startles"] / h <= 40, f"{tag} startles at most 40 an hour", f"{row['startles'] / h:.0f}/h")
            f.check(40 <= row["calls"] / h <= 420, f"{tag} calls 40-420 an hour awake", f"{row['calls'] / h:.0f}/h")
            f.check(row["face_changes"] / (h * 60) <= 8, f"{tag} face changes at most 8 a minute", f"{row['face_changes'] / (h * 60):.1f}/min")
        # following a person who moves about is legitimate travel
        limit = 16000 if (r.c("person") > 0.4).mean() > 0.3 else 12000
        f.check(row["travel_deg"] / h <= limit, f"{tag} the head does not creep (travel <= {limit} deg/h)", f"{row['travel_deg'] / h:.0f} deg/h")
        # expectations stay quiet where nothing taught them (fresh memory every stretch)
        if not shake.any():
            f.check(row["expect_threat_max"] < 0.3, f"{tag} no fear expectation without rough handling", f"max {row['expect_threat_max']:.2f}")
        # LED: lively, and its colour goes with the face
        if touch.any() or bang.any() or shake.any():  # alone in a quiet room an even glow is right
            f.check(np.std(r.act("led_activity")) > 0.02, f"{tag} the LED activity follows what happens", f"std {np.std(r.act('led_activity')):.3f}")
    # stretch-specific rules (they need the full length)
    if full_day:
        if "alone_day" in by:
            f.check(rowby["alone_day"]["boredom_max"] >= 0.6, "[alone_day] boredom grows in a quiet room (>= 0.6)", f"max {rowby['alone_day']['boredom_max']:.2f}")
        if "ignored_at_pc" in by:
            f.check(rowby["ignored_at_pc"]["social_max"] >= 0.45, "[ignored_at_pc] the need for company grows when ignored (>= 0.45)",
                    f"max {rowby['ignored_at_pc']['social_max']:.2f}")
            f.check(rowby["ignored_at_pc"]["invites"] >= 1, "[ignored_at_pc] it invites the owner who ignores it", f"{rowby['ignored_at_pc']['invites']} invitations")
        if "night" in by:
            f.check(rowby["night"]["asleep_frac"] >= 0.5, "[night] it sleeps at night (>= 50 % of the time)", f"{rowby['night']['asleep_frac']:.0%}")
        if "alone_dawn" in by:
            f.check(rowby["alone_dawn"]["asleep_frac"] >= 0.25, "[alone_dawn] it sleeps before dawn (>= 25 %)", f"{rowby['alone_dawn']['asleep_frac']:.0%}")
        if "evening_play" in by:
            e = rowby["evening_play"]
            f.check(e["learned"] >= 1, "[evening_play] it learns something by itself (a link >= 0.25)", f"{e['learned']} links")
            f.check(e["expect_reward_max"] >= 0.4, "[evening_play] it comes to expect petting", f"max {e['expect_reward_max']:.2f}")
            f.check(e["startles"] >= 1, "[evening_play] it startles at claps / shakes", f"{e['startles']}")
            f.check(e["grumbles"] >= 1, "[evening_play] a burst of touches makes it grumble", f"{e['grumbles']}")
        # census findings over the whole day
        for face in FACES:
            if face_total[face] == 0:
                if face in RARE_FACES:
                    f.warn(f"[day] the face {face} appears", "never shown in the whole day (needs a rare situation)")
                else:
                    f.check(False, f"[day] the face {face} appears", "never shown in the whole day")
        tot = sum(rows_["invites"] for rows_ in rows)
        f.check(tot >= 1, "[day] invitations happen at all", f"{tot}")


# ---------------------------------------------------------------------------------------------------------- agreement
def agreement(runs, f):
    """The spiking core's state against the drive teacher on the same inputs (closed loop)."""
    import teacher_drives
    lines = []
    for r in runs:
        if r.bus is None:
            continue
        bus = np.zeros((len(r.bus), len(SNN_INPUT)), np.float32)
        for n in SNN_INPUT:
            bus[:, sin(n)] = r.bus[:, N_IN + EFF.index(n)] if n in EFF else r.bus[:, pin(n)]
        teach = teacher_drives.run(bus)
        for n in ("arousal", "annoyance", "curiosity", "sleep", "playful", "tr_touch", "tr_person", "hab_touch", "surprise"):
            a, b = r.c(n), teach[:, SNN.index(n)]
            err = float(np.abs(a - b).mean())
            corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-4 and b.std() > 1e-4 else float("nan")
            lines.append(f"  {r.name:14s} {n:11s} mean |snn - teacher| {err:.3f}  corr {corr:+.2f}")
            if err > (0.2 if n == "sleep" else 0.15):  # sleep: the level above the asleep threshold matters little
                f.check(False, f"[{r.name}] spiking core follows the drive teacher: {n}", f"mean error {err:.2f}")
        # a drive the deterministic C++ reads must idle near zero when the teacher says zero (the sleep-drain bug)
        day = teach[:, SNN.index("sleep")] < 0.01
        if day.any():
            base = float(r.c("sleep")[day].mean())
            f.check(base < 0.1, f"[{r.name}] the sleep readout idles near 0 when awake", f"mean {base:.3f}")
    return lines


# ---------------------------------------------------------------------------------------------------------- report
def report(rows, face_total, f, agree_lines, path):
    out = []
    out.append("BEHAVIOUR AUDIT\n")
    out.append("== census (per stretch; rates per hour)")
    for row in rows:
        h = max(row["hours"], 1e-6)
        out.append(f"  {row['stretch']:14s} {row['hours']:.2f} h | calls {row['calls'] / h:5.0f} | moves {row['moves'] / h:5.0f} | invites "
                   f"{row['invites']:2d} | startles {row['startles']:2d} | grumbles {row['grumbles']:2d} | faces/min "
                   f"{row['face_changes'] / (h * 60):.1f} | asleep {row.get('asleep_frac', 0):.0%} | learned {row['learned']}")
        if "led" in row:
            out.append(f"  {'':14s} boredom max {row['boredom_max']:.2f} social max {row['social_max']:.2f} annoyance max "
                       f"{row['annoyance_max']:.2f} expect petting {row['expect_reward_max']:.2f} rough {row['expect_threat_max']:.2f} | "
                       f"LED {dict(row['led'].most_common(4))} activity {row['led_activity_mean']:.2f} | travel {row['travel_deg'] / h:.0f} deg/h")
        out.append(f"  {'':14s} faces {dict(row['faces'].most_common())} | voice styles {dict(row['styles'])}")
    out.append("\n== faces over the whole day: " + ", ".join(f"{k} {face_total[k]}" for k in FACES))
    if agree_lines:
        out.append("\n== spiking core vs drive teacher (closed loop)")
        out.extend(agree_lines)
    out.append("\n== rules")
    for status, rule, detail in f.items:
        out.append(f"  {status}  {rule}" + (f"  ({detail})" if detail else ""))
    fails = sum(1 for s, _, _ in f.items if s == "FAIL")
    warns = sum(1 for s, _, _ in f.items if s == "WARN")
    out.append(f"\n{fails} failed, {warns} warnings, {sum(1 for s, _, _ in f.items if s == 'PASS')} passed")
    text = "\n".join(out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(text + "\n")
    return text, fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--robot-log", default=None)
    a = ap.parse_args()
    f = Findings()
    if a.robot_log:
        runs = [Run("robot", log_text=open(a.robot_log, errors="ignore").read())]
        rows, face_total = census(runs)
        rules(runs, rows, face_total, f, full_day=False)
        text, fails = report(rows, face_total, f, [], os.path.join(OUT, "robot_report.txt"))
    else:
        jobs = [(name, text, a.minutes) for name, _, text in stretches(int(a.minutes))]
        with ThreadPoolExecutor(len(jobs)) as ex:
            list(ex.map(run_stretch, jobs))
        runs = [Run(name, os.path.join(OUT, name)) for name, _, _ in jobs]
        rows, face_total = census(runs)
        rules(runs, rows, face_total, f, full_day=a.minutes >= 100)
        agree = agreement(runs, f)
        text, fails = report(rows, face_total, f, agree, os.path.join(OUT, "report.txt"))
    print(text)
    sys.exit(1 if fails else 0)
