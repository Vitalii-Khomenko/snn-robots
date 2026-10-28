"""Behaviour regression of brain v4 on the digital twin (the real firmware, mode v4, the trained networks compiled
in): every touch answered, petting never angers, silent at night, a variety of faces, frightened by shaking,
babbles and looks around by day.   ./py behavior_test_v4.py [--learned FILE]   (builds nothing: run simulator/build.sh
first; --learned: with what the robot learned, the gate of the night sleep)"""
import collections
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SIM = os.path.join(ROOT, "simulator", "build", "stackchan_sim")
SCN = os.path.join(ROOT, "simulator", "scenarios")
LINE = re.compile(r"^[IWE] \((\d+)\) (\w+): (.*)$")


LEARNED = None  # --learned FILE: run every scenario with this learned blob (the night sleep's gate)


# cases that run another scenario with extra simulator options (2026-10-28, an audit A21): the firmware's millisecond
# clock wraps (49.7 days of uptime) 5 minutes into a play session
RUN_AS = {"play_session_wrap": ("play_session", ["--ms-offset", "300"])}
# what the simulator's trace (10 Hz) shows of a channel in a window: (column, from s, to s, max | mean)
TRACE = {"stale_imu": {"tilt_held": ("tilt", 65, 80, max), "tilt_after": ("tilt", 100, 360, "mean")},
         "stale_near": {"near_held": ("near", 60, 70, max), "near_after": ("near", 80, 360, "mean")}}


def trace_metrics(name, out):
    if name not in TRACE:
        return {}
    import csv
    rows = list(csv.DictReader(open(os.path.join(out, "trace.csv"))))
    res = {}
    for key, (col, a, b, how) in TRACE[name].items():
        xs = [float(r[col]) for r in rows if a <= float(r["t_s"]) <= b]
        res[key] = (sum(xs) / len(xs) if how == "mean" else how(xs)) if xs else None
    return res


def run(name, minutes):
    out = os.path.join(ROOT, "simulator", "out", "v4_" + name + ("_learned" if LEARNED else ""))
    scn, extra = RUN_AS.get(name, (name, []))
    cmd = [SIM, os.path.join(SCN, scn + ".scn"), "--out", out, "--minutes", str(minutes)] + extra
    subprocess.run(cmd + (["--learned", LEARNED] if LEARNED else []), check=True, stdout=subprocess.DEVNULL)
    ev = []
    for line in open(os.path.join(out, "log.txt"), errors="ignore"):
        m = LINE.match(line.strip())
        if m:
            ev.append((int(m.group(1)) / 1000.0, m.group(2), m.group(3), line[0]))
    moves = 0  # head movements the simulated servos started (summary.txt, per minute)
    for line in open(os.path.join(out, "summary.txt")):
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 3 and cells[0].isdigit():
            moves += int(cells[3])
    return ev, minutes, moves, out


def _clock(s):
    m, sec = s.split(":")
    return int(m) * 60 + float(sec)


def scene_targets(name):
    """Where the scenario's person and lamps are over time: [(t0, t1, az)] while present / on (lamp: after it was
    switched on)."""
    person, lamps, out = None, {}, {"person": [], "lamp": []}
    for line in open(os.path.join(SCN, RUN_AS.get(name, (name,))[0] + ".scn")):
        w = line.split()
        if len(w) >= 4 and w[0] == "lamp":
            lamps[w[1]] = float(w[2])
        if len(w) < 5 or w[0] != "at":
            continue
        t = _clock(w[1])
        if w[2] == "person" and w[4] == "enter":
            person = (t, float(w[5]))
        elif w[2] == "person" and w[4] == "walk" and person:
            t1 = t + float(w[6])
            out["person"].append((person[0], t, person[1], person[1]))
            out["person"].append((t, t1, person[1], float(w[5])))
            person = (t1, float(w[5]))
        elif w[2] == "person" and w[4] == "leave" and person:
            out["person"].append((person[0], t, person[1], person[1]))
            person = None
        elif w[2] == "lamp" and w[4] == "on":
            out["lamp"].append((t, 1e9, lamps[w[3]], lamps[w[3]]))
    return out


def seek_metrics(name, out):
    """How well the head finds the scenario's person / lamp: the share of the time it points within 25 deg of them
    (the camera sees +-31 deg), and how soon after the lamp came on it first looked there."""
    import csv
    rows = [(float(r["t_s"]), float(r["head_yaw"])) for r in csv.DictReader(open(os.path.join(out, "trace.csv")))]
    tg = scene_targets(name)
    res = {}
    for kind, segs in tg.items():
        near = total = 0
        first = None
        for t, yaw in rows:
            for t0, t1, a0, a1 in segs:
                if t0 <= t < t1:
                    az = a0 + (a1 - a0) * min(1.0, (t - t0) / max(t1 - t0, 1e-6)) if t1 < 1e8 else a0
                    total += 1
                    if abs(yaw - az) < 25.0:
                        near += 1
                        if first is None:
                            first = t - t0
        if segs:
            res[f"on_{kind}"] = near / total if total else 0.0
            res[f"{kind}_found_s"] = first
    return res


def metrics(ev, minutes, moves=0):
    faces = [(t, x.split()[1]) for t, tag, x, _ in ev if tag == "v4" and x.startswith("face:")]
    calls = [t for t, tag, x, _ in ev if tag == "v4" and x.startswith("call")]
    gestures = [(t, x.split()[1]) for t, tag, x, _ in ev if tag == "v4" and x.startswith("gesture:")]
    touches = [t for t, tag, x, _ in ev if tag == "runtime" and x.startswith("head touch (")]
    # isolated touches (5 s after the one before): in a burst of taps every 2 s the robot cannot answer each one (a call
    # in 10 s, a nod in 4 s: the voice budget and the servos' power peaks) - it gets cross instead (the pokes test)
    isolated = [t for i, t in enumerate(touches) if i == 0 or t - touches[i - 1] >= 5.0]
    answered = sum(1 for t in isolated if any(t <= c <= t + 2.0 for c in calls) or
                   any(t <= g <= t + 2.0 for g, _ in gestures))
    share = collections.Counter()
    end = minutes * 60.0
    for i, (t, f) in enumerate(faces):
        share[f] += (faces[i + 1][0] if i + 1 < len(faces) else end) - t
    total = sum(share.values()) or 1.0
    return {
        "touches": len(touches), "touch_answered_frac": answered / len(isolated) if isolated else None,
        "calls": len(calls), "calls_per_h": len(calls) / (minutes / 60.0),
        "n_faces": len(set(f for _, f in faces)), "faces": dict(share),
        "angry_faces": sum(1 for _, f in faces if f in ("Angry", "Annoyed")),
        # an angry / annoyed face in answer to a touch (within 5 s after it)
        "angry_after_touch": sum(1 for t, f in faces if f in ("Angry", "Annoyed") and any(0 <= t - x <= 5.0 for x in touches)),
        "cross": sum(1 for _, f in faces if f in ("Angry", "Annoyed")) + sum(1 for _, g in gestures if g == "shake"),
        "frightened": sum(1 for _, f in faces if f in ("Scared", "Dizzy", "Surprised")) +
                      sum(1 for _, g in gestures if g == "flinch"),
        "nods": sum(1 for _, g in gestures if g == "nod"), "shakes": sum(1 for _, g in gestures if g == "shake"),
        "sleepy_frac": share["Sleepy"] / total,
        "errors": sum(1 for _, _, _, lvl in ev if lvl == "E"),
        "moves_per_min": moves / minutes,
        "nods_per_touch": sum(1 for _, g in gestures if g == "nod") / len(touches) if touches else None,
        # the Muse (docs/CREATIVITY.md): how many kinds of idea came, and how many different faces it made
        "idea_kinds": len(set(x.split()[2] for _, tag, x, _ in ev if tag == "v4" and x.startswith("muse: idea"))),
        "ideas": sum(1 for _, tag, x, _ in ev if tag == "v4" and x.startswith("muse: idea")),
    }


JOYFUL = {"Happy", "VeryHappy", "Laughing", "Wink", "Love"}
LOW = {"Sad", "Bored", "Crying"}


def contingency_metrics(ev, minutes):
    """The sense of "I did that" (docs/CONTINGENCY.md) from the log: the detector's credits ("contingency: gesture
    answered by pet after 1.2 s"), the faces, calls and gestures around them, the scenario's answers ("responder:")."""
    faces = [(t, x.split()[1]) for t, tag, x, _ in ev if tag == "v4" and x.startswith("face:")]
    calls = [t for t, tag, x, _ in ev if tag == "v4" and x.startswith("call")]
    gestures = [t for t, tag, x, _ in ev if tag == "v4" and x.startswith("gesture:")]
    credits = [(t, x.split()[1]) for t, tag, x, _ in ev if tag == "runtime" and x.startswith("contingency:")]
    answers = [t for t, tag, x, _ in ev if tag == "world" and x.startswith("responder:")]

    def face_at(t):
        f = None
        for tf, name in faces:
            if tf > t:
                break
            f = name
        return f

    acts = {"call": calls, "gesture": gestures, "face": [t for t, _ in faces]}
    joy = [face_at(t + 1.5) in JOYFUL for t, _ in credits]
    # the answered act again within 8 s, against the same act within 8 s of any moment (calls are frequent anyway)
    again = [any(t + 0.5 < x <= t + 8.0 for x in acts[a]) for t, a in credits]
    grid = [50.0 + 0.5 * i for i in range(int((minutes * 60 - 60) / 0.5))]
    chance = {a: sum(any(t + 0.5 < x <= t + 8.0 for x in xs) for t in grid) / max(len(grid), 1) for a, xs in acts.items()}
    again_chance = [chance[a] for _, a in credits]
    end = minutes * 60.0
    late = [(t, f) for t, f in faces]
    low = total = 0.0
    for i, (t, f) in enumerate(late):  # the share of low faces in 7:00 .. 11:00 (the still face, before the repair)
        t1 = faces[i + 1][0] if i + 1 < len(faces) else end
        a, b = max(t, 420.0), min(t1, 660.0)
        if b > a:
            total += b - a
            low += (b - a) if f in LOW else 0.0
    social = [t for t in calls + gestures if t > 50.0]
    return {"answers": len(answers), "credits": len(credits),
            "joy_after_answer": sum(joy) / len(joy) if joy else None,
            "again_after_answer": (sum(again) - sum(again_chance)) / len(again) if again else None,
            "acts_per_min": len(social) / max(minutes - 50 / 60.0, 1e-6),
            "low_after_still": low / total if total else None,
            "angry_after_still": sum(1 for t, f in faces if t > 300.0 and f in ("Angry", "Annoyed"))}


# (scenario, minutes, [(metric, min, max, why)])
CASES = [
    ("touch_response", 9, [("touch_answered_frac", 0.9, 1.01, "answers every isolated touch (voice or gesture within 2 s)")]),
    ("pokes", 5, [("cross", 2, 600, "poked every 0.8 s for 60 s it gets cross (an annoyed / angry face or a head shake; 25 s are tolerated since 2026-10-25), 3 seeds")]),
    ("petting", 6, [("angry_after_touch", 0, 0, "petting never angers"), ("touch_answered_frac", 0.8, 1.01, "answers the hand"),
                    ("nods_per_touch", 0.2, 5, "nods 'yes' to the hand (owner 2026-10-24: the head hardly moved)")]),
    ("night_quiet", 40, [("calls", 0, 0, "silent at night (the owner sleeps)"), ("sleepy_frac", 0.3, 1.01, "sleeps")]),
    ("play_session", 12, [("n_faces", 5, 30, "a variety of faces"), ("calls", 3, 400, "talks")]),
    ("day_alone", 20, [("calls_per_h", 40, 600, "babbles by itself by day"),
                       ("moves_per_min", 3, 60, "looks around by itself (owner 2026-10-24: the head hardly moved)"),
                       ("idea_kinds", 4, 5, "the Muse has ideas of most kinds by itself (2026-10-26: creativity)"),
                       # (5 -> 4 on 2026-10-28: with the head input the network was trained on it shows 4-6 faces
                       # over 12 seeds alone, was 5-7 - the extra ones flickers of 2-6 s)
                       ("n_faces", 4, 30, "and the robot's faces vary (2 until the Muse)")]),
    ("learn_rough", 7, [("frightened", 1, 500, "being shaken frightens it (scared / dizzy / surprised / flinch)")]),
    ("hearing", 4, [("frightened", 1, 500, "claps and bangs startle it")]),
    # 2026-10-25, the owner: nothing in the room interested it, not movement, not a bright light
    # report only: the simulated walker is a 0.44 m figure as bright as the walls, it moves ~1/8 of what the robot's
    # camera saw of a real person (2026-10-24: frame share p90 0.135); over 5 seeds it is found as often as by chance
    # (0.37 / 0.37, 2026-10-25). The robot's diary is the test of the motion sense.
    # the sense of "I did that" (docs/CONTINGENCY.md step 3, 2026-10-27)
    # judged over CT_SEEDS (2026-10-27 evening): the gate of 0.35 was one seed's luck (0.42); over 6 seeds joy after an
    # answer was 0.09-0.33, mean 0.18, hardly above any moment's 0.15 and no more than after a pet at random - a report
    # until the joy of being answered is learned (docs/CONTINGENCY.md, open)
    ("contingent", 12, [("joy_after_answer", 0.0, 1.01, "report: joy after an answer to its act (3 seeds)"),
                        ("again_after_answer", -1.0, 1.01, "report: the answered act again within 8 s, over chance")]),
    ("control", 12, [("acts_per_min", 0, 100, "report: calls and gestures a minute when the same pets come at random")]),
    ("still_face", 12, [("low_after_still", 0.2, 1.01, "ignored by someone close it gets quietly sad"),
                        ("angry_after_still", 0, 0, "never angry at being ignored")]),
    # the body budget (docs/BODY.md step 2, 2026-10-27): 14 minutes of bangs and rough handling spend it, a gentle hand
    # refills it; the same quiet stretch and hand without the hard part leave it as it was. (0.45 -> 0.6 on 2026-10-28:
    # with the head input the network was trained on it is frightened less, the low is 0.36-0.53 over 12 seeds, was
    # 0.23-0.42; the control stays >= 0.75)
    ("hard_day", 32, [("budget_low", 0.0, 0.6, "a hard stretch spends the body budget (minutes 12-24)"),
                      ("budget_refill", 0.15, 1.0, "a gentle hand refills it (minute 31 against minute 24)")]),
    ("hard_day_control", 32, [("budget_low", 0.75, 1.0, "a quiet stretch costs little")]),
    ("seek_motion", 5, [("on_person", 0.0, 1.01, "report: how much of the time the head points at someone walking about")]),
    # 2026-10-26: the calm head (a place kept 5-11 s, slow turns) looks round less: 0.5 at the lamp was the old busy
    # head's level; what counts is the lamp against the same runs with it left off, over 3 seeds (SEEK_SEEDS)
    ("seek_light", 4, [("lamp_found_s", 0, 120, "a lamp switched on outside the picture: looks round and finds it within 2 minutes"),
                       ("lamp_over_chance", 0.05, 1.01, "and looks there more than with the lamp left off (3 seeds)")]),
    # faults of the real device (2026-10-28, an audit A21: the simulator never had them)
    ("no_clock_dark", 8, [("calls", 0, 0, "rebooted at 3 am without internet, in the dark: night rules hold (silent)"),
                          ("nods", 0, 0, "and no gestures")]),
    ("no_clock_lit", 8, [("calls", 3, 600, "without internet by day in a lit room: talks again after two quiet minutes")]),
    ("stale_imu", 6, [("tilt_held", 0.1, 1.01, "held tilted sideways it feels it"),
                      ("tilt_after", 0.0, 0.05, "the IMU stops answering: the tilt fades, not frozen for ever")]),
    ("stale_near", 6, [("near_held", 0.1, 1.01, "a hand at its face is felt near"),
                       ("near_after", 0.0, 0.05, "the proximity chip stops answering: nobody near, not someone for ever")]),
    ("play_session_wrap", 12, [("n_faces", 5, 30, "the millisecond clock wraps at minute 5: faces still vary"),
                               ("calls_after_wrap", 3, 400, "and it still talks after the wrap")]),
    # servos that hang like the robot's (2026-10-28: the hardware stays as it is; the watchdog runs in the simulator too)
    ("servo_hangs", 20, [("repowers", 1, 60, "8 % of the switch-ons hang: the watchdog power-cycles the rail"),
                         ("moves_per_min", 3, 60, "and the head keeps looking around (day_alone: 3.8)"),
                         ("calls_per_h", 40, 600, "the voice does not suffer")]),
    ("servo_blocked", 20, [("repowers", 3, 10, "a head blocked for 10 minutes: the power cycles back off (not one every 12 s)"),
                           ("travel_after_free", 100, 100000, "free again, the head moves (minutes 13-20, degrees)")]),
]


SEEK_SEEDS = (9, 1, 2)  # the head's path differs a lot from seed to seed: the lamp's pull is judged over three
CT_SEEDS = (1, 2)  # the contingency scenarios are judged over their own seed and these two (one seed's luck, 2026-10-27)
CT_CASES = ("contingent", "control", "still_face")
# judged as a sum over three seeds too (2026-10-28: the pokes' one seed gave cross 0 for a learned blob that changed 0.4 %
# of the choices; over 12 seeds 1 in 12 runs has none - the gate of the night failed by chance)
SUM_SEEDS = {"pokes": ("cross",)}


def seeded(name, minutes, seed, fn=None):
    """contingency_metrics (or fn's metrics) of one scenario run with another seed."""
    src = open(os.path.join(SCN, name + ".scn")).read().splitlines()
    src = [l for l in src if not l.startswith("seed ")] + ["seed %d" % seed]
    tag = f"{name}_{seed}" + ("_learned" if LEARNED else "")
    scn = os.path.join(ROOT, "simulator", "out", "v4_" + tag + ".scn")
    open(scn, "w").write("\n".join(src) + "\n")
    out = os.path.join(ROOT, "simulator", "out", "v4_" + tag)
    cmd = [SIM, scn, "--out", out, "--minutes", str(minutes)]
    subprocess.run(cmd + (["--learned", LEARNED] if LEARNED else []), check=True, stdout=subprocess.DEVNULL)
    ev = []
    for line in open(os.path.join(out, "log.txt"), errors="ignore"):
        m = LINE.match(line.strip())
        if m:
            ev.append((int(m.group(1)) / 1000.0, m.group(2), m.group(3), line[0]))
    return (fn or contingency_metrics)(ev, minutes)


def yoked(minutes, seed):
    """The yoked control of contingent.scn (2026-10-28, an audit): the same pets at the very moments the contingent run
    of this seed got them, whatever the robot does now - the same stimulation, the dependence on its acts broken.
    -> contingency_metrics of that run."""
    out = os.path.join(ROOT, "simulator", "out", "v4_contingent" + ("" if seed is None else f"_{seed}") +
                       ("_learned" if LEARNED else ""))
    times = [int(m.group(1)) / 1000.0 for m in (LINE.match(l.strip()) for l in open(os.path.join(out, "log.txt"), errors="ignore"))
             if m and m.group(2) == "world" and m.group(3).startswith("responder:")]
    src = [l for l in open(os.path.join(SCN, "contingent.scn")).read().splitlines()
           if not l.startswith("responder") and " responder " not in l]
    if seed is not None:
        src = [l for l in src if not l.startswith("seed ")] + ["seed %d" % seed]
    src += ["at %d:%06.3f pet 1200" % (t // 60, t % 60) for t in times]
    tag = "yoked" + ("" if seed is None else f"_{seed}") + ("_learned" if LEARNED else "")
    scn = os.path.join(ROOT, "simulator", "out", "v4_" + tag + ".scn")
    open(scn, "w").write("\n".join(src) + "\n")
    res = os.path.join(ROOT, "simulator", "out", "v4_" + tag)
    cmd = [SIM, scn, "--out", res, "--minutes", str(minutes)]
    subprocess.run(cmd + (["--learned", LEARNED] if LEARNED else []), check=True, stdout=subprocess.DEVNULL)
    ev = []
    for line in open(os.path.join(res, "log.txt"), errors="ignore"):
        m = LINE.match(line.strip())
        if m:
            ev.append((int(m.group(1)) / 1000.0, m.group(2), m.group(3), line[0]))
    res = contingency_metrics(ev, minutes)
    res["answers"] = len(times)  # given by the scenario here, not by a responder
    return res


def lamp_share(name, minutes, seed, lamp):
    """The share of the time the head points at the scenario's lamp, the lamp switched on or left off, one seed."""
    src = open(os.path.join(SCN, name + ".scn")).read().splitlines()
    src = [("seed %d" % seed) if l.startswith("seed ") else l for l in src]
    if not lamp:
        src = [l for l in src if not (l.startswith("at ") and " lamp " in l and l.endswith(" on"))]
    tag = f"{name}_{seed}_{'on' if lamp else 'off'}" + ("_learned" if LEARNED else "")
    scn = os.path.join(ROOT, "simulator", "out", "v4_" + tag + ".scn")
    open(scn, "w").write("\n".join(src))
    out = os.path.join(ROOT, "simulator", "out", "v4_" + tag)
    cmd = [SIM, scn, "--out", out, "--minutes", str(minutes)]
    subprocess.run(cmd + (["--learned", LEARNED] if LEARNED else []), check=True, stdout=subprocess.DEVNULL)
    return seek_metrics(name, out).get("on_lamp", 0.0)


def check(case):
    name, minutes, checks = case
    ev, minutes, moves, out = run(name, minutes)
    m = metrics(ev, minutes, moves)
    m.update(trace_metrics(name, out))
    m["repowers"] = sum(1 for _, tag, x, _ in ev if tag == "runtime" and x.startswith("servo rail power cycle"))
    travel = [float(l.split("|")[2]) for l in open(os.path.join(out, "summary.txt"))
              if len(l.split("|")) > 3 and l.split("|")[0].strip().isdigit()]
    m["travel_after_free"] = sum(travel[12:])
    if name in RUN_AS:
        m["calls_after_wrap"] = sum(1 for t, tag, x, _ in ev if tag == "v4" and x.startswith("call") and t > 320.0)
    m.update(seek_metrics(name, out))
    m.update(contingency_metrics(ev, minutes))
    if any(k == "lamp_over_chance" for k, *_ in checks):
        on = [m["on_lamp"]] + [lamp_share(name, minutes, s, True) for s in SEEK_SEEDS[1:]]
        off = [lamp_share(name, minutes, s, False) for s in SEEK_SEEDS]
        m["lamp_over_chance"] = sum(on) / len(on) - sum(off) / len(off)
    budget = [float(l.split("|")[12]) for l in open(os.path.join(out, "summary.txt"))
              if len(l.split("|")) > 12 and l.split("|")[0].strip().isdigit()]
    if len(budget) >= 31:  # per minute; [k] is the end of minute k + 1
        m["budget_low"] = min(budget[11:24])
        m["budget_refill"] = budget[30] - budget[23]
        # the quiet stretch after it (minutes 15-24): calls a minute and the share of low faces (BODY step 3)
        m["calls_quiet"] = sum(1 for t, tag, x, _ in ev if tag == "v4" and x.startswith("call") and 900 <= t < 1440) / 9.0
        fl = [(t, x.split()[1]) for t, tag, x, _ in ev if tag == "v4" and x.startswith("face:")]
        low = tot = 0.0
        for i, (t, f) in enumerate(fl):
            a, b = max(t, 900.0), min(fl[i + 1][0] if i + 1 < len(fl) else 1440.0, 1440.0)
            if b > a:
                tot += b - a
                low += (b - a) if f in ("Sad", "Bored", "Sleepy", "Sick", "Crying") else 0.0
        m["low_quiet"] = low / tot if tot else None
    if name in SUM_SEEDS:  # the sum over the scenario's own seed and CT_SEEDS
        runs = [m] + [seeded(name, minutes, s, metrics) for s in CT_SEEDS]
        for k in SUM_SEEDS[name]:
            m[k] = sum(r[k] for r in runs)
    if name in CT_CASES:  # the mean over the seeds (angry: the sum)
        runs = [contingency_metrics(ev, minutes)] + [seeded(name, minutes, s) for s in CT_SEEDS]
        if name == "contingent":  # each seed's yoked control, after its contingent run
            yk = [yoked(minutes, s) for s in (None,) + CT_SEEDS]
            m["acts_yoked"] = sum(r["acts_per_min"] for r in yk) / len(yk)
            m["answers_yoked"] = sum(r["answers"] for r in yk)
            for k in ("again_after_answer", "joy_after_answer"):
                vals = [r[k] for r in yk if r[k] is not None]
                m[k + "_yoked"] = sum(vals) / len(vals) if vals else None
        for k in ("joy_after_answer", "again_after_answer", "acts_per_min", "low_after_still", "angry_after_still"):
            vals = [r[k] for r in runs if r[k] is not None]
            if vals:
                m[k] = sum(vals) if k == "angry_after_still" else sum(vals) / len(vals)
    return name, m, checks


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--learned":
        LEARNED = os.path.abspath(sys.argv[2])
    failed = []
    acts, quiet = {}, {}
    with ThreadPoolExecutor(os.cpu_count() or 4) as ex:
        for name, m, checks in ex.map(check, CASES):
            acts[name] = m["acts_per_min"]
            if "acts_yoked" in m:
                acts["yoked"], acts["answers_yoked"], acts["answers_contingent"] = m["acts_yoked"], m["answers_yoked"], m["answers"]
                acts["again"] = (m["again_after_answer"], m["again_after_answer_yoked"])
                acts["joy"] = (m["joy_after_answer"], m["joy_after_answer_yoked"])
            if "calls_quiet" in m:
                quiet[name] = (m["calls_quiet"], m["low_quiet"])
            extra = (f"   answers {m['answers']} credits {m['credits']} acts/min {m['acts_per_min']:.2f}"
                     if m["answers"] else "")
            print(f"== {name}   faces {', '.join(f'{k} {v:.0f}s' for k, v in sorted(m['faces'].items(), key=lambda kv: -kv[1])[:6])}"
                  f"   calls {m['calls']} nods {m['nods']} shakes {m['shakes']} touches {m['touches']} "
                  f"head moves/min {m['moves_per_min']:.1f}{extra}")
            if m["errors"]:
                failed.append(f"{name}: firmware logged {m['errors']} error(s)")
            for key, lo, hi, why in checks:
                v = m.get(key)
                ok = v is not None and lo <= v <= hi
                print(f"   {'PASS' if ok else 'FAIL'}  {key} = {v if not isinstance(v, float) else round(v, 3)}  (want {lo}..{hi}: {why})")
                if not ok:
                    failed.append(f"{name}: {key} = {v}, wanted {lo}..{hi} ({why})")
    # Answered, the robot acts more than with pets at random. (2026-10-28, an audit: this is not the circular reaction
    # it was taken for - the responder pets twice as often as the random control (86 against 38 on its own seed), and the
    # yoked control below, the same pets at the same moments, gets as many acts. Kept as a gate of what it does show:
    # attention makes the robot livelier.)
    ratio = acts["contingent"] / max(acts["control"], 1e-6)
    # (3 seeds; over 6 seeds 1.10 on 2026-10-27 - the gate 1.1 on one seed was too tight)
    ok = ratio >= 1.05
    print(f"== contingent vs control: calls and gestures {acts['contingent']:.2f} vs {acts['control']:.2f} a minute")
    print(f"   {'PASS' if ok else 'FAIL'}  acts_ratio = {ratio:.2f}  (want >= 1.05: answered it acts more than with fewer pets "
          f"at random - livelier with attention; not yet the contingency, see the yoked control; 3 seeds)")
    # the yoked control (2026-10-28, an audit): the very same pets at the very same moments, not caused by the robot
    yk = acts["contingent"] / max(acts.get("yoked", 0.0), 1e-6)
    print(f"== contingent vs yoked: calls and gestures {acts['contingent']:.2f} vs {acts.get('yoked', 0.0):.2f} a minute "
          f"(the same {acts.get('answers_yoked', 0)} pets at the same moments, 3 seeds)")
    print(f"   REPORT  acts_ratio_yoked = {yk:.2f}  (above 1: the robot's acts multiply because they are answered, not "
          f"because it is petted)")
    fmt = lambda x: "-" if x is None else f"{x:+.3f}"
    if "again" in acts:
        print(f"   REPORT  the answered act again within 8 s, over chance: answered {fmt(acts['again'][0])}, yoked "
              f"{fmt(acts['again'][1])} (Piaget's circular reaction: above the yoked)")
        print(f"   REPORT  a joyful face 1.5 s after a credited answer: answered {fmt(acts['joy'][0])}, yoked "
              f"{fmt(acts['joy'][1])} (Watson's joy of control: above the yoked)")
    if not ok:
        failed.append(f"contingent vs control: acts_ratio = {ratio:.2f}, wanted >= 1.05")
    # the body (BODY step 3): after a hard stretch, alone and low, it saves itself - report until the robot's diary shows
    # real low budgets
    if "hard_day" in quiet and "hard_day_control" in quiet:
        (ch, lh), (cc, lc) = quiet["hard_day"], quiet["hard_day_control"]
        print(f"== hard_day vs control, the quiet minutes 15-24: calls {ch:.2f} vs {cc:.2f} a minute, low faces "
              f"{lh:.2f} vs {lc:.2f}  (report: a low body budget alone saves itself)")
    if failed:
        print("\nFAILED:\n  " + "\n  ".join(failed))
        sys.exit(1)
    print("\nPASS: brain v4 behaviour")
