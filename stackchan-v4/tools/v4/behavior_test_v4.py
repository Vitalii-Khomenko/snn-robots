"""Behaviour regression of brain v4 on the digital twin (the real firmware, mode v4, the trained networks compiled
in): every touch answered, petting never angers, silent at night, a variety of faces, frightened by shaking,
babbles and looks around by day.   ./py behavior_test_v4.py   (builds nothing: run simulator/build.sh first)"""
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


def run(name, minutes):
    out = os.path.join(ROOT, "simulator", "out", "v4_" + name)
    subprocess.run([SIM, os.path.join(SCN, name + ".scn"), "--out", out, "--minutes", str(minutes)],
                   check=True, stdout=subprocess.DEVNULL)
    ev = []
    for line in open(os.path.join(out, "log.txt"), errors="ignore"):
        m = LINE.match(line.strip())
        if m:
            ev.append((int(m.group(1)) / 1000.0, m.group(2), m.group(3), line[0]))
    return ev, minutes


def metrics(ev, minutes):
    faces = [(t, x.split()[1]) for t, tag, x, _ in ev if tag == "v4" and x.startswith("face:")]
    calls = [t for t, tag, x, _ in ev if tag == "v4" and x.startswith("call")]
    gestures = [(t, x.split()[1]) for t, tag, x, _ in ev if tag == "v4" and x.startswith("gesture:")]
    touches = [t for t, tag, x, _ in ev if tag == "runtime" and x.startswith("head touch (")]
    answered = sum(1 for t in touches if any(t <= c <= t + 2.0 for c in calls) or
                   any(t <= g <= t + 2.0 for g, _ in gestures))
    share = collections.Counter()
    end = minutes * 60.0
    for i, (t, f) in enumerate(faces):
        share[f] += (faces[i + 1][0] if i + 1 < len(faces) else end) - t
    total = sum(share.values()) or 1.0
    return {
        "touches": len(touches), "touch_answered_frac": answered / len(touches) if touches else None,
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
    }


# (scenario, minutes, [(metric, min, max, why)])
CASES = [
    ("touch_response", 9, [("touch_answered_frac", 0.8, 1.01, "answers every touch (voice or gesture within 2 s)")]),
    ("pokes", 5, [("cross", 1, 200, "poked every 0.8 s it gets cross (an annoyed / angry face or a head shake)")]),
    ("petting", 6, [("angry_after_touch", 0, 0, "petting never angers"), ("touch_answered_frac", 0.8, 1.01, "answers the hand")]),
    ("night_quiet", 40, [("calls", 0, 0, "silent at night (the owner sleeps)"), ("sleepy_frac", 0.3, 1.01, "sleeps")]),
    ("play_session", 12, [("n_faces", 5, 30, "a variety of faces"), ("calls", 3, 400, "talks")]),
    ("day_alone", 20, [("calls_per_h", 40, 600, "babbles by itself by day"), ("n_faces", 2, 30, "moods change")]),
    ("learn_rough", 7, [("frightened", 1, 500, "being shaken frightens it (scared / dizzy / surprised / flinch)")]),
    ("hearing", 4, [("frightened", 1, 500, "claps and bangs startle it")]),
]


def check(case):
    name, minutes, checks = case
    ev, minutes = run(name, minutes)
    return name, metrics(ev, minutes), checks


if __name__ == "__main__":
    failed = []
    with ThreadPoolExecutor(os.cpu_count() or 4) as ex:
        for name, m, checks in ex.map(check, CASES):
            print(f"== {name}   faces {', '.join(f'{k} {v:.0f}s' for k, v in sorted(m['faces'].items(), key=lambda kv: -kv[1])[:6])}"
                  f"   calls {m['calls']} nods {m['nods']} shakes {m['shakes']} touches {m['touches']}")
            if m["errors"]:
                failed.append(f"{name}: firmware logged {m['errors']} error(s)")
            for key, lo, hi, why in checks:
                v = m.get(key)
                ok = v is not None and lo <= v <= hi
                print(f"   {'PASS' if ok else 'FAIL'}  {key} = {v if not isinstance(v, float) else round(v, 3)}  (want {lo}..{hi}: {why})")
                if not ok:
                    failed.append(f"{name}: {key} = {v}, wanted {lo}..{hi} ({why})")
    if failed:
        print("\nFAILED:\n  " + "\n  ".join(failed))
        sys.exit(1)
    print("\nPASS: brain v4 behaviour")
