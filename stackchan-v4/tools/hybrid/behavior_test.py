"""Behaviour regression on the digital twin: runs the real firmware (with the GRU compiled in) on
scenarios and checks the behaviours the design promises.   simulator/behavior_test.sh"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import evaluate as ev

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SCN = os.path.join(ROOT, "simulator", "scenarios")

# (scenario, minutes, [(metric, minimum, maximum, why)])
CASES = [
    ("play_session", 12, [("person_centred_frac", 0.5, 1.01, "tracks the visitor"), ("calls", 3, 160, "talks back"),
                           ("startles", 1, 60, "flinches at loud things"), ("n_emotions", 5, 30, "shows a variety of faces")]),
    ("night_quiet", 40, [("stats_sleepy_frac", 0.25, 1.01, "falls asleep at night")]),
    ("startle_and_orient", 5, [("startles", 1, 30, "startles at claps"), ("yaw_span", 20, 400, "turns towards sounds")]),
    ("petting", 6, [("angry_faces", 0, 0, "two or three touches are petting: no Angry / Annoyed face"),
                    ("grumble_replies", 0, 0, "and no grumble")]),
    ("ignored", 20, [("nods", 1, 5, "invites a person who ignores it to play (at most every 4 min)")]),
    ("day_alone", 20, [("moves", 5, 400, "looks around by itself")]),
    ("vision_motion", 5, [("person_centred_frac", 0.4, 1.01, "follows a moving person")]),
    ("touch_response", 9, [("touch_answered_frac", 0.8, 1.01, "answers isolated touches"), ("grumble_replies", 1, 100, "gets annoyed by a burst of touches")]),
    ("conversation", 3.5, [("calls_talk_per_min", 3.5, 7, "joins a conversation (>= 3.5 calls a minute while people talk)"), ("calls_in_pause_frac", 0.35, 1.0, "mostly in the pauses between phrases")]),
    ("hearing", 4, [("startles", 3, 100, "flinches at claps and bangs"), ("calls", 3, 300, "answers sounds with its voice")]),
    ("learn_petting", 8, [("learned_petting", 1, 50, "learns by itself what comes before petting"),
                          ("expect_reward_late_max", 0.4, 1.01, "and expects petting when it comes again"),
                          ("learned_rough", 0, 0, "petting teaches nothing about rough handling")]),
    ("learn_rough", 7, [("learned_rough", 1, 50, "learns what comes before being shaken"),
                        ("expect_threat_late_max", 0.4, 1.01, "and expects it at the next clap")]),
]


def run(case):
    name, minutes, checks = case
    out = os.path.join(ROOT, "simulator", "out", "bt_" + name)
    ev.run("builtin", os.path.join(SCN, name + ".scn"), out, minutes)
    m = ev.metrics(out)
    m["n_emotions"] = len(m["emotions"])
    m["yaw_span"] = m["yaw_range"][1] - m["yaw_range"][0]
    return name, m, checks


if __name__ == "__main__":
    failed = []
    with ThreadPoolExecutor(os.cpu_count() or 4) as ex:
        for name, m, checks in ex.map(run, CASES):
            print(f"== {name}")
            if m["errors"]:
                failed.append(f"{name}: firmware logged {m['errors']} error(s)")
            r = m.get("snn_resets/policy_faults/missed")
            if r and (r[0] or r[1]):
                failed.append(f"{name}: snn resets / policy faults {r}")
            for key, lo, hi, why in checks:
                v = m.get(key)
                ok = v is not None and lo <= v <= hi
                print(f"   {'PASS' if ok else 'FAIL'}  {key} = {v}  (want {lo}..{hi}: {why})")
                if not ok:
                    failed.append(f"{name}: {key} = {v}, wanted {lo}..{hi} ({why})")
    if failed:
        print("\nFAILED:\n  " + "\n  ".join(failed))
        sys.exit(1)
    print("\nPASS: behaviour regression")
