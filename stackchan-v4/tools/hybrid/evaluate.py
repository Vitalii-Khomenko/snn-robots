"""Closed-loop evaluation of a policy in the digital twin.

  ./py evaluate.py teacher simulator/scenarios/play_session.scn
  ./py evaluate.py builtin simulator/scenarios/play_session.scn      # the GRU compiled into the sim
Prints behaviour metrics from trace.csv / log.txt."""
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SIM = os.path.join(ROOT, "simulator", "build", "stackchan_sim")


def run(kind, scn, out, minutes=None, seed=None):
    os.makedirs(out, exist_ok=True)
    if kind == "builtin":
        cmd = [SIM, scn, "--out", out]
        if minutes:
            cmd += ["--minutes", str(minutes)]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    else:
        from rollout import run_episode
        run_episode((kind, scn, out, seed, minutes, os.path.join(HERE, "data", "policy_net.pt")))
    return out


def metrics(out):
    import csv
    rows = list(csv.DictReader(open(os.path.join(out, "trace.csv"))))
    log = open(os.path.join(out, "log.txt"), errors="ignore").read()
    f = lambda k: np.array([float(r[k]) for r in rows])
    t = f("t_s")
    m = {}
    m["minutes"] = round((t[-1] - t[0]) / 60, 1)
    px = f("person_x")
    vis = px < 5
    m["person_in_view_frac"] = round(float(vis.mean()), 3)
    if vis.any():
        m["person_centred_frac"] = round(float((np.abs(px[vis]) < 0.35).mean()), 3)
    yaw = f("head_yaw")
    m["yaw_range"] = [round(float(yaw.min())), round(float(yaw.max()))]
    m["moves"] = len(re.findall(r"executor: move d=", log))
    m["calls"] = len(re.findall(r"executor: voice:", log))
    # touches: how many isolated touches got a call within 2.5 s
    def times(pat):
        return [float(x) / 1000.0 for x in re.findall(r"I \((\d+)\) " + pat, log, re.M)]
    touches, calls = times(r"runtime: head touch \((?:held during motion, )?level \d+\)$"), times(r"executor: call: start")
    isolated = [t for i, t in enumerate(touches) if (i == 0 or t - touches[i - 1] > 10) and (i + 1 >= len(touches) or touches[i + 1] - t > 10)]
    if isolated:
        m["touch_answered_frac"] = round(sum(any(0 <= c - t <= 2.5 for c in calls) for t in isolated) / len(isolated), 2)
        m["touches_isolated"] = len(isolated)
    # conversation: calls per minute in the quiet first minute and during the two minutes of talk (scenario `conversation`)
    m["calls_quiet_per_min"] = round(sum(10 <= c < 60 for c in calls) * 60.0 / 50.0, 1)
    m["calls_talk_per_min"] = round(sum(60 <= c < 180 for c in calls) / 2.0, 1)
    m["calls_in_pause_frac"] = round(sum(60 <= c < 180 and (c - 60) % 8 >= 4 for c in calls) / max(1, sum(60 <= c < 180 for c in calls)), 2)
    m["grumble_replies"] = len(re.findall(r"voice: .*style=grumble", log))
    m["startles"] = len(re.findall(r"executor: startle", log))
    m["nods"] = len(re.findall(r"executor: invite", log))
    emos = re.findall(r"emotion=\d+ (\w+)", log)
    m["emotions"] = sorted(set(emos))
    m["angry_faces"] = sum(e in ("Angry", "Annoyed") for e in emos)
    m["emotion_changes"] = len(emos)
    m["stats_sleepy_frac"] = round(float(np.mean([r["emotion"] == "Sleepy" for r in rows])), 3)
    # what it learned by itself (AssocMemory): links that crossed a level, and the expectation in the last third of the run
    m["learned_petting"] = len(re.findall(r"learn: .* -> petting .*\(learned\)", log))
    m["learned_rough"] = len(re.findall(r"learn: .* -> rough handling .*\(learned\)", log))
    late = t >= t[0] + (t[-1] - t[0]) * 2 / 3
    if "expect_reward" in rows[0]:
        m["expect_reward_late_max"] = round(float(f("expect_reward")[late].max()), 3)
        m["expect_threat_late_max"] = round(float(f("expect_threat")[late].max()), 3)
        m["led_activity_mean"] = round(float(f("led_activity").mean()), 3)
    m["errors"] = len(re.findall(r"^E \(", log, re.M))
    last = re.findall(r"snn_resets=(\d+) policy_faults=(\d+) missed_ticks=(\d+)", log)
    if last:
        m["snn_resets/policy_faults/missed"] = list(map(int, last[-1]))
    return m


if __name__ == "__main__":
    kind, scn = sys.argv[1], sys.argv[2]
    minutes = float(sys.argv[3]) if len(sys.argv) > 3 else None
    out = os.path.join(ROOT, "simulator", "out", "eval_" + os.path.basename(scn).replace(".scn", ""))
    run(kind, scn, out, minutes)
    for k, v in metrics(out).items():
        print(f"{k:30s} {v}")
