"""The robot's night sleep on the PC (platform B: the slow learning loop). Every night:

  1. the whole diary (every day so far) -> training / held-out windows (held-out = the last 20 % of every run)
  2. the world model now on the robot (tools/hybrid/data/world_net.pt) is scored on the held-out windows
  3. a candidate starts from it and trains on the training windows (replay of all days: nothing is forgotten on purpose)
  4. the gate: the candidate is not worse than the current model in any group (vision, audio, body) by more than
     0.005 skill - otherwise the robot keeps what it has
  5. weights blob with the candidate (pack_weights.py --override) -> ota.py weights: switched in on the robot without
     a reboot, on trial for a minute; only once the robot confirms it does the candidate become the current model
  6. a few lines of what happened: data/life/<day>/night.txt and data/life/sleep.log

Since 2026-10-14 the critic and the actor sleep too (train_critic.py, train_life_actor.py): the actor (it decides when
the robot moves, where to and when it calls, in mode life) goes to the robot only if its behaviour stays in sane
bounds (0.5-8 moves and 0.3-6 calls a minute by day).

Since 2026-10-17 the eyes and ears sleep too (label_media.py, train_perception.py): the Vision and Auditory nets are
fine-tuned on the day's frames and sounds and go to the robot if they forgot nothing and agree better with the teachers.

  ./py ../life/sleep.py [--epochs 20] [--dry-run]       (from tools/hybrid; systemd user timer life-sleep, 03:00)
"""
import argparse
import json
import datetime as dt
import os
import shutil
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_world as W  # noqa: E402

ROOT = W.L.ROOT
LIVE = os.path.join(ROOT, "tools", "hybrid", "data", "world_net.pt")
GATE = 0.005


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--quant-epochs", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true", help="train and judge, but do not send anything to the robot")
    a = ap.parse_args()
    started = time.time()
    night = dt.date.today() - dt.timedelta(days=1) if dt.datetime.now().hour < 12 else dt.date.today()
    out_dir = os.path.join(ROOT, "data", "sleep", dt.datetime.now().strftime("%Y-%m-%d_%H%M"))
    os.makedirs(out_dir, exist_ok=True)
    lines = []

    def say(m):
        print(m, flush=True)
        lines.append(m)

    days = W.all_days()
    eps = W.episodes(days)
    tr, va = W.windows(eps, 0.8)
    hours = sum(len(e[0]) for e in eps) / 72000
    say(f"night of {night}: diary {len(days)} day(s), {hours:.1f} h; {len(tr)} training / {len(va)} held-out windows")
    if len(tr) < 50 or len(va) < 10:
        say("not enough diary: nothing to learn tonight")
        return finish(night, lines, started)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current = W.WorldNet(len(W.INPUT), len(W.TARGET)).to(dev)
    current.load_state_dict(torch.load(LIVE, map_location=dev))
    current.quant = True
    before = W.skill(current, va, dev)
    say("current world model: " + ", ".join(f"{g} {v:+.3f}" for g, v in before.items()))

    torch.manual_seed(int(time.time()))
    cand = W.WorldNet(len(W.INPUT), len(W.TARGET)).to(dev)
    cand.load_state_dict(current.state_dict())
    W.fit(cand, tr, va, a.epochs, a.quant_epochs, dev, lr=1e-3, log=None)
    cand.quant = True
    after = W.skill(cand, va, dev)
    say("candidate after the night: " + ", ".join(f"{g} {v:+.3f}" for g, v in after.items()))
    worse = [g for g in before if after[g] < before[g] - GATE]
    py = sys.executable
    world_h = os.path.join(out_dir, "world_model_data.h")
    cand_pt = os.path.join(out_dir, "world_net.pt")
    world_new = not worse
    if worse:
        say(f"world gate: NOT passed ({', '.join(worse)} worse by more than {GATE}); the robot keeps its world model")
        W.export(current.cpu(), world_h)
    else:
        say(f"world gate: passed (mean skill {sum(before.values()) / 3:+.3f} -> {sum(after.values()) / 3:+.3f})")
        torch.save(cand.state_dict(), cand_pt)
        W.export(cand.cpu(), world_h)

    # ---- the critic and the actor (stage 3): learned again from every day; the actor goes to the robot only if its
    # behaviour stays in sane bounds (it acts; the world model only predicts)
    actor_h = os.path.join(out_dir, "actor_model_data.h")
    actor_pt = os.path.join(out_dir, "life_actor.pt")
    actor_new = False
    env = dict(os.environ, PYTHONWARNINGS="ignore")
    r = subprocess.run([py, os.path.join(W.L.HERE, "train_critic.py"), "--epochs", "15"], capture_output=True,
                       text=True, env=env)
    say("critic: " + ((r.stdout.strip().splitlines() or ["failed"])[-1].strip())[:120])
    r = subprocess.run([py, os.path.join(W.L.HERE, "train_life_actor.py"), "--out-header", actor_h, "--out-pt",
                        actor_pt], capture_output=True, text=True, env=env)
    probe = next((json.loads(ln[6:]) for ln in r.stdout.splitlines() if ln.startswith("PROBE ")), None)
    if probe is None:
        say("actor: training failed; the robot keeps its actor")
    else:
        sane = 2.0 <= probe["moves / min (day)"] <= 8 and 0.3 <= probe["calls / min (day)"] <= 8 and \
            probe["value: explained variance"] > 0.0
        say("actor: " + ", ".join(f"{k} {v:.2f}" for k, v in probe.items()) + ("" if sane else " - OUT OF BOUNDS"))
        actor_new = sane
    if not actor_new:
        live_actor = os.path.join(ROOT, "tools", "hybrid", "data", "life_actor.pt")
        if os.path.exists(live_actor):
            import train_life_actor as LA
            m = LA.PolicyNet(len(LA.C.STATE), len(LA.OUT))
            m.load_state_dict(torch.load(live_actor, map_location="cpu"))
            LA.export(m, actor_h)
        else:
            actor_h = None
    # ---- perception (since 2026-10-17): the robot's own frames and sounds, labelled by the big nets on the PC
    perc = {}
    r = subprocess.run([py, os.path.join(W.L.HERE, "label_media.py")], capture_output=True, text=True, env=env)
    say(((r.stdout.strip().splitlines() or ["label_media failed"])[-1])[:120])
    r = subprocess.run([py, os.path.join(W.L.HERE, "train_perception.py"), "--out-dir", out_dir], capture_output=True,
                       text=True, env=env)
    for ln in r.stdout.splitlines():
        if " gate: " in ln or "not enough" in ln or "hidden layer moved" in ln:
            say(ln.strip()[:160])
    pp = next((json.loads(ln[6:]) for ln in r.stdout.splitlines() if ln.startswith("PROBE ")), None)
    if pp is None:
        say("perception: training failed; the robot keeps its eyes and ears")
    else:
        perc = {n: os.path.join(out_dir, f"{n}_model_data.h") for n, v in pp.items() if v and v.get("passed")}
    if not world_new and not actor_new and not perc:
        say("nothing new for the robot tonight")
        return finish(night, lines, started)

    blob = os.path.join(out_dir, "weights.bin")
    cmd = [py, os.path.join(W.L.HERE, "pack_weights.py"), "--out", blob, "--override", f"world={world_h}"]
    if actor_h:
        cmd += ["--override", f"actor={actor_h}"]
    for n, h in perc.items():
        cmd += ["--override", f"{n}={h}"]
    subprocess.run(cmd, check=True, capture_output=True)
    if a.dry_run:
        say(f"dry run: candidates kept in {out_dir}, nothing sent")
        return finish(night, lines, started)
    # the home Wi-Fi blinks now and then (2026-10-12 03:00: "Network is unreachable" for two minutes): try again
    for attempt in range(6):
        r = subprocess.run([py, os.path.join(W.L.HERE, "ota.py"), "weights", blob], capture_output=True, text=True)
        if r.returncode == 0 or "went back" in r.stdout + r.stderr:
            break
        say(f"robot not reachable (attempt {attempt + 1}), again in 3 min")
        time.sleep(180)
    say("robot: " + (r.stdout.strip().splitlines() or ["(no answer)"])[-1][:160])
    if r.returncode != 0:
        say("the robot did not take the new weights (unreachable or rolled back); the current models stay")
        if r.stderr.strip():
            say("  " + r.stderr.strip().splitlines()[-1][:160])
        return finish(night, lines, started)
    if world_new:
        shutil.copy(LIVE, os.path.join(out_dir, "world_net_before.pt"))
        shutil.copy(cand_pt, LIVE)
    if actor_new:
        shutil.copy(actor_pt, os.path.join(ROOT, "tools", "hybrid", "data", "life_actor.pt"))
    for n in perc:  # the perception nets the next night starts from (and their history)
        live_pt = os.path.join(ROOT, "tools", "hybrid", "data", f"{n}_net.pt")
        shutil.copy(live_pt, os.path.join(out_dir, f"{n}_net_before.pt"))
        shutil.copy(os.path.join(out_dir, f"{n}_net.pt"), live_pt)
    say(f"live on the robot: world model {'new' if world_new else 'kept'}, actor {'new' if actor_new else 'kept'}"
        + "".join(f", {n} new" for n in perc))
    return finish(night, lines, started)


def finish(night, lines, started):
    lines.append(f"slept {(time.time() - started) / 60:.1f} min")
    folder = os.path.join(ROOT, "data", "life", night.isoformat())
    if os.path.isdir(folder):
        with open(os.path.join(folder, "night.txt"), "a") as f:
            f.write("\n".join(lines) + "\n\n")
    with open(os.path.join(ROOT, "data", "life", "sleep.log"), "a") as f:
        f.write(dt.datetime.now().strftime("%Y-%m-%d %H:%M ") + " | ".join(lines) + "\n")
    print(lines[-1])


if __name__ == "__main__":
    main()
