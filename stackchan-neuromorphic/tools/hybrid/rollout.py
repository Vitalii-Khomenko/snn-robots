"""Closed-loop rollouts of the firmware in the digital twin.

The simulator runs the real firmware (front ends, Vision / Auditory NNs,
spiking core, executor, vocal chain); only the policy is outside, over a
pipe: every 50 ms tick the sim writes the 100 policy inputs and reads the 51
actions. The policy here is the teacher, or a torch policy (DAgger learner).

  ./py rollout.py teacher 40 data/policy/it0          # 40 random scenarios, teacher acting
  ./py rollout.py learner 40 data/policy/it1 --model data/policy_net.pt
"""
import argparse
import os
import random
import struct
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SIM = os.path.join(ROOT, "simulator", "build", "stackchan_sim")
from spec import ACTION  # noqa: E402

from spec import POLICY_INPUT  # noqa: E402

N_IN, N_OUT = len(POLICY_INPUT), len(ACTION)


# AssocMemory::cueName, '_' for spaces
CUES = ["voice", "bang", "tone", "steady_noise", "person", "face", "motion", "approach", "light_change", "owner_home",
        "night"]


def t(sec):
    return f"{int(sec // 60)}:{sec % 60:05.2f}"


def rollout_scenario(seed, minutes, quiet=False):
    """A random but plausible stretch of life: visitors, touches, noises, light, time of day. quiet: alone, or nearly
    (the reinforcement stage needs the idle rhythm, which busy episodes hardly show)."""
    r = random.Random(seed)
    dur = minutes * 60
    clock = r.choice([r.uniform(0, 24), 6.5 + r.uniform(0, 0.5), 21.2 + r.uniform(0, 0.4), 22.3, 9.0, 14.0, 19.5])
    L = [f"duration {minutes}:00", f"clock {int(clock):02d}:{int((clock % 1) * 60):02d}", f"seed {seed}",
         f"room_light {r.choice(['on', 'on', 'off'])}", "sunrise 07:00", "sunset 19:30",
         f"noise_floor {r.choice([15, 25, 25, 40])}"]
    if r.random() < 0.7:
        L.append(f"window {r.uniform(-180, 180):.0f} {r.uniform(5, 25):.0f} {r.uniform(25, 50):.0f} {r.uniform(15, 30):.0f}")
    lamps = []
    for i in range(r.choice([0, 1, 1, 2])):
        L.append(f"lamp l{i} {r.uniform(-170, 170):.0f} {r.uniform(0, 25):.0f} {r.choice(['on', 'on', 'off'])} "
                 f"brightness {r.uniform(3, 8):.1f}")
        lamps.append(f"l{i}")
    L.append(f"owner {r.choice(['home', 'away'])}")
    # experience the robot already has (AssocMemory, learned on earlier days): half of the episodes start with some
    if r.random() < 0.5:
        for _ in range(r.randrange(1, 4)):
            cue = r.choice(CUES)
            if r.random() < 0.75:
                L.append(f"learned {cue} {r.uniform(0.3, 1.0):.2f} 0")
            else:
                L.append(f"learned {cue} 0 {r.uniform(0.3, 1.0):.2f}")
    # a quarter of the lively episodes: somebody sits there the whole time and hardly pays attention (at the PC, reading)
    sitter = not quiet and r.random() < 0.25
    busy = r.choice([0.02, 0.06]) if quiet else 0.08 if sitter else r.choice([0.15, 0.4, 1.0, 2.0])  # lonely .. lively
    if r.random() < 0.06:  # now and then a nearly empty battery (Sick)
        L.append(f"battery {r.choice([4, 6, 7])}")
        L.append("charging off")
    if sitter:
        L.append(f"at 0:05 person v enter {r.uniform(-80, 80):.0f} {r.uniform(0.8, 2.0):.1f}")
    now = r.uniform(2, 20) / busy
    person_in = sitter
    while now < dur - 10:
        c = r.random()
        if not person_in and c < 0.3:
            L.append(f"at {t(now)} person v enter {r.uniform(-90, 90):.0f} {r.uniform(1, 3.5):.1f}")
            person_in = True
            if r.random() < 0.6:
                L.append(f"at {t(now + 0.5)} owner home")
        elif person_in and c < 0.5:
            k = r.random()
            if k < 0.4:
                L.append(f"at {t(now)} person v walk {r.uniform(-100, 100):.0f} {r.uniform(1.5, 6):.1f}")
            elif k < 0.7:
                L.append(f"at {t(now)} person v approach {r.uniform(0.4, 3):.1f} {r.uniform(1, 4):.1f}")
            else:
                L.append(f"at {t(now)} person v step_back {r.uniform(1, 4):.1f}")
        elif person_in and c < 0.58:
            L.append(f"at {t(now)} person v leave")
            person_in = False
        elif c < 0.68 and person_in:
            talk_s = r.uniform(1, 5)
            L.append(f"at {t(now)} talk {r.uniform(-90, 90):.0f} {talk_s:.1f} {r.uniform(0.5, 1):.2f}")
            if r.random() < 0.45:  # a habit to learn: the visitor talks to the robot, then pets it
                L.append(f"at {t(now + talk_s + r.uniform(0.3, 2.0))} touch_head {r.choice([300, 500, 1500])}")
        elif c < 0.76:
            L.append(f"at {t(now)} touch_head {r.choice([300, 500, 1500])}")
            if r.random() < 0.25:  # a burst of repeated touches
                for k in range(1, r.randrange(3, 9)):
                    L.append(f"at {t(now + k * r.uniform(1.0, 3.0))} touch_head 300")
        elif c < 0.80:
            L.append(f"at {t(now)} {r.choice(['tap_screen', 'double_tap_screen'])}")
        elif c < 0.87:
            L.append(f"at {t(now)} clap {r.uniform(-120, 120):.0f} {r.uniform(0.4, 1):.2f}")
        elif c < 0.885:
            if person_in and r.random() < 0.5:  # another habit: someone comes close, then shakes it
                L.append(f"at {t(max(0.0, now - 2.0))} person v approach {r.uniform(0.4, 0.8):.1f} 1.5")
            L.append(f"at {t(now)} shake {r.uniform(1, 2.5):.1f}")
        elif c < 0.895:
            # picked up / turned over for a few seconds, then put down
            L.append(f"at {t(now)} pick_up {r.choice([25, 45, 90, 150, 180, -150])}")
            L.append(f"at {t(now + r.uniform(2.5, 7))} put_down")
        elif c < 0.92 and lamps:
            L.append(f"at {t(now)} lamp {r.choice(lamps)} {r.choice(['on', 'off'])}")
        elif c < 0.94:
            L.append(f"at {t(now)} {r.choice(['bang', 'hiss'])} {r.uniform(-120, 120):.0f}")
        elif c < 0.96:
            L.append(f"at {t(now)} noise {r.choice([0, 250, 400])} {r.choice([0, 0.9])}")
        elif c < 0.97:
            L.append(f"at {t(now)} thing o enter {r.uniform(-60, 60):.0f} {r.uniform(0, 20):.0f} {r.uniform(5, 14):.0f}")
            L.append(f"at {t(now + 1)} thing o move {r.uniform(-90, 90):.0f} {r.uniform(0, 20):.0f} {r.uniform(1, 4):.1f}")
        now += r.expovariate(busy / 25.0) + 0.5
    return "\n".join(L) + "\n"


class Env:
    """One simulator process behind a pipe."""

    def __init__(self, scn, out_dir, dump, seed=None, minutes=None):
        os.makedirs(out_dir, exist_ok=True)
        cmd = [SIM, scn, "--policy-pipe", "--out", out_dir, "--bus-dump", dump]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        if minutes is not None:
            cmd += ["--minutes", str(minutes)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        self.rd = self.p.stdout

    def read_state(self):
        raw = self.rd.read(N_IN * 4)
        if len(raw) < N_IN * 4:
            return None
        return np.frombuffer(raw, np.float32)

    def act(self, action):
        self.p.stdin.write(np.asarray(action, np.float32).tobytes())

    def close(self):
        try:
            self.p.stdin.close()
        except Exception:
            pass
        self.p.wait(timeout=60)


def run_episode(job):
    kind, scn, out_dir, seed, minutes, model = job
    from spec import ACTION
    dump = os.path.join(out_dir, "bus.bin")
    env = Env(scn, out_dir, dump, seed, minutes)
    if kind == "teacher":
        import teacher_policy as tp
        memory = tp.Memory()  # the teacher's memory along the episode (rhythm of glances and calls)

        def policy(x):
            return tp.label(x[None], memory)[0]
    else:
        import torch
        from models import PolicyNet
        from spec import ACTION_SIGNED
        net = PolicyNet()
        net.load_state_dict(torch.load(model, map_location="cpu"))
        net.eval()
        # float weights (train_rl.py runs the reinforcement stage in float: with fake INT8 weights a tiny update flipped
        # rounding levels and moved the outputs in jumps); the export quantizes, the QAT-trained base keeps that close
        signed = torch.tensor([n in ACTION_SIGNED for n in ACTION])
        h = [None]
        # explore (train_rl.py): Gaussian noise on the outputs the reward shapes, held for a second at a time (a trial of
        # "look there for a while", "wait", "speak now", not a jitter); the sampled action is saved for the gradient
        explore = kind == "explore"
        rng = np.random.default_rng(seed)
        sigma = np.zeros(len(ACTION), np.float32)
        if explore:
            from train_rl import NOISE_BLOCK, RL_SIGMA
            for name, s in RL_SIGMA.items():
                sigma[ACTION.index(name)] = s
        noise = np.zeros(len(ACTION), np.float32)
        block = NOISE_BLOCK if explore else 1
        taken = []

        def policy(x):
            nonlocal noise
            with torch.no_grad():
                o, h[0] = net(torch.from_numpy(x.copy()).view(1, 1, -1), h[0])
                o = o[0, 0]
                a = torch.where(signed, torch.tanh(o), torch.sigmoid(o)).numpy()
            if explore:
                if len(taken) % block == 0:
                    noise = (rng.standard_normal(len(ACTION)) * sigma).astype(np.float32)
                a = a + noise
                taken.append(a.copy())
            return a
    n = 0
    while True:
        x = env.read_state()
        if x is None:
            break
        env.act(policy(x))
        n += 1
    env.close()
    if kind == "explore":
        np.asarray(taken, np.float32).tofile(os.path.join(out_dir, "act.bin"))
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["teacher", "learner", "explore"])
    ap.add_argument("count", type=int)
    ap.add_argument("out")
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--first-seed", type=int, default=5000)
    ap.add_argument("--quiet", type=float, default=0.0, help="share of quiet episodes (alone or nearly)")
    ap.add_argument("--model", default=os.path.join(HERE, "data", "policy_net.pt"))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    jobs = []
    for i in range(a.count):
        seed = a.first_seed + i
        d = os.path.join(out, f"ep{seed}")
        os.makedirs(d, exist_ok=True)
        scn = os.path.join(d, "scenario.scn")
        with open(scn, "w") as f:
            f.write(rollout_scenario(seed, int(a.minutes), quiet=i < a.count * a.quiet))
        jobs.append((a.kind, scn, d, seed, a.minutes, a.model))
    with ProcessPoolExecutor(a.jobs) as ex:
        ticks = list(ex.map(run_episode, jobs))
    print(f"{len(jobs)} episodes, {sum(ticks)} ticks -> {out}")
