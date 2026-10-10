"""The reward of the policy's reinforcement stage (train_rl.py): what a moment of the robot's life is worth.

Values, not rules: nothing here says "glance every 8 s" or "at most 6 calls a minute" - it says what moving and talking
cost and what they bring, and the policy finds the rhythm itself. Computed from a rollout's bus dump (policy input,
efference copy, executed action per tick), so it needs nothing the robot does not sense itself.

  ./py reward.py data/rl/it0/ep7000/bus.bin     # the components of one episode
"""
import sys

import numpy as np

from spec import ACTION, EFF, POLICY_INPUT, pin

N_IN, N_EFF, N_OUT = len(POLICY_INPUT), len(EFF), len(ACTION)
REC = N_IN + N_EFF + N_OUT

# what things are worth (per event or per tick)
W = dict(
    petted=0.6,          # a touch while not annoyed: the best thing that can happen
    answer=0.5,          # a call that starts within 1.5 s of a touch or a bang: someone reached out, the robot answers
    person=0.02,         # per tick: a person in view and in the middle of it (attention to people)
    move_deg=-0.002,     # per degree the head travels: servo noise (the robot is deaf while it moves) and wear
    move_start=-0.08,    # per head movement started: every start and stop clicks
    glance_bored=0.35,   # per head movement started, times boredom and the novelty of looking again (the time since the
                         # last movement, full after 10 s: a place seen a second ago is not new)
    talk=-0.003,         # per tick of its own voice: talking costs
    talk_lonely=0.006,   # ... and is worth something times social need (calling for company)
    talk_over=-0.02,     # per tick of its own voice while someone else speaks: do not talk over people
    annoyed=-0.01,       # per tick, times the annoyance above its resting level (0.15)
)


def load(path):
    d = np.fromfile(path, np.float32)
    return d[: d.size // REC * REC].reshape(-1, REC)


def onsets(x, th=0.5):
    on = x > th
    return on & ~np.concatenate([[False], on[:-1]])


def components(d):
    """d: [T, REC] bus dump -> dict of per-tick reward arrays."""
    x = d[:, :N_IN]
    eff = d[:, N_IN:N_IN + N_EFF]
    c = lambda n: x[:, pin(n)]
    T = len(d)
    r = {}
    touch = np.maximum(c("touch_head"), c("touch_screen"))
    t_on = onsets(touch)
    r["petted"] = W["petted"] * (t_on & (c("annoyance") < 0.45))
    bang_on = onsets(np.maximum.reduce([c("clap"), c("sharp"), c("loud")]))
    speak_on = onsets(c("speaker"))
    trigger = np.convolve((t_on | bang_on).astype(float), np.ones(30))[:T] > 0  # within 30 ticks after a touch / bang
    r["answer"] = W["answer"] * (speak_on & trigger)
    r["person"] = W["person"] * ((c("person") > 0.4) & (np.abs(c("target_x")) < 0.25))
    yaw, pitch = c("yaw") * 100.0, c("pitch") * 35.0
    deg = np.concatenate([[0.0], np.hypot(np.diff(yaw), np.diff(pitch))])
    r["move_deg"] = W["move_deg"] * deg
    # a movement starts when the efference copy (commanded minus actual) jumps up
    e = np.hypot(eff[:, 0] * 60.0, eff[:, 1] * 40.0)
    start = (e > 6.0) & (np.concatenate([[0.0], e[:-1]]) < 3.0)
    r["move_start"] = W["move_start"] * start
    since = np.zeros(T)
    last = -10 ** 6
    for t in np.nonzero(start)[0]:
        since[t] = min(1.0, (t - last) / 200.0)  # ticks; 200 = 10 s
        last = t
    r["glance_bored"] = W["glance_bored"] * start * c("slow_boredom") * since
    speaking = c("speaker") > 0.5
    r["talk"] = W["talk"] * speaking
    r["talk_lonely"] = W["talk_lonely"] * speaking * c("slow_social")
    r["talk_over"] = W["talk_over"] * speaking * (c("speech") > 0.5)
    r["annoyed"] = W["annoyed"] * np.maximum(0.0, c("annoyance") - 0.15)
    return r


def reward(d):
    return np.sum(list(components(d).values()), axis=0).astype(np.float32)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        comp = components(load(path))
        T = len(next(iter(comp.values())))
        print(f"{path}: {T} ticks, total {sum(v.sum() for v in comp.values()):+.2f}")
        for k, v in comp.items():
            print(f"  {k:13s} {v.sum():+8.2f}  ({np.count_nonzero(v)} ticks)")
