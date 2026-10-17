"""The robot's temperament on the PC: what the needs are worth and what moving costs. The reward of every tick is
recomputed from the needs the diary records (main/hybrid/motivation.h logs them), so the temperament can be tuned
without touching the robot: r = sum_i w_i (d_i(t-1)^2 - d_i(t)^2) - move_cost * degrees.

2026-10-12: with the robot's own weights (move cost 0.002 / deg) a collector day summed to -135, of which -163 was
moving: an actor would only learn to keep still. The cost is 4x lower here; a stroke (social 0.6 -> 0.36) is worth
~15 moves of 30 deg.
"""
import numpy as np

import lifedata as L

WEIGHTS = {"need_energy": 0.5, "need_social": 1.0, "need_stimulation": 0.7, "need_rest": 0.5, "need_comfort": 1.5}
MOVE_COST = 0.0002  # per degree of head travel (0.0005 until 2026-10-15: the actor learned to hardly move, ~20 deg a minute)
# per call started: until 2026-10-15 the robot's own voice eased its boredom a little (motivation.cpp), the diary keeps
# that, and the actor learned to call ~6 times a minute; a call costs something now
CALL_COST = 0.01


def reward(v):
    """v: [T, columns] diary values (one run, in order) -> [T] reward."""
    r = np.zeros(len(v), np.float32)
    for name, w in WEIGHTS.items():
        d = v[:, L.col(name)]
        r[1:] += w * (d[:-1] ** 2 - d[1:] ** 2)
    r -= MOVE_COST * v[:, L.col("move_deg100")] * 100.0
    snd = v[:, L.col("act_sound_act")] > 0.5
    r[1:] -= CALL_COST * (snd[1:] & ~snd[:-1])
    return r
