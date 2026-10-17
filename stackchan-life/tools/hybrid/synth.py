"""Synthetic sensory/body sequences (20 Hz) for training the spiking core.

A latent timeline of events - people coming and going, things moving, claps,
speech, noise, touches (sometimes a burst of repeated ones), lamps, the
owner, day and night, the robot's own movements and voice - is painted into
the 66-value bus the way the perception networks and body encoders report
them (decaying, noisy, with false alarms and dropouts). It covers hours of
behaviour per second of generation, which the simulator cannot.
"""
import numpy as np

from spec import SNN_INPUT, sin

DT = 0.05
IDX = {n: sin(n) for n in SNN_INPUT}


def pulse(x, i, n, value, decay_ticks):
    """Add value*exp decay starting at tick i (max-combined)."""
    if i >= len(x):
        return
    j = np.arange(min(n, len(x) - i))
    x[i:i + len(j)] = np.maximum(x[i:i + len(j)], value * np.exp(-j / max(decay_ticks, 1e-3)))


def box(x, i, n, value):
    x[i:i + n] = np.maximum(x[i:i + n], value)


def ou(rng, T, tau_s, sigma=0.5):
    a = np.exp(-DT / tau_s)
    n = rng.normal(0, sigma * np.sqrt(1 - a * a), T)
    y = np.empty(T)
    v = rng.normal(0, sigma)
    for t in range(T):
        v = a * v + n[t]
        y[t] = v
    return np.clip(y, -1, 1)


def make_sequence(rng, T, dense=None):
    if dense is None:
        dense = float(np.exp(rng.uniform(np.log(0.05), np.log(2.5))))  # from a lonely quiet room to a busy one
    X = np.zeros((T, len(SNN_INPUT)), np.float32)
    col = lambda n: X[:, IDX[n]]
    sec = lambda s: int(s / DT)

    # ---- time of day: starts at a random hour and runs in real time
    hour0 = rng.uniform(0, 24) if rng.random() < 0.6 else rng.choice([6.55, 21.3, 22.5, 5.5, 12.0]) + rng.uniform(0, 0.2)
    hours = (hour0 + np.arange(T) * DT / 3600.0) % 24
    night = ((hours >= 22) | (hours < 6.5)).astype(np.float32)
    day = ((hours >= 6.5) & (hours < 20)).astype(np.float32)
    dawn = np.clip((hours - 6.6667) / (2 / 60), 0, 1) * ((hours >= 6.6667) & (hours < 7.1167))
    dusk = np.clip((hours - 21.4167) / (2 / 60), 0, 1) * ((hours >= 21.4167) & (hours < 21.8667))
    col("night")[:], col("day")[:], col("dawn")[:], col("dusk")[:] = night, day, dawn, dusk
    col("hour_sin")[:], col("hour_cos")[:] = np.sin(hours / 24 * 2 * np.pi), np.cos(hours / 24 * 2 * np.pi)
    owner = np.zeros(T, np.float32)
    t0 = 0
    state = rng.random() < 0.5
    while t0 < T:
        n = sec(rng.uniform(300, 4000))
        owner[t0:t0 + n] = float(state)
        if state is False and t0 + n < T:
            X[min(t0 + n, T - 1), IDX["owner_arrived"]] = 1.0
        state = not state
        t0 += n
    col("owner_near")[:] = owner

    # ---- the robot's body: random head pose, calm most of the time
    yaw = ou(rng, T, 15.0, 0.4)
    pitch = ou(rng, T, 20.0, 0.25)
    vel_y = np.gradient(yaw) / DT * 100 / 180
    vel_p = np.gradient(pitch) / DT * 35 / 180
    col("yaw")[:], col("pitch")[:] = yaw, pitch
    col("yaw_vel")[:], col("pitch_vel")[:] = np.clip(vel_y, -1, 1), np.clip(vel_p, -1, 1)
    col("real_yaw_vel")[:] = np.clip(vel_y + rng.normal(0, 0.01, T), -1, 1)
    col("real_pitch_vel")[:] = np.clip(vel_p + rng.normal(0, 0.01, T), -1, 1)
    col("imu_pitch")[:] = np.clip(pitch * 0.3 + rng.normal(0, 0.01, T), -1, 1)
    col("imu_roll")[:] = rng.normal(0, 0.01, T)
    col("battery")[:] = rng.uniform(0.3, 1.0)
    stam = np.clip(ou(rng, T, 600.0, 0.5) * 0.5 + rng.uniform(0.4, 1.0), 0.05, 1.0)
    col("stamina")[:] = stam
    col("light_delta")[:] = np.clip(rng.normal(0, 0.01, T), -1, 1)
    moving = (np.abs(vel_y) > 0.04) | (np.abs(vel_p) > 0.04)
    col("own_motion")[:] = np.convolve(moving.astype(float), np.ones(8), "same").clip(0, 1)

    # ---- noise pacemakers
    col("wander_x")[:] = ou(rng, T, 6.0)
    col("wander_y")[:] = ou(rng, T, 8.0, 0.4)
    col("wander_slow")[:] = ou(rng, T, 90.0)
    col("jitter")[:] = ou(rng, T, 0.6, 0.4)

    # ---- people
    t = sec(rng.exponential(60 / dense))
    while t < T:
        dur = sec(rng.uniform(8, 150))
        px = rng.uniform(-0.8, 0.8)
        walk = rng.uniform(-0.02, 0.02)
        facing = rng.random() < 0.7
        for k in range(t, min(T, t + dur)):
            px = np.clip(px + walk + rng.normal(0, 0.01), -0.95, 0.95)
            X[k, IDX["person"]] = max(X[k, IDX["person"]], 0.9 if rng.random() > 0.03 else 0.2)
            if facing:
                X[k, IDX["face"]] = max(X[k, IDX["face"]], 0.7)
            X[k, IDX["target_x"]] = px
            X[k, IDX["target_y"]] = -0.1 + rng.normal(0, 0.03)
            X[k, IDX["target_conf"]] = 0.9
        # activity while someone moves about
        for _ in range(rng.integers(2, 10)):
            i = t + rng.integers(0, max(dur, 1))
            if i < T:
                box(col("activity"), i, sec(rng.uniform(0.5, 3)), rng.uniform(0.3, 0.9))
                m = "motion_l" if px < -0.2 else "motion_r" if px > 0.2 else "motion_c"
                box(col(m), i, sec(1.0), 0.6)
        if rng.random() < 0.4:  # approaching
            i = t + rng.integers(0, max(dur, 1))
            box(col("loom"), i, sec(rng.uniform(0.5, 2)), rng.uniform(0.5, 1.0))
        if rng.random() < 0.3:
            i = min(T - 1, t + dur - sec(1))
            box(col("recede"), i, sec(1.0), 0.6)
        if rng.random() < 0.7:  # speech while there
            for _ in range(rng.integers(1, 5)):
                i = t + rng.integers(0, max(dur, 1))
                n = sec(rng.uniform(1, 5))
                box(col("speech"), i, n, rng.uniform(0.6, 0.95))
                box(col("energy"), i, n, rng.uniform(0.3, 0.7))
                pulse(col("novelty"), i, sec(2), 0.6, sec(0.6))
                pulse(col("onset"), i, 6, 0.5, 2)
        t += dur + sec(rng.exponential(120 / dense))

    # ---- moving things and light changes
    for _ in range(rng.poisson(T * DT / 45 * dense)):
        i = rng.integers(0, T)
        n = sec(rng.uniform(0.5, 4))
        side = rng.integers(0, 3)
        box(col(["motion_l", "motion_c", "motion_r"][side]), i, n, rng.uniform(0.4, 0.9))
        box(col("activity"), i, n, rng.uniform(0.3, 0.8))
        box(col("target_conf"), i, n, 0.5)
        X[i:i + n, IDX["target_x"]] = (side - 1) * 0.6 + rng.normal(0, 0.1)
    for _ in range(rng.poisson(T * DT / 400)):
        i = rng.integers(0, T)
        pulse(col("light_change"), i, sec(0.6), rng.choice([-1, 1]) * rng.uniform(0.4, 1.0), sec(0.2))
        pulse(col("activity"), i, sec(0.6), 0.6, sec(0.2))
    col("brightness")[:] = np.clip(rng.uniform(0.15, 0.6) + 0.1 * day, 0, 1)
    ls = rng.random() < 0.4
    if ls:
        col("light_strength")[:] = rng.uniform(0.3, 0.9)
        col("light_x")[:] = rng.uniform(-0.8, 0.8)
        col("light_y")[:] = rng.uniform(-0.5, 0.3)

    # ---- sounds
    def sound(i, kind):
        lvl = rng.uniform(0.4, 1.0)
        if kind == "clap":
            pulse(col("clap"), i, 8, 0.9, 3)
            pulse(col("loud"), i, 8, lvl, 3)
            pulse(col("onset"), i, 6, 0.9, 2)
            pulse(col("energy"), i, 10, lvl, 3)
            pulse(col("novelty"), i, sec(2), 0.8, sec(0.8))
        elif kind == "sharp":
            n = sec(rng.uniform(0.1, 0.6))
            box(col("sharp"), i, n, 0.9)
            pulse(col("loud"), i, 8, lvl, 3)
            pulse(col("energy"), i, n, lvl, sec(0.3))
            pulse(col("novelty"), i, sec(2), 0.7, sec(0.8))
            box(col(rng.choice(["low", "high"])), i, n, 0.8)
        elif kind == "tone":
            n = sec(rng.uniform(0.3, 2.5))
            box(col("tonal"), i, n, 0.9)
            box(col("energy"), i, n, lvl * 0.5)
            pulse(col("novelty"), i, sec(1.5), 0.5, sec(0.6))
        elif kind == "noise":
            n = sec(rng.uniform(10, 120))
            box(col("continuous"), i, n, 0.9)
            box(col("energy"), i, n, rng.uniform(0.15, 0.5))
            pulse(col("novelty"), i, sec(3), 0.4, sec(1.5))
    for _ in range(rng.poisson(T * DT / 70 * dense)):
        sound(rng.integers(0, T), rng.choice(["clap", "clap", "sharp", "tone", "noise"]))
    # a clap train now and then (rhythm, habituation to loud sounds)
    for _ in range(rng.poisson(T * DT / 600)):
        i, gap = rng.integers(0, T), sec(rng.uniform(0.3, 1.0))
        for k in range(rng.integers(3, 12)):
            sound(i + k * gap, "clap")

    # ---- touches: single, and bursts of repeated ones (annoyance)
    def touch(i, name):
        box(col(name), i, 3, 1.0)
    for _ in range(rng.poisson(T * DT / 120 * dense)):
        touch(rng.integers(0, max(T - 3, 1)), rng.choice(["touch_head", "touch_head", "touch_screen"]))
    for _ in range(rng.poisson(T * DT / 500 * dense)):
        i, gap = rng.integers(0, T), sec(rng.uniform(1.0, 3.0))
        for k in range(rng.integers(4, 12)):
            if i + k * gap < T - 3:
                touch(i + k * gap, "touch_head")
    for _ in range(rng.poisson(T * DT / 900)):
        box(col("shake"), rng.integers(0, T), sec(1.5), 1.0)
    for _ in range(rng.poisson(T * DT / 1500)):
        box(col("tilt"), rng.integers(0, T), sec(2.0), 1.0)

    # ---- own voice and movement intent
    for _ in range(rng.poisson(T * DT / 40)):
        i, n = rng.integers(0, T), sec(rng.uniform(0.3, 1.5))
        box(col("eff_sound"), i, n, 1.0)
        box(col("speaker"), i, n, 1.0)
        box(col("self_voice"), i, n, rng.uniform(0.3, 0.9))
    # stalls: a command that the body did not follow
    for _ in range(rng.poisson(T * DT / 600)):
        i = rng.integers(0, T)
        box(col("pe_motor"), i, sec(2), rng.uniform(0.4, 0.9))
    X[:, IDX["pe_motor"]] = np.maximum(X[:, IDX["pe_motor"]], np.abs(rng.normal(0, 0.02, T)))
    X[:, IDX["pe_vision"]] = np.abs(rng.normal(0, 0.03, T))
    col("eff_yaw")[:] = np.clip(vel_y * 2, -1, 1)
    col("eff_pitch")[:] = np.clip(vel_p * 2, -1, 1)

    # ---- perception noise: small false alarms and jitter on every feature
    for n in ("person", "face", "activity", "motion_l", "motion_c", "motion_r", "motion_up", "motion_down", "loom"):
        col(n)[:] = np.clip(col(n) + np.abs(rng.normal(0, 0.012, T)), 0, 1)
    for n in ("silence", "speech", "clap", "sharp", "continuous", "low", "high", "tonal", "energy", "novelty"):
        col(n)[:] = np.clip(col(n) + np.abs(rng.normal(0, 0.01, T)), 0, 1)
    ext = np.maximum.reduce([col(n) for n in ("speech", "clap", "sharp", "continuous", "tonal")])
    col("silence")[:] = np.clip(1.0 - ext, 0, 1)
    # features in the bus are what the NNs say, never exactly the truth
    return X
