"""Synthetic lives in raw sensor signals (20 Hz) for training brain v4's spiking analyzer.

A latent timeline - time of day and lamps, the owner coming and going, talk, TV, the vacuum cleaner, claps and
bangs, petting, pokes, taps on the face, being picked up, shaken, laid on the side, dropped, covered for the
night, the battery running down - is painted into the raw inputs of channels.py (RAW), with sensor quirks: phantom
head touches, false speech at night, light flicker, dropped camera frames. channels.encode turns them into the same
13 channels the firmware computes, so the network trains on exactly the dynamics it will see.

  make_raw(rng, T) -> dict of raw arrays;  make_channels(rng, T) -> [T, 15]
"""
import numpy as np
from numba import njit

from channels import DT, RAW, encode

SEC = lambda s: max(1, int(round(s / DT)))  # noqa: E731


@njit(cache=True)
def _ar1(n, a):
    y = np.empty(len(n))
    v = 0.0
    for t in range(len(n)):
        v = a * v + n[t]
        y[t] = v
    return y


def _ou(rng, T, tau, sigma):
    a = np.exp(-DT / tau)
    return _ar1(rng.normal(0, sigma * np.sqrt(1 - a * a), T), a)


def _segments(rng, T, rate_per_h, dur_s):
    """Random episodes: list of (start, end) ticks, Poisson starts, durations drawn from dur_s=(lo, hi) log-uniform."""
    out = []
    t = 0
    while True:
        t += int(rng.exponential(3600.0 / max(rate_per_h, 1e-6)) / DT)
        if t >= T:
            return out
        d = SEC(float(np.exp(rng.uniform(np.log(dur_s[0]), np.log(dur_s[1])))))
        out.append((t, min(T, t + d)))
        t += d


def make_raw(rng, T, busy=None, hour0=None, rich=None):
    """One synthetic stretch of life, T ticks. busy: 0 (empty flat) .. 3 (party), log-uniform when None. rich: rare
    events (handling, bangs, pokes) this many times more often than in life, so a network sees enough of them."""
    if rich is None:
        rich = float(np.exp(rng.uniform(0, np.log(30.0))))
    if busy is None:
        busy = float(np.exp(rng.uniform(np.log(0.05), np.log(3.0))))
    r = {n: np.zeros(T) for n in RAW}
    if hour0 is None:
        hour0 = rng.uniform(0, 24)
    hours = (hour0 + np.arange(T) * DT / 3600.0) % 24
    night = (hours >= 21.75) | (hours < 6.0)
    r["hour"] = hours
    r["clock_ok"] = np.ones(T) if rng.random() > 0.03 else np.zeros(T)

    # ---- the room: light, background noise
    dark_hours = night.mean() > 0.5
    lamp_on = rng.random() < (0.1 if dark_hours else 0.7)
    base_light = (10 ** rng.uniform(2.0, 3.5)) if lamp_on else (10 ** rng.uniform(-0.3, 1.2))
    light = np.full(T, base_light) * np.exp(_ou(rng, T, 120.0, 0.15))
    # the brightest spot in the picture (a lamp, a window, the TV), in view now and then as the head turns
    lstr, lx, ly = np.zeros(T), np.zeros(T), np.zeros(T)
    if rng.random() < (0.6 if lamp_on or not dark_hours else 0.1):
        for s, e in _segments(rng, T, 60, (2, 40)):
            lstr[s:e] = rng.uniform(0.1, 1.0)
            lx[s:e], ly[s:e] = rng.uniform(-0.9, 0.9), rng.uniform(-0.8, 0.4)
    for s, e in _segments(rng, T, 0.6 * busy, (20, 900)):  # lamps switched on / off, curtains
        k = 10 ** rng.uniform(-1.5, 1.5)
        light[s:e] *= k
        if k > 3 and rng.random() < 0.5:  # the lamp is in view: a new bright spot with the light
            d = min(e, s + SEC(rng.uniform(5, 60)))
            lstr[s:d] = rng.uniform(0.5, 1.0)
            lx[s:d], ly[s:d] = rng.uniform(-0.9, 0.9), rng.uniform(-0.8, 0.4)
    for s, e in _segments(rng, T, 0.15 * rich, (2, 20)):  # a torch in the face
        light[s:e] = 10 ** rng.uniform(3.8, 4.6)
    r["als"] = np.clip(light, 0, 65535)
    r["als_ok"] = np.ones(T)
    floor = np.clip(rng.uniform(0.0, 0.08) + _ou(rng, T, 300.0, 0.02), 0, 1)
    energy = floor.copy()
    speech = np.zeros(T)

    # ---- people: present episodes (the owner, sometimes guests)
    present = np.zeros(T, bool)
    p_present = np.clip(0.15 + 0.25 * busy, 0, 0.95) * (0.25 if dark_hours else 1.0)
    t = 0
    state = rng.random() < p_present
    while t < T:
        d = SEC(rng.exponential(1200 if state else 1500))
        present[t:t + d] = state
        t += d
        state = rng.random() < p_present
    motion = np.zeros(T)
    mx = np.zeros(T)
    my = np.zeros(T)
    near = np.clip(_ou(rng, T, 60.0, 15.0), 0, None)
    # walking around in view
    walk_x = _ou(rng, T, 8.0, 0.6)
    for s, e in _segments(rng, T, 25 * busy + 2, (2, 40)):
        if not present[s]:
            continue
        # a person moving shows in some frames only (2026-10-24 evening: 3 % of the frames, median share 0.015,
        # p90 0.135): it was a steady 0.02-0.3 here
        frac = rng.uniform(0.01, 0.15) * (1 + busy * 0.3)
        seen = rng.random(e - s) < rng.uniform(0.1, 0.6)
        motion[s:e] = np.maximum(motion[s:e], seen * frac * np.clip(1 + _ou(rng, e - s, 1.0, 0.4), 0.2, 2))
        mx[s:e] = np.clip(walk_x[s:e], -1, 1)
        my[s:e] = rng.uniform(-0.3, 0.5)
    for s, e in _segments(rng, T, 2 * busy * rich, (2, 10)):  # waving / flailing in front of the camera
        if present[s]:
            motion[s:e] = np.maximum(motion[s:e], rng.uniform(0.35, 0.8))
            mx[s:e] = rng.uniform(-0.4, 0.4)
            my[s:e] = rng.uniform(-0.4, 0.4)
    for s, e in _segments(rng, T, 0.3, (60, 900)):  # curtains moving, a fan, TV light in view: weak constant motion
        motion[s:e] = np.maximum(motion[s:e], rng.uniform(0.01, 0.06))
    # talk
    for s, e in _segments(rng, T, 10 * busy + 1, (3, 120)):
        if not present[s]:
            continue
        on = (np.sin(np.arange(e - s) * DT * rng.uniform(0.5, 2.0)) + _ou(rng, e - s, 0.7, 0.8)) > 0
        speech[s:e] = np.maximum(speech[s:e], on * rng.uniform(0.5, 0.95))
        energy[s:e] = np.maximum(energy[s:e], on * rng.uniform(0.15, 0.45) + floor[s:e])
    # TV / music
    for s, e in _segments(rng, T, 0.25 * busy, (300, 3600)):
        lvl = rng.uniform(0.12, 0.45)
        energy[s:e] = np.maximum(energy[s:e], lvl * np.clip(1 + _ou(rng, e - s, 2.0, 0.3), 0.3, 1.6))
        if rng.random() < 0.6:
            speech[s:e] = np.maximum(speech[s:e], (rng.random(e - s) < 0.4) * rng.uniform(0.4, 0.8))
    # vacuum cleaner / drill / blender / shouting
    for s, e in _segments(rng, T, 0.12 * busy * rich, (20, 900)):
        energy[s:e] = np.maximum(energy[s:e], rng.uniform(0.5, 0.95) * np.clip(1 + _ou(rng, e - s, 3.0, 0.15), 0.5, 1.2))
    # false speech at night (snoring, a fan: 2026-10-21 showed 40 % "speech" at 03:00)
    if night.mean() > 0.3 and rng.random() < 0.5:
        fake = (rng.random(T) < rng.uniform(0.1, 0.4)) & night
        speech = np.maximum(speech, fake * rng.uniform(0.4, 0.6))
        energy = np.maximum(energy, fake * rng.uniform(0.05, 0.2))

    # ---- sharp sounds: claps, knocks, bangs (the microphone's onset events)
    sharp = np.zeros(T)
    for s, e in _segments(rng, T, 3 + 30 * busy, (0.5, 8)):
        n = rng.integers(1, 6) if rng.random() < 0.3 else 1
        for k in range(n):
            i = s + k * SEC(rng.uniform(0.3, 1.0))
            if i < T:
                sharp[i] = rng.uniform(0.5, 1.0)
    for s, _ in _segments(rng, T, (0.4 + 0.5 * busy) * rich, (1, 2)):  # a door slams, something falls
        sharp[s] = rng.uniform(1.1, 1.5)
        energy[s:s + SEC(0.5)] = np.maximum(energy[s:s + SEC(0.5)], 0.8)
    for s, e in _segments(rng, T, 0.1 * busy * rich, (5, 30)):  # a series of bangs (hammer, a game)
        for i in range(s, e, SEC(rng.uniform(0.4, 1.5))):
            sharp[i] = rng.uniform(0.8, 1.4)

    # ---- touches
    head_ev = np.zeros(T)
    head_hold = np.zeros(T)
    screen = np.zeros(T)
    for s, e in _segments(rng, T, 4 * busy + 0.5, (5, 120)):  # petting: strokes every 2-6 s, the hand rests a while
        if not present[s]:
            continue
        near[s:e] = np.maximum(near[s:e], rng.uniform(150, 500))
        i = s
        while i < e:
            head_ev[i] = rng.choice([0.7, 1.0, 1.0, 1.3])
            hold = SEC(rng.uniform(0.2, 2.5))
            head_hold[i:i + hold] = 1
            i += hold + SEC(rng.uniform(1.0, 5.0))
    for s, e in _segments(rng, T, 1.0 * busy * rich, (3, 30)):  # poking / drumming on the head
        if not present[s]:
            continue
        i = s
        gap = rng.uniform(0.25, 1.0)
        while i < e:
            head_ev[i] = 1.3
            i += SEC(gap * rng.uniform(0.7, 1.3))
    for s, e in _segments(rng, T, 0.3 * busy * rich, (5, 60)):  # a hand held on the head
        if present[s]:
            head_ev[s] = 1.0
            head_hold[s:e] = 1
    for s, e in _segments(rng, T, 3 * busy + 0.3, (1, 30)):  # taps on the face (screen)
        if not present[s]:
            continue
        n = 1 if rng.random() < 0.5 else rng.integers(2, 25)
        i = s
        for _ in range(n):
            if i >= T:
                break
            screen[i] = 1
            i += SEC(rng.uniform(0.2, 3.0))
    for s, _ in _segments(rng, T, 1.5 if night.mean() > 0.5 else 0.3, (1, 2)):  # phantom pad touches (servo, at night)
        head_ev[s] = 0.7
    # someone at the face (proximity)
    for s, e in _segments(rng, T, 3 * busy + 0.2, (2, 60)):
        if present[s]:
            prof = rng.uniform(150, 900) * np.clip(np.sin(np.linspace(0, np.pi, e - s)) * 1.5, 0, 1)
            near[s:e] = np.maximum(near[s:e], prof)

    # ---- handling: picked up, carried, shaken, laid on the side, dropped
    accel = np.abs(rng.normal(0, 0.008, T))
    imu_pitch = rng.normal(0, 0.003, T)
    imu_roll = 0.4 + rng.normal(0, 0.004, T) + 0.06 * np.tanh(_ou(rng, T, 30.0, 1.0))  # the head's own pitch
    for s, e in _segments(rng, T, 0.6 * busy * rich, (3, 60)):  # picked up and carried
        if not present[s]:
            continue
        n = e - s
        accel[s:e] += np.abs(rng.normal(0.08, 0.05, n))
        imu_pitch[s:e] += np.clip(_ou(rng, n, 2.0, 0.15), -0.4, 0.4)
        imu_roll[s:e] += np.clip(_ou(rng, n, 2.0, 0.08), -0.2, 0.2)
    for s, e in _segments(rng, T, 0.4 * busy * rich, (1, 6)):  # shaken
        if present[s]:
            accel[s:e] += rng.uniform(0.3, 1.3) * np.abs(np.sin(np.arange(e - s) * 2.0)) + 0.1
    for s, e in _segments(rng, T, 0.1 * busy * rich, (10, 900)):  # laid on its side / on its back
        if present[s]:
            imu_pitch[s:e] += rng.choice([-1, 1]) * rng.uniform(0.4, 1.0)
    for s, e in _segments(rng, T, (0.05 * busy + 0.01) * rich, (5, 120)):  # dropped: a jolt, then lying
        accel[s:s + SEC(0.3)] += rng.uniform(1.2, 2.0)
        imu_pitch[s + SEC(0.3):e] += rng.choice([-1, 1]) * rng.uniform(0.5, 1.0)
        sharp[s] = max(sharp[s], 1.3)
    # covered for the night: dark and something right at the face
    if night.mean() > 0.3 and rng.random() < 0.6:
        cov = night.copy()
        near = np.where(cov, np.maximum(near, rng.uniform(500, 1500)), near)
        r["als"] = np.where(cov, np.minimum(r["als"], rng.uniform(0, 3)), r["als"])

    # ---- camera frames (7 per second, some unusable), our own motion masks
    frame = (np.arange(T) % 3 == 0) & (rng.random(T) > 0.05)
    own = np.zeros(T, bool)
    for s, e in _segments(rng, T, 200, (0.5, 3)):  # the head turning (frames masked, mic events gone)
        own[s:e] = True
    frame &= ~own
    blind = r["als"] < 2
    frame &= ~(blind & (rng.random() < 0.9))
    sharp[own] = 0

    # ---- battery and servos
    b0 = rng.uniform(0.3, 1.0) if rng.random() < 0.3 else 1.0
    drain = rng.uniform(0, 0.3) / 3600 * DT if b0 < 1 else 0
    r["battery"] = np.clip(b0 - drain * np.arange(T), 0, 1)
    r["stamina"] = np.clip(1 - np.abs(_ou(rng, T, 600, 0.2)), 0, 1)

    r.update(energy=np.clip(energy, 0, 1), speech=np.clip(speech, 0, 1), sharp_ev=sharp, head_ev=head_ev,
             head_hold=head_hold, screen_ev=screen, accel_sd=accel, imu_pitch=np.clip(imu_pitch, -1, 1),
             imu_roll=np.clip(imu_roll, -1, 1), imu_ok=np.ones(T), near_raw=np.clip(near, 0, 2047),
             near_ok=np.ones(T), motion_frac=np.clip(motion, 0, 1), motion_x=np.clip(mx, -1, 1),
             motion_y=np.clip(my, -1, 1), frame_ok=frame.astype(float),
             light_x=lx, light_y=ly, light_str=np.where(r["als"] < 2, 0.0, lstr))
    return r


def make_channels(rng, T, **kw):
    """-> [T, 18] channels; the encoder starts warm (tilt baseline known, like a robot that has been up a while)."""
    warm = SEC(70)
    raw = make_raw(rng, T + warm, **kw)
    X = encode(raw)
    return X[warm:]


if __name__ == "__main__":
    import sys
    from spec_v4 import CHANNELS, ZONES, zones_of
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    X = np.concatenate([make_channels(rng, 12000) for _ in range(40)])
    print(f"{len(X) / 72000:.1f} h synthetic")
    for i, ch in enumerate(CHANNELS):
        zs = zones_of(ch, X[:, i])
        print(f"{ch:13s} p50 {np.median(X[:, i]):.2f} p99 {np.percentile(X[:, i], 99):.2f} | " +
              ", ".join(f"{z} {zs[z].mean() * 100:4.1f}%" for z in ZONES if z in zs))
