"""The 13 sense channels: Python twin of main/v4/channel_encoder.cpp (keep the two identical).

encode(raw) turns per-tick raw sensor arrays (20 Hz) into the channels 0..1 and the 2 "where" values, in the
order of spec_v4.SNN_IN. The same code serves the diary calibration (stage 0, from the diary's bus columns), the
synthetic generator (synth_v4.py paints raw signals, this encodes them) and the parity test with the firmware.

Raw inputs (dict of float32 arrays [T], missing ones default to 0):
  energy      room loudness over the room's own quiet floor, 0..1, 0 while we speak (AuditoryNet AUD_ENERGY)
  speech      speech probability 0..1 (Auditory NN, gated by level, 0 while we speak)
  sharp_ev    a sharp sound started this tick: its strength (the microphone's onset or loud event, 0.5..1.5), else 0
  head_ev     a head touch started this tick: weight by pad level (0.7 / 1.0 / 1.3), else 0
  head_hold   1 while a finger stays on the head pad
  screen_ev   a screen touch started this tick (1), else 0
  accel_sd    standard deviation of |acceleration| over the last 0.5 s, g (0 while our own servos move)
  imu_pitch, imu_roll   gravity direction, -1..1 (BrainLoop: pitch /(pi/2), roll /pi)
  imu_ok      1 when the IMU values are valid
  near_raw    proximity reading over its slow baseline, sensor counts (near_ok = 1 when valid)
  als         ambient light, raw counts of channel 0 (als_ok = 1 when valid)
  motion_frac share of the picture that moved by itself (head motion searched out), 0..1
  motion_x, motion_y    centre of that movement, -1..1
  frame_ok    1 in a tick with a fresh, usable camera frame (not blind, not during our own motion); 0.5: a fresh frame
              good for the bright spot only (the head turning slowly, 2026-10-26)
  light_x, light_y, light_str   the picture's brightest spot (-1..1) and its excess over the frame (0..1)
  battery     0..1, stamina 0..1 (the servos' tiredness), hour 0..24, clock_ok 1 when the clock is synced
"""
import numpy as np
from numba import njit

from spec_v4 import CHANNELS, RAW, SNN_IN

DT = 0.05
# ---- constants (mirrored in main/v4/channel_encoder.h)
LOUD_TAU, LOUD_GAIN = 10.0, 1.6
SHARP_TAU, SHARP_FULL = 15.0, 6.0  # 3.0 until 2026-10-25 (twice as tolerable)
VOICE_TAU = 60.0
TOUCH_TAU, HEAD_K, SCREEN_K, HOLD_RATE = 30.0, 18.6, 9.5, 0.5
SHAKE_FLOOR, SHAKE_FULL, SHAKE_DECAY = 0.03, 0.8, 3.0
TILT_FIT_TAU, TILT_TAU, TILT_WARM, TILT_ROLL_BAND, TILT_ACCEPT_S = 1800.0, 0.5, 1200.0, 0.18, 600.0
NEAR_FULL, NEAR_TAU = 600.0, 1.0
LIGHT_FULL, LIGHT_TAU = 20000.0, 2.0
MOTION_GAIN, MOTION_TAU, MOTION_FALL_TAU, MOTION_HOLD_DECAY = 3.0, 0.5, 4.0, 2.0  # 1.5, 5 s mean until 2026-10-25
SPOT_SLOW_TAU, SPOT_NOV_GAIN, SPOT_HAB_TAU, SPOT_HAB_RATE, SPOT_HAB_CUT = 4.0, 2.0, 180.0, 0.001, 0.7
SPOT_CAM_HALF_DEG, SPOT_PLACE_DEG, SPOT_FORGET_TAU, SPOT_PLACES = 30.0, 15.0, 900.0, 13  # novelty per room direction
SPOT_FALL_TAU, SPOT_MARGIN = 120.0, 0.15  # a place keeps its usual peak
LONELY_TAU, LONELY_MOTION_CAP, LONELY_START = 2400.0, 300.0, 600.0
CONTACT_SPEECH, CONTACT_NEAR, CONTACT_MOTION = 0.5, 0.3, 0.1



def night_ramp(h):
    """0 by day, rises 21:00 -> 21:45, 1 through the night, falls 06:00 -> 06:45 (the owner's sleep)."""
    if 21.0 <= h < 21.75:
        return (h - 21.0) / 0.75
    if h >= 21.75 or h < 6.0:
        return 1.0
    if 6.0 <= h < 6.75:
        return 1.0 - (h - 6.0) / 0.75
    return 0.0


_night_ramp = njit(cache=True)(night_ramp)


@njit(cache=True)
def _clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


@njit(cache=True)
def _encode(R, out, state):
    """R [T, len(RAW)], out [T, 18]. state [STATE_N] carries the filters between calls (zeros = fresh boot)."""
    a_loud = 1.0 - np.exp(-DT / LOUD_TAU)
    a_voice = 1.0 - np.exp(-DT / VOICE_TAU)
    k_sharp = np.exp(-DT / SHARP_TAU)
    k_touch = np.exp(-DT / TOUCH_TAU)
    k_shake = np.exp(-DT / SHAKE_DECAY)
    a_fit = 1.0 - np.exp(-DT / TILT_FIT_TAU)
    a_tilt = 1.0 - np.exp(-DT / TILT_TAU)
    a_near = 1.0 - np.exp(-DT / NEAR_TAU)
    a_light = 1.0 - np.exp(-DT / LIGHT_TAU)
    a_motion = 1.0 - np.exp(-DT / MOTION_TAU)
    a_mfall = 1.0 - np.exp(-DT / MOTION_FALL_TAU)
    a_spot = 1.0 - np.exp(-DT / SPOT_SLOW_TAU)
    k_spot_hab = np.exp(-DT / SPOT_HAB_TAU)
    k_forget = np.exp(-DT / SPOT_FORGET_TAU)
    a_spot_fall = 1.0 - np.exp(-DT / SPOT_FALL_TAU)
    k_mhold = np.exp(-DT / MOTION_HOLD_DECAY)
    loud, sharp, voice, head, screen, shake = state[0], state[1], state[2], state[3], state[4], state[5]
    # tilt: what upright reads (p0, r0), learned while upright and still
    n, p0, r0, steady, tilt = state[6], state[7], state[8], state[9], state[10]
    near, light, motion = state[11], state[12], state[13]
    lonely_t = state[14]
    mx, my = state[15], state[16]
    lx, ly, spot, spot_hab = state[17], state[18], state[19], state[20]
    place = state[21:21 + SPOT_PLACES]  # a view: updated in place
    for t in range(R.shape[0]):
        energy, speech, sharp_ev = R[t, 0], R[t, 1], R[t, 2]
        head_ev, head_hold, screen_ev = R[t, 3], R[t, 4], R[t, 5]
        accel_sd, ip, ir, imu_ok = R[t, 6], R[t, 7], R[t, 8], R[t, 9]
        near_raw, near_ok, als, als_ok = R[t, 10], R[t, 11], R[t, 12], R[t, 13]
        mfrac, mxr, myr, frame_ok = R[t, 14], R[t, 15], R[t, 16], R[t, 17]
        battery, stamina, hour, clock_ok = R[t, 18], R[t, 19], R[t, 20], R[t, 21]
        lxr, lyr, lstr, head_yaw = R[t, 22], R[t, 23], R[t, 24], R[t, 25]

        loud += (energy - loud) * a_loud
        sharp = sharp * k_sharp + (0.0 if sharp_ev <= 0 else 1.0 + 5.0 * _clamp((sharp_ev - 1.0) / 0.5))
        voice += ((1.0 if speech > 0.4 else 0.0) - voice) * a_voice
        head = head * k_touch + head_ev + head_hold * HOLD_RATE * DT
        screen = screen * k_touch + screen_ev
        shake = max(_clamp((accel_sd - SHAKE_FLOOR) / SHAKE_FULL), shake * k_shake)

        tilt_now = 0.0
        if imu_ok > 0:
            # Sideways the IMU reads clean (0.5 deg noise when upright). Forward / back it turns with the pitch servo,
            # and the commanded pitch does not tell where the head really is (the servos often stop short: roll
            # residual 13 deg, 2026-10-19..21), so that axis only counts beyond the band the head itself can make.
            dev = max(abs(ip - p0), 2.0 * max(0.0, abs(ir - r0) - TILT_ROLL_BAND))
            tilt_now = _clamp(dev) if n >= TILT_WARM else 0.0
            steady = steady + DT if (tilt_now > 0.11 and shake < 0.05 and abs(tilt_now - tilt) < 0.03) else 0.0
            # upright and still (or the first minute after boot, or the same lean for 10 minutes: the new normal)
            if n < TILT_WARM or (tilt_now < 0.11 and shake < 0.05) or steady > TILT_ACCEPT_S:
                n = min(n + 1.0, 1e9)
                a = max(a_fit, 1.0 / n) if steady <= TILT_ACCEPT_S else 0.01
                p0 += (ip - p0) * a
                r0 += (ir - r0) * a
        tilt += (tilt_now - tilt) * a_tilt

        if near_ok > 0:
            near += (_clamp(near_raw / NEAR_FULL) - near) * a_near
        if als_ok > 0:
            light += (_clamp(np.log10(1.0 + max(als, 0.0)) / np.log10(1.0 + LIGHT_FULL)) - light) * a_light
        if frame_ok > 0.75:
            target = _clamp(mfrac * MOTION_GAIN)
            motion += (target - motion) * (a_motion if target > motion else a_mfall)
            if mfrac > 0.01:
                mx, my = mxr, myr
        else:
            motion *= 1.0 - (1.0 - k_mhold) * 0.25  # no usable frame: fade slowly, do not jump to 0
        if frame_ok > 0.25:
            s = _clamp(lstr)
            nov = 0.0
            if s > 0:
                lx, ly = lxr, lyr
                p = _clamp((head_yaw + lxr * SPOT_CAM_HALF_DEG + 90.0) / SPOT_PLACE_DEG, 0.0, SPOT_PLACES - 1.0)
                i = min(int(p), SPOT_PLACES - 2)
                w = p - i
                known = place[i] * (1.0 - w) + place[i + 1] * w
                nov = max(0.0, s - known - SPOT_MARGIN)
                # both by the same step: the read value learns at a_spot (up) / a_spot_fall (down)
                step = (s - known) * (a_spot if s > known else a_spot_fall)
                place[i] += step
                place[i + 1] += step
            spot = _clamp(SPOT_NOV_GAIN * nov) * (1.0 - SPOT_HAB_CUT * spot_hab)
        else:
            spot *= 1.0 - (1.0 - k_mhold) * 0.25
        spot_hab = _clamp(spot_hab * k_spot_hab + spot * SPOT_HAB_RATE)
        for k in range(SPOT_PLACES):
            place[k] *= k_forget
        if motion < 0.02:
            mx *= 0.98
            my *= 0.98
        if spot < 0.02:
            lx *= 0.98
            ly *= 0.98

        hunger = _clamp(1.0 - battery)
        tired = _clamp(0.75 * (_night_ramp(hour) if clock_ok > 0 else 0.0) + 0.25 * (1.0 - stamina))
        lonely_t += DT
        if head_ev > 0 or screen_ev > 0 or head_hold > 0 or speech > CONTACT_SPEECH or near > CONTACT_NEAR:
            lonely_t = 0.0
        elif motion > CONTACT_MOTION and lonely_t > LONELY_MOTION_CAP:
            lonely_t = LONELY_MOTION_CAP
        lonely = 1.0 - np.exp(-lonely_t / LONELY_TAU)

        out[t, 0] = _clamp(loud * LOUD_GAIN)
        out[t, 1] = _clamp(sharp / SHARP_FULL)
        out[t, 2] = _clamp(voice)
        out[t, 3] = 1.0 - np.exp(-head / HEAD_K)
        out[t, 4] = 1.0 - np.exp(-screen / SCREEN_K)
        out[t, 5] = shake
        out[t, 6] = _clamp(tilt)
        out[t, 7] = near
        out[t, 8] = light
        out[t, 9] = motion
        out[t, 10] = hunger
        out[t, 11] = tired
        out[t, 12] = lonely
        out[t, 13] = mx
        out[t, 14] = my
        out[t, 15] = lx
        out[t, 16] = ly
        out[t, 17] = spot
    state[0], state[1], state[2], state[3], state[4], state[5] = loud, sharp, voice, head, screen, shake
    state[6], state[7], state[8], state[9], state[10] = n, p0, r0, steady, tilt
    state[11], state[12], state[13], state[14], state[15], state[16] = near, light, motion, lonely_t, mx, my
    state[17], state[18], state[19], state[20] = lx, ly, spot, spot_hab


STATE_N = 21 + SPOT_PLACES


def new_state():
    s = np.zeros(STATE_N, np.float64)
    s[14] = LONELY_START
    return s


def encode(raw, state=None):
    """raw: dict name -> [T] array. -> [T, len(SNN_IN)] float32 (13 channels + the "where" values)."""
    T = len(next(iter(raw.values())))
    R = np.zeros((T, len(RAW)), np.float64)
    for i, n in enumerate(RAW):
        if n in raw:
            R[:, i] = np.nan_to_num(np.asarray(raw[n], np.float64))
    if "battery" not in raw:
        R[:, RAW.index("battery")] = 1.0
    if "stamina" not in raw:
        R[:, RAW.index("stamina")] = 1.0
    out = np.zeros((T, len(SNN_IN)), np.float64)
    _encode(R, out, new_state() if state is None else state)
    assert out.shape[1] == len(CHANNELS) + 5
    return out.astype(np.float32)
