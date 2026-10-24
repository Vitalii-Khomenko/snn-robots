"""Platform B's diary layout (kind 1 datagrams: int16 x 4096 per value), the robot's life until brain v4 took over
(2026-10-11 .. 2026-10-24). Kept to train and calibrate on those real days.

  values, unix = load("2026-10-20", hours=[13])   # [T, len(COLUMNS)] float32 (NaN where an older layout had no column)
  raw, hours = raw_for_channels(values, unix)      # channels.encode's input, reconstructed after the fact

The layout grew in steps (KNOWN_LAYOUTS): the v3 policy input, action and efference copy, then the motivation, the
world model, the online learner, the perception embeddings, expect_talk and finally brain v4's shadow columns.
"""
import datetime
import struct

import numpy as np

from diary import HEADER, datagrams

SCALE = 4096.0
KIND_LEGACY = 1
POLICY_INPUT = ["motion_l", "motion_c", "motion_r", "motion_up", "motion_down", "person", "face", "loom", "recede",
                "brightness", "light_change", "target_x", "target_y", "target_conf", "activity", "light_x", "light_y",
                "light_strength", "silence", "speech", "clap", "sharp", "continuous", "low", "high", "tonal", "approach",
                "energy", "novelty", "self_voice", "own_motion", "onset", "loud", "yaw", "pitch", "yaw_vel", "pitch_vel",
                "real_yaw_vel", "real_pitch_vel", "imu_pitch", "imu_roll", "touch_head", "touch_screen", "shake", "tilt",
                "battery", "stamina", "light_delta", "speaker", "night", "day", "dawn", "dusk", "owner_near", "hour_sin",
                "hour_cos", "owner_arrived", "pe_motor", "pe_vision", "wander_x", "wander_y", "wander_slow", "jitter",
                "curiosity", "boredom", "arousal", "annoyance", "social", "playful", "fatigue", "sleep", "music", "alert",
                "att_x", "att_y", "att_strength", "nov_vis", "nov_aud", "nov_touch", "tr_touch", "tr_sound", "tr_person",
                "tr_motion", "tr_novelty", "tr_reward", "hab_touch", "hab_sound", "surprise", "prev_yaw", "prev_pitch",
                "prev_speed", "prev_sound", "prev_hold", "prev_explore", "prev_valence", "prev_arousal", "slow_boredom",
                "slow_social", "expect_reward", "expect_threat"]
ACTION = ["yaw", "pitch", "speed", "vigor", "gaze_x", "gaze_y", "sound_act", "sound_arousal", "sound_valence", "blink",
          "led_activity", "led_r", "led_g", "led_b", "hold", "explore", "voice_register", "voice_slope", "voice_expr",
          "voice_length", "voice_rate", "style_calm", "style_ask", "style_play", "style_grumble", "face_valence",
          "face_arousal", "lids", "invite", "startle", "emo_normal", "emo_happy", "emo_very_happy", "emo_laughing",
          "emo_sad", "emo_crying", "emo_sleepy", "emo_surprised", "emo_angry", "emo_annoyed", "emo_love", "emo_shy",
          "emo_wink", "emo_curious", "emo_confused", "emo_thinking", "emo_bored", "emo_scared", "emo_dizzy", "emo_cool",
          "emo_sick"]
EFF = ["eff_yaw", "eff_pitch", "eff_sound"]
MOTIVATION = ["need_energy", "need_social", "need_stimulation", "need_rest", "need_comfort", "reward", "novelty",
              "move_deg100"]
WORLD = ["wm_rms_vision", "wm_rms_audio", "wm_rms_body", "wm_lp10"]
ONLINE = ["online_delta", "online_value"]
EMB = [f"vemb{i}" for i in range(64)] + [f"aemb{i}" for i in range(32)]
V4_CHANNELS = ["loud", "sharp", "voice", "touch_head", "touch_screen", "shake", "tilt", "near", "light", "motion",
               "hunger", "tired", "lonely", "motion_x", "motion_y"]
V4 = ["ch_" + c for c in V4_CHANNELS] + [f"v4snn{i}" for i in range(32)] + [f"v4nn{i}" for i in range(61)]
COLUMNS = POLICY_INPUT + ["act_" + a for a in ACTION] + EFF + MOTIVATION + WORLD + ONLINE + EMB + ["expect_talk"] + V4
_n = len(COLUMNS)
_m = _n - len(V4)
KNOWN_LAYOUTS = (_m - 1 - len(EMB) - len(ONLINE) - len(WORLD) - len(MOTIVATION),
                 _m - 1 - len(EMB) - len(ONLINE) - len(WORLD), _m - 1 - len(EMB) - len(ONLINE), _m - 1 - len(EMB),
                 _m - 1, _m, _n)


def col(name):
    return COLUMNS.index(name)


def load(day, hours=None, root=None):
    """-> values [T, len(COLUMNS)], unix [T] (s, NaN before the clock synced); (None, None) without legacy ticks."""
    values, unix, keys = [], [], []
    for t, kind, n, nv, b, d in datagrams(day, hours, root):
        if kind != KIND_LEGACY:
            continue
        if nv not in KNOWN_LAYOUTS:
            raise ValueError(f"{day}: {nv} values per tick, this reader knows {KNOWN_LAYOUTS}")
        (u0,) = struct.unpack_from("<q", d, HEADER.size)
        rec = np.dtype([("seq", "<u4"), ("ms", "<u4"), ("flags", "<u2"), ("v", "<i2", (nv,))])
        r = np.frombuffer(d, rec, count=n, offset=HEADER.size + 8)
        vals = r["v"].astype(np.float32) / SCALE
        if nv < _n:
            vals = np.concatenate([vals, np.full((n, _n - nv), np.nan, np.float32)], axis=1)
        values.append(vals)
        unix.append(np.where(u0 > 0, u0 / 1000.0 + (r["ms"] - r["ms"][0]) / 1000.0, np.nan))
        keys.append(np.stack([np.full(n, t), r["seq"].astype(np.float64)], 1))
    if not values:
        return None, None
    k = np.concatenate(keys)
    order = np.lexsort((k[:, 1], k[:, 0]))
    return np.concatenate(values)[order], np.concatenate(unix)[order]


def raw_for_channels(v, unix):
    """The ChannelEncoder's inputs reconstructed from the v3 bus: touch, sound, voice, tilt, hunger, tiredness and
    loneliness are close to what the firmware computes; shake is the old shake event, motion the Vision NN's activity;
    proximity and light were not logged."""
    c = lambda n: np.nan_to_num(v[:, col(n)])
    T = len(v)

    def edges(x, thr=0.5):
        on = x > thr
        e = np.zeros(T, bool)
        e[1:] = on[1:] & ~on[:-1]
        return e

    # the microphone's onset detector (and its loud events): 2-4 an hour in a quiet room, where the Auditory NN's
    # clap/sharp classes fire 20-40 times an hour on nothing (2026-10-21)
    onset_ev = edges(c("onset"))
    loud_ev = edges(c("loud"), 0.6)
    # (loud events: intensity 0.6 .. 1.0 -> sharp strength 0.5 .. 1.5, as BrainLoop::hear)
    sharp_ev = np.where(loud_ev, 0.5 + (c("loud") - 0.6) / 0.4, np.where(onset_ev, np.maximum(c("onset"), 0.5), 0.0))
    hours = np.array([np.nan if not np.isfinite(u) else
                      (lambda d: d.hour + d.minute / 60 + d.second / 3600)(datetime.datetime.fromtimestamp(u))
                      for u in unix[::200]])
    hours = np.repeat(hours, 200)[:T]
    moving = (np.abs(c("yaw_vel")) + np.abs(c("pitch_vel"))) > 0.01
    return dict(energy=c("energy"), speech=c("speech"), sharp_ev=sharp_ev,
                head_ev=edges(c("touch_head")).astype(np.float32), screen_ev=edges(c("touch_screen")).astype(np.float32),
                accel_sd=np.where(c("shake") > 0.5, 0.4, 0.0), imu_pitch=c("imu_pitch"), imu_roll=c("imu_roll"),
                head_pitch_deg=c("pitch") * 35.0, imu_ok=np.ones(T), motion_frac=c("activity") * 0.3,
                frame_ok=(~moving).astype(np.float32), battery=c("battery"), stamina=c("stamina"),
                hour=np.nan_to_num(hours, nan=12.0), clock_ok=np.isfinite(hours).astype(np.float32)), hours
