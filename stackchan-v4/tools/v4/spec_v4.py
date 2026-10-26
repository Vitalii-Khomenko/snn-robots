"""Single source of truth for brain v4 (BRAIN_V4_PLAN.txt, BRAIN_V4_INPUT_SCALES.txt).

`./py spec_v4.py` regenerates main/v4/v4_spec.h. The 13 sense channels with their innate zones, the 2 "where"
inputs, the 32 numbers the spiking analyzer reports and the main network's inputs and outputs; the order here is
the wire order between the firmware, the diary, the simulator and the training code.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# ---- the senses: 13 channels 0..1 (main/v4/channel_encoder.cpp computes them, tools/v4/channels.py mirrors it)
CHANNELS = ["loud", "sharp", "voice", "touch_head", "touch_screen", "shake", "tilt", "near", "light", "motion",
            "hunger", "tired", "lonely"]
EXTERNAL = CHANNELS[:10]          # the world and the body's handling (salience + hedonic each)
INTERNAL = CHANNELS[10:]          # the body's needs
WHERE = ["motion_x", "motion_y",  # where the movement in the picture is (-1..1), no zones
         "light_x", "light_y", "light_spot"]  # the bright spot in the picture (-1..1) and how new it is (0..1, 2026-10-25)
SNN_IN = CHANNELS + WHERE

ZONES = ["boredom", "comfort", "interest", "anxiety", "anger"]
# Innate zone borders per channel (BRAIN_V4_INPUT_SCALES.txt, summary table). None = the channel has no such zone.
# Borders are soft (EDGE wide on each side), neighbouring zones share their border region. Experience may later
# move a border by at most SHIFT_MAX (plan section 4, stage 6).
ZONE_TABLE = {
    "loud":         {"boredom": (0.00, 0.15), "comfort": (0.15, 0.45), "interest": (0.45, 0.70),
                     "anxiety": (0.70, 0.85), "anger": (0.85, 1.00)},
    "sharp":        {"interest": (0.10, 0.35), "anxiety": (0.35, 0.65), "anger": (0.65, 1.00)},
    "voice":        {"boredom": (0.00, 0.05), "comfort": (0.05, 0.60), "interest": (0.60, 0.85),
                     "anxiety": (0.85, 1.00)},
    "touch_head":   {"boredom": (0.00, 0.05), "comfort": (0.05, 0.45), "interest": (0.45, 0.70),
                     "anxiety": (0.70, 0.85), "anger": (0.85, 1.00)},
    "touch_screen": {"interest": (0.05, 0.25), "comfort": (0.25, 0.40), "anxiety": (0.40, 0.65),
                     "anger": (0.65, 1.00)},
    "shake":        {"interest": (0.05, 0.20), "anxiety": (0.20, 1.00), "anger": (0.45, 1.00)},
    "tilt":         {"comfort": (0.00, 0.12), "interest": (0.12, 0.30), "anxiety": (0.30, 1.00)},
    "near":         {"boredom": (0.00, 0.10), "comfort": (0.10, 0.50), "interest": (0.50, 0.80),
                     "anxiety": (0.80, 1.00)},
    "light":        {"comfort": (0.10, 0.60), "interest": (0.60, 0.85), "anger": (0.85, 1.00)},
    "motion":       {"boredom": (0.00, 0.03), "interest": (0.03, 0.25), "anxiety": (0.25, 0.55),
                     "anger": (0.55, 1.00)},
    "hunger":       {"comfort": (0.00, 0.50), "anxiety": (0.50, 1.00)},
    "tired":        {"comfort": (0.00, 0.60)},
    "lonely":       {"comfort": (0.00, 0.25), "boredom": (0.25, 1.00)},
}
EDGE = 0.03
SHIFT_MAX = 0.15
# How each zone feels (the analyzer teacher's valence weights) and how much it excites.
ZONE_VALENCE = {"boredom": -0.25, "comfort": 0.6, "interest": 0.45, "anxiety": -0.55, "anger": -0.9}
ZONE_AROUSAL = {"boredom": 0.05, "comfort": 0.2, "interest": 0.55, "anxiety": 0.75, "anger": 0.95}


def membership(v, lo, hi, edge=EDGE):
    """Soft box [lo, hi]: 1 inside, linear ramps 2*edge wide centred on the borders (0 and 1 are open ends)."""
    v = np.asarray(v, np.float32)
    up = np.ones_like(v) if lo <= 0.0 else np.clip((v - lo) / (2 * edge) + 0.5, 0.0, 1.0)
    down = np.ones_like(v) if hi >= 1.0 else np.clip((hi - v) / (2 * edge) + 0.5, 0.0, 1.0)
    return up * down


def zones_of(channel, v, shift=None):
    """-> dict zone -> membership array for one channel's values (shift: per-zone border offsets, stage 6)."""
    out = {}
    for z, (lo, hi) in ZONE_TABLE[channel].items():
        d = 0.0 if shift is None else shift.get(z, 0.0)
        out[z] = membership(v, lo + d, hi + d)
    return out


# ---- what the spiking analyzer reports: 32 numbers (plan section 5)
SNN_GLOBAL = ["valence", "arousal", "under", "over", "threat", "surprise"]
SNN_SALIENCE = [f"sal_{c}" for c in EXTERNAL]
SNN_HEDONIC = [f"hed_{c}" for c in EXTERNAL]
SNN_NEEDS = ["hunger_n", "sleepiness", "loneliness"]
SNN_ATT = ["att_x", "att_y", "att_strength"]
SNN_OUT = SNN_GLOBAL + SNN_SALIENCE + SNN_HEDONIC + SNN_NEEDS + SNN_ATT
SNN_SIGNED = {"att_x", "att_y"}
assert len(SNN_OUT) == 32

# ---- the main network (plan section 6)
EMOTIONS = ["normal", "happy", "very_happy", "laughing", "sad", "crying", "sleepy", "surprised", "angry", "annoyed",
            "love", "shy", "wink", "curious", "confused", "thinking", "bored", "scared", "dizzy", "cool", "sick"]
# what the network did a moment ago, fed back (BrainV4 fills them)
PREV = ["prev_valence", "prev_arousal", "emo_age", "since_sound", "since_gesture", "head_yaw", "head_pitch",
        "led_level", "touch_answered", "pace_fast", "pace_slow", "pace_jitter",
        # the Muse's ideas (tools/v4/muse.py, docs/CREATIVITY.md; 2026-10-26) - the pace_* come from it too now
        "idea_play", "idea_show", "idea_still", "idea_social", "idea_surprise"]
NN_IN = SNN_OUT + PREV
SOUND_STYLES = ["calm", "ask", "play", "grumble"]
LED_COLORS = ["white", "yellow", "orange", "red", "pink", "violet", "blue", "cyan"]
LED_RGB = {"white": (255, 240, 220), "yellow": (255, 200, 40), "orange": (255, 110, 20), "red": (255, 20, 10),
           "pink": (255, 70, 150), "violet": (150, 40, 255), "blue": (30, 60, 255), "cyan": (20, 200, 230)}
LED_PATTERNS = ["steady", "breathe", "pulse", "flicker", "wave"]
HEAD_MODES = ["hold", "look", "scan", "nod", "shake", "flinch", "look_down", "look_up", "doze"]
NN_OUT = (["emo_" + e for e in EMOTIONS] + ["emo_strength"] +
          ["say"] + ["style_" + s for s in SOUND_STYLES] + ["snd_arousal", "snd_valence", "snd_register",
                                                             "snd_length", "snd_rate"] +
          ["led_" + c for c in LED_COLORS] + ["pat_" + p for p in LED_PATTERNS] + ["led_bright", "led_rate"] +
          ["head_" + m for m in HEAD_MODES] + ["gaze_yaw", "gaze_pitch", "gest_amp", "gest_speed", "gest_reps"])
# output groups: softmax groups (one choice) and plain values
NN_SOFTMAX = {"emo": ["emo_" + e for e in EMOTIONS], "style": ["style_" + s for s in SOUND_STYLES],
              "color": ["led_" + c for c in LED_COLORS], "pattern": ["pat_" + p for p in LED_PATTERNS],
              "head": ["head_" + m for m in HEAD_MODES]}
NN_SIGNED = {"gaze_yaw", "gaze_pitch", "snd_valence"}

# The diary (main/net/life_log.h -> tools/v4/diary.py): every brain tick as float16 - the raw sense readings the
# channels are made from (BrainLoop -> ChannelEncoder::Raw), the channels (after the learned zone shift: what the
# analyzer saw), the analyzer's numbers, the main network's outputs, what the body did, the network's own state.
RAW = ["energy", "speech", "sharp_ev", "head_ev", "head_hold", "screen_ev", "accel_sd", "imu_pitch", "imu_roll",
       "imu_ok", "near_raw", "near_ok", "als", "als_ok", "motion_frac", "motion_x", "motion_y",
       "frame_ok", "battery", "stamina", "hour", "clock_ok",
       "light_x", "light_y", "light_str",  # the camera's brightest spot (2026-10-25; older diaries: NaN)
       "head_yaw"]  # where the head points, degrees: places the spot in the room
BODY = ["head_yaw_deg", "head_pitch_deg", "face", "said", "gesture"]
# + the main network's own-state inputs (PREV) of the tick it last decided: with the SNN numbers they are its whole
# input, so the PC can replay every decision exactly (stage 6, sleep_v4.py)
DIARY = (["raw_" + r for r in RAW] + ["ch_" + c for c in SNN_IN] + ["snn_" + o for o in SNN_OUT] +
         ["nn_" + o for o in NN_OUT] + BODY + ["pv_" + p for p in PREV])

GROUPS = [("Raw", "RAW", RAW), ("Body", "BODY", BODY), ("Channel", "CH", CHANNELS), ("Where", "WH", WHERE), ("SnnOut", "SO", SNN_OUT), ("Prev", "PV", PREV),
          ("NnOut", "NO", NN_OUT), ("Zone", "ZONE", ZONES)]


def idx(names, n):
    return names.index(n)


def header():
    out = ["// GENERATED by tools/v4/spec_v4.py - do not edit.", "#pragma once", "#include <cstdint>", "",
           "namespace v4 {"]
    for title, tag, names in GROUPS:
        out.append(f"enum {title} : uint8_t {{")
        for n in names:
            out.append(f"    {tag}_{n.upper()},")
        out.append(f"    {tag}_COUNT")
        out.append("};")
        out.append(f"inline constexpr const char* k{title}Names[] = {{" + ", ".join(f'"{n}"' for n in names) + "};")
    out.append("")
    # zone borders: [channel][zone] = {lo, hi}, lo > hi marks a missing zone
    out.append("struct ZoneBorder { float lo, hi; };")
    out.append(f"inline constexpr float kZoneEdge = {EDGE}f;")
    out.append(f"inline constexpr float kZoneShiftMax = {SHIFT_MAX}f;")
    out.append("inline constexpr ZoneBorder kZones[CH_COUNT][ZONE_COUNT] = {")
    for c in CHANNELS:
        row = []
        for z in ZONES:
            lo, hi = ZONE_TABLE[c].get(z, (1.0, 0.0))
            row.append(f"{{{lo:.2f}f, {hi:.2f}f}}")
        out.append(f"    {{{', '.join(row)}}},  // {c}")
    out.append("};")
    for name, names in NN_SOFTMAX.items():
        out.append(f"inline constexpr int kNo{name.capitalize()}At = NO_{names[0].upper()};")
        out.append(f"inline constexpr int kNo{name.capitalize()}Count = {len(names)};")
    for i, m in enumerate(HEAD_MODES):
        out.append(f"inline constexpr int kHead{''.join(w.capitalize() for w in m.split('_'))} = {i};")
    out.append("inline constexpr uint8_t kLedRgb[][3] = {" +
               ", ".join("{%d, %d, %d}" % LED_RGB[c] for c in LED_COLORS) + "};")
    out.append(f"inline constexpr int kSnnInputs = {len(SNN_IN)};")
    out.append(f"inline constexpr int kNnInputs = {len(NN_IN)};")
    out.append(f"inline constexpr int kDiaryValues = {len(DIARY)};  // RAW, channels, SO, NO, BODY, PV (tools/v4/diary.py)")
    out.append("}  // namespace v4")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    path = os.path.join(ROOT, "main", "v4", "v4_spec.h")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(header())
    print(f"wrote {path}: {len(CHANNELS)} channels, {len(SNN_OUT)} SNN outputs, NN {len(NN_IN)} -> {len(NN_OUT)}")
