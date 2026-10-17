#!/usr/bin/env python3
"""Draws the hybrid brain as an SVG:   python3 tools/hybrid_map.py   ->  docs/BRAIN_MAP.svg
Numbers come from tools/hybrid/spec.py so the picture follows the code."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "hybrid"))
import spec  # noqa: E402

W, H = 2400, 1600
C = dict(bg="#0f1420", panel="#171e2e", line="#3a4660", text="#e6ebf5", dim="#9aa6bf",
         nn="#2f6fdd", snn="#d9534f", det="#3a9d6e", old="#b06bd1", out="#d9a441", bus="#26a6a6")
o = []


def add(s):
    o.append(s)


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, t, size=15, color=None, weight="normal", anchor="start"):
    add(f'<text x="{x}" y="{y}" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="{size}" '
        f'fill="{color or C["text"]}" font-weight="{weight}" text-anchor="{anchor}">{esc(t)}</text>')


def box(x, y, w, h, color, title, lines=(), tsize=17, lsize=13, fill=None):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill or C["panel"]}" stroke="{color}" stroke-width="2.5"/>')
    add(f'<rect x="{x}" y="{y}" width="{w}" height="30" rx="10" fill="{color}"/>')
    add(f'<rect x="{x}" y="{y + 18}" width="{w}" height="12" fill="{color}"/>')
    text(x + 12, y + 21, title, tsize, "#ffffff", "bold")
    for i, l in enumerate(lines):
        text(x + 12, y + 52 + i * (lsize + 5), l, lsize, C["text"] if not l.startswith("·") else C["dim"])


def arrow(x1, y1, x2, y2, color=None, label=None, dash=False, lx=None, ly=None, width=2.2):
    col = color or C["dim"]
    add(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{col}" stroke-width="{width}" fill="none" '
        f'{"stroke-dasharray=\"7 5\"" if dash else ""} marker-end="url(#a{col[1:]})"/>')
    if label:
        text(lx if lx is not None else (x1 + x2) / 2 + 6, ly if ly is not None else (y1 + y2) / 2 - 5, label, 12, col)


def poly(points, color=None, dash=False, label=None, lx=0, ly=0, width=2.2):
    col = color or C["dim"]
    d = "M" + " L".join(f"{x},{y}" for x, y in points)
    add(f'<path d="{d}" stroke="{col}" stroke-width="{width}" fill="none" '
        f'{"stroke-dasharray=\"7 5\"" if dash else ""} marker-end="url(#a{col[1:]})"/>')
    if label:
        text(lx, ly, label, 12, col)


add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
add("<defs>")
for k in set(C.values()):
    if k.startswith("#"):
        add(f'<marker id="a{k[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{k}"/></marker>')
add("</defs>")
add(f'<rect width="{W}" height="{H}" fill="{C["bg"]}"/>')
text(40, 52, "StackChan, platform B: a robot that learns from its own life (mode life)", 32, weight="bold")
text(40, 82, "M5Stack CoreS3 / ESP32-S3, 20 Hz brain tick (50 ms)  -  fast loop on the robot, slow loop (the night sleep) on the home PC  -  safety stays deterministic C++", 16, C["dim"])

# legend
lx = 1500
for i, (k, name) in enumerate([("nn", "trained network (INT8)"), ("snn", "spiking network"), ("det", "deterministic C++"),
                                ("old", "droid voice / PC"), ("out", "actuators / outputs"), ("bus", "FeatureBus")]):
    add(f'<rect x="{lx + (i % 3) * 290}" y="{30 + (i // 3) * 30}" width="22" height="16" rx="3" fill="{C[k]}"/>')
    text(lx + (i % 3) * 290 + 32, 43 + (i // 3) * 30, name, 14, C["dim"])
text(lx, 100, "dashed = feedback / efference   |   numbers: params, MACs, measured time on the robot", 13, C["dim"])

# ---------------------------------------------------------------- sensors
y0 = 130
box(40, y0, 250, 150, C["det"], "Camera GC0308", ["160x120 RGB565, ~7 fps", "exposure locked", "task: core 1, prio 2", "(below the microphone task)"])
box(40, y0 + 175, 250, 150, C["det"], "Microphones (2x ES7210)", ["24 kHz stereo, 128-sample blocks", "mic task: core 1, prio 3", "own-speaker suppression", "loud / onset events"])
box(40, y0 + 350, 250, 130, C["det"], "IMU BMI270", ["accelerometer: shake, tilt", "gyro: REAL head velocity", "(vestibular, self-calibrated)"])
box(40, y0 + 505, 250, 110, C["det"], "Touch + screen", ["Si12T head pad (2-3 ms pulse)", "LCD tap", "ignored during own motion"])
box(40, y0 + 640, 250, 130, C["det"], "Clock / Wi-Fi / power", ["NTP hour, owner phone nearby", "battery level, light sensor", "stamina (servo travel budget)"])

# ---------------------------------------------------------------- front ends
fx = 350
box(fx, y0, 330, 150, C["det"], "Vision front end", ["64x48 sample -> 32x24 luminance", "global image shift search (ego-motion)", "3 channels: lum, compensated diff, raw diff", "exact: brightness, light change, bright spot"], lsize=13)
box(fx, y0 + 175, 330, 150, C["det"], "Audio front end (mic task)", ["16 log-spaced band-pass biquads", "120 Hz .. 7.5 kHz, frame every 21 ms", "per-band quiet baseline (min tracker)", "level, L/R balance, onsets"])
box(fx, y0 + 350, 330, 265, C["det"], "Body / context encoders (BodySense)",
    ["head yaw, pitch, velocities (model + gyro)", "touch 150 ms; shake held 2 s; tilt 1.5 s", "battery, stamina, light delta, speaker",
     "time of day: night, day, dawn, dusk windows", "owner near / arrived", "motor prediction error (gyro vs model)",
     "efference copy: commanded vs actual", "own-motion flag, self-voice flag"])

# ---------------------------------------------------------------- perception nets
px = 740
box(px, y0, 360, 150, C["nn"], "Vision NN (CNN)",
    ["3x24x32 -> conv8 -> conv16 -> conv24 (fused pool)", "+ dense 64 -> 13 outputs; 24 k params, 0.57 M MAC", "sim + COCO + this camera; NIGHTLY: its own frames", "labelled by YOLOv8n + YuNet on the PC; ~26 ms"], lsize=13)
box(px, y0 + 175, 360, 150, C["nn"], "Auditory NN (MLP)",
    ["16 bands x 6 frames + level/flags = 100", "-> dense 64 -> 32 -> 8 outputs; 9 k params", "sim + ESC-50 + LibriSpeech + this microphone;", "NIGHTLY: its own sound, labelled by", "Silero VAD + PANNs CNN14 on the PC"], lsize=13)
text(px + 12, y0 + 328 + 14, "classes: silence speech clap sharp cont. low high tonal", 12, C["dim"])
text(px + 12, y0 + 328 + 30, "hidden layers 64 + 32 -> the diary (bus v4)", 12, C["dim"])
box(px, y0 + 560, 360, 150, C["snn"], "Kept from v3 (mode brain, the way back)", ["Spiking core: 768 LIF neurons, 25 drives", "Policy GRU: 100 -> 51 actions, imitation", "In mode life both are off: the SNN state", "section of the bus is zero (CPU 52 -> 36 %)"], lsize=13, fill="#2a1717")
arrow(fx + 330, y0 + 60, px, y0 + 60, C["det"])
arrow(fx + 330, y0 + 235, px, y0 + 235, C["det"])
arrow(290, y0 + 75, fx, y0 + 75, C["det"])
arrow(290, y0 + 250, fx, y0 + 250, C["det"])
arrow(290, y0 + 415, fx, y0 + 450, C["det"])
arrow(290, y0 + 560, fx, y0 + 520, C["det"])
arrow(290, y0 + 700, fx, y0 + 560, C["det"])

# ---------------------------------------------------------------- bus
bx, bw = 1150, 330
box(bx, y0, bw, 770, C["bus"], "FeatureBus", [], fill="#10262b")
sections = [("Vision", spec.VISION), ("Audio", spec.AUDIO), ("Body", spec.BODY), ("Ctx", spec.CTX), ("Prediction error", spec.PE),
            ("Efference copy", spec.EFF), ("Noise pacemakers", spec.NOISE), ("SNN state (0 in mode life)", spec.SNN), ("Previous action", spec.PREV),
            ("Slow needs", spec.SLOW), ("Learned (on the robot)", spec.LEARN)]
yy = y0 + 46
for name, names in sections:
    text(bx + 12, yy, f"{name} ({len(names)})", 14, C["bus"], "bold")
    yy += 17
    line, lines = "", []
    for n in names:
        if len(line) + len(n) > 40:
            lines.append(line)
            line = ""
        line += n + ", "
    lines.append(line.rstrip(", "))
    for l in lines:
        text(bx + 18, yy, l, 11, C["dim"])
        yy += 14
    yy += 9
text(bx + 12, y0 + 760, f"SNN input {spec.__dict__['SNN_INPUT'].__len__()}   policy input {len(spec.POLICY_INPUT)}   action {len(spec.ACTION)}", 13, C["text"], "bold")
arrow(px + 360, y0 + 70, bx, y0 + 70, C["bus"])
text(px + 366, y0 + 100, "13 + 5 exact", 12, C["bus"])
arrow(px + 360, y0 + 245, bx, y0 + 245, C["bus"])
text(px + 366, y0 + 275, "8 + 7 exact", 12, C["bus"])
arrow(fx + 330, y0 + 470, bx, y0 + 470, C["det"], "body, ctx, PE, efference", lx=fx + 345, ly=y0 + 462)

# ---------------------------------------------------------------- the learning brain (mode life)
sx = 1560
box(sx, y0, 560, 150, C["nn"], "World model GRU (predicts the next tick)",
    ["154 inputs (policy input, action, efference) -> 64 -> GRU 64", "-> 48 -> 49 predicted changes of vision, hearing, body",
     "RMS error per group + learning progress -> diary", "retrained every night from the whole diary (gate: skill)"], lsize=12.5)
arrow(bx + bw, y0 + 60, sx, y0 + 60, C["bus"], "154", lx=bx + bw + 14, ly=y0 + 52)
box(sx, y0 + 170, 560, 170, C["det"], "Motivation: 5 needs + reward (the temperament)",
    ["energy, social, stimulation, rest, comfort (0..1, grow by themselves)",
     "reward = sum w (D(t-1) - D(t)) - 0.0002 / deg,  D = d^2  (Keramati-Gutkin)",
     "CURIOSITY: world-model error well above its habitual level (60 s)", "eases the stimulation need; own voice / motion are not new",
     "needs kept in NVS over a reboot"], lsize=12.5)
arrow(sx + 280, y0 + 150, sx + 280, y0 + 170, C["nn"])
text(sx + 290, y0 + 165, "surprise", 12, C["nn"])
box(sx, y0 + 360, 560, 90, C["snn"], "Associative memory (learned on the robot)",
    ["cue cells -> petting / rough handling / being talked to", "three-factor plastic synapses, NVS -> 3 expectations"], lsize=12.5, fill="#2a1717")
box(sx, y0 + 470, 560, 200, C["nn"], "Actor GRU + online learner (decides)",
    ["state 71 (senses, needs, world errors, expectations) -> 64 -> GRU 64 -> 48 -> 24",
     "move? where (Gaussian yaw / pitch), speed, call?, face (12), voice style (4), value",
     "night: advantage-weighted regression on the diary (critic on the PC)",
     "day: plastic dW 8 x 48 on its hidden layer, TD(lambda) actor-critic,",
     "  delta = r + gamma V(t) - V(t-1): the dopamine; by day only; |dW| <= 1",
     "gate: 2-8 moves and 0.3-8 calls a minute"], lsize=12.5)
arrow(bx + bw, y0 + 560, sx, y0 + 560, C["bus"], "state 71", lx=bx + bw + 14, ly=y0 + 552)
arrow(sx + 140, y0 + 340, sx + 140, y0 + 470, C["det"])
text(sx + 150, y0 + 462, "reward", 12, C["det"])
arrow(sx + 420, y0 + 450, sx + 420, y0 + 470, C["snn"])
box(sx, y0 + 690, 560, 130, C["det"], "Collector: samples the decisions, keeps the rules",
    ["every touch answered (a rule: masked in the actor's training)", "quiet night 21:30-06:30, exploration floor",
     "faces, voice colour, LEDs; motor babbling when no actor"], lsize=12.5)
arrow(sx + 280, y0 + 670, sx + 280, y0 + 690, C["nn"])

# ---------------------------------------------------------------- executor + outputs
ey = y0 + 845
box(sx, ey, 560, 190, C["det"], "BodyExecutor + safety (deterministic C++)",
    ["head: joint windows, steps <= 30 deg, speed by stamina, 10 servo packets/s", "voice <= half of the last minute (mode life)",
     "face of 21 (debounced), gaze, blink; LEDs: breathing + heartbeat",
     "servo stall (gyro) -> rail power cycle, back-off 1-60 min; panel recover", "touch storm filter, stuck pad re-setup, task WDT, battery",
     "OTA: A/B firmware with rollback; weights blob, NaN -> restart"], lsize=12.5)
arrow(sx + 280, y0 + 820, sx + 280, ey, C["det"], "normalized actions", lx=sx + 290, ly=y0 + 838)
ox = 2160
box(ox, ey, 215, 190, C["out"], "Actuators", ["head yaw / pitch servos", "LCD face + eyes", "12 RGB LEDs", "speaker 24 kHz"], lsize=13)
arrow(sx + 560, ey + 95, ox, ey + 95, C["out"])
vy = ey + 210
box(sx, vy, 560, 110, C["old"], "Droid voice: BabbleNet GRU + analogue synth (own task)",
    ["a phrase note by note: 10 kinds of droid sounds, glide, ending, pitch,", "FM, harmonics; the shown face picks the character; no speech model"], lsize=12.5)
arrow(sx + 560, vy + 55, ox + 107, ey + 190, C["out"])
poly([(ox + 107, ey + 190), (ox + 107, vy + 140), (320, vy + 140), (320, y0 + 790), (290, y0 + 790)], C["dim"], True,
     "body feedback: real head velocity (gyro), servo state, battery, own voice", 330, vy + 130)

# ---------------------------------------------------------------- the PC: the slow loop
py0 = vy + 170
add(f'<rect x="40" y="{py0}" width="{W - 80}" height="{H - py0 - 30}" rx="10" fill="{C["panel"]}" stroke="{C["old"]}" stroke-width="2"/>')
text(60, py0 + 32, "The home PC: the slow loop (nothing leaves the home network)", 20, weight="bold")
steps = [("Diary", ["UDP, every 50 ms tick:", "265 values + log lines", "lifelogd -> hourly files", "~1.7 M ticks a day"]),
         ("Media takes", ["every ~3 min: a frame", "+ 2 s of sound (mediad)", "GET /snapshot, /audio", "~480 takes a day"]),
         ("Teachers", ["YOLOv8n (persons, objects)", "YuNet (faces)", "Silero VAD (speech)", "PANNs CNN14 (527 sounds)"]),
         ("Night sleep 03:30", ["world model, critic, actor", "from the whole diary;", "perception fine-tuned on", "takes + 75 % replay"]),
         ("Gates", ["world: no group worse", "actor: sane rates", "perception: no forgetting,", "closer to the teachers"]),
         ("Weights blob -> robot", ["SCW1, CRC32, over Wi-Fi", "spare flash slot, switched", "between two ticks, no reboot", "1 min trial, NaN -> back"])]
bwid, gap = 330, 52
for i, (t, ls) in enumerate(steps):
    x = 60 + i * (bwid + gap)
    box(x, py0 + 52, bwid, 140, C["old"], t, ls, lsize=13)
    if i:
        arrow(x - gap, py0 + 122, x, py0 + 122, C["old"])
poly([(bx + 165, y0 + 770), (bx + 165, py0 - 10), (225, py0 - 10), (225, py0 + 52)], C["bus"], True, "the diary (every tick)", bx - 150, py0 - 18)
poly([(2140, py0 + 52), (2140, y0 + 75), (sx + 560, y0 + 75)], C["old"], True)
poly([(2140, y0 + 570), (sx + 560, y0 + 570)], C["old"], True)
text(2148, y0 + 110, "nightly", 12, C["old"])
text(2148, y0 + 126, "weights", 12, C["old"])
add("</svg>")
out = os.path.join(ROOT, "docs", "BRAIN_MAP.svg")
open(out, "w").write("\n".join(o))
print("wrote", out)
