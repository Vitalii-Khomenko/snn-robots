#!/usr/bin/env python3
"""Draws the hybrid brain as an SVG:   python3 tools/hybrid_map.py   ->  docs/BRAIN_MAP.svg
Numbers come from tools/hybrid/spec.py so the picture follows the code."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "hybrid"))
import spec  # noqa: E402

W, H = 2400, 1730
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
text(40, 52, "StackChan hybrid brain: perception, internal state, learned policy, safe body", 32, weight="bold")
text(40, 82, "M5Stack CoreS3 / ESP32-S3  -  one closed sensorimotor loop, 20 Hz brain tick (50 ms)  -  every neural module has a safe fallback", 16, C["dim"])

# legend
lx = 1500
for i, (k, name) in enumerate([("nn", "trained network (INT8)"), ("snn", "spiking network"), ("det", "deterministic C++"),
                                ("old", "droid voice"), ("out", "actuators / outputs"), ("bus", "FeatureBus")]):
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
    ["3x24x32 -> conv8 -> conv16 -> conv24 (fused pool)", "+ dense 64 -> 13 outputs; 24 k params, 0.57 M MAC", "sim + COCO + 13 real sessions of THIS camera", "person > 0.30 believed; hold 1.2 s; ~26 ms"], lsize=13)
box(px, y0 + 175, 360, 150, C["nn"], "Auditory NN (MLP)",
    ["16 bands x 6 frames + level/flags = 100", "-> dense 64 -> 32 -> 9 outputs; 9 k params", "trained: simulator + ESC-50 + LibriSpeech", "+ 28 real sessions of THIS robot's microphone", "(claps recorded with a parked head; bang = clap|knock)"], lsize=13)
text(px + 12, y0 + 328 + 14, "classes: silence speech clap sharp cont. low high tonal dir.", 12, C["dim"])
text(px + 12, y0 + 328 + 30, "log: hear: clap|sharp|speech (rising edges)", 12, C["dim"])
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
            ("Efference copy", spec.EFF), ("Noise pacemakers", spec.NOISE), ("SNN state", spec.SNN), ("Previous action", spec.PREV),
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

# ---------------------------------------------------------------- SNN
sx = 1560
box(sx, y0, 560, 540, C["snn"], "Spiking core - 768 LIF neurons (internal state)", [], fill="#2a1717")
pops = [("Attention map 12x8", 96), ("Novelty", 64), ("Habituation", 64), ("Curiosity", 64), ("Arousal", 64), ("Boredom", 48),
        ("Annoyance", 48), ("Social need", 48), ("Playfulness", 48), ("Fatigue / sleep", 48), ("Music drive", 32),
        ("Sensory traces (6)", 80), ("Prediction / surprise", 64)]
for i, (n, k) in enumerate(pops):
    col, row = i % 2, i // 2
    x = sx + 14 + col * 272
    y = y0 + 44 + row * 38
    add(f'<rect x="{x}" y="{y}" width="260" height="30" rx="6" fill="#3a1f1f" stroke="{C["snn"]}" stroke-width="1.2"/>')
    text(x + 8, y + 20, f"{n}", 13)
    text(x + 252, y + 20, str(k), 13, C["dim"], anchor="end")
for i, l in enumerate(["LIF neurons, 2 substeps x 25 ms per 50 ms tick, noise current in every",
                       "neuron: the core is never silent. INT8 weights: dense input projection",
                       "(258 x 768) and a sparse recurrent matrix (47 k synapses, event driven).",
                       "The slow drives (seconds to minutes) live in leaky readout cells; 25",
                       "readouts form the SNN state of the bus. Trained with surrogate gradients",
                       "on synthetic days and rollouts; teacher = designed drive dynamics.",
                       "Robot: ~7 ms per tick (input 4.6 ms, neurons + synapses 4.7 ms)."]):
    text(sx + 14, y0 + 338 + i * 20, l, 12.5, C["dim"])
arrow(bx + bw, y0 + 120, sx, y0 + 120, C["bus"])
text(bx + bw + 14, y0 + 112, "SNN input 66", 12, C["bus"])
add(f'<rect x="{sx + 14}" y="{y0 + 468}" width="532" height="40" rx="6" fill="#3a1f1f" stroke="{C["snn"]}" stroke-width="1.6" stroke-dasharray="5,3"/>')
text(sx + 22, y0 + 484, "Associative memory - LEARNED ON THE ROBOT: 11 cue x 2 outcome plastic synapses,", 12.5, C["snn"])
text(sx + 22, y0 + 501, "three-factor rule -> expect petting / rough handling (2), kept in NVS", 12.5, C["snn"])
text(sx + 14, y0 + 528, "annoyance clamp (C++): anger needs touch / shake / loud evidence", 12.5, C["det"])
arrow(sx, y0 + 200, bx + bw, y0 + 200, C["snn"])
text(bx + bw + 14, y0 + 222, "SNN state (25)", 12, C["snn"])

# ---------------------------------------------------------------- policy
ploy = y0 + 580
box(sx, ploy, 560, 200, C["nn"], "Policy GRU (action selection)",
    ["100 inputs -> dense 64 -> GRU 64 -> dense 48 -> 51 outputs; ~37 k INT8 params",
     "trained by imitation of a designed teacher, then DAgger rollouts in the digital twin",
     "robot: ~3.5 ms/tick;  invalid output -> rejected / clamped, neutral action",
     "outputs: head, gaze, blink, WHICH FACE (21), face valence, sound intent",
     "(pitch, glide, vibrato, length, syllables, 4 families), sing, invite, startle"], lsize=12.5)
arrow(bx + bw, ploy + 60, sx, ploy + 60, C["bus"], "policy input 100", lx=bx + bw + 6, ly=ploy + 52)
poly([(sx + 100, ploy + 200), (sx + 100, ploy + 240), (bx + 160, ploy + 240), (bx + 160, y0 + 770)], C["nn"], True, "previous action", bx + 30, ploy + 262)

# ---------------------------------------------------------------- executor + outputs
ey = y0 + 840
box(sx, ey, 560, 230, C["det"], "BodyExecutor + safety (deterministic C++)",
    ["head: yaw +-100, pitch -20..35 deg, no jump over 70 deg,", "<= 8 servo packets/s, speed 12..110 deg/s scaled by stamina",
     "primitives: singing sway, nod series, startle flinch, call to every bang, turns in a talk", "voice: intent + face -> BabbleVoice, <= 6 calls/min (reserve for touch / bang)", "song governor: windows only, <= 45 s, rest 4 min, hush by touch",
     "face: the policy's choice of 21 (debounced), gaze, blink, person symbol", "LEDs: breathing + heartbeat, brightness / wave = led_activity",
     "watchdogs: servo stall (gyro) -> rail power cycle, task WDT, battery"], lsize=12.5)
arrow(sx + 280, ploy + 200 + 8, sx + 280, ey, C["nn"], "51 normalized actions", lx=sx + 290, ly=ploy + 222)

ox = 2160
box(ox, y0 + 840, 215, 215, C["out"], "Actuators", ["head yaw / pitch servos", "LCD face + eyes", "12 RGB LEDs", "speaker 24 kHz"], lsize=13)
arrow(sx + 560, ey + 105, ox, y0 + 950, C["out"])

# voice chain
vy = y0 + 1090
box(sx, vy, 560, 150, C["old"], "Droid voice: BabbleNet + analogue droid synth (own audio task)",
    ["BabbleNet GRU 48 (31 k INT8) + the face's emotion: a phrase, note by note -", "1 of 10 kinds (whistle chirp pip burble trill warble giggle squeal",
     "growl coo), glide, ending flick, pitch, FM, harmonics, rate, stop.", "Synth: analogue-style droid (sine + band-limited harmonics + FM), 24 kHz,",
     "no speech model. Audio task on core 1 prio 5: never starved."], lsize=12.5)
arrow(sx + 280, ey + 230, sx + 280, vy, C["old"], "voice intent (12)", lx=sx + 290, ly=ey + 246)
arrow(sx + 560, vy + 65, ox + 107, y0 + 1005, C["out"])

# ---------------------------------------------------------------- feedback loops
poly([(ox + 107, y0 + 1005), (ox + 107, vy + 190), (320, vy + 190), (320, y0 + 790), (290, y0 + 790)], C["dim"], True,
     "body feedback: real head velocity (gyro), servo state, battery, own voice", 330, vy + 180)
add(f'<rect x="40" y="{vy + 215}" width="{W - 80}" height="{H - vy - 255}" rx="10" fill="{C["panel"]}" stroke="{C["line"]}" stroke-width="2"/>')
text(60, vy + 245, "What changed relative to v2 (NeuroBrain, 6780 LIF neurons, one recurrent network)", 18, weight="bold")
rows = [
    ("v2", "one recurrent spiking net did vision, hearing, drives, motor and voice (6780 cells, ~210 k synapses, 10-12 ms/tick)"),
    ("v3", "specialised modules around one FeatureBus: Vision NN, Auditory NN, 768-neuron SNN, GRU policy (also picks the face), droid voice"),
    ("why", "each part is small, testable, replaceable and retrainable on its own; the policy can later be refined by reward without touching perception"),
    ("data", "perception: simulator ground truth + COCO photos + ESC-50 / LibriSpeech + real recordings from this robot's microphone (parked head for claps) and camera; voice: teacher recipes per face"),
    ("loop", "perception -> SNN state -> policy -> executor -> body -> feedback (gyro, efference, prediction error) -> perception"),
    ("next", "vision: no false persons in an empty room (threshold, new empty frames, retrain), claps while the head moves, SNN on real rollouts, reward refinement"),
]
for i, (k, v) in enumerate(rows):
    text(60, vy + 278 + i * 24, k, 14, C["snn"], "bold")
    text(130, vy + 278 + i * 24, v, 14, C["dim"])
add("</svg>")
out = os.path.join(ROOT, "docs", "BRAIN_MAP.svg")
open(out, "w").write("\n".join(o))
print("wrote", out)
