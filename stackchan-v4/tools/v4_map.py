#!/usr/bin/env python3
"""Draws brain v4 as two SVGs:   tools/v4/py ../v4_map.py   ->  docs/BRAIN_V4_MAP.svg, docs/NETWORKS_V4.svg
Channels, zones, outputs and layer sizes come from tools/v4/spec_v4.py and snn_v4.py, so the pictures follow the code."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "v4"))
import spec_v4 as S  # noqa: E402

C = dict(bg="#0f1420", panel="#171e2e", line="#3a4660", text="#e6ebf5", dim="#9aa6bf",
         nn="#2f6fdd", snn="#d9534f", det="#3a9d6e", pc="#b06bd1", out="#d9a441", bus="#26a6a6")
ZC = dict(boredom="#6c7a96", comfort="#3a9d6e", interest="#d9a441", anxiety="#e8743b", anger="#d9534f")
FONT = "Inter,Segoe UI,Arial,sans-serif"


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.o = w, h, []
        self.add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">')
        self.add("<defs>")
        for k in set(list(C.values()) + list(ZC.values())):
            self.add(f'<marker id="a{k[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                     f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{k}"/></marker>')
        self.add("</defs>")
        self.add(f'<rect width="{w}" height="{h}" fill="{C["bg"]}"/>')

    def add(self, s):
        self.o.append(s)

    def text(self, x, y, t, size=15, color=None, weight="normal", anchor="start"):
        t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.add(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{color or C["text"]}" '
                 f'font-weight="{weight}" text-anchor="{anchor}">{t}</text>')

    def rect(self, x, y, w, h, fill, stroke=None, rx=4, sw=1.5):
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"{st}/>')

    def box(self, x, y, w, h, color, title, lines=(), tsize=17, lsize=13, fill=None):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill or C["panel"]}" '
                 f'stroke="{color}" stroke-width="2.5"/>')
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="30" rx="10" fill="{color}"/>')
        self.add(f'<rect x="{x}" y="{y + 18}" width="{w}" height="12" fill="{color}"/>')
        self.text(x + 12, y + 21, title, tsize, "#ffffff", "bold")
        for i, ln in enumerate(lines):
            self.text(x + 12, y + 52 + i * (lsize + 5), ln, lsize, C["dim"] if ln.startswith("·") else C["text"])

    def arrow(self, pts, color=None, dash=False, label=None, lx=None, ly=None, width=2.2):
        col = color or C["dim"]
        d = "M" + " L".join(f"{x},{y}" for x, y in pts)
        da = ' stroke-dasharray="7 5"' if dash else ""
        self.add(f'<path d="{d}" stroke="{col}" stroke-width="{width}" fill="none"{da} marker-end="url(#a{col[1:]})"/>')
        if label:
            (x1, y1), (x2, y2) = pts[0], pts[-1]
            self.text(lx if lx is not None else (x1 + x2) / 2 + 6, ly if ly is not None else (y1 + y2) / 2 - 6,
                      label, 12, col)

    def legend(self, x, y, items, cols=3, step=270):
        for i, (col, name) in enumerate(items):
            xx, yy = x + (i % cols) * step, y + (i // cols) * 28
            self.rect(xx, yy, 22, 16, col, rx=3)
            self.text(xx + 32, yy + 13, name, 14, C["dim"])

    def save(self, name):
        self.add("</svg>")
        path = os.path.join(ROOT, "docs", name)
        with open(path, "w") as f:
            f.write("\n".join(self.o) + "\n")
        print("wrote", path)


def zone_bar(s, x, y, w, h, ch):
    """The channel's 0..1 scale with its innate zones; a zone overlapping an earlier one goes into the lower half."""
    s.rect(x, y, w, h, "#0b0f18", C["line"], rx=3, sw=1)
    drawn = []
    for z, (a, b) in S.ZONE_TABLE[ch].items():
        lower = any(a < bb and b > aa for aa, bb in drawn)
        drawn.append((a, b))
        yy, hh = (y + h / 2, h / 2) if lower else (y, h if not lower else h / 2)
        s.rect(x + a * w, yy, (b - a) * w, hh, ZC[z], rx=2)


NOTES = {
    "loud": "loudness, slow average (10 s)", "sharp": "loud peaks: claps, bangs",
    "voice": "share of speech (60 s)", "touch_head": "strokes on the head pad",
    "touch_screen": "taps / pokes on the face", "shake": "accelerometer jitter",
    "tilt": "lean from the learned upright", "near": "IR proximity (LTR-553)",
    "light": "ambient light, log scale", "motion": "camera: share of moving cells",
    "hunger": "1 - battery", "tired": "night ramp + low stamina", "lonely": "time without contact (40 min)",
}

N_ENC, N_ZONE, N_MIX, N_ATT = len(S.CHANNELS) * 10, len(S.CHANNELS) * len(S.ZONES) * 4, 282, 96
N_SNN = N_ENC + N_ZONE + N_MIX + N_ATT
SNN_KB = (N_ENC + N_ZONE) * N_MIX + N_MIX * N_MIX + N_MIX * N_ATT + N_SNN * len(S.SNN_OUT)
NIN, H = len(S.SNN_OUT) + len(S.PREV), 64
N_EMO = len(S.EMOTIONS)
N_ACT = len(S.NN_OUT) - N_EMO - 1
NN_PARAMS = 3 * (NIN * H + H * H + 2 * H) + H * (N_EMO + 1) + (N_EMO + 1) + (H + N_EMO) * 64 + 64 + 64 * N_ACT + N_ACT
G = {g: len(v) for g, v in S.NN_SOFTMAX.items()}
VOICE = 1 + G["style"] + sum(1 for n in S.NN_OUT if n.startswith("snd_"))
LEDS = G["color"] + G["pattern"] + 2
HEAD = G["head"] + sum(1 for n in S.NN_OUT if n.startswith("gaze_") or n.startswith("gest_"))
assert VOICE + LEDS + HEAD == N_ACT, (VOICE, LEDS, HEAD, N_ACT)


# ====================================================================== the whole robot
def brain_map():
    s = Svg(2400, 1490)
    s.text(40, 52, "StackChan, brain v4: a spiking stimulus analyzer and one decision network", 32, weight="bold")
    s.text(40, 82, "M5Stack CoreS3 / ESP32-S3  -  20 Hz tick (50 ms): channels and SNN every tick, the main network every "
                   "2nd tick  -  everything runs on the robot; the home PC records and trains", 16, C["dim"])
    s.legend(1560, 26, [(C["snn"], "spiking network (int8)"), (C["nn"], "trained network (float)"),
                        (C["det"], "deterministic C++"), (C["bus"], "numbers passed on"), (C["out"], "body / actuators"),
                        (C["pc"], "home PC")])
    s.text(1560, 100, "dashed = feedback", 13, C["dim"])

    # ---------------------------------------------------------------- sensors
    sens = [("Microphones (2x ES7210)", ["24 kHz, loudness, onsets,", "speech detector"], ["loud", "sharp", "voice"]),
            ("Head touch pad", ["3 pads, strokes and holds"], ["touch_head"]),
            ("Touchscreen", ["taps on the face"], ["touch_screen"]),
            ("IMU BMI270", ["accelerometer jitter and lean", "(masked while servos re-power)"], ["shake", "tilt"]),
            ("LTR-553", ["IR proximity + ambient light"], ["near", "light"]),
            ("Camera GC0308", ["160x120, motion grid", "+ where it moves (x, y)"], ["motion"]),
            ("Battery, clock, timers", ["AXP2101, NTP time,", "time since contact"], ["hunger", "tired", "lonely"])]
    y, ys = 130, {}
    for title, lines, chans in sens:
        hh = 52 + 18 * len(lines)
        s.box(40, y, 290, hh, C["det"], title, lines, tsize=15)
        for c in chans:
            ys[c] = y + hh / 2
        y += hh + 16

    # ---------------------------------------------------------------- channel encoder
    ex, ey, ew = 390, 130, 580
    rowh = 54
    eh = 70 + rowh * len(S.CHANNELS) + 120
    s.box(ex, ey, ew, eh, C["det"], f"Channel encoder: {len(S.CHANNELS)} channels, each a number 0..1")
    s.text(ex + 150, ey + 52, "0", 11, C["dim"])
    s.text(ex + 400, ey + 52, "1", 11, C["dim"], anchor="end")
    for i, ch in enumerate(S.CHANNELS):
        ry = ey + 62 + i * rowh
        s.text(ex + 14, ry + 13, f"{i + 1:2d}  {ch}", 14, weight="bold")
        zone_bar(s, ex + 150, ry, 250, 16, ch)
        s.text(ex + 412, ry + 13, NOTES[ch], 11, C["dim"])
        s.arrow([(330, ys[ch]), (360, ys[ch]), (360, ry + 8), (ex - 2, ry + 8)], C["det"], width=1.4)
    ly = ey + 70 + rowh * len(S.CHANNELS)
    for i, z in enumerate(S.ZONES):
        s.rect(ex + 14 + i * 105, ly, 18, 12, ZC[z], rx=2)
        s.text(ex + 38 + i * 105, ly + 11, z, 12, C["dim"])
    s.text(ex + 14, ly + 36, "+ where: motion_x, motion_y (-1..1) for attention", 13)
    s.text(ex + 14, ly + 56, f"soft zone borders (±{S.EDGE}); innate table, may shift ±{S.SHIFT_MAX} with experience", 12, C["dim"])
    s.text(ex + 14, ly + 74, "slow clocks (minutes, hours) stay in code: a spiking net cannot count them", 12, C["dim"])
    s.text(ex + 14, ly + 92, "on the robot and in the trainer: the same code (parity test, error 0)", 12, C["dim"])

    # ---------------------------------------------------------------- SNN
    sx, sy, sw = 1010, 130, 390
    sh = eh
    s.box(sx, sy, sw, sh, C["snn"], f"Spiking analyzer (SNN): {N_SNN} LIF neurons")
    layers = [(f"ENC {N_ENC}", ["8 Gaussian receptive fields per channel", "+ ON / OFF change cells"]),
              (f"ZONE {N_ZONE}", [f"{len(S.CHANNELS)} channels x {len(S.ZONES)} zones x 4 neurons", "fixed innate wiring,",
                                  "habituation 1 / 8 / 30 / 90 s"]),
              (f"MIX {N_MIX}", ["recurrent, 80 % excitatory / 20 % inhib.", "context across channels"]),
              (f"ATT {N_ATT}", ["12 x 8 attention map: where to look"]),
              (f"READOUT {len(S.SNN_OUT)}", ["leaky integrators, 0.15 .. 3 s"])]
    yy = sy + 44
    prev = None
    for name, lines in layers:
        hh = 34 + 17 * len(lines)
        s.rect(sx + 16, yy, sw - 32, hh, "#22192a", C["snn"], rx=8)
        s.text(sx + 28, yy + 22, name, 15, "#ffb3ad", "bold")
        for k, ln in enumerate(lines):
            s.text(sx + 28, yy + 42 + k * 17, ln, 12)
        if prev is not None:
            s.arrow([(sx + sw / 2, prev), (sx + sw / 2, yy - 2)], C["snn"], width=1.8)
        prev = yy + hh
        yy += hh + 18
    for k, ln in enumerate(["2 substeps of 25 ms per tick, event-driven C++",
                            f"int8 weights per row: {SNN_KB / 1024:.0f} KB (in PSRAM)",
                            "3-5 ms per tick on the ESP32-S3",
                            "· trained: surrogate gradients, TBPTT, int8 QAT",
                            "· matches its teacher at 0.96-0.99 (correlation)"]):
        s.text(sx + 16, yy + 6 + k * 19, ln, 13, C["dim"] if ln.startswith("·") else C["text"])
    s.arrow([(ex + ew, ey + eh / 2), (sx - 2, sy + sh / 2)], C["bus"], width=3)
    s.text(ex + ew + 20, ey + eh / 2 - 26, "15", 13, C["bus"], "bold", "middle")
    s.text(ex + ew + 20, ey + eh / 2 - 10, "nums", 11, C["bus"], anchor="middle")

    # ---------------------------------------------------------------- 32 numbers
    bx, by, bw = 1440, 130, 250
    groups = [("core affect 6", ["valence, arousal, under-", "and over-stimulation,", "threat, surprise"]),
              ("salience 10", ["how much each outer", "channel matters now"]),
              ("hedonic 10", ["pleasant / unpleasant", "per outer channel"]),
              ("drives 3", ["hunger, sleepiness,", "loneliness"]),
              ("attention 3", ["x, y, strength"])]
    bh = 44 + sum(26 + 16 * len(g[1]) for g in groups)
    s.box(bx, by, bw, bh, C["bus"], f"{len(S.SNN_OUT)} numbers to the NN")
    yy = by + 52
    for name, lines in groups:
        s.text(bx + 14, yy, name, 14, "#8fe3e3", "bold")
        for k, ln in enumerate(lines):
            s.text(bx + 14, yy + 18 + k * 16, ln, 12)
        yy += 26 + 16 * len(lines)
    s.arrow([(sx + sw, sy + 200), (bx - 2, sy + 200)], C["bus"], width=3)

    # ---------------------------------------------------------------- main network
    nx, ny, nw = 1730, 130, 330
    nh = eh
    s.box(nx, ny, nw, nh, C["nn"], "Main network (the one decider)")
    s.text(nx + 14, ny + 56, f"input {NIN} = {len(S.SNN_OUT)} from the SNN + {len(S.PREV)} own state", 13)
    s.text(nx + 14, ny + 74, "runs at 10 Hz, float, " + f"{NN_PARAMS / 1000:.0f} k parameters", 13, C["dim"])
    parts = [(f"GRU {H}", ["memory of the last seconds"]),
             (f"Emotion head: {N_EMO} + strength", ["picks the face / mood first"]),
             (f"Action heads (hidden 64): {N_ACT}", [f"voice {VOICE}: say, style, mood, register,", "         length, rate",
                                                     f"LEDs {LEDS}: colour, pattern, brightness, rate",
                                                     f"head {HEAD}: mode, gaze, gesture size,", "         speed, repeats"])]
    yy = ny + 96
    prev = None
    for name, lines in parts:
        hh = 34 + 17 * len(lines)
        s.rect(nx + 16, yy, nw - 32, hh, "#16213a", C["nn"], rx=8)
        s.text(nx + 28, yy + 22, name, 15, "#a9c6ff", "bold")
        for k, ln in enumerate(lines):
            s.text(nx + 28, yy + 42 + k * 17, ln, 12)
        if prev is not None:
            s.arrow([(nx + nw / 2, prev), (nx + nw / 2, yy - 2)], C["nn"], width=1.8)
        prev = yy + hh
        yy += hh + 18
    s.text(nx + 14, yy + 6, f"own state ({len(S.PREV)}):", 13, weight="bold")
    for k, ln in enumerate(["last mood, how long the face is shown,", "time since a sound / a gesture, head pose,",
                            "LED level, was the touch answered,", "3 pacemakers (slow, fast, jitter): the",
                            "randomness the network can see",
                            "· trained: imitation of a teacher with", "· memory, then DAgger; agrees 0.90"]):
        s.text(nx + 14, yy + 26 + k * 18, ln, 12, C["dim"] if ln.startswith("·") else C["text"])
    s.arrow([(bx + bw, by + bh / 2), (nx - 2, by + bh / 2)], C["bus"], width=3)

    # ---------------------------------------------------------------- executor + body
    xx, xw = 2090, 280
    s.box(xx, 130, xw, 310, C["det"], "Executor (safety in C++)",
          [f"takes the {len(S.NN_OUT)} outputs every 100 ms", "face shown >= 1.5 s, confirmed 2 ticks", "voice budget: 5 tokens, 1 per 10 s;",
           "  reactions may spend the last one", "night 21:45-06:00: silent,", "  LEDs <= 6 %, no nods / shakes",
           "nod / shake at most every 4 s", "servo bus <= 8 commands / s", "yaw ±100°, pitch -20..35°",
           "· the network decides WHAT,", "· the executor only HOW"], tsize=15)
    s.arrow([(nx + nw, ny + 150), (xx - 2, ny + 150)], C["nn"], width=3)
    body = [("Display: the face", [f"{N_EMO} emotions, smooth morph,", "eyes follow attention"]),
            ("Speaker: droid babble", ["child-like phrases, style and", "mood from the network"]),
            ("LED ring", [f"{len(S.LED_COLORS)} colours x {len(S.LED_PATTERNS)} patterns", "steady breathe pulse flicker wave"]),
            ("Head servos (yaw, pitch)", ["look, scan, look up / down, doze,", "nod YES, shake NO, flinch"])]
    yy = 460
    for title, lines in body:
        s.box(xx, yy, xw, 88, C["out"], title, lines, tsize=15)
        s.arrow([(xx + 20, yy - 30 if yy == 460 else yy - 14), (xx + 20, yy - 2)], C["out"], width=1.6)
        yy += 102
    s.arrow([(xx + xw - 20, 430), (xx + xw - 20, 1010), (nx + nw / 2, 1010), (nx + nw / 2, ny + nh + 4)], C["det"],
            dash=True, label="what really happened -> own state", lx=nx + 20, ly=1002)

    # ---------------------------------------------------------------- the home PC
    py0 = 1060
    s.rect(30, py0, 2340, 410, "#140f1c", C["pc"], rx=14, sw=2)
    s.text(50, py0 + 34, "Home PC (192.168.8.110): records every tick, learns at night, sends firmware and what the robot learned", 22, "#e2c4f2", "bold")
    s.box(50, py0 + 60, 330, 170, C["pc"], "Diary (UDP, 20 Hz)",
          [f"{len(S.RAW)} raw readings, 13 + 2 channels", f"{len(S.SNN_OUT)} SNN numbers, {len(S.NN_OUT)} outputs",
           f"body + {len(S.PREV)} own-state inputs", "float16, + every log line", "-> lifelogd -> data/life/<day>",
           "gzipped daily (~5x)"], tsize=15)
    steps = [("0  Calibration", ["7 days of real diary:", "where each channel sits,", "which zones it visits"]),
             ("1  Teachers", ["innate appraisal -> 32", "behaviour teacher with", "memory -> 61"]),
             ("2  Train the SNN", ["1400 synthetic + 400 real", "5-minute lives, surrogate", "gradients, int8 QAT"]),
             ("3  Train the NN", ["imitation of the teacher,", "2 DAgger rounds: the net", "acts, the teacher corrects"]),
             ("4  Check", ["export -> C++, exact parity,", "simulator (digital twin):", "behaviour tests"]),
             ("5  OTA over Wi-Fi", ["new firmware on trial,", "confirmed after a healthy", "minute, else rolled back"])]
    x = 420
    for i, (title, lines) in enumerate(steps):
        s.box(x, py0 + 60, 300, 130, C["pc"], title, lines, tsize=15)
        if i:
            s.arrow([(x - 20, py0 + 125), (x - 2, py0 + 125)], C["pc"])
        x += 320
    s.arrow([(380, py0 + 125), (418, py0 + 125)], C["pc"])
    s.rect(420, py0 + 220, 1900, 100, "#1d1428", C["pc"], rx=10, sw=1.5)
    s.text(440, py0 + 250, "6  Night sleep 03:30: learning from its own life (sleep_v4.py)", 17, "#e2c4f2", "bold")
    s.text(440, py0 + 276, "replay every decision from the diary -> reward by people's answers (petting, talk, coming close / "
                           "shaking, rough pokes, leaving) -> AWR on the action heads -> zones shift (habituation, "
                           f"±{S.SHIFT_MAX})", 14)
    s.text(440, py0 + 300, "-> gates (behaviour tests in the simulator) -> learned blob to the robot, switched in without a "
                           "reboot; the SNN and the emotions stay innate", 13, C["dim"])
    s.text(50, py0 + 280, "teacher-free on the robot:", 14, weight="bold")
    s.text(50, py0 + 300, "the teachers exist only on the PC", 13, C["dim"])
    s.text(50, py0 + 360, "The firmware holds only brain v4 and what it learned (zone shifts, action heads); the PC records, "
                          "analyses and trains. No camera frames or sound leave the robot.", 15, C["dim"])
    s.text(50, py0 + 388, "The simulator runs the real firmware code (channels, SNN, NN, executor) in a virtual room "
                          "with scripted people, sounds, touches and light.", 15, C["dim"])
    s.arrow([(ex + 60, ey + eh), (ex + 60, 1035), (185, 1035), (185, py0 + 58)], C["pc"],
            label="diary: channels, SNN, NN", lx=195, ly=1028)
    s.arrow([(2220, py0 + 58), (2220, 876)], C["pc"], label="OTA", lx=2230, ly=960)
    s.save("BRAIN_V4_MAP.svg")


# ====================================================================== inside the two networks
def networks():
    s = Svg(2400, 1500)
    s.text(40, 52, "Brain v4: inside the two networks", 32, weight="bold")
    s.text(40, 82, "left: how a channel's number becomes spikes and zone activity; right: what the main network chooses",
           16, C["dim"])

    # ---- 1. one channel through the SNN (example: touch_head)
    ch = "touch_head"
    x0, y0 = 40, 130
    s.box(x0, y0, 1120, 560, C["snn"], f"One channel through the analyzer (example: {ch})")
    s.text(x0 + 20, y0 + 64, "value 0..1", 14, weight="bold")
    bw = 900
    zone_bar(s, x0 + 160, y0 + 50, bw, 20, ch)
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        s.text(x0 + 160 + v * bw, y0 + 88, f"{v:g}", 11, C["dim"], anchor="middle")
    s.text(x0 + 20, y0 + 130, "ENC: 8 Gaussian", 14, weight="bold")
    s.text(x0 + 20, y0 + 148, "receptive fields", 12, C["dim"])
    import math
    for k in range(8):
        c = (k + 0.5) / 8
        pts = []
        for i in range(61):
            v = i / 60
            pts.append((x0 + 160 + v * bw, y0 + 170 - 60 * math.exp(-((v - c) / 0.075) ** 2 / 2)))
        d = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in pts)
        s.add(f'<path d="{d}" stroke="{C["snn"]}" stroke-width="1.8" fill="none" opacity="0.85"/>')
    s.text(x0 + 160, y0 + 196, "+ ON cell (value rising) and OFF cell (falling): a touch onset fires at once", 12, C["dim"])
    s.text(x0 + 20, y0 + 240, "ZONE: 4 neurons", 14, weight="bold")
    s.text(x0 + 20, y0 + 258, "per zone", 12, C["dim"])
    zx = x0 + 160
    for i, z in enumerate(S.ZONES):
        xx = zx + i * 182
        s.rect(xx, y0 + 226, 170, 120, "#22192a", ZC[z], rx=8)
        s.text(xx + 10, y0 + 248, z, 14, ZC[z], "bold")
        for k, tau in enumerate((1, 8, 30, 90)):
            cx = xx + 24 + k * 38
            s.add(f'<circle cx="{cx}" cy="{y0 + 282}" r="12" fill="{ZC[z]}" opacity="{1 - k * 0.2:.1f}"/>')
            s.text(cx, y0 + 312, f"{tau}s", 11, C["dim"], anchor="middle")
        s.text(xx + 10, y0 + 334, f"val {S.ZONE_VALENCE[z]:+.2f}  aro {S.ZONE_AROUSAL[z]:.2f}", 11)
    s.text(x0 + 160, y0 + 370, "habituation: each neuron of a zone tires with its own time constant, so a steady stimulus "
                               "fades (1 s .. 90 s) and a new one stands out", 12, C["dim"])
    s.text(x0 + 20, y0 + 410, "MIX 282", 14, weight="bold")
    s.text(x0 + 160, y0 + 410, "recurrent pool, 80 / 20 excitatory / inhibitory (Dale's law), 12 % connections: "
                               "context - touch while it is loud, shake after a touch ...", 12)
    s.text(x0 + 20, y0 + 436, "ATT 96", 14, weight="bold")
    for i in range(12):
        for j in range(8):
            g = math.exp(-((i - 7.5) ** 2 / 6 + (j - 3) ** 2 / 3))
            s.rect(x0 + 160 + i * 18, y0 + 424 + j * 11, 16, 9, C["snn"] if g > 0.15 else "#2a2030", rx=1)
    s.text(x0 + 400, y0 + 446, "12 x 8 map; a bump where motion is seen, weighted by salience", 12)
    s.text(x0 + 400, y0 + 466, "-> att_x, att_y, att_strength (the eyes and the head follow it)", 12, C["dim"])
    s.text(x0 + 20, y0 + 542, "READOUT", 14, weight="bold")
    s.text(x0 + 160, y0 + 542, f"{len(S.SNN_OUT)} leaky integrators read all {N_SNN} neurons: smooth numbers for the "
                               "main network", 12)

    # ---- 2. the 32 outputs
    x1 = 1200
    s.box(x1, y0, 1160, 560, C["bus"], f"The analyzer's {len(S.SNN_OUT)} numbers")
    cols = [("core affect", S.SNN_OUT[:6]), ("salience", S.SNN_OUT[6:16]), ("hedonic", S.SNN_OUT[16:26]),
            ("drives + attention", S.SNN_OUT[26:])]
    for i, (name, items) in enumerate(cols):
        xx = x1 + 24 + i * 285
        s.text(xx, y0 + 62, name, 16, "#8fe3e3", "bold")
        for k, it in enumerate(items):
            s.text(xx, y0 + 90 + k * 22, it, 13)
    s.text(x1 + 24, y0 + 360, "core affect: valence (pleasant .. unpleasant), arousal (calm .. excited),", 13)
    s.text(x1 + 24, y0 + 380, "under / over (too little / too much happening: the inverted U of Wundt and Berlyne),", 13)
    s.text(x1 + 24, y0 + 400, "threat, surprise. salience: how much a channel matters now; hedonic: is it nice.", 13)
    s.text(x1 + 24, y0 + 440, "zone values that shape them:", 13, weight="bold")
    for i, z in enumerate(S.ZONES):
        xx = x1 + 24 + i * 220
        s.rect(xx, y0 + 458, 18, 12, ZC[z], rx=2)
        s.text(xx + 26, y0 + 469, f"{z}: val {S.ZONE_VALENCE[z]:+.2f}", 12, C["dim"])
    s.text(x1 + 24, y0 + 510, "the SNN gets only numbers and knows which channel each comes from; "
                              "it does not choose emotions", 12, C["dim"])

    # ---- 3. the main network
    y1 = 730
    s.box(40, y1, 2320, 730, C["nn"], f"Main network: one decider, {len(S.NN_OUT)} outputs")
    blocks = [(60, f"input {NIN}", [f"{len(S.SNN_OUT)} SNN numbers"] + [f"+ {p}" for p in S.PREV]),
              (380, f"GRU {H}", ["gated recurrent unit:", "remembers the last", "seconds"]),
              (660, f"emotion head", [f"{N_EMO} emotions (softmax)", "+ strength"]),
              (960, "action hidden 64", ["sees the GRU state", "and the chosen", "emotion"])]
    for bx, title, lines in blocks:
        hh = 50 + 17 * len(lines)
        s.rect(bx, y1 + 50, 260, hh, "#16213a", C["nn"], rx=8)
        s.text(bx + 12, y1 + 74, title, 15, "#a9c6ff", "bold")
        for k, ln in enumerate(lines):
            s.text(bx + 12, y1 + 96 + k * 17, ln, 12)
    s.arrow([(320, y1 + 100), (378, y1 + 100)], C["nn"])
    s.arrow([(640, y1 + 100), (658, y1 + 100)], C["nn"])
    s.arrow([(920, y1 + 100), (958, y1 + 100)], C["nn"])
    s.arrow([(640, y1 + 150), (940, y1 + 150), (958, y1 + 150)], C["nn"], dash=True)
    s.text(380, y1 + 180, f"{NN_PARAMS} float parameters, 10 Hz", 12, C["dim"])

    # emotions grid
    s.text(660, y1 + 240, "emotions", 16, "#a9c6ff", "bold")
    for i, e in enumerate(S.EMOTIONS):
        s.text(660 + (i % 3) * 95, y1 + 266 + (i // 3) * 20, e, 12)

    heads = [("voice", VOICE, ["say (0..1)", "style: " + ", ".join(S.SOUND_STYLES),
                               "mood: arousal, valence", "register, length, rate"]),
             ("LEDs", LEDS, ["colour: " + ", ".join(S.LED_COLORS[:4]) + ",", "  " + ", ".join(S.LED_COLORS[4:]),
                             "pattern: " + ", ".join(S.LED_PATTERNS), "brightness, rate"]),
             ("head", HEAD, ["mode: " + ", ".join(S.HEAD_MODES[:5]) + ",", "  " + ", ".join(S.HEAD_MODES[5:]),
                             "gaze yaw / pitch (-1..1)", "gesture amplitude, speed, repeats"])]
    for i, (name, n, lines) in enumerate(heads):
        bx = 1280 + i * 360
        s.rect(bx, y1 + 50, 340, 150, "#16213a", C["out"], rx=8)
        s.text(bx + 12, y1 + 74, f"{name}: {n} outputs", 15, C["out"], "bold")
        for k, ln in enumerate(lines):
            s.text(bx + 12, y1 + 98 + k * 20, ln, 12)
    s.arrow([(1220, y1 + 100), (1278, y1 + 100)], C["nn"])

    # gestures
    gy = y1 + 240
    s.text(1280, gy, "gesture library (the executor plays it on the network's wish)", 16, C["out"], "bold")
    gest = [("nod = YES", "pitch down / up, 5-12°, x1-3"), ("shake = NO", "yaw left / right, 5-14°, x1-3"),
            ("flinch", "head jerks back +12°, 0.6 s"), ("look / scan", "gaze x 60° yaw, 30° pitch"),
            ("look down / up", "-15° / +22° pitch"), ("doze", "head sinks -18°, slow")]
    for i, (a, b) in enumerate(gest):
        s.text(1280 + (i % 2) * 520, gy + 30 + (i // 2) * 24, a, 13, weight="bold")
        s.text(1280 + (i % 2) * 520 + 130, gy + 30 + (i // 2) * 24, b, 13, C["dim"])

    # training
    ty = y1 + 450
    s.rect(60, ty, 2280, 270, "#121a2c", C["line"], rx=10)
    s.text(80, ty + 32, "How it was taught", 18, weight="bold")
    lines = [
        "SNN: an innate appraisal (teacher_snn.py) turns channel histories into the 32 numbers; the SNN learns to reproduce "
        "them from its own spikes",
        "      (surrogate gradients, TBPTT; the last 30 % of the epochs with int8 fake quantisation, as on the robot).",
        "Data: 1400 synthetic 5-minute lives (quiet nights, play, being alone, rough handling) + 400 slices of the real diary.",
        "NN: a behaviour teacher with memory (teacher_nn.py) acts on the SNN's numbers and the network imitates it; then "
        "DAgger: the network acts, its own choices",
        "      feed back into its inputs and the teacher labels the states it really visits. Emotion agreement acting "
        "alone: 0.90 (real lives 0.92).",
        "Checks: C++ on the robot gives the same numbers as PyTorch (parity tests); simulator behaviour tests: every touch "
        "answered, petting is not annoying,",
        "      pokes are, nights are silent, play shows many faces, a lonely robot talks, shakes and claps frighten.",
        "Life (stage 6): every night the action heads are fine-tuned on the robot's real days with AWR, rewarded by how people "
        "answered; the zones of the environment channels shift by experience.",
    ]
    for k, ln in enumerate(lines):
        s.text(80, ty + 64 + k * 26, ln, 14)
    s.save("NETWORKS_V4.svg")


if __name__ == "__main__":
    brain_map()
    networks()
