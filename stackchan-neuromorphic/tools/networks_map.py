#!/usr/bin/env python3
"""Draws the five networks on the robot with their sizes:   python3 tools/networks_map.py   ->  docs/NETWORKS.svg
Sizes are read from the generated model headers and the PyTorch checkpoints, so the picture follows the files."""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def size(p):
    return os.path.getsize(os.path.join(ROOT, p))


def kb(n):
    return f"{n / 1024:.1f} KB" if n < 1024 * 1024 else f"{n / 1024 / 1024:.2f} MB"


def flash_bytes(header):
    """Weight data the firmware carries: int8 weights + float scales/biases + int16/int32 tables."""
    s = open(os.path.join(ROOT, header)).read()
    f = lambda t: sum(int(n) for n in re.findall(t + r"\s+\w+\[(\d+)\]", s))
    return f("int8_t"), f("float") * 4 + f("int16_t") * 2 + f("int32_t") * 4, f("int8_t")


NETS = [
    ("1  Vision CNN", "#2f6fdd", "main/hybrid/vision_model_data.h", "tools/hybrid/data/vision_net.pt",
     ["3 conv layers (8, 16, 24 filters, 3x3, pooling) + dense 64 -> 13 outputs",
      "Input: 32x24 image, 3 channels (luminance, ego-motion-compensated difference, raw difference) + 4 side values",
      "Output: motion per region, person, face, approaching / receding, target x/y/confidence, activity",
      "INT8, ~26 ms per frame at 7 fps (camera task, core 1)"]),
    ("2  Auditory MLP", "#2f6fdd", "main/hybrid/auditory_model_data.h", "tools/hybrid/data/auditory_net.pt",
     ["dense 64 -> 32 -> 8", "Input: 100 = 16 filter-bank bands x 6 frames + level, balance, 'I speak', 'my servos run'",
      "Output: silence, speech, clap, sharp, continuous noise, low, high, tonal",
      "INT8, ~1.5 ms per brain tick"]),
    ("3  Spiking core", "#d9534f", "main/hybrid/snn_model_data.h", "tools/hybrid/data/snn.pt",
     ["768 leaky integrate-and-fire neurons, 258 inputs (dense 258x768 projection), ~47.7 k sparse recurrent synapses",
      "Input: perception, body, context, prediction errors, noise pacemakers",
      "Output: 25 inner-state readouts - curiosity, arousal, boredom, annoyance, social need, playfulness, fatigue, sleep, attention",
      "INT8, ~7 ms per brain tick (20 Hz); trained with surrogate gradients"]),
    ("4  Policy GRU", "#2f6fdd", "main/hybrid/policy_model_data.h", "tools/hybrid/data/policy_net.pt",
     ["dense 64 -> GRU 64 -> dense 48 -> 51 outputs", "Input: 100 = all perception + SNN state + previous action + 2 slow-need clocks + 2 learned expectations",
      "Output: head target, gaze, blink, which of 21 faces, sound intent (pitch, glide, length, style), sing, invite, startle",
      "INT8, ~4 ms per brain tick; imitation of a designed teacher + DAgger"]),
    ("5  BabbleNet GRU", "#b06bd1", "main/voice/babble_model_data.h", "tools/voice/data/babble_net.pt",
     ["dense 56 -> 64, GRU 48, dense 48 -> 48, 171 outputs", "Input: sound intent (12) + the shown face (21 one-hot) + the phrase state",
      "Output: next note - kind (10 droid sounds), glide, ending, pitch, FM, harmonics, rate, stop",
      "INT8, own audio task (core 1, priority 5); the sound itself is an analogue-style droid synthesizer, not a network"]),
]

W, H = 1700, 1010
o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
     f'<rect width="{W}" height="{H}" fill="#0f1420"/>']
T = lambda x, y, t, s=15, c="#e6ebf5", w="normal", a="start": o.append(
    f'<text x="{x}" y="{y}" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="{s}" fill="{c}" font-weight="{w}" text-anchor="{a}">'
    f'{t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")}</text>')
T(40, 52, "The five trained networks on the StackChan robot", 32, w="bold")
T(40, 82, "M5Stack CoreS3 / ESP32-S3, one brain tick every 50 ms. Everything else (filter bank, front ends, safety limits, synthesizer) is ordinary C++.", 16, "#9aa6bf")
total = 0
y = 110
for title, col, header, ckpt, lines in NETS:
    i8, other, _ = flash_bytes(header)
    weights = i8 + other
    total += weights
    o.append(f'<rect x="40" y="{y}" width="{W - 80}" height="152" rx="10" fill="#171e2e" stroke="{col}" stroke-width="2.5"/>')
    o.append(f'<rect x="40" y="{y}" width="{W - 80}" height="32" rx="10" fill="{col}"/><rect x="40" y="{y + 20}" width="{W - 80}" height="12" fill="{col}"/>')
    T(54, y + 23, title, 19, "#ffffff", "bold")
    for k, l in enumerate(lines):
        T(54, y + 58 + k * 22, l, 15, "#e6ebf5" if k == 0 else "#b8c2d9")
    x = 1230
    T(x, y + 58, "Files and sizes", 14, "#9aa6bf", "bold")
    T(x, y + 80, f"weights in flash: {kb(weights)}", 15)
    T(x + 0, y + 100, f"  ({i8:,} INT8 weights + {kb(other)} scales / tables)", 13, "#9aa6bf")
    T(x, y + 120, f"C++ header {os.path.basename(header)}: {kb(size(header))}", 14)
    T(x, y + 140, f"PyTorch checkpoint {os.path.basename(ckpt)}: {kb(size(ckpt))}", 14)
    y += 166
o.append(f'<rect x="40" y="{y}" width="{W - 80}" height="62" rx="10" fill="#10262b" stroke="#26a6a6" stroke-width="2"/>')
T(54, y + 26, f"Total weight data of all five networks in flash: {kb(total)}  -  whole firmware image {kb(size('build/stackchan_neuromorphic.bin'))}", 17, "#e6ebf5", "bold")
T(54, y + 49, "Weights are INT8 with per-layer scales; the PyTorch checkpoints and the C++ headers are generated by tools/hybrid/train_*.py and tools/voice/train_babble.py.", 14, "#9aa6bf")
o.append("</svg>")
open(os.path.join(ROOT, "docs", "NETWORKS.svg"), "w").write("\n".join(o))
print("wrote docs/NETWORKS.svg", kb(total))
