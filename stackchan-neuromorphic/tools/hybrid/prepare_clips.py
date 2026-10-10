"""Real sounds for the Auditory NN: ESC-50 (environmental sounds) and LibriSpeech
(speech) resampled to the robot's 24 kHz and written as raw int16 files the
simulator plays back (scenario verb `clip`), with the class mask of each.

  ./py prepare_clips.py        -> data/clips/*.raw + data/clips/index.json
"""
import csv
import glob
import json
import os
import random

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.abspath(os.path.join(HERE, "..", "..", "..", "datasets"))
OUT = os.path.join(HERE, "data", "clips")
os.makedirs(OUT, exist_ok=True)

# ESC-50 category -> our classes (voice-like sounds count as speech-like)
CLASSES = {
    "clapping": "clap",
    "door_wood_knock": "sharp", "mouse_click": "sharp", "keyboard_typing": "sharp", "glass_breaking": "sharp",
    "can_opening": "sharp", "clock_tick": "sharp", "footsteps": "sharp", "dog": "sharp", "door_wood_creaks": "sharp",
    "vacuum_cleaner": "continuous", "washing_machine": "continuous", "engine": "continuous",
    "airplane": "continuous", "train": "continuous", "helicopter": "continuous", "wind": "continuous",
    "rain": "continuous", "sea_waves": "continuous", "crackling_fire": "continuous", "pouring_water": "continuous",
    "toilet_flush": "continuous", "brushing_teeth": "continuous", "hand_saw": "continuous", "chainsaw": "continuous",
    "water_drops": "sharp", "thunderstorm": "continuous+low", "snoring": "continuous+low",
    "church_bells": "tonal", "clock_alarm": "tonal+high", "siren": "tonal+high", "car_horn": "tonal",
    "chirping_birds": "tonal+high", "cat": "tonal+high", "insects": "continuous+high", "crickets": "continuous+high",
    "crow": "tonal", "rooster": "tonal", "hen": "tonal", "cow": "tonal+low", "pig": "tonal+low", "sheep": "tonal",
    "frog": "tonal", "laughing": "speech", "crying_baby": "speech+high", "coughing": "speech", "sneezing": "speech",
    "breathing": "", "drinking_sipping": "", "mouse_click ": "sharp", "fireworks": "sharp+low", "keyboard": "sharp",
}


def to24k(x, sr):
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(1)
    g = np.gcd(24000, sr)
    y = resample_poly(x, 24000 // g, sr // g)
    return np.clip(y * 32767, -32768, 32767).astype(np.int16)


index = []
rows = list(csv.DictReader(open(os.path.join(DS, "ESC-50-master", "meta", "esc50.csv"))))
for r in rows:
    cat = r["category"]
    mask = CLASSES.get(cat)
    if mask is None or mask == "":
        continue
    x, sr = sf.read(os.path.join(DS, "ESC-50-master", "audio", r["filename"]))
    y = to24k(x, sr)
    name = f"esc_{r['filename'][:-4]}.raw"
    y.tofile(os.path.join(OUT, name))
    index.append(dict(file=name, classes=mask, source="esc50:" + cat, fold=int(r["fold"])))

# speech: LibriSpeech utterances, 3-8 s pieces
rng = random.Random(1)
flacs = sorted(glob.glob(os.path.join(DS, "ls", "LibriSpeech", "dev-clean", "*", "*", "*.flac")))
rng.shuffle(flacs)
for i, f in enumerate(flacs[:900]):
    x, sr = sf.read(f)
    y = to24k(x, sr)
    n = int(rng.uniform(3, 8) * 24000)
    start = rng.randrange(0, max(1, len(y) - n))
    seg = y[start:start + n]
    name = f"ls_{i:04d}.raw"
    seg.tofile(os.path.join(OUT, name))
    index.append(dict(file=name, classes="speech", source="librispeech", fold=1 + i % 5))
json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=0)
print(len(index), "clips", {k: sum(1 for c in index if c["classes"] == k) for k in sorted({c["classes"] for c in index})})
