"""Renders phrases of the droid teacher (or any list of phrases) with the C++ synthesizer.
  ../vocal_decoder/.venv/bin/python render_teacher.py out.wav [n_per_mood]"""
import os
import subprocess
import sys

import numpy as np

import droid_teacher as T

HERE = os.path.dirname(os.path.abspath(__file__))
IDX = {k: i for i, k in enumerate(T.KINDS)}
SH = {k: i for i, k in enumerate(T.SHAPES)}
EN = {k: i for i, k in enumerate(T.ENDINGS)}

MOODS = [
    ("question", dict(valence=.2, arousal=.6, register=.6, slope=.4, expr=.6, length=.4, rate=.55, calm=0, ask=1, play=0, grumble=0, sing=0)),
    ("play", dict(valence=.7, arousal=.85, register=.75, slope=.1, expr=.8, length=.8, rate=.85, calm=0, ask=.1, play=1, grumble=0, sing=0)),
    ("calm", dict(valence=.1, arousal=.2, register=.35, slope=-.4, expr=.5, length=.5, rate=.2, calm=1, ask=0, play=0, grumble=0, sing=0)),
    ("grumble", dict(valence=-.8, arousal=.6, register=.15, slope=-.5, expr=.4, length=.4, rate=.5, calm=0, ask=0, play=0, grumble=1, sing=0)),
    ("startle", dict(valence=-.2, arousal=.95, register=.8, slope=.5, expr=.7, length=.2, rate=.8, calm=0, ask=.8, play=.2, grumble=0, sing=0)),
    ("song", dict(valence=.6, arousal=.5, register=.6, slope=0, expr=.8, length=.8, rate=.3, calm=.5, ask=0, play=.5, grumble=0, sing=1)),
]


def spec_lines(phrase, reg):
    base = 520.0 * 2 ** (1.2 * reg)
    out = []
    for s in phrase:
        out.append(f"{base:.1f} {IDX[s['kind']]} {SH[s['shape']]} {EN[s['ending']]} {s['dur']:.0f} {s['f0']:.2f} {s['slope']:.2f} "
                   f"{s['loud']:.2f} {s['fm']:.2f} {s['harm']:.2f} {s['vib']:.2f} {s['rate']:.2f} {s['gap']:.0f}")
    return "\n".join(out) + "\n\n"


def render(phrases_regs, out):
    spec = os.path.join(os.path.dirname(out) or ".", "spec.tmp")
    with open(spec, "w") as f:
        for ph, reg in phrases_regs:
            f.write(spec_lines(ph, reg))
    subprocess.run([os.path.join(HERE, "voice_preview"), out, spec], check=True)
    os.remove(spec)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/droid_teacher.wav"
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    rng = np.random.default_rng(4)
    items = []
    for emo in T.EMOTIONS:
        it = T.random_intent(rng)
        for _ in range(per):
            ph = T.compose(it, rng, emo)
            print(f"{emo:10s}", T.text(ph))
            items.append((ph, it["register"]))
    render(items, out)
