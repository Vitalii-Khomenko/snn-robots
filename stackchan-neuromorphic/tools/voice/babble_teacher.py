"""The babble teacher: how a small creature with moods babbles.

Given a voice intent (mood, energy, register, melody tilt, expressiveness,
length, rate, four styles and "singing") it composes a phrase of 1-6 syllables:
consonant+vowel[+nasal] units (ba, di, mo, a-u, ne...) with phonotactics that
depend on the style, and a pitch melody taken from a REAL speech phrase
(prosody_bank.py), scaled by arousal. Only a training source: the robot runs a
small recurrent network that learned this distribution (train_babble.py).
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ONS = ["", "b", "d", "g", "p", "t", "k", "m", "n", "w", "y", "l", "r", "h", "s"]
NUC = ["a", "e", "i", "o", "u", "x", "A", "U"]
CODA = ["", "m", "n"]
INTENT = ["valence", "arousal", "register", "slope", "expr", "length", "rate", "calm", "ask", "play", "grumble", "sing"]

# onset weights per style, nucleus weights per style, coda probability, duration factor
STYLE = {
    "calm": dict(ons={"m": .32, "n": .18, "": .22, "h": .12, "w": .08, "l": .08}, nuc={"o": .28, "u": .28, "x": .2, "a": .16, "U": .08},
                 coda=.35, dur=1.2, breath=.55, creak=.0, maxn=3),
    "ask": dict(ons={"b": .16, "d": .14, "g": .08, "w": .1, "y": .1, "": .16, "m": .1, "n": .08, "h": .05, "l": .03},
                nuc={"a": .3, "e": .24, "i": .2, "o": .1, "u": .04, "A": .12}, coda=.08, dur=1.0, breath=.1, creak=.0, maxn=4),
    "play": dict(ons={"b": .2, "d": .16, "g": .08, "m": .12, "n": .06, "p": .08, "t": .07, "k": .05, "w": .06, "y": .06, "l": .06},
                 nuc={"a": .34, "i": .2, "e": .14, "o": .1, "u": .08, "A": .08, "U": .06}, coda=.05, dur=.85, breath=.0, creak=.0, maxn=6),
    "grumble": dict(ons={"g": .24, "k": .12, "n": .14, "m": .1, "b": .08, "": .22, "h": .1},
                    nuc={"u": .3, "x": .3, "o": .2, "a": .2}, coda=.42, dur=.82, breath=.0, creak=.6, maxn=2),
}
SONG = dict(ons={"l": .3, "n": .2, "m": .15, "w": .15, "": .12, "y": .08}, nuc={"a": .3, "o": .22, "u": .18, "i": .16, "e": .14})
PENTATONIC = np.array([-5, -3, 0, 2, 4, 7, 9, 12])

_bank = None


def bank():
    global _bank
    if _bank is None:
        d = np.load(os.path.join(HERE, "data", "prosody_bank.npz"))
        _bank = (d["contours"], d["desc"])
    return _bank


def pick(rng, table):
    k = list(table)
    p = np.array([table[x] for x in k], float)
    return k[rng.choice(len(k), p=p / p.sum())]


def template(rng, style, ask):
    c, d = bank()
    final, rng_st = d[:, 1], d[:, 2]
    if ask > 0.5:
        ok = np.where(final > 1.5)[0]
    elif style in ("calm", "grumble"):
        ok = np.where(final < -0.3)[0]
    elif style == "play":
        ok = np.where(rng_st > 12)[0]
    else:
        ok = np.arange(len(c))
    if len(ok) < 20:
        ok = np.arange(len(c))
    return c[rng.choice(ok)]


def compose(intent, rng):
    """intent: dict of INTENT -> float. Returns list of syllable dicts."""
    v, a, reg, slope, expr, L, rate = (intent[k] for k in INTENT[:7])
    sw = np.array([intent["calm"], intent["ask"], intent["play"], intent["grumble"]], float) + 1e-3
    names = ["calm", "ask", "play", "grumble"]
    main = names[int(np.argmax(sw))]
    ask = intent["ask"] / (sw.sum())
    sing = intent["sing"] > 0.5
    n = int(np.clip(round(1 + L * 4.5 + rng.normal(0, 0.6)), 1, 6))
    if not sing:
        n = min(n, STYLE[main]["maxn"] + (1 if L > 0.8 else 0))
    else:
        n = int(np.clip(round(2 + L * 4 + rng.normal(0, 0.7)), 2, 7))
    out = []
    # melody contour
    scale = float(np.clip(0.45 + 1.1 * a * (0.5 + 0.5 * expr), 0.3, 1.6))
    cont = template(rng, main, ask)
    for i in range(n):
        st = rng.choice(len(sw), p=sw / sw.sum()) if not sing else 1
        style = STYLE[names[st]] if not sing else None
        if sing:
            ons = pick(rng, SONG["ons"])
            nuc = pick(rng, SONG["nuc"])
            coda = ""
            dur = (300 + 170 * (1 - rate)) * rng.uniform(0.85, 1.25)
            f0 = float(PENTATONIC[min(len(PENTATONIC) - 1, max(0, (i * 2 + rng.integers(-1, 3)) % len(PENTATONIC)))]) * 0.5
            sl = rng.normal(0, 0.8)
            breath, creak = 0.05, 0.0
            loud = 0.55 + 0.3 * a
            vib = 0.5 + 0.5 * expr
        else:
            ons = pick(rng, style["ons"])
            nuc = pick(rng, style["nuc"])
            coda = ("m" if rng.random() < .5 else "n") if rng.random() < style["coda"] and i == n - 1 else ""
            if i > 0 and names[st] == "play" and rng.random() < 0.5:   # reduplication: ba-ba, di-di
                ons, nuc = out[-1]["onset"], out[-1]["nucleus"]
            dur = (275 - 150 * rate) * style["dur"] * rng.uniform(0.85, 1.18)
            if i == n - 1:
                dur *= 1.25                                              # phrase-final lengthening
            u = (i + 0.5) / n
            ci = lambda x: np.interp(np.clip(x, 0, 1) * 31, np.arange(32), cont)
            f0 = float(ci(u)) * scale + 1.6 * v + 4.0 * slope * (u - 0.5)
            sl = float((ci(u + 0.5 / n) - ci(u - 0.5 / n)) * scale * 0.5)
            if names[st] == "play":
                f0 += 2.2 * (1 if i % 2 else -1) * (0.4 + a)               # bounce
            if i == n - 1 and (names[st] == "ask" or ask > 0.5):
                sl += (3.0 + 5.0 * a) * (0.6 + 0.4 * expr)                 # the question rises
                f0 += 1.0
            if names[st] in ("calm", "grumble") and i == n - 1:
                sl -= 2.0 + 2.0 * (1 - a)
            breath, creak = style["breath"] * rng.uniform(.6, 1.2), style["creak"] * rng.uniform(.6, 1.2)
            loud = float(np.clip(0.42 + 0.45 * a + (0.1 if i == 0 else 0) + rng.normal(0, 0.05) - 0.12 * (names[st] == "calm"), 0.2, 1.0))
            vib = 0.0
        out.append(dict(onset=ons, nucleus=nuc, coda=coda, dur=float(np.clip(dur, 110, 560)), f0=float(np.clip(f0, -7, 11)),
                        slope=float(np.clip(sl, -9, 9)), loud=loud, breath=float(np.clip(breath, 0, 1)), creak=float(np.clip(creak, 0, 1)),
                        gap=float(np.clip(40 - 30 * rate + rng.normal(0, 6), 5, 70)), vib=float(vib)))
    return out


def random_intent(rng):
    k = rng.dirichlet(np.full(4, 0.35))
    calm, ask, play, grumble = k
    a = np.clip(0.5 + 0.35 * (play - calm) + 0.2 * (grumble) + rng.normal(0, 0.2), 0, 1)
    v = np.clip(0.4 * play + 0.15 * ask - 0.7 * grumble + 0.1 * calm + rng.normal(0, 0.25), -1, 1)
    return dict(valence=v, arousal=a, register=rng.uniform(0, 1), slope=np.clip(rng.normal(0, 0.5), -1, 1) + 0.5 * (ask - grumble),
                expr=rng.uniform(0, 1), length=rng.uniform(0, 1), rate=np.clip(0.5 + 0.35 * (play - calm) + rng.normal(0, 0.2), 0, 1),
                calm=calm, ask=ask, play=play, grumble=grumble, sing=float(rng.random() < 0.1))


def intent_vec(d):
    return np.array([d[k] for k in INTENT], np.float32)


def text(phrase):
    return "-".join((s["onset"] + s["nucleus"] + s["coda"].upper()) for s in phrase)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    for _ in range(12):
        it = random_intent(rng)
        ph = compose(it, rng)
        print({k: round(float(v), 2) for k, v in it.items() if k in ("arousal", "valence", "calm", "ask", "play", "grumble")}, text(ph))
