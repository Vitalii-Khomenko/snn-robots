"""The droid teacher: how an astromech-style robot talks (R2-D2 school), as notes for droid_synth.

A phrase is a short list of notes. Each note is one of ten electronic sounds - whistle (long sliding tone),
chirp (fast sweep), pip (tiny metallic beep), burble (computer chatter), trill (fluttering tone), warble
(slow deep wobble), giggle (a run of rising blips), squeal (short loud scream), growl (low buzz with
amplitude chop) and coo (soft pure swell) - with a glide curve, an optional ending flick, FM depth,
harmonics, vibrato, speed and loudness.

Every face the robot can show has its own voice: EMOTIONS below gives, per emotion, the kinds it uses
(probabilities), a pitch offset, loudness, speed, direction of the glides and the ending. The network gets
the emotion as an input (21 values) next to the 12-value voice intent. The four styles (calm / ask / play /
grumble) and singing still colour the phrase: questions end with an upward flick, songs are pentatonic
whistles and coos.  Same interface as babble_teacher.py: compose(intent, rng, emotion), random_intent,
intent_vec, text.
"""
import numpy as np

import babble_teacher as B

INTENT = B.INTENT
random_intent = B.random_intent
intent_vec = B.intent_vec
pick = B.pick

KINDS = ["whistle", "chirp", "pip", "burble", "trill", "warble", "giggle", "squeal", "growl", "coo"]
LETTER = {"whistle": "W", "chirp": "C", "pip": "P", "burble": "B", "trill": "T", "warble": "V", "giggle": "G", "squeal": "S", "growl": "R", "coo": "O"}
SHAPES = ["linear", "fast", "slow", "scoop"]
ENDINGS = ["none", "up", "down"]
CONT = ["dur", "f0", "slope", "loud", "fm", "harm", "vib", "rate", "gap"]
PENTA = np.array([-5, -3, 0, 2, 4, 7, 9, 12])

# the order of face_style.h's Emotion enum
EMOTIONS = ["Normal", "Happy", "VeryHappy", "Laughing", "Sad", "Crying", "Sleepy", "Surprised", "Angry", "Annoyed", "Love", "Shy",
            "Wink", "Curious", "Confused", "Thinking", "Bored", "Scared", "Dizzy", "Cool", "Sick"]
#                       whistle chirp pip  burble trill warble giggle squeal growl coo
# per emotion: kind weights, pitch offset (st), loudness x, speed x (1 = normal; <1 slower), glide dir (+1 up, -1 down, 0 style), end
EMO = {
    "Normal":    ([.25, .15, .25, .20, .05, 0, 0, 0, 0, .10], 0.0, 1.0, 1.0, 0, None),
    "Happy":     ([.25, .25, .20, .10, 0, 0, .10, 0, 0, .10], 1.5, 1.0, 1.1, 1, None),
    "VeryHappy": ([.15, .25, .20, .15, 0, 0, .25, 0, 0, 0], 3.0, 1.0, 1.3, 1, "up"),
    "Laughing":  ([0, .20, .15, 0, 0, .10, .55, 0, 0, 0], 3.0, 1.0, 1.4, 1, None),
    "Sad":       ([.45, 0, 0, 0, .10, .10, 0, 0, 0, .35], -3.0, 0.8, 0.6, -1, "down"),
    "Crying":    ([.35, 0, 0, 0, .10, .40, 0, 0, 0, .15], -1.0, 0.85, 0.7, -1, "down"),
    "Sleepy":    ([.30, 0, 0, 0, .10, 0, 0, 0, 0, .60], -4.0, 0.6, 0.5, -1, "down"),
    "Surprised": ([0, .25, .25, 0, 0, 0, 0, .50, 0, 0], 4.0, 1.0, 1.4, 1, "up"),
    "Angry":     ([0, 0, .10, .15, .25, 0, 0, 0, .50, 0], -5.0, 1.0, 1.2, -1, "down"),
    "Annoyed":   ([0, 0, .20, .20, .30, 0, 0, 0, .30, 0], -3.0, 0.95, 1.1, -1, None),
    "Love":      ([.30, .10, 0, 0, 0, 0, .10, 0, 0, .50], 0.5, 0.75, 0.7, 1, None),
    "Shy":       ([.20, .10, .30, 0, 0, 0, 0, 0, 0, .40], 0.0, 0.65, 0.8, 0, None),
    "Wink":      ([.15, .30, .40, 0, 0, 0, .15, 0, 0, 0], 2.0, 0.9, 1.2, 1, "up"),
    "Curious":   ([.30, .35, .20, .15, 0, 0, 0, 0, 0, 0], 1.5, 0.95, 1.0, 1, "up"),
    "Confused":  ([.15, .15, 0, .30, 0, .40, 0, 0, 0, 0], 0.0, 0.9, 0.9, 0, "up"),
    "Thinking":  ([.10, 0, .30, .45, .15, 0, 0, 0, 0, 0], -1.0, 0.8, 0.6, 0, None),
    "Bored":     ([.50, 0, .15, 0, .15, 0, 0, 0, 0, .20], -3.0, 0.7, 0.5, -1, "down"),
    "Scared":    ([0, .15, 0, .15, 0, .25, 0, .45, 0, 0], 4.0, 1.0, 1.5, 1, None),
    "Dizzy":     ([.20, 0, 0, .20, 0, .60, 0, 0, 0, 0], 0.0, 0.9, 0.9, 0, None),
    "Cool":      ([.30, .10, .30, 0, 0, 0, 0, 0, .10, .20], -3.0, 0.85, 0.8, -1, "down"),
    "Sick":      ([0, 0, 0, .30, .30, .20, 0, 0, .20, 0], -4.0, 0.6, 0.6, -1, "down"),
}


def emotion_of(intent, rng):
    """A plausible face for the intent (the executor's own rules, roughly); sometimes any face."""
    v, a = intent["valence"], intent["arousal"]
    if rng.random() < 0.15:
        return rng.choice(EMOTIONS)
    if a < 0.15 and intent["calm"] > 0.5:
        return "Sleepy"
    if v > 0.3:
        if v > 0.55 and a < 0.35:
            return "Love"
        if rng.random() < 0.08:
            return rng.choice(["Shy", "Wink"])
        return "Laughing" if a > 0.65 else "VeryHappy" if a > 0.35 else "Happy"
    if v < -0.2:
        if a > 0.65:
            return "Scared"
        if a > 0.35:
            return "Angry" if v < -0.6 else "Annoyed"
        return "Crying" if v < -0.5 and rng.random() < 0.4 else "Sad"
    if intent["ask"] > 0.5 and a > 0.8:
        return "Surprised"
    if a > 0.55:
        return "Curious"
    r = rng.random()
    return "Thinking" if r < 0.2 else "Bored" if r < 0.35 else "Confused" if r < 0.42 else "Cool" if r < 0.48 else "Normal"


def note(kind, rng, sgn, a, expr, speed):
    k = KINDS[kind]
    n = dict(kind=k, ending="none", vib=0.0, rate=0.5, shape="linear")
    if k == "whistle":
        n["dur"] = rng.uniform(280, 750) / speed
        n["slope"] = sgn * rng.uniform(3, 14) * (0.6 + 0.6 * a)
        n["shape"] = pick(rng, {"linear": .4, "fast": .25, "slow": .15, "scoop": .2})
        n["fm"], n["harm"], n["vib"], n["rate"] = rng.uniform(.1, .45), rng.uniform(.05, .3), 0.1 + 0.4 * expr * rng.random(), rng.uniform(.2, .9)
    elif k == "chirp":
        n["dur"] = rng.uniform(100, 230) / speed
        n["slope"] = sgn * rng.uniform(7, 18) * (0.6 + 0.5 * a)
        n["shape"] = pick(rng, {"fast": .6, "linear": .2, "slow": .2})
        n["fm"], n["harm"] = rng.uniform(.3, .7), rng.uniform(.35, .8)
    elif k == "pip":
        n["dur"] = rng.uniform(45, 115) / speed
        n["slope"] = rng.uniform(-3, 3)
        n["fm"], n["harm"] = rng.uniform(.5, .9), rng.uniform(.4, .9)
    elif k == "burble":
        n["dur"] = rng.uniform(200, 650) / speed
        n["slope"] = sgn * rng.uniform(4, 14) * (0.5 + 0.7 * a)
        n["shape"] = pick(rng, {"linear": .6, "fast": .2, "slow": .2})
        n["fm"], n["harm"] = rng.uniform(.5, .95), rng.uniform(.3, .7)
        n["rate"] = float(np.clip(0.2 + 0.5 * a + 0.3 * (speed - 1) + rng.normal(0, .12), 0, 1))
    elif k == "trill":
        n["dur"] = rng.uniform(150, 420) / speed
        n["slope"] = sgn * rng.uniform(0, 8)
        n["shape"] = pick(rng, {"linear": .6, "fast": .2, "slow": .2})
        n["fm"], n["harm"] = rng.uniform(.2, .6), rng.uniform(.25, .6)
        n["rate"] = float(np.clip(0.25 + 0.5 * a + rng.normal(0, .15), 0, 1))
    elif k == "warble":
        n["dur"] = rng.uniform(250, 700) / speed
        n["slope"] = sgn * rng.uniform(0, 6)
        n["fm"], n["harm"], n["vib"], n["rate"] = rng.uniform(.1, .4), rng.uniform(.05, .2), rng.uniform(.4, 1.0), rng.uniform(.3, 1.0)
    elif k == "giggle":
        n["dur"] = rng.uniform(220, 600) / speed
        n["slope"] = abs(sgn or 1) * rng.uniform(1, 8) * rng.choice([1, 1, -1])
        n["fm"], n["harm"], n["rate"] = rng.uniform(.2, .5), rng.uniform(.3, .6), rng.uniform(.3, 1.0)
    elif k == "squeal":
        n["dur"] = rng.uniform(90, 260) / speed
        n["slope"] = rng.uniform(6, 16)
        n["shape"] = "fast"
        n["fm"], n["harm"] = rng.uniform(.5, .9), rng.uniform(.6, 1.0)
    elif k == "growl":
        n["dur"] = rng.uniform(150, 450) / speed
        n["slope"] = -rng.uniform(0, 6)
        n["shape"] = pick(rng, {"slow": .4, "linear": .6})
        n["fm"], n["harm"], n["rate"] = rng.uniform(.3, .7), rng.uniform(.6, 1.0), rng.uniform(.3, 1.0)
    else:  # coo
        n["dur"] = rng.uniform(300, 800) / speed
        n["slope"] = sgn * rng.uniform(1, 7)
        n["shape"] = pick(rng, {"linear": .6, "scoop": .4})
        n["fm"], n["harm"], n["vib"], n["rate"] = rng.uniform(0, .15), rng.uniform(0, .1), rng.uniform(.2, .7), rng.uniform(.2, .6)
    return n


KIND_PITCH = {"squeal": 3.0, "growl": -6.0, "coo": -1.5}


def compose(intent, rng, emotion=None):
    v, a, reg, slope, expr, L, rate = (intent[k] for k in INTENT[:7])
    emotion = emotion or emotion_of(intent, rng)
    weights, pitch_off, loud_mul, speed, direction, end = EMO[emotion]
    styles = ["calm", "ask", "play", "grumble"]
    sw = np.array([intent[s] for s in styles], float) + 1e-3
    main = styles[int(np.argmax(sw))]
    sing = intent["sing"] > 0.5
    n = int(np.clip(round(1 + L * 5 + rng.normal(0, 0.7)), 1, 7))
    if sing:
        n = int(np.clip(round(3 + L * 4 + rng.normal(0, 0.7)), 3, 7))
    out = []
    pitch = rng.uniform(-2, 3) + 3.0 * v + pitch_off
    total = 0.0
    wk = np.array(weights, float)
    wk = wk / wk.sum()
    sp = speed * (1.0 + 0.5 * rate - 0.25)
    for i in range(n):
        last = i == n - 1
        if sing:
            kind = KINDS.index(pick(rng, {"whistle": .7, "coo": .3}))
            nt = note(kind, rng, 0, a, expr, 1.0)
            nt.update(dur=rng.uniform(250, 480) * (0.8 + 0.5 * (1 - rate)), slope=rng.normal(0, 1.6), shape="linear", vib=0.5 + 0.5 * expr)
            f0 = float(PENTA[(rng.integers(0, len(PENTA)) + i) % len(PENTA)]) * 0.8 - 1
        else:
            kind = int(rng.choice(10, p=wk))
            # pip-pip-pip, chirp-chirp, giggle-giggle: repeats
            if i > 0 and out[-1]["kind"] in ("pip", "chirp") and main == "play" and rng.random() < 0.4:
                kind = KINDS.index(out[-1]["kind"])
            sgn = direction if direction != 0 else {"ask": 1, "calm": -1, "grumble": -1, "play": rng.choice([-1, 1])}[main]
            nt = note(kind, rng, sgn, a, expr, sp)
            pitch += rng.normal(0, 3.0) * (0.5 + a) if main in ("play", "ask") or direction > 0 else rng.normal(-0.8, 1.3)
            if main == "ask":
                pitch += 1.2
            f0 = pitch + KIND_PITCH.get(nt["kind"], 0.0)
        if last:
            e = end or ("up" if main == "ask" and rng.random() < 0.75 else "down" if main in ("calm", "grumble") and rng.random() < 0.6 else None)
            if e and rng.random() < 0.8:
                nt["ending"] = e
            elif rng.random() < 0.1:
                nt["ending"] = rng.choice(["up", "down"])
            if nt["ending"] != "none" and nt["dur"] < 140:
                nt["dur"] = 140.0
        nt["f0"] = float(np.clip(f0, -9, 12))
        nt["loud"] = float(np.clip((0.55 + 0.4 * a + rng.normal(0, 0.05)) * loud_mul, 0.3, 1.0))
        nt["gap"] = float(np.clip((70 - 55 * rate + rng.normal(0, 8) + (30 if main == "calm" else 0)) / max(speed, 0.5) ** 0.5, 12, 130))
        nt["dur"] = float(np.clip(nt["dur"], 45, 800))
        nt["slope"] = float(np.clip(nt["slope"], -18, 18))
        for key in ("fm", "harm", "vib", "rate"):
            nt[key] = float(np.clip(nt[key], 0, 1))
        total += nt["dur"] + nt["gap"]
        out.append(nt)
        if total > 2800 and i >= 1:
            break
    return out


def emotion_vec(emotion):
    v = np.zeros(len(EMOTIONS), np.float32)
    v[EMOTIONS.index(emotion)] = 1
    return v


def text(phrase):
    return " ".join(LETTER[s["kind"]] + ("+" if s["slope"] > 1 else "-" if s["slope"] < -1 else "") + {"none": "", "up": "?", "down": "."}[s["ending"]]
                    for s in phrase)


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    for e in EMOTIONS:
        it = random_intent(rng)
        print(f"{e:10s}", text(compose(it, rng, e)), "|", text(compose(it, rng, e)))
