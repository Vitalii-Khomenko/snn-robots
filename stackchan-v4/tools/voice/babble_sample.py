"""Samples phrases from the trained BabbleNet (as the robot does) and renders them with the C++ synthesizer.
  ../vocal_decoder/.venv/bin/python babble_sample.py [out.wav] [n_per_mood]"""
import os
import sys

import numpy as np
import torch

import droid_teacher as T
from render_teacher import MOODS, render
from train_babble import BabbleNet, MAXN, N_IN, NC, NEMO, NK, P_CONT, P_END, P_KIND, P_SHAPE, P_STEP, unpack

HERE = os.path.dirname(os.path.abspath(__file__))
net = BabbleNet()
net.load_state_dict(torch.load(os.path.join(HERE, "data", "babble_net.pt"), map_location="cpu"))
net.eval()


def _std():
    # residual noise as exported (read back from the header so the sampler matches the robot)
    import re
    txt = open(os.path.join(HERE, "..", "..", "main", "voice", "babble_model_data.h")).read()
    m = re.search(r"cont_std\[\d+\]\s*=\s*\{([^}]*)\}", txt)
    return np.array([float(x) for x in m.group(1).replace("f", "").split(",") if x.strip()]).reshape(NK, NC)


STD = _std()


def sample(intent, rng, emotion, temp=1.0):
    iv = T.intent_vec(intent)
    ev = T.emotion_vec(emotion)
    h = None
    prev = None
    phrase = []
    for j in range(MAXN + 1):
        x = np.zeros(N_IN, np.float32)
        x[:12] = iv
        x[12:12 + NEMO] = ev
        if prev:
            x[P_KIND + prev["k"]] = 1
            x[P_SHAPE + prev["s"]] = 1
            x[P_END + prev["e"]] = 1
            x[P_CONT:P_CONT + 4] = [prev["f0"] / 5, prev["slope"] / 8, (prev["dur"] - 250) / 150, prev["loud"]]
        x[P_STEP] = j / 6
        x[P_STEP + 1] = 1.0 if j == 0 else 0.0
        with torch.no_grad():
            o, h = net(torch.from_numpy(x).view(1, 1, -1), h)
        stop, kind, shape, ending, cont = [t[0, 0].numpy() for t in unpack(o)]
        if j >= 1 and (1 / (1 + np.exp(-stop))) > rng.random():
            break
        pick = lambda lg: int(rng.choice(len(lg), p=np.exp((lg - lg.max()) / temp) / np.exp((lg - lg.max()) / temp).sum()))
        k = pick(kind)
        s, e = pick(shape[k]), pick(ending[k])
        c = cont[k] + (rng.random(NC) - 0.5) * 1.732 * STD[k] * 0.5
        dur = float(np.clip(c[0] * 150 + 250, 45, 800))
        if e and dur < 140:
            dur = 140.0
        syl = dict(k=k, s=s, e=e, kind=T.KINDS[k], shape=T.SHAPES[s], ending=T.ENDINGS[e], dur=dur, f0=float(np.clip(c[1] * 5, -9, 12)),
                   slope=float(np.clip(c[2] * 8, -18, 18)), loud=float(np.clip(c[3], .3, 1)), fm=float(np.clip(c[4], 0, 1)),
                   harm=float(np.clip(c[5], 0, 1)), vib=float(np.clip(c[6], 0, 1)), rate=float(np.clip(c[7], 0, 1)),
                   gap=float(np.clip((c[8] + 0.5) * 80, 10, 130)))
        phrase.append(syl)
        prev = syl
    return phrase


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/droid_net.wav"
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    rng = np.random.default_rng(11)
    items = []
    for emo in T.EMOTIONS:
        it = T.random_intent(rng)
        for _ in range(per):
            ph = sample(it, rng, emo)
            print(f"{emo:10s}", T.text(ph))
            items.append((ph, it["register"]))
    render(items, out)
