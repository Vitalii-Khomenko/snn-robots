"""Trains brain v4's main network: imitation of teacher_nn.py on the trained spiking analyzer's outputs, then DAgger
(the network acts, its own choices feed back into its inputs and the teacher's memory, the teacher labels the
states the network really visits).

  ./py train_nn_v4.py data              # SNN outputs of the analyzer's dataset (out/snn_data.npz) -> out/nn_snn.npz
  ./py train_nn_v4.py train [dagger]    # -> data/nn_v4.pt, report out/nn_report.txt
"""
import collections
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from nn_v4 import ACT_IDX, EMO_IDX, GROUPS, SIG, SIGNED, STRENGTH, NnV4, activate
from snn_v4 import SnnV4
from spec_v4 import EMOTIONS, HEAD_MODES, NN_OUT, PREV, SNN_OUT
import contingency as C
import muse as M
from teacher_nn import Teacher, acts_of

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
DATA = os.path.join(HERE, "data")
O = {n: i for i, n in enumerate(NN_OUT)}
SAY = O["say"]
GAZE = [O["gaze_yaw"], O["gaze_pitch"]]
LOOKS = [O["head_look"], O["head_scan"]]


def build():
    d = np.load(os.path.join(OUT, "snn_data.npz"))
    X = d["X"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SnnV4().to(dev)
    m.load_state_dict(torch.load(os.path.join(DATA, "snn_v4.pt"), weights_only=False)["state"])
    m.quant = True
    m.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), 25):  # the spike rates of a batch are kept for the whole life (GPU memory)
            xb = torch.tensor(X[i:i + 25], device=dev)
            o, _, _ = m(xb, m.init_state(len(xb), dev))
            outs.append(o[:, ::2].cpu().numpy())  # the main network runs at 10 Hz
            print(i, flush=True)
    S = np.concatenate(outs).astype(np.float32)
    np.savez_compressed(os.path.join(OUT, "nn_snn.npz"), S=S, real=d["real"])
    print("SNN outputs", S.shape)


def label(S, rng, policy=None, dev="cpu"):
    """Teacher rollouts over SNN output sequences S [N, T, 32]. policy=None: the teacher acts itself (imitation);
    else the network acts (DAgger) and the teacher labels. People answer what is really done or not
    (contingency.AnswerWorld, one per life). -> inputs [N, T, 55], targets [N, T, 61]."""
    N, T, _ = S.shape
    S = np.array(S, np.float32)  # a copy: the people's answers are written into it (contingency.inject)
    X = np.zeros((N, T, len(SNN_OUT) + len(PREV)), np.float32)
    Y = np.zeros((N, T, len(NN_OUT)), np.float32)
    teachers = [Teacher() for _ in range(N)]
    worlds = [C.AnswerWorld(np.random.default_rng(int(rng.integers(1 << 31)))) for _ in range(N)]
    pace = M.run(S.astype(np.float32), rng)  # the Muse: pacemakers + ideas, driven by each life's mood
    h = None
    for t in range(T):
        for n in range(N):
            X[n, t, :32] = S[n, t]
            X[n, t, 32:] = teachers[n].prev_inputs(pace[n, t], worlds[n].out)
        if policy is None:
            for n in range(N):
                Y[n, t] = teachers[n].step(S[n, t], pace[n, t], worlds[n].out)
                worlds[n].step(*acts_of(Y[n, t]))
                C.inject(S[n], t + 1, worlds[n].answers)
        else:
            with torch.no_grad():
                raw, h = policy(torch.tensor(X[:, t:t + 1], device=dev), h)
                act = activate(raw)[:, 0].cpu().numpy()
            for n in range(N):
                executed = decisive(act[n])
                Y[n, t] = teachers[n].step(S[n, t], pace[n, t], worlds[n].out, executed=executed)
                worlds[n].step(*acts_of(executed))
                C.inject(S[n], t + 1, worlds[n].answers)
    return X, Y


def decisive(a):
    """The network's output as the executor uses it: the strongest choice per group, say above 0.5."""
    e = a.copy()
    for idx in [EMO_IDX] + list(GROUPS.values()):
        k = idx[int(np.argmax(a[idx]))]
        e[idx] = 0.0
        e[k] = 1.0
    e[SAY] = 1.0 if a[SAY] > 0.5 else 0.0
    return e


def class_weights(Y, idx, dev, power=0.5, cap=5.0, keep=()):
    """Rare choices weigh more (inverse frequency ** power, 0.3..cap; `keep`: positions left at 1). For the head
    modes: without it the network looked around 8x less than its teacher (scan 0.6 % vs 4.8 %); with the full square
    root it looked around 2x more and flinched more (2026-10-24) - so power 0.3, cap 3 and the flinch left at 1."""
    freq = Y[..., idx].reshape(-1, len(idx)).mean(0) + 1e-3
    w = np.clip((freq.mean() / freq) ** power, 0.3, cap)
    for k in keep:
        w[k] = 1.0
    return torch.tensor(w, dtype=torch.float32, device=dev)


def loss_fn(raw, y, emo_w, head_w=None):
    l_emo = F.cross_entropy(raw[..., EMO_IDX].reshape(-1, len(EMO_IDX)), y[..., EMO_IDX].argmax(-1).reshape(-1),
                            weight=emo_w)
    l_grp = sum(F.cross_entropy(raw[..., idx].reshape(-1, len(idx)), y[..., idx].argmax(-1).reshape(-1),
                                weight=head_w if g == "head" else None)
                for g, idx in GROUPS.items())
    l_say = F.binary_cross_entropy_with_logits(raw[..., SAY], y[..., SAY], pos_weight=torch.tensor(4.0, device=raw.device))
    sig = [STRENGTH] + [i for i in SIG if i != SAY]
    l_sig = F.mse_loss(torch.sigmoid(raw[..., sig]), y[..., sig])
    looking = y[..., LOOKS].sum(-1, keepdim=True)
    l_gaze = ((torch.tanh(raw[..., GAZE]) - y[..., GAZE]) ** 2 * (0.1 + looking)).mean()
    sv = [i for i in SIGNED if i not in GAZE]
    l_sv = F.mse_loss(torch.tanh(raw[..., sv]), y[..., sv])
    return l_emo + 0.5 * l_grp + 1.5 * l_say + 2.0 * l_sig + 2.0 * l_gaze + l_sv


def fit(model, X, Y, epochs, dev, emo_w, log, head_w=None):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n_val = max(20, len(X) // 12)
    tr, va = np.arange(n_val, len(X)), np.arange(n_val)
    rng = np.random.default_rng(0)
    best, best_state = 1e9, None
    for ep in range(epochs):
        t0 = time.time()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // 48)):
            xb, yb = torch.tensor(X[b], device=dev), torch.tensor(Y[b], device=dev)
            raw, _ = model(xb)
            loss = loss_fn(raw, yb, emo_w, head_w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        with torch.no_grad():  # in chunks: with three DAgger rounds the whole validation set no longer fits the GPU
            vls, accs = [], []
            for c in np.array_split(va, max(1, len(va) // 64)):
                raw, _ = model(torch.tensor(X[c], device=dev))
                vls.append(float(loss_fn(raw, torch.tensor(Y[c], device=dev), emo_w, head_w)) * len(c))
                accs.append(float((raw[..., EMO_IDX].argmax(-1).cpu() == torch.tensor(Y[c][..., EMO_IDX]).argmax(-1))
                                  .float().mean()) * len(c))
            vl, acc = sum(vls) / len(va), sum(accs) / len(va)
        msg = f"  epoch {ep + 1}/{epochs} val {vl:.3f} emotion acc {acc:.3f} {time.time() - t0:.0f}s"
        print(msg, flush=True)
        log.append(msg)
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best


def census(model, S, rng, dev):
    """The network acting alone on held-out lives: emotions, head modes, calls per hour, touches answered."""
    X, Y = label(S, rng, policy=model, dev=dev)
    with torch.no_grad():
        raw, _ = model(torch.tensor(X, device=dev))
        A = activate(raw).cpu().numpy()
    emo = collections.Counter(EMOTIONS[i] for i in A[..., EMO_IDX].argmax(-1).ravel())
    head = collections.Counter(HEAD_MODES[i] for i in A[..., GROUPS["head"]].argmax(-1).ravel())
    say = A[..., SAY] > 0.5
    calls = int((say[:, 1:] & ~say[:, :-1]).sum())
    hours = A.shape[0] * A.shape[1] * 0.1 / 3600
    agree = float((A[..., EMO_IDX].argmax(-1) == Y[..., EMO_IDX].argmax(-1)).mean())
    n = A.shape[0] * A.shape[1]
    return [f"calls/h {calls / hours:.0f}, emotion agreement with the teacher {agree:.3f}",
            "emotions: " + ", ".join(f"{k} {v / n * 100:.1f}%" for k, v in emo.most_common()),
            "head: " + ", ".join(f"{k} {v / n * 100:.1f}%" for k, v in head.most_common())]


def train(dagger_rounds):
    d = np.load(os.path.join(OUT, "nn_snn.npz"))
    S, real = d["S"], d["real"]
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(S))
    test, pool = idx[:60], idx[60:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log = []
    t0 = time.time()
    X, Y = label(S[pool], rng)
    print(f"imitation labels {X.shape} in {time.time() - t0:.0f}s", flush=True)
    freq = Y[..., EMO_IDX].reshape(-1, len(EMO_IDX)).mean(0) + 1e-3
    emo_w = torch.tensor(np.clip((freq.mean() / freq) ** 0.5, 0.3, 5.0), dtype=torch.float32, device=dev)
    head_w = class_weights(Y, GROUPS["head"], dev, power=0.3, cap=3.0, keep=(HEAD_MODES.index("flinch"),))
    log.append("head mode weights: " + ", ".join(f"{m} {w:.2f}" for m, w in zip(HEAD_MODES, head_w.tolist())))
    model = NnV4().to(dev)
    log.append("imitation")
    fit(model, X, Y, 25, dev, emo_w, log, head_w)
    for r in range(dagger_rounds):
        sub = rng.choice(pool, size=min(600, len(pool)), replace=False)
        t0 = time.time()
        Xd, Yd = label(S[sub], rng, policy=model, dev=dev)
        print(f"DAgger round {r + 1}: {Xd.shape} labelled in {time.time() - t0:.0f}s", flush=True)
        X, Y = np.concatenate([X, Xd]), np.concatenate([Y, Yd])
        log.append(f"DAgger round {r + 1}")
        fit(model, X, Y, 15, dev, emo_w, log, head_w)
    os.makedirs(DATA, exist_ok=True)
    torch.save({"state": model.state_dict()}, os.path.join(DATA, "nn_v4.pt"))
    lines = census(model, S[test], rng, dev)
    lines_real = census(model, S[test][real[test]], rng, dev) if real[test].any() else []
    text = "\n".join(log + ["", "held-out lives, the network acting alone:"] + lines +
                     (["", "real diary lives only:"] + lines_real if lines_real else [])) + "\n"
    open(os.path.join(OUT, "nn_report.txt"), "w").write(text)
    print(text)


if __name__ == "__main__":
    assert len(ACT_IDX) + len(EMO_IDX) + 1 == len(NN_OUT)
    if sys.argv[1] == "data":
        build()
    else:
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
