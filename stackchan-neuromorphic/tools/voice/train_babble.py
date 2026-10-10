"""Trains BabbleNet: a small GRU that composes droid phrases. Given the voice intent it generates a phrase
note by note (kind, glide shape, ending, duration, pitch, slope, loudness, FM, harmonics, vibrato, rate,
gap, and when to stop; the shape / ending / continuous heads exist once per kind). Exports main/voice/babble_model_data.h.
  ../vocal_decoder/.venv/bin/python train_babble.py [--phrases 80000]"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "hybrid"))
import droid_teacher as T  # noqa: E402
from nn import FakeQuant, Header  # noqa: E402

NK, NS, NE, NC = 10, 4, 3, 9
NEMO = 21
N_IN = 12 + NEMO + NK + NS + NE + 4 + 2
P_KIND, P_SHAPE, P_END, P_CONT, P_STEP = 12 + NEMO, 12 + NEMO + NK, 12 + NEMO + NK + NS, 12 + NEMO + NK + NS + NE, 12 + NEMO + NK + NS + NE + 4
N_OUT = 1 + NK + NK * NS + NK * NE + NK * NC
MAXN = 7
SHAPE_AT, ENDING_AT = 1 + NK, 1 + NK + NK * NS
CONT_AT = ENDING_AT + NK * NE


def cont_vec(s):
    return [(s["dur"] - 250) / 150, s["f0"] / 5, s["slope"] / 8, s["loud"], s["fm"], s["harm"], s["vib"], s["rate"], s["gap"] / 80 - 0.5]


def encode_phrase(intent, emotion, phrase):
    iv = T.intent_vec(intent)
    ev = T.emotion_vec(emotion)
    X = np.zeros((MAXN + 1, N_IN), np.float32)
    Y = dict(stop=np.zeros(MAXN + 1, np.float32), kind=np.zeros(MAXN + 1, np.int64), shape=np.zeros(MAXN + 1, np.int64),
             ending=np.zeros(MAXN + 1, np.int64), cont=np.zeros((MAXN + 1, NC), np.float32), mask=np.zeros(MAXN + 1, np.float32))
    prev = None
    for j in range(len(phrase) + 1):
        X[j, :12] = iv
        X[j, 12:12 + NEMO] = ev
        if prev is not None:
            X[j, P_KIND + T.KINDS.index(prev["kind"])] = 1
            X[j, P_SHAPE + T.SHAPES.index(prev["shape"])] = 1
            X[j, P_END + T.ENDINGS.index(prev["ending"])] = 1
            X[j, P_CONT:P_CONT + 4] = [prev["f0"] / 5, prev["slope"] / 8, (prev["dur"] - 250) / 150, prev["loud"]]
        X[j, P_STEP] = j / 6
        X[j, P_STEP + 1] = 1.0 if j == 0 else 0.0
        Y["mask"][j] = 1
        if j == len(phrase):
            Y["stop"][j] = 1
            break
        s = phrase[j]
        Y["kind"][j], Y["shape"][j], Y["ending"][j] = T.KINDS.index(s["kind"]), T.SHAPES.index(s["shape"]), T.ENDINGS.index(s["ending"])
        Y["cont"][j] = cont_vec(s)
        prev = s
    return X, Y


class BabbleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(N_IN, 64)
        self.gru = nn.GRU(64, 48, batch_first=True)
        self.fc1 = nn.Linear(48, 48)
        self.fc2 = nn.Linear(48, N_OUT)
        self.quant = False

    def qw(self, w):
        return FakeQuant.apply(w) if self.quant else w

    def forward(self, x, h0=None):
        d = F.relu(F.linear(x, self.qw(self.inp.weight), self.inp.bias))
        if h0 is None:
            h0 = d.new_zeros(1, d.shape[0], 48)
        if self.quant:
            o, h = torch._VF.gru(d, h0, [self.qw(self.gru.weight_ih_l0), self.qw(self.gru.weight_hh_l0), self.gru.bias_ih_l0,
                                         self.gru.bias_hh_l0], True, 1, 0.0, self.training, False, True)
        else:
            o, h = self.gru(d, h0)
        o = F.relu(F.linear(o, self.qw(self.fc1.weight), self.fc1.bias))
        return F.linear(o, self.qw(self.fc2.weight), self.fc2.bias), h


def unpack(o):
    lead = o.shape[:-1]
    return (o[..., 0], o[..., 1:1 + NK], o[..., SHAPE_AT:ENDING_AT].reshape(*lead, NK, NS),
            o[..., ENDING_AT:CONT_AT].reshape(*lead, NK, NE), o[..., CONT_AT:].reshape(*lead, NK, NC))


def pick_kind(block, kind):
    """block [..., NK, D]; the sampled kind's row -> [..., D]"""
    idx = kind[..., None, None].expand(*kind.shape, 1, block.shape[-1])
    return block.gather(-2, idx).squeeze(-2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrases", type=int, default=150000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--qat", type=int, default=8)
    a = ap.parse_args()
    rng = np.random.default_rng(5)
    Xs, Ys = [], []
    for _ in range(a.phrases):
        it = T.random_intent(rng)
        emo = T.emotion_of(it, rng)
        p = T.compose(it, rng, emo)
        x, y = encode_phrase(it, emo, p)
        Xs.append(x)
        Ys.append(y)
    X = torch.tensor(np.stack(Xs))
    Y = {k: torch.tensor(np.stack([y[k] for y in Ys])) for k in Ys[0]}
    print("phrases", len(X))
    dev = torch.device("cuda")
    X, Y = X.to(dev), {k: v.to(dev) for k, v in Y.items()}
    n_val = 4000
    model = BabbleNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    bs = 512
    steps = (len(X) - n_val) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=(a.epochs + a.qat) * steps, pct_start=0.1)

    def loss_of(idx):
        o, _ = model(X[idx])
        stop, kind, shape, ending, cont = unpack(o)
        m = Y["mask"][idx]
        live = m * (1 - Y["stop"][idx])
        k = Y["kind"][idx]
        ce = lambda lg, y: F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="none").reshape(y.shape)
        l = (F.binary_cross_entropy_with_logits(stop, Y["stop"][idx], reduction="none") * m).sum() / m.sum() * 2
        l = l + (ce(kind, k) * live).sum() / live.sum()
        l = l + (ce(pick_kind(shape, k), Y["shape"][idx]) * live).sum() / live.sum() * 0.6
        l = l + (ce(pick_kind(ending, k), Y["ending"][idx]) * live).sum() / live.sum() * 0.6
        l = l + 2.0 * (((pick_kind(cont, k) - Y["cont"][idx]) ** 2).sum(-1) * live).sum() / live.sum()
        return l

    for ep in range(a.epochs + a.qat):
        model.quant = ep >= a.epochs
        model.train()
        perm = torch.randperm(len(X) - n_val, device=dev) + n_val
        agg = 0.0
        for i in range(steps):
            loss = loss_of(perm[i * bs:(i + 1) * bs])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            agg += loss.item()
        if ep % 5 == 4:
            model.eval()
            with torch.no_grad():
                print(f"epoch {ep + 1} loss {agg / steps:.4f} val {loss_of(torch.arange(n_val, device=dev)).item():.4f} quant={model.quant}", flush=True)
    # residual std of the continuous outputs per kind (sampling noise on the robot)
    model.eval()
    with torch.no_grad():
        idx = torch.arange(n_val, device=dev)
        o, _ = model(X[idx])
        cont = unpack(o)[4]
        live = (Y["mask"][idx] * (1 - Y["stop"][idx])).bool()
        std = np.zeros((NK, NC), np.float32)
        for kk in range(NK):
            sel = live & (Y["kind"][idx] == kk)
            std[kk] = ((cont[:, :, kk] - Y["cont"][idx])[sel]).std(0).cpu().numpy()
    print("residual std per kind\n", std.round(3))
    torch.save(model.state_dict(), os.path.join(HERE, "data", "babble_net.pt"))
    h = Header("babble_model", True)
    cpu = model.cpu()
    for n in ("inp", "fc1", "fc2"):
        m = getattr(cpu, n)
        h.layer(n, m.weight, m.bias)
    h.layer("gru_i", cpu.gru.weight_ih_l0, cpu.gru.bias_ih_l0)
    h.layer("gru_h", cpu.gru.weight_hh_l0, cpu.gru.bias_hh_l0)
    h.float("cont_std", std.reshape(-1))
    h.write(os.path.join(ROOT, "main", "voice", "babble_model_data.h"))
    print("exported", sum(p.numel() for p in cpu.parameters()), "parameters")


if __name__ == "__main__":
    main()
