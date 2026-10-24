"""Brain v4's main network (PyTorch twin of main/v4/nn_v4.cpp). BRAIN_V4_PLAN.txt section 6.

One network makes every decision, in two steps so that they agree:
  input (32 from the spiking analyzer + 12 "what I just did") -> GRU 64 -> EMOTION head (21 + strength)
  [GRU state + chosen emotion's probabilities] -> hidden 64 -> ACTION heads (voice, LEDs, head)
Since every action head reads the emotion, an angry face with a happy chirp cannot happen by construction.
10 Hz.
"""
import torch
import torch.nn as nn

from spec_v4 import NN_IN, NN_OUT, NN_SIGNED, NN_SOFTMAX

NIN, NOUT, H, HA = len(NN_IN), len(NN_OUT), 64, 64
EMO_IDX = [NN_OUT.index(n) for n in NN_SOFTMAX["emo"]]
STRENGTH = NN_OUT.index("emo_strength")
ACT_IDX = [i for i in range(NOUT) if i not in EMO_IDX and i != STRENGTH]
GROUPS = {g: [NN_OUT.index(n) for n in names] for g, names in NN_SOFTMAX.items() if g != "emo"}
SIGNED = [NN_OUT.index(n) for n in NN_SIGNED]
SOFT = sorted(i for g in GROUPS.values() for i in g)
SIG = [i for i in ACT_IDX if i not in SOFT and i not in SIGNED]


class NnV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(NIN, H, batch_first=True)
        self.emo = nn.Linear(H, len(EMO_IDX) + 1)
        self.act1 = nn.Linear(H + len(EMO_IDX), HA)
        self.act2 = nn.Linear(HA, len(ACT_IDX))

    def forward(self, x, h=None):
        """x [B, T, 44] -> raw logits [B, T, 61] (activations: see activate), h."""
        z, h = self.gru(x, h)
        e = self.emo(z)
        p = torch.softmax(e[..., :len(EMO_IDX)], -1)
        a = self.act2(torch.tanh(self.act1(torch.cat([z, p], -1))))
        out = torch.zeros(*x.shape[:2], NOUT, device=x.device)
        out[..., EMO_IDX] = e[..., :len(EMO_IDX)]
        out[..., STRENGTH] = e[..., -1]
        out[..., ACT_IDX] = a
        return out, h


def activate(raw):
    """Logits -> what the robot uses: softmax per group, tanh for signed values, sigmoid for the rest."""
    y = torch.empty_like(raw)
    y[..., EMO_IDX] = torch.softmax(raw[..., EMO_IDX], -1)
    for idx in GROUPS.values():
        y[..., idx] = torch.softmax(raw[..., idx], -1)
    y[..., SIGNED] = torch.tanh(raw[..., SIGNED])
    rest = [STRENGTH] + SIG
    y[..., rest] = torch.sigmoid(raw[..., rest])
    return y
