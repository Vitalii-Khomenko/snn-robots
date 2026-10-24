"""The spiking core (PyTorch twin of main/hybrid/spiking_core.cpp).

768 LIF neurons in populations, a dense trained input projection, a sparse
trained recurrent matrix, endogenous noise, and leaky-integrator readout
cells (the slow drives live in the readout filters; the spiking core itself
supplies the fast dynamics, the topographic attention map and the traces).
Two 25 ms substeps per 50 ms brain tick.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from spec import SNN, SNN_INPUT, sin

POPS = [("ATT", 96), ("NOV", 64), ("HAB", 64), ("CUR", 64), ("ARO", 64), ("BOR", 48), ("ANN", 48), ("SOC", 48),
        ("PLAY", 48), ("FAT", 48), ("MUS", 32), ("TRC", 80), ("PRED", 64)]
N = sum(s for _, s in POPS)
assert N == 768
POP_AT = {}
_o = 0
for _n, _s in POPS:
    POP_AT[_n] = (_o, _s)
    _o += _s

# which population each readout reads, and its natural time constant (s)
READOUT = {
    "curiosity": ("CUR", 4.0), "boredom": ("BOR", 150.0), "arousal": ("ARO", 1.5), "annoyance": ("ANN", 5.0),
    "social": ("SOC", 200.0), "playful": ("PLAY", 5.0), "fatigue": ("FAT", 10.0), "sleep": ("FAT", 10.0),
    "music": ("MUS", 3.0), "alert": ("ARO", 1.0), "att_strength": ("ATT", 0.5), "nov_vis": ("NOV", 0.5),
    "nov_aud": ("NOV", 0.5), "nov_touch": ("NOV", 0.5), "tr_touch": ("TRC", 2.0), "tr_sound": ("TRC", 1.5),
    "tr_person": ("TRC", 3.0), "tr_motion": ("TRC", 1.0), "tr_novelty": ("TRC", 2.0), "tr_reward": ("TRC", 3.0),
    "hab_touch": ("HAB", 8.0), "hab_sound": ("HAB", 8.0), "surprise": ("PRED", 0.5),
}
SIGMOID_OUT = [n for n in SNN if n in READOUT]           # readouts through a sigmoid
MAP_W, MAP_H = 12, 8                                     # attention map
NBUS = len(SNN_INPUT)
NIN = NBUS + 2 * MAP_W * MAP_H                           # bus + target bump + light bump
DT_TICK = 0.05
SUBSTEPS = 2
KEEP_ATT = 0.8                                           # attention-rate EMA per tick

IDX = {n: sin(n) for n in SNN_INPUT}
_gx = (np.arange(MAP_W) + 0.5) / MAP_W * 2 - 1
_gy = (np.arange(MAP_H) + 0.5) / MAP_H * 2 - 1
MAP_X = np.tile(_gx, MAP_H).astype(np.float32)           # cell (row-major, y rows)
MAP_Y = np.repeat(_gy, MAP_W).astype(np.float32)


def encode(x):
    """bus [..., 66] -> core input [..., NIN]: the bus plus two fixed place codes."""
    g = lambda n: x[..., IDX[n]]
    mx, my = torch.as_tensor(MAP_X, device=x.device), torch.as_tensor(MAP_Y, device=x.device)

    def bump(cx, cy, amp):
        return amp[..., None] * torch.exp(-((mx - cx[..., None]) ** 2 / (2 * 0.17 ** 2) + (my - cy[..., None]) ** 2 / (2 * 0.25 ** 2)))

    target = bump(g("target_x"), g("target_y"), g("target_conf"))
    light = bump(g("light_x"), g("light_y"), g("light_strength") * 0.7)
    return torch.cat([x, target, light], -1)


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_minus_theta):
        ctx.save_for_backward(v_minus_theta)
        return (v_minus_theta > 0).float()

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * torch.clamp(1 - x.abs() / 0.6, min=0) * 0.8


def row_quant(w, dim):
    s = w.abs().amax(dim=dim, keepdim=True).clamp_min(1e-8) / 127.0
    return torch.round(w / s).clamp(-127, 127) * s


class FQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, dim):
        return row_quant(w, dim)

    @staticmethod
    def backward(ctx, g):
        return g, None


def make_mask(seed=7):
    rng = np.random.default_rng(seed)
    m = rng.random((N, N)) < 0.045
    for name, (o, s) in POP_AT.items():
        blk = rng.random((s, s)) < 0.22
        m[o:o + s, o:o + s] = blk
    # hub-like cross links to arousal / attention / traces
    for src in ("ARO", "NOV", "TRC", "ATT"):
        so, ss = POP_AT[src]
        m[:, so:so + ss] |= rng.random((N, ss)) < 0.06
    np.fill_diagonal(m, False)
    return m


class SpikingCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(N, NIN) * 0.05)
        self.W_rec = nn.Parameter(torch.randn(N, N) * 0.05)
        self.register_buffer("mask", torch.as_tensor(make_mask()))
        self.bias = nn.Parameter(torch.full((N,), 0.05))
        # membrane decay per 25 ms step, spread over 40 ms .. 8 s
        tau = np.exp(np.random.default_rng(3).uniform(np.log(0.04), np.log(8.0), N))
        self.beta_logit = nn.Parameter(torch.as_tensor(np.log(np.exp(-0.025 / tau) / (1 - np.exp(-0.025 / tau))), dtype=torch.float32))
        self.theta = nn.Parameter(torch.ones(N))
        self.noise = 0.12
        # readouts
        self.names = [n for n in SNN if n in READOUT]
        rmask = torch.zeros(len(self.names), N)
        for k, n in enumerate(self.names):
            o, s = POP_AT[READOUT[n][0]]
            rmask[k, o:o + s] = 1.0
        self.register_buffer("rmask", rmask)
        self.R = nn.Parameter(torch.randn(len(self.names), N) * 0.05)
        self.rbias = nn.Parameter(torch.zeros(len(self.names)))
        a0 = np.array([math.exp(-DT_TICK / READOUT[n][1]) for n in self.names])
        self.alpha_logit = nn.Parameter(torch.as_tensor(np.log(a0 / (1 - a0)), dtype=torch.float32))
        o, s = POP_AT["ATT"]
        self.att_slice = slice(o, o + s)
        self.register_buffer("att_x", torch.as_tensor(MAP_X))
        self.register_buffer("att_y", torch.as_tensor(MAP_Y))
        self.quant = False

    def weights(self):
        w_in, w_rec, r = self.W_in, self.W_rec * self.mask, self.R * self.rmask
        if self.quant:
            w_in = FQ.apply(w_in, 0)        # per input column
            w_rec = FQ.apply(w_rec, 1)      # per presynaptic row (column of the post-by-pre matrix) -> dim 0 of transposed
            r = FQ.apply(r, 1)
        return w_in, w_rec, r

    def init_state(self, B, dev):
        return dict(v=torch.zeros(B, N, device=dev), s=torch.zeros(B, N, device=dev),
                    z=torch.zeros(B, len(self.names), device=dev), att=torch.zeros(B, MAP_W * MAP_H, device=dev))

    def forward(self, x, state, noise=True):
        """x: bus [B,T,66] -> readouts dict-like tensors; returns (y [B,T,25], rates [B,T,N] mean spikes, state)."""
        B, T, _ = x.shape
        w_in, w_rec, r = self.weights()
        u = encode(x)
        i_in = u @ w_in.t()  # [B,T,N]
        beta = torch.sigmoid(self.beta_logit)
        alpha = torch.sigmoid(self.alpha_logit)
        v, s, z, att = state["v"], state["s"], state["z"], state["att"]
        ys, rate_sum = [], []
        wt = w_rec.t()
        for t in range(T):
            cnt = 0
            for _ in range(SUBSTEPS):
                cur = i_in[:, t] + s @ wt + self.bias
                if noise and self.noise > 0:
                    cur = cur + torch.randn_like(cur) * self.noise
                v = beta * v * (1 - s) + cur
                s = Spike.apply(v - self.theta)
                cnt = cnt + s
            n = cnt / SUBSTEPS
            z = alpha * z + (1 - alpha) * (n @ r.t())
            att = KEEP_ATT * att + (1 - KEEP_ATT) * n[:, self.att_slice]
            ys.append(torch.cat([z + self.rbias, att], 1))
            rate_sum.append(n)
        y = torch.stack(ys, 1)
        out = dict(v=v, s=s, z=z, att=att)
        return y, torch.stack(rate_sum, 1), out

    def decode(self, y):
        """raw [B,T,K+96] -> the 25 SNN values in spec order."""
        k = len(self.names)
        sig = torch.sigmoid(y[..., :k])
        att = y[..., k:]
        tot = att.sum(-1)
        ax = (att * self.att_x).sum(-1) / (tot + 0.05)
        ay = (att * self.att_y).sum(-1) / (tot + 0.05)
        cols = []
        for n in SNN:
            if n == "att_x":
                cols.append(ax)
            elif n == "att_y":
                cols.append(ay)
            else:
                cols.append(sig[..., self.names.index(n)])
        return torch.stack(cols, -1)


def export_header(core, path, trained=True):
    """Quantized weights exactly as SpikingCore::tick() reads them."""
    from nn import Header
    core = core.cpu()
    core.quant = False
    with torch.no_grad():
        w_in = core.W_in.detach().numpy()                       # [N, NIN]
        s_in = np.maximum(np.abs(w_in).max(0), 1e-8) / 127.0    # per input column
        q_in = np.clip(np.round(w_in / s_in), -127, 127).astype(np.int8)
        w_rec = (core.W_rec * core.mask).detach().numpy()       # [post, pre]
        s_rec = np.maximum(np.abs(w_rec).max(1), 1e-8) / 127.0  # per post row
        q_rec = np.clip(np.round(w_rec / s_rec[:, None]), -127, 127).astype(np.int8)
        r = (core.R * core.rmask).detach().numpy()
        s_r = np.maximum(np.abs(r).max(1), 1e-8) / 127.0
        q_r = np.clip(np.round(r / s_r[:, None]), -127, 127).astype(np.int8)
        alpha = torch.sigmoid(core.alpha_logit).numpy()
        beta = torch.sigmoid(core.beta_logit).numpy()
    h = Header("snn_model", trained)
    h.const("kSynapses", int((q_rec != 0).sum()))
    h.const("kReadouts", len(core.names))
    h.const("att_start", POP_AT["ATT"][0])
    h.int8("win_w", q_in.T.copy())                      # [NIN][N]
    h.float("win_s", s_in)
    h.float("bias", core.bias.detach().numpy())
    h.float("beta", beta)
    h.float("theta", core.theta.detach().numpy())
    # recurrent CSR by presynaptic neuron
    off, syn = [0], []
    for j in range(N):
        rows = np.nonzero(q_rec[:, j])[0]
        syn.extend(((rows.astype(np.uint32) << 8) | (q_rec[rows, j].astype(np.int16) & 0xFF).astype(np.uint32)).tolist())
        off.append(len(syn))
    h.u32("rec_off", off)
    h.u32("rec_syn", syn)
    h.float("rec_s", s_rec)
    # readouts over their population slice
    starts, sizes, woffs, wl = [], [], [], []
    for k, n in enumerate(core.names):
        o, sz = POP_AT[READOUT[n][0]]
        starts.append(o)
        sizes.append(sz)
        woffs.append(len(wl))
        wl.extend(q_r[k, o:o + sz].tolist())
    h.u16("ro_start", starts)
    h.u16("ro_size", sizes)
    h.u32("ro_woff", woffs)
    h.int8("ro_w", np.asarray(wl, np.int8))
    h.float("ro_s", s_r)
    h.float("ro_alpha", alpha)
    h.float("ro_bias", core.rbias.detach().numpy())
    h.u16("ro_out", [SNN.index(n) for n in core.names])
    h.write(path)
