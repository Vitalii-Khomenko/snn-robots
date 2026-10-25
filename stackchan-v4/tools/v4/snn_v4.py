"""Brain v4's spiking analyzer (PyTorch twin of main/v4/snn_v4.cpp). BRAIN_V4_PLAN.txt section 4.

768 leaky integrate-and-fire neurons in four layers, 20 Hz ticks of 2 substeps (25 ms):

  ENC   130  per channel 8 level neurons (Gaussian receptive fields over 0..1) + ON / OFF (rise / fall); fixed
  ZONE  260  per channel 5 zones x 4 neurons, innate wiring from ENC (the zone table), adaptive thresholds (ALIF:
             a constant stimulus fires them less and less - habituation, 4 time constants per zone); fixed
  MIX   282  recurrent (sparse, 80 % excitatory / 20 % inhibitory), trained input from ENC and ZONE
  ATT    96  12 x 8 attention map: a fixed bump where the movement is, trained input from MIX

Readout: 32 leaky integrators over the spikes of all 768 (trained), sigmoid (tanh for att_x / att_y) -> SNN_OUT.
Weights are int8 per output row (quantization-aware training at the end), like on the robot.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from spec_v4 import CHANNELS, SNN_IN, SNN_OUT, SNN_SIGNED, WHERE, ZONE_TABLE, ZONES, membership

NC = len(CHANNELS)
N_GRF = 8
N_ENC = NC * (N_GRF + 2)            # 130
N_ZPER = 4
N_ZONE = NC * len(ZONES) * N_ZPER   # 260
N_MIX = 282
N_EXC = 226
MAP_W, MAP_H = 12, 8
N_ATT = MAP_W * MAP_H               # 96
N = N_ENC + N_ZONE + N_MIX + N_ATT
assert N == 768
NOUT = len(SNN_OUT)
SUBSTEPS = 2
DT_SUB = 0.025

GRF_CENTERS = (np.arange(N_GRF) + 0.5) / N_GRF
GRF_SIGMA = 0.6 / N_GRF
GRF_GAIN = 1.4
ONOFF_GAIN = 25.0                  # a 0.04 step per tick drives an ON neuron to threshold
BETA_ENC = math.exp(-DT_SUB / 0.05)
BETA_ZONE = math.exp(-DT_SUB / 0.08)
BETA_MIX = math.exp(-DT_SUB / 0.10)
BETA_ATT = math.exp(-DT_SUB / 0.06)
ZONE_ADAPT_TAU = (1.0, 8.0, 30.0, 90.0)   # s, one per neuron of a zone group
ZONE_ADAPT_STEP = (0.0, 0.08, 0.05, 0.03)  # threshold rise per spike (the first neuron does not adapt)
ZONE_GAIN = 0.55
REC_DENSITY = 0.12
# readout time constants (s): how fast each number may change
RO_TAU = {"valence": 1.0, "arousal": 0.6, "under": 3.0, "over": 1.5, "threat": 0.6, "surprise": 0.15,
          "hunger_n": 3.0, "sleepiness": 3.0, "loneliness": 3.0, "att_x": 0.25, "att_y": 0.25, "att_strength": 0.25}
for _n in SNN_OUT:
    if _n.startswith("sal_"):
        RO_TAU[_n] = 0.25
    elif _n.startswith("hed_"):
        RO_TAU[_n] = 0.6

_gx = (np.arange(MAP_W) + 0.5) / MAP_W * 2 - 1
_gy = (np.arange(MAP_H) + 0.5) / MAP_H * 2 - 1
MAP_X = np.tile(_gx, MAP_H).astype(np.float32)
MAP_Y = np.repeat(_gy, MAP_W).astype(np.float32)
BUMP_SX, BUMP_SY, BUMP_GAIN = 0.2, 0.3, 1.6


def zone_weights():
    """Fixed ENC(GRF) -> ZONE weights [N_ZONE, NC * N_GRF]: a zone neuron listens to the level neurons of its own
    channel whose centre lies inside the zone (soft borders), nothing else. Missing zones stay silent."""
    W = np.zeros((N_ZONE, NC * N_GRF), np.float32)
    for c, ch in enumerate(CHANNELS):
        for z, zone in enumerate(ZONES):
            if zone not in ZONE_TABLE[ch]:
                continue
            lo, hi = ZONE_TABLE[ch][zone]
            m = membership(GRF_CENTERS, lo, hi, edge=0.5 / N_GRF)
            if m.sum() == 0:  # a zone narrower than a receptive field: the nearest one
                m[np.argmin(np.abs(GRF_CENTERS - (lo + hi) / 2))] = 1.0
            for j in range(N_ZPER):
                W[(c * len(ZONES) + z) * N_ZPER + j, c * N_GRF:(c + 1) * N_GRF] = m * ZONE_GAIN
    return W


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (1.0 + 4.0 * x.abs()) ** 2


spike = Spike.apply


def fake_quant_rows(w):
    s = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
    q = torch.round(w / s).clamp(-127, 127) * s
    return w + (q - w).detach()


class SnnV4(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("grf_c", torch.tensor(GRF_CENTERS, dtype=torch.float32))
        self.register_buffer("w_zone", torch.tensor(zone_weights()))
        adapt_tau = np.tile(ZONE_ADAPT_TAU, N_ZONE // N_ZPER)
        adapt_step = np.tile(ZONE_ADAPT_STEP, N_ZONE // N_ZPER)
        self.register_buffer("adapt_keep", torch.tensor(np.exp(-DT_SUB / adapt_tau), dtype=torch.float32))
        self.register_buffer("adapt_step", torch.tensor(adapt_step, dtype=torch.float32))
        self.register_buffer("map_x", torch.tensor(MAP_X))
        self.register_buffer("map_y", torch.tensor(MAP_Y))
        mask = (torch.rand(N_MIX, N_MIX, generator=g) < REC_DENSITY).float()
        mask.fill_diagonal_(0)
        self.register_buffer("rec_mask", mask)
        sign = torch.ones(N_MIX)
        sign[N_EXC:] = -1.0
        self.register_buffer("rec_sign", sign)  # by presynaptic neuron (Dale)
        self.w_in = nn.Parameter(torch.randn(N_MIX, N_ENC + N_ZONE, generator=g) * 0.6 / math.sqrt(40))
        self.w_rec = nn.Parameter(torch.rand(N_MIX, N_MIX, generator=g) * 0.25 / math.sqrt(N_MIX * REC_DENSITY))
        self.b_mix = nn.Parameter(torch.full((N_MIX,), 0.05))
        self.w_att = nn.Parameter(torch.randn(N_ATT, N_MIX, generator=g) * 0.1 / math.sqrt(N_MIX))
        self.b_att = nn.Parameter(torch.zeros(N_ATT))
        self.w_ro = nn.Parameter(torch.randn(NOUT, N, generator=g) * 0.5 / math.sqrt(N))
        self.b_ro = nn.Parameter(torch.zeros(NOUT))
        keep = np.array([math.exp(-0.05 / RO_TAU[n]) for n in SNN_OUT], np.float32)
        self.register_buffer("ro_keep", torch.tensor(keep))
        self.register_buffer("signed", torch.tensor([n in SNN_SIGNED for n in SNN_OUT]))
        self.quant = False

    # ---- weights as the robot uses them
    def W(self, name):
        if name == "rec":
            w = self.w_rec.abs() * self.rec_mask * self.rec_sign[None, :]
        else:
            w = getattr(self, "w_" + name)
        return fake_quant_rows(w) if self.quant else w

    def init_state(self, B, device):
        z = lambda n: torch.zeros(B, n, device=device)  # noqa: E731
        return dict(v_enc=z(N_ENC), v_zone=z(N_ZONE), a_zone=z(N_ZONE), v_mix=z(N_MIX), s_mix=z(N_MIX),
                    v_att=z(N_ATT), ro=z(NOUT), prev=None)

    def encode(self, x, prev):
        """x [B, 18] -> constant input currents of ENC [B, 130] and the attention bump [B, 96] for this tick."""
        ch = x[:, :NC]
        grf = GRF_GAIN * torch.exp(-(ch[:, :, None] - self.grf_c) ** 2 / (2 * GRF_SIGMA ** 2))  # [B, NC, 8]
        d = ch - (prev if prev is not None else ch)
        onoff = torch.stack([torch.relu(d), torch.relu(-d)], -1) * ONOFF_GAIN                 # [B, NC, 2]
        i_enc = torch.cat([grf.reshape(len(x), -1), onoff.reshape(len(x), -1)], 1)
        # a bump where the movement is, and one where a new bright spot is (2026-10-25)
        mx, my = x[:, NC + WHERE.index("motion_x")], x[:, NC + WHERE.index("motion_y")]
        lx, ly = x[:, NC + WHERE.index("light_x")], x[:, NC + WHERE.index("light_y")]
        amp = (ch[:, CHANNELS.index("motion")] * 4.0).clamp(0, 1) * BUMP_GAIN
        lamp = x[:, NC + WHERE.index("light_spot")].clamp(0, 1) * BUMP_GAIN

        def g(cx, cy):
            return torch.exp(-((self.map_x - cx[:, None]) ** 2 / (2 * BUMP_SX ** 2) +
                               (self.map_y - cy[:, None]) ** 2 / (2 * BUMP_SY ** 2)))
        bump = amp[:, None] * g(mx, my) + lamp[:, None] * g(lx, ly)
        return i_enc, bump

    def step(self, x, st):
        """One 50 ms tick. x [B, 18]. Returns out [B, 32] and spike counts per layer (for the rate penalty)."""
        i_enc, bump = self.encode(x, st["prev"])
        st["prev"] = x[:, :NC].detach() if not torch.is_grad_enabled() else x[:, :NC]
        w_in, w_rec, w_att, w_ro = self.W("in"), self.W("rec"), self.W("att"), self.W("ro")
        acc = 0.0
        for _ in range(SUBSTEPS):
            v = BETA_ENC * st["v_enc"] + i_enc
            s_enc = spike(v - 1.0)
            st["v_enc"] = v - s_enc
            grf_spikes = s_enc[:, :NC * N_GRF]
            v = BETA_ZONE * st["v_zone"] + grf_spikes @ self.w_zone.t()
            s_zone = spike(v - 1.0 - st["a_zone"])
            st["v_zone"] = v - s_zone * (1.0 + st["a_zone"])
            st["a_zone"] = st["a_zone"] * self.adapt_keep + s_zone * self.adapt_step
            pre = torch.cat([s_enc, s_zone], 1)
            v = BETA_MIX * st["v_mix"] + pre @ w_in.t() + st["s_mix"] @ w_rec.t() + self.b_mix
            s_mix = spike(v - 1.0)
            st["v_mix"] = v - s_mix
            st["s_mix"] = s_mix
            v = BETA_ATT * st["v_att"] + bump + s_mix @ w_att.t() + self.b_att
            s_att = spike(v - 1.0)
            st["v_att"] = v - s_att
            acc = acc + torch.cat([s_enc, s_zone, s_mix, s_att], 1)
        rate = acc / SUBSTEPS
        st["ro"] = st["ro"] * self.ro_keep + (1 - self.ro_keep) * (rate @ w_ro.t() + self.b_ro)
        z = st["ro"]
        out = torch.where(self.signed, torch.tanh(z), torch.sigmoid(z))
        return out, rate

    def forward(self, X, st):
        """X [B, T, 18] -> Y [B, T, 32], mean rates per layer [4]."""
        outs, rates = [], []
        for t in range(X.shape[1]):
            o, r = self.step(X[:, t], st)
            outs.append(o)
            rates.append(r)
        R = torch.stack(rates, 1)
        layer = torch.stack([R[..., :N_ENC].mean(), R[..., N_ENC:N_ENC + N_ZONE].mean(),
                             R[..., N_ENC + N_ZONE:N_ENC + N_ZONE + N_MIX].mean(), R[..., -N_ATT:].mean()])
        return torch.stack(outs, 1), layer, R


def detach_state(st):
    return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in st.items()}


assert SNN_IN[:NC] == CHANNELS
