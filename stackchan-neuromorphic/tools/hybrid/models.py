"""Network definitions shared by training and export."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn import FakeQuant

VISION_OUT = ["motion_l", "motion_c", "motion_r", "motion_up", "motion_down", "person", "face", "loom", "recede",
              "target_x", "target_y", "target_conf", "activity"]
VISION_SIGNED = {"target_x", "target_y"}


class QuantMixin:
    quant = False

    def qw(self, w):
        return FakeQuant.apply(w) if self.quant else w


class VisionNet(nn.Module, QuantMixin):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 24, 3, padding=1)
        self.fc1 = nn.Linear(24 * 3 * 4 + 4, 64)
        self.fc2 = nn.Linear(64, len(VISION_OUT))

    def forward(self, img, side):
        x = F.relu(F.conv2d(img, self.qw(self.conv1.weight), self.conv1.bias, padding=1))
        x = F.max_pool2d(x, 2)
        x = F.relu(F.conv2d(x, self.qw(self.conv2.weight), self.conv2.bias, padding=1))
        x = F.max_pool2d(x, 2)
        x = F.relu(F.conv2d(x, self.qw(self.conv3.weight), self.conv3.bias, padding=1))
        x = F.max_pool2d(x, 2).flatten(1)
        x = F.relu(F.linear(torch.cat([x, side], 1), self.qw(self.fc1.weight), self.fc1.bias))
        return F.linear(x, self.qw(self.fc2.weight), self.fc2.bias)

    def export(self, path, trained=True):
        from nn import Header
        h = Header("vision_model", trained)
        for n in ("conv1", "conv2", "conv3", "fc1", "fc2"):
            m = getattr(self, n)
            h.layer(n, m.weight, m.bias)
        h.write(path)


AUDIO_OUT = ["silence", "speech", "clap", "sharp", "continuous", "low", "high", "tonal"]
AUDIO_IN = 16 * 6 + 4


class AuditoryNet(nn.Module, QuantMixin):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(AUDIO_IN, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, len(AUDIO_OUT))

    def forward(self, x):
        x = F.relu(F.linear(x, self.qw(self.fc1.weight), self.fc1.bias))
        x = F.relu(F.linear(x, self.qw(self.fc2.weight), self.fc2.bias))
        return F.linear(x, self.qw(self.fc3.weight), self.fc3.bias)

    def export(self, path, trained=True):
        from nn import Header
        h = Header("auditory_model", trained)
        for n in ("fc1", "fc2", "fc3"):
            m = getattr(self, n)
            h.layer(n, m.weight, m.bias)
        h.write(path)


class PolicyNet(nn.Module, QuantMixin):
    """100 -> dense 64 -> GRU 64 -> dense 48 -> 51 actions (pre-activation)."""

    def __init__(self, n_in=None, n_out=None):
        from spec import ACTION, POLICY_INPUT
        n_in = n_in or len(POLICY_INPUT)
        n_out = n_out or len(ACTION)
        super().__init__()
        self.inp = nn.Linear(n_in, 64)
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 48)
        self.fc2 = nn.Linear(48, n_out)

    def forward(self, x, h0=None):
        d = F.relu(F.linear(x, self.qw(self.inp.weight), self.inp.bias))
        if h0 is None:
            h0 = d.new_zeros(1, d.shape[0], 64)
        if self.quant:
            out, h = torch._VF.gru(d, h0, [self.qw(self.gru.weight_ih_l0), self.qw(self.gru.weight_hh_l0),
                                           self.gru.bias_ih_l0, self.gru.bias_hh_l0], True, 1, 0.0, self.training,
                                   False, True)
        else:
            out, h = self.gru(d, h0)
        o = F.relu(F.linear(out, self.qw(self.fc1.weight), self.fc1.bias))
        return F.linear(o, self.qw(self.fc2.weight), self.fc2.bias), h

    def export(self, path, trained=True):
        from nn import Header
        h = Header("policy_model", trained)
        h.layer("inp", self.inp.weight, self.inp.bias)
        h.layer("gru_i", self.gru.weight_ih_l0, self.gru.bias_ih_l0)
        h.layer("gru_h", self.gru.weight_hh_l0, self.gru.bias_hh_l0)
        h.layer("fc1", self.fc1.weight, self.fc1.bias)
        h.layer("fc2", self.fc2.weight, self.fc2.bias)
        h.write(path)
