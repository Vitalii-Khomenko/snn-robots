"""Reinforcement stage of the policy GRU (after imitation + DAgger): PPO on closed-loop rollouts in the digital twin.

The reward (reward.py) says what moving, talking, being petted and being annoyed are worth; the policy finds out WHEN
to move the head, when to hold still and when to speak - the timing the stateless teacher cannot teach and that the
executor used to impose by hand (a glance at most every 8 s, a call budget). Only those outputs explore and are
shaped by the reward (RL_SIGMA); every output stays anchored to the teacher (imitation loss), the explored ones only
lightly, so faces, voice and colour are not lost.

  tools/hybrid/rl.sh ITERATIONS        (explore rollouts -> this -> export, repeated)
  ./py train_rl.py data/rl/it3          (one update from the rollouts in that folder)
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn

import reward as R
import teacher_policy as tp
from models import PolicyNet
from spec import ACTION, ACTION_SIGNED, POLICY_INPUT

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
N_IN, N_OUT = len(POLICY_INPUT), len(ACTION)

# the outputs that explore (noise std in their 0..1 / -1..1 range) and that the reward shapes
# Only the timing explores - when to hold still and when to speak; where to look stays the teacher's (noise on the head
# target made the head move by itself, and the policy learned to resist that noise instead of a rhythm)
RL_SIGMA = {"hold": 0.6, "sound_act": 0.25}  # hold idles near 0 and stops the head above 0.6: the noise must reach it
NOISE_BLOCK = 60  # ticks a noise sample is held (3 s: "wait a while", "speak now")
GAMMA, LAMBDA = 0.995, 0.95      # per 50 ms tick: rewards ~10 s ahead still count
CLIP, CHUNK = 0.2, 200
ANCHOR, ANCHOR_RL = 1.0, 0.02     # imitation weight on the other outputs / on the explored ones


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_IN, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)[..., 0]


def load_episodes(folder):
    xs, acts, rews = [], [], []
    for ep in sorted(glob.glob(os.path.join(folder, "ep*"))):
        bus, act = os.path.join(ep, "bus.bin"), os.path.join(ep, "act.bin")
        if not (os.path.exists(bus) and os.path.exists(act)):
            continue
        d = R.load(bus)
        a = np.fromfile(act, np.float32)
        a = a[: a.size // N_OUT * N_OUT].reshape(-1, N_OUT)
        T = min(len(d), len(a))
        if T < 400:
            continue
        xs.append(d[:T, :N_IN])
        acts.append(a[:T])
        rews.append(R.reward(d[:T]))
    T = min(len(x) for x in xs)
    return np.stack([x[:T] for x in xs]), np.stack([a[:T] for a in acts]), np.stack([r[:T] for r in rews])


def squash(o):
    sig = torch.tensor([n in ACTION_SIGNED for n in ACTION], device=o.device)
    return torch.where(sig, torch.tanh(o), torch.sigmoid(o))


def gae(rew, val):
    """rew, val: [B, T] numpy -> advantages, returns."""
    B, T = rew.shape
    adv = np.zeros_like(rew)
    last = np.zeros(B, np.float32)
    nxt = np.zeros(B, np.float32)
    for t in range(T - 1, -1, -1):
        delta = rew[:, t] + GAMMA * nxt - val[:, t]
        last = delta + GAMMA * LAMBDA * last
        adv[:, t] = last
        nxt = val[:, t]
    return adv, adv + val


def means(model, X, dev):
    """the policy's action means over whole episodes (no grad), and the hidden state at every chunk start."""
    model.eval()
    out, hs = [], []
    h = None
    with torch.no_grad():
        for c in range(0, X.shape[1], CHUNK):
            hs.append(h)
            o, h = model(X[:, c:c + CHUNK], h)
            out.append(squash(o))
    return torch.cat(out, 1), hs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--model", default=os.path.join(HERE, "data", "policy_net.pt"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    dev = torch.device("cuda")
    X, A, Rw = load_episodes(a.folder)
    B, T = X.shape[:2]
    print(f"{B} episodes x {T} ticks, mean reward per episode {Rw.sum(1).mean():+.2f}")
    labels = np.stack([tp.label(x) for x in X])
    Xt, At, Lt = (torch.as_tensor(v, device=dev) for v in (X, A, labels))
    rl_idx = torch.tensor([ACTION.index(n) for n in RL_SIGMA], device=dev)
    sig = torch.tensor(list(RL_SIGMA.values()), device=dev)
    model = PolicyNet().to(dev)
    model.load_state_dict(torch.load(a.model))
    model.quant = False  # float (see rollout.py): fake INT8 made tiny updates jump; the export quantizes
    vpath = os.path.join(HERE, "data", "value_net.pt")
    value = ValueNet().to(dev)
    if os.path.exists(vpath):
        value.load_state_dict(torch.load(vpath))
    # old policy (the one that acted) and the advantages
    old_mean, hs = means(model, Xt, dev)
    old_logp = (-0.5 * ((At[..., rl_idx] - old_mean[..., rl_idx]) / sig) ** 2).sum(-1)
    with torch.no_grad():
        val = value(Xt).cpu().numpy()
    adv, ret = gae(Rw, val)
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    Adv, Ret = torch.as_tensor(adv, device=dev), torch.as_tensor(ret, device=dev)
    w_anchor = torch.full((N_OUT,), ANCHOR, device=dev)
    w_anchor[rl_idx] = ANCHOR_RL
    from train_policy import loss_fn as imitation_loss, W as IW
    iw = torch.tensor([IW[n] for n in ACTION], device=dev) * w_anchor
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    vopt = torch.optim.Adam(value.parameters(), lr=1e-3)
    n_chunks = (T + CHUNK - 1) // CHUNK
    for ep in range(a.epochs):
        model.train()
        agg = dict(pg=0.0, imit=0.0, v=0.0, clipfrac=0.0)
        n = 0
        # a few minibatches of whole episodes per epoch, one optimizer step each (the gradient is accumulated chunk by
        # chunk over the episode)
        for mb in np.array_split(np.random.permutation(B), a.minibatches):
            mb = torch.as_tensor(mb, device=dev)
            opt.zero_grad()
            vopt.zero_grad()
            for k, c in enumerate(range(0, T, CHUNK)):
                sl = slice(c, c + CHUNK)
                h0 = None if hs[k] is None else hs[k][:, mb]
                x = Xt[mb, sl]
                logits, _ = model(x, h0)
                mean = squash(logits)
                logp = (-0.5 * ((At[mb, sl][..., rl_idx] - mean[..., rl_idx]) / sig) ** 2).sum(-1)
                ratio = torch.exp(logp - old_logp[mb, sl])
                adv_c = Adv[mb, sl]
                pg = -torch.min(ratio * adv_c, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv_c).mean()
                imit = imitation_loss(logits, Lt[mb, sl], iw, torch.ones_like(logits))
                ((pg + imit) / n_chunks).backward()
                vloss = ((value(x) - Ret[mb, sl]) ** 2).mean()
                (vloss / n_chunks).backward()
                agg["pg"] += pg.item()
                agg["imit"] += imit.item()
                agg["v"] += vloss.item()
                agg["clipfrac"] += ((ratio - 1).abs() > CLIP).float().mean().item()
                n += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            vopt.step()
        print(f"  epoch {ep + 1}: " + " ".join(f"{k} {v / n:.4f}" for k, v in agg.items()), flush=True)
    torch.save(model.state_dict(), os.path.join(HERE, "data", "policy_net.pt"))
    torch.save(value.state_dict(), vpath)
    model.cpu().export(os.path.join(ROOT, "main", "hybrid", "policy_model_data.h"))
    print("exported")
