"""Colab T4 GPU job (Stage 7): Batched GPU World-Model Simulator + Dreamer/MBPO Ensemble RL.

1. Trains a 3-model Stochastic Multi-Step World Model Ensemble on 147 hours of real life (world_rl_data.npz)
   using scheduled sampling (autoregressive rollout loss) + heteroscedastic Gaussian next-obs + human-event heads.
2. Runs a 512-parallel batched GPU virtual-life simulator (WorldEnsemble -> SnnV4 -> Muse -> NnV4 -> WorldEnsemble)
   and measures virtual-time throughput (simulated hours / second on Tesla T4).
3. Performs uncertainty-penalized MBPO/AWR policy optimization on NnV4's action heads (act1, act2) with KL anchor
   to innate heads, evaluates Innate vs Dreamed policy across 512 held-out virtual lives, and exports:
     /content/out/world_ensemble.pt
     /content/out/gpu_dream_heads.npz
     /content/out/gpu_sim_report.txt
"""
import os
import sys
import tarfile
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

t0 = time.time()
os.makedirs("/content/sim_work", exist_ok=True)
src_tar = "/content/sim_bundle.tar.gz" if os.path.exists("/content/sim_bundle.tar.gz") else "/content/bundle.tar.gz"
with tarfile.open(src_tar, "r:gz") as tf:
    tf.extractall("/content/sim_work")

os.chdir("/content/sim_work")
if "/content/sim_work" not in sys.path:
    sys.path.insert(0, "/content/sim_work")

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU World-Model Simulator on", dev, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

from nn_v4 import ACT_IDX, EMO_IDX, GROUPS, NnV4, activate
from snn_v4 import SnnV4, detach_state
from spec_v4 import CHANNELS, NN_OUT, PREV, SNN_OUT
import muse as MU

# ---- 1. Load 147 hours of real 2 Hz trajectories
z = np.load("/content/world_rl_data.npz", allow_pickle=True)
obs = z["obs"].astype(np.float32)       # [N, 17]: 13 channels + sin/cos(h) + head_yaw/60 + head_pitch/30
act = z["act"].astype(np.float32)       # [N, 10]: said, gest, 4 styles, head_look, head_scan, gy, gp
rew = z["rew"].astype(np.float32)       # [N]
done = z["done"].astype(bool)
day_idx = z["day_idx"]
days = [str(d) for d in z["days"]]

# Discretize human response event per 0.5 s step: [pos_touch(pet/tap), pos_social(talk/near), neg(shake/tilt/poke/left)]
ev_pos_touch = (rew >= 0.45).astype(np.float32)
ev_pos_any = (rew > 0.15).astype(np.float32)
ev_neg = (rew < -0.15).astype(np.float32)
ev_mat = np.stack([ev_pos_touch, ev_pos_any, ev_neg], axis=-1).astype(np.float32)

SEQ = 32  # 16 s windows at 2 Hz for multi-step world-model training
starts = np.array([i for i in range(0, len(obs) - SEQ - 1, SEQ // 2) if not done[i:i + SEQ - 1].any()], np.int64)
held_day = int(day_idx.max())
val_m = day_idx[starts] == held_day
tr_s, val_s = starts[~val_m], starts[val_m]
print(f"Real-life windows: {len(tr_s)} train, {len(val_s)} val (held-out: {days[held_day]})", flush=True)


class StochWorldModel(nn.Module):
    """Stochastic GRU-128 world model predicting next 13 sensory channels (mean + logstd) & 3 human-event probs."""
    def __init__(self, d_obs=17, d_act=10, hid=128):
        super().__init__()
        self.gru = nn.GRU(d_obs + d_act, hid, batch_first=True)
        self.mu = nn.Linear(hid, 13)
        self.logstd = nn.Linear(hid, 13)
        self.ev = nn.Linear(hid, 3)

    def step(self, o_t, a_t, h=None):
        """Single 0.5 s step: o_t [B, 17], a_t [B, 10] -> mu [B,13], std [B,13], ev_logits [B,3], h_next."""
        x = torch.cat([o_t, a_t], -1).unsqueeze(1)
        out, h_next = self.gru(x, h)
        z = out[:, 0]
        mu = torch.sigmoid(self.mu(z))
        std = torch.exp(self.logstd(z).clamp(-4.5, -1.0))
        ev = self.ev(z)
        return mu, std, ev, h_next


def batch_win(idx):
    ii = idx[:, None] + np.arange(SEQ)[None, :]
    return (torch.from_numpy(obs[ii]).to(dev),
            torch.from_numpy(act[ii]).to(dev),
            torch.from_numpy(obs[ii + 1]).to(dev),
            torch.from_numpy(ev_mat[ii]).to(dev))


# ---- 2. Load pre-trained 3-model Stochastic World Model Ensemble if provided, else train with Scheduled Sampling
N_ENS = 3
ensemble = [StochWorldModel().to(dev) for _ in range(N_ENS)]
ev_pw = torch.tensor([15.0, 10.0, 12.0], device=dev)
lines = []
t_wm = time.time()

pre_ens = "/content/world_ensemble.pt" if os.path.exists("/content/world_ensemble.pt") else None
if pre_ens:
    ckpt = torch.load(pre_ens, weights_only=False, map_location=dev)
    for k, wm in enumerate(ensemble):
        wm.load_state_dict(ckpt["ensemble"][k])
        wm.eval()
        with torch.no_grad():
            errs = []
            for i in range(0, len(val_s), 512):
                o_seq, a_seq, o_next, _ = batch_win(val_s[i:i + 512])
                h = None
                cur_o = o_seq[:, 0]
                for t in range(16):
                    mu, _, _, h = wm.step(cur_o, a_seq[:, t], h)
                    errs.append(((mu - o_next[:, t, :13]) ** 2).mean(-1).cpu().numpy())
                    if t + 1 < 16:
                        cur_o = torch.cat([mu, o_seq[:, t + 1, 13:]], -1)
            ar_rmse = float(np.sqrt(np.mean(np.concatenate(errs))))
        msg = f"WorldModel[{k}] loaded (14 ep): held-out 8s autoregressive RMSE = {ar_rmse:.4f}"
        print(msg, flush=True)
        lines.append(msg)
else:
    for k, wm in enumerate(ensemble):
        torch.manual_seed(100 + k)
        rng = np.random.default_rng(100 + k)
        opt = torch.optim.AdamW(wm.parameters(), lr=2.5e-3, weight_decay=1e-4)
        EPOCHS = 14
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        BS = 256
        for ep in range(EPOCHS):
            wm.train()
            # Scheduled sampling probability ramps 0.0 -> 0.55 so autoregressive rollouts stay calibrated
            ss_prob = min(0.55, max(0.0, (ep - 2) / 10.0 * 0.55))
            perm = rng.permutation(tr_s)
            tot, nb = 0.0, 0
            for i in range(0, len(perm), BS):
                o_seq, a_seq, o_next, ev_t = batch_win(perm[i:i + BS])
                B = o_seq.shape[0]
                h = None
                cur_o = o_seq[:, 0]
                loss_nll = 0.0
                loss_ev = 0.0
                for t in range(SEQ):
                    mu, std, ev_l, h = wm.step(cur_o, a_seq[:, t], h)
                    tgt = o_next[:, t, :13]
                    nll = 0.5 * (((tgt - mu) / std) ** 2 + 2.0 * torch.log(std)).mean()
                    loss_nll = loss_nll + nll
                    loss_ev = loss_ev + F.binary_cross_entropy_with_logits(ev_l, ev_t[:, t], pos_weight=ev_pw)
                    if t + 1 < SEQ:
                        use_pred = (torch.rand(B, 1, device=dev) < ss_prob)
                        next_ch = torch.where(use_pred, mu.detach(), o_seq[:, t + 1, :13])
                        cur_o = torch.cat([next_ch, o_seq[:, t + 1, 13:]], -1)
                loss = (loss_nll + 0.5 * loss_ev) / SEQ
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(wm.parameters(), 2.0)
                opt.step()
                tot += float(loss.detach())
                nb += 1
            sched.step()
        # Evaluate 16-step (8 s) free-running autoregressive RMSE on held-out day
        wm.eval()
        with torch.no_grad():
            errs = []
            for i in range(0, len(val_s), 512):
                o_seq, a_seq, o_next, _ = batch_win(val_s[i:i + 512])
                h = None
                cur_o = o_seq[:, 0]
                for t in range(16):
                    mu, _, _, h = wm.step(cur_o, a_seq[:, t], h)
                    errs.append(((mu - o_next[:, t, :13]) ** 2).mean(-1).cpu().numpy())
                    if t + 1 < 16:
                        cur_o = torch.cat([mu, o_seq[:, t + 1, 13:]], -1)
            ar_rmse = float(np.sqrt(np.mean(np.concatenate(errs))))
        msg = f"WorldModel[{k}] trained ({EPOCHS} ep): held-out 8s autoregressive RMSE = {ar_rmse:.4f}"
        print(msg, flush=True)
        lines.append(msg)

wm_s = time.time() - t_wm
lines.append(f"3-model stochastic ensemble ready in {wm_s:.1f}s")
os.makedirs("/content/out", exist_ok=True)
torch.save({"ensemble": [m.state_dict() for m in ensemble], "days": days}, "/content/out/world_ensemble.pt")

# ---- 3. Load freshly trained SnnV4 + NnV4 + Muse for 512-parallel GPU simulation
snn = SnnV4().to(dev)
snn.load_state_dict(torch.load("/content/sim_work/data/snn_v4.pt", weights_only=False, map_location=dev)["state"])
snn.quant = True
snn.eval()

nn_ckpt = "/content/out/nn_v4.pt" if os.path.exists("/content/out/nn_v4.pt") else "/content/sim_work/data/nn_v4.pt"
nn_innate = NnV4().to(dev)
nn_innate.load_state_dict(torch.load(nn_ckpt, weights_only=False, map_location=dev)["state"])
nn_innate.eval()

# Batched PyTorch Muse helper on GPU (exact match to tools/v4/muse.py)
mz = np.load("/content/sim_work/data/muse_v4.npz")
W_rec_m = torch.tensor(mz["W"], dtype=torch.float32, device=dev)
W_in_m = torch.tensor(mz["Win"], dtype=torch.float32, device=dev)
b_m = torch.tensor(mz["b"], dtype=torch.float32, device=dev)
W_out_m = torch.tensor(mz["Wout"], dtype=torch.float32, device=dev)
W_u_m = torch.tensor(mz["Wu"], dtype=torch.float32, device=dev)
b_out_m = torch.tensor(mz["bout"], dtype=torch.float32, device=dev)
alpha_m = torch.tensor(MU.A, dtype=torch.float32, device=dev)
noise_sd_m = torch.tensor(MU.NOISE, dtype=torch.float32, device=dev)
DRIVE_IDX_M = torch.tensor(MU.DRIVE_IDX, dtype=torch.long, device=dev)
HAB_RATE_M = float(MU.DT / MU.TAU_HAB)
HAB_FB_M = float(MU.HAB_FEEDBACK)
HAB_CUT_M = float(MU.HAB_CUT)

EMO_VAL = torch.tensor([0.55, 0.80, 0.90, 0.90, 0.20, 0.10, 0.45, 0.55, 0.10, 0.25, 0.95,
                        0.70, 0.80, 0.65, 0.45, 0.55, 0.35, 0.10, 0.35, 0.70, 0.20], device=dev)
EMO_ARO = torch.tensor([0.30, 0.55, 0.80, 0.85, 0.25, 0.50, 0.05, 0.85, 0.90, 0.65, 0.45,
                        0.35, 0.55, 0.55, 0.50, 0.35, 0.15, 0.90, 0.70, 0.35, 0.15], device=dev)
SAY_I = NN_OUT.index("say")
STYLE_I = [NN_OUT.index("style_" + s) for s in ("calm", "ask", "play", "grumble")]
HEAD_I = GROUPS["head"]
HEAD_LOOK, HEAD_SCAN = NN_OUT.index("head_look"), NN_OUT.index("head_scan")
HEAD_GEST = [NN_OUT.index("head_nod"), NN_OUT.index("head_shake"), NN_OUT.index("head_flinch")]
GY_I, GP_I, LED_I = NN_OUT.index("gaze_yaw"), NN_OUT.index("gaze_pitch"), NN_OUT.index("led_bright")
ACT_LIST = [i for i in range(len(NN_OUT)) if i not in EMO_IDX and i != NN_OUT.index("emo_strength")]
GROUPS_ACT = {g: [ACT_LIST.index(i) for i in idx] for g, idx in GROUPS.items()}
SAY_ACT = ACT_LIST.index(SAY_I)


@torch.no_grad()
def rollout_gpu(policy, seed_idx, steps=60, seed=0, stoch=True):
    """Run B = len(seed_idx) parallel virtual lives on Tesla T4 (steps x 0.5 s = 30 s per life).
    Returns dict of features, actions, rewards, uncertainties, and throughput metrics."""
    torch.manual_seed(seed)
    B = len(seed_idx)
    cur_o = torch.from_numpy(obs[seed_idx]).to(dev)  # [B, 17]
    h_wm = [None] * N_ENS
    snn_st = snn.init_state(B, dev)
    h_nn = None
    muse_x = torch.zeros(B, 64, device=dev)
    muse_hab = torch.zeros(B, 5, device=dev)
    # Batched body/prev state on GPU
    emo = torch.zeros(B, dtype=torch.long, device=dev)
    emo_age = torch.zeros(B, device=dev)
    since_snd = torch.full((B,), 10.0, device=dev)
    since_gst = torch.full((B,), 30.0, device=dev)
    head_y = cur_o[:, 15].clone()
    head_p = cur_o[:, 16].clone()
    led_lvl = torch.full((B,), 0.3, device=dev)
    touch_open = torch.zeros(B, dtype=torch.bool, device=dev)
    prev_touch = torch.zeros(B, device=dev)
    agency = torch.zeros(B, device=dev)
    answered = torch.zeros(B, device=dev)
    ignored = torch.zeros(B, device=dev)
    call_budget = torch.ones(B, device=dev)

    feats_l, rew_l, unc_l, raw_act_l = [], [], [], []
    act_choices = {g: [] for g in GROUPS_ACT}
    act_say = []
    tot_calls = 0

    t_sim = time.time()
    for t in range(steps):
        # 1. Step SnnV4 (2 ticks of 50 ms per 0.5 s macro-step for high GPU throughput)
        ch18 = torch.cat([cur_o[:, :13], torch.zeros(B, 5, device=dev)], -1)
        so, _ = snn.step(ch18, snn_st)
        so, _ = snn.step(ch18, snn_st)  # [B, 32]

        # 2. Step batched Muse on GPU (exact match to muse.py + habituation feedback)
        u_m = so[:, DRIVE_IDX_M] - 0.5
        hab_fb = (muse_hab @ W_out_m[3:]) * HAB_FB_M
        pre = muse_x @ W_rec_m.T + u_m @ W_in_m.T + b_m - hab_fb
        muse_x = torch.clamp((1.0 - alpha_m) * muse_x + alpha_m * torch.tanh(pre)
                             + noise_sd_m * torch.randn_like(muse_x), -3.0, 3.0)
        m_y = torch.tanh(muse_x @ W_out_m.T + u_m @ W_u_m.T + b_out_m)
        m_idea = m_y[:, 3:]
        muse_hab = muse_hab + (torch.clamp(m_idea, min=0.0) - muse_hab) * HAB_RATE_M
        m_idea_out = torch.where(m_idea > 0.0, m_idea * (1.0 - HAB_CUT_M * muse_hab), m_idea)
        muse_8 = torch.cat([m_y[:, :3], m_idea_out], -1)

        # 3. Assemble 56-input vector [32 SNN + 24 PREV]
        touch_now = torch.maximum(so[:, SNN_OUT.index("sal_touch_head")], so[:, SNN_OUT.index("sal_touch_screen")])
        touch_open = touch_open | ((touch_now > 0.12) & ((touch_now - prev_touch) > 0.04))
        prev_touch = touch_now

        prev24 = torch.stack([
            EMO_VAL[emo], EMO_ARO[emo], (emo_age / 20.0).clamp(0, 1),
            (since_snd / 30.0).clamp(0, 1), (since_gst / 30.0).clamp(0, 1),
            head_y.clamp(-1, 1), head_p.clamp(-1, 1), led_lvl,
            (~touch_open).float(),
            muse_8[:, 0], muse_8[:, 1], muse_8[:, 2],
            muse_8[:, 3], muse_8[:, 4], muse_8[:, 5], muse_8[:, 6], muse_8[:, 7],
            agency, answered, ignored, agency * 0.8, agency * 0.5, agency * 0.3,
            torch.zeros(B, device=dev),  # expectation
        ], dim=-1)
        x56 = torch.cat([so, prev24], -1)

        # 4. Step NnV4 (GRU + emotion + action heads)
        z_gru, h_nn = policy.gru(x56.unsqueeze(1), h_nn)
        e_out = policy.emo(z_gru)
        p_emo = F.softmax(e_out[..., :len(EMO_IDX)], -1)
        feat = torch.cat([z_gru[:, 0], p_emo[:, 0]], -1)  # [B, 85]
        a_logits = policy.act2(torch.tanh(policy.act1(feat)))  # [B, 39]

        # Reconstruct full 61-dim output for executor/action extraction
        raw61 = torch.zeros(B, len(NN_OUT), device=dev)
        raw61[:, EMO_IDX] = e_out[:, 0, :len(EMO_IDX)]
        raw61[:, NN_OUT.index("emo_strength")] = e_out[:, 0, len(EMO_IDX)]
        raw61[:, ACT_LIST] = a_logits
        act61 = activate(raw61)

        # Sample or argmax choices (stochastic during dream RL exploration!)
        if stoch:
            say_p = torch.sigmoid(a_logits[:, SAY_ACT]).clamp(1e-4, 1 - 1e-4)
            said_wish = (torch.rand(B, device=dev) < say_p)
            ch_g = {g: torch.multinomial(F.softmax(a_logits[:, idx], -1), 1)[:, 0] for g, idx in GROUPS_ACT.items()}
        else:
            said_wish = act61[:, SAY_I] > 0.55
            ch_g = {g: a_logits[:, idx].argmax(-1) for g, idx in GROUPS_ACT.items()}

        # Executor call budget & refractory
        said = said_wish & (since_snd >= 1.5) & (call_budget >= 0.25)
        call_budget = (call_budget + 0.04 - said.float() * 0.35).clamp(0.0, 1.0)
        tot_calls += int(said.sum())
        head_pick = ch_g["head"]
        gest = ((head_pick == 3) | (head_pick == 4) | (head_pick == 5)) & (since_gst >= 2.0)

        # Build 10-dim action vector for WorldModelEnsemble
        style_oh = F.one_hot(ch_g["style"], 4).float()
        h_look = (head_pick == 1).float()
        h_scan = (head_pick == 2).float()
        gy = torch.tanh(raw61[:, GY_I])
        gp = torch.tanh(raw61[:, GP_I])
        a10 = torch.cat([said.float()[:, None], gest.float()[:, None], style_oh,
                         h_look[:, None], h_scan[:, None], gy[:, None], gp[:, None]], dim=-1)

        # 5. Step 3-model Stochastic WorldModel Ensemble on GPU
        mus, stds, evs = [], [], []
        for k, wm in enumerate(ensemble):
            mu_k, sd_k, ev_k, h_wm[k] = wm.step(cur_o, a10, h_wm[k])
            mus.append(mu_k)
            stds.append(sd_k)
            evs.append(torch.sigmoid(ev_k))
        mus_t = torch.stack(mus, 0)       # [3, B, 13]
        mu_mean = mus_t.mean(0)
        epistemic = mus_t.std(0).mean(-1)  # [B] disagreement across ensemble members
        ev_mean = torch.stack(evs, 0).mean(0)  # [B, 3]: pos_touch, pos_any, neg

        # Sample next world state from a random ensemble member (PETS trajectory sampling)
        pick_m = int(torch.randint(0, N_ENS, (1,)).item())
        next_ch = (mus[pick_m] + (torch.randn_like(mu_mean) * stds[pick_m] if stoch else 0.0)).clamp(0.0, 1.0)

        # Compute virtual reward + uncertainty penalty (MBPO)
        r_step = 0.8 * ev_mean[:, 0] + 0.4 * ev_mean[:, 1] - 0.6 * ev_mean[:, 2] - 2.0 * epistemic
        # Penalize excessive spamming when call budget is low
        r_step = r_step - 0.05 * (said_wish & ~said).float()

        # Update body/contingency/prev states
        new_emo = p_emo[:, 0].argmax(-1)
        emo_age = torch.where(new_emo == emo, emo_age + 0.5, torch.zeros_like(emo_age))
        emo = new_emo
        since_snd = torch.where(said, torch.zeros_like(since_snd), since_snd + 0.5)
        since_gst = torch.where(gest, torch.zeros_like(since_gst), since_gst + 0.5)
        touch_open = touch_open & ~(said | gest)
        head_y = torch.where((head_pick == 1) | (head_pick == 2), 0.7 * head_y + 0.3 * gy, head_y)
        head_p = torch.where((head_pick == 1) | (head_pick == 2), 0.7 * head_p + 0.3 * gp, head_p)
        led_lvl = act61[:, LED_I]
        ans_now = (ev_mean[:, 1] > 0.20).float()
        answered = torch.clamp(answered * 0.75 + ans_now * 0.5, 0.0, 1.0)
        agency = torch.clamp(agency * 0.98 + ans_now * (said | gest).float() * 0.08, 0.0, 1.0)
        ignored = torch.where(ans_now > 0.5, torch.zeros_like(ignored),
                              torch.clamp(ignored * 0.98 + (said & (ch_g["style"] == 1)).float() * 0.08, 0.0, 1.0))

        cur_o = torch.cat([next_ch, cur_o[:, 13:15], head_y[:, None], head_p[:, None]], -1)

        feats_l.append(feat)
        rew_l.append(r_step)
        unc_l.append(epistemic)
        act_say.append(said_wish.float())
        for g in GROUPS_ACT:
            act_choices[g].append(ch_g[g])

    sim_s = time.time() - t_sim
    sim_hours = (B * steps * 0.5) / 3600.0
    R = torch.stack(rew_l, 1)  # [B, T]
    # Discounted returns (gamma = 0.97 at 2 Hz -> ~16 s horizon)
    G = torch.zeros_like(R)
    acc = torch.zeros(B, device=dev)
    for t in range(steps - 1, -1, -1):
        acc = R[:, t] + 0.97 * acc
        G[:, t] = acc
    return dict(
        feats=torch.stack(feats_l, 1).reshape(-1, 85),
        G=G.reshape(-1),
        mean_ret=float(R.sum(1).mean()),
        mean_unc=float(torch.stack(unc_l, 1).mean()),
        calls_per_h=tot_calls / max(sim_hours, 1e-6),
        sim_s=sim_s,
        sim_hours=sim_hours,
        speed_x=(B * steps * 0.5) / max(sim_s, 1e-6),
        act_say=torch.stack(act_say, 1).reshape(-1),
        act_g={g: torch.stack(act_choices[g], 1).reshape(-1) for g in GROUPS_ACT},
    )


# ---- 4. Benchmark 512-parallel GPU virtual-life simulation + Dreamed MBPO/AWR policy optimization
rng_sim = np.random.default_rng(7)
B_SIM = 512
eval_seeds = rng_sim.choice(val_s, size=min(B_SIM, len(val_s)), replace=len(val_s) < B_SIM)
train_seeds = rng_sim.choice(tr_s, size=B_SIM, replace=False)

base_eval = rollout_gpu(nn_innate, eval_seeds, steps=60, seed=42, stoch=False)
msg = (f"GPU Sim Benchmark ({B_SIM} parallel lives x 30 s = {base_eval['sim_hours']:.2f} virt-h in {base_eval['sim_s']:.2f}s "
       f"= {base_eval['speed_x']:.0f}x real-time): Innate ret={base_eval['mean_ret']:+.4f}, "
       f"unc={base_eval['mean_unc']:.4f}, calls/h={base_eval['calls_per_h']:.0f}")
print(msg, flush=True)
lines.append(msg)

# Train dreamed action heads via KL-anchored AWR inside 3 rounds x 512 parallel GPU virtual lives
dreamed = NnV4().to(dev)
dreamed.load_state_dict(nn_innate.state_dict())
for p in dreamed.parameters():
    p.requires_grad_(False)
head_params = list(dreamed.act1.parameters()) + list(dreamed.act2.parameters())
for p in head_params:
    p.requires_grad_(True)
opt_h = torch.optim.Adam(head_params, lr=5e-4)

for rnd in range(3):
    seeds_r = rng_sim.choice(tr_s, size=B_SIM, replace=False)
    ro = rollout_gpu(dreamed, seeds_r, steps=60, seed=1000 + rnd, stoch=True)
    F_all = ro["feats"]
    G_all = ro["G"]
    adv = (G_all - G_all.mean()) / (G_all.std() + 1e-6)
    adv = adv.clamp(-2.5, 2.5)
    with torch.no_grad():
        old_l = nn_innate.act2(torch.tanh(nn_innate.act1(F_all)))
        old_p = {g: F.softmax(old_l[:, idx], -1) for g, idx in GROUPS_ACT.items()}
        old_say = torch.sigmoid(old_l[:, SAY_ACT])
    # AWR weighted likelihood + strong KL anchor (3.0) to innate heads so safety & reflexes stay intact
    w_awr = torch.exp(adv / 2.5).clamp(0.2, 4.0)
    other = [i for i in range(len(ACT_LIST)) if i != SAY_ACT and not any(i in v for v in GROUPS_ACT.values())]
    N_f = len(F_all)
    for _ in range(8):
        perm = torch.randperm(N_f, device=dev)
        for i in range(0, N_f, 2048):
            b = perm[i:i + 2048]
            out = dreamed.act2(torch.tanh(dreamed.act1(F_all[b])))
            loss = 0.0
            for g, idx in GROUPS_ACT.items():
                logp = F.log_softmax(out[:, idx], -1)
                nll = F.nll_loss(logp, ro["act_g"][g][b], reduction="none")
                kl = F.kl_div(logp, old_p[g][b], reduction="batchmean")
                loss = loss + (nll * w_awr[b]).mean() + 3.0 * kl
            bce = F.binary_cross_entropy_with_logits(out[:, SAY_ACT], ro["act_say"][b], reduction="none")
            kl_s = F.binary_cross_entropy_with_logits(out[:, SAY_ACT], old_say[b])
            loss = loss + (bce * w_awr[b]).mean() + 3.0 * kl_s
            loss = loss + 3.0 * F.mse_loss(out[:, other], old_l[b][:, other])
            opt_h.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 1.0)
            opt_h.step()
    ev_r = rollout_gpu(dreamed, eval_seeds, steps=60, seed=42, stoch=False)
    # Measure KL & choice flip vs innate on eval features
    with torch.no_grad():
        lo = nn_innate.act2(torch.tanh(nn_innate.act1(ev_r["feats"])))
        ln = dreamed.act2(torch.tanh(dreamed.act1(ev_r["feats"])))
        kls, flips = [], []
        for g, idx in GROUPS_ACT.items():
            po, pn = F.softmax(lo[:, idx], -1), F.softmax(ln[:, idx], -1)
            kls.append(float((po * (po.clamp_min(1e-7).log() - pn.clamp_min(1e-7).log())).sum(-1).mean()))
            flips.append(float((po.argmax(-1) != pn.argmax(-1)).float().mean()))
    msg = (f"Dream RL round {rnd+1}/3: eval ret={ev_r['mean_ret']:+.4f} (delta {ev_r['mean_ret'] - base_eval['mean_ret']:+.4f}), "
           f"unc={ev_r['mean_unc']:.4f}, calls/h={ev_r['calls_per_h']:.0f}, KL={np.mean(kls):.4f}, flip={100*np.mean(flips):.1f}%")
    print(msg, flush=True)
    lines.append(msg)

os.makedirs("/content/out", exist_ok=True)
torch.save({"ensemble": [m.state_dict() for m in ensemble], "days": days}, "/content/out/world_ensemble.pt")
np.savez_compressed(
    "/content/out/gpu_dream_heads.npz",
    act1_w=dreamed.act1.weight.detach().cpu().numpy(),
    act1_b=dreamed.act1.bias.detach().cpu().numpy(),
    act2_w=dreamed.act2.weight.detach().cpu().numpy(),
    act2_b=dreamed.act2.bias.detach().cpu().numpy(),
)
lines.append(f"Total GPU World-Model Sim + Dream RL time: {(time.time() - t0) / 60:.1f} min")
with open("/content/out/gpu_sim_report.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("=== GPU SIM + DREAM RL DONE ===", flush=True)
