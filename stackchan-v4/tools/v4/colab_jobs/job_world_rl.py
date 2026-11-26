"""Colab T4 GPU job (Stage 5):
  1. Train GRU-128 Neural World Model on all 34 days of real life (predicting next 0.5 s sensory channels
     + probability of positive human response in the next 15 s)
  2. Train Multi-Horizon Offline IQL Critic V(s) & Q(s,a) (30 s horizon, gamma=0.985 at 2 Hz)
  3. Report held-out world-model RMSE, human-response AUROC, and action-advantage insights across styles/calls/gestures.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

t0 = time.time()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("World Model + IQL on", dev, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

z = np.load("/content/world_rl_data.npz", allow_pickle=True)
obs = z["obs"].astype(np.float32)
act = z["act"].astype(np.float32)
rew = z["rew"].astype(np.float32)
done = z["done"].astype(bool)
day_idx = z["day_idx"]
days = [str(d) for d in z["days"]]

# Compute 15 s future positive response target (30 steps at 2 Hz) & 30 s discounted return (gamma=0.985)
N = len(obs)
pos15 = np.zeros(N, np.float32)
ret30 = np.zeros(N, np.float32)
g = 0.0
fut_pos = 0
for i in range(N - 1, -1, -1):
    if done[i]:
        g = 0.0
        fut_pos = 0
    g = rew[i] + 0.985 * g
    ret30[i] = g
    if rew[i] > 0.2:
        fut_pos = 30
    else:
        fut_pos = max(0, fut_pos - 1)
    pos15[i] = 1.0 if fut_pos > 0 else 0.0

# Slice into clean 64-step (32 s) windows without crossing done boundaries
SEQ = 64
starts = []
for i in range(0, N - SEQ - 1, SEQ // 2):
    if not done[i:i + SEQ - 1].any():
        starts.append(i)
starts = np.array(starts, np.int64)
held_day = int(day_idx.max())
val_m = day_idx[starts] == held_day
tr_s, val_s = starts[~val_m], starts[val_m]
print(f"Windows: {len(tr_s)} train, {len(val_s)} val (held-out day: {days[held_day]}), pos15 rate={pos15.mean():.4f}", flush=True)


class WorldModel(nn.Module):
    def __init__(self, d_obs=17, d_act=10, hid=128):
        super().__init__()
        self.gru = nn.GRU(d_obs + d_act, hid, batch_first=True)
        self.next_obs = nn.Linear(hid, 13)   # predict next 13 sensory channels
        self.resp_logit = nn.Linear(hid, 1)  # P(positive human response within 15 s)
        self.val_head = nn.Linear(hid, 1)    # V(s) 30 s return critic
        self.q_head = nn.Sequential(nn.Linear(hid + d_act, 64), nn.ReLU(), nn.Linear(64, 1))  # Q(s, a)

    def forward(self, o, a):
        h, _ = self.gru(torch.cat([o, a], -1))
        no = torch.sigmoid(self.next_obs(h))
        rl = self.resp_logit(h).squeeze(-1)
        v = self.val_head(h).squeeze(-1)
        q = self.q_head(torch.cat([h, a], -1)).squeeze(-1)
        return no, rl, v, q


def batch_windows(idx):
    ii = idx[:, None] + np.arange(SEQ)[None, :]
    o = torch.from_numpy(obs[ii]).to(dev)
    a = torch.from_numpy(act[ii]).to(dev)
    no_t = torch.from_numpy(obs[ii + 1, :13]).to(dev)
    p15_t = torch.from_numpy(pos15[ii]).to(dev)
    r30_t = torch.from_numpy(ret30[ii]).to(dev)
    return o, a, no_t, p15_t, r30_t


def auroc(y_true, y_score):
    y_true = y_true.astype(bool)
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    return float((ranks[y_true].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


model = WorldModel().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
EPOCHS = 30
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
rng = np.random.default_rng(0)
BS = 256
best_loss = 1e9
best_state = None
lines = []
pos_weight = torch.tensor([min(15.0, float((1.0 - pos15.mean()) / max(pos15.mean(), 1e-4)))], device=dev)

for ep in range(EPOCHS):
    model.train()
    perm = rng.permutation(tr_s)
    tot, nb = 0.0, 0
    for i in range(0, len(perm), BS):
        o, a, no_t, p15_t, r30_t = batch_windows(perm[i:i + BS])
        no, rl, v, q = model(o, a)
        loss_obs = F.mse_loss(no, no_t)
        loss_resp = F.binary_cross_entropy_with_logits(rl, p15_t, pos_weight=pos_weight)
        # IQL expectile loss (tau=0.7) for V(s) and MSE for Q(s, a)
        diff = r30_t - v
        w_exp = torch.where(diff > 0, 0.7, 0.3)
        loss_v = (w_exp * diff.pow(2)).mean()
        loss_q = F.mse_loss(q, r30_t)
        loss = 5.0 * loss_obs + 0.5 * loss_resp + loss_v + loss_q
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        tot += float(loss)
        nb += 1
    sched.step()
    model.eval()
    with torch.no_grad():
        obs_errs, r_probs, r_trues, v_errs, adv_said, adv_gest, adv_styles = [], [], [], [], [], [], [[] for _ in range(4)]
        for i in range(0, len(val_s), BS):
            o, a, no_t, p15_t, r30_t = batch_windows(val_s[i:i + BS])
            no, rl, v, q = model(o, a)
            obs_errs.append(F.mse_loss(no, no_t, reduction="none").mean(-1).cpu().numpy())
            r_probs.append(torch.sigmoid(rl).cpu().numpy().ravel())
            r_trues.append(p15_t.cpu().numpy().ravel())
            v_errs.append((q - r30_t).abs().cpu().numpy().ravel())
            adv = (q - v).cpu().numpy()
            an = a.cpu().numpy()
            if (an[..., 0] > 0.5).any():
                adv_said.append(adv[an[..., 0] > 0.5])
            if (an[..., 1] > 0.5).any():
                adv_gest.append(adv[an[..., 1] > 0.5])
            st = np.argmax(an[..., 2:6], axis=-1)
            for k in range(4):
                m_k = (an[..., 0] > 0.5) & (st == k)
                if m_k.any():
                    adv_styles[k].append(adv[m_k])
        rmse = float(np.sqrt(np.mean(np.concatenate(obs_errs))))
        auc = auroc(np.concatenate(r_trues), np.concatenate(r_probs))
        q_mae = float(np.mean(np.concatenate(v_errs)))
        vloss = rmse + (1.0 - auc) * 0.1
        msg = f"ep {ep+1:02d}/{EPOCHS} train {tot/max(1,nb):.4f} | val obs_RMSE={rmse:.4f} resp_AUROC={auc:.3f} Q_MAE={q_mae:.4f}"
        print(msg, flush=True)
        lines.append(msg)
        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

lines.append("\n=== HELD-OUT REAL-LIFE WORLD MODEL & OFFLINE IQL SUMMARY ===")
lines.append(f"Days trained: {len(days)-1} ({days[0]} .. {days[-2]}), held-out: {days[held_day]}")
lines.append(f"Best obs RMSE: {rmse:.4f}, 15s human-response AUROC: {auc:.3f}, 30s Q_MAE: {q_mae:.4f}")
if adv_said:
    lines.append(f"Mean IQL advantage when calling: {float(np.mean(np.concatenate(adv_said))):+.4f}")
if adv_gest:
    lines.append(f"Mean IQL advantage when gesturing: {float(np.mean(np.concatenate(adv_gest))):+.4f}")
for k, name in enumerate(("calm", "ask", "play", "grumble")):
    if adv_styles[k]:
        lines.append(f"  style_{name:8s} mean advantage: {float(np.mean(np.concatenate(adv_styles[k]))):+.4f} (n={sum(len(x) for x in adv_styles[k])})")

os.makedirs("/content/out", exist_ok=True)
torch.save({"state": best_state, "days": days, "rmse": rmse, "auroc": auc}, "/content/out/world_model.pt")
with open("/content/out/world_model_report.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines[-12:]), flush=True)
print(f"Done in {(time.time() - t0)/60:.1f} min", flush=True)
