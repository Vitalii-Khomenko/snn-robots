"""The night sleep of brain v4 (stage 6, BRAIN_V4_PLAN.txt 8.4): the robot learns from its own life while the owner
sleeps, and gets the result without a reboot.

  1. replay   every decision of the main network on the last days, exactly (the diary keeps its whole input: the
              analyzer's 32 numbers + its 12 own-state inputs); checked against the emotions the robot logged
  2. reward   how people answered (reward_v4.py), discounted over ~5 s -> the return of every decision
  3. heads    the action heads (voice, LEDs, head) move towards the choices that were followed by more reward than
              expected and away from those followed by less (a centred advantage: train_heads), anchored to what
              they were (KL); the GRU, the emotion head and the analyzer stay innate; a step that changes them too
              much is taken in part (blend). Only wishes the body carried out count (carried_out: a call the
              executor held back, a nod never made, a gaze the head's own reflex overrode teach nothing)
  4. zones    habituation: a channel that sits in its alarm zone a lot without anything bad following moves its
              alarm border up (0.01 a night); one whose alarm is followed by rough handling moves it down; one that
              is rarely alarming drifts back; |shift| <= 0.15 (learned.py, main/v4/zone_shift.h)
  4c. memory  the most vivid moments of these days, good and bad, with their context (episodes.table): by day the
              robot remembers one when the present is like it (main/v4/memory_v4.h, docs/EPISODES.md)
  4b. muse    the Muse's ideas (muse.py, docs/CREATIVITY.md): an idea after which people answered more than usual
              (reward over 20 s), or after which the world turned out other than the robot expected (curiosity: the
              error of a 1 s forward model of the channels over 5 s), comes a little more often - its bias up 0.05 at
              most a night; one nobody answered, a little less; none below -0.4 (every idea stays possible)
  5. gates    enough life; boots of the network being trained only (its id in the boot line); the replay matches
              (<= 1 % of a boot's decisions off by > 0.05: heads, zones and the Muse learn only from such boots); the
              heads change little (KL, choices); the same step learned without the newest day does not favour, on
              that day, what turned out worse (held_out); all behaviour tests pass in the simulator with the new
              blob - otherwise the robot keeps what it has
  6. send     tools/robot/ota.py learned (switched in between two ticks, a minute on trial, no reboot)

  ./py sleep_v4.py                 the night (run by tools/robot/sleep-v4.timer at 03:30)
  ./py sleep_v4.py --dry-run       everything but sending; --days N (default 3); --min-hours H (default 1)

Writes data/learned/v<version>.bin, live.bin (what the robot has), report out/sleep/<date>.txt, a line in
data/life/sleep_v4.log.
"""
import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import diary as D
import episodes as EP
import learned as LR
import muse as MU
from nn_v4 import EMO_IDX, GROUPS, NnV4
from reward_v4 import reward
from spec_v4 import CHANNELS, HEAD_MODES, NN_OUT, SHIFT_MAX, SNN_OUT, PREV

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
STORE = os.path.join(ROOT, "data", "learned")
LIVE = os.path.join(STORE, "live.bin")
REPORTS = os.path.join(HERE, "out", "sleep")
LOG = os.path.join(D.LIFE, "sleep_v4.log")

GAMMA = 0.98          # per decision (10 Hz): ~5 s
A_CLIP = 3.0          # the centred advantage, in units of its spread, is clipped here (train_heads)
ETA = 2.0             # its tilt of a choice's odds: exp(A / ETA), at most exp(1.5) = 4.5x
ANCHOR = 1.0          # weight of the KL to the heads the robot has
MIN_HOURS = 1.0       # of daytime life before anything is learned (--min-hours)
MIN_EVENTS = 5
KL_MAX, FLIP_MAX = 0.05, 0.15
ALPHA_MIN = 0.1      # the smallest share of a night's step on the heads (blend)
REPLAY_BAD_ERR = 0.05   # a decision replays wrong when an emotion's probability is off by more (replay)
REPLAY_MAX_BAD = 0.01   # a boot is learned from when at most this share of its decisions replays wrong
ADAPTIVE = ["loud", "sharp", "voice", "near", "light", "motion"]  # touch, shake, tilt keep the owner's innate rules
ZONE_STEP, ZONE_RELAX = 0.01, 0.005
SAY = NN_OUT.index("say")
# the Muse (learn_muse)
MUSE_STEP, MUSE_ETA, MUSE_FLOOR, MUSE_CURIOSITY = 0.05, 0.05, -0.4, 0.3
MUSE_MIN_EPISODES = 3
IDEA0 = len(SNN_OUT) + PREV.index("idea_play")
# what the body did with the wishes (carried_out; V4Executor, BabbleVoice)
SAY_ON = 0.55                     # a call starts when say rises above this
HEAD_GESTURES = [HEAD_MODES.index(m) for m in ("nod", "shake", "flinch")]
FREE_HEAD = [HEAD_MODES.index(m) for m in ("hold", "look", "scan")]
GAZE_OWN = 32                     # V4Executor::kGazeOwn in the diary's `held`
ATT_LOOK, WATCH_TICKS = 0.12, 200  # diaries before `held`: the orienting reflex holds the head 10 s (kAttLook, kWatchMs)
HELD = [(1, "call: night"), (2, "call: budget spent"), (4, "call: busy / too soon"), (8, "gesture: night"),
        (16, "gesture: too soon"), (32, "gaze: the head's own")]


def say(lines, text):
    print(text, flush=True)
    lines.append(text)


def days_with_v4(n):
    days = sorted(d for d in os.listdir(D.LIFE) if d[:2] == "20" and d >= "2026-10-24")
    today = dt.date.today().isoformat()
    return [d for d in days if d < today or d == today][-n:]


def boot_ids(lines):
    """boot -> (main network id or None, firmware sha or None) from the boot's own lines (RobotRuntime::begin; since
    2026-10-28 evening)."""
    ids = {}
    for _, b, _, text in lines:
        nn, fw = ids.get(b, (None, None))
        if "main network" in text and "(id " in text:
            nn = int(text.split("(id ")[1].split(")")[0], 16)
        elif text.find("firmware in") >= 0 and ", sha " in text:
            fw = text.split(", sha ")[1].split()[0]
        ids[b] = (nn, fw)
    return ids


def load_life(days):
    """Per boot: the decided ticks with a full input (the own-state columns exist since 2026-10-24 13:22)."""
    S0, P0 = D.col("snn_" + SNN_OUT[0]), D.col("pv_" + PREV[0])
    N0 = D.col("nn_" + NN_OUT[0])
    boots = []
    for day in days:
        t, lines = D.load(day)
        if t is None:
            continue
        ids = boot_ids(lines)
        r, _ = reward(t)
        for b in np.unique(t["boot"]):
            m = t["boot"] == b
            v, f, seq = t["v"][m], t["flags"][m], t["seq"][m]
            o = np.argsort(seq)
            v, f, seq, rb = v[o], f[o], seq[o], r[m][o]
            dec = (f & D.FLAG_DECIDED) > 0
            # every own-state input logged (the Muse's ideas since 2026-10-26 evening: older boots ran another network)
            if dec.sum() < 100 or not np.isfinite(v[dec][:, P0:P0 + len(PREV)]).all():
                continue
            # the reward between two decisions goes to the earlier one
            idx = np.flatnonzero(dec)
            rd = np.add.reduceat(rb, idx) if len(idx) else np.zeros(0)
            boots.append(dict(day=day, dec_idx=idx, nn_id=ids.get(b, (None, None))[0], fw=ids.get(b, (None, None))[1],
                              said=v[:, D.col("said")] > 0.5, gest=v[:, D.col("gesture")] > 0.5,
                              held=v[:, D.col("held")], att=v[:, D.col("snn_att_strength")],
                              x=np.concatenate([v[dec][:, S0:S0 + len(SNN_OUT)], v[dec][:, P0:P0 + len(PREV)]], 1),
                              nn=v[dec][:, N0:N0 + len(NN_OUT)], r=rd.astype(np.float32),
                              day_mask=((f[dec] & D.FLAG_DRIVES) > 0) & ((f[dec] & D.FLAG_NIGHT) == 0),
                              gaps=int(np.sum(np.diff(seq[dec].astype(np.int64)) != 2)),
                              ch=v[:, D.col("ch_" + CHANNELS[0]):D.col("ch_" + CHANNELS[0]) + len(CHANNELS)],
                              raw_r=rb, flags=f, hour=v[:, D.col("raw_hour")]))
    return boots


def returns(r):
    g = np.zeros_like(r)
    acc = 0.0
    for i in range(len(r) - 1, -1, -1):
        acc = r[i] + GAMMA * acc
        g[i] = acc
    return g


def base_model():
    m = NnV4()
    m.load_state_dict(torch.load(os.path.join(HERE, "data", "nn_v4.pt"), weights_only=False, map_location="cpu")["state"])
    version, shift, mbias = 0, np.zeros(len(CHANNELS), np.float32), np.zeros(len(MU.OUTS), np.float32)
    if os.path.exists(LIVE):
        version, arrays = LR.unpack(open(LIVE, "rb").read())
        # the live heads only if they were made for this network (a retrained one starts from its own innate heads)
        if "nn.act1_w" in arrays and LR.id_of(arrays) == LR.nn_id(m.state_dict()):
            with torch.no_grad():
                m.act1.weight.copy_(torch.tensor(arrays["nn.act1_w"]).view_as(m.act1.weight))
                m.act1.bias.copy_(torch.tensor(arrays["nn.act1_b"]))
                m.act2.weight.copy_(torch.tensor(arrays["nn.act2_w"]).view_as(m.act2.weight))
                m.act2.bias.copy_(torch.tensor(arrays["nn.act2_b"]))
        if "zones.shift" in arrays:
            shift = arrays["zones.shift"].astype(np.float32)
        if "muse.bias" in arrays:
            mbias = arrays["muse.bias"].astype(np.float32)
    m.eval()
    return m, version, shift, mbias


def replay(m, boots):
    """GRU + emotion head over every boot's decisions -> features [h, emotion probs], and per boot the share of
    decisions whose replayed emotions are off by more than REPLAY_BAD_ERR, and the 99th percentile of the error.

    (2026-10-28, an audit: the gate was the median - half of the decisions could be wrong and pass. Real boots: p99
    <= 0.007, share above 0.05 <= 0.00014, the rare misses after diary gaps.)"""
    feats, errs = [], []
    with torch.no_grad():
        for b in boots:
            x = torch.tensor(b["x"], dtype=torch.float32)[None]
            z, _ = m.gru(x)
            e = m.emo(z)
            p = torch.softmax(e[..., :len(EMO_IDX)], -1)
            feats.append(torch.cat([z, p], -1)[0])
            logged = torch.tensor(b["nn"][:, EMO_IDX], dtype=torch.float32)
            err = (p[0] - logged).abs().max(-1).values.numpy()
            errs.append((float((err > REPLAY_BAD_ERR).mean()), float(np.quantile(err, 0.99))))
    return feats, errs


def replay_ok(err):
    bad_share, p99 = err
    return bad_share <= REPLAY_MAX_BAD and p99 <= REPLAY_BAD_ERR


def heads_out(m, f):
    return m.act2(torch.tanh(m.act1(f)))  # [T, 39] logits in ACT order


ACT = [i for i in range(len(NN_OUT)) if i not in EMO_IDX and i != NN_OUT.index("emo_strength")]
GROUPS_ACT = {g: [ACT.index(i) for i in idx] for g, idx in GROUPS.items()}
SAY_ACT = ACT.index(SAY)


def taken(nn_logged):
    """What the network chose at each decision: the strongest choice per group, say above SAY_ON (the voice's own
    threshold; 0.5 until 2026-10-28)."""
    t = {g: np.argmax(nn_logged[:, idx], 1) for g, idx in GROUPS.items()}
    t["say"] = (nn_logged[:, SAY] > SAY_ON).astype(np.float32)
    return t


def runs(mask):
    """The True runs of a bool array: [(start, end)], end exclusive."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def gaze_own(b):
    """Per tick: the head followed its own reflex, a watch or a glance, not the network's gaze. The diary's `held`
    since 2026-10-28 evening; before it the orienting reflex is inferred (attention above ATT_LOOK in the last 10 s;
    the glances are not seen)."""
    held = b["held"]
    if np.isfinite(held).all():
        return (held.astype(np.int64) & GAZE_OWN) > 0
    att = np.nan_to_num(b["att"]) > ATT_LOOK
    c = np.r_[0, np.cumsum(att)]
    i = np.arange(len(att))
    return (c[i + 1] - c[np.maximum(0, i + 1 - WATCH_TICKS)]) > 0


def carried_out(b):
    """Per decision and action group: was the wish carried out? (2026-10-28, an audit: the night took every wish for a
    deed - on 10-27 45 % of the runs of a wanted call never sounded (the call budget, a phrase still playing) and ~40 %
    of the wanted nods and flinches were never made (too soon after the last), yet each was credited with what came
    after it.) A run of decisions wanting a call counts when a call started during it; a gesture mode when its gesture
    started; look / scan / hold not while the head followed its own reflex or a glance; a sound's style only where a
    call starts. Colours and patterns are always shown. Wishes not carried out ask for no change (train_heads)."""
    idx, nn, T = b["dec_idx"], b["nn"], len(b["said"])
    end = np.r_[idx[1:], T]  # a decision's ticks: idx[i] .. end[i]
    said_c, gest_c = np.r_[0, np.cumsum(b["said"])], np.r_[0, np.cumsum(b["gest"])]
    own_c = np.r_[0, np.cumsum(gaze_own(b))]
    ok = {}
    wish = nn[:, SAY] > SAY_ON
    ok["say"] = ~wish
    for s, e in runs(wish):
        a, z = idx[s], min(T, end[e - 1] + 2)
        ok["say"][s:e] = said_c[z] - said_c[a] > 0
    ok["style"] = said_c[end] - said_c[idx] > 0
    head = np.argmax(nn[:, GROUPS["head"]], 1)
    ok["head"] = ~(np.isin(head, FREE_HEAD) & (own_c[end] - own_c[idx] > 0))
    for g in HEAD_GESTURES:
        for s, e in runs(head == g):
            a, z = idx[s], min(T, end[e - 1] + 2)
            ok["head"][s:e] = gest_c[z] - gest_c[a] > 0
    ok["color"] = ok["pattern"] = np.ones(len(idx), bool)
    return ok


def train_heads(m, F_all, act, adv, ok=None, epochs=40):
    """The action heads towards the choices that turned out better than expected and away from those that turned out
    worse. adv: the centred advantage of each decision (mean ~0, in units of its spread). The target of a decision is
    the heads' own distribution with the odds of the choice made tilted by exp(adv / ETA): a decision of average
    outcome asks for no change at all, and no decision for more than exp(A_CLIP / ETA) in the odds. ok: per group,
    the decisions whose wish the body carried out (carried_out); the others are not tilted. Anchored (KL) to the heads
    the robot has.

    2026-10-28: it was advantage-weighted regression (weights exp(A) > 0 on the robot's own greedy choices). With ~400
    rewarded moments in 14 h the weights were ~1 almost everywhere: the step was the network imitating itself (KL 0.003
    from a step with all weights 1) and cut the entropy of every choice by ~35 % a night - a slowly monotonous robot
    (saved only by the KL gate, which dropped every night's step). Signed weights on -log p were unbounded; on
    log(1 - p) for A < 0 they flattened every choice (entropy +20 % a night)."""
    new = NnV4()
    new.load_state_dict(m.state_dict())
    for p in new.parameters():
        p.requires_grad_(False)
    params = list(new.act1.parameters()) + list(new.act2.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=1e-3)
    with torch.no_grad():
        old = heads_out(m, F_all)
        tilt = torch.tensor(adv) / ETA
        done = {g: torch.ones(len(adv)) if ok is None else torch.tensor(ok[g], dtype=torch.float32)
                for g in list(GROUPS_ACT) + ["say"]}
        targets = {}
        for g, idx in GROUPS_ACT.items():
            q = F.softmax(old[:, idx], -1) * torch.exp((tilt * done[g])[:, None] * F.one_hot(torch.tensor(act[g]), len(idx)))
            targets[g] = q / q.sum(-1, keepdim=True)
        p1 = torch.sigmoid(old[:, SAY_ACT])
        t = torch.tensor(act["say"])
        k = torch.exp(tilt * done["say"])
        up = torch.where(t > 0.5, p1 * k, p1)          # the odds of the outcome that happened, times exp(tilt)
        down = torch.where(t > 0.5, 1 - p1, (1 - p1) * k)
        say_q = up / (up + down)
    other = [i for i in range(len(ACT)) if i != SAY_ACT and not any(i in v for v in GROUPS_ACT.values())]
    n = len(F_all)
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        for b in np.array_split(rng.permutation(n), max(1, n // 4096)):
            out = heads_out(new, F_all[b])
            loss = 0.0
            for g, idx in GROUPS_ACT.items():
                logp = F.log_softmax(out[:, idx], -1)
                loss = loss + F.kl_div(logp, targets[g][b], reduction="batchmean")
                loss = loss + ANCHOR * F.kl_div(logp, F.softmax(old[b][:, idx], -1), reduction="batchmean")
            loss = loss + F.binary_cross_entropy_with_logits(out[:, SAY_ACT], say_q[b])
            loss = loss + ANCHOR * F.binary_cross_entropy_with_logits(out[:, SAY_ACT], p1[b])
            loss = loss + ANCHOR * F.mse_loss(out[:, other], old[b][:, other])
            opt.zero_grad()
            loss.backward()
            opt.step()
    new.eval()
    return new


def prepare(m, boots, idx, feats, wv=None):
    """The daytime decisions of the boots `idx` for the heads: features, the centred advantage, the choices made and
    which were carried out. wv: a value baseline fitted elsewhere (held_out); None fits it here. -> F, A, act, ok, wv,
    spread (of the raw advantage) and each decision's minute (for a bootstrap by minutes)."""
    F_all = torch.cat([feats[i][torch.from_numpy(boots[i]["day_mask"])] for i in idx])
    G = np.concatenate([returns(boots[i]["r"])[boots[i]["day_mask"]] for i in idx])
    X = np.concatenate([F_all.numpy(), np.ones((len(F_all), 1), np.float32)], 1)
    if wv is None:  # value baseline: ridge regression of the return on the features
        wv = np.linalg.solve(X.T @ X + 1.0 * np.eye(X.shape[1]), X.T @ G)
    A = G - X @ wv
    spread = A.std() + 1e-6
    act = taken(np.concatenate([boots[i]["nn"][boots[i]["day_mask"]] for i in idx]))
    outs = [carried_out(boots[i]) for i in idx]
    ok = {g: np.concatenate([o[g][boots[i]["day_mask"]] for o, i in zip(outs, idx)]) for g in outs[0]}
    minute = np.concatenate([k * 100000 + np.arange(int(boots[i]["day_mask"].sum())) // 600 for k, i in enumerate(idx)])
    return F_all, A, act, ok, wv, spread, minute


def improvement(m, new, feat, act, ok, A):
    """Per decision, how the step moves the odds of what was chosen and carried out, times how that turned out:
    sum over the groups of (p_new / p_old - 1) x advantage. Its mean is the step's first-order change of the return
    (the policy-gradient estimate), > 0 when the choices it favours were followed by more reward."""
    with torch.no_grad():
        a, b = heads_out(m, feat), heads_out(new, feat)
    s = np.zeros(len(feat))
    for g, idx in GROUPS_ACT.items():
        ch = torch.tensor(act[g])[:, None]
        pa = torch.softmax(a[:, idx], -1).gather(1, ch)[:, 0].numpy()
        pb = torch.softmax(b[:, idx], -1).gather(1, ch)[:, 0].numpy()
        s += ok[g] * (pb / pa - 1.0)
    on = act["say"] > 0.5
    pa = torch.sigmoid(a[:, SAY_ACT]).numpy()
    pb = torch.sigmoid(b[:, SAY_ACT]).numpy()
    s += ok["say"] * np.where(on, pb / pa, (1 - pb) / (1 - pa)) - ok["say"]
    return s * A


def held_out(m, boots, use, feats, lines, rng=None):
    """Does a night's step generalise? (2026-10-28, an audit A11: the gates only checked that the heads changed little
    and broke no test.) The step is learned on every day but the newest and judged on the newest: does it favour the
    choices that were followed by more reward there? The mean improvement() over that day with a 95 % interval from a
    bootstrap over its minutes. -> (mean, lo, hi) in units of the train days' advantage spread, x 1000; None when
    there is one day only."""
    days = sorted({boots[i]["day"] for i in use})
    if len(days) < 2:
        say(lines, "  held-out check: one day only - not judged")
        return None
    tr = [i for i in use if boots[i]["day"] != days[-1]]
    te = [i for i in use if boots[i]["day"] == days[-1]]
    F_tr, A_tr, act_tr, ok_tr, wv, spread, _ = prepare(m, boots, tr, feats)
    A_tr = np.clip((A_tr - A_tr.mean()) / spread, -A_CLIP, A_CLIP).astype(np.float32)
    new = train_heads(m, F_tr, act_tr, A_tr, ok_tr)
    F_te, A_te, act_te, ok_te, _, _, minute = prepare(m, boots, te, feats, wv)
    A_te = np.clip((A_te - A_te.mean()) / spread, -A_CLIP, A_CLIP)
    s = improvement(m, new, F_te, act_te, ok_te, A_te)
    s_tr = improvement(m, new, F_tr, act_tr, ok_tr, A_tr)
    _, inv = np.unique(minute, return_inverse=True)
    sums, counts = np.bincount(inv, s), np.bincount(inv)
    rng = rng or np.random.default_rng(0)
    boot = [sums[k].sum() / counts[k].sum() for k in (rng.integers(0, len(sums), len(sums)) for _ in range(400))]
    mean, lo, hi = 1000 * s.mean(), 1000 * np.quantile(boot, 0.025), 1000 * np.quantile(boot, 0.975)
    say(lines, f"  held-out check (learned on {', '.join(days[:-1])}, judged on {days[-1]}, {len(s)} decisions): "
               f"improvement {mean:+.3f} (95 % {lo:+.3f} .. {hi:+.3f}) x 1e-3; on the days it learned from "
               f"{1000 * s_tr.mean():+.3f}")
    return mean, lo, hi


def blend(m, new, alpha):
    """The heads a share `alpha` of the way from m's to new's (the rest of the network is m's)."""
    out = NnV4()
    out.load_state_dict(m.state_dict())
    with torch.no_grad():
        for name in ("act1", "act2"):
            for p_out, p_old, p_new in zip(getattr(out, name).parameters(), getattr(m, name).parameters(),
                                           getattr(new, name).parameters()):
                p_out.copy_(p_old + alpha * (p_new - p_old))
    out.eval()
    return out


def change(m, new, F_all):
    """Mean KL per choice group and the share of decisions whose choice flips, new heads vs old."""
    with torch.no_grad():
        a, b = heads_out(m, F_all), heads_out(new, F_all)
    kl, flips = [], []
    for idx in GROUPS_ACT.values():
        pa, lb = F.softmax(a[:, idx], -1), F.log_softmax(b[:, idx], -1)
        kl.append(float(F.kl_div(lb, pa, reduction="batchmean")))
        flips.append(float((a[:, idx].argmax(-1) != b[:, idx].argmax(-1)).float().mean()))
    flips.append(float(((a[:, SAY_ACT] > 0) != (b[:, SAY_ACT] > 0)).float().mean()))
    return max(kl), max(flips)


def zones(boots, shift, lines):
    """One night's step of the zone shifts (habituation / sensitization / relaxing back)."""
    new = shift.copy()
    for name in ADAPTIVE:
        c = CHANNELS.index(name)
        a = LR.alarm(c)
        if a is None:
            continue
        alarm_t, bad_after, onsets, total = 0, 0, 0, 0
        for b in boots:
            day = ((b["flags"] & D.FLAG_DRIVES) > 0) & ((b["flags"] & D.FLAG_NIGHT) == 0)
            x = b["ch"][:, c]
            on = (x >= a) & day
            total += int(day.sum())
            alarm_t += int(on.sum())
            starts = np.flatnonzero(on[1:] & ~on[:-1]) + 1
            bad = b["raw_r"] <= -0.5   # shaken, tilted, poked hard
            for s in starts:
                onsets += 1
                bad_after += bool(bad[s:s + 600].any())  # within 30 s
        if total < MIN_HOURS * 72000:  # ticks at 20 Hz
            continue
        share = alarm_t / total
        if onsets >= 5 and bad_after / onsets > 0.3:
            new[c] -= ZONE_STEP
            why = f"alarm followed by rough handling {bad_after}/{onsets}: sensitized"
        elif share > 0.05:
            new[c] += ZONE_STEP
            why = f"in alarm {100 * share:.1f} % of the day, nothing bad after: habituated"
        elif share < 0.01 and new[c] != 0:
            new[c] -= np.sign(new[c]) * min(ZONE_RELAX, abs(new[c]))
            why = "rarely alarming: back towards innate"
        else:
            continue
        new[c] = float(np.clip(new[c], -SHIFT_MAX, SHIFT_MAX))
        say(lines, f"  zone {name}: shift {shift[c]:+.3f} -> {new[c]:+.3f} ({why})")
    return new


def surprise_of(ch, day):
    """The channels' unexpectedness per tick: |error| of a ridge forward model ch(t) -> ch(t + 1 s), fitted on the day."""
    lag = 20
    e = np.full(len(ch), np.nan, np.float32)
    ok = np.isfinite(ch).all(1)
    i = np.flatnonzero(day[:-lag] & ok[:-lag] & ok[lag:])
    if len(i) < 2000:
        return e
    fit = i[::5]
    X = np.concatenate([ch[fit], np.ones((len(fit), 1), np.float32)], 1)
    W = np.linalg.solve(X.T @ X + 1.0 * np.eye(X.shape[1]), X.T @ ch[fit + lag])
    pred = np.concatenate([ch[i], np.ones((len(i), 1), np.float32)], 1) @ W
    e[i] = np.abs(ch[i + lag] - pred).mean(1)
    return e


def learn_muse(boots, bias, lines):
    """One night's step of the Muse's idea biases (docs/CREATIVITY.md step 3 + 4)."""
    new = bias.copy()
    stats = {k: ([], []) for k in MU.IDEAS}
    for b in boots:
        day, r = b["day_mask"], b["r"]
        if day.sum() < 6000:  # 10 min of daytime decisions
            continue
        win_r = np.convolve(r, np.ones(200), "full")[199:][:len(r)]  # reward over the next 20 s of each decision
        e = surprise_of(b["ch"], ((b["flags"] & D.FLAG_DRIVES) > 0) & ((b["flags"] & D.FLAG_NIGHT) == 0))
        e_dec = np.array([np.nanmean(e[t:t + 100]) if np.isfinite(e[t:t + 100]).any() else np.nan for t in b["dec_idx"]])
        rb, rs = win_r[day].mean(), win_r[day].std() + 1e-6
        cb, cs = np.nanmean(e_dec[day]), np.nanstd(e_dec[day]) + 1e-6
        for k, name in enumerate(MU.IDEAS):
            on = (b["x"][:, IDEA0 + k] > MU.IDEA_ON) & day
            last = -10 ** 9
            for i in np.flatnonzero(on[1:] & ~on[:-1]) + 1:
                if i - last < 100 or i + 200 > len(r):
                    continue
                last = i
                stats[name][0].append((win_r[i] - rb) / rs)
                if np.isfinite(e_dec[i]) and np.isfinite(cb):
                    stats[name][1].append((e_dec[i] - cb) / cs)
    for k, name in enumerate(MU.IDEAS):
        R, C = stats[name]
        j = 3 + k
        if len(R) < MUSE_MIN_EPISODES:
            if new[j] != 0:  # not had: back towards innate
                new[j] -= np.sign(new[j]) * min(0.005, abs(new[j]))
            continue
        adv = float(np.mean(R)) + MUSE_CURIOSITY * (float(np.mean(C)) if C else 0.0)
        step = float(np.clip(MUSE_ETA * adv * min(1.0, len(R) / 10), -MUSE_STEP, MUSE_STEP))
        new[j] = float(np.clip(new[j] + step, MUSE_FLOOR, MU.BIAS_MAX))
        say(lines, f"  muse {name}: {len(R)} ideas, answered {np.mean(R):+.2f}, curiosity "
                   f"{np.mean(C) if C else 0:+.2f} -> bias {bias[j]:+.3f} -> {new[j]:+.3f}")
    return new.astype(np.float32)


def behaviour_gate(blob_path, lines):
    r = subprocess.run([sys.executable, os.path.join(HERE, "behavior_test_v4.py"), "--learned", blob_path],
                       capture_output=True, text=True, cwd=HERE)
    ok = r.returncode == 0
    fails = [l.strip() for l in r.stdout.splitlines() if l.strip().startswith("FAIL")]
    say(lines, "  behaviour tests with the new blob: " + ("all PASS" if ok else "FAILED: " + "; ".join(fails[:4])))
    return ok


def main():
    global MIN_HOURS
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-hours", type=float, default=MIN_HOURS, help="daytime life needed (tests: less)")
    a = ap.parse_args()
    MIN_HOURS = a.min_hours
    started = time.time()
    lines = []
    tonight = dt.date.today().isoformat()
    days = days_with_v4(a.days)
    say(lines, f"night of {tonight}: days {', '.join(days) or '-'}")
    m, live_version, shift, mbias = base_model()
    boots = load_life(days)
    # boots of another network are not this network's life (the boot line's id, 2026-10-28 evening on)
    nn_id = LR.nn_id(m.state_dict())
    other = [b for b in boots if b["nn_id"] is not None and b["nn_id"] != nn_id]
    boots = [b for b in boots if b["nn_id"] is None or b["nn_id"] == nn_id]
    n_dec = sum(len(b["x"]) for b in boots)
    n_day = sum(int(b["day_mask"].sum()) for b in boots)
    fws = sorted({b["fw"] for b in boots if b["fw"]})
    say(lines, f"  {len(boots)} boot(s), {n_dec} decisions ({n_day / 36000:.1f} h by day), live blob version {live_version}"
               f"; network {nn_id:08x}" + (f", firmware {', '.join(fws)}" if fws else "")
               + (f"; {len(other)} boot(s) of another network left out" if other else ""))
    changed = False
    parts = []
    heads_new = None
    feats, errs = replay(m, boots) if boots else ([], [])
    if boots:
        say(lines, "  replay, per boot (share of decisions off by > %.2f / p99 of the error): %s"
            % (REPLAY_BAD_ERR, ", ".join(f"{a:.4f}/{q:.3f}" for a, q in errs)))
    use = [i for i, e in enumerate(errs) if replay_ok(e)]
    if len(use) < len(boots):
        say(lines, f"  {len(boots) - len(use)} boot(s) replay wrong: not learned from (heads, zones, Muse)")
    if n_day >= MIN_HOURS * 36000:  # decisions at 10 Hz
        n_use = sum(int(boots[i]["day_mask"].sum()) for i in use)
        events = sum(int((np.abs(boots[i]["r"][boots[i]["day_mask"]]) > 0).sum()) for i in use)
        say(lines, f"  reward events by day {events}")
        if n_use >= MIN_HOURS * 36000 and events >= MIN_EVENTS:
            F_all, A, act, ok, _, spread, _ = prepare(m, boots, use, feats)
            A = np.clip((A - A.mean()) / spread, -A_CLIP, A_CLIP).astype(np.float32)
            wished = {"say": act["say"] > 0.5, "head": np.isin(act["head"], HEAD_GESTURES)}
            say(lines, "  wishes carried out by day: calls %.0f %%, nods / shakes / flinches %.0f %%, the network's gaze %.0f %%"
                % (100 * ok["say"][wished["say"]].mean() if wished["say"].any() else 100,
                   100 * ok["head"][wished["head"]].mean() if wished["head"].any() else 100,
                   100 * ok["head"][np.isin(act["head"], FREE_HEAD)].mean()))
            held = np.concatenate([boots[i]["held"][np.isfinite(boots[i]["held"])].astype(np.int64) for i in use])
            if len(held):
                say(lines, "  held back (ticks, diary `held`): " + ", ".join(
                    f"{why} {int(((held & bit) > 0).sum())}" for bit, why in HELD))
            new = train_heads(m, F_all, act, A, ok)
            kl, flip = change(m, new, F_all)
            say(lines, f"  the action heads: KL {kl:.4f} (max {KL_MAX}), choices changed {100 * flip:.1f} % "
                       f"(max {100 * FLIP_MAX:.0f} %)")
            # Too big a step is taken in part (2026-10-28: three nights in a row the whole step was 0.060-0.062 against
            # 0.05 and was dropped - the heads never learned anything): half, a quarter ... of the way, the largest
            # share that passes the gate.
            alpha = 1.0
            while (kl > KL_MAX or flip > FLIP_MAX) and alpha > ALPHA_MIN:
                alpha /= 2
                cand = blend(m, new, alpha)
                kl, flip = change(m, cand, F_all)
                say(lines, f"    {alpha:g} of the step: KL {kl:.4f}, choices changed {100 * flip:.1f} %")
            # the same step learned without the newest day, judged on it: one that clearly favours what turned out
            # worse there is not taken (2026-10-28; 10-27 -> 10-28: +7.3 (+6.4 .. +8.2) x 1e-3, with the outcomes
            # shuffled -0.1 (-0.3 .. +0.1))
            ho = held_out(m, boots, use, feats, lines)
            if kl > KL_MAX or flip > FLIP_MAX:
                say(lines, "  gate: the heads would change too much even in part - kept")
            elif ho is not None and ho[2] < 0:
                say(lines, "  gate: on the held-out day the step favours what turned out worse - kept")
            else:
                heads_new = LR.heads_of(new if alpha == 1.0 else cand)
                changed = True
                parts.append(f"heads ({alpha:g} of the step, KL {kl:.4f}"
                             + (f", held-out {ho[0]:+.2f}e-3)" if ho else ")"))
        else:
            say(lines, "  not enough rewarded daytime life for the heads yet - kept")
    else:
        say(lines, f"  less than {MIN_HOURS:g} h of daytime decisions - heads kept")
    good = [boots[i] for i in use]
    new_shift = zones(good, shift, lines)
    if not np.allclose(new_shift, shift):
        changed = True
        parts.append(f"zones ({int((~np.isclose(new_shift, shift)).sum())} channel(s))")
    new_mbias = learn_muse(good, mbias, lines)
    if not np.allclose(new_mbias, mbias):
        changed = True
        parts.append(f"Muse ({int((~np.isclose(new_mbias, mbias)).sum())} idea bias(es))")
    mem = EP.table(days)
    if len(mem):
        changed = True
        parts.append(f"memories ({int((mem[:, 1] > 0).sum())} good, {int((mem[:, 1] < 0).sum())} bad)")
        say(lines, "  memories, the strongest: " + "; ".join(EP.describe(r) for r in mem[:3]))
    if not changed:
        return finish(lines, started, "nothing new: the robot keeps what it has")
    say(lines, "  new: " + ", ".join(parts) + ("" if heads_new is not None else "; the heads as they were"))
    if heads_new is None:
        heads_new = LR.heads_of(m)
    version = int(dt.datetime.now().strftime("%y%m%d%H"))
    # a dry run's candidate goes with the reports, not into the store of the robot's versions
    store = REPORTS if a.dry_run else STORE
    os.makedirs(store, exist_ok=True)
    blob_path = os.path.join(store, f"v{version}{'_dry' if a.dry_run else ''}.bin")
    open(blob_path, "wb").write(LR.pack(version, heads_new, new_shift, new_mbias, nn=LR.nn_id(m.state_dict()),
                                        episodes=mem if len(mem) else None))
    say(lines, f"  candidate {os.path.relpath(blob_path, ROOT)} (version {version})")
    if not behaviour_gate(blob_path, lines):
        return finish(lines, started, "gate failed: the robot keeps what it has")
    if a.dry_run:
        return finish(lines, started, "dry run: not sent")
    for attempt in range(3):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "robot", "ota.py"), "learned", blob_path],
                           capture_output=True, text=True)
        if r.returncode == 0:
            shutil.copy(blob_path, LIVE)
            return finish(lines, started, f"live on the robot: version {version}")
        say(lines, f"  robot: {(r.stdout + r.stderr).strip().splitlines()[-1:]} (attempt {attempt + 1})")
        time.sleep(120)
    return finish(lines, started, "the robot did not take it: it keeps what it has")


def finish(lines, started, result):
    say(lines, f"  {result}; slept {(time.time() - started) / 60:.1f} min")
    os.makedirs(REPORTS, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    open(os.path.join(REPORTS, f"{stamp}.txt"), "w").write("\n".join(lines) + "\n")
    with open(LOG, "a") as f:
        f.write(dt.datetime.now().strftime("%Y-%m-%d %H:%M ") + " | ".join(l.strip() for l in lines) + "\n")


if __name__ == "__main__":
    main()
