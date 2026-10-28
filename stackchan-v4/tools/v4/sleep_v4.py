"""The night sleep of brain v4 (stage 6, BRAIN_V4_PLAN.txt 8.4): the robot learns from its own life while the owner
sleeps, and gets the result without a reboot.

  1. replay   every decision of the main network on the last days, exactly (the diary keeps its whole input: the
              analyzer's 32 numbers + its 12 own-state inputs); checked against the emotions the robot logged
  2. reward   how people answered (reward_v4.py), discounted over ~5 s -> the return of every decision
  3. heads    the action heads (voice, LEDs, head) move towards the choices that were followed by more reward than
              expected and away from those followed by less (a centred advantage: train_heads), anchored to what
              they were (KL); the GRU, the emotion head and the analyzer stay innate; a step that changes them too
              much is taken in part (blend)
  4. zones    habituation: a channel that sits in its alarm zone a lot without anything bad following moves its
              alarm border up (0.01 a night); one whose alarm is followed by rough handling moves it down; one that
              is rarely alarming drifts back; |shift| <= 0.15 (learned.py, main/v4/zone_shift.h)
  4b. muse    the Muse's ideas (muse.py, docs/CREATIVITY.md): an idea after which people answered more than usual
              (reward over 20 s), or after which the world turned out other than the robot expected (curiosity: the
              error of a 1 s forward model of the channels over 5 s), comes a little more often - its bias up 0.05 at
              most a night; one nobody answered, a little less; none below -0.4 (every idea stays possible)
  5. gates    enough life; the replay matches (<= 1 % of a boot's decisions off by > 0.05); the heads change little (KL, choices); all behaviour tests pass in the
              simulator with the new blob - otherwise the robot keeps what it has
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
import learned as LR
import muse as MU
from nn_v4 import EMO_IDX, GROUPS, NnV4
from reward_v4 import reward
from spec_v4 import CHANNELS, NN_OUT, SHIFT_MAX, SNN_OUT, PREV

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


def say(lines, text):
    print(text, flush=True)
    lines.append(text)


def days_with_v4(n):
    days = sorted(d for d in os.listdir(D.LIFE) if d[:2] == "20" and d >= "2026-10-24")
    today = dt.date.today().isoformat()
    return [d for d in days if d < today or d == today][-n:]


def load_life(days):
    """Per boot: the decided ticks with a full input (the own-state columns exist since 2026-10-24 13:22)."""
    S0, P0 = D.col("snn_" + SNN_OUT[0]), D.col("pv_" + PREV[0])
    N0 = D.col("nn_" + NN_OUT[0])
    boots = []
    for day in days:
        t, _ = D.load(day)
        if t is None:
            continue
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
            boots.append(dict(day=day, dec_idx=idx,
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
    """What the robot did at each decision: the strongest choice per group, say above 0.5."""
    t = {g: np.argmax(nn_logged[:, idx], 1) for g, idx in GROUPS.items()}
    t["say"] = (nn_logged[:, SAY] > 0.5).astype(np.float32)
    return t


def train_heads(m, F_all, act, adv, epochs=40):
    """The action heads towards the choices that turned out better than expected and away from those that turned out
    worse. adv: the centred advantage of each decision (mean ~0, in units of its spread). The target of a decision is
    the heads' own distribution with the odds of the choice made tilted by exp(adv / ETA): a decision of average
    outcome asks for no change at all, and no decision for more than exp(A_CLIP / ETA) in the odds. Anchored (KL) to
    the heads the robot has.

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
        targets = {}
        for g, idx in GROUPS_ACT.items():
            q = F.softmax(old[:, idx], -1) * torch.exp(tilt[:, None] * F.one_hot(torch.tensor(act[g]), len(idx)))
            targets[g] = q / q.sum(-1, keepdim=True)
        p1 = torch.sigmoid(old[:, SAY_ACT])
        t = torch.tensor(act["say"])
        k = torch.exp(tilt)
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
    n_dec = sum(len(b["x"]) for b in boots)
    n_day = sum(int(b["day_mask"].sum()) for b in boots)
    say(lines, f"  {len(boots)} boot(s), {n_dec} decisions ({n_day / 36000:.1f} h by day), live blob version {live_version}")
    changed = False
    heads_new = None
    if n_day >= MIN_HOURS * 36000:  # decisions at 10 Hz
        feats, errs = replay(m, boots)
        say(lines, "  replay, per boot (share of decisions off by > %.2f / p99 of the error): %s"
            % (REPLAY_BAD_ERR, ", ".join(f"{a:.4f}/{q:.3f}" for a, q in errs)))
        use = [i for i, e in enumerate(errs) if replay_ok(e)]
        if len(use) < len(boots):
            say(lines, f"  {len(boots) - len(use)} boot(s) replay wrong: not learned from")
        F_all = torch.cat([feats[i][torch.from_numpy(boots[i]["day_mask"])] for i in use]) if use else torch.zeros(0, 85)
        G = np.concatenate([returns(boots[i]["r"])[boots[i]["day_mask"]] for i in use]) if use else np.zeros(0)
        events = sum(int((np.abs(boots[i]["r"][boots[i]["day_mask"]]) > 0).sum()) for i in use)
        say(lines, f"  reward events by day {events}, mean return {G.mean() if len(G) else 0:+.3f}")
        if len(F_all) >= MIN_HOURS * 36000 and events >= MIN_EVENTS:
            # value baseline: ridge regression of the return on the features
            X = np.concatenate([F_all.numpy(), np.ones((len(F_all), 1), np.float32)], 1)
            wv = np.linalg.solve(X.T @ X + 1.0 * np.eye(X.shape[1]), X.T @ G)
            A = G - X @ wv
            A = np.clip((A - A.mean()) / (A.std() + 1e-6), -A_CLIP, A_CLIP).astype(np.float32)
            act = taken(np.concatenate([boots[i]["nn"][boots[i]["day_mask"]] for i in use]))
            new = train_heads(m, F_all, act, A)
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
            if kl <= KL_MAX and flip <= FLIP_MAX:
                heads_new = LR.heads_of(new if alpha == 1.0 else cand)
                changed = True
            else:
                say(lines, "  gate: the heads would change too much even in part - kept")
        else:
            say(lines, "  not enough rewarded daytime life for the heads yet - kept")
    else:
        say(lines, f"  less than {MIN_HOURS:g} h of daytime decisions - heads kept")
    new_shift = zones(boots, shift, lines)
    if not np.allclose(new_shift, shift):
        changed = True
    new_mbias = learn_muse(boots, mbias, lines)
    if not np.allclose(new_mbias, mbias):
        changed = True
    if not changed:
        return finish(lines, started, "nothing new: the robot keeps what it has")
    if heads_new is None:
        heads_new = LR.heads_of(m)
    version = int(dt.datetime.now().strftime("%y%m%d%H"))
    # a dry run's candidate goes with the reports, not into the store of the robot's versions
    store = REPORTS if a.dry_run else STORE
    os.makedirs(store, exist_ok=True)
    blob_path = os.path.join(store, f"v{version}{'_dry' if a.dry_run else ''}.bin")
    open(blob_path, "wb").write(LR.pack(version, heads_new, new_shift, new_mbias, nn=LR.nn_id(m.state_dict())))
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
