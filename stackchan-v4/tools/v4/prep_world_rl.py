#!/usr/bin/env python3
"""Prepare 2 Hz real-life trajectories from all v4 days (2026-10-25..latest) for Colab T4 World Model + Offline RL (IQL).
Writes out/world_rl_data.npz (~15-30 MB):
  obs [N, 17]: 13 channels + sin(hour), cos(hour), head_yaw/60, head_pitch/30
  act [N, 10]: said, gesture, 4 styles (calm/ask/play/grumble), head_look, head_scan, gaze_yaw, gaze_pitch
  rew [N]:     summed real human reward over each 0.5 s step
  done [N]:    1 at episode/gap boundaries
  day_idx [N]: day index (newest full day held out for validation)
"""
import os
import numpy as np

import diary as D
from reward_v4 import reward
from spec_v4 import CHANNELS, HEAD_MODES, NN_OUT

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "world_rl_data.npz")
STEP = 10  # 20 Hz -> 2 Hz (0.5 s per step)


def main():
    days = sorted(d for d in os.listdir(D.LIFE) if d[:2] == "20" and d >= "2026-10-25")
    if len(days) > 16:
        pick = np.linspace(0, len(days) - 1, 16).round().astype(int)
        days = [days[i] for i in sorted(set(pick))]
    ch0 = D.col("ch_" + CHANNELS[0])
    nn0 = D.col("nn_" + NN_OUT[0])
    i_hr = D.col("raw_hour")
    i_yaw = D.col("head_yaw_deg")
    i_pit = D.col("head_pitch_deg")
    i_said = D.col("said")
    i_gest = D.col("gesture")
    st_idx = [NN_OUT.index("style_" + s) for s in ("calm", "ask", "play", "grumble")]
    hm_look = NN_OUT.index("head_look")
    hm_scan = NN_OUT.index("head_scan")
    i_gy = NN_OUT.index("gaze_yaw")
    i_gp = NN_OUT.index("gaze_pitch")

    obs_l, act_l, rew_l, done_l, day_l = [], [], [], [], []
    for di, day in enumerate(days):
        t, _ = D.load(day)
        if t is None or len(t["v"]) < 2000:
            continue
        r, _ = reward(t)
        v, f, seq, boot = t["v"], t["flags"], t["seq"], t["boot"]
        day_m = ((f & D.FLAG_DRIVES) > 0) & ((f & D.FLAG_NIGHT) == 0)
        if day_m.sum() < 2000:
            continue
        # Process per boot in chunks of STEP ticks
        for b in np.unique(boot):
            m = boot == b
            vb, fb, sq, rb, dm = v[m], f[m], seq[m], r[m], day_m[m]
            o = np.argsort(sq)
            vb, fb, sq, rb, dm = vb[o], fb[o], sq[o], rb[o], dm[o]
            n = (len(vb) // STEP) * STEP
            if n < STEP * 20:
                continue
            # Keep daytime steps only
            dm_s = dm[:n].reshape(-1, STEP).all(1)
            if dm_s.sum() < 20:
                continue
            ch = np.nan_to_num(vb[:n:STEP, ch0:ch0 + len(CHANNELS)], 0.0)
            hr = np.nan_to_num(vb[:n:STEP, i_hr], 12.0) * (2.0 * np.pi / 24.0)
            hy = np.clip(np.nan_to_num(vb[:n:STEP, i_yaw], 0.0) / 60.0, -1.5, 1.5)
            hp = np.clip(np.nan_to_num(vb[:n:STEP, i_pit], 0.0) / 30.0, -1.5, 1.5)
            obs = np.column_stack([ch, np.sin(hr), np.cos(hr), hy, hp]).astype(np.float16)

            said_s = (np.nan_to_num(vb[:n, i_said], 0.0).reshape(-1, STEP).max(1) > 0.5).astype(np.float32)
            gest_s = (np.nan_to_num(vb[:n, i_gest], 0.0).reshape(-1, STEP).max(1) > 0.5).astype(np.float32)
            nn_s = np.nan_to_num(vb[:n:STEP, nn0:nn0 + len(NN_OUT)], 0.0)
            act = np.column_stack([said_s, gest_s, nn_s[:, st_idx], nn_s[:, hm_look], nn_s[:, hm_scan],
                                   nn_s[:, i_gy], nn_s[:, i_gp]]).astype(np.float16)
            rew = rb[:n].reshape(-1, STEP).sum(1).astype(np.float16)
            sq_s = sq[:n:STEP].astype(np.int64)
            gap = np.r_[np.diff(sq_s) > STEP * 3, True]
            done = (gap | ~np.r_[dm_s[1:], False]).astype(np.uint8)
            keep = dm_s
            obs_l.append(obs[keep])
            act_l.append(act[keep])
            rew_l.append(rew[keep])
            done_l.append(done[keep])
            day_l.append(np.full(int(keep.sum()), di, np.int16))
        print(f"world {day}: total steps {sum(len(x) for x in obs_l)}", flush=True)

    obs = np.concatenate(obs_l)
    act = np.concatenate(act_l)
    rew = np.concatenate(rew_l)
    done = np.concatenate(done_l)
    day_idx = np.concatenate(day_l)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, obs=obs, act=act, rew=rew, done=done, day_idx=day_idx, days=np.array(days))
    print(f"wrote {OUT}: {len(obs)} 2Hz steps ({len(obs)/7200:.1f} daytime hours, {os.path.getsize(OUT)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
