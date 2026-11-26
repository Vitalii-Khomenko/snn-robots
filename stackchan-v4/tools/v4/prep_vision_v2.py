#!/usr/bin/env python3
"""Prepare compact Vision v2 dataset (out/vision_v2_data.npz) for Core 1 int8 VisionNetV2:
  5 focused outputs: [person, face, target_x, target_y, target_conf]
  Combines:
    - All 15,853 real room .snp frames from data/media/*/*.snp + YOLOv8n/YuNet .lab.npz (held-out day: 2026-10-24)
    - All 14 real_*.bin camera recordings from stackchan-life/tools/hybrid/data/vision/
    - All 20 coco_*.bin real photos
    - Stratified synthetic vision_*.bin frames
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LIFE_HYB = os.path.abspath(os.path.join(ROOT, "..", "stackchan-life", "tools", "hybrid"))
LIFE_TOOLS = os.path.abspath(os.path.join(ROOT, "..", "stackchan-life", "tools", "life"))
sys.path.insert(0, LIFE_HYB)
sys.path.insert(0, LIFE_TOOLS)

import media  # noqa: E402
import train_vision as TV  # noqa: E402

MEDIA_DIR = os.path.join(ROOT, "data", "media")
OLD_VIS = os.path.join(LIFE_HYB, "data", "vision")
OUT_PATH = os.path.join(HERE, "out", "vision_v2_data.npz")
# Indices into old 13-output VISION_OUT: person(5), face(6), target_x(9), target_y(10), target_conf(11)
KEEP_COLS = [5, 6, 9, 10, 11]


def load_media_vision():
    imgs, sides, labs, is_val = [], [], [], []
    for lp in sorted(glob.glob(os.path.join(MEDIA_DIR, "*", "*.lab.npz"))):
        base = lp[:-len(".lab.npz")]
        snp_p = base + ".snp"
        if not os.path.exists(snp_p):
            continue
        day = os.path.basename(os.path.dirname(base))
        try:
            L = np.load(lp)
            s = media.read_snapshot(snp_p)
        except Exception:
            continue
        lab = np.full(5, np.nan, np.float32)
        if s["contrast"] < 3.5:
            lab[[0, 1, 4]] = 0.0
        else:
            p, fc = float(L["vis_person"]), float(L["vis_face"])
            lab[0] = 1.0 if p >= 0.5 else (0.0 if p < 0.2 and fc < 0.5 else np.nan)
            lab[1] = 1.0 if fc >= 0.75 else (0.0 if lab[0] == 0.0 or fc < 0.2 else np.nan)
            tx, ty, conf = L["vis_target"]
            if conf > 0:
                lab[2], lab[3], lab[4] = tx, ty, 1.0
            elif lab[0] == 0.0:
                lab[4] = 0.0
        imgs.append(np.clip(np.round(s["image"] * 60.0), -127, 127).astype(np.int8))
        sides.append(s["side"].astype(np.float16))
        labs.append(lab.astype(np.float16))
        is_val.append(day == "2026-10-24")
    return np.stack(imgs), np.stack(sides), np.stack(labs), np.array(is_val, dtype=bool)


def load_old(files, max_n=None, rng=None):
    if not files:
        return np.zeros((0, 3, 24, 32), np.int8), np.zeros((0, 4), np.float16), np.zeros((0, 5), np.float16)
    im, sd, lb = TV.load(files)
    lb = lb[:, KEEP_COLS]
    if max_n is not None and len(im) > max_n:
        sel = rng.choice(len(im), max_n, replace=False)
        im, sd, lb = im[sel], sd[sel], lb[sel]
    return im.astype(np.int8), sd.astype(np.float16), lb.astype(np.float16)


def main():
    rng = np.random.default_rng(42)
    print("Loading media room .snp frames...", flush=True)
    m_im, m_sd, m_lb, m_val = load_media_vision()
    print(f"  media frames: {len(m_im)} (val {m_val.sum()}, person_pos={(m_lb[:,0]==1).sum()}, face_pos={(m_lb[:,1]==1).sum()})", flush=True)

    all_bins = sorted(glob.glob(os.path.join(OLD_VIS, "*.bin")))
    real_bins = [f for f in all_bins if os.path.basename(f).startswith("real_")]
    coco_bins = [f for f in all_bins if os.path.basename(f).startswith("coco_")]
    syn_bins = [f for f in all_bins if os.path.basename(f).startswith("vision_")]

    val_real = [f for f in real_bins if os.path.basename(f) in ("real_empty_4.bin", "real_front_7.bin")]
    tr_real = [f for f in real_bins if f not in set(val_real)]
    val_coco = coco_bins[-2:]
    tr_coco = coco_bins[:-2]
    syn_pick = [syn_bins[i] for i in np.linspace(0, len(syn_bins) - 1, min(30, len(syn_bins))).round().astype(int)]
    val_syn = syn_pick[-3:]
    tr_syn = syn_pick[:-3]

    r_im_tr, r_sd_tr, r_lb_tr = load_old(tr_real)
    r_im_v, r_sd_v, r_lb_v = load_old(val_real)
    c_im_tr, c_sd_tr, c_lb_tr = load_old(tr_coco, 25000, rng)
    c_im_v, c_sd_v, c_lb_v = load_old(val_coco, 3000, rng)
    s_im_tr, s_sd_tr, s_lb_tr = load_old(tr_syn, 20000, rng)
    s_im_v, s_sd_v, s_lb_v = load_old(val_syn, 2500, rng)

    m_tr_i, m_v_i = np.flatnonzero(~m_val), np.flatnonzero(m_val)
    im_tr = np.concatenate([m_im[m_tr_i], r_im_tr, c_im_tr, s_im_tr])
    sd_tr = np.concatenate([m_sd[m_tr_i], r_sd_tr, c_sd_tr, s_sd_tr])
    lb_tr = np.concatenate([m_lb[m_tr_i], r_lb_tr, c_lb_tr, s_lb_tr])
    src_tr = np.concatenate([
        np.full(len(m_tr_i), 3, np.uint8),
        np.full(len(r_im_tr), 2, np.uint8),
        np.full(len(c_im_tr), 1, np.uint8),
        np.full(len(s_im_tr), 0, np.uint8),
    ])

    im_v = np.concatenate([m_im[m_v_i], r_im_v, c_im_v, s_im_v])
    sd_v = np.concatenate([m_sd[m_v_i], r_sd_v, c_sd_v, s_sd_v])
    lb_v = np.concatenate([m_lb[m_v_i], r_lb_v, c_lb_v, s_lb_v])
    src_v = np.concatenate([
        np.full(len(m_v_i), 3, np.uint8),
        np.full(len(r_im_v), 2, np.uint8),
        np.full(len(c_im_v), 1, np.uint8),
        np.full(len(s_im_v), 0, np.uint8),
    ])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(OUT_PATH, im_tr=im_tr, sd_tr=sd_tr, lb_tr=lb_tr, src_tr=src_tr,
                        im_v=im_v, sd_v=sd_v, lb_v=lb_v, src_v=src_v)
    print(f"wrote {OUT_PATH}: train {im_tr.shape} val {im_v.shape} ({os.path.getsize(OUT_PATH)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
