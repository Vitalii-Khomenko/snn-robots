"""Real photos for the Vision NN: COCO val2017 crops turned into the robot's view.

Each photo gives several 4:3 crops at 160x120 greyscale with a camera-like look
(soft, exposure and gamma jitter, sensor noise), as (previous, current) frame
pairs - the previous one shifted by a few pixels (the head turning) or equal
(nothing moves). Labels come from COCO's person boxes and face keypoints:
person, face, target. Nothing in a still photo moves or approaches, so the
motion / loom / recede labels are zero (valid: the room is static).

  ./py make_coco_vision.py [--crops 6]   -> data/vision/coco_*.bin (same format as the simulator's)
"""
import argparse
import json
import os
import subprocess

import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.abspath(os.path.join(HERE, "..", "..", "..", "datasets"))
FEED = os.path.abspath(os.path.join(HERE, "..", "..", "simulator", "build", "frames_feed"))
OUT = os.path.join(HERE, "data", "vision")
W, H = 160, 120


def rgb565(gray):
    v = gray.astype(np.uint16)
    px = ((v >> 3) << 11) | ((v >> 2) << 5) | (v >> 3)
    return np.stack([(px >> 8).astype(np.uint8), (px & 0xFF).astype(np.uint8)], -1).reshape(-1).tobytes()


def look(img, rng, gain, gamma, noise):
    g = np.asarray(img.convert("L").filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2))), np.float32) / 255.0
    g = np.clip(gain * g ** gamma, 0, 1) * 255.0
    g = g + rng.normal(0, noise, g.shape)
    return np.clip(g, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=int, default=6)
    ap.add_argument("--per-file", type=int, default=1500)
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    ann = json.load(open(os.path.join(DS, "annotations", "person_keypoints_val2017.json")))
    persons = {}
    for p in ann["annotations"]:
        if p["iscrowd"] or p["area"] < 1500:
            continue
        persons.setdefault(p["image_id"], []).append(p)
    images = ann["images"]
    rng.shuffle(images)
    pairs, labels = [], []
    file_idx = 0

    def flush():
        nonlocal pairs, labels, file_idx
        if not pairs:
            return
        raw = os.path.join(OUT, "coco_frames.tmp")
        with open(raw, "wb") as f:
            for p, c in pairs:
                f.write(p + c)
        feed = os.path.join(OUT, "coco_feed.tmp")
        subprocess.run([FEED, raw, feed], check=True, stdout=subprocess.DEVNULL)
        rec = 2304 + 16 + 4
        d = np.fromfile(feed, np.uint8)
        n = d.size // rec
        d = d[: n * rec].reshape(n, rec)
        out = bytearray()
        for i in range(n):
            lab = np.array(labels[i], np.float32).copy()
            contrast = d[i, 2320:2324].copy().view(np.float32)[0]
            if contrast < 3.5:  # blind
                lab[[5, 6, 7, 8, 9, 10, 11]] = 0
            out += d[i, :2320].tobytes() + lab.tobytes() + np.zeros(2, np.float32).tobytes()
        with open(os.path.join(OUT, f"coco_{file_idx:03d}.bin"), "wb") as f:
            f.write(out)
        file_idx += 1
        pairs, labels = [], []
        os.remove(raw)
        os.remove(feed)

    for k, im in enumerate(images):
        path = os.path.join(DS, "val2017", im["file_name"])
        img = Image.open(path).convert("RGB")
        iw, ih = img.size
        plist = persons.get(im["id"], [])
        for _ in range(a.crops):
            cw = rng.uniform(0.4, 0.8) * min(iw, ih * 4 / 3)
            ch = cw * 3 / 4
            x0 = rng.uniform(0, max(1e-3, iw - cw))
            y0 = rng.uniform(0, max(1, ih - ch))
            sx, sy = (int(rng.integers(-3, 4)), int(rng.integers(-3, 4))) if rng.random() < 0.6 else (0, 0)
            gain, gamma, noise = rng.uniform(0.35, 1.8), rng.uniform(0.7, 1.5), rng.uniform(0.8, 3.0)
            crop = lambda dx: img.resize((W, H), Image.BILINEAR, box=(x0 + dx, y0, x0 + dx + cw, y0 + ch))
            shift_px = sx * cw / W * 5  # shift of 32x24 pooled pixels = 5 sample pixels
            if not (0 <= x0 - shift_px <= iw - cw):
                sx, shift_px = 0, 0.0
            cur = look(crop(0), rng, gain, gamma, noise)
            prev = look(crop(-shift_px), rng, gain, gamma, noise)
            # labels
            person = face = tx = ty = conf = 0.0
            best_area = 0.0
            for p in plist:
                bx, by, bw, bh = p["bbox"]
                ix0, iy0, ix1, iy1 = max(bx, x0), max(by, y0), min(bx + bw, x0 + cw), min(by + bh, y0 + ch)
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                inter = (ix1 - ix0) * (iy1 - iy0)
                vis = inter / (bw * bh)
                size = inter / (cw * ch)
                if size < 0.04:
                    continue
                person = max(person, min(1.0, 2 * vis))
                kp = np.array(p["keypoints"]).reshape(17, 3)
                head = kp[:3]
                head_ok = (head[:, 2] == 2).all() and (head[:, 0] > x0).all() and (head[:, 0] < x0 + cw).all() \
                    and (head[:, 1] > y0).all() and (head[:, 1] < y0 + ch).all() and bh > 0.22 * ch
                if head_ok:
                    face = 1.0
                if inter > best_area:
                    best_area = inter
                    if head_ok:
                        hx, hy = head[:, 0].mean(), head[:, 1].mean()
                    else:
                        hx, hy = (ix0 + ix1) / 2, iy0 + 0.25 * (iy1 - iy0)
                    tx, ty, conf = (hx - x0) / cw * 2 - 1, (hy - y0) / ch * 2 - 1, 1.0
            labels.append([0, 0, 0, 0, 0, person, face, 0, 0, tx, ty, conf, 0])
            pairs.append((rgb565(prev), rgb565(cur)))
            if len(pairs) >= a.per_file:
                flush()
        if k % 500 == 499:
            print(k + 1, "images", flush=True)
    flush()
    print("coco files:", file_idx)


main()
