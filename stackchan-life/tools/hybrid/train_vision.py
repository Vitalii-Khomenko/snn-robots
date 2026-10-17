"""Trains the Vision NN on data/vision/*.bin (simulator/datagen) and exports
main/hybrid/vision_model_data.h.   ./py train_vision.py [--epochs 30]"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

from models import VISION_OUT, VISION_SIGNED, VisionNet

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REC = 2304 + 16 + 52 + 8


def load(files):
    imgs, sides, labs = [], [], []
    for f in files:
        raw = np.fromfile(f, dtype=np.uint8)
        n = raw.size // REC
        a = raw[: n * REC].reshape(n, REC)
        imgs.append(a[:, :2304].copy().view(np.int8).reshape(n, 3, 24, 32))
        sides.append(a[:, 2304:2320].copy().view(np.float32))
        labs.append(a[:, 2320:2372].copy().view(np.float32))
    return np.concatenate(imgs), np.concatenate(sides), np.concatenate(labs)


def augment(img, side, lab, flip_prob=0.5):
    """img float [B,3,24,32]. Horizontal mirror, contrast, sensor noise."""
    B = img.shape[0]
    dev = img.device
    flip = torch.rand(B, device=dev) < flip_prob
    img = torch.where(flip[:, None, None, None], img.flip(3), img)
    side = side.clone()
    side[:, 0] = torch.where(flip, -side[:, 0], side[:, 0])
    lab = lab.clone()
    ml, mr = lab[:, 0].clone(), lab[:, 2].clone()
    lab[:, 0] = torch.where(flip, mr, ml)
    lab[:, 2] = torch.where(flip, ml, mr)
    lab[:, 9] = torch.where(flip, -lab[:, 9], lab[:, 9])
    gain = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.6, 1.5)
    img = img * torch.cat([gain, gain, gain], 1) if False else img * gain
    # the scene must not be memorised: small shifts and a brightness ramp across the picture (window light, a lamp at one side)
    dx, dy = int(torch.randint(-2, 3, (1,))), int(torch.randint(-2, 3, (1,)))
    img = torch.roll(img, (dy, dx), (2, 3))
    ramp = torch.linspace(-1, 1, img.shape[3], device=dev)[None, None, None, :] * torch.empty(B, 1, 1, 1, device=dev).uniform_(-0.35, 0.35)
    img = torch.cat([img[:, :1] + ramp * img[:, :1].abs().mean((1, 2, 3), keepdim=True), img[:, 1:]], 1)
    sigma = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.0, 0.06)
    img = img + torch.randn_like(img) * sigma
    return img, side, lab


def loss_fn(logits, lab):
    loss = 0.0
    parts = {}
    sig_idx = [i for i, n in enumerate(VISION_OUT) if n not in VISION_SIGNED]
    ls = lab[:, sig_idx]
    ok = ~torch.isnan(ls)  # real recordings leave unknowable labels as NaN
    ls = torch.nan_to_num(ls, 0.0).clamp(0, 1)
    bce = F.binary_cross_entropy_with_logits(logits[:, sig_idx], ls, reduction="none")
    # presence outputs matter more than the (frequent) zeros
    w = (1.0 + 2.0 * ls) * ok
    for n in ("person", "face"):  # a false person / face costs as much as a missed one
        w[:, sig_idx.index(VISION_OUT.index(n))] = 3.0 * ok[:, sig_idx.index(VISION_OUT.index(n))]
    loss = (bce * w).sum() / ok.sum().clamp(min=1)
    parts["bce"] = loss.item()
    conf = torch.nan_to_num(lab[:, VISION_OUT.index("target_conf")], 0.0)
    for n in VISION_SIGNED:
        i = VISION_OUT.index(n)
        tgt = lab[:, i]
        okt = (~torch.isnan(tgt)).float()
        err = (torch.tanh(logits[:, i]) - torch.nan_to_num(tgt, 0.0)) ** 2
        l = (err * okt * (0.05 + (conf > 0.3).float())).mean()
        loss = loss + 2.0 * l
        parts[n] = l.item()
    return loss, parts


@torch.no_grad()
def evaluate(model, img, side, lab, dev, bs=2048):
    model.eval()
    out = []
    for i in range(0, len(img), bs):
        x = torch.from_numpy(img[i:i + bs]).to(dev).float() / 60.0
        s = torch.from_numpy(side[i:i + bs]).to(dev)
        out.append(model(x, s).cpu())
    o = torch.cat(out)
    pred = torch.where(torch.tensor([n in VISION_SIGNED for n in VISION_OUT]), torch.tanh(o), torch.sigmoid(o)).numpy()
    rep = {}
    for i, n in enumerate(VISION_OUT):
        y, p = lab[:, i], pred[:, i]
        if n in VISION_SIGNED:
            m = (lab[:, VISION_OUT.index("target_conf")] > 0.3) & ~np.isnan(y)
            rep[n] = f"mae {np.abs(y[m] - p[m]).mean():.3f}"
        else:
            k = ~np.isnan(y)
            y, p = y[k], p[k]
            thr = y > 0.5
            tp = ((p > 0.5) & thr).sum()
            rep[n] = f"mae {np.abs(y - p).mean():.3f} rec {tp / max(1, thr.sum()):.2f} prec {tp / max(1, (p > 0.5).sum()):.2f} pos {thr.mean():.3f}"
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--qat", type=int, default=6)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--holdout", default="", help="comma list of real files kept out of training, e.g. real_empty_2.bin")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(HERE, "data", "vision", "*.bin")))
    isreal = lambda f: os.path.basename(f).startswith('real_')
    val_files = [f for f in files[:: 10] if not isreal(f)] + [f for f in files if os.path.basename(f) in a.holdout.split(',')]  # held-out real sessions (default none: every recording trains)
    train_files = [f for f in files if f not in val_files]
    print(len(train_files), "train files,", len(val_files), "val files")
    train_files = train_files + [f for f in train_files if os.path.basename(f).startswith('coco_')] * 3  # real photos x4
    train_files = train_files + [f for f in train_files if isreal(f)] * 7  # the robot's own camera x8
    timg, tside, tlab = load(train_files)
    vimg, vside, vlab = load(val_files)
    isc = np.concatenate([np.full(len(load([f])[0]), os.path.basename(f).startswith('coco_')) for f in val_files])
    isr = np.concatenate([np.full(len(load([f])[0]), isreal(f)) for f in val_files])
    print("frames", len(timg), len(vimg))
    dev = torch.device("cuda")
    model = VisionNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    total = (a.epochs + a.qat) * (len(timg) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=total, pct_start=0.1)
    timg_t = torch.from_numpy(timg)
    tside_t, tlab_t = torch.from_numpy(tside), torch.from_numpy(tlab)
    for ep in range(a.epochs + a.qat):
        model.quant = ep >= a.epochs
        model.train()
        perm = torch.randperm(len(timg_t))
        agg = 0.0
        steps = len(timg_t) // a.bs
        for i in range(steps):
            idx = perm[i * a.bs:(i + 1) * a.bs]
            x = timg_t[idx].to(dev).float() / 60.0
            s, y = tside_t[idx].to(dev), tlab_t[idx].to(dev)
            x, s, y = augment(x, s, y)
            loss, parts = loss_fn(model(x, s), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            agg += loss.item()
        if ep % 5 == 4 or ep == a.epochs + a.qat - 1:
            print(f"epoch {ep + 1} loss {agg / steps:.4f} quant={model.quant}")
    for name, m in (('SIM', ~isc & ~isr), ('COCO', isc), ('ROBOT CAMERA', isr)):
        if not m.any():
            continue
        print(name)
        rep = evaluate(model, vimg[m], vside[m], vlab[m], dev)
        for k, v in rep.items():
            print(f"  {k:12s} {v}")
    torch.save(model.state_dict(), os.path.join(HERE, "data", "vision_net.pt"))
    model.cpu().export(os.path.join(ROOT, "main", "hybrid", "vision_model_data.h"))
    print("exported")
