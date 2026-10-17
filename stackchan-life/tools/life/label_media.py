"""Labels the robot's media takes (mediad.py) with the big nets on the PC (teachers.py).

  ../hybrid/py ../life/label_media.py [--redo]

Every take data/media/DATE/HHMMSS.snp / .aud gets HHMMSS.lab.npz next to it:
  vis_objects [80]   the detector's max confidence per COCO class
  vis_person, vis_face        best person / face confidence (0 when none)
  vis_target [3]     x, y (-1..1 image coordinates, the Vision NN's convention) and conf of the face, else of the
                     biggest person's head (top quarter of the box)
  aud_classes [527]  PANNs clip probabilities (AudioSet)
  aud_speech [ticks] Silero VAD speech probability for each 50 ms brain tick of the robot's features
Labels are kept, not recomputed: only new takes are labelled (a night's ~480 takes take a minute on the GPU).
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import media  # noqa: E402

MEDIA = os.path.abspath(os.path.join(HERE, "..", "..", "data", "media"))


def target_of(lab):
    faces = [f for f in lab["faces"] if f[0] >= 0.8]
    if faces:
        _, (x0, y0, x1, y1) = max(faces, key=lambda f: (f[1][2] - f[1][0]) * (f[1][3] - f[1][1]))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    else:
        persons = [p for p in lab["persons"] if p[0] >= 0.5]
        if not persons:
            return np.array([0.0, 0.0, 0.0], np.float32)
        _, (x0, y0, x1, y1) = max(persons, key=lambda p: (p[1][2] - p[1][0]) * (p[1][3] - p[1][1]))
        cx, cy = (x0 + x1) / 2, y0 + 0.25 * (y1 - y0)
    return np.array([cx * 2 - 1, cy * 2 - 1, 1.0], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    import teachers
    vis = snd = None
    done = 0
    for snp in sorted(glob.glob(os.path.join(MEDIA, "*", "*.snp"))):
        base = snp[:-4]
        out = base + ".lab.npz"
        if os.path.exists(out) and not a.redo:
            continue
        rec = {}
        try:
            s = media.read_snapshot(snp)
        except Exception as e:  # a torn take (the PC slept, the robot rebooted): skipped
            print("skip", snp, e)
            continue
        vis = vis or teachers.Vision()
        lab = vis.label(s["rgb"])
        rec["vis_objects"] = lab["objects"]
        rec["vis_person"] = np.float32(max([p[0] for p in lab["persons"]], default=0.0))
        rec["vis_face"] = np.float32(max([f[0] for f in lab["faces"]], default=0.0))
        rec["vis_target"] = target_of(lab)
        if os.path.exists(base + ".aud"):
            try:
                au = media.read_audio(base + ".aud")
                snd = snd or teachers.Sound()
                al = snd.label(au["pcm"], au["rate"], len(au["features"]))
                rec["aud_classes"] = al["classes"].astype(np.float16)
                rec["aud_speech"] = al["speech"]
            except Exception as e:
                print("skip sound", base, e)
        np.savez_compressed(out, **rec)
        done += 1
    print(f"label_media: {done} new takes labelled")


if __name__ == "__main__":
    main()
