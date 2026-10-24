#!/usr/bin/env python3
"""Takes what the robot sees and hears now and then, for the perception networks' training on the PC (platform B;
the owner allowed frames and sound to the home PC, 2026-10-17). Nothing leaves the home network.

Every ~30 s (at random in between) it fetches GET /snapshot, and with every third frame GET /audio (main/net/ota_server.h) and keeps
them: data/media/YYYY-MM-DD/HHMMSS.snp (raw RGB565 frame + the front end's frame + the Vision NN's outputs) and
.aud (2 s of sound + the Auditory NN's inputs). tools/life/media.py reads them.

  python3 tools/life/mediad.py [--every 30] [--audio-every 3]   (frames every 30 s since 2026-10-18; was 180 s)        (systemd user service tools/life/mediad.service)
"""
import argparse
import os
import random
import sys
import time
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ota  # noqa: E402 - address and token

ROOT = ota.ROOT


def fetch(path):
    import urllib.request
    req = urllib.request.Request(f"http://{ota.address()}{path}", headers={"X-Token": ota.token()})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=30.0, help="mean seconds between frames")
    ap.add_argument("--audio-every", type=int, default=3, help="sound with every n-th frame")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    print(f"mediad: a frame every ~{a.every:.0f} s, sound with every {a.audio_every}. -> "
          f"{os.path.join(ROOT, 'data', 'media')}", flush=True)
    n = 0
    while True:
        stamp = time.strftime("%Y-%m-%d/%H%M%S")
        folder = os.path.join(ROOT, "data", "media", stamp.split("/")[0])
        os.makedirs(folder, exist_ok=True)
        takes = (("/snapshot", "snp"), ("/audio", "aud")) if n % a.audio_every == 0 else (("/snapshot", "snp"),)
        n += 1
        for path, ext in takes:
            try:
                data = fetch(path)
                with open(os.path.join(folder, f"{stamp.split('/')[1]}.{ext}"), "wb") as f:
                    f.write(data)
            except (urllib.error.URLError, OSError) as e:
                print(f"mediad: {path} failed: {e}", flush=True)
        if a.once:
            return
        time.sleep(random.uniform(0.3, 1.7) * a.every)


if __name__ == "__main__":
    main()
