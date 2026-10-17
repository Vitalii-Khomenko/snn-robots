#!/usr/bin/env python3
"""Takes what the robot sees and hears now and then, for the perception networks' training on the PC (platform B;
the owner allowed frames and sound to the home PC, 2026-10-17). Nothing leaves the home network.

Every few minutes (and at random in between) it fetches GET /snapshot and GET /audio (main/net/ota_server.h) and keeps
them: data/media/YYYY-MM-DD/HHMMSS.snp (raw RGB565 frame + the front end's frame + the Vision NN's outputs) and
.aud (2 s of sound + the Auditory NN's inputs). tools/life/media.py reads them.

  python3 tools/life/mediad.py [--every 180]        (systemd user service tools/life/mediad.service)
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
    ap.add_argument("--every", type=float, default=180.0, help="mean seconds between takes")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    print(f"mediad: a take every ~{a.every:.0f} s -> {os.path.join(ROOT, 'data', 'media')}", flush=True)
    while True:
        stamp = time.strftime("%Y-%m-%d/%H%M%S")
        folder = os.path.join(ROOT, "data", "media", stamp.split("/")[0])
        os.makedirs(folder, exist_ok=True)
        for path, ext in (("/snapshot", "snp"), ("/audio", "aud")):
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
