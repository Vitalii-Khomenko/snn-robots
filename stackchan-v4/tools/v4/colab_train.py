#!/usr/bin/env python3
"""Run one or more training scripts on a single Google Colab T4 GPU session via `colab` CLI,
with automatic retry on transient connection drops, output download, Google Drive mirroring
(`~/Work/MLPojekt/stackchan-v4`), and guaranteed `colab stop` in `finally`.
"""
import argparse
import os
import shutil
import subprocess
import time

DRIVE_DIR = os.path.expanduser("~/Work/MLPojekt/stackchan-v4")


def run(cmd, check=True, timeout=None, retries=1):
    for attempt in range(1, retries + 1):
        print(f"+ (try {attempt}/{retries}) " + " ".join(cmd), flush=True)
        r = subprocess.run(cmd, check=False, timeout=timeout)
        if r.returncode == 0 or not check:
            return r
        if attempt < retries:
            time.sleep(5 * attempt)
    raise subprocess.CalledProcessError(r.returncode, cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Colab session name")
    ap.add_argument("--gpu", default="T4")
    ap.add_argument("--upload", action="append", default=[], help="local:remote upload pair")
    ap.add_argument("--script", action="append", required=True, help="local Python script(s) to execute sequentially")
    ap.add_argument("--download", action="append", default=[], help="remote:local download pair")
    ap.add_argument("--timeout", type=int, default=0, help="colab exec timeout (0 = unlimited)")
    ap.add_argument("--mirror", action="store_true", default=True)
    a = ap.parse_args()

    t0 = time.time()
    created = False
    try:
        # Clean any stale session of the same name first
        subprocess.run(["colab", "stop", "-s", a.name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        run(["colab", "new", "--gpu", a.gpu, "-s", a.name], retries=3)
        created = True
        for spec in a.upload:
            local, remote = spec.split(":", 1)
            run(["colab", "upload", "-s", a.name, local, remote], retries=3)
        for script in a.script:
            run(["colab", "exec", "-s", a.name, "--timeout", str(a.timeout), "-f", script])
        os.makedirs(os.path.join(DRIVE_DIR, a.name), exist_ok=True)
        for spec in a.download:
            remote, local = spec.split(":", 1)
            os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
            run(["colab", "download", "-s", a.name, remote, local], retries=3)
            if a.mirror and os.path.isdir(os.path.expanduser("~/Work/MLPojekt")):
                shutil.copy2(local, os.path.join(DRIVE_DIR, a.name, os.path.basename(local)))
        print(f"[colab_train] finished '{a.name}' in {(time.time() - t0) / 60:.1f} min", flush=True)
    finally:
        if created:
            print(f"[colab_train] releasing Colab session '{a.name}'...", flush=True)
            subprocess.run(["colab", "stop", "-s", a.name], check=False)


if __name__ == "__main__":
    main()
