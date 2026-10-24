#!/usr/bin/env python3
"""Compresses the diary's finished days: data/life/<day>/<hour>.scl -> <hour>.scl.gz and log.txt -> log.txt.gz
(gzip; the binary hours shrink ~5x, the text ~12x). Today is never touched: lifelogd.py is still writing it.
tools/v4/diary.py and legacy_diary.py read either form.

  python3 tools/robot/compress_diary.py            every finished day
  python3 tools/robot/compress_diary.py --dry-run  only say what it would do
"""
import argparse
import datetime
import glob
import gzip
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LIFE = os.path.join(ROOT, "data", "life")


def gzip_file(path):
    """path -> path.gz, checked by reading it back whole, then the original is removed."""
    tmp = path + ".gz.part"
    with open(path, "rb") as src, gzip.open(tmp, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
    with open(path, "rb") as a, gzip.open(tmp, "rb") as b:
        if a.read() != b.read():
            os.remove(tmp)
            raise RuntimeError(f"{path}: the compressed copy differs, original kept")
    os.replace(tmp, path + ".gz")
    os.remove(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    today = datetime.date.today().isoformat()
    before = after = 0
    for day in sorted(d for d in os.listdir(LIFE) if d[:2] == "20" and d < today):
        files = sorted(glob.glob(os.path.join(LIFE, day, "*.scl")) + glob.glob(os.path.join(LIFE, day, "log.txt")))
        if not files:
            continue
        size = sum(os.path.getsize(f) for f in files)
        if a.dry_run:
            print(f"{day}: {len(files)} files, {size / 1e6:.0f} MB")
            continue
        for f in files:
            gzip_file(f)
        packed = sum(os.path.getsize(f + ".gz") for f in files)
        before, after = before + size, after + packed
        print(f"{day}: {size / 1e6:.0f} MB -> {packed / 1e6:.0f} MB", flush=True)
    if before:
        print(f"total {before / 1e9:.2f} GB -> {after / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
