#!/usr/bin/env python3
"""Packs the networks' weights into one blob for a weights_a / weights_b flash partition (main/net/weight_store.h).

Reads the arrays the training scripts export as C headers (main/hybrid/*_model_data.h, main/voice/babble_model_data.h)
and names them "<net>.<array>" (policy.inp_w, snn.win_w, ...). The firmware takes a net's weights from the blob only
when every array that net needs is there with the size it expects.

  python3 tools/life/pack_weights.py [--out build/weights.bin] [--version N]
"""
import argparse
import os
import re
import struct
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NETS = {
    "policy": "main/hybrid/policy_model_data.h",
    "snn": "main/hybrid/snn_model_data.h",
    "vision": "main/hybrid/vision_model_data.h",
    "auditory": "main/hybrid/auditory_model_data.h",
    "babble": "main/voice/babble_model_data.h",
    "world": "main/hybrid/world_model_data.h",
    "actor": "main/hybrid/actor_model_data.h",
}
TYPES = {"int8_t": (1, np.int8), "float": (2, np.float32), "uint32_t": (3, np.uint32)}
ARRAY = re.compile(r"static const (int8_t|float|uint32_t) (\w+)\[(\d+)\] = \{(.*?)\};", re.S)
HEADER = struct.Struct("<4sIIII12x")
ENTRY = struct.Struct("<32sIIII")


def arrays(path):
    text = open(path if os.path.isabs(path) else os.path.join(ROOT, path)).read()
    for ctype, name, n, body in ARRAY.findall(text):
        tcode, dtype = TYPES[ctype]
        values = np.array([float(v.rstrip("f")) for v in body.replace("\n", "").split(",") if v.strip()])
        if len(values) != int(n):
            raise ValueError(f"{path}: {name} has {len(values)} values, declared {n}")
        yield name, tcode, values.astype(dtype)


def pack(version, override=None):
    entries = []
    for net, path in {**NETS, **(override or {})}.items():
        for name, tcode, values in arrays(path):
            full = f"{net}.{name}"
            if len(full) >= 32:
                raise ValueError(f"name too long for the blob: {full}")
            entries.append((full, tcode, values))
    offset = HEADER.size + ENTRY.size * len(entries)
    table, data = b"", b""
    for full, tcode, values in entries:
        pad = (-(offset + len(data))) % 16
        data += b"\0" * pad
        table += ENTRY.pack(full.encode(), tcode, len(values), offset + len(data), 0)
        data += values.tobytes()
    body = table + data
    head = HEADER.pack(b"SCW1", version, len(entries), HEADER.size + len(body), zlib.crc32(body) & 0xFFFFFFFF)
    return head + body, entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "weights.bin"))
    ap.add_argument("--version", type=int, default=int(time.time()) // 60, help="default: minutes since 1970")
    ap.add_argument("--override", action="append", default=[], metavar="NET=HEADER",
                    help="take a net from another header (a nightly candidate), e.g. world=data/sleep/x/world_model_data.h")
    a = ap.parse_args()
    blob, entries = pack(a.version, dict(o.split("=", 1) for o in a.override))
    if len(blob) > 0x180000:
        sys.exit(f"blob is {len(blob)} bytes, the partition holds {0x180000}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(blob)
    per_net = {}
    for full, _, values in entries:
        per_net[full.split(".")[0]] = per_net.get(full.split(".")[0], 0) + values.nbytes
    print(f"{a.out}: version {a.version}, {len(entries)} arrays, {len(blob)} bytes "
          + ", ".join(f"{k} {v // 1024} KB" for k, v in per_net.items()))


if __name__ == "__main__":
    main()
