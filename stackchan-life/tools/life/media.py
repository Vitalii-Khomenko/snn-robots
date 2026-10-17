"""Reads the robot's media takes (mediad.py): snapshots (.snp) and sound (.aud).

  snp = read_snapshot(path) -> dict(rgb=[h, w, 3] uint8, image=[3, 24, 32] float, side=[4], contrast, outputs=[18])
  aud = read_audio(path)    -> dict(rate, pcm=[n] int16, features=[ticks, 100] float)
"""
import struct

import numpy as np


def read_snapshot(path):
    d = open(path, "rb").read()
    magic, w, h, raw_len, ms = struct.unpack_from("<4sHHII", d)
    assert magic == b"SNP1", path
    o = struct.calcsize("<4sHHII")
    raw = np.frombuffer(d, np.uint8, raw_len, o)
    o += raw_len
    # RGB565, big-endian bytes as the OV camera sends them
    px = raw.reshape(-1, 2).astype(np.uint16)
    v = (px[:, 0] << 8) | px[:, 1]
    rgb = np.stack([((v >> 11) & 31) * 255 // 31, ((v >> 5) & 63) * 255 // 63, (v & 31) * 255 // 31], 1)
    rgb = rgb.reshape(h, w, 3).astype(np.uint8)
    image = np.frombuffer(d, np.float32, 3 * 768, o).reshape(3, 24, 32)
    o += 3 * 768 * 4
    side = np.frombuffer(d, np.float32, 4, o)
    o += 16
    (contrast,) = struct.unpack_from("<f", d, o)
    o += 4
    outputs = np.frombuffer(d, np.float32, (len(d) - o) // 4, o)
    return dict(rgb=rgb, image=image, side=side, contrast=contrast, outputs=outputs, ms=ms)


def read_audio(path):
    d = open(path, "rb").read()
    magic, rate, n, ticks, dim, ms = struct.unpack_from("<4sIIIII", d)
    assert magic == b"AUD1", path
    o = struct.calcsize("<4sIIIII")
    pcm = np.frombuffer(d, np.int16, n, o)
    o += 2 * n
    feat = np.frombuffer(d, np.float32, ticks * dim, o).reshape(ticks, dim)
    return dict(rate=rate, pcm=pcm, features=feat, ms=ms)
