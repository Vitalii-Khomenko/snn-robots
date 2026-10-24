"""The big nets on the PC that label the robot's own frames and sounds (platform B: perception learns from its life).

  person / object detector   YOLOv8n (COCO 80 classes)        data/teachers/yolo/yolov8n.pt
  faces                      YuNet (OpenCV zoo)               data/teachers/yunet/face_detection_yunet_2023mar.onnx
  speech or not, per 32 ms   Silero VAD (pip silero-vad)
  what sound it is (527)     PANNs CNN14 (AudioSet)           data/teachers/panns/Cnn14_mAP=0.431.pth

Nothing leaves the PC. Loaded lazily, on the GPU when there is one.
"""
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
TEACH = os.path.abspath(os.path.join(HERE, "..", "..", "data", "teachers"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- vision
class Vision:
    def __init__(self):
        import cv2
        from ultralytics import YOLO
        self.cv2 = cv2
        self.yolo = YOLO(os.path.join(TEACH, "yolo", "yolov8n.pt"))
        self.names = [self.yolo.names[i] for i in range(len(self.yolo.names))]
        self.face = cv2.FaceDetectorYN.create(os.path.join(TEACH, "yunet", "face_detection_yunet_2023mar.onnx"), "",
                                              (640, 480), 0.6, 0.3, 50)

    def label(self, rgb):
        """rgb [h, w, 3] uint8 -> dict(objects=[80] max confidence per COCO class, persons=[(conf, box)], faces=[(score, box)])
        with boxes (x0, y0, x1, y1) in 0..1 image coordinates."""
        h, w = rgb.shape[:2]
        r = self.yolo.predict(rgb[:, :, ::-1].copy(), imgsz=320, conf=0.1, verbose=False, device=DEV)[0]
        objects = np.zeros(len(self.names), np.float32)
        persons = []
        for c, k, b in zip(r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.xyxyn.cpu().numpy()):
            objects[k] = max(objects[k], c)
            if self.names[k] == "person":
                persons.append((float(c), [float(v) for v in b]))
        big = self.cv2.resize(rgb[:, :, ::-1], (640, 480), interpolation=self.cv2.INTER_CUBIC)  # faces at 160x120 are tiny
        _, det = self.face.detect(big)
        faces = []
        if det is not None:
            for d in det:
                x, y, bw, bh = d[:4]
                faces.append((float(d[-1]), [float(x / 640), float(y / 480), float((x + bw) / 640), float((y + bh) / 480)]))
        return dict(objects=objects, persons=persons, faces=faces)


# ---------------------------------------------------------------- sound
def _block(cin, cout):
    return nn.ModuleDict(dict(conv1=nn.Conv2d(cin, cout, 3, 1, 1, bias=False), conv2=nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
                              bn1=nn.BatchNorm2d(cout), bn2=nn.BatchNorm2d(cout)))


class Cnn14(nn.Module):
    """PANNs CNN14 (Kong et al. 2020), 32 kHz, 64 mel; the published checkpoint's layout."""

    def __init__(self):
        super().__init__()
        from torchlibrosa.stft import LogmelFilterBank, Spectrogram
        self.spectrogram_extractor = Spectrogram(n_fft=1024, hop_length=320, win_length=1024, window="hann", center=True,
                                                 pad_mode="reflect", freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=32000, n_fft=1024, n_mels=64, fmin=50, fmax=14000, ref=1.0, amin=1e-10,
                                                 top_db=None, freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(64)
        chans = [1, 64, 128, 256, 512, 1024, 2048]
        for i in range(6):
            setattr(self, f"conv_block{i + 1}", _block(chans[i], chans[i + 1]))
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, 527)

    def forward(self, wav):
        x = self.logmel_extractor(self.spectrogram_extractor(wav))  # [B, 1, T, 64]
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for i in range(6):
            b = getattr(self, f"conv_block{i + 1}")
            x = F.relu(b["bn1"](b["conv1"](x)))
            x = F.relu(b["bn2"](b["conv2"](x)))
            x = F.avg_pool2d(x, 2) if i < 5 else F.avg_pool2d(x, 1)
        x = x.mean(3)
        x = x.max(2)[0] + x.mean(2)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc_audioset(x))


class Sound:
    def __init__(self):
        from silero_vad import load_silero_vad
        self.vad = load_silero_vad()
        self.panns = Cnn14()
        ck = torch.load(os.path.join(TEACH, "panns", "Cnn14_mAP=0.431.pth"), map_location="cpu", weights_only=True)
        self.panns.load_state_dict(ck["model"], strict=False)
        self.panns.to(DEV).eval()
        with open(os.path.join(TEACH, "panns", "class_labels_indices.csv")) as f:
            self.names = [row["display_name"] for row in csv.DictReader(f)]

    @staticmethod
    def resample(pcm, rate, to):
        from scipy.signal import resample_poly
        g = np.gcd(rate, to)
        return resample_poly(pcm.astype(np.float32) / 32768.0, to // g, rate // g).astype(np.float32)

    @torch.no_grad()
    def label(self, pcm, rate, ticks, tick_ms=50):
        """pcm int16 -> dict(classes=[527] clip probabilities, speech=[ticks] fraction of each brain tick that is speech).
        The ticks are the last `ticks` x 50 ms of the clip (the robot's feature ring ends with the clip)."""
        w32 = self.resample(pcm, rate, 32000)
        classes = self.panns(torch.from_numpy(w32)[None].to(DEV))[0].cpu().numpy()
        w16 = torch.from_numpy(self.resample(pcm, rate, 16000))
        probs = []
        self.vad.reset_states()
        for i in range(0, len(w16) - 511, 512):  # 32 ms windows
            probs.append(float(self.vad(w16[i:i + 512], 16000)))
        probs = np.array(probs, np.float32)
        t_win = (np.arange(len(probs)) + 0.5) * 0.032  # window centres, s
        dur = len(pcm) / rate
        speech = np.zeros(ticks, np.float32)
        for k in range(ticks):
            t0 = dur - (ticks - k) * tick_ms / 1000.0
            m = (t_win >= t0 - 0.016) & (t_win < t0 + tick_ms / 1000.0 + 0.016)
            speech[k] = probs[m].max() if m.any() else 0.0
        return dict(classes=classes, speech=speech)
