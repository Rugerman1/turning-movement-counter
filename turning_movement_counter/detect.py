"""Run YOLO + ByteTrack over a video and feed the tracks to the counter.

Ultralytics, PyTorch and OpenCV are imported inside the functions so the rest
of the app (and the tests) load without them.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable

from .counting import CountSettings, Movement, TrackAccumulator
from .zones import ZoneSet

# COCO class ids the stock YOLO models use for vehicles.
COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

MODELS = {
    "yolo26n.pt": "YOLO26 nano (bundled, fastest)",
    "yolo26s.pt": "YOLO26 small (downloads on first use, more accurate)",
    "yolo26m.pt": "YOLO26 medium (downloads on first use, best with a GPU)",
    "yolo11n.pt": "YOLO11 nano (downloads on first use)",
}
DEFAULT_MODEL = "yolo26n.pt"


def app_dir() -> str:
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(base, "TurningMovementCounter")


def model_path(name: str = DEFAULT_MODEL, download: bool = True) -> str:
    """Bundled copy first (inside the exe), then the user's model folder, downloading if needed."""
    here = getattr(sys, "_MEIPASS", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    for d in (os.path.join(here, "models"), os.path.join(app_dir(), "models")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    target = os.path.join(app_dir(), "models", name)
    if download:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        from ultralytics.utils.downloads import attempt_download_asset

        attempt_download_asset(target)
    return target


@dataclass
class VideoInfo:
    fps: float
    frames: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps else 0.0


def video_info(path: str) -> VideoInfo:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise OSError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    info = VideoInfo(fps, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


def first_frame(path: str, at_s: float = 0.0):
    """RGB numpy array of the frame at `at_s` seconds."""
    import cv2

    cap = cv2.VideoCapture(path)
    if at_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, at_s * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise OSError(f"Could not read a frame from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


@dataclass
class RunOptions:
    model: str = DEFAULT_MODEL
    analysis_fps: float = 10.0     # frames per second sent to the detector
    conf: float = 0.3
    imgsz: int = 960
    device: str | None = None      # None lets Ultralytics pick (GPU if available)
    annotated_path: str | None = None
    count: CountSettings = field(default_factory=CountSettings)


@dataclass
class CountRun:
    source: str
    movements: list[Movement]
    stats: dict
    info: VideoInfo
    seconds_analyzed: float


def load_model(name: str = DEFAULT_MODEL):
    from ultralytics import YOLO

    return YOLO(model_path(name))


def count_video(path: str, zones: ZoneSet, opts: RunOptions | None = None,
                progress: Callable[[float, str], None] | None = None,
                cancelled: Callable[[], bool] | None = None, model=None) -> CountRun:
    import cv2

    opts = opts or RunOptions()
    info = video_info(path)
    stride = max(1, round(info.fps / opts.analysis_fps))
    opts.count.frame_diag = math.hypot(info.width, info.height)
    zs = _scaled(zones, info.width, info.height)
    acc = TrackAccumulator(zs, opts.count)
    model = model or load_model(opts.model)
    tracker = "bytetrack.yaml"

    writer = None
    if opts.annotated_path:
        writer = cv2.VideoWriter(opts.annotated_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 info.fps / stride, (info.width, info.height))

    cap = cv2.VideoCapture(path)
    idx = 0
    last_t = 0.0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % stride:
                idx += 1
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = idx / info.fps
            last_t = t
            res = model.track(frame, persist=True, tracker=tracker, classes=list(COCO_VEHICLES),
                              conf=opts.conf, imgsz=opts.imgsz, device=opts.device, verbose=False)[0]
            boxes = res.boxes
            if boxes is not None and boxes.id is not None:
                for (x1, y1, x2, y2), tid, c in zip(boxes.xyxy.tolist(), boxes.id.int().tolist(),
                                                    boxes.cls.int().tolist()):
                    acc.add(tid, t, (x1 + x2) / 2, y2, COCO_VEHICLES.get(c, "car"))
            if writer is not None:
                writer.write(_annotate(res.plot(), zs))
            if progress and info.frames:
                progress(min(idx / info.frames, 1.0), f"{t / 60:.1f} of {info.duration_s / 60:.1f} min")
            if cancelled and cancelled():
                break
            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    movements, stats = acc.movements()
    return CountRun(path, movements, stats, info, last_t)


def _scaled(zones: ZoneSet, width: int, height: int) -> ZoneSet:
    """Zones are drawn on the preview frame; rescale if the video size differs."""
    if not zones.frame_size or tuple(zones.frame_size) == (width, height):
        return zones
    sx, sy = width / zones.frame_size[0], height / zones.frame_size[1]
    from .zones import Zone

    return ZoneSet([Zone(z.approach, [(x * sx, y * sy) for x, y in z.points]) for z in zones.zones],
                   frame_size=(width, height))


def _annotate(bgr, zones: ZoneSet):
    import cv2
    import numpy as np

    for z in zones.zones:
        pts = np.array(z.points, np.int32).reshape(-1, 1, 2)
        cv2.polylines(bgr, [pts], True, (0, 220, 255), 2)
        x, y = pts[0][0]
        cv2.putText(bgr, z.approach, (int(x) + 6, int(y) + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
    return bgr


def selftest_model() -> int:
    """Load the bundled model and track one blank frame; used by the exe self-test in CI."""
    import numpy as np

    model = load_model()
    frame = np.zeros((480, 640, 3), np.uint8)
    model.track(frame, persist=True, tracker="bytetrack.yaml", classes=list(COCO_VEHICLES), verbose=False)
    return len(model.names)
