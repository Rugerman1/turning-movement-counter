"""A drawn four-leg intersection and fake vehicle tracks with known counts.

Used by the tests (the counting logic is checked against the exact number of
vehicles generated for each movement) and by Help > Run demo count, so the
app can be tried without a video or the YOLO model.
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from .zones import BOUND, Zone, ZoneSet, turn_for

W, H = 1280, 720
CX, CY = 640, 360

# Lane centerlines, right-hand traffic.  Inbound = toward the intersection.
INBOUND = {"N": ("x", 600), "S": ("x", 680), "E": ("y", 330), "W": ("y", 390)}
OUTBOUND = {"N": ("x", 680), "S": ("x", 600), "E": ("y", 390), "W": ("y", 330)}
FAR = {"N": -40, "S": H + 40, "E": W + 40, "W": -40}


def demo_zones() -> ZoneSet:
    return ZoneSet([
        Zone("N", [(550, 30), (730, 30), (730, 270), (550, 270)]),
        Zone("E", [(770, 290), (1250, 290), (1250, 430), (770, 430)]),
        Zone("S", [(550, 450), (730, 450), (730, 690), (550, 690)]),
        Zone("W", [(30, 290), (510, 290), (510, 430), (30, 430)]),
    ], frame_size=(W, H))


def demo_frame() -> np.ndarray:
    """An RGB picture of the intersection the demo zones fit."""
    img = np.full((H, W, 3), (86, 120, 74), np.uint8)       # grass
    img[:, 560:720] = (60, 60, 64)                          # N-S road
    img[300:420, :] = (60, 60, 64)                          # E-W road
    img[:300, 638:642] = img[420:, 638:642] = (230, 200, 40)  # center lines
    img[358:362, :560] = img[358:362, 720:] = (230, 200, 40)
    for y in range(0, H, 40):                               # lane edge dashes
        img[y:y + 20, 562:564] = img[y:y + 20, 716:718] = 220
    img[282:288, 560:640] = img[432:438, 640:720] = 240     # stop bars
    img[300:360, 542:548] = img[360:420, 732:738] = 240
    return img


def _point(leg: str, inbound: bool, along: float) -> tuple[float, float]:
    """A point on a lane, `along` = 0 at the far end of the leg, 1 at the intersection edge."""
    axis, c = (INBOUND if inbound else OUTBOUND)[leg]
    edge = {"N": CY - 60, "S": CY + 60, "E": CX + 80, "W": CX - 80}[leg]
    pos = FAR[leg] + (edge - FAR[leg]) * along
    return (c, pos) if axis == "x" else (pos, c)


def path(entry: str, exit_: str) -> list[tuple[float, float]]:
    pts = [_point(entry, True, 0.0), _point(entry, True, 1.0)]
    if turn_for(entry, exit_) == "U":
        pts.append((CX, CY))
    pts += [_point(exit_, False, 1.0), _point(exit_, False, 0.0)]
    return pts


def _sample(pts: list[tuple[float, float]], speed: float, fps: float) -> list[tuple[float, float]]:
    out = []
    step = speed / fps
    carry = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        seg = float(np.hypot(x2 - x1, y2 - y1))
        d = carry
        while d < seg:
            f = d / seg
            out.append((x1 + (x2 - x1) * f, y1 + (y2 - y1) * f))
            d += step
        carry = d - seg
    out.append(pts[-1])
    return out


DEFAULT_HOURLY = {
    ("S", "N"): 420, ("S", "W"): 90, ("S", "E"): 60,
    ("N", "S"): 380, ("N", "E"): 70, ("N", "W"): 55,
    ("E", "W"): 300, ("E", "S"): 80, ("E", "N"): 45,
    ("W", "E"): 260, ("W", "N"): 65, ("W", "S"): 40, ("W", "W"): 6,
}


def generate(duration_s: float = 3600.0, hourly: dict | None = None, fps: float = 10.0,
             break_rate: float = 0.0, jitter: float = 1.5, seed: int = 7):
    """Return (points, truth).

    points: list of (track_id, t, x, y, class) sorted by time, like tracker output.
    truth:  Counter of (direction, turn, class) for every vehicle generated.
    break_rate: share of vehicles whose track is split in two mid intersection
    (the occlusion case), each half with its own track id.
    """
    rng = random.Random(seed)
    hourly = hourly or DEFAULT_HOURLY
    points = []
    truth: Counter = Counter()
    next_id = 1
    for (entry, exit_), per_hour in hourly.items():
        n = int(round(per_hour * duration_s / 3600.0))
        # Demand rises toward the middle of the count so there is a clear peak.
        times = sorted(duration_s * (rng.random() + rng.random()) / 2 for _ in range(n))
        for t0 in times:
            cls = rng.choices(["car", "truck", "bus", "motorcycle"], [90, 6, 2, 2])[0]
            speed = rng.uniform(110, 170)
            samples = _sample(path(entry, exit_), speed, fps)
            truth[(BOUND[entry], turn_for(entry, exit_), cls)] += 1
            split = None
            if rng.random() < break_rate:
                # break while the vehicle is crossing the box between the zones
                mid = [i for i, (x, y) in enumerate(samples) if abs(x - CX) < 70 and abs(y - CY) < 50]
                if len(mid) > 4:
                    split = mid[len(mid) // 2]
            tid = next_id
            next_id += 1
            for i, (x, y) in enumerate(samples):
                if split is not None and split <= i < split + 4:
                    continue                          # hidden for 0.4 s
                if split is not None and i == split + 4:
                    tid = next_id
                    next_id += 1
                cls_seen = cls if rng.random() > 0.1 else rng.choice(["car", "truck"])
                points.append((tid, t0 + i / fps, x + rng.gauss(0, jitter), y + rng.gauss(0, jitter), cls_seen))
    points.sort(key=lambda p: p[1])
    return points, truth
