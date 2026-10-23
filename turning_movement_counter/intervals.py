"""Bin movements into count intervals and find the peak hour.

Peak hour = the four consecutive 15-minute intervals with the highest total.
Peak hour factor (PHF) = peak hour volume / (4 x the highest 15-minute volume
inside that hour), as in the Highway Capacity Manual.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .counting import HEAVY_CLASSES, Movement
from .zones import DIRECTIONS, TURNS

Key = tuple[str, str]  # (direction, turn), e.g. ("NB", "L")


@dataclass
class Interval:
    start_s: float
    end_s: float
    full: bool                       # False for a last interval cut short by the end of the video
    counts: Counter = field(default_factory=Counter)        # (direction, turn) -> vehicles
    by_class: dict[str, Counter] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def heavy(self) -> int:
        return sum(sum(c.values()) for k, c in self.by_class.items() if k in HEAVY_CLASSES)


@dataclass
class PeakHour:
    first: int          # index of the first 15-minute interval in the hour
    start_s: float
    end_s: float
    volume: int
    max_15: int
    phf: float
    counts: Counter     # (direction, turn) totals for the hour
    heavy_pct: float


@dataclass
class CountTable:
    intervals: list[Interval]
    interval_min: int
    start_clock: datetime | None
    peak: PeakHour | None

    def label(self, seconds: float) -> str:
        if self.start_clock is None:
            m, s = divmod(int(round(seconds)), 60)
            h, m = divmod(m, 60)
            return f"{h:d}:{m:02d}:{s:02d}"
        return (self.start_clock + timedelta(seconds=seconds)).strftime("%H:%M")

    def totals(self) -> Counter:
        out: Counter = Counter()
        for iv in self.intervals:
            out.update(iv.counts)
        return out

    def class_totals(self) -> Counter:
        out: Counter = Counter()
        for iv in self.intervals:
            for cls, c in iv.by_class.items():
                out[cls] += sum(c.values())
        return out

    def approach_totals(self, counts: Counter | None = None) -> dict[str, int]:
        counts = self.totals() if counts is None else counts
        return {d: sum(counts[(d, t)] for t in TURNS) for d in DIRECTIONS}


def bin_movements(movements: list[Movement], duration_s: float, interval_min: int = 15) -> list[Interval]:
    width = interval_min * 60.0
    n = max(1, int(-(-duration_s // width)))  # ceil
    ivs = [Interval(i * width, min((i + 1) * width, duration_s), (i + 1) * width <= duration_s + 1e-6)
           for i in range(n)]
    for m in movements:
        i = min(int(m.t // width), n - 1)
        key = (m.direction, m.turn)
        ivs[i].counts[key] += 1
        ivs[i].by_class.setdefault(m.vehicle_class, Counter())[key] += 1
    return ivs


def peak_hour(intervals_15: list[Interval]) -> PeakHour | None:
    """Needs at least four full 15-minute intervals; returns None otherwise."""
    full = [iv for iv in intervals_15 if iv.full]
    if len(full) < 4:
        return None
    best = max(range(len(full) - 3), key=lambda i: (sum(iv.total for iv in full[i:i + 4]), -i))
    hour = full[best:best + 4]
    volume = sum(iv.total for iv in hour)
    max_15 = max(iv.total for iv in hour)
    counts: Counter = Counter()
    for iv in hour:
        counts.update(iv.counts)
    heavy = sum(iv.heavy for iv in hour)
    return PeakHour(
        first=best, start_s=hour[0].start_s, end_s=hour[-1].end_s, volume=volume, max_15=max_15,
        phf=round(volume / (4 * max_15), 2) if max_15 else 0.0, counts=counts,
        heavy_pct=round(100 * heavy / volume, 1) if volume else 0.0,
    )


def build_table(movements: list[Movement], duration_s: float, interval_min: int = 15,
                start_clock: datetime | None = None) -> CountTable:
    if interval_min not in (5, 15):
        raise ValueError("Count intervals must be 5 or 15 minutes.")
    ivs = bin_movements(movements, duration_s, interval_min)
    peak = peak_hour(ivs if interval_min == 15 else bin_movements(movements, duration_s, 15))
    return CountTable(ivs, interval_min, start_clock, peak)
