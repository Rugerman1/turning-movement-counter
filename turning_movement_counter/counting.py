"""Turn tracker output into one counted movement per vehicle.

The detector/tracker gives a stream of (track id, time, x, y, class) points,
where (x, y) is the bottom center of the box (where the tires meet the road).
For each track we record the zone visits in order, then:

* entry leg = the first zone visited, exit leg = the last one;
* a visit only counts after `min_samples` consecutive points inside a zone, so
  a box jittering on a zone edge does not register;
* first leg == last leg is a U-turn only when the vehicle was outside the zone
  for at least `u_turn_gap_s` in between (otherwise it is one long visit);
* the vehicle class is the majority vote over the track's points.

Tracks break when a car is hidden behind a truck.  The halves look like a
vehicle that entered and vanished plus a vehicle that appeared mid
intersection and left.  `stitch` joins those halves when the second starts
shortly after and close to where the first ended, so the vehicle is counted
once instead of not at all.  A track that is never seen in two different
legs is not counted, which errs on the side of undercounting.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from .zones import BOUND, ZoneSet, turn_for

VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]
HEAVY_CLASSES = {"truck", "bus"}


@dataclass
class Visit:
    leg: str
    t_in: float
    t_out: float


@dataclass
class Track:
    track_id: int
    visits: list[Visit] = field(default_factory=list)
    classes: Counter = field(default_factory=Counter)
    t_first: float = 0.0
    t_last: float = 0.0
    xy_first: tuple[float, float] = (0.0, 0.0)
    xy_last: tuple[float, float] = (0.0, 0.0)
    n_points: int = 0
    # state for the in-zone debounce
    _cand_leg: str | None = None
    _cand_n: int = 0
    _cand_t: float = 0.0
    _in_leg: str | None = None

    @property
    def vehicle_class(self) -> str:
        return self.classes.most_common(1)[0][0] if self.classes else "car"

    @property
    def starts_outside(self) -> bool:
        """True if the track appeared before reaching any zone (a broken track's second half)."""
        return not self.visits or self.visits[0].t_in > self.t_first

    @property
    def ends_outside(self) -> bool:
        """True if the track was last seen outside the zone it last visited."""
        return not self.visits or self.visits[-1].t_out < self.t_last


@dataclass
class Movement:
    t: float            # seconds from the start of the video when the vehicle left its entry zone
    entry: str          # leg entered from: N, E, S, W
    exit: str
    direction: str      # SB, WB, NB, EB
    turn: str           # R, T, L, U
    vehicle_class: str
    track_ids: tuple[int, ...]


@dataclass
class CountSettings:
    min_samples: int = 2        # consecutive points inside a zone before a visit counts
    u_turn_gap_s: float = 1.0   # time outside the zone before a return counts as a U-turn
    count_u_turns: bool = True
    stitch: bool = True
    stitch_gap_s: float = 2.0   # max time between the end of one half and the start of the other
    stitch_dist: float = 0.0    # max pixel distance; 0 means 8% of the frame diagonal
    frame_diag: float = 1000.0


class TrackAccumulator:
    """Feed tracker points one at a time, then call `movements()`."""

    def __init__(self, zones: ZoneSet, settings: CountSettings | None = None):
        self.zones = zones
        self.s = settings or CountSettings()
        self.tracks: dict[int, Track] = {}

    def add(self, track_id: int, t: float, x: float, y: float, vehicle_class: str) -> None:
        tr = self.tracks.get(track_id)
        if tr is None:
            tr = self.tracks[track_id] = Track(track_id, t_first=t, xy_first=(x, y))
        tr.t_last = t
        tr.xy_last = (x, y)
        tr.n_points += 1
        tr.classes[vehicle_class] += 1
        leg = self.zones.zone_at(x, y)

        if leg is not None and leg == tr._in_leg:
            tr.visits[-1].t_out = t
            tr._cand_leg, tr._cand_n = leg, self.s.min_samples
            return
        if leg != tr._cand_leg:
            tr._cand_leg, tr._cand_n, tr._cand_t = leg, 0, t
        tr._cand_n += 1
        if leg is None:
            if tr._cand_n >= self.s.min_samples:
                tr._in_leg = None
            return
        if tr._cand_n >= self.s.min_samples:
            tr._in_leg = leg
            tr.visits.append(Visit(leg, tr._cand_t, t))

    # --- turning visits into movements -------------------------------------------------

    def _legs(self, visits: list[Visit]) -> list[Visit]:
        """Merge back-to-back visits to the same leg unless the gap is long enough for a U-turn."""
        merged: list[Visit] = []
        for v in visits:
            if merged and merged[-1].leg == v.leg and v.t_in - merged[-1].t_out < self.s.u_turn_gap_s:
                merged[-1].t_out = v.t_out
            else:
                merged.append(Visit(v.leg, v.t_in, v.t_out))
        return merged

    def _movement(self, visits: list[Visit], classes: Counter, ids: tuple[int, ...]) -> Movement | None:
        legs = self._legs(visits)
        if len(legs) < 2:
            return None
        entry, exit_ = legs[0].leg, legs[-1].leg
        turn = turn_for(entry, exit_)
        if turn == "U" and not self.s.count_u_turns:
            return None
        cls = classes.most_common(1)[0][0] if classes else "car"
        return Movement(legs[0].t_out, entry, exit_, BOUND[entry], turn, cls, ids)

    def _stitch(self, tracks: list[Track]) -> list[list[Track]]:
        """Pair 'entered then vanished' halves with 'appeared then exited' halves."""
        max_d = self.s.stitch_dist or 0.08 * self.s.frame_diag
        # Only fragments seen in exactly one leg are candidates: a complete track needs no help.
        single = [t for t in tracks if len(self._legs(t.visits)) == 1]
        heads = [t for t in single if t.ends_outside]
        tails = [t for t in single if t.starts_outside]
        groups: dict[int, list[Track]] = {t.track_id: [t] for t in tracks}
        root: dict[int, int] = {t.track_id: t.track_id for t in tracks}
        used_tails: set[int] = set()
        for h in sorted(heads, key=lambda t: t.t_last):
            best, best_cost = None, None
            for tl in tails:
                if tl.track_id in used_tails or tl.track_id == h.track_id:
                    continue
                gap = tl.t_first - h.t_last
                if not (0 < gap <= self.s.stitch_gap_s):
                    continue
                d = math.dist(h.xy_last, tl.xy_first)
                if d > max_d:
                    continue
                if tl.visits[0].leg == h.visits[-1].leg:
                    continue  # rejoining the same leg would invent a U-turn
                if (h.vehicle_class in HEAVY_CLASSES) != (tl.vehicle_class in HEAVY_CLASSES):
                    continue
                cost = d / max_d + gap / self.s.stitch_gap_s
                if best_cost is None or cost < best_cost:
                    best, best_cost = tl, cost
            if best is not None:
                used_tails.add(best.track_id)
                r = root[h.track_id]
                for t in groups.pop(best.track_id):
                    root[t.track_id] = r
                    groups[r].append(t)
        return list(groups.values())

    def movements(self) -> tuple[list[Movement], dict]:
        tracks = sorted(self.tracks.values(), key=lambda t: t.t_first)
        groups = self._stitch(tracks) if self.s.stitch else [[t] for t in tracks]
        out: list[Movement] = []
        stitched = 0
        for g in groups:
            visits = [v for t in g for v in t.visits]
            classes = sum((t.classes for t in g), Counter())
            m = self._movement(visits, classes, tuple(t.track_id for t in g))
            if m is not None:
                out.append(m)
                if len(g) > 1:
                    stitched += 1
        out.sort(key=lambda m: m.t)
        stats = {
            "tracks": len(tracks),
            "counted": len(out),
            "stitched": stitched,
            "not_counted": len(groups) - len(out),
        }
        return out, stats
