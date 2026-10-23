"""Approach zones and the entry/exit rule that names a turning movement.

Each zone is a polygon drawn over one approach of the intersection (inbound
and outbound lanes together) and labeled with the compass leg it covers:
N, E, S or W.  A vehicle that enters through the S zone and leaves through
the N zone is a northbound through movement.

Turns come from the clockwise order of the legs N -> E -> S -> W, so the
camera can point any direction as long as the zones are labeled with the
real compass legs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

APPROACHES = ["N", "E", "S", "W"]  # clockwise

# A vehicle entering from the north leg is traveling southbound, and so on.
BOUND = {"N": "SB", "E": "WB", "S": "NB", "W": "EB"}
BOUND_NAME = {"SB": "Southbound", "WB": "Westbound", "NB": "Northbound", "EB": "Eastbound"}

# Column order used by most count sheets: SB, WB, NB, EB, each Right, Thru, Left, U-turn.
DIRECTIONS = ["SB", "WB", "NB", "EB"]
TURNS = ["R", "T", "L", "U"]
TURN_NAME = {"R": "Right", "T": "Thru", "L": "Left", "U": "U-turn"}


def turn_for(entry: str, exit_: str) -> str:
    """Movement for a vehicle entering on leg `entry` and leaving on leg `exit_`.

    Going clockwise from the entry leg: one leg over is a left turn (seen
    from the driver heading into the intersection), two legs is through,
    three is a right turn, and the same leg is a U-turn.
    """
    i, j = APPROACHES.index(entry), APPROACHES.index(exit_)
    return {0: "U", 1: "L", 2: "T", 3: "R"}[(j - i) % 4]


def point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    """Even-odd ray casting test."""
    inside = False
    n = len(poly)
    for k in range(n):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


@dataclass
class Zone:
    approach: str
    points: list[tuple[float, float]] = field(default_factory=list)

    def contains(self, x: float, y: float) -> bool:
        return len(self.points) >= 3 and point_in_polygon(x, y, self.points)


@dataclass
class ZoneSet:
    zones: list[Zone] = field(default_factory=list)
    frame_size: tuple[int, int] | None = None  # (width, height) the zones were drawn on

    def zone_at(self, x: float, y: float) -> str | None:
        for z in self.zones:
            if z.contains(x, y):
                return z.approach
        return None

    def validate(self) -> list[str]:
        problems = []
        legs = [z.approach for z in self.zones]
        for leg in set(legs):
            if legs.count(leg) > 1:
                problems.append(f"Leg {leg} has more than one zone.")
        for z in self.zones:
            if z.approach not in APPROACHES:
                problems.append(f"Unknown leg {z.approach!r}.")
            if len(z.points) < 3:
                problems.append(f"Zone {z.approach} needs at least 3 points.")
        if len(set(legs)) < 2:
            problems.append("Draw zones on at least two legs.")
        return problems

    def to_json(self) -> str:
        return json.dumps({
            "frame_size": list(self.frame_size) if self.frame_size else None,
            "zones": [{"approach": z.approach, "points": [list(p) for p in z.points]} for z in self.zones],
        }, indent=1)

    @classmethod
    def from_json(cls, text: str) -> "ZoneSet":
        data = json.loads(text)
        size = data.get("frame_size")
        return cls(
            zones=[Zone(z["approach"], [tuple(p) for p in z["points"]]) for z in data["zones"]],
            frame_size=tuple(size) if size else None,
        )
