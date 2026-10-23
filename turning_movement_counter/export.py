"""CSV exports.

`write_tmc_csv` writes the sheet layout traffic engineers expect: one row per
interval, columns grouped Southbound, Westbound, Northbound, Eastbound with
Right, Thru, Left, U-turn under each, then a total.  One block for all
vehicles and one per vehicle class, followed by the peak hour summary.

`write_long_csv` writes one row per (interval, movement, class) for Excel
pivot tables or pandas.
"""

from __future__ import annotations

import csv
from collections import Counter

from .counting import VEHICLE_CLASSES
from .intervals import CountTable
from .zones import BOUND_NAME, DIRECTIONS, TURN_NAME, TURNS

PLURAL = {"car": "Cars", "truck": "Trucks", "bus": "Buses", "motorcycle": "Motorcycles"}
FROM_LEG = {"SB": "North", "WB": "East", "NB": "South", "EB": "West"}


def _header_rows() -> list[list[str]]:
    top = ["Interval start"]
    sub = [""]
    for d in DIRECTIONS:
        top += [f"{BOUND_NAME[d]} (from {FROM_LEG[d]})"] + [""] * (len(TURNS) - 1)
        sub += [TURN_NAME[t] for t in TURNS]
    top.append("Total")
    sub.append("")
    return [top, sub]


def _row(label: str, counts: Counter) -> list:
    row = [label] + [counts[(d, t)] for d in DIRECTIONS for t in TURNS]
    row.append(sum(counts.values()))
    return row


def _block(table: CountTable, title: str, counts_of) -> list[list]:
    rows = [[title], *_header_rows()]
    total: Counter = Counter()
    for iv in table.intervals:
        c = counts_of(iv)
        total.update(c)
        label = table.label(iv.start_s) + ("" if iv.full else " (partial)")
        rows.append(_row(label, c))
    rows.append(_row("Total", total))
    rows.append([])
    return rows


def tmc_rows(table: CountTable, source: str = "", stats: dict | None = None) -> list[list]:
    rows: list[list] = [["Turning Movement Count"], ["Video", source],
                        ["Count start", table.label(0.0)], ["Interval (min)", table.interval_min], []]
    rows += _block(table, "All vehicles", lambda iv: iv.counts)
    present = [c for c in VEHICLE_CLASSES if table.class_totals()[c]]
    for cls in present:
        rows += _block(table, PLURAL[cls], lambda iv, cls=cls: iv.by_class.get(cls, Counter()))
    p = table.peak
    if p is None:
        rows.append(["Peak hour", "Needs at least 60 minutes of video"])
    else:
        rows.append(["Peak hour", f"{table.label(p.start_s)} to {table.label(p.end_s)}"])
        rows.append(["Peak hour volume", p.volume])
        rows.append(["Highest 15-min volume in peak hour", p.max_15])
        rows.append(["Peak hour factor", f"{p.phf:.2f}"])
        rows.append(["Heavy vehicles in peak hour (%)", p.heavy_pct])
        rows.append([])
        rows += [["Peak hour by movement"], *_header_rows(), _row("Peak hour", p.counts)]
    if stats:
        rows.append([])
        rows.append(["Tracks seen", stats.get("tracks", "")])
        rows.append(["Vehicles counted", stats.get("counted", "")])
        rows.append(["Broken tracks joined", stats.get("stitched", "")])
        rows.append(["Tracks not counted (seen in fewer than two legs)", stats.get("not_counted", "")])
    return rows


def write_tmc_csv(path: str, table: CountTable, source: str = "", stats: dict | None = None) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(tmc_rows(table, source, stats))


def write_long_csv(path: str, table: CountTable) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["interval_start", "interval_end", "full_interval", "direction", "from_leg",
                    "movement", "vehicle_class", "count"])
        for iv in table.intervals:
            for cls in sorted(iv.by_class):
                for (d, t), n in sorted(iv.by_class[cls].items()):
                    w.writerow([table.label(iv.start_s), table.label(iv.end_s), iv.full, d,
                                FROM_LEG[d], TURN_NAME[t], cls, n])
