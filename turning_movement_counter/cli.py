"""Headless counting: python -m turning_movement_counter count VIDEO --zones ZONES.json"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

from .detect import DEFAULT_MODEL, RunOptions, count_video
from .export import write_long_csv, write_tmc_csv
from .intervals import build_table
from .zones import ZoneSet


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="turning_movement_counter count")
    ap.add_argument("video")
    ap.add_argument("--zones", required=True, help="zones JSON saved from the app")
    ap.add_argument("--start", help="clock time the video starts, e.g. '2026-10-22 07:00'")
    ap.add_argument("--interval", type=int, default=15, choices=[5, 15])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--fps", type=float, default=10.0, help="frames per second to analyze")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--annotated", help="write an annotated video here")
    ap.add_argument("--out", help="CSV path (default: next to the video)")
    a = ap.parse_args(argv)

    with open(a.zones, encoding="utf-8") as f:
        zones = ZoneSet.from_json(f.read())
    problems = zones.validate()
    if problems:
        ap.error("; ".join(problems))
    opts = RunOptions(model=a.model, analysis_fps=a.fps, conf=a.conf, annotated_path=a.annotated)

    def progress(frac, msg):
        print(f"\r{frac * 100:5.1f}%  {msg}", end="", flush=True)

    run = count_video(a.video, zones, opts, progress=progress)
    print()
    start = datetime.fromisoformat(a.start) if a.start else None
    table = build_table(run.movements, run.info.duration_s, a.interval, start)
    out = a.out or os.path.splitext(a.video)[0] + ".tmc.csv"
    write_tmc_csv(out, table, os.path.basename(a.video), run.stats)
    write_long_csv(os.path.splitext(out)[0] + ".long.csv", table)
    print(f"{len(run.movements)} vehicles counted -> {out}")
    if table.peak:
        p = table.peak
        print(f"Peak hour {table.label(p.start_s)}-{table.label(p.end_s)}: {p.volume} veh, PHF {p.phf:.2f}")
    return 0
