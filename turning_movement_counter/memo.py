"""Optional Claude-written count memo.

Claude only sees the computed numbers in `summary_dict` and is told to quote
them, never to compute anything.  Without an API key the app writes a
template memo so it works fully offline.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .counting import VEHICLE_CLASSES
from .intervals import CountTable
from .zones import BOUND_NAME, DIRECTIONS, TURN_NAME, TURNS

MODEL = "claude-haiku-5-5"
API_URL = "https://api.anthropic.com/v1/messages"

SYSTEM = (
    "You are a traffic engineer writing a one-page memo that summarizes an intersection "
    "turning movement count. You are given a JSON summary that software already computed "
    "from video. Rules: quote numbers exactly as given and never compute, round differently, "
    "or invent values; lead with the peak hour, its volume and the peak hour factor; name the "
    "heaviest approach and the heaviest turning movements; mention heavy vehicle share; if "
    "there is no peak hour say the video was shorter than an hour; end with one sentence on "
    "count method and that the counts are automated and should be spot checked against a "
    "hand count. Under 250 words, short paragraphs, no headings, no bullet points, no em dashes."
)


def summary_dict(table: CountTable, source: str = "", stats: dict | None = None) -> dict:
    totals = table.totals()
    movements = {f"{BOUND_NAME[d]} {TURN_NAME[t]}": totals[(d, t)] for d in DIRECTIONS for t in TURNS
                 if totals[(d, t)]}
    out = {
        "video": os.path.basename(source),
        "count_start": table.label(0.0),
        "count_end": table.label(table.intervals[-1].end_s),
        "interval_minutes": table.interval_min,
        "total_vehicles": sum(totals.values()),
        "approach_totals": {BOUND_NAME[d]: v for d, v in table.approach_totals().items()},
        "movement_totals": movements,
        "vehicle_classes": {c: table.class_totals()[c] for c in VEHICLE_CLASSES if table.class_totals()[c]},
        "peak_hour": None,
        "tracking": stats or {},
    }
    p = table.peak
    if p:
        ph = table.approach_totals(p.counts)
        out["peak_hour"] = {
            "start": table.label(p.start_s), "end": table.label(p.end_s), "volume": p.volume,
            "highest_15_min_volume": p.max_15, "phf": p.phf, "heavy_vehicle_pct": p.heavy_pct,
            "approach_volumes": {BOUND_NAME[d]: v for d, v in ph.items()},
        }
    return out


def template_memo(s: dict) -> str:
    approaches = sorted(s["approach_totals"].items(), key=lambda kv: -kv[1])
    top_moves = sorted(s["movement_totals"].items(), key=lambda kv: -kv[1])[:3]
    text = (f"This memo summarizes an automated turning movement count from {s['video'] or 'video'}, "
            f"{s['count_start']} to {s['count_end']}, in {s['interval_minutes']}-minute intervals. "
            f"{s['total_vehicles']} vehicles were counted.\n\n")
    p = s["peak_hour"]
    if p:
        text += (f"The peak hour is {p['start']} to {p['end']} with {p['volume']} vehicles. The highest "
                 f"15-minute volume in that hour is {p['highest_15_min_volume']}, for a peak hour factor of "
                 f"{p['phf']:.2f}. Heavy vehicles (trucks and buses) are {p['heavy_vehicle_pct']}% of the "
                 f"peak hour volume.\n\n")
    else:
        text += "The video is shorter than one hour, so no peak hour or peak hour factor is reported.\n\n"
    text += (f"The heaviest approach is {approaches[0][0]} with {approaches[0][1]} vehicles. "
             "The largest movements are " + ", ".join(f"{k} ({v})" for k, v in top_moves) + ".\n\n")
    text += ("Counts were produced by YOLO vehicle detection with ByteTrack tracking and approach zones; "
             "they are automated and should be spot checked against a hand count.")
    return text


def api_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY") or _settings().get("anthropic_api_key") or None


def _settings_path() -> str:
    from .detect import app_dir

    return os.path.join(app_dir(), "settings.json")


def _settings() -> dict:
    try:
        with open(_settings_path(), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_api_key(key: str) -> None:
    path = _settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = _settings()
    data["anthropic_api_key"] = key.strip()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def write_memo(summary: dict, timeout: float = 30.0) -> tuple[str, str]:
    """Return (memo, source) where source is 'claude' or 'template'."""
    key = api_key()
    if not key:
        return template_memo(summary), "template"
    body = json.dumps({
        "model": MODEL,
        "max_tokens": 700,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": "Count summary:\n" + json.dumps(summary, indent=1)}],
    }).encode()
    req = urllib.request.Request(API_URL, data=body, method="POST", headers={
        "content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        text = "".join(b.get("text", "") for b in data.get("content", []))
        if text.strip():
            return text.strip(), "claude"
    except (urllib.error.URLError, ValueError, KeyError, OSError):
        pass
    return template_memo(summary), "template"
