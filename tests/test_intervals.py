from datetime import datetime

import pytest

from turning_movement_counter.counting import Movement
from turning_movement_counter.export import tmc_rows, write_long_csv, write_tmc_csv
from turning_movement_counter.intervals import build_table


def mv(t, d="NB", turn="T", cls="car"):
    return Movement(t, "S", "N", d, turn, cls, (0,))


def test_bins_and_peak_hour_factor():
    # 15-minute volumes 100, 120, 150, 130, 90 -> peak hour = intervals 1..4? no: 0..3 = 500, 1..4 = 490
    vols = [100, 120, 150, 130, 90]
    ms = []
    for i, v in enumerate(vols):
        ms += [mv(i * 900 + 900 * (k + 0.5) / v, cls="truck" if k < 10 else "car") for k in range(v)]
    table = build_table(ms, duration_s=4500, start_clock=datetime(2026, 10, 22, 7, 0))
    assert [iv.total for iv in table.intervals] == vols
    p = table.peak
    assert p.volume == 500 and p.max_15 == 150
    assert p.phf == pytest.approx(500 / 600, abs=0.005)
    assert table.label(p.start_s) == "07:00" and table.label(p.end_s) == "08:00"
    assert p.heavy_pct == pytest.approx(100 * 40 / 500, abs=0.05)


def test_peak_hour_slides_to_the_busiest_hour():
    vols = [50, 100, 120, 150, 130]
    ms = [mv(i * 900 + 1 + k) for i, v in enumerate(vols) for k in range(v)]
    p = build_table(ms, duration_s=4500).peak
    assert p.first == 1 and p.volume == 500


def test_short_video_has_no_peak_hour_and_partial_last_interval():
    table = build_table([mv(10), mv(1000)], duration_s=2000)
    assert table.peak is None
    assert [iv.full for iv in table.intervals] == [True, True, False]


def test_five_minute_table_still_uses_15_minute_phf():
    ms = [mv(i * 60 + 1) for i in range(60)]  # one vehicle a minute for an hour
    table = build_table(ms, duration_s=3600, interval_min=5)
    assert len(table.intervals) == 12
    assert table.peak.volume == 60 and table.peak.phf == 1.0


def test_csv_layout(tmp_path):
    ms = [mv(5, "NB", "L"), mv(6, "SB", "R", "bus"), mv(7, "EB", "U")]
    table = build_table(ms, duration_s=900, start_clock=datetime(2026, 1, 1, 16, 30))
    rows = tmc_rows(table, "clip.mp4")
    header = rows[6]
    assert header[0] == "Interval start" and header[1].startswith("Southbound") and header[-1] == "Total"
    sub = rows[7]
    assert sub[1:5] == ["Right", "Thru", "Left", "U-turn"]
    first = rows[8]
    assert first[0] == "16:30"
    assert first[1] == 1          # SB right
    assert first[9 + 2] == 1      # NB left
    assert first[13 + 3] == 1     # EB U-turn
    assert first[-1] == 3
    write_tmc_csv(tmp_path / "a.csv", table, "clip.mp4")
    write_long_csv(tmp_path / "b.csv", table)
    assert "Buss" not in (tmp_path / "a.csv").read_text()
    assert (tmp_path / "b.csv").read_text().count("\n") == 4
