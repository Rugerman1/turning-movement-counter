from collections import Counter

from turning_movement_counter.counting import CountSettings, TrackAccumulator
from turning_movement_counter.synthetic import DEFAULT_HOURLY, demo_zones, generate, path, _sample


def run(points, **kw):
    acc = TrackAccumulator(demo_zones(), CountSettings(frame_diag=1469, **kw))
    for tid, t, x, y, cls in points:
        acc.add(tid, t, x, y, cls)
    return acc.movements()


def counted(movements):
    return Counter((m.direction, m.turn, m.vehicle_class) for m in movements)


def test_one_vehicle_each_movement():
    for i, (entry, exit_) in enumerate(DEFAULT_HOURLY):
        pts = [(1, k / 10, x, y, "car") for k, (x, y) in enumerate(_sample(path(entry, exit_), 140, 10))]
        movements, stats = run(pts)
        assert len(movements) == 1, (entry, exit_)
        m = movements[0]
        assert (m.entry, m.exit) == (entry, exit_)


def test_hour_of_clean_tracks_matches_truth_exactly():
    points, truth = generate(duration_s=3600, break_rate=0.0)
    movements, stats = run(points)
    assert counted(movements) == truth
    assert stats["not_counted"] == 0


def test_broken_tracks_are_joined_not_dropped():
    points, truth = generate(duration_s=1800, break_rate=0.3, seed=3)
    total = sum(truth.values())
    no_stitch, _ = run(points, stitch=False)
    stitched, stats = run(points)
    assert len(no_stitch) < 0.8 * total          # broken halves alone are never counted
    assert stats["stitched"] > 0.2 * total
    assert abs(len(stitched) - total) <= 0.02 * total
    by_move = Counter((m.direction, m.turn) for m in stitched)
    want = Counter()
    for (d, t, _), n in truth.items():
        want[(d, t)] += n
    for k, n in want.items():
        assert abs(by_move[k] - n) <= max(2, 0.05 * n), k


def test_zone_edge_jitter_does_not_make_a_u_turn():
    # A car parked on the edge of the N zone, flickering in and out, never leaves the leg.
    pts = [(1, k / 10, 600, 270 + (2 if k % 2 else -2), "car") for k in range(100)]
    movements, _ = run(pts)
    assert movements == []


def test_u_turn_needs_time_outside_the_zone():
    pts = [(1, k / 10, x, y, "car") for k, (x, y) in enumerate(_sample(path("W", "W"), 140, 10))]
    movements, _ = run(pts)
    assert [(m.direction, m.turn) for m in movements] == [("EB", "U")]
    movements, _ = run(pts, count_u_turns=False)
    assert movements == []


def test_track_seen_in_one_leg_is_not_counted():
    pts = [(1, k / 10, 600, 40 + 10 * k, "car") for k in range(20)]   # drives down the N leg, then lost
    movements, stats = run(pts)
    assert movements == [] and stats["not_counted"] == 1


def test_class_is_majority_vote():
    pts = [(1, k / 10, x, y, "truck" if k % 4 else "car")
           for k, (x, y) in enumerate(_sample(path("S", "N"), 140, 10))]
    movements, _ = run(pts)
    assert movements[0].vehicle_class == "truck"
