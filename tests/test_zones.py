import pytest

from turning_movement_counter.zones import BOUND, Zone, ZoneSet, point_in_polygon, turn_for


@pytest.mark.parametrize("entry,exit_,turn", [
    ("S", "N", "T"), ("S", "W", "L"), ("S", "E", "R"), ("S", "S", "U"),   # northbound
    ("N", "S", "T"), ("N", "E", "L"), ("N", "W", "R"),                    # southbound
    ("E", "W", "T"), ("E", "S", "L"), ("E", "N", "R"),                    # westbound
    ("W", "E", "T"), ("W", "N", "L"), ("W", "S", "R"),                    # eastbound
])
def test_turn_names_follow_driver_view(entry, exit_, turn):
    assert turn_for(entry, exit_) == turn


def test_direction_of_travel():
    assert BOUND == {"N": "SB", "E": "WB", "S": "NB", "W": "EB"}


def test_point_in_polygon_concave():
    # an L shape
    poly = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]
    assert point_in_polygon(2, 8, poly)
    assert point_in_polygon(8, 2, poly)
    assert not point_in_polygon(8, 8, poly)
    assert not point_in_polygon(-1, 5, poly)


def test_zoneset_json_round_trip_and_validation():
    zs = ZoneSet([Zone("N", [(0, 0), (5, 0), (5, 5)]), Zone("S", [(0, 10), (5, 10), (5, 15)])], (640, 480))
    back = ZoneSet.from_json(zs.to_json())
    assert back.frame_size == (640, 480)
    assert back.zones[1].points[2] == (5, 15)
    assert back.validate() == []
    assert ZoneSet([Zone("N", [(0, 0), (1, 1)])]).validate()
