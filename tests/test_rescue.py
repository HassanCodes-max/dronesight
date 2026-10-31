"""Mistakes here would send a team to the wrong person or the wrong place, so cases are checked by hand."""
import math

import numpy as np
import pytest

from deploy.build_mission_data import hazard_at
from dronesight.app.rescue import Person, Telemetry, locate, location_error_m, rank

CENTRE = Telemetry(latitude=33.7, longitude=73.0, altitude_m=100, heading_degrees=0, camera_fov_degrees=90, aspect=1)


def metres_north(latitude):
    return (latitude - CENTRE.latitude) * 111_320


def metres_east(longitude):
    return (longitude - CENTRE.longitude) * 111_320 * math.cos(math.radians(CENTRE.latitude))


def test_image_centre_is_directly_below_the_drone():
    assert locate(0.5, 0.5, CENTRE) == pytest.approx((33.7, 73.0))


def test_ground_size_comes_from_height_and_field_of_view():
    assert CENTRE.ground_width_m == pytest.approx(200)


def test_top_of_image_is_ahead_when_flying_north():
    latitude, longitude = locate(0.5, 0.0, CENTRE)
    assert metres_north(latitude) == pytest.approx(100)
    assert metres_east(longitude) == pytest.approx(0)


def test_right_of_image_is_east_when_flying_north():
    latitude, longitude = locate(1.0, 0.5, CENTRE)
    assert metres_north(latitude) == pytest.approx(0)
    assert metres_east(longitude) == pytest.approx(100)


def test_heading_rotates_the_ground_position():
    east_facing = Telemetry(33.7, 73.0, 100, 90, 90, 1)
    latitude, longitude = locate(0.5, 0.0, east_facing)  # ahead of a drone facing east is east
    assert metres_east(longitude) == pytest.approx(100)
    assert metres_north(latitude) == pytest.approx(0)
    latitude, longitude = locate(1.0, 0.5, east_facing)  # its right side is south
    assert metres_north(latitude) == pytest.approx(-100)


def test_location_error_grows_with_height():
    low = Telemetry(33.7, 73.0, 50, 0, 90)
    high = Telemetry(33.7, 73.0, 150, 0, 90)
    assert location_error_m(low) < location_error_m(high)
    assert location_error_m(high) > 5


def person(person_id=1, **values):
    return Person(id=person_id, x=0.5, y=0.5, confidence=values.pop("confidence", 0.9), **values)


def test_person_far_from_safe_ground_outranks_person_at_the_edge():
    edge = person(1, hazard="flood water", depth_m=1, neighbours=3)
    deep = person(2, hazard="flood water", depth_m=12, neighbours=3)
    assert [p.id for p in rank([edge, deep])] == [2, 1]
    assert deep.level == "CRITICAL" and edge.level == "HIGH"
    assert deep.reasons == ["in flood water", "about 12 m from safe ground"]


def test_person_in_fire_is_critical_and_outranks_shallow_water():
    fire = person(1, hazard="fire", neighbours=3)
    water = person(2, hazard="flood water", depth_m=1, neighbours=3)
    assert [p.id for p in rank([water, fire])] == [1, 2]
    assert fire.level == "CRITICAL"


def test_person_in_water_outranks_person_near_it():
    near = person(1, distance_m=4, nearest_hazard="flood water", neighbours=3)
    inside = person(2, hazard="flood water", depth_m=1, neighbours=3)
    assert [p.id for p in rank([near, inside])] == [2, 1]


def test_alone_and_still_raise_priority_and_are_explained():
    alone = person(1, hazard="flood water", depth_m=1, neighbours=0, still=True)
    grouped = person(2, hazard="flood water", depth_m=1, neighbours=4, still=False)
    assert [p.id for p in rank([grouped, alone])] == [1, 2]
    assert alone.reasons == ["in flood water", "alone", "not moving"]


def test_equal_scores_go_to_whoever_is_furthest_from_safety():
    near = person(1, hazard="flood water", depth_m=9, neighbours=1)
    far = person(2, hazard="flood water", depth_m=28, neighbours=1, confidence=0.6)
    assert [p.id for p in rank([near, far])] == [2, 1]


def test_smoke_alone_is_only_a_watch_however_far_inside():
    smoke = rank([person(1, hazard="smoke", depth_m=30, neighbours=3)])[0]
    assert smoke.level == "WATCH" and smoke.reasons == ["in smoke"]


def test_smoke_beside_flames_counts_both():
    beside = person(1, hazard="smoke", distance_m=4, nearest_hazard="fire", neighbours=3)
    away = person(2, hazard="smoke", distance_m=45, nearest_hazard="fire", neighbours=3)
    assert [p.id for p in rank([away, beside])] == [1, 2]
    assert beside.level == "HIGH" and beside.reasons == ["in smoke", "about 4 m from fire"]


def test_person_in_flames_outranks_person_in_smoke_beside_them():
    flames = person(1, hazard="fire", neighbours=3)
    beside = person(2, hazard="smoke", distance_m=1, nearest_hazard="fire", neighbours=3)
    assert [p.id for p in rank([beside, flames])] == [1, 2]


def test_person_far_from_danger_is_clear_even_if_alone():
    safe = rank([person(1, distance_m=200, nearest_hazard="fire", neighbours=0, still=True)])[0]
    assert safe.level == "CLEAR" and safe.score == 0 and safe.reasons == []


def test_unknown_movement_adds_nothing():
    unknown = rank([person(1, hazard="flood water", neighbours=2, still=None)])[0]
    assert unknown.score == 45 and "not moving" not in unknown.reasons


def test_uncertain_detection_says_so_and_ties_prefer_the_surer_one():
    weak = person(1, hazard="flood water", neighbours=1, confidence=0.3)
    strong = person(2, hazard="flood water", neighbours=1, confidence=0.8)
    ranked = rank([weak, strong])
    assert [p.id for p in ranked] == [2, 1]
    assert "detection uncertain, check the image" in weak.reasons


def test_dry_land_beside_water_is_a_watch_but_beside_flames_is_high():
    water = rank([person(1, distance_m=4, nearest_hazard="flood water", neighbours=3)])[0]
    flames = rank([person(2, distance_m=4, nearest_hazard="fire", neighbours=3)])[0]
    assert water.level == "WATCH" and flames.level == "HIGH"


def test_distance_is_measured_from_flames_not_from_smoke_boxes():
    fire, smoke = np.zeros((100, 100), np.uint8), np.zeros((100, 100), np.uint8)
    fire[:, 80:] = 1
    smoke[:, 40:] = 1
    inside, _, nearest, distance = hazard_at({"fire": fire, "smoke": smoke}, 0.1, 0.5, 1)
    assert inside is None and nearest == "fire" and distance == pytest.approx(70, abs=1)
