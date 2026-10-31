"""Rank people by danger and turn image positions into map positions.

The ranking is rules, not a trained model, because no dataset labels who is really at risk, and a
rescue team must be able to see why someone ranked first. It is a research prototype: the weights
below are reasoned choices that have not been validated by emergency responders.
"""
import math
from dataclasses import dataclass, field

METRES_PER_DEGREE = 111_320
GPS_ERROR_M = 5.0  # typical consumer drone GPS
TILT_ERROR_DEGREES = 2.0  # how far a gimbal may be from pointing straight down

HAZARD_POINTS = {"fire": 65, "flood water": 45, "smoke": 10}  # smoke boxes are coarse, so smoke alone is only a watch
DEEP_POINTS = 20  # someone far from safe ground cannot simply walk out
DEEP_METRES = 3
# Metres from the hazard and points. Fire spreads and burns at a distance. Dry land beside water is
# safer, so it stays a watch unless other reasons add up. Smoke boxes are too coarse to measure from.
NEAR_POINTS = {"fire": ((5, 40), (20, 25), (50, 10)), "flood water": ((5, 25), (20, 10))}
ALONE_POINTS = 15
STILL_POINTS = 15
ALONE_RADIUS_M = 10
LEVELS = ((65, "CRITICAL"), (30, "HIGH"), (10, "WATCH"), (0, "CLEAR"))
UNCERTAIN_BELOW = 0.5


@dataclass(frozen=True)
class Telemetry:
    latitude: float
    longitude: float
    altitude_m: float
    heading_degrees: float
    camera_fov_degrees: float
    aspect: float = 16 / 9

    @property
    def ground_width_m(self):
        return 2 * self.altitude_m * math.tan(math.radians(self.camera_fov_degrees) / 2)

    @property
    def ground_height_m(self):
        return self.ground_width_m / self.aspect


@dataclass
class Person:
    id: int
    x: float  # 0 to 1 across the image, at the person's feet
    y: float
    confidence: float
    hazard: str | None = None  # what they are standing in
    depth_m: float = 0  # how far inside it, to the nearest safe ground
    distance_m: float | None = None  # to the nearest hazard
    nearest_hazard: str | None = None
    neighbours: int = 0  # people within ALONE_RADIUS_M
    still: bool | None = None  # None when one image cannot tell
    score: int = 0
    level: str = "CLEAR"
    reasons: list = field(default_factory=list)


def locate(x, y, telemetry):
    """Map position of an image point, for a camera pointing straight down."""
    right = (x - 0.5) * telemetry.ground_width_m
    forward = (0.5 - y) * telemetry.ground_height_m
    heading = math.radians(telemetry.heading_degrees)
    north = forward * math.cos(heading) - right * math.sin(heading)
    east = forward * math.sin(heading) + right * math.cos(heading)
    latitude = telemetry.latitude + north / METRES_PER_DEGREE
    longitude = telemetry.longitude + east / (METRES_PER_DEGREE * math.cos(math.radians(telemetry.latitude)))
    return latitude, longitude


def location_error_m(telemetry):
    """Rough horizontal error, combining GPS error with the gimbal not pointing exactly down."""
    tilt = telemetry.altitude_m * math.tan(math.radians(TILT_ERROR_DEGREES))
    return math.hypot(GPS_ERROR_M, tilt)


def assess(person):
    """Score one person and record each reason, so the result can be explained."""
    score = 0
    reasons = []
    if person.hazard:
        score += HAZARD_POINTS[person.hazard]
        reasons.append(f"in {person.hazard}")
        if person.hazard != "smoke" and person.depth_m >= DEEP_METRES:
            score += DEEP_POINTS
            reasons.append(f"about {round(person.depth_m)} m from safe ground")
    if person.hazard in (None, "smoke") and person.distance_m is not None:
        for metres, points in NEAR_POINTS[person.nearest_hazard]:
            if person.distance_m <= metres:
                score += points
                reasons.append(f"about {round(person.distance_m)} m from {person.nearest_hazard}")
                break
    if score and person.neighbours == 0:
        score += ALONE_POINTS
        reasons.append("alone")
    if score and person.still:
        score += STILL_POINTS
        reasons.append("not moving")
    if person.confidence < UNCERTAIN_BELOW:
        reasons.append("detection uncertain, check the image")
    person.score = score
    person.level = next(level for minimum, level in LEVELS if score >= minimum)
    person.reasons = reasons
    return person


def rank(people):
    """Most urgent first. Ties go to whoever is furthest from safety, then to the surer detection."""
    return sorted((assess(person) for person in people), key=lambda p: (-p.score, -p.depth_m, -p.confidence, p.id))
