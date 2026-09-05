"""Put the player on the badminton court.

Joint angles tell you how a shot was played; they say nothing about *where*
from, or whether the player got back to base afterwards, which is most of what
decides a rally.

The court is flat and the feet are on it, so one homography relates image pixels
to court metres exactly -- no camera calibration, no depth estimate. Four tapped
corners determine it.

Assumptions worth stating: the camera does not move after calibration (a phone
on a bag or tripod is fine, a hand-held one drifts), and the feet are on the
floor (during a jump the mapped position creeps towards the camera).

Mirrors ``web/js/core/court.js``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import landmarks as L


class Court:
    """BWF court dimensions, in metres."""

    length = 13.40
    width = 6.10
    half_length = 6.70
    singles_width = 5.18
    short_service_line = 1.98
    doubles_long_service_line = 0.76
    net_height_centre = 1.524
    net_height_posts = 1.55


#: Court coordinates for the near half, in metres: ``x`` runs -3.05 (left) to
#: +3.05 (right) as the player faces the net, ``y`` 0 at the net to 6.70 at the
#: back boundary.
HALF_COURT_CORNERS = {
    "netLeft": (-Court.width / 2, 0.0),
    "netRight": (Court.width / 2, 0.0),
    "backRight": (Court.width / 2, Court.half_length),
    "backLeft": (-Court.width / 2, Court.half_length),
}

#: The order corners are asked for.
CALIBRATION_ORDER = ["netLeft", "netRight", "backRight", "backLeft"]

#: Singles base position: centre, a little behind mid-court.
BASE_POSITION = (0.0, 3.0)

ZONES = {
    "depth": [("front", 0.0, 2.2), ("mid", 2.2, 4.5), ("rear", 4.5, Court.half_length)],
    "lateral": [("left", -math.inf, -1.0), ("centre", -1.0, 1.0), ("right", 1.0, math.inf)],
}


def solve_linear(A, b):
    """Solve ``A x = b`` by Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [list(row) + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            return None
        M[col], M[pivot] = M[pivot], M[col]
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col] / M[col][col]
            if factor == 0:
                continue
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]
    return [M[i][n] / M[i][i] for i in range(n)]


def homography_from_quad(src, dst):
    """Homography mapping four source points to four destination points.

    With h33 fixed at 1 the eight remaining unknowns are determined exactly by
    four correspondences, so this is a linear solve rather than a least-squares
    fit -- which is why the four taps must be the four *corners*, and why tapping
    them accurately matters.
    """
    if len(src) != 4 or len(dst) != 4:
        raise ValueError("need exactly four points")
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.append(v)
    h = solve_linear(A, b)
    if h is None:
        return None
    return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]]


def apply_homography(H, point):
    """Apply a homography to a 2D point. ``None`` behind the horizon."""
    x, y = point
    w = H[2][0] * x + H[2][1] * y + H[2][2]
    if abs(w) < 1e-12:
        return None
    return ((H[0][0] * x + H[0][1] * y + H[0][2]) / w,
            (H[1][0] * x + H[1][1] * y + H[1][2]) / w)


def invert3x3(M):
    a, b, c = M[0]
    d, e, f = M[1]
    g, h, i = M[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-12:
        return None
    return [
        [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
        [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
        [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
    ]


def zone_of(point) -> dict:
    """Which of the nine zones a court position falls in."""
    x, y = point
    depth = next((z for z in ZONES["depth"] if z[1] <= y < z[2]), None)
    if depth is None:
        depth = ZONES["depth"][0] if y < 0 else ZONES["depth"][-1]
    lateral = next((z for z in ZONES["lateral"] if z[1] <= x < z[2]), ZONES["lateral"][1])
    return {"depth": depth[0], "lateral": lateral[0], "name": f"{depth[0]}-{lateral[0]}"}


def distance_from_base(point) -> float:
    return math.hypot(point[0] - BASE_POSITION[0], point[1] - BASE_POSITION[1])


def on_court(point, margin: float = 0.5) -> bool:
    x, y = point
    return abs(x) <= Court.width / 2 + margin and -margin <= y <= Court.half_length + margin


def is_convex_quad(points, min_edge: float = 0.02) -> bool:
    """Is the quad simple and convex?

    The four corners of a real court always project to a convex quadrilateral. If
    the taps came in a non-cyclic order the quad self-intersects into a bow tie,
    and the resulting homography inverts perfectly well while mapping the player
    to nonsense. Winding direction is deliberately *not* checked: filming the
    same half from the other side of the hall genuinely mirrors it.
    """
    if len(points) != 4:
        return False
    for i in range(4):
        for j in range(i + 1, 4):
            if math.dist(points[i], points[j]) < min_edge:
                return False
    sign = 0
    for i in range(4):
        a, b, c = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) < 1e-9:
            return False
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


class CourtCalibration:
    """A calibrated court: normalised image points in, court metres out.

    Points are stored normalised (0..1) so the calibration survives the video
    being displayed at a different size, which it always is between a phone in
    portrait and a laptop.
    """

    def __init__(self, points):
        self.points = [(float(p[0]), float(p[1])) for p in points]
        dst = [HALF_COURT_CORNERS[k] for k in CALIBRATION_ORDER]
        self.to_court = homography_from_quad(self.points, dst)
        self.to_image = invert3x3(self.to_court) if self.to_court else None

    @property
    def valid(self) -> bool:
        if self.to_court is None or not is_convex_quad(self.points):
            return False
        centre = apply_homography(self.to_court, (
            sum(p[0] for p in self.points) / 4,
            sum(p[1] for p in self.points) / 4,
        ))
        return centre is not None and on_court(centre, 1.5)

    @property
    def mirrored(self) -> bool:
        a, b, c = self.points[:3]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        return cross > 0

    def swap_sides(self) -> "CourtCalibration":
        nl, nr, br, bl = self.points
        return CourtCalibration([nr, nl, bl, br])

    def court_point(self, image_point):
        return apply_homography(self.to_court, image_point) if self.to_court else None

    def image_point(self, court_point):
        return apply_homography(self.to_image, court_point) if self.to_image else None

    def to_json(self) -> dict:
        return {"points": [list(p) for p in self.points]}

    @staticmethod
    def from_json(data):
        pts = (data or {}).get("points")
        return CourtCalibration(pts) if pts and len(pts) == 4 else None


def foot_position(image) -> dict | None:
    """Where the player is standing, from their ankles.

    Ankles rather than foot tips: they are the more reliable landmark, and
    heel-to-toe is small next to the size of a zone.
    """
    left, right = image[L.LEFT_ANKLE], image[L.RIGHT_ANKLE]
    vis = min(left[3] if len(left) > 3 else 1.0, right[3] if len(right) > 3 else 1.0)
    return {"point": ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2), "visibility": vis}


def court_track(frames, calibration: CourtCalibration, min_visibility: float = 0.4) -> list:
    """Court position per frame, or ``None`` where it is not trustworthy."""
    positions = []
    for f in frames:
        image = f.get("image")
        if not image:
            positions.append(None)
            continue
        feet = foot_position(image)
        if feet is None or feet["visibility"] < min_visibility:
            positions.append(None)
            continue
        court = calibration.court_point(feet["point"])
        if court and on_court(court, 2.0):
            positions.append({
                "t": f["t"], "x": court[0], "y": court[1],
                "zone": zone_of(court), "base": distance_from_base(court),
            })
        else:
            positions.append(None)
    return positions


def distance_covered(positions, max_step: float = 1.2) -> float:
    """Total ground covered, in metres, ignoring gaps and obvious jitter."""
    total = 0.0
    previous = None
    for p in positions:
        if p is None:
            previous = None
            continue
        if previous is not None:
            step = math.hypot(p["x"] - previous["x"], p["y"] - previous["y"])
            # A larger step between adjacent frames is a tracking glitch, not a
            # movement any player makes in 1/30 s.
            if step <= max_step:
                total += step
        previous = p
    return total


def zone_occupancy(positions) -> dict:
    """How long each zone was occupied, in seconds."""
    seconds: dict[str, float] = {}
    for prev, p in zip(positions, positions[1:]):
        if p is None or prev is None:
            continue
        dt = p["t"] - prev["t"]
        if not (0 < dt <= 0.5):
            continue
        seconds[p["zone"]["name"]] = seconds.get(p["zone"]["name"], 0.0) + dt
    return seconds


def recovery_times(strokes, positions, radius: float = 1.0, limit: float = 3.0) -> list:
    """Time from each stroke until the player is back near base.

    Recovery is the habit that separates players who keep a rally going from
    players caught out of position, and it is invisible in a shot-by-shot angle
    readout.
    """
    known = [p for p in positions if p is not None]
    out = []
    for s in strokes:
        after = [p for p in known if s["t"] < p["t"] <= s["t"] + limit]
        home = next((p for p in after if p["base"] <= radius), None)
        at = next((p for p in known if p["t"] >= s["t"]), None)
        out.append({
            "stroke_index": s["index"],
            "t": s["t"],
            "position": {"x": at["x"], "y": at["y"], "zone": at["zone"]["name"], "base": at["base"]}
            if at else None,
            "recovery_seconds": home["t"] - s["t"] if home else None,
            "recovered": home is not None,
        })
    return out
