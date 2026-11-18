"""Coordinate handling.

Longitude/latitude are angles, so Euclidean distance in degrees is not a distance in metres.
Each trip covers only a city-sized area, so we use a local equirectangular projection centred on
the trip: x = R*dlon*cos(lat0), y = R*dlat (radians). Over ~20 km the error is far below GPS noise.
"""
import numpy as np

EARTH_R = 6371008.8


class LocalProjection:
    def __init__(self, lon0, lat0):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self.kx = EARTH_R * np.cos(np.radians(self.lat0)) * np.pi / 180.0
        self.ky = EARTH_R * np.pi / 180.0

    @classmethod
    def from_track(cls, track):
        return cls(np.median(track[:, 0]), np.median(track[:, 1]))

    def to_xy(self, lonlat):
        lonlat = np.asarray(lonlat, dtype=float)
        return np.column_stack([(lonlat[:, 0] - self.lon0) * self.kx,
                                (lonlat[:, 1] - self.lat0) * self.ky])


def point_segment(px, py, ax, ay, bx, by):
    """Vectorised distance from points P to segments AB.

    Returns (distance, t) where t in [0,1] is the position of the closest point along AB.
    All inputs are broadcastable arrays (metres).
    """
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * dx + (py - ay) * dy) / L2
    t = np.where(L2 > 0, np.clip(t, 0.0, 1.0), 0.0)
    cx, cy = ax + t * dx, ay + t * dy
    return np.hypot(px - cx, py - cy), t


def angle_diff_deg(a, b):
    """Absolute difference between two bearings (degrees), in [0,180]."""
    d = np.abs((a - b + 180.0) % 360.0 - 180.0)
    return d


def bearing_deg(dx, dy):
    """Bearing of vector (dx,dy) measured from the x (east) axis, degrees in [0,360)."""
    return np.degrees(np.arctan2(dy, dx)) % 360.0
