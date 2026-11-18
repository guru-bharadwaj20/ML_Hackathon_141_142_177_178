"""Candidate generation: for each GPS point, the K nearest road segments (arcs) within a radius."""
import numpy as np
import shapely
from shapely import STRtree

from .geo import point_segment
from . import config as C


class ArcIndex:

    def __init__(self, nodes_xy, arcs):
        self.arcs = arcs
        self.A = nodes_xy[arcs[:, 0]]
        self.B = nodes_xy[arcs[:, 1]]
        self.length = np.hypot(*(self.B - self.A).T)
        self.bearing = np.degrees(np.arctan2(self.B[:, 1] - self.A[:, 1], self.B[:, 0] - self.A[:, 0])) % 360.0
        self.lines = shapely.linestrings(np.stack([self.A, self.B], axis=1))
        self.tree = STRtree(self.lines)

    def within(self, xy, radius):
        pts = shapely.points(xy)
        pi, ai = self.tree.query(pts, predicate="dwithin", distance=radius)
        return pi, ai


def generate_candidates(xy, index, radius=C.CAND_RADIUS_M, k=C.CAND_K, fallback=C.FALLBACK_RADIUS_M):
    pi, ai = index.within(xy, radius)
    d, t = point_segment(xy[pi, 0], xy[pi, 1], index.A[ai, 0], index.A[ai, 1], index.B[ai, 0], index.B[ai, 1])
    keep = d <= radius
    pi, ai, d, t = pi[keep], ai[keep], d[keep], t[keep]
    fb = np.zeros(len(pi), dtype=bool)

    have = np.zeros(len(xy), dtype=bool)
    have[pi] = True
    miss = np.where(~have)[0]
    if len(miss):
        idx, dist = index.tree.query_nearest(shapely.points(xy[miss]), max_distance=fallback,
                                             return_distance=True, all_matches=False)
        p2 = miss[idx[0]]
        a2 = idx[1]
        d2, t2 = point_segment(xy[p2, 0], xy[p2, 1], index.A[a2, 0], index.A[a2, 1], index.B[a2, 0], index.B[a2, 1])
        pi = np.concatenate([pi, p2]); ai = np.concatenate([ai, a2])
        d = np.concatenate([d, d2]); t = np.concatenate([t, t2])
        fb = np.concatenate([fb, np.ones(len(p2), dtype=bool)])

    order = np.lexsort((ai, d, pi))
    pi, ai, d, t, fb = pi[order], ai[order], d[order], t[order], fb[order]
    start = np.r_[0, np.flatnonzero(np.diff(pi)) + 1]
    counts = np.diff(np.r_[start, len(pi)])
    rank = np.arange(len(pi)) - np.repeat(start, counts)
    keep = rank < k
    return {"pt": pi[keep], "arc": ai[keep], "dist": d[keep], "tproj": t[keep],
            "rank": rank[keep], "fallback": fb[keep]}
