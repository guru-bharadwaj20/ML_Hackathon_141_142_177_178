"""Feature engineering: one row per (GPS point, candidate arc) pair."""
import numpy as np
import pandas as pd

from .geo import point_segment, angle_diff_deg, bearing_deg

FEATURES = ["dist", "rank", "dist_minus_min", "heading_diff", "speed", "arc_len", "tproj",
            "n_cand", "nbr_dist", "has_reverse", "fallback"]
EXTRA_FEATURES = ["heading_diff_long", "disp_long"]
LONG_W = 10


def motion(xy, t, w=2):
    n = len(xy)
    i0 = np.clip(np.arange(n) - w, 0, n - 1)
    i1 = np.clip(np.arange(n) + w, 0, n - 1)
    dx, dy = xy[i1, 0] - xy[i0, 0], xy[i1, 1] - xy[i0, 1]
    dtime = np.maximum(t[i1] - t[i0], 1e-6)
    speed = np.hypot(dx, dy) / dtime
    heading = bearing_deg(dx, dy)
    return heading, speed


def long_motion(xy, w=LONG_W):
    n = len(xy)
    i0 = np.clip(np.arange(n) - w, 0, n - 1)
    i1 = np.clip(np.arange(n) + w, 0, n - 1)
    d = xy[i1] - xy[i0]
    return bearing_deg(d[:, 0], d[:, 1]), np.hypot(d[:, 0], d[:, 1])


def reverse_flags(arcs):
    key = arcs[:, 0].astype(np.int64) * (arcs.max() + 1) + arcs[:, 1]
    rkey = arcs[:, 1].astype(np.int64) * (arcs.max() + 1) + arcs[:, 0]
    return np.isin(rkey, key).astype(np.int8)


def build_features(xy, t, cand, index, has_rev):
    pt, arc = cand["pt"], cand["arc"]
    heading, speed = motion(xy, t)
    df = pd.DataFrame({"pt": pt, "arc": arc, "dist": cand["dist"], "rank": cand["rank"],
                       "tproj": cand["tproj"], "fallback": cand["fallback"].astype(np.int8)})
    mind = df.groupby("pt")["dist"].transform("min").to_numpy()
    df["dist_minus_min"] = df["dist"].to_numpy() - mind
    df["n_cand"] = df.groupby("pt")["arc"].transform("size").to_numpy()
    df["heading_diff"] = angle_diff_deg(heading[pt], index.bearing[arc])
    df["speed"] = speed[pt]
    df["arc_len"] = index.length[arc]
    df["has_reverse"] = has_rev[arc]
    n = len(xy)
    acc = np.zeros(len(df))
    for off in (-2, -1, 1, 2):
        q = np.clip(pt + off, 0, n - 1)
        d, _ = point_segment(xy[q, 0], xy[q, 1], index.A[arc, 0], index.A[arc, 1], index.B[arc, 0], index.B[arc, 1])
        acc += d
    df["nbr_dist"] = acc / 4.0
    h_long, disp_long = long_motion(xy)
    df["heading_diff_long"] = angle_diff_deg(h_long[pt], index.bearing[arc])
    df["disp_long"] = disp_long[pt]
    return df
