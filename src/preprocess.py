"""Per-trip preprocessing shared by training, evaluation and the API."""
import numpy as np

from .geo import LocalProjection
from .candidates import ArcIndex, generate_candidates
from .features import build_features, reverse_flags
from .labels import align_route
from . import config as C


def clean_track(track):
    ok = np.isfinite(track).all(1) & (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    track = track[ok]
    keep = np.r_[True, np.diff(track[:, 2]) > 0]
    return track[keep], ok


def prepare_trip(trip, radius=C.CAND_RADIUS_M, k=C.CAND_K):
    track, _ = clean_track(trip["track"])
    proj = LocalProjection.from_track(track)
    xy = proj.to_xy(track[:, :2])
    nodes_xy = proj.to_xy(trip["nodes"])
    index = ArcIndex(nodes_xy, trip["arcs"])
    cand = generate_candidates(xy, index, radius=radius, k=k)
    has_rev = reverse_flags(trip["arcs"])
    feats = build_features(xy, track[:, 2], cand, index, has_rev)
    return {"trip_id": trip["trip_id"], "track": track, "xy": xy, "proj": proj,
            "index": index, "feats": feats}


def attach_labels(prep, route):
    idx = prep["index"]
    k, dist = align_route(prep["xy"], idx.A[route], idx.B[route])
    ref_arc = route[k]
    prep["ref_arc"] = ref_arc
    prep["ref_dist"] = dist
    f = prep["feats"]
    f["y"] = (f["arc"].to_numpy() == ref_arc[f["pt"].to_numpy()]).astype(np.int8)
    f["ref_dist"] = dist[f["pt"].to_numpy()]
    return prep
