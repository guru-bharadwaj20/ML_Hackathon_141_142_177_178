"""Reading raw trip files (.track, .nodes, .arcs, .route)."""
from pathlib import Path
import numpy as np


def list_trips(partition_dir):
    """Return sorted trip ids (folder names) inside a partition folder."""
    p = Path(partition_dir)
    if not p.exists():
        return []
    return sorted(d.name for d in p.iterdir() if d.is_dir())


def _read(path, dtype, ncols):
    if not path.exists():
        return None
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    if arr.size == 0:
        return np.zeros((0, ncols), dtype=dtype)
    assert arr.shape[1] == ncols, f"{path}: expected {ncols} columns, got {arr.shape[1]}"
    return arr


def load_trip(partition_dir, trip_id, with_route=True):
    """Load one trip.

    Returns dict with:
      track : (N,3) float  lon, lat, t
      nodes : (V,2) float  lon, lat    (row index = node id)
      arcs  : (E,2) int    from, to    (row index = arc id)
      route : (M,)  int    reference arc ids, or None
    with_route=False never touches the .route file (used by the prediction pipeline).
    """
    d = Path(partition_dir) / trip_id
    out = {
        "trip_id": trip_id,
        "track": _read(d / f"{trip_id}.track", float, 3),
        "nodes": _read(d / f"{trip_id}.nodes", float, 2),
        "arcs": _read(d / f"{trip_id}.arcs", np.int64, 2),
        "route": None,
    }
    if with_route:
        r = d / f"{trip_id}.route"
        if r.exists():
            out["route"] = np.loadtxt(r, dtype=np.int64, ndmin=1)
    return out


def load_route(partition_dir, trip_id):
    r = Path(partition_dir) / trip_id / f"{trip_id}.route"
    return np.loadtxt(r, dtype=np.int64, ndmin=1) if r.exists() else None
