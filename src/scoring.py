"""Server-side scoring against the reference route. Called only after prediction is complete."""
import numpy as np

from .data_io import load_route
from .labels import align_route


def route_metrics(pred, ref, t, arcs):
    top1 = float((pred == ref).mean())
    m = (np.diff(t) <= 10) & (pred[:-1] >= 0) & (pred[1:] >= 0)
    a0, a1 = pred[:-1][m], pred[1:][m]
    cont = float(((a0 == a1) | (arcs[a0, 1] == arcs[a1, 0])).sum() / max(1, m.sum()))
    ps, rs = set(pred[pred >= 0].tolist()), set(ref.tolist())
    inter = len(ps & rs)
    return {"top1_accuracy": top1, "route_continuity": cont,
            "route_arc_precision": inter / max(1, len(ps)), "route_arc_recall": inter / max(1, len(rs))}


def score(part_dir, result):
    route = load_route(part_dir, result["trip_id"])
    if route is None:
        return None
    prep, arcs = result["prep"], result["trip"]["arcs"]
    idx = prep["index"]
    k, _ = align_route(prep["xy"], idx.A[route], idx.B[route])
    ref = route[k]
    t = prep["track"][:, 2]
    return {"hmm": route_metrics(result["pred_hmm"], ref, t, arcs),
            "classifier_only": route_metrics(result["pred_clf"], ref, t, arcs),
            "points": int(len(ref))}
