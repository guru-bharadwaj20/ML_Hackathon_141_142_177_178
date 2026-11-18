"""Derive per-GPS-point reference labels from the .route file."""
import numpy as np
from .geo import point_segment


# Uses the .route file: training and evaluation only, never the prediction pipeline.
def align_route(xy, A, B):
    N, M = len(xy), len(A)
    dt = np.int16 if M < 32000 else np.int32
    back = np.empty((N, M), dtype=dt)
    ar = np.arange(M)
    d0, _ = point_segment(xy[0, 0], xy[0, 1], A[:, 0], A[:, 1], B[:, 0], B[:, 1])
    cost = d0
    back[0] = ar
    for t in range(1, N):
        cm = np.minimum.accumulate(cost)
        arg = np.maximum.accumulate(np.where(cost <= cm, ar, 0))
        dt_, _ = point_segment(xy[t, 0], xy[t, 1], A[:, 0], A[:, 1], B[:, 0], B[:, 1])
        cost = cm + dt_
        back[t] = arg
    k = np.empty(N, dtype=np.int64)
    k[-1] = int(np.argmin(cost))
    for t in range(N - 1, 0, -1):
        k[t - 1] = back[t, k[t]]
    dist, _ = point_segment(xy[:, 0], xy[:, 1], A[k, 0], A[k, 1], B[k, 0], B[k, 1])
    return k, dist
