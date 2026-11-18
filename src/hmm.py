"""HMM map matching over classifier candidates, decoded with Viterbi."""
import heapq
import numpy as np

BACK_TOL_M = 15.0
MAX_GAP_S = 120.0
MIN_CUTOFF_M = 250.0
MAX_CUTOFF_M = 5000.0


class RoadGraph:
    def __init__(self, arcs, length):
        n = int(arcs.max()) + 1
        order = np.argsort(arcs[:, 0], kind="stable")
        self.ptr = np.searchsorted(arcs[order, 0], np.arange(n + 1)).tolist()
        self.to = arcs[order, 1].tolist()
        self.w = length[order].tolist()
        self.cache = {}

    def reach(self, src, cutoff):
        hit = self.cache.get(src)
        if hit is not None and hit[0] >= cutoff:
            return hit[1]
        if len(self.cache) > 20000:
            self.cache.clear()
        ptr, to, w = self.ptr, self.to, self.w
        dist = {src: 0.0}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for e in range(ptr[u], ptr[u + 1]):
                v, nd = to[e], d + w[e]
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        self.cache[src] = (cutoff, dist)
        return dist


def route_distance(graph, arcs, length, a0, t0, a1, t1, cutoff):
    D = np.full((len(a0), len(a1)), np.inf)
    for i in range(len(a0)):
        ai, ti = a0[i], t0[i]
        rem = (1.0 - ti) * length[ai]
        reach = None
        for j in range(len(a1)):
            aj, tj = a1[j], t1[j]
            if ai == aj:
                along = (tj - ti) * length[ai]
                if along >= -BACK_TOL_M:
                    D[i, j] = max(along, 0.0)
                    continue
            if reach is None:
                reach = graph.reach(arcs[ai, 1], cutoff)
            dn = reach.get(arcs[aj, 0])
            if dn is not None:
                D[i, j] = rem + dn + tj * length[aj]
    return D


def prepare_steps(cand, xy, t, arcs, length):
    graph = RoadGraph(arcs, length)
    pt = cand["pt"].to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(pt)) + 1, len(pt)]
    pts = pt[starts[:-1]]
    arc = cand["arc"].to_numpy()
    tp = cand["tproj"].to_numpy()
    costs = [None]
    for k in range(1, len(pts)):
        p0, p1 = pts[k - 1], pts[k]
        dt = t[p1] - t[p0]
        if dt > MAX_GAP_S:
            costs.append(None)
            continue
        gc = float(np.hypot(*(xy[p1] - xy[p0])))
        cutoff = min(max(MIN_CUTOFF_M, 2.0 * gc + 100.0), MAX_CUTOFF_M)
        s0, s1 = slice(starts[k - 1], starts[k]), slice(starts[k], starts[k + 1])
        D = route_distance(graph, arcs, length, arc[s0], tp[s0], arc[s1], tp[s1], cutoff)
        costs.append(np.abs(D - gc))
    return {"pts": pts, "starts": starts, "arc": arc, "costs": costs}


def viterbi(steps, logp, beta, w_em=1.0):
    pts, starts, arc, costs = steps["pts"], steps["starts"], steps["arc"], steps["costs"]
    n = len(pts)
    back = [None] * n
    em = [w_em * logp[starts[k]:starts[k + 1]] for k in range(n)]
    score = em[0].copy() if n else None
    seg_start = [0]
    finals = []
    for k in range(1, n):
        C = costs[k]
        if C is not None:
            M = score[:, None] - C / beta
            best = M.argmax(0)
            new = M[best, np.arange(M.shape[1])]
        if C is None or not np.isfinite(new).any():
            finals.append((k - 1, int(np.argmax(score))))
            seg_start.append(k)
            score = em[k].copy()
            continue
        back[k] = best
        score = new + em[k]
    if n:
        finals.append((n - 1, int(np.argmax(score))))
    choice = np.empty(n, dtype=np.int64)
    for (end, s), beg in zip(finals, seg_start):
        choice[end] = s
        for k in range(end, beg, -1):
            choice[k - 1] = back[k][choice[k]]
    pred = arc[starts[:-1] + choice] if n else np.zeros(0, dtype=np.int64)
    return pts, pred, len(seg_start) - 1
