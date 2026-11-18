"""Stage 2: preprocessing, candidate generation, label derivation and feature extraction."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from src import config as C
from src.data_io import list_trips, load_trip
from src.preprocess import prepare_trip, attach_labels, clean_track

OUT = C.RESULTS_DIR / "stage2"
OUT.mkdir(parents=True, exist_ok=True)


def radius_k_grid():
    recs = []
    for tid in list_trips(C.TRAIN_DIR):
        T = load_trip(C.TRAIN_DIR, tid)
        p = attach_labels(prepare_trip(T, radius=100.0, k=20), T["route"])
        f = p["feats"]
        n = len(p["xy"])
        for R in (20, 30, 50, 75, 100):
            for K in (2, 4, 6, 8, 10, 15):
                sel = f[(f["dist"] <= R) & (f["rank"] < K) & (f["fallback"] == 0)]
                hit = sel.groupby("pt")["y"].max().sum()
                recs.append({"trip_id": tid, "R": R, "K": K, "n_points": n, "hits": hit, "rows": len(sel)})
    g = pd.DataFrame(recs).groupby(["R", "K"])[["n_points", "hits", "rows"]].sum()
    g["candidate_miss_rate"] = 1 - g["hits"] / g["n_points"]
    g["candidates_per_point"] = g["rows"] / g["n_points"]
    g = g.reset_index()
    g.to_csv(OUT / "candidate_radius_k_grid.csv", index=False)
    print(g.pivot(index="R", columns="K", values="candidate_miss_rate").round(4))
    print(g.pivot(index="R", columns="K", values="candidates_per_point").round(2))


def build(partition, pdir):
    feats, pts, summ = [], [], []
    for tid in list_trips(pdir):
        T = load_trip(pdir, tid)
        if T["arcs"] is None or T["route"] is None:
            summ.append({"partition": partition, "trip_id": tid, "status": "skipped: missing .arcs/.route"})
            continue
        t0 = time.time()
        _, ok = clean_track(T["track"])
        p = attach_labels(prepare_trip(T), T["route"])
        f = p["feats"]
        f.insert(0, "trip_id", tid)
        n = len(p["xy"])
        covered = np.zeros(n, bool)
        covered[f.loc[f["y"] == 1, "pt"].to_numpy()] = True
        nc = np.bincount(f["pt"], minlength=n)
        pts.append(pd.DataFrame({"trip_id": tid, "pt": np.arange(n), "t": p["track"][:, 2],
                                 "ref_arc": p["ref_arc"], "ref_dist": p["ref_dist"], "covered": covered,
                                 "n_cand": nc}))
        feats.append(f)
        summ.append({"partition": partition, "trip_id": tid, "status": "ok", "raw_points": len(T["track"]),
                     "dropped_invalid": int((~ok).sum()), "points": n, "rows": len(f),
                     "no_candidate_points": int((nc == 0).sum()),
                     "fallback_points": int(f.loc[f["fallback"] == 1, "pt"].nunique()),
                     "candidate_miss_rate": 1 - covered.mean(),
                     "label_unreliable_points(ref_dist>R)": int((p["ref_dist"] > C.CAND_RADIUS_M).sum()),
                     "sec": round(time.time() - t0, 2)})
        print(partition, tid, n, len(f), flush=True)
    F = pd.concat(feats, ignore_index=True)
    Pt = pd.concat(pts, ignore_index=True)
    F.to_pickle(C.FEATURES_DIR / f"{partition}.pkl")
    Pt.to_pickle(C.FEATURES_DIR / f"points_{partition}.pkl")
    return pd.DataFrame(summ)


if __name__ == "__main__":
    if "--skip-grid" not in sys.argv:
        radius_k_grid()
    s = pd.concat([build("train", C.TRAIN_DIR), build("student_test", C.STUDENT_TEST_DIR)])
    s.to_csv(OUT / "preprocessing_summary.csv", index=False)
    print(s.groupby("partition")[["points", "rows", "no_candidate_points", "fallback_points",
                                  "label_unreliable_points(ref_dist>R)"]].sum())
    ok = s[s.status == "ok"]
    for part, g in ok.groupby("partition"):
        print(part, "candidate miss rate (point-weighted):",
              round((g.candidate_miss_rate * g.points).sum() / g.points.sum(), 4))
