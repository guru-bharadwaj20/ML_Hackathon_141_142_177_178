"""Stage 5: HMM + Viterbi route reconstruction, beta tuned on out-of-fold train probabilities.

python scripts/s5_hmm.py
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier

from src import config as C
from src.data_io import load_trip
from src.geo import LocalProjection
from src.preprocess import clean_track
from src import hmm
from scripts.s3_models import load_train

OUT = C.RESULTS_DIR / "stage5"
OUT.mkdir(parents=True, exist_ok=True)
BETAS = [0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 100]


def trip_geometry(part_dir, tid):
    trip = load_trip(part_dir, tid, with_route=False)
    track, _ = clean_track(trip["track"])
    proj = LocalProjection.from_track(track)
    xy, nodes = proj.to_xy(track[:, :2]), proj.to_xy(trip["nodes"])
    arcs = trip["arcs"]
    length = np.hypot(*(nodes[arcs[:, 1]] - nodes[arcs[:, 0]]).T)
    return xy, track[:, 2], arcs, length


def build(F, part_dir):
    out = {}
    t0 = time.time()
    for tid, g in F.groupby("trip_id", sort=True):
        xy, t, arcs, length = trip_geometry(part_dir, tid)
        g = g.sort_values(["pt", "rank"])
        steps = hmm.prepare_steps(g, xy, t, arcs, length)
        out[tid] = {"steps": steps, "logp": np.log(np.clip(g["p"].to_numpy(), 1e-6, 1)),
                    "argmax": g.loc[g.groupby("pt")["p"].idxmax(), ["pt", "arc"]].to_numpy(), "arcs": arcs}
    print(f"built HMM inputs for {len(out)} trips in {time.time() - t0:.0f}s", flush=True)
    return out


def decode(H, beta):
    rows = []
    for tid, h in H.items():
        if beta is None:
            pts, pred, br = h["argmax"][:, 0], h["argmax"][:, 1], 0
        else:
            pts, pred, br = hmm.viterbi(h["steps"], h["logp"], beta)
        rows.append(pd.DataFrame({"trip_id": tid, "pt": pts, "pred": pred, "breaks": br}))
    return pd.concat(rows, ignore_index=True)


def evaluate(D, points, H, mask=None):
    P = points.merge(D[["trip_id", "pt", "pred"]], on=["trip_id", "pt"], how="left")
    P["pred"] = P["pred"].fillna(-1).astype(int)
    if mask is not None:
        P = P[mask(P)]
    top1 = float((P.pred == P.ref_arc).mean())
    n = ok = 0
    per = []
    for tid, g in P.groupby("trip_id"):
        arcs = H[tid]["arcs"]
        a, t = g.pred.to_numpy(), g.t.to_numpy()
        m = (np.diff(t) <= 10) & (a[:-1] >= 0) & (a[1:] >= 0)
        a0, a1 = a[:-1][m], a[1:][m]
        c = int(((a0 == a1) | (arcs[a0, 1] == arcs[a1, 0])).sum())
        ok += c; n += int(m.sum())
        ps, rs = set(a[a >= 0].tolist()), set(g.ref_arc.tolist())
        inter = len(ps & rs)
        per.append({"trip_id": tid, "points": len(g), "top1": float((g.pred == g.ref_arc).mean()),
                    "continuity": c / max(1, int(m.sum())), "arc_precision": inter / max(1, len(ps)),
                    "arc_recall": inter / max(1, len(rs))})
    per = pd.DataFrame(per)
    pr, rc = per.arc_precision.mean(), per.arc_recall.mean()
    return {"top1": top1, "continuity": ok / max(1, n), "route_arc_precision": pr, "route_arc_recall": rc,
            "route_arc_f1": 2 * pr * rc / (pr + rc), "trips_top1_ge_0.9": float((per.top1 >= 0.9).mean())}, per


def main():
    fm = json.load(open(C.CONFIGS_DIR / "final_model.json"))
    SEL, params = fm["features"], fm["params"]
    make = lambda: HistGradientBoostingClassifier(early_stopping=False, random_state=C.SEED, **params)

    X = load_train()
    folds = GroupKFold(n_splits=5).split(X, X["y"], X["trip_id"])
    F = pd.read_pickle(C.FEATURES_DIR / "train.pkl")
    F["p"] = np.nan
    t0 = time.time()
    for tr, va in folds:
        m = make().fit(X.iloc[tr][SEL], X["y"].iloc[tr])
        held = np.isin(F.trip_id, X.trip_id.iloc[va].unique())
        F.loc[held, "p"] = m.predict_proba(F.loc[held, SEL])[:, 1]
    assert F.p.notna().all()
    print(f"out-of-fold train probabilities in {time.time() - t0:.0f}s", flush=True)
    Ptr = pd.read_pickle(C.FEATURES_DIR / "points_train.pkl")
    Htr = build(F, C.TRAIN_DIR)
    reliable = lambda P: P.ref_dist <= C.CAND_RADIUS_M

    T = pd.read_pickle(C.FEATURES_DIR / "student_test.pkl")
    model = joblib.load(C.MODELS_DIR / fm["file"])
    T["p"] = model.predict_proba(T[SEL])[:, 1]
    Pte = pd.read_pickle(C.FEATURES_DIR / "points_student_test.pkl")
    Hte = build(T, C.STUDENT_TEST_DIR)

    grid = []
    for beta in [None] + BETAS:
        t0 = time.time()
        Dtr = decode(Htr, beta)
        s_tr, _ = evaluate(Dtr, Ptr, Htr, reliable)
        dec_s = time.time() - t0
        s_te, _ = evaluate(decode(Hte, beta), Pte, Hte)
        grid.append({"beta": "classifier only" if beta is None else beta,
                     **{f"train_oof_{k}": v for k, v in s_tr.items()},
                     **{f"test_{k}": v for k, v in s_te.items()},
                     "train_breaks": int(Dtr.groupby("trip_id").breaks.first().sum()), "train_decode_s": dec_s})
        print(grid[-1]["beta"], round(s_tr["top1"], 4), round(s_te["top1"], 4), round(s_te["continuity"], 4), flush=True)
    G = pd.DataFrame(grid)
    G.to_csv(OUT / "hmm_beta_grid.csv", index=False)

    hm = G.iloc[1:]
    best_beta = float(hm.loc[hm.train_oof_top1.idxmax(), "beta"])
    json.dump({"beta_m": best_beta, "emission": "log P(candidate is the true arc) from the final classifier",
               "transition": "-|route distance - GPS distance| / beta", "back_tolerance_m": hmm.BACK_TOL_M,
               "max_gap_s": hmm.MAX_GAP_S, "cutoff_m": [hmm.MIN_CUTOFF_M, "2*gps_dist+100", hmm.MAX_CUTOFF_M],
               "chosen_by": "best Top-1 on out-of-fold train probabilities (points within 50 m of the route)"},
              open(C.CONFIGS_DIR / "hmm.json", "w"), indent=2)

    t0 = time.time()
    Dh = decode(Hte, best_beta)
    t_dec = time.time() - t0
    s_c, per_c = evaluate(decode(Hte, None), Pte, Hte)
    s_h, per_h = evaluate(Dh, Pte, Hte)
    s_ref, _ = evaluate(Pte.assign(pred=Pte.ref_arc)[["trip_id", "pt", "pred"]], Pte, Hte)
    cmp_ = pd.DataFrame([{"method": "classifier only (argmax)", **s_c},
                         {"method": f"classifier + HMM (beta={best_beta:g})", **s_h},
                         {"method": "reference route", **s_ref}])
    cmp_["chain_breaks"] = [0, int(Dh.groupby("trip_id").breaks.first().sum()), 0]
    cmp_.to_csv(OUT / "classifier_vs_hmm.csv", index=False)
    per = per_c.merge(per_h, on=["trip_id", "points"], suffixes=("_classifier", "_hmm"))
    per.to_csv(OUT / "per_trip_student_test.csv", index=False)
    Dh.to_csv(OUT / "hmm_predictions_student_test.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    b = hm.beta.astype(float)
    ax[0].semilogx(b, hm.train_oof_top1, "o-", label="train (out-of-fold)")
    ax[0].semilogx(b, hm.test_top1, "s-", label="student test")
    ax[0].axhline(G.train_oof_top1[0], ls="--", color="C0", alpha=.5); ax[0].axhline(G.test_top1[0], ls="--", color="C1", alpha=.5)
    ax[0].axvline(best_beta, color="grey", ls=":")
    ax[0].set_xlabel("beta (m)"); ax[0].set_ylabel("Top-1 accuracy"); ax[0].set_title("HMM beta (dashed = classifier only)"); ax[0].legend()
    x = np.arange(len(per)); w = .4
    ax[1].bar(x - w / 2, per.top1_classifier, w, label="classifier only")
    ax[1].bar(x + w / 2, per.top1_hmm, w, label="classifier + HMM")
    ax[1].set_xticks(x, per.trip_id, rotation=30); ax[1].set_ylim(0, 1.05); ax[1].set_title("Student-test Top-1 per trip"); ax[1].legend()
    fig.tight_layout(); fig.savefig(C.PLOTS_DIR / "s5_hmm_beta_per_trip.png", dpi=130); plt.close(fig)

    print(cmp_.round(4).to_string(index=False))
    print(per.round(3).to_string(index=False))
    print("beta", best_beta, "student-test decode seconds", round(t_dec, 2))


if __name__ == "__main__":
    main()
