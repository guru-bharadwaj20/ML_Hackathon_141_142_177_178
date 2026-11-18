"""Stage 3: feature selection + training/comparison of 7 models from different families.

Training/CV : 80 train trips, GroupKFold(5) by trip (no trip in both train and validation folds).
Comparison  : 9 usable student_test trips (00000000 has no .arcs file).
Task        : binary classification of each (GPS point, candidate arc) row: is this the reference arc?
Outputs     : results/stage3/*.csv, results/plots/s3_*.png, models/stage3_<model>.joblib
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, log_loss, confusion_matrix

from src import config as C
from src.features import FEATURES
from src.data_io import load_trip

OUT = C.RESULTS_DIR / "stage3"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.RandomState(C.SEED)


def load_train():
    F = pd.read_pickle(C.FEATURES_DIR / "train.pkl")
    F = F[F["ref_dist"] <= C.CAND_RADIUS_M]                    # drop unreliable labels (off-network GPS)
    keep = []
    for tid, g in F.groupby("trip_id"):                         # cap points per trip (uniform stride)
        pts = np.unique(g["pt"].to_numpy())
        step = max(1, int(np.ceil(len(pts) / C.MAX_POINTS_PER_TRIP)))
        keep.append(g[g["pt"].isin(pts[::step])])
    return pd.concat(keep, ignore_index=True)


def models():
    return {
        "LogisticRegression": ("Linear", "C=1.0, lbfgs, max_iter=1000, standardised",
                               make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))),
        "GaussianNB": ("Probabilistic (Bayesian)", "var_smoothing=1e-9",
                       GaussianNB()),
        "KNN": ("Instance-based", "k=25, distance-weighted, euclidean, standardised",
                make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1))),
        "DecisionTree": ("Tree-based", "criterion=entropy, max_depth=10, min_samples_leaf=20",
                         DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_leaf=20, random_state=C.SEED)),
        "RandomForest": ("Ensemble (bagging)", "200 trees, min_samples_leaf=5, max_features=sqrt",
                         RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features="sqrt",
                                                n_jobs=-1, random_state=C.SEED)),
        "GradientBoosting": ("Boosting", "HistGradientBoosting, lr=0.1, max_iter=300, max_leaf_nodes=31",
                             HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300, random_state=C.SEED)),
        "MLP": ("Neural network (ANN)", "hidden=(64,32), relu, adam, lr=1e-3, early_stopping, standardised",
                make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), learning_rate_init=1e-3,
                                                              max_iter=200, early_stopping=True, n_iter_no_change=8,
                                                              random_state=C.SEED))),
    }


def top1_and_continuity(df, prob, points, arcs_by_trip):
    """Observation-level Top-1 accuracy and route continuity of the argmax predictions.

    Top-1 is over ALL GPS points of the partition (points without candidates count as wrong).
    Continuity: share of consecutive point pairs (gap <= 10 s) whose predicted arcs are the same arc
    or directly connected (to_node of first == from_node of second).
    """
    d = df[["trip_id", "pt", "arc"]].copy()
    d["p"] = prob
    best = d.loc[d.groupby(["trip_id", "pt"])["p"].idxmax()]
    P = points.merge(best[["trip_id", "pt", "arc"]], on=["trip_id", "pt"], how="left")
    P["arc"] = P["arc"].fillna(-1).astype(int)
    top1 = float((P["arc"] == P["ref_arc"]).mean())
    cont_n = cont_ok = ref_ok = 0
    for tid, g in P.groupby("trip_id"):
        arcs = arcs_by_trip[tid]
        a, t, r = g["arc"].to_numpy(), g["t"].to_numpy(), g["ref_arc"].to_numpy()
        m = (np.diff(t) <= 10) & (a[:-1] >= 0) & (a[1:] >= 0)
        a0, a1 = a[:-1][m], a[1:][m]
        cont_ok += int(((a0 == a1) | (arcs[a0, 1] == arcs[a1, 0])).sum())
        r0, r1 = r[:-1][m], r[1:][m]
        ref_ok += int(((r0 == r1) | (arcs[r0, 1] == arcs[r1, 0])).sum())
        cont_n += int(m.sum())
    return top1, cont_ok / cont_n, ref_ok / cont_n, P


def scores(y, p):
    yhat = (p >= 0.5).astype(int)
    return {"macro_f1": f1_score(y, yhat, average="macro"), "accuracy": accuracy_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0), "recall": recall_score(y, yhat),
            "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))}


def feature_selection(X, y, groups):
    """Information-gain (entropy) importance from a Random Forest, then backward elimination by CV Macro F1."""
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, criterion="entropy", n_jobs=-1, random_state=C.SEED)
    rf.fit(X[FEATURES], y)
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    gkf = GroupKFold(n_splits=5)
    recs = []
    feats = list(imp.index)
    while len(feats) >= 3:
        f1s = []
        for tr, va in gkf.split(X, y, groups):
            m = HistGradientBoostingClassifier(max_iter=150, random_state=C.SEED).fit(X.iloc[tr][feats], y[tr])
            f1s.append(f1_score(y[va], m.predict(X.iloc[va][feats]), average="macro"))
        recs.append({"n_features": len(feats), "features": ",".join(feats), "dropped_next": feats[-1],
                     "cv_macro_f1": np.mean(f1s), "cv_std": np.std(f1s)})
        print("FS", len(feats), round(np.mean(f1s), 4), flush=True)
        feats = feats[:-1]
    fs = pd.DataFrame(recs)
    best = fs["cv_macro_f1"].max()
    chosen = fs[fs["cv_macro_f1"] >= best - 0.002].sort_values("n_features").iloc[0]
    return imp, fs, chosen["features"].split(",")


def main():
    X = load_train()
    y = X["y"].to_numpy()
    groups = X["trip_id"].to_numpy()
    print("train rows", len(X), "points", X.groupby(["trip_id", "pt"]).ngroups, "positive rate", y.mean().round(3))

    imp, fs, SEL = feature_selection(X, y, groups)
    imp.rename("rf_entropy_importance").to_csv(OUT / "feature_importance.csv")
    fs.to_csv(OUT / "feature_selection_backward.csv", index=False)
    json.dump({"selected_features": SEL, "all_features": FEATURES}, open(C.CONFIGS_DIR / "features.json", "w"), indent=2)
    print("selected features:", SEL)

    T = pd.read_pickle(C.FEATURES_DIR / "student_test.pkl")
    Tpts = pd.read_pickle(C.FEATURES_DIR / "points_student_test.pkl")
    arcs_by_trip = {tid: load_trip(C.STUDENT_TEST_DIR, tid, with_route=False)["arcs"] for tid in Tpts.trip_id.unique()}
    Xt, yt = T[SEL], T["y"].to_numpy()

    gkf = GroupKFold(n_splits=5)
    rows, folds = [], []
    for name, (family, cfg, est) in models().items():
        f1s, lls = [], []
        for k, (tr, va) in enumerate(gkf.split(X, y, groups)):
            m = joblib.loads(joblib.dumps(est)) if False else __import__("sklearn").base.clone(est)
            m.fit(X.iloc[tr][SEL], y[tr])
            pv = m.predict_proba(X.iloc[va][SEL])[:, 1]
            s = scores(y[va], pv)
            f1s.append(s["macro_f1"]); lls.append(s["log_loss"])
            folds.append({"model": name, "fold": k, **s})
        t0 = time.time()
        m = __import__("sklearn").base.clone(est).fit(X[SEL], y)
        train_time = time.time() - t0
        tr_s = scores(y, m.predict_proba(X[SEL])[:, 1])
        t0 = time.time()
        pt = m.predict_proba(Xt)[:, 1]
        inf_time = time.time() - t0
        s = scores(yt, pt)
        top1, cont, ref_cont, _ = top1_and_continuity(T, pt, Tpts, arcs_by_trip)
        cm = confusion_matrix(yt, (pt >= 0.5).astype(int))
        rows.append({"model": name, "family": family, "config": cfg,
                     "cv_macro_f1": np.mean(f1s), "cv_macro_f1_std": np.std(f1s), "cv_log_loss": np.mean(lls),
                     "train_macro_f1": tr_s["macro_f1"],
                     "test_macro_f1": s["macro_f1"], "test_accuracy": s["accuracy"], "test_precision": s["precision"],
                     "test_recall": s["recall"], "test_log_loss": s["log_loss"], "test_top1_accuracy": top1,
                     "test_route_continuity": cont, "reference_route_continuity": ref_cont,
                     "train_time_s": train_time, "inference_time_s": inf_time,
                     "inference_ms_per_1k_rows": 1000 * inf_time / len(Xt),
                     "tn": cm[0, 0], "fp": cm[0, 1], "fn": cm[1, 0], "tp": cm[1, 1]})
        joblib.dump(m, C.MODELS_DIR / f"stage3_{name}.joblib")
        if name == "MLP":
            mlp = m[-1]
            pd.DataFrame({"epoch": np.arange(1, len(mlp.loss_curve_) + 1), "train_loss": mlp.loss_curve_,
                          "val_accuracy": mlp.validation_scores_}).to_csv(OUT / "mlp_history.csv", index=False)
        print(pd.Series(rows[-1]).drop(["config"]).to_string(), "\n", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "model_comparison.csv", index=False)
    pd.DataFrame(folds).to_csv(OUT / "cv_folds.csv", index=False)
    plots(R, pd.DataFrame(folds), imp, fs)
    print(R[["model", "cv_macro_f1", "cv_macro_f1_std", "train_macro_f1", "test_macro_f1", "test_top1_accuracy",
             "test_log_loss", "test_route_continuity", "train_time_s", "inference_time_s"]].round(4).to_string())


def plots(R, folds, imp, fs):
    P = C.PLOTS_DIR
    order = R.sort_values("cv_macro_f1")["model"]
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [folds[folds.model == m]["macro_f1"] for m in order]
    ax.boxplot(data, vert=False, tick_labels=list(order))
    ax.set_xlabel("Macro F1 per CV fold (GroupKFold=5, train trips)"); ax.set_title("Cross-validation Macro F1")
    fig.tight_layout(); fig.savefig(P / "s3_cv_macro_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(R)); w = 0.38
    ax.bar(x - w / 2, R["test_top1_accuracy"], w, label="Top-1 accuracy")
    ax.bar(x + w / 2, R["test_macro_f1"], w, label="Macro F1")
    ax.set_xticks(x, R["model"], rotation=20); ax.set_ylim(0, 1); ax.legend(); ax.set_title("Student-test performance")
    for i, (a, b) in enumerate(zip(R["test_top1_accuracy"], R["test_macro_f1"])):
        ax.text(i - w / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=7); ax.text(i + w / 2, b + 0.01, f"{b:.3f}", ha="center", fontsize=7)
    fig.tight_layout(); fig.savefig(P / "s3_test_top1_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(R["model"], R["test_log_loss"], color="C2"); ax.set_ylabel("log loss (lower is better)")
    ax.set_title("Student-test log loss"); plt.setp(ax.get_xticklabels(), rotation=20)
    fig.tight_layout(); fig.savefig(P / "s3_test_log_loss.png", dpi=130); plt.close(fig)

    fig, axs = plt.subplots(1, len(R), figsize=(3 * len(R), 3))
    for ax, (_, r) in zip(axs, R.iterrows()):
        cm = np.array([[r.tn, r.fp], [r.fn, r.tp]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"]); ax.set_title(r.model, fontsize=9)
    fig.tight_layout(); fig.savefig(P / "s3_confusion_matrices.png", dpi=130); plt.close(fig)

    h = pd.read_csv(OUT / "mlp_history.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h.epoch, h.train_loss, label="training loss (log loss)")
    ax2 = ax.twinx(); ax2.plot(h.epoch, h.val_accuracy, color="C1", label="validation accuracy")
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss"); ax2.set_ylabel("validation accuracy (10% hold-out)")
    ax.set_title("MLP training history (only model with epoch-wise history)")
    fig.legend(loc="center right"); fig.tight_layout(); fig.savefig(P / "s3_mlp_history.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    imp.sort_values().plot.barh(ax=ax[0]); ax[0].set_title("Random-Forest information-gain importance")
    ax[1].plot(fs.n_features, fs.cv_macro_f1, "o-"); ax[1].invert_xaxis()
    ax[1].set_xlabel("number of features kept (least important dropped first)"); ax[1].set_ylabel("CV Macro F1")
    ax[1].set_title("Backward feature elimination")
    fig.tight_layout(); fig.savefig(P / "s3_feature_selection.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
