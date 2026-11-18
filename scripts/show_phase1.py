"""Print the Phase 1 answers (preprocessing, k values, feature selection, models, choice) from the saved results."""
import sys, json
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C

S2, S3 = C.RESULTS_DIR / "stage2", C.RESULTS_DIR / "stage3"
W = 96
pd.set_option("display.width", 200)


def title(t):
    print("\n" + "=" * W + "\n " + t + "\n" + "=" * W)


def main():
    pre = pd.read_csv(S2 / "preprocessing_summary.csv")
    ok = pre[pre.status == "ok"]
    grid = pd.read_csv(S2 / "candidate_radius_k_grid.csv")
    imp = pd.read_csv(S3 / "feature_importance.csv", index_col=0).iloc[:, 0]
    fs = pd.read_csv(S3 / "feature_selection_backward.csv")
    sel = json.load(open(C.CONFIGS_DIR / "features.json"))["selected_features"]
    M = pd.read_csv(S3 / "model_comparison.csv")

    title("1. PREPROCESSING")
    for part, g in ok.groupby("partition"):
        miss = (g.candidate_miss_rate * g.points).sum() / g.points.sum()
        print(f" {part:<13} trips={len(g):>3}  GPS points={int(g.points.sum()):>8,}  candidate rows={int(g.rows.sum()):>10,}"
              f"  candidate miss rate={miss:.2%}")
    print(" Skipped:", ", ".join(f"{r.partition} {r.trip_id:08d} ({r.status})" for r in pre[pre.status != "ok"].itertuples()))
    print(" Steps: clean track -> metre projection -> candidates (R, K) -> labels by route alignment -> 11 features")
    print(" Training rows: drop points > 50 m from route, cap 1,000 points per trip")

    title("2. K VALUES")
    print(" (a) Candidate K: how many nearby roads we keep per GPS point (grid on train trips)")
    g = grid[grid.R == C.CAND_RADIUS_M][["K", "candidate_miss_rate", "candidates_per_point"]]
    print(g.rename(columns={"candidate_miss_rate": "miss_rate", "candidates_per_point": "cands_per_point"})
          .round(4).to_string(index=False))
    print(f"     -> chosen R = {C.CAND_RADIUS_M:.0f} m, K = {C.CAND_K}")
    print("\n (b) KNN k: number of neighbours = 25, distance-weighted (one setting, not tuned)")
    print("\n (c) Cross-validation folds: k = 5 (GroupKFold, grouped by trip)")

    title("3. FEATURE SELECTION  (RF entropy importance + backward elimination, scored by CV macro F1)")
    print(" Importance ranking:")
    for f, v in imp.items():
        print(f"   {f:<15} {v:.3f}  {'#' * int(round(v * 100))}")
    print("\n Backward elimination:")
    print(fs[["n_features", "dropped_next", "cv_macro_f1", "cv_std"]].round(4).to_string(index=False))
    print(f"\n -> kept {len(sel)} features: {', '.join(sel)}")
    print(f" -> dropped: {', '.join(f for f in imp.index if f not in sel)}")

    title("4. ALL MODELS  (CV on 80 train trips, test on 9 student-test trips)")
    cols = {"model": "Model", "cv_macro_f1": "CV F1", "train_macro_f1": "Train F1", "test_macro_f1": "Test F1",
            "test_accuracy": "Test Acc", "test_precision": "Prec", "test_recall": "Recall",
            "test_top1_accuracy": "Top-1", "test_log_loss": "LogLoss", "test_route_continuity": "Continuity",
            "train_time_s": "Train s", "inference_time_s": "Infer s"}
    T = M[list(cols)].rename(columns=cols)
    T["Model"] = T["Model"].replace({"GradientBoosting": "HistGradientBoosting"})
    print(T.round(4).to_string(index=False))

    title("5. FINAL CHOICE")
    best = M.loc[M.cv_macro_f1.idxmax()]
    print(f" HistGradientBoostingClassifier  (best CV macro F1 = {best.cv_macro_f1:.4f}, "
          f"lowest test log loss = {M.test_log_loss.min():.3f})")
    print(" Why: ties Random Forest on accuracy, but overfits less (train-CV gap 0.012 vs 0.036),")
    print("      gives better probabilities for the HMM, is 0.56 MB vs 178 MB and predicts ~3x faster.")
    print()


if __name__ == "__main__":
    main()
