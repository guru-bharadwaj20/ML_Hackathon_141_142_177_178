"""Stage 4: long-heading feature ablation, then Optuna TPE tuning of HistGradientBoostingClassifier.

python scripts/s4_optuna.py [n_trials]
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import joblib
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier

from src import config as C
from src.data_io import load_trip
from src.features import EXTRA_FEATURES
from scripts.s3_models import load_train, scores, top1_and_continuity

OUT = C.RESULTS_DIR / "stage4"
OUT.mkdir(parents=True, exist_ok=True)
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 50

SPACE = {
    "learning_rate": ("float_log", 0.02, 0.3),
    "max_iter": ("int", 100, 800),
    "max_leaf_nodes": ("int_log", 15, 255),
    "min_samples_leaf": ("int_log", 10, 400),
    "l2_regularization": ("float_log", 1e-4, 10.0),
    "max_features": ("float", 0.5, 1.0),
}


def suggest(trial):
    p = {}
    for k, (kind, lo, hi) in SPACE.items():
        if kind == "float_log":
            p[k] = trial.suggest_float(k, lo, hi, log=True)
        elif kind == "float":
            p[k] = trial.suggest_float(k, lo, hi)
        elif kind == "int_log":
            p[k] = trial.suggest_int(k, lo, hi, log=True)
        else:
            p[k] = trial.suggest_int(k, lo, hi)
    return p


def make(p):
    return HistGradientBoostingClassifier(early_stopping=False, random_state=C.SEED, **p)


def cv_eval(X, y, folds, feats, p):
    cv = [scores(y[va], make(p).fit(X.iloc[tr][feats], y[tr]).predict_proba(X.iloc[va][feats])[:, 1]) for tr, va in folds]
    f = [c["macro_f1"] for c in cv]
    return {"cv_macro_f1": np.mean(f), "cv_macro_f1_std": np.std(f), "cv_log_loss": np.mean([c["log_loss"] for c in cv])}


def main():
    P1 = json.load(open(C.CONFIGS_DIR / "features.json"))["selected_features"]
    SEL = P1 + EXTRA_FEATURES
    X = load_train()
    y, groups = X["y"].to_numpy(), X["trip_id"].to_numpy()
    folds = list(GroupKFold(n_splits=5).split(X, y, groups))
    XS = X[SEL]
    base = {"learning_rate": 0.1, "max_iter": 150, "max_leaf_nodes": 31, "min_samples_leaf": 20,
            "l2_regularization": 1e-4, "max_features": 1.0}
    abl = []
    for name, feats in (("Phase 1 (9 features)", P1), ("+ heading_diff_long", P1 + EXTRA_FEATURES[:1]),
                        ("+ heading_diff_long + disp_long", SEL)):
        abl.append({"feature_set": name, "n_features": len(feats), **cv_eval(X, y, folds, feats, base)})
        print(abl[-1], flush=True)
    pd.DataFrame(abl).to_csv(OUT / "feature_ablation.csv", index=False)

    def objective(trial):
        p = suggest(trial)
        f1s, lls = [], []
        t0 = time.time()
        for k, (tr, va) in enumerate(folds):
            m = make(p).fit(XS.iloc[tr], y[tr])
            s = scores(y[va], m.predict_proba(XS.iloc[va])[:, 1])
            f1s.append(s["macro_f1"]); lls.append(s["log_loss"])
            trial.report(float(np.mean(f1s)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("cv_std", float(np.std(f1s)))
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        trial.set_user_attr("seconds", time.time() - t0)
        return float(np.mean(f1s))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="hgb_tpe",
                                sampler=optuna.samplers.TPESampler(seed=C.SEED, n_startup_trials=10, multivariate=True),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
    study.enqueue_trial(base)
    t_search = time.time()
    study.optimize(objective, n_trials=N_TRIALS,
                   callbacks=[lambda s, t: print(t.number, t.state.name, t.value and round(t.value, 4),
                                                 {k: round(v, 4) for k, v in t.params.items()}, flush=True)])
    t_search = time.time() - t_search

    df = study.trials_dataframe(attrs=("number", "value", "params", "state", "duration", "user_attrs"))
    df.to_csv(OUT / "optuna_trials.csv", index=False)
    best = study.best_trial
    bp = dict(best.params)
    json.dump({"model": "HistGradientBoostingClassifier", "objective": "mean CV macro F1, GroupKFold(5) by trip",
               "sampler": "TPESampler(seed=42, multivariate=True)", "pruner": "MedianPruner",
               "search_space": {k: list(v) for k, v in SPACE.items()},
               "fixed": {"early_stopping": False, "loss": "log_loss", "max_bins": 255, "random_state": C.SEED},
               "best_trial": best.number, "best_cv_macro_f1": best.value, "best_params": bp},
              open(C.CONFIGS_DIR / "hgb_tuned.json", "w"), indent=2)

    T = pd.read_pickle(C.FEATURES_DIR / "student_test.pkl")
    Tpts = pd.read_pickle(C.FEATURES_DIR / "points_student_test.pkl")
    arcs_by_trip = {tid: load_trip(C.STUDENT_TEST_DIR, tid, with_route=False)["arcs"] for tid in Tpts.trip_id.unique()}
    rows = []
    fitted = {}
    for name, feats, p in (("Phase 1 model (9 features, untuned)", P1, base), ("11 features, untuned", SEL, base),
                           ("11 features, tuned (Optuna best)", SEL, bp)):
        cvs = cv_eval(X, y, folds, feats, p)
        t0 = time.time()
        m = make(p).fit(X[feats], y)
        tt = time.time() - t0
        fitted[name] = (m, feats, p)
        tr_s = scores(y, m.predict_proba(X[feats])[:, 1])
        t0 = time.time()
        pt = m.predict_proba(T[feats])[:, 1]
        it = time.time() - t0
        s = scores(T["y"].to_numpy(), pt)
        top1, cont, ref_cont, _ = top1_and_continuity(T, pt, Tpts, arcs_by_trip)
        rows.append({"model": name, **cvs, "train_macro_f1": tr_s["macro_f1"], "train_accuracy": tr_s["accuracy"],
                     "test_macro_f1": s["macro_f1"], "test_accuracy": s["accuracy"], "test_precision": s["precision"],
                     "test_recall": s["recall"], "test_log_loss": s["log_loss"], "test_top1_accuracy": top1,
                     "test_route_continuity": cont, "train_time_s": tt, "inference_time_s": it})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "tuned_vs_untuned.csv", index=False)
    cand = R.iloc[1:].sort_values(["cv_macro_f1", "cv_log_loss"], ascending=[False, True])
    pick = cand.iloc[0]["model"]
    m, feats, p = fitted[pick]
    joblib.dump(m, C.MODELS_DIR / "final_hgb.joblib")
    json.dump({"model": "HistGradientBoostingClassifier", "chosen": pick, "chosen_by": "highest CV macro F1 on train trips",
               "features": feats, "params": p, "file": "final_hgb.joblib"},
              open(C.CONFIGS_DIR / "final_model.json", "w"), indent=2)

    done = df[df.state == "COMPLETE"]
    cost = {"n_trials": len(df), "completed": int(len(done)), "pruned": int((df.state == "PRUNED").sum()),
            "search_seconds": t_search, "mean_seconds_per_completed_trial": float(done["user_attrs_seconds"].mean()),
            "folds_per_trial": 5, "train_rows": int(len(X))}
    json.dump(cost, open(OUT / "search_cost.json", "w"), indent=2)
    imp = optuna.importance.get_param_importances(study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=C.SEED))
    pd.Series(imp, name="fanova_importance").to_csv(OUT / "param_importance.csv")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(done.number, done.value, "o", ms=4, label="completed trial")
    ax[0].plot(done.number, done.value.cummax(), "-", color="C3", label="best so far")
    ax[0].axhline(R.cv_macro_f1[1], ls="--", color="grey", label="untuned (11 features)")
    ax[0].set_xlabel("trial"); ax[0].set_ylabel("CV macro F1"); ax[0].set_title("Optuna TPE optimisation history"); ax[0].legend()
    s = pd.Series(imp).sort_values()
    ax[1].barh(s.index, s.values); ax[1].set_title("Hyperparameter importance (fANOVA)")
    fig.tight_layout(); fig.savefig(C.PLOTS_DIR / "s4_optuna_history_importance.png", dpi=130); plt.close(fig)

    print(R.round(4).T.to_string())
    print("final model:", pick)
    print(json.dumps(cost, indent=1), "\nbest", best.number, round(best.value, 4), bp)


if __name__ == "__main__":
    main()
