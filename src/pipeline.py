"""Prediction pipeline used by the API: raw trip -> candidates -> features -> classifier -> HMM/Viterbi."""
import json, time
import numpy as np
import joblib

from . import config as C
from .data_io import load_trip
from .preprocess import prepare_trip
from . import hmm


class Matcher:
    def __init__(self):
        fm = json.load(open(C.CONFIGS_DIR / "final_model.json"))
        self.features = fm["features"]
        self.model = joblib.load(C.MODELS_DIR / fm["file"])
        self.model_name = fm["chosen"]
        self.beta = float(json.load(open(C.CONFIGS_DIR / "hmm.json"))["beta_m"])

    def match(self, part_dir, trip_id):
        tm = {}
        t0 = time.perf_counter()
        trip = load_trip(part_dir, trip_id, with_route=False)
        tm["load_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        prep = prepare_trip(trip)
        f = prep["feats"]
        tm["candidates_features_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        p = self.model.predict_proba(f[self.features])[:, 1]
        tm["classifier_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        n = len(prep["xy"])
        pt, arc = f["pt"].to_numpy(), f["arc"].to_numpy()
        pred_clf = np.full(n, -1, dtype=np.int64)
        order = np.lexsort((-p, pt))
        first = order[np.r_[True, np.diff(pt[order]) != 0]]
        pred_clf[pt[first]] = arc[first]
        steps = hmm.prepare_steps(f, prep["xy"], prep["track"][:, 2], trip["arcs"], prep["index"].length)
        pts, pred, breaks = hmm.viterbi(steps, np.log(np.clip(p, 1e-6, 1)), self.beta)
        pred_hmm = np.full(n, -1, dtype=np.int64)
        pred_hmm[pts] = pred
        tm["hmm_viterbi_s"] = time.perf_counter() - t0

        return {"trip_id": trip_id, "trip": trip, "prep": prep, "candidate_arcs": np.unique(arc),
                "pred_clf": pred_clf, "pred_hmm": pred_hmm, "breaks": breaks, "timings": tm}
