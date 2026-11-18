"""Per-trip count of student-test points the HMM fixed or broke compared with the classifier alone."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from src import config as C
from src.data_io import list_trips, load_route
from src.labels import align_route
from src.pipeline import Matcher
mt = Matcher(); rows = []
for tid in list_trips(C.STUDENT_TEST_DIR):
    if not (C.STUDENT_TEST_DIR / tid / f"{tid}.arcs").exists(): continue
    r = mt.match(C.STUDENT_TEST_DIR, tid)
    route = load_route(C.STUDENT_TEST_DIR, tid); idx = r["prep"]["index"]
    k, _ = align_route(r["prep"]["xy"], idx.A[route], idx.B[route]); ref = route[k]
    c, h = r["pred_clf"] == ref, r["pred_hmm"] == ref
    f = r["prep"]["feats"]; incand = f[f.arc.values == ref[f.pt.values]].pt.nunique()
    rows.append(dict(trip=tid, n=len(ref), fixed=int((~c & h).sum()), broken=int((c & ~h).sum()),
                     both_wrong=int((~c & ~h).sum()), ref_not_in_cand=len(ref) - incand, breaks=r["breaks"]))
D = pd.DataFrame(rows)
D.to_csv(C.RESULTS_DIR / "stage5" / "hmm_fixed_broken_student_test.csv", index=False)
print(D.to_string(index=False))
