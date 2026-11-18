"""Stage 6: Folium maps of student-test trips (GPS, network, reference, classifier-only, HMM).

python scripts/s6_maps.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from src import config as C
from src.data_io import list_trips, load_route
from src.pipeline import Matcher
from src.scoring import score
from src.maps import trip_map

OUT = C.RESULTS_DIR / "stage6"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    mt = Matcher()
    rows = []
    for tid in list_trips(C.STUDENT_TEST_DIR):
        if not (C.STUDENT_TEST_DIR / tid / f"{tid}.arcs").exists():
            continue
        r = mt.match(C.STUDENT_TEST_DIR, tid)
        s = score(C.STUDENT_TEST_DIR, r)
        rows.append({"trip_id": tid, "points": s["points"], "top1_classifier": s["classifier_only"]["top1_accuracy"],
                     "top1_hmm": s["hmm"]["top1_accuracy"], "continuity_hmm": s["hmm"]["route_continuity"],
                     **{k: round(v, 3) for k, v in r["timings"].items()}})
        title = (f"Student-test trip {tid}: Top-1 classifier {rows[-1]['top1_classifier']:.3f}, "
                 f"classifier + HMM {rows[-1]['top1_hmm']:.3f}")
        m = trip_map(r["prep"]["track"], r["trip"]["nodes"], r["trip"]["arcs"], network_arc_ids=r["candidate_arcs"],
                     reference_arc_ids=load_route(C.STUDENT_TEST_DIR, tid),
                     predicted={"classifier only": r["pred_clf"], "classifier + HMM": r["pred_hmm"]}, title=title)
        m.save(str(C.MAPS_DIR / f"student_test_{tid}.html"))
        print(rows[-1], flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "pipeline_student_test.csv", index=False)
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
