"""PathSnap API (laptop 1): uvicorn api.main:app --host 0.0.0.0 --port 8000"""
import sys, time, logging, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src import config as C
from src.data_io import list_trips
from src.pipeline import Matcher
from src.scoring import score

C.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(C.LOGS_DIR / "api.log")])
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap map-matching API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
matcher = Matcher()
log.info("model loaded: %s | features=%d | HMM beta=%.1f m | trips folder=%s",
         matcher.model_name, len(matcher.features), matcher.beta, C.EVAL_DIR)


class PredictRequest(BaseModel):
    trip_id: str


def eval_trips():
    need = ("track", "nodes", "arcs")
    return [t for t in list_trips(C.EVAL_DIR) if all((C.EVAL_DIR / t / f"{t}.{e}").exists() for e in need)]


def lines(nodes, arcs, ids, merge=True):
    out, cur, last = [], [], None
    for a in ids:
        if a < 0 or (merge and a == last):
            continue
        u, v = arcs[a]
        p, q = [round(nodes[u, 1], 6), round(nodes[u, 0], 6)], [round(nodes[v, 1], 6), round(nodes[v, 0], 6)]
        if merge and cur and cur[-1] == p:
            cur.append(q)
        else:
            if cur:
                out.append(cur)
            cur = [p, q]
        last = a
    if cur:
        out.append(cur)
    return out


@app.get("/health")
def health():
    return {"status": "ok", "model": matcher.model_name, "hmm_beta_m": matcher.beta}


@app.get("/trips")
def trips():
    ids = eval_trips()
    log.info("GET /trips -> %d trips", len(ids))
    return {"trips": ids}


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trip_id.strip()
    if not re.fullmatch(r"\d{8}", tid) or tid not in eval_trips():
        log.warning("POST /predict rejected unknown trip_id=%r", tid)
        raise HTTPException(status_code=404, detail="unknown trip id")
    log.info("POST /predict trip=%s | started", tid)
    t0 = time.perf_counter()
    try:
        r = matcher.match(C.EVAL_DIR, tid)
    except Exception as e:
        log.exception("trip=%s | prediction failed", tid)
        raise HTTPException(status_code=422, detail=f"prediction failed: {e}")
    tm = r["timings"]
    track, nodes, arcs = r["prep"]["track"], r["trip"]["nodes"], r["trip"]["arcs"]
    n, ncand = len(track), int((r["prep"]["feats"].groupby("pt").size() > 0).sum())
    log.info("trip=%s | %d GPS points, %d with candidates, %d candidate rows | load %.2fs, features %.2fs, "
             "classifier %.2fs, HMM %.2fs | chain breaks=%d", tid, n, ncand, len(r["prep"]["feats"]),
             tm["load_s"], tm["candidates_features_s"], tm["classifier_s"], tm["hmm_viterbi_s"], r["breaks"])
    t1 = time.perf_counter()
    s = score(C.EVAL_DIR, r)
    tm["scoring_s"] = time.perf_counter() - t1
    tm["total_s"] = time.perf_counter() - t0
    if s:
        log.info("trip=%s | Top-1 accuracy: HMM %.4f, classifier-only %.4f | continuity %.4f | total %.2fs",
                 tid, s["hmm"]["top1_accuracy"], s["classifier_only"]["top1_accuracy"],
                 s["hmm"]["route_continuity"], tm["total_s"])
    step = max(1, n // 3000)
    near = r["candidate_arcs"]
    return {"trip_id": tid, "model": matcher.model_name, "hmm_beta_m": matcher.beta,
            "n_points": n, "points_with_candidates": ncand, "matched_points": int((r["pred_hmm"] >= 0).sum()),
            "chain_breaks": r["breaks"], "metrics": s, "timings_s": {k: round(v, 3) for k, v in tm.items()},
            "gps": [[round(track[i, 1], 6), round(track[i, 0], 6)] for i in range(0, n, step)],
            "predicted_route": lines(nodes, arcs, r["pred_hmm"]),
            "classifier_route": lines(nodes, arcs, r["pred_clf"]),
            "road_network": lines(nodes, arcs, near[:20000], merge=False),
            "predicted_arcs": [int(a) for a in r["pred_hmm"]]}
