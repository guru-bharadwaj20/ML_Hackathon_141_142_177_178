"""Central configuration: paths and fixed constants used by every stage."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
TRAIN_DIR = DATA_RAW / "train"
STUDENT_TEST_DIR = DATA_RAW / "student_test"
# Evaluator trips are supplied on the day; the API reads them from this folder.
EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", str(DATA_RAW / "evaluator_test")))

MODELS_DIR = ROOT / "models"
CONFIGS_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MAPS_DIR = ROOT / "maps"
LOGS_DIR = ROOT / "logs"
FEATURES_DIR = ROOT / "data" / "features"

SEED = 42

# Candidate generation
CAND_RADIUS_M = 50.0      # search radius around each GPS point
CAND_K = 8                # keep at most K nearest road segments
FALLBACK_RADIUS_M = 200.0  # if nothing within CAND_RADIUS_M, take the nearest arc within this

# Training-set size control: at most this many GPS points per trip (uniform stride)
MAX_POINTS_PER_TRIP = 1000

for _d in (MODELS_DIR, CONFIGS_DIR, RESULTS_DIR, PLOTS_DIR, MAPS_DIR, LOGS_DIR, FEATURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
