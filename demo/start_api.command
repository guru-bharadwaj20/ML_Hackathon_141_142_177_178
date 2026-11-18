#!/bin/zsh
# Laptop 1 (Mac): double-click to start the PathSnap API.
cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate
export PATHSNAP_EVAL_DIR="${PATHSNAP_EVAL_DIR:-$PWD/data/raw/evaluator_test}"
IP=$(ipconfig getifaddr en0)
echo "Trips found: $(ls "$PATHSNAP_EVAL_DIR" | wc -l | tr -d ' ')"
cat "$PATHSNAP_EVAL_DIR"/*/* > /dev/null 2>&1
echo "Laptop 2: http://$IP:8000/health   then PATHSNAP_API_URL=http://$IP:8000"
caffeinate -dis &
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
