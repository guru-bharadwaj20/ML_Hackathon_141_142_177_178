#!/bin/zsh
# Laptop 2 (Mac): double-click, then type laptop 1's IP.
cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate
read "IP?Laptop 1 IP: "
export PATHSNAP_API_URL="http://$IP:8000"
curl -s "$PATHSNAP_API_URL/health" || echo "Cannot reach the API at $PATHSNAP_API_URL"
python3 -m streamlit run ui/app.py
