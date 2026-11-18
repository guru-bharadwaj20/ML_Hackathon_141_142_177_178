# Running the evaluator demo

Two laptops on the same network (a phone hotspot is the safest; college Wi-Fi often blocks laptop-to-laptop traffic). Start the API first.

## One-time setup (both laptops)

Mac:
```bash
git clone https://github.com/guru-bharadwaj20/ML_Hackathon_141_142_177_178
cd ML_Hackathon_141_142_177_178
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
git clone https://github.com/guru-bharadwaj20/ML_Hackathon_141_142_177_178
cd ML_Hackathon_141_142_177_178
pip install -r requirements.txt
```

On a Mac, keep the repo out of an iCloud-synced Desktop. Files there are fetched on first read, and the first request can take 20+ seconds.

## Laptop 1: API

The evaluator trips go in `data/raw/evaluator_test/<trip_id>/` (`.track .nodes .arcs .route`).

- Mac: double-click `demo/start_api.command` (first time: right-click, Open).
- Windows: double-click `demo/start_api.bat`, and allow Python through the firewall (Private and Public).

The window prints this laptop's IP and then `model loaded: 11 features, tuned (Optuna best) | ... HMM beta=5.0 m`. Keep it open: the evaluator reads the request logs here (also saved in `logs/api.log`).

## Laptop 2: UI

- Mac: double-click `demo/start_ui.command` and type laptop 1's IP.
- Windows: double-click `demo/start_ui.bat` and type laptop 1's IP.

The browser opens at `http://localhost:8501` with the 10 trips. Pick one and press **Run map matching**.

## During the demo

1. The evaluator picks a trip on laptop 2 and presses Run.
2. Show laptop 1's console: `started`, the timings line, and the Top-1 line for HMM and classifier only.
3. On laptop 2, read the tiles, then the map: blue = GPS, red = predicted route, grey = nearby roads. Switch on the orange classifier-only layer to show what the HMM fixed (trip 00000030 shows it best).

## If something goes wrong

| Problem | Fix |
|---|---|
| `/health` doesn't load from laptop 2 | wrong IP, firewall, or network blocking: move both to a hotspot and use the new IP |
| trip list is empty | trip folders must sit directly inside `data/raw/evaluator_test/` |
| `Address already in use` | Mac: `lsof -ti:8000 \| xargs kill`; Windows: close the old window, or use port 8001 on both sides |
| map is grey | `git pull` (OpenStreetMap tiles are the default now) |

## Questions to expect

- **Where is the ground truth?** Only on laptop 1 in the `.route` files. `src/pipeline.py` loads trips with `with_route=False`; `src/scoring.py` reads the route after prediction. The response never contains the reference route.
- **Hidden states?** The candidate roads of each GPS point.
- **Emission?** log P(true road) from the tuned HistGradientBoosting classifier.
- **Transition?** `-|driving distance - GPS distance| / beta`, beta = 5 m, driving distance by Dijkstra on the trip's road graph.
- **Gaps?** The chain restarts after 120 s, or when no candidate can reach the next.
- **Did the HMM help?** Student test Top-1 88.2% → 91.3%, continuity 93.5% → 98.0%; every trip improved.
- **Why HistGradientBoosting?** Tied with Random Forest on accuracy, but best CV macro F1 (0.9289) and log loss (0.107), less overfitting (0.012 vs 0.036 gap), 0.56 MB vs 178 MB, about 3× faster.
- **Tuned hyperparameters?** learning_rate, max_iter, max_leaf_nodes, min_samples_leaf, l2_regularization, max_features; Optuna TPE, 50 trials, objective = mean 5-fold group CV macro F1.
