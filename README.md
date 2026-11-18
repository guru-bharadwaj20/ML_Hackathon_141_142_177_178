# PathSnap: map matching noisy GPS to road arcs

PathSnap takes a trip of noisy GPS points and a road network and says which road arc each point was on. Every GPS point gets a short list of nearby candidate arcs. A HistGradientBoosting classifier scores each (point, arc) pair, and an HMM with Viterbi decoding picks the sequence of arcs that is both likely and drivable.

Phase 1 (preprocessing, features, seven classifiers, model choice) is written up in [Phase 1 Deliverables/README.md](Phase%201%20Deliverables/README.md). This file covers the full pipeline and the evaluator demo.

## Layout

| Folder | Contents |
|---|---|
| `src/` | Data loading, preprocessing, candidates, labels, features, HMM, prediction pipeline, scoring |
| `scripts/` | One script per stage (`s1` to `s6`) |
| `api/` | FastAPI server that runs the pipeline (laptop 1) |
| `ui/` | Streamlit and Folium interface (laptop 2) |
| `configs/` | Selected features, tuned hyperparameters, final model choice, HMM settings |
| `models/` | Saved models (`final_hgb.joblib` is the one the API uses) |
| `results/` | Every table and plot quoted below |
| `maps/` | Folium maps of the student-test trips |
| `report/` | Technical report PDF, the script that builds it from `results/`, and its screenshots |
| `demo/` | One-click start scripts for both laptops (Mac `.command`, Windows `.bat`) and the demo guide |
| `presentation/` | Evaluation slides, UI screenshots and the presenter split |

Raw data goes in `data/raw/train`, `data/raw/student_test` and `data/raw/evaluator_test`. It is not in git.

## Reproducing the results

```powershell
pip install -r requirements.txt
python scripts\s1_eda.py
python scripts\s2_features.py --skip-grid
python scripts\s3_models.py
python scripts\s4_optuna.py 50
python scripts\s5_hmm.py
python scripts\s6_maps.py
```

`s2_features.py` without `--skip-grid` also reruns the candidate radius and K grid from Phase 1.

## Stage 4: new features and Bayesian tuning

Error analysis of the Phase 1 model showed most mistakes happen at low speed, and 61% of them pick the reverse direction of the true arc. At walking or crawling speed the heading between two consecutive points is mostly noise. We added two features that look at a wider window of ±10 points: `heading_diff_long` (the heading over that window compared with the arc bearing) and `disp_long` (how far the vehicle moved over that window). Neither uses the route file.

| Feature set | CV macro F1 | CV std | CV log loss |
|---|---|---|---|
| Phase 1 (9 features) | 0.9285 | 0.0086 | 0.0994 |
| + heading_diff_long | 0.9386 | 0.0093 | 0.0900 |
| + heading_diff_long + disp_long | 0.9400 | 0.0089 | 0.0869 |

The gain (+0.0115) is larger than the spread across folds, so both features stay.

We then tuned the 11-feature model with Optuna: TPE sampler (seed 42, 10 random start-up trials), median pruner, 50 trials, each scored by mean macro F1 over 5 GroupKFold folds split by trip. The untuned settings were enqueued as trial 0 so the search could only match or beat them. The search took 18.5 minutes.

| Model | CV macro F1 | CV log loss | Test macro F1 | Test Top-1 (classifier only) |
|---|---|---|---|---|
| Phase 1 model (9 features, untuned) | 0.9285 | 0.0994 | 0.9156 | 0.8630 |
| 11 features, untuned | 0.9400 | 0.0869 | 0.9239 | 0.8768 |
| 11 features, tuned | 0.9419 | 0.0822 | 0.9245 | 0.8824 |

Tuning adds a little (+0.0019 CV F1) on top of the new features. The best trial uses learning rate 0.022, 279 iterations, 120 leaves, at least 193 samples per leaf, L2 0.015 and 82% of features per split. Learning rate explains 68% of the variation between trials (fANOVA). The tuned model is the final one because it has the highest CV macro F1 on train trips. Full trial list: `results/stage4/optuna_trials.csv`.

## Stage 5: HMM and Viterbi

The classifier looks at one point at a time, so its choices can jump between roads that are not connected. The HMM fixes that. The hidden states are the candidate arcs of each point. The emission score is the log probability from the classifier. The transition score is `-|route distance - GPS distance| / beta`: moving from one arc to the next should cost about as much road distance as the GPS points moved in a straight line. Route distance comes from a bounded Dijkstra on the trip's road graph. The chain restarts after a gap of more than 120 s or when no candidate can reach any other.

We picked beta from 12 values between 0.5 and 100 m using out-of-fold classifier probabilities on the train trips. Beta = 5 m gave the best train Top-1 (0.922).

| Student test (9 trips) | Top-1 | Continuity | Route arc precision | Route arc recall |
|---|---|---|---|---|
| Classifier only | 0.8824 | 0.9351 | 0.8902 | 0.9800 |
| Classifier + HMM (beta 5) | 0.9125 | 0.9801 | 0.9555 | 0.9906 |
| Reference route | 1.0000 | 0.9832 | 1.0000 | 1.0000 |

Continuity is the share of consecutive point pairs (at most 10 s apart) that sit on the same arc or on two connected arcs. The HMM lifts it to 0.980, close to the 0.983 of the true route itself. All 9 trips improve. Trip 77 is still the hardest (0.799) because it is slow throughout, with a median speed of 3.7 m/s. Decoding all 9 trips takes 0.08 s.

## Leakage rules

- The prediction pipeline (`src/pipeline.py`) loads trips with `with_route=False`, so the model never sees the route file.
- Accuracy is computed in `src/scoring.py` after prediction finishes, on the API server only. The API response carries metrics but no reference geometry.
- Features, hyperparameters, the model and beta were all chosen on train trips (and checked on student test). The evaluator trips were not used for any choice.

## Evaluator demo on two laptops

Shortcut: double-click `demo/start_api.command` (Mac) or `demo/start_api.bat` (Windows) on laptop 1, then `demo/start_ui.command` or `demo/start_ui.bat` on laptop 2 and type laptop 1's IP. The full guide, with fixes and likely questions, is in [demo/DEMO.md](demo/DEMO.md).

Both laptops need this repository and `pip install -r requirements.txt`, and both must be on the same network.

Laptop 1 runs the API. Put the evaluator trips in `data/raw/evaluator_test` (one folder per trip with `.track`, `.nodes`, `.arcs` and `.route` files), then:

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

On a Mac use `python3` and find the address with `ipconfig getifaddr en0`.

The console prints one log line per request: points, candidate rows, time per stage, chain breaks and the Top-1 accuracy of the HMM and of the classifier alone. The same lines go to `logs/api.log`. Find this laptop's address with `ipconfig` (the IPv4 address), and allow Python through the Windows firewall if asked.

Laptop 2 runs the interface:

```powershell
$env:PATHSNAP_API_URL = "http://<laptop-1-ip>:8000"
python -m streamlit run ui\app.py
```

On a Mac: `export PATHSNAP_API_URL=http://<laptop-1-ip>:8000` then `python3 -m streamlit run ui/app.py`.

Before the demo, open `http://<laptop-1-ip>:8000/health` in laptop 2's browser. If it doesn't load, the two laptops can't see each other: check the network (a phone hotspot is the safest) and the firewall.

Pick one of the 10 trips and press "Run map matching". The page shows the accuracy tiles and a map with the nearby road network, the GPS points, the predicted route and (as a layer you can switch on) the classifier-only route.

To use another folder of trips, set `PATHSNAP_EVAL_DIR` on laptop 1 before starting the API. A trip shows up in the list only if it has `.track`, `.nodes` and `.arcs` files.

API endpoints: `GET /health`, `GET /trips`, `POST /predict` with body `{"trip_id": "00000015"}`.
