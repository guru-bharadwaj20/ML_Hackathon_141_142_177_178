# PathSnap: map matching noisy GPS to road arcs :

UE24CS352A Machine Learning hackathon, PES University. Team: Debargho Banerji (PES1UG24CS141), Deekshith Gowda G (PES1UG24CS142), Guru R Bharadwaj (PES1UG24CS177), Gyaan S Shetty (PES1UG24CS178).

PathSnap takes a trip of noisy GPS points and a road network and says which road arc each point was on. Every GPS point gets up to eight nearby candidate arcs. A HistGradientBoosting classifier scores each (point, arc) pair, and an HMM with Viterbi decoding picks the sequence of arcs that is both likely and drivable.

## Deliverables :

| Phase | What | Link |
|---|---|---|
| Phase 1 | Preprocessing, feature selection, seven models and their accuracy, model choice | [Phase 1 Deliverables/README.md](Phase%201%20Deliverables/README.md) |
| Phase 2 | Technical report (PDF) | [Phase 2 Deliverables/PathSnap_Technical_Report.pdf](Phase%202%20Deliverables/PathSnap_Technical_Report.pdf) |
| Phase 2 | Where each required component lives in this repository | [Phase 2 Deliverables/README.md](Phase%202%20Deliverables/README.md) |

## Layout :

| Folder | Contents |
|---|---|
| [src/](src/) | Data loading, preprocessing, candidates, labels, features, HMM, prediction pipeline, scoring, maps |
| [scripts/](scripts/) | One script per stage, `s1` to `s6` |
| [api/](api/) | FastAPI server that runs the pipeline (laptop 1) |
| [ui/](ui/) | Streamlit and Folium interface (laptop 2) |
| [configs/](configs/) | Selected features, tuned hyperparameters, final model choice, HMM settings |
| [models/](models/) | Saved models; the API loads `final_hgb.joblib` |
| [results/](results/) | Every table and plot quoted in the report |
| [logs/](logs/) | Logs from the tuning, HMM, map and API runs |
| [maps/](maps/) | Interactive Folium maps of the student-test trips and one train trip |
| [report/](report/) | Script that builds the report PDF, and its screenshots |

## Installation :

```powershell
pip install -r requirements.txt
```

The raw data is not in git. Put it in `data/raw/train`, `data/raw/student_test` and `data/raw/evaluator_test`, one folder per trip with `.track`, `.nodes`, `.arcs` and `.route` files.

## Reproducing the results :

```powershell
python scripts\s1_eda.py
python scripts\s2_features.py --skip-grid
python scripts\s3_models.py
python scripts\s4_optuna.py 50
python scripts\s5_hmm.py
python scripts\s6_maps.py
python report\hmm_flips.py
python report\make_report.py
```

`s2_features.py` without `--skip-grid` also reruns the candidate radius and K grid. `make_report.py` writes the PDF into `Phase 2 Deliverables/`.

## Evaluator demo on two laptops

Both laptops need this repository and `pip install -r requirements.txt`, and both must be on the same network (a phone hotspot is the safest).

Laptop 1 runs the API and holds the evaluator trips in `data/raw/evaluator_test`:

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The console prints one log line per request: points, candidate rows, time per stage, chain breaks and the Top-1 accuracy of the HMM and of the classifier alone. The same lines go to `logs/api.log`. Find the laptop's address with `ipconfig` (Windows) or `ipconfig getifaddr en0` (Mac), and allow Python through the firewall if asked.

Laptop 2 runs the interface:

```powershell
$env:PATHSNAP_API_URL = "http://<laptop-1-ip>:8000"
python -m streamlit run ui\app.py
```

On a Mac: `export PATHSNAP_API_URL=http://<laptop-1-ip>:8000` then `python3 -m streamlit run ui/app.py`.

Before the demo, open `http://<laptop-1-ip>:8000/health` in laptop 2's browser. If it doesn't load, the laptops can't reach each other, so check the network and the firewall.

Pick one of the 10 trips and press "Run map matching". The page shows the accuracy tiles and a map with the nearby road network, the GPS points, the predicted route and, as a layer you can switch on, the classifier-only route.

API endpoints: `GET /health`, `GET /trips`, `POST /predict` with body `{"trip_id": "00000015"}`. To serve another folder of trips, set `PATHSNAP_EVAL_DIR` on laptop 1 before starting the API.

Observation  :
"As seen in the reports , trip 30 showed the HMM improving the accuracy by 6% . Trip 87 showed very similar accuracy even after HMM ."

## Leakage rules :

- The prediction pipeline ([src/pipeline.py](src/pipeline.py)) loads trips with `with_route=False`, so the model never sees the route file.
- Accuracy is computed in [src/scoring.py](src/scoring.py) after prediction, on the API server only. The API response carries metrics but no reference geometry.
- Features, hyperparameters, the model and beta were chosen on train trips. The evaluator trips were not used for any choice.
