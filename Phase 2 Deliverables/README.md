# Phase 2 deliverables

Phase 2 has two deliverables: this public repository and one technical report PDF.

## Deliverable 2: technical report

[PathSnap_Technical_Report.pdf](PathSnap_Technical_Report.pdf) follows the required structure. [report/make_report.py](../report/make_report.py) builds it, reading every number from `results/` and `configs/`.

| Report section | Pages |
|---|---|
| 2. Dataset understanding and EDA | 3 to 8 |
| 3. Model experimentation and comparison (seven models, compulsory questions for each) | 9 to 15 |
| 4. Bayesian hyperparameter optimisation | 16 to 17 |
| 5. HMM and Viterbi route reconstruction | 18 to 20 |
| 6. Geographical visualisation with Folium | 21 to 22 |
| 7. API, UI and deployment (UI and API console screenshots) | 23 to 24 |

## Deliverable 1: repository contents

| Required component | Where it is |
|---|---|
| Data processing | [src/data_io.py](../src/data_io.py), [src/preprocess.py](../src/preprocess.py), [src/geo.py](../src/geo.py), [scripts/s1_eda.py](../scripts/s1_eda.py) |
| Feature engineering | [src/candidates.py](../src/candidates.py), [src/labels.py](../src/labels.py), [src/features.py](../src/features.py), [scripts/s2_features.py](../scripts/s2_features.py) |
| Model training (all seven models) | [scripts/s3_models.py](../scripts/s3_models.py) |
| Model evaluation (metrics, plots, confusion matrices) | [scripts/s3_models.py](../scripts/s3_models.py), [src/scoring.py](../src/scoring.py), [results/stage3/](../results/stage3/), [results/plots/](../results/plots/) |
| Hyperparameter optimisation (Optuna TPE) | [scripts/s4_optuna.py](../scripts/s4_optuna.py), [results/stage4/optuna_trials.csv](../results/stage4/optuna_trials.csv), [configs/hgb_tuned.json](../configs/hgb_tuned.json) |
| HMM and Viterbi | [src/hmm.py](../src/hmm.py), [scripts/s5_hmm.py](../scripts/s5_hmm.py), [results/stage5/](../results/stage5/) |
| API (endpoints and model loading) | [api/main.py](../api/main.py), [src/pipeline.py](../src/pipeline.py) |
| UI (evaluator interface) | [ui/app.py](../ui/app.py) |
| Deployment | [requirements.txt](../requirements.txt), [configs/](../configs/), the two-laptop steps in the [main README](../README.md#evaluator-demo-on-two-laptops) |
| Visualisations | [src/maps.py](../src/maps.py), [scripts/s6_maps.py](../scripts/s6_maps.py), interactive maps in [maps/](../maps/), report screenshots from [report/shots.py](../report/shots.py) |
| Supporting results | [results/](../results/) (CSV and JSON), [logs/](../logs/) (run logs), [configs/](../configs/) |
| Saved models | [models/](../models/). The API loads `final_hgb.joblib`. KNN and Random Forest are left out for size and are rebuilt by `s3_models.py` |

## Final results

| Student test (9 trips) | Top-1 | Route continuity |
|---|---|---|
| Phase 1 model | 0.8630 | 0.9241 |
| Tuned 11-feature model | 0.8824 | 0.9351 |
| Tuned model + HMM (beta 5 m) | 0.9125 | 0.9801 |

On the ten evaluator trips, run through the deployed API after every choice was frozen, mean Top-1 is 0.958 with the HMM and 0.935 for the classifier alone ([results/stage6/evaluator_api_check.csv](../results/stage6/evaluator_api_check.csv)).
