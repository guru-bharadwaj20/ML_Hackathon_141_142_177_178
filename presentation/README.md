# Presentation

`PathSnap_Presentation.pptx` is the 19-slide deck used in the evaluation. The `screenshots/` folder holds the UI captures it uses, taken from the running API and Streamlit UI on evaluator trips 00000087 and 00000030.

## Slide order and who presents

| # | Slide | Presenter |
|---|---|---|
| 1–4 | Title, problem, two-laptop architecture, six-step pipeline | Gyaan |
| 5–7 | EDA, candidates and labels (K = 8 within 50 m), features | Gyaan |
| 8–9 | Seven models compared, why HistGradientBoosting | Guru |
| 10–13 | Optuna tuning, HMM and Viterbi, HMM results, per-trip chart | Debargo |
| 14–19 | UI home, trip 87 result, trip 30 classifier vs HMM, API console, integrity, summary | Deekshit |

After slide 19 the team switches to the live demo (see `demo/DEMO.md`).

## Numbers used in the deck

| | Source file |
|---|---|
| Seven-model table | `results/stage3/model_comparison.csv` |
| K grid | `results/stage2/candidate_radius_k_grid.csv` |
| Tuned vs untuned | `results/stage4/tuned_vs_untuned.csv` |
| Classifier vs HMM | `results/stage5/classifier_vs_hmm.csv` |
| Per-trip chart | `results/stage5/per_trip_student_test.csv` |
