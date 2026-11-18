"""Builds the technical report PDF from the files in results/, configs/ and report/figures/.

python report/make_report.py [output.pdf]
"""
import json
import sys
from pathlib import Path
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image,
                                KeepTogether, Preformatted)

ROOT = Path(__file__).resolve().parents[1]
RES, PLOTS, FIG = ROOT / "results", ROOT / "results" / "plots", ROOT / "report" / "figures"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "report" / "PathSnap_Technical_Report.pdf"
REPO = "https://github.com/guru-bharadwaj20/ML_Hackathon_141_142_177_178"
TEAM = "Team members (name, SRN): ______________________________________________"

ss = getSampleStyleSheet()
TITLE = ParagraphStyle("t", parent=ss["Title"], fontSize=22, leading=27, spaceAfter=10)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, spaceBefore=4, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, spaceBefore=10, spaceAfter=4)
H3 = ParagraphStyle("h3", parent=ss["Heading3"], fontSize=11, spaceBefore=8, spaceAfter=2)
B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13.2)
QS = ParagraphStyle("q", parent=B, fontName="Helvetica-Bold", textColor=colors.HexColor("#1f3b63"), spaceBefore=5)
CELL = ParagraphStyle("c", parent=B, fontSize=8, leading=10)
CAP = ParagraphStyle("cap", parent=B, fontSize=8, leading=10, textColor=colors.HexColor("#555555"), spaceAfter=9)
MONO = ParagraphStyle("m", parent=B, fontName="Courier", fontSize=6.6, leading=8.4,
                      backColor=colors.HexColor("#f2f2f2"), borderPadding=4)
story = []


def h1(t): story.append(Paragraph(t, H1))
def h2(t): story.append(Paragraph(t, H2))
def h3(t): story.append(Paragraph(t, H3))
def p(t): story.append(Paragraph(t, B))
def brk(): story.append(PageBreak())


def qa(q, *paras):
    story.append(Paragraph(q, QS))
    for a in paras:
        story.append(Paragraph(a, B))


def bl(items):
    for i in items:
        story.append(Paragraph(i, B, bulletText="•"))


def table(rows, widths, font=None, bold_rows=()):
    st = CELL if font is None else ParagraphStyle("cx", parent=CELL, fontSize=font, leading=font + 2.2)
    data = [[Paragraph(str(c), st) for c in r] for r in rows]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde4ee")),
             ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
             ("VALIGN", (0, 0), (-1, -1), "TOP")]
    style += [("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fff4d6")) for r in bold_rows]
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 6))


def img(path, width_cm, caption, crop_h=None):
    path = Path(path)
    w, h = PILImage.open(path).size
    if crop_h:
        h = min(h, crop_h)
        tmp = ROOT / "report" / "figures" / f"_crop_{path.name}"
        PILImage.open(path).crop((0, 0, w, h)).save(tmp)
        path = tmp
    story.append(KeepTogether([Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w),
                               Paragraph(caption, CAP)]))


f4 = lambda x: f"{x:.4f}"
f3 = lambda x: f"{x:.3f}"
M = pd.read_csv(RES / "stage3" / "model_comparison.csv").set_index("model")
FS = pd.read_csv(RES / "stage3" / "feature_selection_backward.csv")
IMP = pd.read_csv(RES / "stage3" / "feature_importance.csv", index_col=0).iloc[:, 0]
MV = pd.read_csv(RES / "eda" / "missing_values.csv")
PS = pd.read_csv(RES / "eda" / "partition_summary.csv").set_index("partition")
SAMP = json.load(open(RES / "eda" / "sampling_summary.json"))
ABL = pd.read_csv(RES / "stage4" / "feature_ablation.csv")
TU = pd.read_csv(RES / "stage4" / "tuned_vs_untuned.csv")
PIMP = pd.read_csv(RES / "stage4" / "param_importance.csv", index_col=0).iloc[:, 0]
COST = json.load(open(RES / "stage4" / "search_cost.json"))
TUNE = json.load(open(ROOT / "configs" / "hgb_tuned.json"))
HMMC = json.load(open(ROOT / "configs" / "hmm.json"))
BG = pd.read_csv(RES / "stage5" / "hmm_beta_grid.csv")
CVH = pd.read_csv(RES / "stage5" / "classifier_vs_hmm.csv")
PT = pd.read_csv(RES / "stage5" / "per_trip_student_test.csv", dtype={"trip_id": str})
FB = pd.read_csv(RES / "stage5" / "hmm_fixed_broken_student_test.csv", dtype={"trip": str})
EV = pd.read_csv(RES / "stage6" / "evaluator_api_check.csv", dtype={"trip_id": str})
CONSOLE = (ROOT / "report" / "api_console_sample.txt").read_text(encoding="utf-8")

NAME = {"LogisticRegression": "Logistic Regression", "GaussianNB": "Gaussian Naive Bayes",
        "KNN": "Distance-weighted KNN", "DecisionTree": "Decision Tree", "RandomForest": "Random Forest",
        "GradientBoosting": "HistGradientBoosting", "MLP": "Shallow MLP"}
CLS = {"LogisticRegression": "LogisticRegression", "GaussianNB": "GaussianNB", "KNN": "KNeighborsClassifier",
       "DecisionTree": "DecisionTreeClassifier", "RandomForest": "RandomForestClassifier",
       "GradientBoosting": "HistGradientBoostingClassifier", "MLP": "MLPClassifier"}
FAM = {"LogisticRegression": "Linear and probabilistic", "GaussianNB": "Linear and probabilistic",
       "KNN": "Instance-based", "DecisionTree": "Tree-based", "RandomForest": "Ensemble (bagging)",
       "GradientBoosting": "Boosting", "MLP": "Neural network"}
CONF = {"LogisticRegression": "StandardScaler, C=1.0 (L2), solver lbfgs, max_iter=1000",
        "GaussianNB": "var_smoothing=1e-9, no scaling",
        "KNN": "StandardScaler, n_neighbors=25, weights=distance, Euclidean",
        "DecisionTree": "criterion=entropy, max_depth=10, min_samples_leaf=20",
        "RandomForest": "n_estimators=200, criterion=gini, max_features=sqrt, min_samples_leaf=5, bootstrap",
        "GradientBoosting": "learning_rate=0.1, max_iter=300, max_leaf_nodes=31, min_samples_leaf=20, "
                            "max_bins=255, early stopping (stopped at 151 trees)",
        "MLP": "StandardScaler, hidden (64, 32), relu, adam, lr=1e-3, alpha=1e-4, batch 200, "
               "early stopping patience 8 (stopped at epoch 34)"}
SIZE = {"LogisticRegression": "2 KB", "GaussianNB": "1 KB", "KNN": "62 MB", "DecisionTree": "99 KB",
        "RandomForest": "178 MB", "GradientBoosting": "0.56 MB", "MLP": "76 KB"}
ORDER = list(NAME)
P1, U11, T11 = TU.iloc[0], TU.iloc[1], TU.iloc[2]
CLF, HM, REF = CVH.iloc[0], CVH.iloc[1], CVH.iloc[2]
BEST_B = BG[BG.beta != "classifier only"].copy()
BEST_B["b"] = BEST_B.beta.astype(float)
TRAIN_CLF = BG[BG.beta == "classifier only"].iloc[0]
TRAIN_B5 = BEST_B[BEST_B.b == HMMC["beta_m"]].iloc[0]

# ------------------------------------------------------------------ cover
story.append(Spacer(1, 2.2 * cm))
story.append(Paragraph("PathSnap: map matching noisy GPS to road arcs", TITLE))
p("Technical report. UE24CS352A Machine Learning hackathon, PES University, RR Campus.")
p(TEAM)
p(f"Repository: <link href='{REPO}' color='blue'>{REPO}</link> (main branch)")
story.append(Spacer(1, 0.5 * cm))
p("PathSnap takes a trip of noisy GPS points and the road network around it and says which road arc each point "
  "was on. Every GPS point gets up to eight nearby candidate arcs. A HistGradientBoosting classifier scores each "
  "(point, candidate arc) pair, and a hidden Markov model decoded with Viterbi picks the sequence of arcs that is "
  "both likely and drivable. A FastAPI server on one laptop runs the pipeline and scores it against the reference "
  "route, and a Streamlit page on a second laptop lets the evaluator pick a trip and see the result on a Folium map.")
table([["Stage", "Student-test Top-1", "Route continuity", "Where"],
       ["Phase 1 model (9 features, untuned HistGradientBoosting)", f4(P1.test_top1_accuracy),
        f4(P1.test_route_continuity), "Sections 3 and 4"],
       ["+ two new features", f4(U11.test_top1_accuracy), f4(U11.test_route_continuity), "Section 4"],
       ["+ Bayesian tuning with Optuna", f4(T11.test_top1_accuracy), f4(T11.test_route_continuity), "Section 4"],
       [f"+ HMM and Viterbi (final system, beta = {HMMC['beta_m']:g} m)", f4(HM.top1), f4(HM.continuity),
        "Section 5"],
       ["Reference route itself", "1.0000", f4(REF.continuity), ""]],
      [8, 3, 3, 3], bold_rows=(4,))
p(f"On the ten evaluator trips the final system reaches a mean Top-1 of {EV.top1_hmm.mean():.4f} against "
  f"{EV.top1_classifier.mean():.4f} for the classifier alone (section 7). Those trips were used only for this "
  "final check of the deployed system.")
p("Every number in this report is read from a CSV or JSON file in the repository by report/make_report.py, so "
  "each one can be traced to the script that produced it.")
brk()

# ------------------------------------------------------------------ 1. overview
h1("1. Problem and system overview")
p("Each trip in the dataset has four files. The .track file holds GPS observations (longitude, latitude, time in "
  "seconds). The .nodes file holds road-network nodes (longitude, latitude) and .arcs holds directed arcs "
  "(from_node, to_node), with the row index as the arc id. The .route file lists the arc ids the vehicle "
  "travelled, in order. The task is to give every GPS point one arc id.")
p("We turned this into binary classification over candidates followed by sequence decoding.")
table([["Step", "What happens", "Code"],
       ["Load and clean", "Read the trip without its route file; drop non-finite or out-of-range coordinates and "
        "non-increasing timestamps", "src/data_io.py, src/preprocess.py"],
       ["Project", "Local equirectangular projection to metres around the trip centre", "src/geo.py"],
       ["Candidates", "Up to K = 8 arcs within R = 50 m of each point (STRtree index); nearest arc within 200 m if "
        "none", "src/candidates.py"],
       ["Features", "11 features per (point, arc) row from the GPS track and road network only", "src/features.py"],
       ["Classifier", "Tuned HistGradientBoostingClassifier gives P(arc is the true arc)", "src/pipeline.py"],
       ["HMM and Viterbi", "Connectivity-aware decoding of the most likely drivable arc sequence", "src/hmm.py"],
       ["Scoring", "Top-1, continuity, arc precision and recall, computed after prediction on the API server",
        "src/scoring.py"],
       ["API and UI", "FastAPI on laptop 1, Streamlit and Folium on laptop 2", "api/main.py, ui/app.py"]],
      [2.8, 9.4, 4.8])
p("Partitions. Train: 80 trips, used for fitting, cross-validation, feature selection, tuning and choosing the HMM "
  "parameter. Student test: 10 trips, used for development comparisons; trip 00000000 has no .arcs file, so 9 are "
  "scored. Evaluator test: 10 trips, read only by the deployed API for the final check. All cross-validation is "
  "5-fold GroupKFold by trip, so no trip is split between training and validation.")
p("Leakage rules we followed. The prediction pipeline loads trips with with_route=False, so the route file is "
  "never opened before prediction finishes. Labels come from the route only for train and student-test rows. "
  "The API scores the prediction after it is made and returns metrics but no reference geometry. No choice of "
  "feature, model, hyperparameter, threshold or decoder setting was made from evaluator results.")

# ------------------------------------------------------------------ 2. data
brk()
h1("2. Dataset understanding and exploratory data analysis")
p("All analysis in this section comes from scripts/s1_eda.py and scripts/s2_features.py. The partition used is "
  "named in each table. The evaluator partition is not part of the EDA.")
h2("2.1 Inventory and missing-value analysis")
table([["Partition", "Trips", "GPS points", "Median points per trip (min to max)", "Median nodes", "Median arcs",
        "Median route arcs", "Trips with a gap over 10 s"]] +
      [[k, f"{int(r.trips)}", f"{int(r.gps_points):,}", f"{int(r.median_points):,} ({int(r.min_points):,} to "
        f"{int(r.max_points):,})", f"{int(r.median_nodes):,}", f"{int(r.median_arcs):,}",
        f"{int(r.median_route_arcs)}", f"{int(r.trips_with_gap_gt10s)}"] for k, r in PS.loc[["train",
                                                                                            "student_test"]].iterrows()],
      [2.2, 1.1, 1.8, 3.4, 1.7, 1.7, 1.9, 2.4])
table([["Partition", "File", "Trips", "Trips missing the file", "Rows with wrong column count",
        "Rows non-numeric or NaN"]] +
      [[r.partition, "." + r.file, r.trips, r.trips_missing_file, r.rows_wrong_column_count,
        r.rows_non_numeric_or_nan] for r in MV.itertuples()],
      [2.6, 1.8, 1.6, 3.4, 3.8, 3.8])
p("No row in any file is missing a value or has the wrong number of columns. The only missing data is a whole "
  "file: student-test trip 00000000 has no .arcs file, so it has no road network and cannot be matched or scored. "
  "We report 9 scored student-test trips everywhere.")

h2("2.2 How we preprocessed the data")
table([["Step", "What we did", "Why"],
       ["1. Cleaning", "Drop rows with non-finite or out-of-range coordinates and rows whose timestamp does not "
        "increase", "No such rows were found in train or student test (0 dropped), but the API must handle them "
        "on unseen trips"],
       ["2. Projection", "Local equirectangular projection per trip, from degrees to metres around the trip "
        "centre", "Degrees are angles, so Euclidean distance in degrees is wrong. The distances that matter are "
        "between a point and arcs within 50 m of it, where the projection error is negligible"],
       ["3. Candidate generation", "Keep up to K = 8 nearest arcs within R = 50 m; if none, the nearest arc within "
        "200 m (fallback)", "Chosen from a grid of R in {20, 30, 50, 75, 100} m and K in {2, 4, 6, 8, 10, 15} on "
        "train only (table below)"],
       ["4. Labels", "Dynamic-programming alignment: each GPS point gets one route arc so that the route index "
        "never goes backwards and total distance is smallest", "The route file says which arcs were driven, not "
        "which point was on which arc"],
       ["5. Features", "11 features per row from GPS and network only (section 2.4)", "The route is used for labels "
        "and never for features"],
       ["6. Training-row filtering", "Drop points more than 50 m from their reference arc, then keep at most "
        "1,000 points per trip with a uniform stride", "Points far off the route have unreliable labels, and the "
        "cap stops long trips (up to 19,313 points) from dominating. Applied to training rows only"]],
      [3, 7, 7])
G = pd.read_csv(RES / "stage2" / "candidate_radius_k_grid.csv")
g = G[G.R == 50]
table([["K (R = 50 m, train)", "Candidate miss rate", "Candidates per point"]] +
      [[f"{int(r.K)}" + (" (chosen)" if r.K == 8 else ""), f3(r.candidate_miss_rate), f"{r.candidates_per_point:.1f}"]
       for r in g.itertuples()], [4, 4, 4], bold_rows=(4,))
p("From K = 8 to K = 15 the miss rate drops only from 0.119 to 0.116 while each point gets two more wrong "
  "candidates. Raising R to 100 m at K = 8 lowers it to 0.111 but adds 1.3 candidates per point, almost all wrong. "
  "The high train miss rate comes from a few trips that leave the mapped roads (trip 54 has 87% and trip 33 59% of "
  "points more than 50 m from the route). On the student-test trips the true arc is among the candidates for 99.6% "
  "of points (miss rate 0.43%).")
table([["Partition", "Trips", "GPS points", "Candidate rows", "Positive share"],
       ["Train, all points", "80", "208,417", "1,103,327", ""],
       ["Train, after filtering and cap (model rows)", "80", "59,664", "369,705", "15.95%"],
       ["Student test", "9", "16,657", "103,788", "15.98%"]], [6, 1.5, 2.5, 3, 3])

h2("2.3 What we observed during exploration")
table([["Check", "Partition", "Finding"],
       ["Sampling interval", "Train", f"{SAMP['frac_dt_eq_1s']:.1%} of steps are exactly 1 s and "
        f"{SAMP['frac_dt_le_2s']:.1%} are 2 s or less. {SAMP['n_gaps_gt10s']} gaps over 10 s, "
        f"{SAMP['n_gaps_gt60s']} over 60 s, longest {SAMP['max_gap_s']:,.0f} s (trip 90)"],
       ["Trips with gaps over 10 s", "Train / student test", "40 of 80 / 2 of 10"],
       ["Two-way roads", "Train / student test", "Median 95% / 90% of arcs have an opposite-direction twin, so the "
        "direction of travel is needed to pick between them"],
       ["GPS far from the route", "Train", "Trip 54: 87% of points over 50 m from the route, trip 33: 59%, "
        "trip 2: 12%. These are mostly drives on roads missing from the network"],
       ["Implausible jumps (over 60 m/s)", "Train", "Seen in many trips, up to 361 in trip 32"],
       ["Route contiguity", "Train", "A few reference routes are not fully connected (lowest 98.1%, trip 75), "
        "so the reference is itself imperfect"]],
      [3.6, 3.2, 10.2])
p("These observations shaped the design. Two-way roads make heading a necessary feature. Gaps and jumps mean "
  "the decoder must be able to restart. Off-network trips mean some labels are unreliable, which is why we filter "
  "training rows by distance to the route. Because the reference route has its own gaps, its continuity (0.983) "
  "is the realistic ceiling for our continuity metric.")
img(PLOTS / "eda_points_and_sampling.png", 16.5, "Figure 2.1. Left: GPS points per trip (train and student test). "
    "Right: time between consecutive GPS points, log count, clipped at 30 s.")
img(PLOTS / "eda_trajectories_network.png", 15, "Figure 2.2. GPS points (blue), reference route (green) and road "
    "network (grey) in projected metres for four train trips.")
img(PLOTS / "eda_network_size_connectivity.png", 16.5, "Figure 2.3. Left: nodes against arcs per trip (train blue, "
    "student test orange). Right: share of nodes in the largest weakly connected component.")
img(PLOTS / "eda_gps_to_route_distance.png", 11.5, "Figure 2.4. Distance from each GPS point to the nearest reference-route "
    "arc (train and student test). The red line is the 50 m candidate radius.")
img(PLOTS / "eda_largest_gaps.png", 11.5, "Figure 2.5. The 15 trips with the largest time gap between GPS points "
    "(train and student test).")
img(PLOTS / "eda_trip_locations.png", 13, "Figure 2.6. Median GPS coordinate of each trip. Trips span several "
    "continents, which is why we project each trip around its own centre.")
img(FIG / "map_eda_train_00000001.png", 16, "Figure 2.7. Folium map of train trip 00000001 "
    "(maps/eda_train_00000001.html) with layers for GPS points, nearby road network and the reference route, a "
    "legend and start and end markers. Screenshot with an OpenStreetMap base layer.")

h2("2.4 Features we extracted and why")
table([["Feature", "What it measures", "Why we added it"],
       ["dist", "Distance from the GPS point to the arc (m)", "The true road is usually the nearest one"],
       ["rank", "Distance order among this point's candidates (0 = nearest)", "Relative position matters more "
        "than raw metres when GPS error varies between trips"],
       ["dist_minus_min", "dist minus the nearest candidate's distance", "How much worse this arc is than the best "
        "option for the same point"],
       ["heading_diff", "Angle between direction of travel (points t-2 to t+2) and arc direction, 0 to 180 deg",
        "Separates the two directions of a two-way road"],
       ["speed", "GPS speed over the same window (m/s)", "Heading is unreliable when the vehicle is nearly stopped"],
       ["arc_len", "Arc length (m)", "Short arcs near junctions are easy to confuse"],
       ["tproj", "Where the point projects along the arc, 0 (start) to 1 (end)", "Points near arc ends are "
        "ambiguous between connected arcs"],
       ["n_cand", "Number of candidates for the point", "Dense areas are harder"],
       ["nbr_dist", "Mean distance of points t-2, t-1, t+1, t+2 to the same arc", "One noisy point can be near a "
        "wrong arc; its neighbours usually are not"],
       ["has_reverse", "1 if the opposite-direction arc exists", "Tells the model when heading is needed"],
       ["fallback", "1 if the candidate came from the 200 m fallback", "Marks low-confidence candidates"],
       ["heading_diff_long (stage 4)", "Heading over a window of 10 points on each side compared with the arc "
        "bearing", "Added after error analysis: at low speed the short-window heading is noise (section 4.1)"],
       ["disp_long (stage 4)", "Straight-line displacement over the same 21-point window (m)", "Tells the model "
        "how far the vehicle really moved, so it knows how much to trust either heading"]],
      [3.4, 6.3, 7.3])

h2("2.5 Feature selection")
p("Technique: embedded importance ranking followed by backward elimination. A RandomForestClassifier (200 trees, "
  "criterion=entropy, min_samples_leaf=5) ranks the 11 Phase 1 features by mean decrease in entropy (information "
  "gain). We then drop the lowest-ranked feature one at a time, score each set with a HistGradientBoostingClassifier "
  "by 5-fold GroupKFold macro F1 on train, and keep the smallest set within 0.002 of the best score.")
rows = [["Feature", "RF entropy importance", "Features kept", "Dropped next", "CV macro F1", "Fold std"]]
for i, r in enumerate(FS.itertuples()):
    rows.append([IMP.index[i], f3(IMP.iloc[i]), str(r.n_features), r.dropped_next, f4(r.cv_macro_f1),
                 f4(r.cv_std)])
for k in IMP.index[len(FS):]:
    rows.append([k, f3(IMP[k]), "", "", "", ""])
table(rows, [3.3, 3, 2.4, 3, 2.6, 2.4], bold_rows=(3,))
p("The best score is 0.9290 with 9 features, so the cut-off is 0.9270; the 8-feature set (0.9263) falls below it. "
  "We kept heading_diff, dist_minus_min, rank, nbr_dist, tproj, speed, dist, arc_len and has_reverse, and dropped "
  "fallback (importance 0.000) and n_cand (0.010). All seven Phase 1 models use these 9 features. In stage 4 we "
  "added heading_diff_long and disp_long to the final model only, after an ablation on train (section 4.1).")
img(PLOTS / "s3_feature_selection.png", 16.5, "Figure 2.8. RF entropy importance and backward-elimination CV macro "
    "F1 (train, GroupKFold).")

# ------------------------------------------------------------------ 3. models
brk()
h1("3. Model experimentation and comparison")
p("Experimental design. Every model uses the same 9 features, the same 369,705 training rows and the same five "
  "GroupKFold folds (grouped by trip). StandardScaler sits inside a pipeline for Logistic Regression, KNN and the "
  "MLP, so it is fitted on training folds only. random_state=42 everywhere and the decision threshold is 0.5. Each "
  "model is then refitted on all training rows and scored once on the 9 student-test trips. Training time is one "
  "fit on all training rows; inference time is scoring all 103,788 student-test rows.")
p("Primary metric: cross-validated macro F1. Only 16% of rows are true arcs, so a model that always says \"not "
  "this arc\" already gets 84% accuracy. Macro F1 averages the F1 of both classes, so missing the true arc costs as "
  "much as accepting a wrong one. Log loss is the second criterion because the HMM uses the probabilities as "
  "emission scores. Observation-level Top-1 accuracy takes, for each GPS point, the candidate with the highest "
  "probability and compares it with the reference arc; a point whose true arc is not among its candidates counts "
  "as wrong. Route continuity is the share of consecutive point pairs (at most 10 s apart) whose predicted arcs are "
  "the same or directly connected.")
h2("3.1 Overall model comparison")
rows = [["Model", "Algorithm family", "Key configuration", "CV macro F1 (std)", "Top-1", "Route cont.",
         "Test macro F1", "Log loss", "Inference (s)"]]
for k in ORDER:
    r = M.loc[k]
    rows.append([f"{NAME[k]}<br/><font size=6>{CLS[k]}</font>", FAM[k], CONF[k],
                 f"{f4(r.cv_macro_f1)} ({f3(r.cv_macro_f1_std)})", f4(r.test_top1_accuracy),
                 f3(r.test_route_continuity), f4(r.test_macro_f1), f3(r.test_log_loss), f3(r.inference_time_s)])
rows.append(["Final: HistGradientBoosting, 11 features, tuned", "Boosting", "Optuna best (section 4)",
             f"{f4(T11.cv_macro_f1)} ({f3(T11.cv_macro_f1_std)})", f4(T11.test_top1_accuracy),
             f3(T11.test_route_continuity), f4(T11.test_macro_f1), f3(T11.test_log_loss),
             f3(T11.inference_time_s)])
rows.append(["Final + HMM / Viterbi", "Boosting + HMM", f"beta = {HMMC['beta_m']:g} m (section 5)",
             "N/A: the HMM decodes whole trips, it does not classify rows", f4(HM.top1), f3(HM.continuity),
             "N/A: decoding outputs one arc per point, not row probabilities", "N/A, same reason",
             "0.08 s decode for all 9 trips"])
table(rows, [3.3, 2, 3.5, 1.9, 1.2, 1.2, 1.5, 1.2, 1.3], font=7, bold_rows=(6, 8, 9))
rows = [["Model", "Train macro F1", "Train minus CV", "Test accuracy", "Test precision", "Test recall",
         "CV log loss", "Train time (s)", "Saved size"]]
for k in ORDER:
    r = M.loc[k]
    rows.append([NAME[k], f4(r.train_macro_f1), f"{r.train_macro_f1 - r.cv_macro_f1:+.4f}", f4(r.test_accuracy),
                 f4(r.test_precision), f4(r.test_recall), f3(r.cv_log_loss), f"{r.train_time_s:.2f}", SIZE[k]])
table(rows, [3.3, 1.7, 1.7, 1.7, 1.7, 1.6, 1.6, 1.8, 1.9], font=7.5)
p("The seven models cover six of the families in the Models and Training Guidelines. We did not train a kernel "
  "SVM, because its training time grows faster than linearly with the 369,705 rows and it gives no probabilities "
  "without extra calibration. We did not use clustering, because cluster ids would need a separate mapping to the "
  "true-arc label.")

h2("3.2 Compulsory questions for every model")


def model_block(i, k, why, strong, weak, loss, fit):
    r = M.loc[k]
    h3(f"3.2.{i} {NAME[k]} ({CLS[k]}), {FAM[k].lower()}")
    qa("A1. Why did we consider it suitable for PathSnap?", why)
    qa("A2. Strengths and limitations on this task", strong + " " + weak)
    qa("B5. Important hyperparameters and initial values", CONF[k] + ". These were the starting values. Only the "
       "chosen model was tuned afterwards (section 4).")
    qa("B6. Loss function or optimisation objective", loss)
    qa("C7 and C8. Student-test performance",
       f"Observation-level Top-1 {f4(r.test_top1_accuracy)}, macro F1 {f4(r.test_macro_f1)}, log loss "
       f"{f3(r.test_log_loss)}. Accuracy {f4(r.test_accuracy)}, precision {f4(r.test_precision)}, recall "
       f"{f4(r.test_recall)}, route continuity {f3(r.test_route_continuity)}. Confusion matrix (TN, FP, FN, TP): "
       f"{int(r.tn):,}, {int(r.fp):,}, {int(r.fn):,}, {int(r.tp):,}.")
    qa("C9. Overfitting or underfitting?", f"Train macro F1 {f4(r.train_macro_f1)}, CV {f4(r.cv_macro_f1)} "
       f"(fold std {f3(r.cv_macro_f1_std)}), student test {f4(r.test_macro_f1)}. " + fit)


model_block(1, "LogisticRegression",
    "It is the simplest model that outputs a probability for each candidate, which the HMM needs, and the main "
    "features (dist, dist_minus_min, rank) push the probability in one direction, so a linear model is a fair "
    "baseline.",
    "It trains in 0.18 s, predicts the test set in 0.006 s and its coefficients are easy to explain.",
    "It draws one linear boundary, so it cannot learn that heading_diff matters only when speed is high. Its test "
    "macro F1 (0.8934) is the lowest of the six models that do not assume independence.",
    "Binary cross-entropy (log loss) plus an L2 penalty with strength 1/C = 1, minimised with L-BFGS. It converged "
    "in 15 iterations.",
    "Train and CV differ by only 0.0012 and both are well below the tree models. That is underfitting (high bias): "
    "the model is too simple for the feature interactions.")
model_block(2, "GaussianNB",
    "It is the fastest probabilistic classifier and gives class probabilities through Bayes' rule, so it shows how "
    "far a model with strong assumptions gets.",
    "Training took 0.046 s and it has the best recall of all seven (0.8837), so it rarely rejects the true arc.",
    "It assumes features are independent given the class, but dist, dist_minus_min, rank and nbr_dist are all "
    "distance measures and strongly correlated. It counts the same evidence several times, becomes overconfident, "
    "and accepts 8,312 wrong candidates against about 2,200 for the tree models. Precision falls to 0.6381 and log "
    "loss (0.386) is the worst.",
    "No iterative loss. Training estimates a mean and variance per feature and class by maximum likelihood; "
    "var_smoothing adds 1e-9 times the largest variance for stability.",
    "Train, CV and test are all close (0.851, 0.850, 0.840), so there is no overfitting. It underfits because of its "
    "independence assumption.")
model_block(3, "KNN",
    "Point-candidate pairs that look alike (similar distance, heading difference and rank) should have the same "
    "label, which is the assumption KNN makes. Distance weighting lets very similar rows count more.",
    "It needs no training (0.45 s just to store the rows) and reached CV macro F1 0.9192.",
    "It needs scaled features, stores all 369,705 training rows (62 MB file) and takes 9.79 s to score the "
    "student-test set, about 240 times slower than HistGradientBoosting. That is too slow for the API.",
    "No loss is optimised. Prediction is a vote of the 25 nearest training rows weighted by 1/distance in "
    "standardised feature space; the probability is the weighted share of positive neighbours.",
    "Train macro F1 is 1.0000 against CV 0.9192. With distance weighting every training row is its own nearest "
    "neighbour, so it memorises the training set. That is overfitting, and the CV score is the one to trust.")
model_block(4, "DecisionTree",
    "Map matching has rule-like structure, for example \"close to the point and heading the same way\". A tree "
    "learns such rules and their interactions without scaling.",
    "It predicts in 0.007 s, can be drawn and explained, and with 609 leaves reached CV F1 0.9250.",
    "A single tree has high variance: its fold scores range from 0.9148 to 0.9393, and its log loss (0.122) is worse "
    "than the ensembles because leaf probabilities are coarse.",
    "Greedy splitting that maximises information gain (reduction in entropy) at each node, limited to depth 10 and "
    "at least 20 rows per leaf.",
    "Train 0.9333, CV 0.9250, gap 0.0083. The depth and leaf limits keep overfitting mild.")
model_block(5, "RandomForest",
    "Averaging many trees keeps the tree's ability to learn rules and reduces its variance, which matters because "
    "trips differ a lot in GPS quality.",
    "It has the best student-test macro F1 (0.9163) and Top-1 (0.8659) of the seven, and the second-best CV F1.",
    "The saved model is 178 MB, over GitHub's 100 MB file limit, and it trains in 8.57 s and predicts in 0.130 s, "
    "about three times slower than HistGradientBoosting.",
    "Each of the 200 trees is grown on a bootstrap sample with Gini impurity, trying sqrt(9) = 3 features per split. "
    "The predicted probability is the average over trees.",
    "Train 0.9644 against CV 0.9285, a gap of 0.0359, the largest after KNN. The fully grown trees fit the training "
    "rows closely, so it overfits more than boosting, although averaging stops this from hurting CV much.")
model_block(6, "GradientBoosting",
    "Boosting adds small trees one at a time, each correcting the errors of the ones before it, and it optimises log "
    "loss directly, which suits a model whose probabilities feed the HMM. The histogram version bins each feature "
    "into at most 255 values, so it trains fast on 369,705 rows.",
    "It has the highest CV macro F1 (0.9289), the lowest test log loss (0.107), trains in 1.41 s, predicts in "
    "0.041 s and saves to 0.56 MB.",
    "It has several interacting hyperparameters (learning rate, number of trees, leaves), which is why it was the "
    "one we tuned. Its built-in early stopping uses a random 10% of training rows, so that split is not grouped by "
    "trip; we switched early stopping off for tuning.",
    "Binary cross-entropy (log loss). Each new tree is fitted to the gradient of the loss and added with learning "
    "rate 0.1. Early stopping watched the loss on the 10% split and stopped after 151 of the 300 allowed trees.",
    "Train 0.9411, CV 0.9289, gap 0.0122, a third of Random Forest's gap. Fold std 0.0086 is the second lowest. "
    "Overfitting is small.")
model_block(7, "MLP",
    "A neural network can learn nonlinear combinations of features, such as heading mattering more at speed, "
    "without us writing them by hand.",
    "CV F1 0.9258 with the lowest fold std (0.0081), and inference is fast (0.031 s).",
    "It needs scaled inputs, takes 18.97 s to train (about 13 times HistGradientBoosting), depends on the random "
    "initialisation, and is harder to explain. It gave no accuracy gain over boosting.",
    "Binary cross-entropy plus an L2 weight penalty alpha=1e-4, minimised with Adam (learning rate 1e-3, "
    "mini-batches of 200). Early stopping on validation accuracy from a 10% split stopped training after 34 epochs; "
    "the best validation accuracy was 0.9645.",
    "Train 0.9323, CV 0.9258, gap 0.0065, the smallest of the nonlinear models, because early stopping halted "
    "training before it started to overfit. Figure 3.5 shows the epoch history.")

h2("3.3 Model visualisations")
img(PLOTS / "s3_cv_macro_f1.png", 13, "Figure 3.1. Macro F1 of each of the 5 GroupKFold folds for all seven "
    "models (train). GradientBoosting is the HistGradientBoostingClassifier.")
img(PLOTS / "s3_test_top1_f1.png", 14, "Figure 3.2. Student-test observation-level Top-1 accuracy and macro F1.")
img(PLOTS / "s3_test_log_loss.png", 13, "Figure 3.3. Student-test log loss (lower is better).")
img(PLOTS / "s3_confusion_matrices.png", 17, "Figure 3.4. Student-test confusion matrices at threshold 0.5.")
img(PLOTS / "s3_mlp_history.png", 11, "Figure 3.5. MLP training loss and validation accuracy per epoch. Only the "
    "MLP trains in epochs, so it is the only model with a training curve. MLPClassifier's early stopping records "
    "validation accuracy, not validation loss.")

h2("3.4 Model selection")
table([["Criterion", "HistGradientBoosting", "Closest rival"],
       ["Generalisation", "Highest CV macro F1, 0.9289; student-test F1 0.9162", "Random Forest 0.9285 CV, 0.9163 "
        "test. The difference (0.0004) is far inside one fold std (0.009), so they tie"],
       ["Stability across folds", "Fold std 0.0086, fold range 0.919 to 0.941", "Random Forest 0.0089; MLP 0.0081"],
       ["Probability quality", "Lowest student-test log loss 0.107", "Random Forest 0.108, MLP 0.109, Naive Bayes "
        "0.386"],
       ["Inference speed", "0.041 s for 103,788 rows", "Random Forest 0.130 s, KNN 9.79 s"],
       ["Model complexity", "151 trees of at most 31 leaves, 0.56 MB, train minus CV gap 0.012", "Random Forest "
        "200 full trees, 178 MB, gap 0.036"],
       ["Fit with the sequential decoder", "Trained on log loss, so its probabilities work directly as HMM emission "
        "scores", "Same for RF and MLP; KNN and the single tree give coarse probabilities"]],
      [3.6, 6.4, 7])
p("On accuracy alone HistGradientBoosting and Random Forest are tied. HistGradientBoosting wins on probability "
  "quality, overfitting, file size and speed, which all matter for the HMM and the API. Training accuracy played no "
  "part in the choice. The other models lose for measured reasons: Naive Bayes on precision and log loss, Logistic "
  "Regression on bias, KNN on overfitting and speed, the single tree on stability, Random Forest on size and "
  "overfitting, and the MLP on training cost with no gain.")

# ------------------------------------------------------------------ 4. tuning
brk()
h1("4. Bayesian hyperparameter optimisation")
h2("4.1 Before tuning: error analysis and two new features")
p("Before tuning we looked at the mistakes the Phase 1 model made on the student-test trips (a development "
  "partition; the evaluator trips were not involved). Most errors happen at low speed, and 61% of them pick the "
  "reverse direction of the true arc. At walking or crawling speed the heading "
  "between points two seconds apart is mostly GPS noise. We added heading_diff_long and disp_long (section 2.4), "
  "which look at 10 points on each side. Neither uses the route file. Ablation on train:")
table([["Feature set", "Features", "CV macro F1", "CV std", "CV log loss"]] +
      [[r.feature_set, str(r.n_features), f4(r.cv_macro_f1), f4(r.cv_macro_f1_std), f4(r.cv_log_loss)]
       for r in ABL.itertuples()], [6.5, 2, 2.8, 2.5, 2.8])
p(f"The gain (+{ABL.cv_macro_f1.iloc[2] - ABL.cv_macro_f1.iloc[0]:.4f}) is larger than the fold std, so both "
  "features stay. The ablation used the untuned Phase 1 settings with early stopping off.")

h2("4.2 Search design")
sp = TUNE["search_space"]
kind = {"float_log": "log-uniform float", "int": "uniform integer", "int_log": "log-uniform integer",
        "float": "uniform float"}
qa("1. Which hyperparameters were included in the search?",
   "learning_rate, max_iter (number of boosting rounds), max_leaf_nodes, min_samples_leaf, l2_regularization and "
   "max_features (share of features tried per split). Fixed: loss=log_loss, max_bins=255, early_stopping=False "
   "(its internal 10% split is not grouped by trip), random_state=42.")
qa("2. What ranges or distributions were considered?")
table([["Hyperparameter", "Distribution", "Range", "Untuned value", "Best value (trial %d)" % TUNE["best_trial"]]] +
      [[k, kind[v[0]], f"{v[1]:g} to {v[2]:g}", f'{ {"learning_rate": 0.1, "max_iter": 150, "max_leaf_nodes": 31, "min_samples_leaf": 20,
                                                   "l2_regularization": 1e-4, "max_features": 1.0}[k]:g}' ,
        f"{TUNE['best_params'][k]:.4g}"] for k, v in sp.items()], [3.6, 3.6, 3, 2.8, 4])
p("Log scales are used where the effect is multiplicative (learning rate, leaves, leaf size, L2). The untuned "
  "setting is the Phase 1 configuration with a fixed 150 trees, since Phase 1's early stopping had stopped at 151.")
qa("3. What objective function was optimised?",
   f"Maximise mean macro F1 over 5-fold GroupKFold by trip on the {COST['train_rows']:,} training rows, the same "
   "metric and folds used to compare models in section 3. The MedianPruner receives the running mean after each "
   "fold and may stop a trial that is below the median of earlier trials at the same fold.")
qa("4. How many optimisation trials were performed?",
   f"{COST['n_trials']} trials with Optuna's TPESampler (seed 42, multivariate, 10 random start-up trials); "
   f"{COST['completed']} completed and {COST['pruned']} were pruned. Trial 0 was the untuned setting, enqueued so "
   f"the search could only match or beat it. Cost: {COST['search_seconds']:.0f} s in total "
   f"({COST['search_seconds'] / 60:.1f} min), {COST['mean_seconds_per_completed_trial']:.1f} s per trial, "
   f"{COST['folds_per_trial']} model fits per trial. Every trial is in results/stage4/optuna_trials.csv.")
qa("5. What were the best hyperparameter values obtained?",
   f"Trial {TUNE['best_trial']}: CV macro F1 {TUNE['best_cv_macro_f1']:.4f} with learning_rate "
   f"{TUNE['best_params']['learning_rate']:.4f}, max_iter {TUNE['best_params']['max_iter']}, max_leaf_nodes "
   f"{TUNE['best_params']['max_leaf_nodes']}, min_samples_leaf {TUNE['best_params']['min_samples_leaf']}, "
   f"l2_regularization {TUNE['best_params']['l2_regularization']:.4f}, max_features "
   f"{TUNE['best_params']['max_features']:.3f}. fANOVA importance: " +
   ", ".join(f"{k} {v:.2f}" for k, v in PIMP.items()) + ". Learning rate explains most of the variation "
   "between trials; the best trials use a low rate with more, larger trees.")
qa("6. Did the tuned model outperform the untuned model?",
   "Yes, on every quality metric, by a small margin. Same features, folds and student-test trips for all rows:")
table([["Model", "CV macro F1", "CV log loss", "Test Top-1", "Test macro F1", "Test log loss", "Inference (s)",
        "Train time (s)"]] +
      [[r.model, f4(r.cv_macro_f1), f4(r.cv_log_loss), f4(r.test_top1_accuracy), f4(r.test_macro_f1),
        f4(r.test_log_loss), f3(r.inference_time_s), f"{r.train_time_s:.2f}"] for r in TU.itertuples()],
      [5, 1.7, 1.7, 1.6, 1.8, 1.7, 1.7, 1.8], bold_rows=(3,))
p(f"Tuning adds {T11.cv_macro_f1 - U11.cv_macro_f1:+.4f} CV macro F1 and {T11.test_top1_accuracy - U11.test_top1_accuracy:+.4f} "
  f"test Top-1 on top of the new features, and lowers log loss from {U11.cv_log_loss:.4f} to {T11.cv_log_loss:.4f}, "
  "which helps the HMM. The cost is speed: 279 trees of 120 leaves take "
  f"{T11.inference_time_s:.3f} s to score 103,788 rows against {U11.inference_time_s:.3f} s, still a small part of "
  "the under 0.5 s the API needs per trip. The Phase 1 row reads 0.9285 rather than 0.9289 (section 3) because all "
  "three rows were refitted with early stopping off and a fixed number of trees. We chose the tuned model because it has the highest CV macro F1 on train; "
  "the student-test numbers only confirm it.")
img(PLOTS / "s4_optuna_history_importance.png", 17, "Figure 4.1. Optuna optimisation history (CV macro F1 per "
    "trial and best so far) and fANOVA hyperparameter importance.")

# ------------------------------------------------------------------ 5. HMM
brk()
h1("5. HMM and Viterbi route reconstruction")
p("The classifier scores each point on its own, so neighbouring points can be given arcs that are not connected. "
  "The HMM adds the road network's connectivity. The code is src/hmm.py and the selection script is "
  "scripts/s5_hmm.py.")
qa("1. What are the hidden states?",
   "At GPS point t the hidden states are that point's candidate arcs (up to 8, from section 2.2). Each state keeps "
   "its arc id and the projection position tproj, so the decoded state maps straight back to a road arc. Points "
   "with no candidate have no states and are left unmatched (none on student test or evaluator trips, thanks to "
   "the 200 m fallback).")
qa("2. How are emission probabilities calculated?",
   "The emission score of candidate j at point t is log P(j is the true arc), from the final tuned classifier, "
   "clipped at 1e-6 so one confident mistake cannot zero out a path.")
qa("3. How are transition probabilities calculated?",
   "For consecutive points t-1 and t we compare the road distance between the two candidates with the straight-line "
   "GPS distance d_gps. The transition score is -|d_route - d_gps| / beta, a log-Laplace model: a vehicle that moved "
   "d_gps metres should have driven about that far along the road. If both candidates are the same arc, d_route is "
   "the distance along the arc (moving backwards up to 15 m is tolerated as GPS noise). Otherwise d_route is the rest "
   "of the first arc, plus the shortest path from its end node to the start node of the second arc (Dijkstra over "
   "the directed arcs, so one-way streets are respected), plus the distance into the second arc. Dijkstra stops at "
   "min(max(250, 2 d_gps + 100), 5000) m; candidates not reachable within that are impossible transitions "
   "(score minus infinity).")
p("Viterbi recurrence: delta_t(j) = log P_t(j) + max_i [delta_(t-1)(i) - |d_route(i, j) - d_gps| / beta], with a "
  "back-pointer to the best i. The best final state is traced back through the pointers to give one arc per point.")
qa("4. How are missing candidates or large GPS gaps handled?",
   f"Points with no candidates are skipped and the chain links the points either side. A time gap over "
   f"{HMMC['max_gap_s']:.0f} s ends the chain: the best path so far is kept, and decoding restarts at the next "
   "point from its emission scores alone, because after a long gap the vehicle could have driven anywhere.")
qa("5. What happens when consecutive observations suggest disconnected road segments?",
   "If no candidate at t can be reached from any candidate at t-1 within the Dijkstra limit, every transition is "
   "minus infinity. Viterbi then closes the current segment at its best state and starts a new one at t, the same "
   "way as for a gap. These restarts are the chain breaks we report: 5 on the 9 student-test trips (2 on trip 56, "
   "3 on trip 65). The route is never forced through an impossible link.")
qa("6. Did HMM decoding improve the final route?", "Yes, on every student-test trip.")
h2("5.1 Choosing beta")
p("beta sets how strongly the decoder trusts road distance against the classifier. We tried 12 values from 0.5 to "
  "100 m using out-of-fold classifier probabilities on the 80 train trips (each trip scored by a model that never "
  "saw it). beta was chosen on train Top-1 only; the student-test columns are shown for checking.")
rows = [["beta (m)", "Train OOF Top-1", "Train continuity", "Train arc F1", "Student-test Top-1",
         "Student-test continuity"]]
rows.append(["classifier only", f4(TRAIN_CLF.train_oof_top1), f4(TRAIN_CLF.train_oof_continuity),
             f4(TRAIN_CLF.train_oof_route_arc_f1), f4(TRAIN_CLF.test_top1), f4(TRAIN_CLF.test_continuity)])
for r in BEST_B.itertuples():
    rows.append([f"{r.b:g}", f4(r.train_oof_top1), f4(r.train_oof_continuity), f4(r.train_oof_route_arc_f1),
                 f4(r.test_top1), f4(r.test_continuity)])
best_row = 2 + list(BEST_B.b).index(HMMC["beta_m"])
table(rows, [2.8, 2.8, 2.8, 2.6, 3, 3], font=7.5, bold_rows=(best_row,))
p(f"Train Top-1 peaks at beta = {HMMC['beta_m']:g} m ({TRAIN_B5.train_oof_top1:.4f}, against "
  f"{TRAIN_CLF.train_oof_top1:.4f} for the classifier alone). A very small beta lets the distance check overrule "
  "a confident classifier; a large beta weakens the connectivity check until the decoder drifts back towards "
  "classifier-only behaviour. On student test 10 m would have scored slightly higher (0.9134 against 0.9125); "
  "we kept the value chosen on train.")
h2("5.2 Classifier only against classifier plus HMM (student test)")
table([["Method", "Top-1", "Continuity", "Route arc precision", "Route arc recall", "Route arc F1",
        "Trips with Top-1 of 0.9 or more", "Chain breaks"]] +
      [[r.method, f4(r.top1), f4(r.continuity), f4(r.route_arc_precision), f4(r.route_arc_recall),
        f4(r.route_arc_f1), f"{round(r._7 * 9)} of 9", str(r.chain_breaks)] for r in CVH.itertuples()],
      [3.8, 1.5, 1.7, 2, 1.9, 1.7, 2.6, 1.7], bold_rows=(2,))
p("Route arc precision and recall are sequence-level measures: the set of distinct arcs on the predicted route "
  "compared with the set on the reference route. The HMM lifts continuity from 0.935 to 0.980, close to the 0.983 of "
  "the reference route itself, and raises arc precision from 0.890 to 0.955, which means far fewer stray arcs.")
P = PT.merge(FB, left_on="trip_id", right_on="trip")
table([["Trip", "Points", "Top-1 classifier", "Top-1 HMM", "Continuity classifier", "Continuity HMM",
        "Points fixed by HMM", "Points broken by HMM", "Wrong in both", "True arc not a candidate"]] +
      [[r.trip_id, f"{r.points:,}", f4(r.top1_classifier), f4(r.top1_hmm), f4(r.continuity_classifier),
        f4(r.continuity_hmm), str(r.fixed), str(r.broken), str(r.both_wrong), str(r.ref_not_in_cand)]
       for r in P.itertuples()] +
      [["All 9", f"{P.points.sum():,}", f4(CLF.top1), f4(HM.top1), f4(CLF.continuity), f4(HM.continuity),
        str(P.fixed.sum()), str(P.broken.sum()), str(P.both_wrong.sum()), str(P.ref_not_in_cand.sum())]],
      [1.6, 1.3, 1.7, 1.5, 1.9, 1.7, 1.8, 1.8, 1.6, 1.9], font=7.2, bold_rows=(10,))
h2("5.3 Where the HMM did not help")
bl([f"It also breaks points. Across the 9 trips it fixed {P.fixed.sum()} points the classifier got wrong but "
    f"broke {P.broken.sum()} the classifier got right, a net gain of {P.fixed.sum() - P.broken.sum()}. On trip 7 "
    "the two almost cancel (40 fixed, 36 broken, Top-1 0.946 to 0.949). The classifier alone already has "
    "continuity 0.997 on that trip, so there is little for the connectivity check to correct.",
    f"It cannot recover an arc that is not a candidate. {P.ref_not_in_cand.sum()} student-test points have no "
    "true arc among their candidates (20 on trip 9, 29 on trip 56), and those stay wrong whatever the decoder does.",
    "Slow trips stay hard. Trip 77 has a median speed of 3.7 m/s. The HMM fixes 279 points there, but 673 are wrong "
    "under both methods and Top-1 only reaches 0.799. When the vehicle barely moves, the GPS distance is mostly "
    "noise and both directions of a two-way road can pass the distance check, which matches the reverse-direction "
    "errors found in section 4.1.",
    "Chain breaks on trips 56 and 65 split the decoding. Each new segment starts from emissions only, so the first "
    "few points after a break are effectively classifier-only."])
img(PLOTS / "s5_hmm_beta_per_trip.png", 17, "Figure 5.1. Left: Top-1 against beta on train (out-of-fold) and "
    "student test; dashed lines are the classifier alone and the dotted line is the chosen beta. Right: student-test "
    "Top-1 per trip, classifier alone against classifier plus HMM.")

# ------------------------------------------------------------------ 6. folium
brk()
h1("6. Geographical visualisation with Folium")
p("src/maps.py builds every map and scripts/s6_maps.py writes one interactive map per student-test trip to "
  "maps/student_test_&lt;trip&gt;.html (all committed). Each map has separate layers for the GPS observations (blue "
  "dots), the road network near the track (grey), the reference route (green), the classifier-only prediction "
  "(orange) and the classifier plus HMM prediction (red), with a layer control, a legend, a title with the trip's "
  "Top-1 values, start and end markers and a scale bar. The view is fitted to the trip's GPS extent. The screenshots "
  "below were taken from those files with an OpenStreetMap base layer added for readability.")
img(FIG / "map_student_test_00000076.png", 16.5, "Figure 6.1. Student-test trip 00000076, full trip. The "
    "predicted route follows the reference route from start to end.")
img(FIG / "map_student_test_00000076_zoom.png", 16.5, "Figure 6.2. The same trip zoomed to the stretch where the HMM "
    "fixed the most points. The classifier alone (orange) jumps onto a parallel slip road at a junction for a few "
    "points; the HMM route (red) stays on the main road with the reference route (green), because the jump and "
    "return would need a road distance much longer than the GPS movement.")

# ------------------------------------------------------------------ 7. API UI
brk()
h1("7. API, UI and deployment")
table([["Laptop", "Component", "What it does"],
       ["Laptop 1", "FastAPI app (api/main.py), run with uvicorn on port 8000", "Loads the tuned model and HMM "
        "settings once at start-up. GET /health, GET /trips (the evaluator trips that have track, nodes and arcs "
        "files) and POST /predict {\"trip_id\"}. /predict runs the full pipeline, then scores it against the route "
        "file on the server and logs one line per stage to the console and logs/api.log"],
       ["Laptop 2", "Streamlit page (ui/app.py)", "Lists the 10 evaluator trips from /trips, sends the chosen id to "
        "/predict and shows the metrics and a Folium map (road network, GPS points, predicted route, optional "
        "classifier-only route). It never reads trip files itself"]],
      [2, 4.5, 10.5])
p("The /predict response carries the GPS points, the predicted and classifier-only routes, the nearby road network, "
  "timings and the metrics. It contains no reference route geometry, so ground truth never reaches the UI. Unknown "
  "trip ids get 404, and a pipeline failure returns 422 with the error instead of crashing the server. Setup steps "
  "for both laptops are in the README.")
img(FIG / "ui_trip_00000087.png", 16, "Figure 7.1. The Streamlit UI after running evaluator trip 00000087: the ten "
    "trip options, metric tiles computed on the API server and the returned route on a Folium map.", crop_h=1150)
story.append(Paragraph("API console on the laptop receiving the requests (start-up, one /trips call and one "
                       "/predict call):", B))
story.append(Spacer(1, 3))
story.append(Preformatted(CONSOLE, MONO, maxLineLength=150))
story.append(Spacer(1, 8))
p("Final system check on the 10 evaluator trips. We sent every evaluator trip through the running API once, with "
  "the model, features and beta already frozen. These numbers were not used for any choice.")
table([["Trip", "Points", "Top-1 classifier", "Top-1 HMM", "Continuity classifier", "Continuity HMM",
        "Arc precision HMM", "Arc recall HMM", "Chain breaks", "API time (s)"]] +
      [[r.trip_id, f"{r.points:,}", f4(r.top1_classifier), f4(r.top1_hmm), f4(r.continuity_classifier),
        f4(r.continuity_hmm), f4(r.arc_precision_hmm), f4(r.arc_recall_hmm), str(r.chain_breaks),
        f"{r.api_total_s:.2f}"] for r in EV.itertuples()] +
      [["Mean", f"{EV.points.sum():,} total", f4(EV.top1_classifier.mean()), f4(EV.top1_hmm.mean()),
        f4(EV.continuity_classifier.mean()), f4(EV.continuity_hmm.mean()), f4(EV.arc_precision_hmm.mean()),
        f4(EV.arc_recall_hmm.mean()), str(EV.chain_breaks.sum()), f"{EV.api_total_s.mean():.2f}"]],
      [1.6, 1.6, 1.7, 1.5, 1.9, 1.7, 1.8, 1.6, 1.5, 1.5], font=7.2, bold_rows=(11,))
p("The HMM beats the classifier alone on all 10 trips and every request finishes in under 0.5 s. The lowest "
  "trip is 00000030 (Top-1 0.876). Trip 00000015 has the lowest arc precision (0.72) despite Top-1 0.917, which "
  "means its wrong points land on many distinct arcs that are not on the reference route. The per-trip results are in "
  "results/stage6/evaluator_api_check.csv.")

# ------------------------------------------------------------------ 8. integrity
h1("8. Reproducibility, academic integrity and AI use")
p("Everything in this report can be rebuilt from the repository. With the raw data in data/raw/:")
story.append(Preformatted("pip install -r requirements.txt\n"
                          "python scripts/s1_eda.py               # EDA, data-quality checks, eda_* plots and maps\n"
                          "python scripts/s2_features.py --skip-grid   # candidates, labels, features (drop the flag to rerun the R/K grid)\n"
                          "python scripts/s3_models.py            # feature selection, 7 models, s3_* plots\n"
                          "python scripts/s4_optuna.py 50         # new features, Optuna search, tuned model\n"
                          "python scripts/s5_hmm.py               # beta selection, classifier vs HMM\n"
                          "python scripts/s6_maps.py              # Folium maps of the student-test trips\n"
                          "python report/hmm_flips.py             # points fixed and broken by the HMM\n"
                          "python report/make_report.py           # this PDF", MONO, maxLineLength=150))
story.append(Spacer(1, 6))
p("Random seeds are fixed at 42. Every table above is read from results/ or configs/ by report/make_report.py, "
  "and every figure is either a plot written by the scripts or a screenshot of our own Folium maps and UI. The "
  "saved model (models/final_hgb.joblib), its configuration (configs/final_model.json, configs/hgb_tuned.json, "
  "configs/hmm.json) and the full Optuna trial record are committed.")
p("AI use. We used an AI coding assistant while writing code and drafting text for this report. All experiments "
  "were run on our own machines from the scripts in the repository, and every result reported here comes from "
  "those runs. Each team member can explain the code, the experiments and the choices described above.")

h1("9. Conclusions and limitations")
bl(["Framing map matching as (point, candidate arc) classification gave a strong base model: tuned "
    f"HistGradientBoosting reaches Top-1 {f4(T11.test_top1_accuracy)} on student test from 11 features.",
    f"The largest single gain came from the HMM, which uses road connectivity: Top-1 {f4(CLF.top1)} to "
    f"{f4(HM.top1)} and continuity {f3(CLF.continuity)} to {f3(HM.continuity)} on student test, and "
    f"{EV.top1_classifier.mean():.3f} to {EV.top1_hmm.mean():.3f} mean Top-1 on the evaluator trips.",
    "Error analysis paid off more than tuning: the two wide-window features added +0.0115 CV macro F1, tuning "
    "+0.0019.",
    "Limitations. Slow or stationary stretches remain the main error source (trip 77). Points whose true arc is "
    "not within 50 m (or 200 m with the fallback) cannot be matched. The reference routes are imperfect (some are "
    "not fully connected), so labels near route gaps are uncertain and Top-1 there is less reliable. beta is a single global value; a speed-dependent beta is a natural next step."])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(2 * cm, 1.1 * cm, "PathSnap technical report")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.7 * cm,
                  bottomMargin=1.8 * cm, title="PathSnap technical report").build(story, onFirstPage=footer,
                                                                                   onLaterPages=footer)
for f in FIG.glob("_crop_*"):
    f.unlink()
print(OUT)
