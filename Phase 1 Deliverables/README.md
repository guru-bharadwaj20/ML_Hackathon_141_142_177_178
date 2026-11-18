# Phase 1 deliverables

Phase 1 asks for four things: preprocess the dataset, select features, train at least five models and record their accuracy, then pick one model and say why. This folder answers all four in plain words. The code lives in `src/` and `scripts/`, and every number below comes from the files in `results/`.

## How to run it

From the `pathsnap` folder, in PowerShell:

```powershell
pip install -r requirements.txt
python scripts\s1_eda.py          # data checks and EDA plots
python scripts\s2_features.py     # preprocessing, candidates, labels, features
python scripts\s3_models.py       # feature selection + all 7 models
python scripts\show_phase1.py     # prints every answer below in one screen
```

The raw data goes in `data/raw/train` and `data/raw/student_test`; it is not in git because of its size. The printed output of `show_phase1.py` from our results is saved in this folder as `output.txt`.

## The problem in simple words

Each trip has noisy GPS points and a road map. We have to say which road each GPS point was on. For every GPS point we take a few nearby roads (the candidates) and ask a classifier "is this the right road, yes or no?". So each row of data is one GPS point paired with one nearby road, with label 1 for the true road and 0 for the others.

## 1. Preprocessing

| Step | What we do | Why |
|---|---|---|
| Clean | Remove rows with broken coordinates or time going backwards | Bad rows would break distances and speeds. Our data had none, but new trips might |
| Convert to metres | Turn latitude and longitude into metres around each trip's centre | Distances and angles only make sense in metres |
| Find candidates | Keep up to K = 8 roads within 50 m of each GPS point (if none, take the nearest road within 200 m) | The true road is almost always close by |
| Make labels | Match each GPS point to one road of the true route, keeping the route order | The route file lists the roads driven, not which point was on which road |
| Make features | 11 numbers per row (distance, direction difference, speed and so on) | These are what the model learns from. None of them uses the true route |
| Clean training rows | Drop points more than 50 m from the route, keep at most 1,000 points per trip | Far-away points have unreliable labels, and long trips should not dominate |

| Partition | Trips | GPS points | Rows (point + candidate) | True road missing from candidates |
|---|---|---|---|---|
| Train | 80 | 208,417 | 1,103,327 | 11.42% |
| Train rows used for models | 80 | 59,664 | 369,705 (16% are true roads) | |
| Student test | 9 | 16,657 | 103,788 | 0.43% |

Student-test trip `00000000` has no `.arcs` file, so 9 of the 10 test trips are used. The train miss rate is higher because a few train trips drive off the mapped roads (trip 54 has 87% of its points more than 50 m from the route).

## 2. The k values

Three different things in this project are called k.

### Candidate K: how many nearby roads we keep per GPS point

We tried radius 20, 30, 50, 75 and 100 m and K from 2 to 15 on the train trips only. "Miss rate" is how often the true road is not in the candidate list. These rows use a 50 m radius.

| K | Miss rate | Candidates per point |
|---|---|---|
| 2 | 0.183 | 1.8 |
| 4 | 0.137 | 3.2 |
| 6 | 0.124 | 4.3 |
| **8 (chosen)** | **0.119** | **5.2** |
| 10 | 0.117 | 5.9 |
| 15 | 0.116 | 7.1 |

After K = 8 the miss rate hardly drops (0.119 to 0.116) while each point gets two more wrong roads to choose from. So we stopped at 8. Full grid: `results/stage2/candidate_radius_k_grid.csv`.

### KNN k: how many neighbours KNN looks at

We used k = 25 with distance weighting (closer neighbours get a bigger vote). With a small k such as 5, one noisy GPS point can decide the vote. With a very large k, rows from different situations get mixed in. 25 sits between the two, and distance weighting still lets the closest rows count most. We used this one setting and did not tune k, because KNN was not a contender: it is about 240 times slower to predict than the chosen model.

### Cross-validation k: number of folds

We use 5 folds with GroupKFold, grouped by trip. All points of one trip are in the same fold. GPS points one second apart are almost copies of each other, so if a trip were split across folds the model would be tested on near-copies of its training data and the score would look better than it really is.

## 3. Feature selection

Technique: rank the features by importance, then remove the weakest one at a time and keep the smallest set that still scores as well as the best.

1. A Random Forest with `criterion="entropy"` (information gain) is trained on all 11 features and gives each feature an importance score.
2. We remove the least important feature, retrain, and score with 5-fold grouped cross-validation (macro F1). We repeat until 3 features are left.
3. We keep the smallest feature set whose score is within 0.002 of the best.

| Feature | What it means | Importance |
|---|---|---|
| heading_diff | Angle between the direction the car moves and the direction of the road | 0.301 |
| dist_minus_min | How much farther this road is than the closest candidate | 0.233 |
| rank | 0 for the closest road, 1 for the next, and so on | 0.132 |
| nbr_dist | How far the 2 GPS points before and after are from this road | 0.088 |
| tproj | Where along the road the point falls (0 start, 1 end) | 0.071 |
| speed | Speed of the car in m/s | 0.060 |
| dist | Distance from the GPS point to the road in metres | 0.051 |
| arc_len | Length of the road piece | 0.038 |
| has_reverse | 1 if the road is two-way | 0.015 |
| n_cand | Number of candidate roads for this point | 0.010 |
| fallback | 1 if the road was found with the wider 200 m search | 0.000 |

| Features kept | Feature removed next | CV macro F1 |
|---|---|---|
| 11 | fallback | 0.9280 |
| 10 | n_cand | 0.9280 |
| **9** | has_reverse | **0.9290 (best)** |
| 8 | arc_len | 0.9263 |
| 7 | dist | 0.9249 |
| 6 | speed | 0.9242 |
| 5 | tproj | 0.9230 |
| 4 | nbr_dist | 0.9137 |
| 3 | rank | 0.9062 |

Result: 9 features are kept (heading_diff, dist_minus_min, rank, nbr_dist, tproj, speed, dist, arc_len, has_reverse). `fallback` and `n_cand` are dropped because removing them did not lower the score at all. Going down to 8 features drops below the 0.002 margin, so we stop at 9. The list is saved in `configs/features.json`, and every model uses it.

## 4. The models

The college asks for at least five models. We trained seven, from six algorithm families in the college's list. All use the same 9 features, the same rows and the same 5 grouped folds.

| Model (sklearn class) | Family | Settings | In simple words |
|---|---|---|---|
| Logistic Regression (`LogisticRegression`) | Linear | C=1.0, scaled inputs | Draws one straight line between right and wrong roads |
| Gaussian Naive Bayes (`GaussianNB`) | Probabilistic | defaults | Uses Bayes' rule and assumes features are independent |
| KNN (`KNeighborsClassifier`) | Instance-based | k=25, distance-weighted, scaled | Looks at the 25 most similar training rows and lets them vote |
| Decision Tree (`DecisionTreeClassifier`) | Tree | entropy, max depth 10, min 20 rows per leaf | Learns yes/no questions like "is the angle small?" |
| Random Forest (`RandomForestClassifier`) | Ensemble (bagging) | 200 trees, sqrt features per split, min 5 rows per leaf | Many trees on random samples, then an average |
| HistGradientBoosting (`HistGradientBoostingClassifier`) | Boosting | learning rate 0.1, up to 300 trees, 31 leaves, early stopping (stopped at 151 trees) | Adds small trees one by one, each fixing the mistakes of the ones before |
| MLP (`MLPClassifier`) | Neural network | hidden layers 64 and 32, relu, adam, early stopping (stopped at epoch 34) | A small neural network |

## 5. Accuracy of every model

CV = 5-fold grouped cross-validation on the 80 train trips. Test = the 9 student-test trips. Top-1 = for each GPS point, is the road with the highest score the true road. Log loss = how good the probabilities are (lower is better).

| Model | CV F1 | Train F1 | Test F1 | Test accuracy | Precision | Recall | Top-1 | Log loss | Train time (s) | Predict time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.9005 | 0.9017 | 0.8934 | 0.9427 | 0.821 | 0.821 | 0.8457 | 0.160 | 0.18 | 0.006 |
| Gaussian Naive Bayes | 0.8501 | 0.8507 | 0.8401 | 0.9013 | 0.638 | 0.884 | 0.8382 | 0.386 | 0.05 | 0.018 |
| KNN (k=25) | 0.9192 | 1.0000 | 0.9119 | 0.9536 | 0.872 | 0.831 | 0.8553 | 0.153 | 0.45 | 9.790 |
| Decision Tree | 0.9250 | 0.9333 | 0.9138 | 0.9541 | 0.865 | 0.844 | 0.8631 | 0.122 | 1.64 | 0.007 |
| Random Forest | 0.9285 | 0.9644 | 0.9163 | 0.9559 | 0.879 | 0.839 | 0.8659 | 0.108 | 8.57 | 0.130 |
| **HistGradientBoosting** | **0.9289** | 0.9411 | 0.9162 | 0.9553 | 0.865 | 0.853 | 0.8647 | **0.107** | 1.41 | 0.041 |
| MLP | 0.9258 | 0.9323 | 0.9152 | 0.9548 | 0.864 | 0.851 | 0.8598 | 0.109 | 18.97 | 0.031 |

We compare models on macro F1, not accuracy. Only 16% of rows are true roads, so a model that always says "wrong road" would already get 84% accuracy. Macro F1 gives the true-road class equal weight.

Two checks that the numbers are real: picking the closest road for every point gives Top-1 0.536, and the best possible Top-1 with our candidates is 0.996. Our models sit between the two. No feature uses the true route.

Plots: [CV F1](../results/plots/s3_cv_macro_f1.png), [test Top-1 and F1](../results/plots/s3_test_top1_f1.png), [log loss](../results/plots/s3_test_log_loss.png), [confusion matrices](../results/plots/s3_confusion_matrices.png), [feature selection](../results/plots/s3_feature_selection.png), [MLP history](../results/plots/s3_mlp_history.png).

## 6. Which model we chose and why

We chose **HistGradientBoostingClassifier**.

It has the best CV macro F1 (0.9289) and the lowest log loss (0.107). Random Forest is equally accurate (0.9285 CV, the difference is far smaller than the 0.009 variation between folds), so the tie was broken on four practical points:

- It overfits less. The gap between train F1 and CV F1 is 0.012, against 0.036 for Random Forest.
- Its probabilities are the best. The next phase (HMM with Viterbi) uses these probabilities to build one connected route, so good probabilities matter.
- The saved model is 0.56 MB. Random Forest is 178 MB, too large for GitHub.
- It predicts about three times faster than Random Forest, which helps the API.

Why the others lost: Naive Bayes assumes the features are independent, but several of them are all distances, so it becomes overconfident (precision 0.638, worst log loss). Logistic Regression can only draw a straight line and misses that direction matters more when the car is moving. KNN memorises the training data (train F1 1.0) and takes almost 10 s to predict. A single Decision Tree changes more from fold to fold than the ensembles. The MLP is as good as boosting but takes 13 times longer to train.

The selection note is also in `results/stage3/model_selection_remark.md`.
