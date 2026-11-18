# Model selection remark (stage 3)

Each row is one GPS point paired with one candidate road arc, and the label says whether that arc is the true one. Models were scored with 5-fold GroupKFold CV on the 80 training trips (grouped by trip, so no trip appears in both train and validation folds) and then checked once on the 9 usable student-test trips. All models used the same 9 features picked by backward elimination.

## Results

| Model | CV macro F1 (std) | Train macro F1 | Test macro F1 | Test accuracy | Test Top-1 | Test log loss | Route continuity | Inference (s) |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.9005 (0.011) | 0.9017 | 0.8934 | 0.9427 | 0.8457 | 0.160 | 0.910 | 0.006 |
| Gaussian Naive Bayes | 0.8501 (0.009) | 0.8507 | 0.8401 | 0.9013 | 0.8382 | 0.386 | 0.904 | 0.018 |
| KNN (k=25, distance weighted) | 0.9192 (0.009) | 1.0000 | 0.9119 | 0.9536 | 0.8553 | 0.153 | 0.918 | 9.790 |
| Decision Tree (entropy) | 0.9250 (0.010) | 0.9333 | 0.9138 | 0.9541 | 0.8631 | 0.122 | 0.921 | 0.007 |
| Random Forest | 0.9285 (0.009) | 0.9644 | 0.9163 | 0.9559 | 0.8659 | 0.108 | 0.926 | 0.130 |
| Gradient Boosting | 0.9289 (0.009) | 0.9411 | 0.9162 | 0.9553 | 0.8647 | 0.107 | 0.927 | 0.041 |
| MLP (64, 32) | 0.9258 (0.008) | 0.9323 | 0.9152 | 0.9548 | 0.8598 | 0.109 | 0.927 | 0.031 |

The reference labels give a route continuity of 0.983, which is the ceiling for this measure.

## Choice: Gradient Boosting

Gradient Boosting has the highest CV macro F1 (0.9289) and the lowest test log loss (0.107). Log loss matters more than usual here because the HMM in stage 5 uses the classifier probability as its emission score, so a model with better calibrated probabilities gives Viterbi better input. Random Forest is within 0.0004 on CV F1 and 0.0003 on test Top-1, which is inside one fold standard deviation, so on accuracy alone the two are tied. Gradient Boosting wins the tie on three practical points. Its train/CV gap is 0.012 against 0.036 for Random Forest, so it overfits less. Its saved model is 0.56 MB against 178 MB, which matters for loading it inside the FastAPI backend. It predicts about three times faster.

The other models lose for clear reasons. Naive Bayes assumes the features are independent, but distance, rank and dist_minus_min are strongly correlated, and its precision drops to 0.64. Logistic Regression draws one linear boundary and cannot capture the interaction between heading difference and speed (heading is unreliable when the vehicle is nearly stopped). KNN memorises the training set (train F1 1.0) and takes almost 10 s to score the test set, which is too slow for an API. The single Decision Tree is close but less stable than the ensembles built from it. The MLP is competitive but trains about 13 times slower and needs feature scaling, with no accuracy gain.

Stage 4 will tune Gradient Boosting with Optuna (TPE sampler) using the same grouped CV.
