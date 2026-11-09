# Baseline logistic regression — validation only

Using random seed 42 and stratified random splits, 18,000 records were used for training and 6,000 for validation. The remaining 6,000 records are reserved for later testing; they were not scored in this run. The record allocation is in `data_split.csv`. All records had previously been included in EDA, so this same-dataset holdout has already been indirectly explored; disclose this when reporting future results.

The model excludes ID and the target from inputs. It standardizes numeric features and one-hot encodes SEX, EDUCATION and MARRIAGE inside a pipeline fitted only on training data. Repayment codes are treated as numeric in this simple baseline; their coding and this assumption require further review. The classifier is logistic regression with default settings and no threshold optimization. This is an initial benchmark, not a calibrated or production-ready credit score.

At a 0.5 probability threshold, the validation confusion matrix is `[[4541, 132], [994, 333]]`: 4,541 true negatives, 132 false positives, 994 false negatives, and 333 true positives. Precision is 0.7161; recall is 0.2509; ROC-AUC is 0.7259; average precision is 0.5001. Average precision is a precision–recall summary; it is not necessarily identical to trapezoidal PR-AUC.

The threshold captures about one quarter of observed validation defaults, so the remaining approximately three quarters are missed. The threshold is illustrative, not an approved risk policy. No test-set metric, calibration claim, profit or causal effect is reported.
