# Repository and model audit

## Current assets

- The original UCI workbook is `data/raw/default of credit card clients.xls`. The recorded customer-ID split is `reports/data_split.csv`; it contains 18,000 training, 6,000 validation, and 6,000 test records.
- The final fixed model is specified in `risk_model.py` and `final_evaluation.py`: training-only StandardScaler plus random forest (100 trees, minimum leaf size 25, seed 42). There is **no saved fitted model artifact** and no project-owned notebook in the inspected workspace. No pickle or other serialized estimator is loaded.
- `reports/test_risk_scores.csv` contains recorded probabilities and outcomes for the original test split. `reports/final_test_metrics.csv` and the original research report document the prior results. The simulator reconstructs the pipeline only after the user submits the form.
- `tests/test_model_reproducibility.py` refits the fixed pipeline using only the original training split and checks all 6,000 predictions against the saved scores. This check passed in the local Python 3.12 environment at an absolute tolerance of 1e-10. This verifies local reproduction, not cross-platform model equivalence.

## Feature schema and prediction time

The final input has 20 ordered numeric fields: `LIMIT_BAL`, `AGE`, `PAY_0`, `PAY_2`–`PAY_6`, `BILL_AMT1`–`BILL_AMT6`, and `PAY_AMT1`–`PAY_AMT6`. ID, target, split, SEX, EDUCATION and MARRIAGE are excluded. The preprocessing pipeline standardizes those fields using training observations only.

According to the [official UCI descriptions](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), payment status, bill statements and previous payments refer to April–September 2005; the outcome is next-month default. Their use is temporally plausible when the prediction is made after those records are available. Actual availability, reporting delays and the target observation window must be confirmed in any real operating environment. Codes 0 and −2 are not fully described by that page. The app uses the original codes and does not infer their meaning.

The training pipeline avoids fitting preprocessing on validation or test data. However, exploratory analysis used the full sample before splitting, and later extended diagnostics use the already-scored test set. This limits independence and creates possible analysis-selection bias; no same-sample result should be represented as independent external validation.

## Remaining gaps

No fitted estimator artifact, SHAP analysis, external or temporal validation, independently verified real bank policy thresholds, or true intervention/profit outcomes are available. Do not claim those items are completed. The simulator uses a synthetic combination of training medians and hypothetical bands; observed training ranges are not a complete validity specification. Public deployment should be tested on the target Linux/Python environment before being described as verified.
