# Final held-out evaluation protocol

Written before computing held-out metrics. This is a student demonstration, not a production policy.

- Model: random forest with 100 trees, `min_samples_leaf=25`, seed 42. Exclude ID, SEX, EDUCATION and MARRIAGE from inputs; retain SEX only for descriptive group evaluation. This feature choice is motivated by the earlier validation comparison, not proof of fairness.
- Fit the same training-only preprocessing and estimator on the original 18,000 training rows. Do not fit on the validation or test rows.
- Evaluate once on the 6,000 reserved test rows: ROC-AUC, average precision, Brier score, and confusion counts/precision/recall at threshold 0.5 and 1/6. Threshold 1/6 comes from the previously stated hypothetical FP cost 1 and FN cost 5, zero cost for correct outcomes, and calibrated-probability assumptions; it is not an approval or intervention policy.
- Capacity experiment: rank test rows by predicted risk and inspect the top 10% and 20%. Report number flagged, actual defaults captured, precision and recall; break score ties deterministically by ID only for ranking. ID is not a model feature. This simulates allocation of fixed review slots, with no claim that intervention prevents default or yields profit.
- Describe test results for the two recorded SEX codes at threshold 0.5; include counts, false-positive rates and recall. No fairness or legal compliance conclusion follows from this limited diagnostic.
- Report model and threshold results, including the capacity exercise, as exploratory because earlier EDA used the complete dataset before the split, and model choice was informed by validation. Same-dataset results do not establish external or temporal generalization.
- No model or threshold tuning based on test results. Future changes need a fresh evaluation dataset.
