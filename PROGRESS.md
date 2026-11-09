# Progress

## Phase 1 — technical setup and initial data inspection (2026-11-09)

- Completed: project folder and existing Git repository checked; Python 3.12 virtual environment created; official UCI workbook downloaded; initial inspection scripted and run.
- Actual results: 30,000 rows and 25 columns; 0 missing cells; 0 duplicated customer IDs; 23,364 non-default and 6,636 default records, a 22.12% default rate. Mean age 35.49 years; median age 34. Mean credit limit NT$167,484.32; median NT$140,000.
- Issues: the initial default Python lacked `venv`, so a separate Python 3.12 environment was installed. Some observed category codes require further documentation. The dataset is old and geographically limited.
- Technical milestones: Python environment and dependencies initialized; Pandas data ingestion, `head()` and `shape` verification; schema extraction and target distribution confirmed.
- Next: Phase 2 exploratory analysis, starting with category values and data quality, followed by plots and default rates. No model results yet.

## Phase 2 — initial exploratory analysis (2026-11-09)

- Completed: reviewed missingness and duplicate IDs; category counts and default rates; all six repayment-status months; credit limit groups, selected numerical ranges and correlations; generated a four-panel figure.
- Actual results: limit groups from <= NT$50,000 to > NT$200,000 had default rates of 31.79%, 25.80%, 19.48% and 14.73%. `PAY_0` code 1 had 3,688 cases with 33.95% default; code 2 had 2,667 cases with 69.14% default.
- Issues: some category meanings remain undocumented on the UCI page, rare repayment codes have small sample sizes, and 590 September bill amounts are negative. Values were not altered. The full dataset was explored before splitting; this must be disclosed in future evaluation.
- Learning: a group default rate divides defaults by the number of records in that group; an association does not prove a causal effect.
- Next: resolve data documentation as practical, then split before further label-guided model development. Baseline modelling is planned, not complete.

## Phase 3 — first baseline and validation (2026-11-09)

- Completed: stratified split into 18,000 training, 6,000 validation and 6,000 reserved test records; training-only preprocessing with logistic regression; validation evaluation at threshold 0.5.
- Validation results: TN 4,541; FP 132; FN 994; TP 333; precision 0.7161; recall 0.2509; ROC-AUC 0.7259; average precision 0.5001. The test set was not scored.
- Limitations: the full dataset was previously explored; repayment codes are treated numerically; category definitions and probability calibration need further work. No claim of real lending performance.
- Next: assess model improvements and calibration without tuning on the test set, then determine an evaluation plan that explicitly discloses previous full-dataset exploration.

## Phase 4 and 5 — initial model comparison and decision scenarios (2026-11-09)

- Completed: compared logistic regression, random forest and training-only 3-fold sigmoid-calibrated forest on validation; calculated ROC-AUC, average precision and Brier scores; generated calibration plot, fixed-model validation bootstrap intervals and impurity feature importance; compared four thresholds under explicitly hypothetical false-positive and false-negative costs.
- Actual validation results: AUCs 0.725912 / 0.789860 / 0.786664; Brier scores 0.145218 / 0.133710 / 0.134276 in the same model order. Random forest at threshold 0.1667 had recall 77.02% and flagged 44.90% of customers; at 0.50, recall was 34.29% and 11.10% were flagged.
- Limitations: test remains unscored; prior full-data exploration; calibration not shown to improve Brier score; sample bootstrap does not measure all uncertainty; sensitive fields and categorical coding need further scrutiny; no real cost or profit data.
- Next: investigate feature treatment and fairness, model and threshold stability under resampling and alternative costs, then finalize a plan for test scoring and independent validation. Expected-loss analysis remains planned.

## Further robustness and scenarios (2026-11-09)

- Completed: training-only three-fold stability check, validation comparison excluding SEX/EDUCATION/MARRIAGE, preliminary sex-group confusion metrics at 0.5, three hypothetical cost ratios and fully hypothetical PD/LGD/EAD arithmetic.
- Results: training-fold AUC 0.778959–0.787539; validation AUC 0.789860 with all features vs 0.786271 excluding those three. At 0.5, all-feature false-positive rates were 5.96% and 3.63% for SEX codes 1 and 2; removing the fields did not eliminate the difference.
- Limitations: one 3-fold split is not temporal stability; group analysis is descriptive, not a fairness/compliance determination; costs, LGD and EAD are assumptions. Test not scored, and prior full-data EDA must be disclosed.
- Next: lock model and threshold evaluation rules before scoring the held-out split; document test limitations and independently validate on newer or external data when available. Build the portfolio/report and dashboard after results are finalized.

## Held-out test and capacity-constrained review (2026-11-09)

- Wrote a protocol before evaluation: fixed random forest, excluded SEX/EDUCATION/MARRIAGE from features, kept original training split, fixed thresholds and 10%/20% review capacities.
- Same-dataset test: 6,000 records, 1,327 defaults; AUC 0.774123, average precision 0.549650, Brier 0.135837. At threshold 0.5, TP 461, FN 866, FP 237, TN 4,436.
- Fixed capacity: top 10% captured 415 observed defaults among 600 cases (31.27% of all test defaults); top 20% captured 678 among 1,200 (51.09%). Random selection expectations are 132.7 and 265.4, respectively; this is not an intervention or profit result.
- Prior full-data EDA limits independence; test group metrics are descriptive, not a fairness determination. The test set has now been scored and must not be reused as a fresh final test after changes.
- Next: assemble an evidence-based report, README and résumé/interview talking points; find truly independent or later data for external validation; build the planned dashboard if requested.

## Research report and local dashboard (2026-11-09)

- Drafted an English research report using the measured results and documented limitations. Created a local Streamlit demonstration with held-out metrics, capacity scenarios and predictions on public historical examples.
- The dashboard reconstructs the fixed final model in memory from training records; no separate serialized model file is supplied. It is intended for educational exploration, not production decisions.
- Remaining: review and refine the report, create truthful résumé/interview materials, independent validation, and any public deployment requested later.

## Dashboard access and language refinement (2026-11-09)

- Confirmed localhost service health and page returned HTTP 200; the previous in-app browser tab was no longer available. Opened a fresh tab and verified the rendered page.
- Standardized application content, metric descriptions, comparison labels and historical case columns to Chinese, using a restrained native theme.
- Added a double-click Windows launcher and verified service startup from a stopped state. Overview, capacity and historical prediction views passed application checks. Public hosting remains unconfigured.

## English workspace and VS Code access (2026-11-09)

- User clarified that the required display language is English. Replaced Chinese application content with a consistent English research workspace, sidebar navigation, validation comparison, review-capacity chart, historical predictions and in-app report/download.
- VS Code launched successfully and opened the project and app.py; Python/Jupyter extensions were present. Added project interpreter selection and a Streamlit debug configuration.
- Reproduced browser connection refusal when the local service was no longer listening; restarted it and verified page loading. A stopped local service must be started before browsing localhost. Four application views and the 20% capacity selection passed runtime checks.

## Light research workspace and capacity extension (2026-11-09)

- User chose a light professional workspace. Reorganized the English interface into five destinations and 15 secondary views, with consistent chart colours, adaptive metrics, source exploration, validation evidence, scenario controls and report downloads.
- Extended the fixed model's already-scored test diagnostics to a 1%–100% capacity curve and successive 5% marginal bands. Original 600/1,200-slot counts remained 415/678. The first four 300-case bands contained 224/191/137/126 historical defaults.
- Added explicit post-evaluation labelling; no model choice was retuned. The report supplement explains historical coverage and does not infer intervention profit.
- Workspace navigation, capacity totals and expected-loss interactions passed tests; desktop, narrow window and phone layouts were inspected.
- Independent finish review identified and verified fixes for clearable subnavigation and low comparator-line contrast. Final review disposition: ship within the inspected scope. All 15 subpages and calculation checks passed; real-runtime overview and capacity interactions were also checked. This does not constitute a complete accessibility audit.

## Portfolio dashboard and reproducibility audit (2026-11-09)

- Confirmed no saved estimator artifact or project-owned notebook in the workspace. The documented 20-feature model reconstructed from training data reproduced 6,000 saved test probabilities to absolute tolerance 1e-10 in local Python 3.12.
- Added ROC and precision–recall curves from the saved scores, candidate feature importance, and a synthetic-input simulator that explicitly reconstructs the fixed model only after submission. Removed row-level historical account display from the public demo.
- Added repository audit, deployment instructions and a CV draft. Added tests for model reconstruction, schema, all dashboard modules and simulator submission. Both test scripts and dependency checks passed locally.
- Added the requested restrained glass styling. Browser rendering was checked locally; cloud/Linux deployment and a full accessibility audit have not been verified. No external publishing performed.
