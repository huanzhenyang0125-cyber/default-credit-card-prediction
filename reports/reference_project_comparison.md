# Reference project and independent direction

Reviewed on 9 November 2026:
- Repository and README: https://github.com/robertofranceschi/Default-Credit-Card-Prediction
- Implementation inspected statically: https://github.com/robertofranceschi/Default-Credit-Card-Prediction/blob/master/code.ipynb

This comparison is based on the published README and notebook, not a fresh reproduction of the reference experiments. No source code, figures or prose were copied into this project.

## Reference strengths

The reference offers a detailed sequence from exploratory plots through preprocessing and mathematical explanations to model comparison. It compares logistic regression, random forest and support-vector approaches; studies PCA, SMOTE and Cluster Centroids; and discusses hyperparameter search with cross-validation. Its report-style README, notebooks and HTML export are useful presentation examples.

## Differences in research design

| Dimension | Reference emphasis | Credit Risk Decision Lab |
|---|---|---|
| Main question | Compare classification algorithms and preprocessing choices | Relate risk ranking and probability estimates to constrained review and assumed error costs |
| Data handling | Describes removing 399 rows with undocumented demographic codes and modifying repayment encodings | Retains 30,000 records; preserves uncertain codes; excludes selected demographic fields from the final predictor |
| Model development | Wider comparison of algorithms, PCA, sampling and parameter grids | Simpler fixed baselines; calibration checks, training-only cross-validation and validation bootstrap |
| Evaluation | Primarily accuracy, precision, recall and F1 | ROC-AUC, average precision, Brier score, classification errors, group rates and capacity coverage |
| Operational analysis | Classification experiments are the main reported output | Illustrative cost ratios and top-10%/top-20% review capacity, with no intervention-profit claim |
| Delivery | Report-oriented README, notebooks and HTML | Runnable scripts, recorded evaluation protocol, English report and interactive dashboard |

These differences are choices about scope, not evidence of originality across the literature. The current project has not completed the reference's systematic hyperparameter experiments, and neither project should be assumed superior without a common data preparation, split, metric and evaluation budget.

## Practices to assess critically

The reference notebook's stored cell order places MinMaxScaler fitting to the full `data` object before the train/test split. This differs from the README's training-only scaling explanation and presents a preprocessing leakage concern if run in that order. Its PCA is fitted to training/validation records before downstream cross-validation, which also warrants inspection of fold boundaries. These observations are a static implementation review, not a claim that every published result has been reproduced or invalidated.

This project's pipelines fit transformations within each relevant training fold, but its own full-dataset EDA occurred before splitting and remains a disclosed limitation. Removing unknown categories or transforming repayment codes should require documented domain rationale rather than replication of another project's choices.

## Borrowed presentation direction

Use a concise README with a visible research question, measured result tables, a transparent methodology, a reproducible run order and explicit limitations. Link to the full report and underlying result files. Strengthen visual analysis and mathematical explanations when they serve the research question, rather than adding methods solely to increase the method count.

## Planned extensions

Now completed as post-evaluation exploration: a 1%–100% review-capacity curve, incremental coverage in successive 300-case bands, interactive assumed-cost and expected-loss controls, and updated report documentation. The expanded grid was not prespecified before test inspection and is not fresh confirmatory evidence. Capacity uncertainty and workload-constrained subgroup analysis remain planned. Any new tuning or model selection requires new evaluation data. Historical ranking does not establish intervention effectiveness or profit.
