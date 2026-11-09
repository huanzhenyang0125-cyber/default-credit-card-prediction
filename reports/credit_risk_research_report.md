# Credit Risk Decision Lab
## An Explainable Credit Risk and Lending Decision Framework

**Status:** Student research portfolio draft, based on experiments completed on 9 November 2026.  
**Research question:** How can historical default predictions support transparent, risk-sensitive prioritisation when review capacity and costs are constrained?

## Abstract

Credit risk teams must identify potentially risky accounts while operating with limited analytical and review capacity. This study develops a reproducible student research workflow using the UCI Default of Credit Card Clients dataset, a historical sample of 30,000 Taiwanese credit-card accounts. A logistic regression baseline is compared with a random forest and a sigmoid-calibrated random forest. The final demonstration uses a random forest that excludes gender, education and marital status from its predictors. On a 6,000-record held-out portion of the same dataset, the model achieves ROC-AUC 0.774, average precision 0.550 and Brier score 0.136. A fixed-capacity simulation finds that the highest-risk 10% of records contains 415 of 1,327 observed defaults, compared with an expected 132.7 defaults under uniform random selection of the same number of records. This comparison evaluates historical ranking, not the effectiveness or profitability of intervention. Full-dataset exploration before splitting, the age and geographic concentration of the data, and the absence of observed intervention outcomes restrict the conclusions. The project illustrates how predictive modelling, probability assessment and operational constraints can be connected in a transparent educational analysis.

## 1. Introduction

A credit risk model estimates the likelihood that an account will experience an adverse outcome. A probability estimate alone does not determine an operational decision: a lender or risk team must also consider the cost of missed cases, the cost of unnecessary reviews and the number of cases staff can investigate. These decisions depend on business assumptions that are separate from a model's statistical performance.

This project examines two connected questions. First, how well can simple supervised models rank accounts by the next-month default outcome in this particular historical dataset? Second, what does the ranking imply for a hypothetical team able to review only a fixed fraction of accounts? The second question gives the project its fintech focus: it connects a data-driven score to a practical capacity constraint without treating a historical outcome as proof that an intervention would succeed.

The contribution is an auditable workflow combining a baseline model, calibration assessment, subgroup diagnostics, illustrative costs and a capacity-limited review exercise. The methods are established techniques; the study does not claim a novel algorithm. The results are intended for learning and portfolio discussion, not credit approval or production deployment.

## 2. Dataset and research setting

The analysis uses the original UCI Default of Credit Card Clients workbook [1], downloaded on 9 November 2026. It contains 30,000 observations, an identifier, 23 input variables and a binary indicator of default payment in the following month. According to the source documentation, repayment status, bill amounts and previous payments cover April through September 2005. The dataset describes Taiwanese credit-card clients; monetary variables are denominated in New Taiwan dollars (NTD).

The target equals one for default and zero for non-default. In the full sample, 6,636 records are labelled default and 23,364 non-default, corresponding to a 22.12% historical default rate. A classifier predicting non-default for every observation would therefore be correct on 77.88% of these records while finding none of the defaults. This illustrates why accuracy alone is inadequate for the task.

The initial inspection found no missing cells as interpreted by Pandas and no duplicate customer IDs. Those checks do not establish that every value is valid. Some observed category values are not fully explained on the UCI dataset page; 590 records have a negative September bill amount. Negative balances are not automatically errors, and no undocumented values were deleted or reinterpreted. Investigation of their meaning remains an open data-quality task.

The data are historical and geographically specific. They do not provide a current bank's application population, approval decisions, recoveries, exposure at default or intervention outcomes. No claim about current lending practice, a new population or the causal effect of an intervention follows from these observations.

## 3. Methodology

### 3.1 Data splitting and preprocessing

The records were split with a fixed seed and stratification by the outcome into 18,000 training observations, 6,000 validation observations and 6,000 held-out test observations. The original train/validation/test membership is recorded by ID for reproducibility. Test records were not used to fit the models or choose the final specification.

An important qualification is that initial descriptive exploration used the full dataset before this split. Consequently, the held-out rows were indirectly exposed through aggregate exploration, and the resulting test should not be characterised as an entirely untouched or external evaluation. A new dataset is needed for a stronger independent assessment, especially after further model or decision-rule changes.

Preprocessing was fitted inside each model pipeline using only the corresponding training data. The initial comparisons standardised numerical variables and one-hot encoded gender, education and marital status. Customer ID and the target were excluded from predictors. The final forest excludes all three demographic fields as a demonstration of feature sensitivity, motivated by the validation comparison. Excluding them does not guarantee fairness because other inputs can carry correlated information. Repayment codes are treated as numeric in this baseline; their coding and interpretation warrant further review.

### 3.2 Models and probability assessment

The first benchmark is logistic regression, a simple classification method that produces estimated probabilities. The second is a random forest with 100 trees, a minimum leaf size of 25 and a fixed random seed. A third model applies sigmoid calibration to a forest using three-fold cross-validation within the training set. These specifications were fixed examples rather than the outcome of a comprehensive hyperparameter search.

Discrimination is assessed with area under the receiver operating characteristic curve (ROC-AUC) and average precision. ROC-AUC describes ranking across thresholds; average precision summarises the precision–recall relationship and is not necessarily identical to trapezoidal PR-AUC. The Brier score is the mean squared difference between predicted probabilities and observed binary outcomes, with smaller values indicating a better overall score. Calibration curves compare observed event rates with mean predicted probabilities in groups.

The decision analysis uses an illustrative cutoff of 0.5 and a second cutoff of approximately 0.1667. The latter follows from a simplified scenario in which a false positive costs one unit, a false negative costs five units and correct classifications have zero cost, assuming well-calibrated probabilities and constant case-level costs. These units are not observed currency amounts; the assumptions are not a bank's documented cost structure.

### 3.3 Capacity-limited review

The final model was chosen using the validation findings and a preference for excluding the three named demographic fields. Before scoring the held-out sample, the protocol fixed this model and specified reviews of the top 10% and 20% of test records ranked by predicted risk. Equal scores are ordered deterministically by ID, which is not an input feature. The analysis counts historical defaults in each selected group and compares the count with the expected number under uniform random sampling of the same size.

This design is a ranking and resource-allocation illustration. It does not observe whether the accounts would actually have been reviewed, whether an intervention would change their outcomes, or whether a lender would gain or lose money. The random-selection comparator is an arithmetic expectation rather than an executed randomised trial.

## 4. Validation results

The first validation comparison used the 6,000-record validation set. Logistic regression reached ROC-AUC 0.725912, average precision 0.500144 and Brier score 0.145218. The uncalibrated forest reached 0.789860, 0.554455 and 0.133710, respectively. The sigmoid-calibrated forest reached 0.786664, 0.554311 and 0.134276.

These results support a narrower conclusion than “calibration improved the model”: on this validation set, the calibrated forest did not improve the overall Brier score relative to the uncalibrated forest. The calibration plot provides additional visual evidence, but neither a plot nor one score establishes that probabilities will remain reliable in another period or population. A fixed-model bootstrap of the validation observations gave the uncalibrated forest an approximate 95% ROC-AUC interval of 0.773939–0.803708. This interval does not include all uncertainty from retraining, model selection or distribution shift.

In a separate three-fold split within the training data, the same forest specification produced ROC-AUC values of 0.787539, 0.780228 and 0.778959. The close values are encouraging for these three folds, but they are not evidence of temporal or geographic stability. When SEX, EDUCATION and MARRIAGE were excluded, validation ROC-AUC was 0.786271, compared with 0.789860 when they were included. The difference is descriptive and was not tested for statistical significance.

## 5. Held-out evaluation and decision analysis

The fixed final forest was trained on the original 18,000 training observations and evaluated on 6,000 held-out observations, including 1,327 defaults. Its held-out ROC-AUC was 0.774123, average precision was 0.549650 and Brier score was 0.135837. These figures describe performance on a partition from the same historical source; the prior full-data exploration remains a limitation.

At a probability cutoff of 0.5, the model identified 461 defaults and missed 866, while flagging 237 non-default cases. Precision was 66.05% and recall was 34.74%. At the hypothetical 0.1667 cutoff, it identified 982 defaults and missed 345, while flagging 1,729 non-default cases. Precision fell to 36.22%, recall rose to 74.00% and 45.18% of accounts were flagged. This comparison makes the operational trade-off explicit: identifying more defaults also increases the review workload and the number of false positives.

Under the fixed-capacity ranking exercise, the top 600 records contained 415 observed defaults, representing 31.27% of all defaults in the held-out sample. Uniform random selection of 600 records would contain an expected 132.7 defaults at the sample's observed default rate. With 1,200 review slots, the model-ranked group contained 678 defaults, or 51.09% of all defaults, versus a random-selection expectation of 265.4. The first group had a 69.17% observed default rate, while the larger group had a 56.50% rate. More capacity therefore increased the number of historical defaults covered while adding a lower-risk set of cases on average.

The report also includes a simplified expected-loss illustration, not an estimate derived from unavailable bank data. For an assumed probability of default of 5%, loss given default of 45% and exposure at default of NTD 100,000, the simple product PD × LGD × EAD equals NTD 2,250. Recovery timing, discounting, exposure changes and other real-world factors are omitted. The example is arithmetic under assumptions, not a validated forecast.

## 6. Fairness, limitations and interpretation

Group-level diagnostics were computed using the dataset's recorded binary SEX codes, although SEX was excluded from the final model's inputs. At the 0.5 cutoff, the false-positive rate was 5.54% for code 1 and 4.77% for code 2; corresponding recall was 34.05% and 35.25%. These descriptive rates do not establish the cause of differences, whether any difference is statistically meaningful, or whether a system meets a legal or ethical standard. They also do not cover gender identities outside the dataset's coding.

Several limitations affect the practical interpretation. The original records date from 2005, concern a single geographic context and may reflect selection or historical lending practices not represented in the workbook. No prospective or external test was performed. Some category codes and unusual monetary values need further investigation. The prediction probabilities are not guaranteed to be calibrated for a future population, and cost assumptions and review capacities were chosen for demonstration. A production evaluation would require current data, clear outcome definitions, legally appropriate feature governance, stronger subgroup analysis and monitoring.

The same held-out test has now been scored. It should not be repeatedly reused as a fresh final test if model choices change. In addition, full-data exploration before the split must be disclosed whenever its results are presented. A later-period or genuinely independent dataset would offer stronger evidence about generalisation.

## 7. Conclusion

This project constructed a reproducible credit-risk modelling workflow and connected model scores to a constrained-review problem. The final forest achieved a held-out ROC-AUC of 0.774 on the same historical dataset. Under a simulated 10% review capacity, its highest-ranked records contained 415 of 1,327 observed defaults; this is a historical ranking result rather than evidence of prevented losses.

The exercise shows why risk modelling requires more than maximising one metric. Model discrimination, probability quality, error trade-offs, group outcomes and operational capacity each address a different decision question. The next research step is independent and preferably later-period validation, followed by more complete assessment of calibration, data quality, fairness and realistic business costs.

## Post-evaluation supplement: marginal review coverage

After the original held-out evaluation, the same fixed model and test cohort were used to describe review capacities from 1% to 100%. This extension is exploratory: capacities beyond the original 10% and 20% exercises were not prespecified before test inspection. No model parameters or final selection rules were changed.

At 5% capacity, 300 reviewed accounts contain 224 historical defaults, covering 16.88% of all 1,327 test defaults. At 10%, 600 accounts contain 415 defaults (31.27% coverage); at 20%, 1,200 accounts contain 678 (51.09%); and at 30%, 1,800 accounts contain 839 (63.23%). Coverage increases with resources, while the default concentration in the selected group changes.

Incremental bands make the resource trade-off more visible. The first 300 reviews contain 224 historical defaults, the next 300 contain 191, the next 300 contain 137, and the following 300 contain 126. Thus, doubling capacity from 600 to 1,200 adds 263 historical default cases to the review group, rather than doubling the original count. These are sample-specific outcomes, not a guaranteed monotonic pattern for every band in another population.

The operational interpretation is that a team can compare additional workload with additional historical case coverage. Any claim about the economic value of extra reviewers would additionally require actual review costs, intervention effectiveness, recovery and revenue information. No optimal staffing level, causal benefit or validated new threshold is selected from this curve. Supporting files are `review_capacity_curve.csv` and `review_capacity_marginal.csv`.

## References

[1] Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H. Dataset page: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients. Licensed CC BY 4.0. Accessed 9 November 2026.

## Reproducibility record

The project uses Python 3.12, fixed random seeds, the official workbook and scripts in the repository. The split, model-comparison tables, held-out metrics and capacity results are recorded in `reports/`. See `README.md` for execution instructions and `final_evaluation_protocol.md` for the evaluation plan fixed before test scoring. The final model is reconstructed by `final_evaluation.py`; this report does not claim an independent external validation or a deployable lending system.
