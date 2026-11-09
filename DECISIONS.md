# Decisions

- Use the official UCI original workbook; preserve the downloaded archive and workbook as raw data, and document attribution and retrieval date in README.
- Use Python 3.12 in an isolated `.venv` because the initially discovered Python installation lacked the standard `venv` module. The virtual environment is excluded from Git.
- Use Pandas and the `calamine` Excel engine to read the original `.xls`; `header=1` selects the second workbook row as field names.
- Start with a short runnable script and descriptive checks; notebooks, plots and modelling are planned later, not represented as completed work.
- `ID` identifies records and is excluded from model inputs. A fixed stratified 60/20/20 split is recorded in `reports/data_split.csv`; preprocessing is fitted inside the model pipeline using training records only.
- Do not infer undocumented payment or category codes without further verification, and do not interpret association as causation.
- The first EDA used all records before splitting. Treat a future same-dataset test set as previously explored and disclose the limitation; an independent or later-period test is needed for stronger external validation.
- The first baseline uses logistic regression, one-hot encodes demographic categories, and treats repayment codes as numeric; the latter is a simplifying assumption to revisit. A 0.5 threshold is illustrative, not optimized or a lending policy.
- Model comparison uses a fixed random forest specification and a separate training-only three-fold sigmoid calibration. Fixed-model validation bootstrap does not account for retraining or selection. Test records remain unscored.
- Threshold experiment uses hypothetical FP cost 1 and FN cost 5, with TP/TN zero; these are abstract units, not actual monetary losses or approval profits. The validation set is reused for model and scenario comparison, so conclusions are exploratory.
- The fixed-model random forest is additionally checked with training-only three-fold cross-validation. A comparison without SEX, EDUCATION and MARRIAGE and descriptive SEX-group rates are exploratory; removing attributes does not establish fairness.
- Expected-loss examples use entirely assumed PD, LGD and EAD, and omit recovery timing, discounting and other real-world details. They are not individual predictions.
- Before test scoring, recorded `reports/final_evaluation_protocol.md` selecting a fixed forest without SEX/EDUCATION/MARRIAGE, two illustrative thresholds and top 10%/20% review capacity. The previously held-out test is now consumed; future changes need new evaluation data.
