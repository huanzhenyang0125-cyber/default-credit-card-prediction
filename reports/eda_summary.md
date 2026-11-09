# Exploratory analysis — historical sample

The calculations in `eda_results.txt` describe all 30,000 records in the original UCI workbook. Run `python eda.py` from the project directory to regenerate the text and figure. They are descriptive observations, not causal effects or model performance.

## Findings

- There are 6,636 default cases (22.12%) and 23,364 non-default cases (77.88%). Always predicting non-default would achieve 77.88% accuracy on this sample while identifying no defaults.
- The groups with credit limits of at most NT$50,000, NT$50,000–100,000, NT$100,000–200,000, and over NT$200,000 have default rates of 31.79%, 25.80%, 19.48%, and 14.73%, respectively. Boundaries are upper-inclusive (e.g., NT$100,000 is in the second group). These are associations, not evidence that increasing credit limits changes default risk.
- For `PAY_0`, the most recent repayment code, code 1 has 3,688 records and a 33.95% default rate; code 2 has 2,667 records and a 69.14% default rate. Codes 5–8 each have at most 26 records, making their percentages unstable. Six months of repayment code counts and rates are in the text results.
- Pandas found no missing cells and no duplicate IDs. This does not rule out invalid values or other quality issues. Education codes 0, 5 and 6, marriage code 0, and repayment codes 0 and -2 occur but are not fully explained in the UCI page's code descriptions. Preserve them until their meaning is verified.
- There are 590 negative September bill amounts; a negative value is not automatically an error. Inspect their meaning rather than deleting or changing them without evidence. Large monetary values also warrant investigation.
- Selected Pearson correlations are included as rough numeric summaries; category codes are not necessarily continuous quantities. Correlation is not causation.

## Limits and future evaluation

The data describe Taiwan credit-card customers from 2005 and do not establish modern or cross-region performance, real approval outcomes, or profits. These plots and aggregates used **all** records before a train/test split, so a later holdout from these same records cannot be described as completely untouched by exploration. Split before further label-guided analysis or tuning; disclose this earlier exploration when reporting evaluation, and seek independent or later-period data for a stronger final assessment.
