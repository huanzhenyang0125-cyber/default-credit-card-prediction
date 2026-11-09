# Credit Risk Decision Lab 💳

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Dataset: UCI ML Repo](https://img.shields.io/badge/Dataset-UCI%20Credit%20Card%20(350)-blue.svg)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(15%2F15)-brightgreen.svg)](tests/)

> **A reproducible credit-risk machine learning framework connecting default probability estimation with resource-constrained account review and operational decisioning.**

---

## Table of Contents

1. [Executive Summary & Background](#executive-summary--background)
2. [Research Question & Operational Framing](#research-question--operational-framing)
3. [Dataset & Feature Schema](#dataset--feature-schema)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Data Preprocessing & Fairness Compliance](#data-preprocessing--fairness-compliance)
6. [Machine Learning Models & Validation](#machine-learning-models--validation)
7. [Held-Out Test Set Performance](#held-out-test-set-performance)
8. [Capacity-Constrained Account Review](#capacity-constrained-account-review)
9. [Interactive Research Dashboard](#interactive-research-dashboard)
10. [Repository Structure](#repository-structure)
11. [Quickstart & Reproducibility](#quickstart--reproducibility)
12. [Scientific Disclosures & Limitations](#scientific-disclosures--limitations)
13. [References & Attribution](#references--attribution)

---

## Executive Summary & Background

In 2005–2006, card-issuing banks in Taiwan experienced a sharp expansion in cash-card and consumer credit lines followed by an unprecedented default crisis. As consumer delinquency surged, banks were confronted with two fundamental operational challenges:

1. **Risk Discrimination vs. Calibration**: A model that ranks credit applicants accurately (high ROC-AUC) is not necessarily calibrated for actual loss prediction (accurate Expected Loss $EL = PD \times LGD \times EAD$).
2. **Operational Queue Constraints**: In realistic risk operations, human review desks and intervention teams operate under strictly bounded daily capacity (e.g., manual audit queues of 10%–20% of accounts), meaning static classification cutoffs (such as threshold $0.5$) fail to translate to actionable operations.

**Credit Risk Decision Lab** addresses these challenges by developing a rigorous, audit-ready risk scoring pipeline using the benchmark UCI Credit Card Default dataset (30,000 accounts). It evaluates model discrimination, probability calibration, ethical feature exclusion, and operational queue yields.

---

## Research Question & Operational Framing

```
                                  Operational Review Funnel
                                  
   All Accounts (30,000) ────► Preprocessing Pipeline ────► Probability Scoring (Random Forest)
                               (Train-only Standardizer)           │
                                                                   ▼
                         ┌────────────────────────────── Ranked Risk Queue ──────────────────────────────┐
                         │                                                                              │
                         ▼                                                                              ▼
               Top 10% Highest Risk (600 Slots)                               Top 20% Highest Risk (1,200 Slots)
               Captures 415 Historical Defaults (31.27%)                      Captures 678 Historical Defaults (51.09%)
               [3.13x Lift over Random Audit]                                 [2.55x Lift over Random Audit]
```

* **Core Question**: How effectively can calibrated machine learning default probabilities prioritize account interventions when manual underwriting/investigation resources are limited?
* **Methodological Scope**: Rather than treating default risk as an abstract academic classification benchmark, this repository evaluates models through **discrimination (ROC-AUC)**, **calibration (Brier Score)**, **decision thresholds**, and **marginal capacity yield curves**.

---

## Dataset & Feature Schema

The dataset comes from the **UCI Machine Learning Repository** ([Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)), containing 30,000 customer records from April to September 2005 in Taiwan.

### Feature Dictionary

| Attribute | Type | Description | Operational Role |
|:---|:---|:---|:---|
| `LIMIT_BAL` | Numeric | Credit line amount in NT dollars (individual + supplementary) | Core Exposure |
| `SEX` | Categorical | Gender (1=Male, 2=Female) | **Excluded for Fairness** |
| `EDUCATION` | Categorical | Education level (1=Grad school, 2=University, 3=High school, 4=Others) | **Excluded for Fairness** |
| `MARRIAGE` | Categorical | Marital status (1=Married, 2=Single, 3=Others) | **Excluded for Fairness** |
| `AGE` | Numeric | Age in years | Demographic Control |
| `PAY_0` – `PAY_6` | Ordinal | Repayment status from September 2005 (`PAY_0`) back to April 2005 (`PAY_6`). Scale: `-1`=Pay duly, `1`–`8`=Payment delay in months, `9`=Delay ≥9 months | **Primary Signal** |
| `BILL_AMT1` – `BILL_AMT6` | Numeric | Statement bill amount (NT dollars) from September (`BILL_AMT1`) to April 2005 (`BILL_AMT6`) | Historical Balance |
| `PAY_AMT1` – `PAY_AMT6` | Numeric | Amount of previous payment (NT dollars) from September (`PAY_AMT1`) to April 2005 (`PAY_AMT6`) | Repayment Velocity |
| `default payment next month` | Binary | Target indicator: `1` = Defaulted in October 2005; `0` = Paid duly | Ground Truth Target |

*Note: Undocumented codes `0` and `-2` present in the raw data are preserved as observed historical records without unverified heuristic modifications.*

---

## Exploratory Data Analysis (EDA)

![EDA Overview](figures/eda_overview.png)

Key observations from the exploratory phase:
* **Class Imbalance**: The dataset contains 23,364 non-default accounts (77.88%) and 6,636 default accounts (**22.12% default rate**).
* **Payment Delinquency Impact**: `PAY_0` (September repayment status) demonstrates the steepest gradient: accounts with status `0` or `-1` have default rates $< 15\%$, while accounts with status `1` exhibit a **33.95%** default rate, and status `2` (two months overdue) spikes to **69.14%**.
* **Credit Limit Inverse Relationship**: Lower credit limits ($\le \text{NT}\$50,000$) show an observed default rate of **31.79%**, whereas credit lines $> \text{NT}\$200,000$ exhibit a **14.73%** default rate.

---

## Data Preprocessing & Fairness Compliance

1. **Stratified Split**:
   - Total sample: 30,000 records.
   - **Training Set (60%)**: 18,000 records (3,982 defaults, 22.12%).
   - **Validation Set (20%)**: 6,000 records (1,327 defaults, 22.12%).
   - **Held-Out Test Set (20%)**: 6,000 records (1,327 defaults, 22.12%).
2. **Leakage-Free Standard Scaling**:
   - `StandardScaler` fitted strictly on the 18,000 training observations and applied out-of-fold to validation and test records.
3. **Fairness & Regulatory Feature Selection**:
   - Consistent with fair-lending regulatory standards (such as the Equal Credit Opportunity Act principles), protected demographic attributes (`SEX`, `EDUCATION`, and `MARRIAGE`) were **deliberately excluded** from the final 20-feature modeling feature vector.
   - Validation experiments confirmed that removing these features preserved predictive discrimination (AUC 0.7899 with demographics vs. 0.7863 without) while preventing disparate demographic scoring.

---

## Machine Learning Models & Validation

We evaluated candidate pipelines across cross-validation and bootstrap resampling:

1. **Baseline Model**: L2-regularized Logistic Regression ($C=1.0$) with training-only scaling.
2. **Challenger Model**: Random Forest (100 estimators, minimum leaf size 25, seed 42).
3. **Calibrated Model**: 3-fold Out-of-Fold Sigmoid Calibrated Random Forest.

![Model Discrimination and Calibration](figures/report_discrimination.png)

### Validation Diagnostics

| Model Pipeline | Val ROC-AUC | Val Average Precision | Val Brier Score | Notes |
|:---|:---:|:---:|:---:|:---|
| **Logistic Regression (Baseline)** | 0.7259 | 0.5001 | 0.1452 | Fast linear baseline |
| **Random Forest (Final Chosen)** | **0.7899** | **0.5642** | **0.1337** | Superior ranking & discrimination |
| **Calibrated Random Forest** | 0.7867 | 0.5598 | 0.1343 | Comparable Brier score |

* **Internal Stability**: 3-fold cross-validation on the training set demonstrated tight AUC bounds ($0.7789$ – $0.7875$), verifying robustness against data partitioning noise.
* **Bootstrap Confidence Intervals**: 1,000 validation bootstrap draws for the Random Forest produced a 95% CI of $[0.7758, 0.8037]$ for ROC-AUC and $[0.1283, 0.1392]$ for Brier score.

---

## Held-Out Test Set Performance

The pre-registered final 20-feature Random Forest model was scored on the uncompromised 6,000 held-out test records:

| Test Metric | Measured Value | Benchmark Interpretation |
|:---|:---:|:---|
| **ROC-AUC** | **0.7741** | Strong ranking discrimination across operational thresholds |
| **Average Precision (PR-AUC)** | **0.5497** | High positive-class precision relative to 22.12% base rate |
| **Brier Score** | **0.1358** | Low mean squared probability error |
| **Accuracy (at threshold 0.50)** | **81.62%** | High baseline correctness |
| **Test Sample Size** | 6,000 records | 1,327 actual defaults / 4,673 non-defaults |

### Test Confusion Matrix (Fixed Threshold = 0.50)

| | Predicted Non-Default (`0`) | Predicted Default (`1`) | Total Observed |
|:---|:---:|:---:|:---:|
| **Actual Non-Default (`0`)** | 4,436 (TN) | 237 (FP) | 4,673 |
| **Actual Default (`1`)** | 866 (FN) | 461 (TP) | 1,327 |
| **Total Predicted** | 5,302 | 698 | 6,000 |

---

## Capacity-Constrained Account Review

In real credit operations, banks do not take actions using arbitrary cutoffs like $p=0.5$. Instead, operations teams have a fixed budget of manual investigation hours. We evaluated review capacity queues:

![Capacity Analysis Curve](figures/report_capacity.png)

### Review Capacity Benchmarks (Test Set: 6,000 Accounts)

| Review Capacity | Available Slots | Historical Defaults Captured | Share of All Defaults | Lift vs. Random |
|:---:|:---:|:---:|:---:|:---:|
| **Top 5%** | 300 | 224 | 16.88% | **3.38x** |
| **Top 10%** | **600** | **415** | **31.27%** | **3.13x** |
| **Top 15%** | 900 | 552 | 41.60% | **2.77x** |
| **Top 20%** | **1,200** | **678** | **51.09%** | **2.55x** |

* **Operational Insight**: By auditing just **10% of accounts**, risk teams capture nearly a third (**31.27%**) of all defaults. Auditing **20%** captures over half (**51.09%**), representing more than double the discovery rate of random audit policies.

---

## Interactive Research Dashboard

The project features a **Streamlit** research workspace with a custom clean visual design.

![Dashboard Preview](figures/dashboard_portfolio_preview.png)

### Dashboard Features

1. **Data Explorer**: Distribution plots, delinquency status distributions, and data quality checks.
2. **Model Validation**: Side-by-side discrimination curves, Brier reliability curves, and feature importance rankings.
3. **Decision Lab & Capacity Analysis**: Interactive slider to inspect 1%–100% review queues and compute marginal yield per 300-account tier.
4. **Expected Loss & Cost Sensitivity**: Custom false-positive/false-negative cost ratios and assumed LGD/EAD parameter scenarios.
5. **Interactive Risk Simulator**: Submit synthetic applicant parameters to compute real-time default probability via the exact in-memory pipeline.

---

## Repository Structure

```
Credit-Risk-Decision-Lab/
├── app.py                         # Streamlit interactive research dashboard
├── risk_model.py                  # Core modeling pipeline & feature schema definitions
├── read_data.py                   # Data ingestion and verification script
├── eda.py                         # Exploratory data analysis & figure generator
├── baseline.py                    # Logistic regression baseline validation
├── model_validation.py            # Random forest & calibration experiments
├── robustness.py                  # Stability, bootstrap CI & group diagnostic suite
├── final_evaluation.py            # Protocol-controlled held-out test scoring
├── capacity_analysis.py           # Extended 1%-100% capacity curve calculations
│
├── data/
│   └── raw/                       # Official UCI Excel workbook
│
├── figures/                       # High-resolution generated charts
│   ├── eda_overview.png
│   ├── report_discrimination.png
│   ├── report_capacity.png
│   ├── calibration.png
│   └── dashboard_portfolio_preview.png
│
├── reports/                       # Formal research documentation
│   ├── credit_risk_research_report.md
│   ├── Credit_Risk_Research_Report.pdf
│   ├── final_evaluation_summary.md
│   ├── repository_audit.md        # Leakage, schema & reproducibility audit
│   └── cv_project_entry.md        # Verified résumé & interview talking points
│
├── styles/                        # Custom CSS workspace styling
├── tests/                         # Automated test suite
│   ├── test_model_reproducibility.py  # Verifies 6,000 predictions match to 1e-10
│   └── test_workspace.py              # Tests all 15 dashboard views & calculations
│
├── DEPLOYMENT.md                  # Streamlit Cloud & production guide
├── requirements.txt               # Strict version-pinned dependencies
└── README.md                      # Project documentation
```

---

## Quickstart & Reproducibility

### 1. Prerequisites
* Python 3.12
* Git

### 2. Setup Environment
```bash
# Clone the repository
git clone https://github.com/huanzhenyang0125-cyber/default-credit-card-prediction.git
cd default-credit-card-prediction

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
# Run mathematical reproducibility check
python tests/test_model_reproducibility.py

# Run workspace and application test suite
python tests/test_workspace.py
```

### 4. Launch Research Dashboard
```bash
# Launch Streamlit app
streamlit run app.py
```
*On Windows, you can also double-click `Open dashboard.cmd` to start automatically on `http://localhost:8501`.*

---

## Scientific Disclosures & Limitations

In compliance with academic rigor and fair reporting standards:
1. **Full-Sample Pre-exploration**: An exploratory analysis of descriptive distributions was conducted on the complete dataset prior to partitioning. All subsequent model comparisons adhered strictly to training-only fitting.
2. **Historical vs. Live Operational Data**: The UCI dataset reflects Taiwan market dynamics from 2005. Economic regimes, consumer credit products, and inflation have evolved significantly; these findings illustrate methodology rather than production underwriting policies.
3. **Synthetic Loss Scenarios**: The LGD and EAD figures used in the Decision Lab represent hypothetical stress parameters for sensitivity testing rather than empirical bank portfolio recoveries.
4. **Deterministic Reproducibility**: The model pipeline has been verified to reproduce all 6,000 held-out probability scores with zero numerical drift ($\text{tolerance} < 10^{-10}$) within local Python 3.12 environments.

---

## References & Attribution

1. **Yeh, I. C., & Lien, C. H. (2009)**. *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients*. Expert Systems with Applications, 36(2), 2473–2480.
2. **UCI Machine Learning Repository**: Yeh, I. (2009). *Default of Credit Card Clients*. [https://doi.org/10.24432/C55S3H](https://doi.org/10.24432/C55S3H).
3. **Reference Benchmarks**: [Roberto Franceschi's Default Credit Card Prediction](https://github.com/robertofranceschi/default-credit-card-prediction) was reviewed as a structural benchmark for empirical reporting. All pipelines, experimental code, and decisioning frameworks in this repository were independently formulated.

---

**Author**: Huanzhen Yang ([@huanzhenyang0125-cyber](https://github.com/huanzhenyang0125-cyber))  
**License**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
