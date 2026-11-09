from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

root = Path(__file__).resolve().parent
data = pd.read_excel(root / "data/raw/default of credit card clients.xls", header=1, engine="calamine")
split = pd.read_csv(root / "reports/data_split.csv")
assert data.ID.is_unique and split.ID.is_unique and set(data.ID) == set(split.ID)
data = data.merge(split, on="ID", validate="one_to_one")
assert data['split'].value_counts().to_dict() == {'train': 18000, 'validation': 6000, 'test': 6000}
target = "default payment next month"
features = data.drop(columns=["ID", target, "split"])
y = data[target]
train = data["split"] == "train"
valid = data["split"] == "validation"
categorical = ["SEX", "EDUCATION", "MARRIAGE"]
numeric = [col for col in features if col not in categorical]

def make_pipeline(estimator):
    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("preprocess", preprocessing), ("model", estimator)])

logistic = make_pipeline(LogisticRegression(max_iter=1000, random_state=42))
forest = make_pipeline(RandomForestClassifier(n_estimators=100, min_samples_leaf=25, random_state=42, n_jobs=-1))
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
models = {
    "Logistic regression": logistic,
    "Random forest": forest,
    "Calibrated forest (3-fold sigmoid)": CalibratedClassifierCV(estimator=clone(forest), method="sigmoid", cv=cv, n_jobs=1),
}
probabilities = {}
metrics = []
forest_importance = None
print("Training on 18,000 records; evaluating on 6,000 validation records. Test set remains reserved.", flush=True)
for name, model in models.items():
    print(f"Fitting {name}...", flush=True)
    model.fit(features.loc[train], y.loc[train])
    if name == "Random forest":
        forest_importance = pd.DataFrame({"feature": model.named_steps["preprocess"].get_feature_names_out(),
                                          "importance": model.named_steps["model"].feature_importances_})
    p = model.predict_proba(features.loc[valid])[:, 1]
    probabilities[name] = p
    metrics.append({"model": name, "roc_auc": roc_auc_score(y.loc[valid], p),
                    "average_precision": average_precision_score(y.loc[valid], p),
                    "brier": brier_score_loss(y.loc[valid], p)})

metrics = pd.DataFrame(metrics)
metrics.to_csv(root / "reports/model_comparison.csv", index=False, float_format="%.6f")
forest_importance.sort_values("importance", ascending=False).to_csv(root / "reports/forest_importance.csv", index=False, float_format="%.6f")
print("\nValidation model comparison:\n", metrics.to_string(index=False), flush=True)

# Paired validation bootstrap reflects sampling variation with models held fixed.
rng = np.random.default_rng(42)
y_valid = y.loc[valid].to_numpy()
auc_samples = []
for _ in range(500):
    indices = rng.integers(0, len(y_valid), len(y_valid))
    if len(np.unique(y_valid[indices])) < 2:
        continue
    auc_samples.append({name: roc_auc_score(y_valid[indices], p[indices]) for name, p in probabilities.items()})
intervals = pd.DataFrame(auc_samples).quantile([0.025, 0.975]).T
intervals.columns = ["auc_lower_95", "auc_upper_95"]
intervals.to_csv(root / "reports/validation_bootstrap.csv", float_format="%.6f")
print("\nFixed-model validation bootstrap intervals (500 resamples):\n", intervals.to_string(), flush=True)

# Example intervention cost units only: false positive 1, false negative 5.
thresholds = [0.10, 1 / 6, 0.25, 0.50]
rows = []
actual = y.loc[valid]
for name, p in probabilities.items():
    for threshold in thresholds:
        pred = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(actual, pred, labels=[0, 1]).ravel()
        rows.append({"model": name, "threshold": threshold, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                     "precision": precision_score(actual, pred, zero_division=0),
                     "recall": recall_score(actual, pred, zero_division=0),
                     "flagged_fraction": pred.mean(), "hypothetical_cost_units": fp + 5 * fn})
decisions = pd.DataFrame(rows)
decisions.to_csv(root / "reports/threshold_scenarios.csv", index=False, float_format="%.6f")
print("\nThreshold scenarios (false positive = 1 unit, false negative = 5 units):\n", decisions.to_string(index=False), flush=True)

fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
for name, p in probabilities.items():
    observed, predicted = calibration_curve(actual, p, n_bins=10, strategy="quantile")
    ax.plot(predicted, observed, marker="o", label=name)
ax.set(xlabel="Mean predicted probability", ylabel="Observed default rate", title="Validation probability calibration")
ax.legend(fontsize=8)
fig.savefig(root / "figures/calibration.png", dpi=150)
plt.close(fig)
print("\nSaved comparison, threshold scenarios, and calibration figure. No test scoring performed.", flush=True)
