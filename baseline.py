from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

root = Path(__file__).resolve().parent
data = pd.read_excel(root / "data/raw/default of credit card clients.xls", header=1, engine="calamine")
target = "default payment next month"
features = data.drop(columns=["ID", target])
labels = data[target]

# Hold out test records before fitting any transformation or model.
train_val, test = train_test_split(data.index, test_size=0.2, random_state=42, stratify=labels)
train, validation = train_test_split(train_val, test_size=0.25, random_state=42, stratify=labels.loc[train_val])
assert len(set(train) & set(validation)) == len(set(train) & set(test)) == len(set(validation) & set(test)) == 0
split = pd.Series(index=data.index, dtype="str")
split.loc[train], split.loc[validation], split.loc[test] = "train", "validation", "test"
reports = root / "reports"
reports.mkdir(exist_ok=True)
pd.DataFrame({"ID": data["ID"], "split": split}).to_csv(reports / "data_split.csv", index=False)

categorical = ["SEX", "EDUCATION", "MARRIAGE"]
numeric = [column for column in features.columns if column not in categorical]
preprocess = ColumnTransformer([
    ("numeric", StandardScaler(), numeric),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
])
model = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])

print(f"Training: {len(train)}; validation: {len(validation)}; test reserved: {len(test)}")
print("Training logistic regression on the training set...")
model.fit(features.loc[train], labels.loc[train])
probability = model.predict_proba(features.loc[validation])[:, 1]
prediction = (probability >= 0.5).astype(int)
actual = labels.loc[validation]
matrix = confusion_matrix(actual, prediction, labels=[0, 1])
result = (f"Validation results (threshold 0.5)\n"
          f"Validation records: {len(validation)}; actual defaults: {actual.sum()}\n"
          f"Confusion matrix [[TN, FP], [FN, TP]]: {matrix.tolist()}\n"
          f"Precision: {precision_score(actual, prediction):.4f}\n"
          f"Recall: {recall_score(actual, prediction):.4f}\n"
          f"ROC-AUC: {roc_auc_score(actual, probability):.4f}\n"
          f"Average precision (PR summary): {average_precision_score(actual, probability):.4f}\n")
print(result)
(reports / "baseline_validation.txt").write_text(result, encoding="utf-8")
print("Saved reports/baseline_validation.txt and reports/data_split.csv; test set was not scored.")
