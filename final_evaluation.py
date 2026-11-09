from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

root = Path(__file__).resolve().parent
df = pd.read_excel(root / 'data/raw/default of credit card clients.xls', header=1, engine='calamine')
allocation = pd.read_csv(root / 'reports/data_split.csv')
assert df.ID.is_unique and allocation.ID.is_unique and set(df.ID) == set(allocation.ID)
df = df.merge(allocation, on='ID', validate='one_to_one')
train, test = df['split'].eq('train'), df['split'].eq('test')
assert train.sum() == 18000 and test.sum() == 6000
target = 'default payment next month'
X = df.drop(columns=['ID', target, 'split', 'SEX', 'EDUCATION', 'MARRIAGE'])
y = df[target]
model = Pipeline([
    ('preprocess', ColumnTransformer([('numeric', StandardScaler(), list(X.columns))])),
    ('forest', RandomForestClassifier(n_estimators=100, min_samples_leaf=25, random_state=42, n_jobs=-1)),
])
print('Fitting the predetermined model on the original training set...', flush=True)
model.fit(X.loc[train], y.loc[train])
p = model.predict_proba(X.loc[test])[:, 1]
actual = y.loc[test].to_numpy()
ids = df.loc[test, 'ID'].to_numpy()
sex = df.loc[test, 'SEX'].to_numpy()
metrics = {'n': len(actual), 'defaults': int(actual.sum()), 'roc_auc': roc_auc_score(actual, p),
           'average_precision': average_precision_score(actual, p), 'brier': brier_score_loss(actual, p)}
pd.DataFrame([metrics]).to_csv(root / 'reports/final_test_metrics.csv', index=False, float_format='%.6f')
threshold_rows = []
for threshold in [0.5, 1 / 6]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, pred, labels=[0, 1]).ravel()
    threshold_rows.append({'threshold': threshold, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                           'precision': precision_score(actual, pred, zero_division=0),
                           'recall': recall_score(actual, pred, zero_division=0),
                           'flagged_fraction': pred.mean(), 'hypothetical_cost_fp1_fn5': fp + 5 * fn})
pd.DataFrame(threshold_rows).to_csv(root / 'reports/final_test_thresholds.csv', index=False, float_format='%.6f')

# Capacity-limited review: stable deterministic ordering; IDs are not model inputs.
order = np.lexsort((ids, -p))
capacity = []
for fraction in [0.10, 0.20]:
    selected = order[:int(len(actual) * fraction)]
    captured = int(actual[selected].sum())
    capacity.append({'capacity_fraction': fraction, 'review_slots': len(selected), 'captured_defaults': captured,
                     'review_precision': captured / len(selected), 'default_recall': captured / actual.sum(),
                     'random_selection_expected_defaults': len(selected) * actual.mean()})
pd.DataFrame(capacity).to_csv(root / 'reports/final_review_capacity.csv', index=False, float_format='%.6f')

groups = []
for value in sorted(np.unique(sex)):
    mask = sex == value
    a, pred = actual[mask], (p[mask] >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(a, pred, labels=[0, 1]).ravel()
    groups.append({'sex_code': value, 'n': int(mask.sum()), 'defaults': int(a.sum()), 'tn': tn, 'fp': fp,
                   'fn': fn, 'tp': tp, 'recall': tp / (tp + fn), 'false_positive_rate': fp / (fp + tn),
                   'flagged_rate': pred.mean()})
pd.DataFrame(groups).to_csv(root / 'reports/final_test_group_metrics.csv', index=False, float_format='%.6f')
print('\nTest metrics:\n', pd.DataFrame([metrics]).to_string(index=False))
print('\nThresholds:\n', pd.DataFrame(threshold_rows).to_string(index=False))
print('\nFixed review capacity:\n', pd.DataFrame(capacity).to_string(index=False))
print('\nGroup diagnostics:\n', pd.DataFrame(groups).to_string(index=False))
print('\nEvaluation complete. Outputs saved in reports/.')
