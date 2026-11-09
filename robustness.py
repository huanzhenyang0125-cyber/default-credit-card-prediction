from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

root = Path(__file__).resolve().parent
df = pd.read_excel(root / 'data/raw/default of credit card clients.xls', header=1, engine='calamine')
split = pd.read_csv(root / 'reports/data_split.csv')
assert df.ID.is_unique and split.ID.is_unique and set(df.ID) == set(split.ID)
df = df.merge(split, on='ID', validate='one_to_one')
target = 'default payment next month'
train, validation = df['split'].eq('train'), df['split'].eq('validation')
assert (train.sum(), validation.sum(), df['split'].eq('test').sum()) == (18000, 6000, 6000)
y = df[target]
full_features = df.drop(columns=['ID', target, 'split'])
excluded = ['SEX', 'EDUCATION', 'MARRIAGE']

def pipeline(features):
    categories = [c for c in excluded if c in features.columns]
    numerical = [c for c in features.columns if c not in categories]
    transformer = ColumnTransformer([
        ('numeric', StandardScaler(), numerical),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), categories),
    ])
    return Pipeline([('preprocess', transformer), ('forest', RandomForestClassifier(
        n_estimators=100, min_samples_leaf=25, random_state=42, n_jobs=-1))])

print('Checking training-only cross-validation stability...', flush=True)
features = full_features
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)
X_train, y_train = features.loc[train].reset_index(drop=True), y.loc[train].reset_index(drop=True)
folds = []
for i, (fit_idx, check_idx) in enumerate(cv.split(X_train, y_train), start=1):
    model = pipeline(X_train)
    model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
    p = model.predict_proba(X_train.iloc[check_idx])[:, 1]
    folds.append({'fold': i, 'roc_auc': roc_auc_score(y_train.iloc[check_idx], p),
                  'average_precision': average_precision_score(y_train.iloc[check_idx], p),
                  'brier': brier_score_loss(y_train.iloc[check_idx], p)})
    print(f'Fold {i} complete', flush=True)
pd.DataFrame(folds).to_csv(root / 'reports/training_cv.csv', index=False, float_format='%.6f')

print('Comparing feature choices and group outcomes...', flush=True)
comparisons, group_rows = [], []
for name, X in [('all_features', full_features), ('exclude_demographics', full_features.drop(columns=excluded))]:
    model = pipeline(X)
    model.fit(X.loc[train], y.loc[train])
    p = model.predict_proba(X.loc[validation])[:, 1]
    actual = y.loc[validation].to_numpy()
    comparisons.append({'model': name, 'roc_auc': roc_auc_score(actual, p),
                        'average_precision': average_precision_score(actual, p),
                        'brier': brier_score_loss(actual, p)})
    sex = df.loc[validation, 'SEX'].to_numpy()
    for value in sorted(np.unique(sex)):
        group = sex == value
        group_y, group_pred = actual[group], (p[group] >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(group_y, group_pred, labels=[0, 1]).ravel()
        group_rows.append({'model': name, 'sex_code': value, 'n': group.sum(), 'defaults': group_y.sum(),
                           'flagged_rate': group_pred.mean(), 'recall': tp / (tp + fn),
                           'false_positive_rate': fp / (fp + tn), 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp})
pd.DataFrame(comparisons).to_csv(root / 'reports/demographic_feature_comparison.csv', index=False, float_format='%.6f')
pd.DataFrame(group_rows).to_csv(root / 'reports/group_metrics.csv', index=False, float_format='%.6f')

scenarios = pd.read_csv(root / 'reports/threshold_scenarios.csv')
cost_rows = []
for ratio in [2, 5, 10]:
    for _, row in scenarios.iterrows():
        cost_rows.append({'model': row['model'], 'threshold': row['threshold'], 'fn_to_fp_cost': ratio,
                          'cost_units': int(row['fp'] + ratio * row['fn']), 'flagged_fraction': row['flagged_fraction']})
pd.DataFrame(cost_rows).to_csv(root / 'reports/cost_sensitivity.csv', index=False)
loss_scenarios = []
for pd_assumption in [0.01, 0.05, 0.10]:
    for lgd in [0.45, 0.60]:
        ead = 100000
        loss_scenarios.append({'pd_assumption': pd_assumption, 'lgd_assumption': lgd,
                               'ead_assumption_ntd': ead, 'hypothetical_expected_loss_ntd': pd_assumption * lgd * ead})
pd.DataFrame(loss_scenarios).to_csv(root / 'reports/expected_loss_scenarios.csv', index=False)
print('Training cross-validation:\n', pd.DataFrame(folds).to_string(index=False))
print('\nFeature comparison:\n', pd.DataFrame(comparisons).to_string(index=False))
print('\nGroup metrics at threshold 0.5:\n', pd.DataFrame(group_rows).to_string(index=False))
print('\nSaved training CV, feature comparison, group metrics, cost and expected-loss scenarios. No test scoring.')
