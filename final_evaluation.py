"""Protocol-controlled held-out evaluation and capacity-constrained review analysis."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent


def evaluate_held_out_test() -> None:
    """Evaluate predetermined Random Forest model on 6,000 held-out test records."""
    data = pd.read_excel(ROOT / 'data/raw/default of credit card clients.xls', header=1, engine='calamine')
    split = pd.read_csv(ROOT / 'reports/data_split.csv')
    assert data.ID.is_unique and split.ID.is_unique and set(data.ID) == set(split.ID)
    data = data.merge(split, on='ID', validate='one_to_one')

    train_mask = data['split'].eq('train')
    test_mask = data['split'].eq('test')
    assert train_mask.sum() == 18000 and test_mask.sum() == 6000

    target = 'default payment next month'
    features = data.drop(columns=['ID', target, 'split', 'SEX', 'EDUCATION', 'MARRIAGE'])
    labels = data[target]

    model = Pipeline([
        ('preprocess', ColumnTransformer([('numeric', StandardScaler(), list(features.columns))])),
        ('forest', RandomForestClassifier(n_estimators=100, min_samples_leaf=25, random_state=42, n_jobs=-1)),
    ])
    print('Fitting the predetermined model on the original training set...', flush=True)
    model.fit(features.loc[train_mask], labels.loc[train_mask])

    probabilities = model.predict_proba(features.loc[test_mask])[:, 1]
    actual = labels.loc[test_mask].to_numpy()
    customer_ids = data.loc[test_mask, 'ID'].to_numpy()
    sex_codes = data.loc[test_mask, 'SEX'].to_numpy()

    # 1. Main test metrics
    test_metrics = {
        'n': len(actual),
        'defaults': int(actual.sum()),
        'roc_auc': roc_auc_score(actual, probabilities),
        'average_precision': average_precision_score(actual, probabilities),
        'brier': brier_score_loss(actual, probabilities),
    }
    pd.DataFrame([test_metrics]).to_csv(ROOT / 'reports/final_test_metrics.csv', index=False, float_format='%.6f')

    # 2. Threshold sensitivity analysis (0.50 and hypothetical 1/6)
    threshold_records = []
    for threshold in [0.5, 1 / 6]:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(actual, predictions, labels=[0, 1]).ravel()
        threshold_records.append({
            'threshold': threshold,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'precision': precision_score(actual, predictions, zero_division=0),
            'recall': recall_score(actual, predictions, zero_division=0),
            'flagged_fraction': predictions.mean(),
            'hypothetical_cost_fp1_fn5': fp + 5 * fn,
        })
    pd.DataFrame(threshold_records).to_csv(ROOT / 'reports/final_test_thresholds.csv', index=False, float_format='%.6f')

    # 3. Capacity-limited review: deterministic stable ordering (IDs are not model features)
    sort_order = np.lexsort((customer_ids, -probabilities))
    capacity_records = []
    for fraction in [0.10, 0.20]:
        selected_indices = sort_order[:int(len(actual) * fraction)]
        captured_defaults = int(actual[selected_indices].sum())
        capacity_records.append({
            'capacity_fraction': fraction,
            'review_slots': len(selected_indices),
            'captured_defaults': captured_defaults,
            'review_precision': captured_defaults / len(selected_indices),
            'default_recall': captured_defaults / actual.sum(),
            'random_selection_expected_defaults': len(selected_indices) * actual.mean(),
        })
    pd.DataFrame(capacity_records).to_csv(ROOT / 'reports/final_review_capacity.csv', index=False, float_format='%.6f')

    # 4. Demographic group descriptive diagnostics
    group_records = []
    for code in sorted(np.unique(sex_codes)):
        mask = sex_codes == code
        group_actual = actual[mask]
        group_pred = (probabilities[mask] >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(group_actual, group_pred, labels=[0, 1]).ravel()
        group_records.append({
            'sex_code': code,
            'n': int(mask.sum()),
            'defaults': int(group_actual.sum()),
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'recall': tp / (tp + fn),
            'false_positive_rate': fp / (fp + tn),
            'flagged_rate': group_pred.mean(),
        })
    pd.DataFrame(group_records).to_csv(ROOT / 'reports/final_test_group_metrics.csv', index=False, float_format='%.6f')

    print('\nTest metrics:\n', pd.DataFrame([test_metrics]).to_string(index=False))
    print('\nThresholds:\n', pd.DataFrame(threshold_records).to_string(index=False))
    print('\nFixed review capacity:\n', pd.DataFrame(capacity_records).to_string(index=False))
    print('\nGroup diagnostics:\n', pd.DataFrame(group_records).to_string(index=False))
    print('\nEvaluation complete. Outputs saved in reports/.')


if __name__ == '__main__':
    evaluate_held_out_test()
