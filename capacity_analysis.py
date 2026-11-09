"""Post-evaluation diagnostics for review capacity and marginal yields.

Note: No model or threshold selection should be performed on these results.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from risk_model import EXCLUDED, ROOT, TARGET, fit_fixed_model, load_data


def run_capacity_analysis() -> None:
    """Compute 1%-100% capacity curve and 5% marginal review yields on test data."""
    data = load_data()
    model = fit_fixed_model(data)
    test = data.loc[data['split'].eq('test')].copy()
    test['probability'] = model.predict_proba(test.drop(columns=EXCLUDED))[:, 1]
    
    scored = test[['ID', TARGET, 'probability']].sort_values(['probability', 'ID'], ascending=[False, True])
    actual = scored[TARGET].to_numpy()
    n, total_defaults = len(actual), int(actual.sum())
    assert (n, total_defaults) == (6000, 1327)
    assert actual[:600].sum() == 415 and actual[:1200].sum() == 678

    curve = []
    for pct in range(1, 101):
        slots = n * pct // 100
        captured = int(actual[:slots].sum())
        curve.append({
            'capacity_pct': pct,
            'slots': slots,
            'defaults': captured,
            'recall': captured / total_defaults,
            'precision': captured / slots,
            'random_expected_defaults': slots * total_defaults / n,
            'lift': (captured / slots) / (total_defaults / n),
        })

    bands = []
    for pct in range(5, 101, 5):
        start = n * (pct - 5) // 100
        end = n * pct // 100
        found = int(actual[start:end].sum())
        additional_slots = end - start
        bands.append({
            'capacity_band': f'{pct-5}–{pct}%',
            'end_pct': pct,
            'additional_reviews': additional_slots,
            'additional_defaults': found,
            'marginal_precision': found / additional_slots,
            'reviews_per_default': additional_slots / found if found else np.nan,
        })

    scored.to_csv(ROOT / 'reports/test_risk_scores.csv', index=False, float_format='%.12f')
    pd.DataFrame(curve).to_csv(ROOT / 'reports/review_capacity_curve.csv', index=False)
    pd.DataFrame(bands).to_csv(ROOT / 'reports/review_capacity_marginal.csv', index=False)

    print(pd.DataFrame(curve).query('capacity_pct in [5, 10, 15, 20, 30, 50, 100]').to_string(index=False))
    print('\nMarginal review yields:\n', pd.DataFrame(bands).head(6).to_string(index=False))
    print('\nPost-evaluation diagnostics saved; model and evaluation rules unchanged.')


if __name__ == '__main__':
    run_capacity_analysis()
