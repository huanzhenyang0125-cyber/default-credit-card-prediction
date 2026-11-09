"""Post-evaluation diagnostics: no model or threshold selection on these results."""
import numpy as np
import pandas as pd

from risk_model import ROOT, TARGET, EXCLUDED, fit_fixed_model, load_data

data = load_data()
model = fit_fixed_model(data)
test = data.loc[data['split'].eq('test')].copy()
test['probability'] = model.predict_proba(test.drop(columns=EXCLUDED))[:, 1]
scored = test[['ID', TARGET, 'probability']].sort_values(['probability', 'ID'], ascending=[False, True])
actual = scored[TARGET].to_numpy()
n, total = len(actual), int(actual.sum())
assert (n, total) == (6000, 1327)
assert actual[:600].sum() == 415 and actual[:1200].sum() == 678
curve = []
for pct in range(1, 101):
    slots = n * pct // 100
    captured = int(actual[:slots].sum())
    curve.append({'capacity_pct': pct, 'slots': slots, 'defaults': captured,
                  'recall': captured / total, 'precision': captured / slots,
                  'random_expected_defaults': slots * total / n,
                  'lift': (captured / slots) / (total / n)})
bands = []
for pct in range(5, 101, 5):
    start, end = n * (pct - 5) // 100, n * pct // 100
    found = int(actual[start:end].sum())
    bands.append({'capacity_band': f'{pct-5}–{pct}%', 'end_pct': pct, 'additional_reviews': end-start,
                  'additional_defaults': found, 'marginal_precision': found/(end-start),
                  'reviews_per_default': (end-start)/found if found else np.nan})
scored.to_csv(ROOT / 'reports/test_risk_scores.csv', index=False, float_format='%.12f')
pd.DataFrame(curve).to_csv(ROOT / 'reports/review_capacity_curve.csv', index=False)
pd.DataFrame(bands).to_csv(ROOT / 'reports/review_capacity_marginal.csv', index=False)
print(pd.DataFrame(curve).query('capacity_pct in [5,10,15,20,30,50,100]').to_string(index=False))
print('\nMarginal review yields:\n', pd.DataFrame(bands).head(6).to_string(index=False))
print('\nPost-evaluation diagnostics saved; model and evaluation rules unchanged.')
