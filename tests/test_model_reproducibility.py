from pathlib import Path
import sys

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from risk_model import EXCLUDED, TARGET, fit_fixed_model, load_data

data = load_data()
model = fit_fixed_model(data)
test = data.loc[data['split'].eq('test')]
X = test.drop(columns=EXCLUDED)
assert list(model.feature_names_in_) == list(X.columns)
assert TARGET not in X.columns and 'ID' not in X.columns
assert len(X.columns) == 20
predicted = model.predict_proba(X)[:, 1]
saved = pd.read_csv(root / 'reports/test_risk_scores.csv')
comparison = pd.DataFrame({'ID': test.ID, 'prediction': predicted}).merge(saved[['ID', 'probability']], on='ID', validate='one_to_one')
assert len(comparison) == 6000
assert np.allclose(comparison.prediction, comparison.probability, atol=1e-10, rtol=0)
print('PASS: 20-feature training-only pipeline reproduces saved held-out predictions')
