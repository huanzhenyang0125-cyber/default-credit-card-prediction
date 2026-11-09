"""Check browser scoring and review/threshold inputs against the Python model."""
import json
import subprocess
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from risk_model import EXCLUDED, fit_fixed_model, load_data

data = load_data()
features = data.drop(columns=EXCLUDED)
model = fit_fixed_model(data)
sample = features.sample(n=300, random_state=58)
payload = json.loads((ROOT / 'data/web_model.json').read_text(encoding='utf-8'))
assert list(sample.columns) == payload['features']
node = r'''
const fs=require('fs'), vm=require('vm');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const ctx={window:{addEventListener(){}},document:{},console};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync('scripts/workspace.js','utf8')+'\nevidence = '+JSON.stringify(input.evidence)+';',ctx);
const out=input.rows.map(row=>ctx.probability(row));
process.stdout.write(JSON.stringify(out));
'''
result = subprocess.run(['node', '-e', node], cwd=ROOT, text=True, check=True, capture_output=True,
                        input=json.dumps({'evidence':payload, 'rows':sample.to_dict(orient='records')}))
browser = np.array(json.loads(result.stdout))
python = model.predict_proba(sample)[:, 1]
np.testing.assert_allclose(browser, python, rtol=0, atol=1e-12)
scores = pd.read_csv(ROOT / 'reports/test_risk_scores.csv')
assert scores['probability'].is_monotonic_decreasing
assert len(scores) == 6000 and scores['default payment next month'].sum() == 1327
assert scores.iloc[:600]['default payment next month'].sum() == 415
assert round(roc_auc_score(scores['default payment next month'], scores['probability']), 6) == 0.774123
print(f'Browser forest matches Python for {len(sample)} profiles; largest error: {max(abs(browser-python)):.2g}')
print(f"At 31%: {scores.iloc[:1860]['default payment next month'].sum()} observed defaults among 1,860 accounts")
