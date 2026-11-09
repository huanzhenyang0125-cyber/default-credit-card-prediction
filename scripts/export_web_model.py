"""Export the fixed training-split forest and held-out scores for the static site."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from risk_model import EXCLUDED, fit_fixed_model, load_data


def main():
    data = load_data()
    train = data.loc[data['split'].eq('train')]
    x = train.drop(columns=EXCLUDED)
    model = fit_fixed_model(data)
    scaler = model.named_steps['preprocess'].named_transformers_['numeric']
    forest = model.named_steps['forest']
    trees = []
    for estimator in forest.estimators_:
        tree = estimator.tree_
        trees.append({
            'left': tree.children_left.tolist(), 'right': tree.children_right.tolist(),
            'feature': tree.feature.tolist(), 'threshold': tree.threshold.tolist(),
            'probability': tree.value[:, 0, 1].tolist(),
        })
    scores = pd.read_csv(ROOT / 'reports/test_risk_scores.csv')
    payload = {
        'features': list(x.columns), 'defaults': x.median().astype(int).to_dict(),
        'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'trees': trees,
        'scores': scores['probability'].tolist(),
        'outcomes': scores['default payment next month'].astype(int).tolist(),
    }
    out = ROOT / 'data/web_model.json'
    out.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    print(f'Exported {len(trees)} trees and {len(scores)} held-out scores to {out} ({out.stat().st_size:,} bytes)')
    print('Default prediction:', model.predict_proba(pd.DataFrame([payload['defaults']], columns=x.columns))[0, 1])


if __name__ == '__main__':
    main()
