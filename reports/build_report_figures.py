from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

root = Path(__file__).resolve().parents[1]
scores = pd.read_csv(root / 'reports/test_risk_scores.csv')
y, p = scores['default payment next month'], scores.probability
fpr, tpr, _ = roc_curve(y, p)
precision, recall, _ = precision_recall_curve(y, p)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(fpr, tpr, color='#14556b', label='Final forest')
axes[0].plot([0, 1], [0, 1], '--', color='#7c8793', label='Random ranking')
axes[0].set(xlabel='False positive rate', ylabel='Recall', title='ROC · test sample')
axes[0].legend(frameon=False)
axes[1].plot(recall, precision, color='#14556b')
axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision–recall · test sample')
for ax in axes:
    ax.grid(alpha=.18)
fig.tight_layout()
fig.savefig(root / 'figures/report_discrimination.png', dpi=180, bbox_inches='tight')
plt.close(fig)

curve = pd.read_csv(root / 'reports/review_capacity_curve.csv')
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(curve.capacity_pct, 100*curve.recall, color='#14556b', linewidth=2, label='Fixed-model ranking')
ax.plot([0,100], [0,100], '--', color='#7c8793', label='Random selection expectation')
ax.scatter([10,20], [31.2735,51.0927], color='#14556b')
ax.set(xlabel='Review capacity (% of test accounts)', ylabel='Observed defaults covered (%)', title='Review capacity and historical default coverage')
ax.grid(alpha=.18); ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(root / 'figures/report_capacity.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print('Generated research figures from recorded test scores and capacity results.')
