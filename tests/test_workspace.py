"""Exercise navigation and critical research outputs in the dashboard."""
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

root = Path(__file__).resolve().parents[1]
app = AppTest.from_file(str(root / 'app.py'), default_timeout=30).run()
assert not app.exception
for section, widget, options in [
    ('Data explorer', 'data_view', ['Distributions', 'Repayment behaviour', 'Risk scores', 'Data quality']),
    ('Model validation', 'validation_view', ['Performance', 'Calibration', 'Feature importance', 'Stability', 'Group diagnostics']),
    ('Decision lab', 'decision_view', ['Review capacity', 'Marginal coverage', 'Threshold & cost', 'Expected loss']),
]:
    app.radio(key='section').set_value(section).run()
    assert not app.exception, section
    for option in options:
        app.segmented_control(key=widget).set_value(option).run()
        assert not app.exception, (section, option)
        print(f'PASS: {section} / {option}')

app.radio(key='section').set_value('Decision lab').run()
app.segmented_control(key='decision_view').set_value('Review capacity').run()
app.slider(key='review_pct').set_value(20).run()
assert app.metric[0].value == '1,200' and app.metric[1].value == '678'
app.segmented_control(key='decision_view').set_value('Expected loss').run()
assert app.metric[0].value == '2,250.00'
app.number_input(key='assumed_pd').set_value(10.0).run()
assert app.metric[0].value == '4,500.00'
assert not app.exception

app.radio(key='section').set_value('Prediction simulator').run()
assert not app.exception
app.button[0].click().run(timeout=60)
assert not app.exception
assert any('Estimated probability' in metric.label for metric in app.metric)
print('PASS: synthetic prediction simulator')

curve = pd.read_csv(root / 'reports/review_capacity_curve.csv')
bands = pd.read_csv(root / 'reports/review_capacity_marginal.csv')
assert curve.defaults.is_monotonic_increasing
assert curve.iloc[-1].defaults == 1327 and curve.iloc[-1].recall == 1
assert bands.additional_reviews.sum() == 6000 and bands.additional_defaults.sum() == 1327
print('PASS: capacity reconciliation and expected-loss interaction')
