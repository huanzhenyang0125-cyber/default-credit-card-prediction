from pathlib import Path

import altair as alt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from risk_model import EXCLUDED, TARGET, fit_fixed_model, load_data as read_source

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Credit Risk Decision Lab', layout='wide', initial_sidebar_state='expanded')
st.html(f"<style>{(ROOT / 'styles/workspace.css').read_text(encoding='utf-8')}</style>")
TEAL, NAVY, MUTED = '#167D8D', '#172B4D', '#64748B'


@st.cache_data
def table(name):
    return pd.read_csv(ROOT / 'reports' / name)


@st.cache_data
def source():
    return read_source()


@st.cache_resource
def reconstructed_model():
    return fit_fixed_model(source())


def heading(title, description):
    st.title(title)
    st.write(description)


def metrics(items):
    with st.grid(len(items), min_column_width=180):
        for label, value, description in items:
            st.metric(label, value, help=description)


def frame(data, caption):
    st.dataframe(data, hide_index=True, alt=caption)


def curve_chart(selected=None):
    data = table('review_capacity_curve.csv')
    rows = pd.concat([
        pd.DataFrame({'Capacity': data.capacity_pct / 100, 'Coverage': data.recall, 'Series': 'Model ranking'}),
        pd.DataFrame({'Capacity': [0, 1], 'Coverage': [0, 1], 'Series': 'Random allocation expectation'}),
    ], ignore_index=True)
    chart = alt.Chart(rows).mark_line(strokeWidth=3).encode(
        x=alt.X('Capacity:Q', title='Accounts reviewed', axis=alt.Axis(format='.0%'), scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('Coverage:Q', title='Observed defaults covered', axis=alt.Axis(format='.0%'), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('Series:N', scale=alt.Scale(domain=['Model ranking', 'Random allocation expectation'], range=[TEAL, MUTED]), legend=alt.Legend(orient='bottom', title=None)),
        tooltip=[alt.Tooltip('Capacity:Q', format='.0%'), alt.Tooltip('Coverage:Q', format='.1%'), 'Series:N'])
    if selected is not None:
        point = data.loc[data.capacity_pct.eq(selected)].assign(Capacity=lambda x: x.capacity_pct / 100, Coverage=lambda x: x.recall)
        chart += alt.Chart(point).mark_point(size=120, filled=True, color=NAVY).encode(x='Capacity:Q', y='Coverage:Q')
    return chart.properties(height=310)


with st.sidebar:
    st.subheader('Credit Risk\nDecision Lab')
    st.caption('RISK ANALYTICS WORKSPACE')
    section = st.radio('Workspace', ['Overview', 'Data explorer', 'Model validation', 'Decision lab', 'Prediction simulator'], label_visibility='collapsed', key='section')
    st.divider()
    st.write('**30,000** accounts · **22.12%** default')
    st.caption('UCI credit-card portfolio · Taiwan, 2005')

if section == 'Overview':
    heading('Credit risk research', 'Model evidence, review capacity and operational trade-offs.')
    m = table('final_test_metrics.csv').iloc[0]
    metrics([('ROC-AUC', f'{m.roc_auc:.3f}', 'Held-out risk discrimination'), ('Average precision', f'{m.average_precision:.3f}', 'Precision–recall summary'), ('Brier score', f'{m.brier:.3f}', 'Probability error · lower is better'), ('Test accounts', '6,000', '1,327 observed defaults')])
    st.divider()
    st.subheader('Review capacity and default coverage')
    st.altair_chart(curve_chart(), alt='Cumulative historical default coverage by review capacity')
    st.write('**600 review slots · 415 historical defaults · 31.27% coverage.** The highest-risk 10% of test accounts concentrates more observed defaults than the random-selection expectation of 132.7.')
    st.caption('Historical ranking, not an intervention experiment. Open Decision lab to adjust capacity and inspect incremental coverage.')
    st.subheader('An auditable research workflow')
    frame(pd.DataFrame({'Stage': ['Data', 'Development', 'Evaluation', 'Decision analysis'], 'Evidence': ['Official workbook; raw values retained', 'Baselines, calibration and cross-validation', 'Recorded protocol and held-out split', 'Capacity, assumed costs and group diagnostics'], 'Scope': ['30,000 historical records', '18,000 train / 6,000 validation', '6,000 test; already scored', 'Exploratory; no profit claim']}), 'Research workflow and evaluation boundaries')
    with st.expander('Read the evidence boundary'):
        st.write('Full-data exploration preceded splitting. This is not external validation. Extended capacity analysis was added after test scoring, so it is exploratory rather than fresh confirmatory evidence. No prevented-loss or profit estimate is established.')

elif section == 'Data explorer':
    heading('Explore the source data.', 'Inspect distributions and coding before drawing conclusions.')
    sub = st.segmented_control('Data view', ['Distributions', 'Repayment behaviour', 'Risk scores', 'Data quality'], default='Distributions', required=True, key='data_view')
    data = source()
    if sub == 'Distributions':
        variable = st.selectbox('Measure', ['Credit limit', 'Age', 'Latest bill', 'Latest payment'], key='measure')
        col = {'Credit limit': 'LIMIT_BAL', 'Age': 'AGE', 'Latest bill': 'BILL_AMT1', 'Latest payment': 'PAY_AMT1'}[variable]
        cohort = st.segmented_control('Historical outcome', ['All accounts', 'Default', 'Non-default'], default='All accounts', required=True, key='cohort')
        subset = data if cohort == 'All accounts' else data[data[TARGET].eq(int(cohort == 'Default'))]
        metrics([('Accounts', f'{len(subset):,}', 'Full-sample exploratory selection'), ('Median', f'{subset[col].median():,.0f}', 'Years for age; otherwise NTD'), ('Observed default rate', f'{subset[TARGET].mean():.1%}', 'Historical outcome, not a forecast')])
        st.altair_chart(alt.Chart(subset[[col]]).mark_bar(color=TEAL).encode(x=alt.X(f'{col}:Q', bin=alt.Bin(maxbins=35), title=variable), y=alt.Y('count():Q', title='Accounts'), tooltip=['count():Q']).properties(height=330), alt=f'Historical distribution of {variable}')
    elif sub == 'Repayment behaviour':
        months = ['September', 'August', 'July', 'June', 'May', 'April']
        month = st.selectbox('Repayment month', months, key='month')
        field = dict(zip(months, ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']))[month]
        summary = data.groupby(field)[TARGET].agg(Accounts='size', Defaults='sum', Rate='mean').reset_index().rename(columns={field: 'Repayment code'})
        st.altair_chart(alt.Chart(summary).mark_bar(color=TEAL).encode(x=alt.X('Repayment code:O'), y=alt.Y('Rate:Q', axis=alt.Axis(format='.0%'), title='Historical default rate'), tooltip=['Repayment code:O', 'Accounts:Q', alt.Tooltip('Rate:Q', format='.1%')]).properties(height=320), alt='Default rates and account counts by repayment code')
        frame(summary.round(3), 'Monthly repayment codes and historical defaults')
        st.caption('Small categories can have unstable rates. Codes 0 and −2 are not fully documented on the source page.')
    elif sub == 'Risk scores':
        scores = table('test_risk_scores.csv')
        st.caption('Fixed-model scores on already-evaluated test records. This is a post-evaluation descriptive view.')
        st.altair_chart(alt.Chart(scores).mark_bar(color=TEAL).encode(x=alt.X('probability:Q', bin=alt.Bin(maxbins=30), title='Predicted probability'), y=alt.Y('count():Q', title='Accounts')).properties(height=320), alt='Distribution of fixed-model test probabilities')
        st.write('Predicted probabilities describe associations learned from historical data; they do not identify causal effects.')
    else:
        metrics([('Missing cells', f'{data.isna().sum().sum():,}', 'As interpreted by Pandas'), ('Duplicate IDs', f'{data.ID.duplicated().sum():,}', 'Customer identifiers'), ('Negative latest bills', f'{data.BILL_AMT1.lt(0).sum():,}', 'Preserved for investigation')])
        st.write('An undocumented code or negative balance is not automatically an error. Original records are retained; their meaning requires further verification.')
        frame(data[['EDUCATION', 'MARRIAGE', 'PAY_0']].astype(str).apply(lambda s: ', '.join(sorted(s.unique()))).rename('Observed codes').reset_index().rename(columns={'index': 'Field'}), 'Observed category values')

elif section == 'Model validation':
    heading('Examine the model evidence.', 'Separate ranking performance, probability quality and subgroup behaviour.')
    sub = st.segmented_control('Validation view', ['Performance', 'Calibration', 'Feature importance', 'Stability', 'Group diagnostics'], default='Performance', required=True, key='validation_view')
    if sub == 'Performance':
        frame(table('model_comparison.csv').rename(columns={'model':'Model','roc_auc':'ROC-AUC','average_precision':'Average precision','brier':'Brier score'}).round(4), 'Candidate model metrics on validation data')
        st.caption('Candidate comparison uses validation data. Final test metrics concern the forest excluding sex, education and marital status.')
        st.subheader('Final model at the recorded test thresholds')
        frame(table('final_test_thresholds.csv').rename(columns={'threshold':'Threshold','tn':'True negatives','fp':'False positives','fn':'False negatives','tp':'True positives','precision':'Precision','recall':'Recall','flagged_fraction':'Flagged share','hypothetical_cost_fp1_fn5':'Assumed cost units'}).round(3), 'Recorded test classification results')
        scores = table('test_risk_scores.csv')
        fpr, tpr, _ = roc_curve(scores[TARGET], scores.probability)
        precision, recall, _ = precision_recall_curve(scores[TARGET], scores.probability)
        plot_choice = st.segmented_control('Curve', ['ROC', 'Precision–recall'], default='ROC', required=True, key='curve_choice')
        chosen_threshold = st.slider('Explore classification threshold', 0.01, 0.99, 0.50, 0.01, key='curve_threshold')
        prediction = scores.probability.ge(chosen_threshold)
        actual = scores[TARGET].eq(1)
        tp = int((prediction & actual).sum())
        fp = int((prediction & ~actual).sum())
        recall_now = tp / int(actual.sum())
        fpr_now = fp / int((~actual).sum())
        precision_now = tp / int(prediction.sum()) if prediction.sum() else 0.0
        metrics([('Recall', f'{recall_now:.1%}', 'Share of observed defaults identified'), ('False positive rate', f'{fpr_now:.1%}', 'Share of non-defaults flagged'), ('Precision', f'{precision_now:.1%}', 'Share of flagged accounts that defaulted')])
        fig = go.Figure(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC', line={'color': TEAL}))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line={'color': MUTED, 'dash': 'dash'}))
        fig.add_trace(go.Scatter(x=[fpr_now], y=[recall_now], mode='markers', name='Selected threshold', marker={'size': 12, 'color': '#111111'}))
        fig.update_layout(title='ROC curve · held-out sample', xaxis_title='False positive rate', yaxis_title='Recall', height=420, font={'color':'#111111'})
        fig2 = go.Figure(go.Scatter(x=recall, y=precision, mode='lines', name='Precision–recall', line={'color': TEAL}))
        fig2.add_trace(go.Scatter(x=[recall_now], y=[precision_now], mode='markers', name='Selected threshold', marker={'size': 12, 'color': '#111111'}))
        fig2.update_layout(title='Precision–recall curve · held-out sample', xaxis_title='Recall', yaxis_title='Precision', height=420, font={'color':'#111111'})
        st.plotly_chart(fig if plot_choice == 'ROC' else fig2, alt='Held-out model curve with selected threshold')
        st.caption('Threshold exploration is post-evaluation analysis on the already-scored test set, not a validated policy or fresh test.')
        threshold_row = table('final_test_thresholds.csv').iloc[0]
        f1 = 2 * threshold_row.precision * threshold_row.recall / (threshold_row.precision + threshold_row.recall)
        st.caption(f'At threshold 0.50: precision {threshold_row.precision:.3f}, recall {threshold_row.recall:.3f}, F1 {f1:.3f}. Curves use recorded held-out probabilities.')
    elif sub == 'Calibration':
        st.image(str(ROOT / 'figures/calibration.png'), alt='Candidate-model probability calibration on validation data', caption='Validation candidates, before final demographic-feature exclusion.')
        st.write('Calibration did not improve the overall Brier score in this candidate comparison. Ranking quality and probability reliability are different properties.')
    elif sub == 'Feature importance':
        importance = table('forest_importance.csv').head(12).copy()
        st.bar_chart(importance, x='feature', y='importance', horizontal=True, alt='Training impurity importance of the candidate random forest')
        st.caption('Candidate forest including demographic fields, not the final restricted model. Impurity importance may be biased by feature structure and correlation; it is not a causal explanation.')
    elif sub == 'Stability':
        frame(table('training_cv.csv').round(4), 'Training-only three-fold forest results')
        frame(table('validation_bootstrap.csv').rename(columns={'Unnamed: 0':'Model'}).round(4), 'Fixed-model validation bootstrap intervals')
        st.caption('These checks do not establish stability across time, geography or a new population.')
    else:
        frame(table('final_test_group_metrics.csv').rename(columns={'sex_code':'Recorded sex code','n':'Accounts','defaults':'Observed defaults','recall':'Recall','false_positive_rate':'False positive rate','flagged_rate':'Flagged share'}).round(4), 'Final model descriptive group diagnostics')
        st.write('Recorded code 1 denotes male and code 2 female. These descriptive comparisons do not establish fairness, discrimination causes or compliance.')

elif section == 'Decision lab':
    heading('Turn risk scores into questions.', 'Explore review workload and explicit assumptions without claiming intervention benefits.')
    sub = st.segmented_control('Decision view', ['Review capacity', 'Marginal coverage', 'Threshold & cost', 'Expected loss'], default='Review capacity', required=True, key='decision_view')
    curve = table('review_capacity_curve.csv')
    if sub == 'Review capacity':
        pct = st.slider('Accounts available for review (%)', 1, 100, 10, key='review_pct')
        row = curve.loc[curve.capacity_pct.eq(pct)].iloc[0]
        metrics([('Review slots', f'{row.slots:,.0f}', f'{pct}% of 6,000 accounts'), ('Defaults captured', f'{row.defaults:,.0f}', 'Historical outcomes'), ('Default coverage', f'{row.recall:.1%}', 'Share of 1,327 observed defaults'), ('Review precision', f'{row.precision:.1%}', 'Default rate in selected accounts')])
        st.altair_chart(curve_chart(pct), alt='Capacity curve with selected review budget')
        st.caption(f'Random-selection expectation: {row.random_expected_defaults:,.1f} defaults. Historical concentration lift: {row.lift:.2f}×; this is not a causal improvement.')
        st.caption('Extended capacity analysis is a post-evaluation diagnostic on already-scored test data.')
        st.download_button('Download capacity data', curve.to_csv(index=False), 'review_capacity_curve.csv', 'text/csv')
    elif sub == 'Marginal coverage':
        bands = table('review_capacity_marginal.csv')
        st.subheader('What do the next 300 reviews add?')
        st.altair_chart(alt.Chart(bands).mark_bar(color=TEAL).encode(x=alt.X('capacity_band:O', sort=None, title='Additional review capacity'), y=alt.Y('additional_defaults:Q', title='Additional historical defaults'), tooltip=['capacity_band:N','additional_reviews:Q','additional_defaults:Q',alt.Tooltip('marginal_precision:Q',format='.1%')]).properties(height=330), alt='Historical defaults in successive five-percent review bands')
        frame(bands.rename(columns={'capacity_band':'Capacity band','additional_reviews':'Additional reviews','additional_defaults':'Additional defaults','marginal_precision':'Marginal precision','reviews_per_default':'Reviews per observed default'}).drop(columns='end_pct').round(3), 'Incremental coverage by successive review band')
        st.caption('Post-evaluation exploration; marginal yields are historical observations, not defaults prevented.')
    elif sub == 'Threshold & cost':
        st.caption('Post-evaluation exploration. These controls do not select a validated policy.')
        left, right = st.columns(2)
        cutoff = left.slider('Probability threshold', 0.01, 0.99, 0.50, 0.01, key='threshold')
        ratio = right.slider('Assumed missed-default / false-alarm cost', 1, 20, 5, key='cost_ratio')
        scores = table('test_risk_scores.csv')
        actual, pred = scores[TARGET].to_numpy(), scores.probability.ge(cutoff).to_numpy()
        tp, fp = int(((actual == 1) & pred).sum()), int(((actual == 0) & pred).sum())
        fn, tn = int(((actual == 1) & ~pred).sum()), int(((actual == 0) & ~pred).sum())
        metrics([('Recall', f'{tp/(tp+fn):.1%}', 'Observed defaults identified'), ('Flagged accounts', f'{pred.sum():,}', f'{pred.mean():.1%} of test records'), ('Assumed error cost', f'{fp+ratio*fn:,}', 'Abstract units; not money')])
        frame(pd.DataFrame({'Actual outcome':['Non-default','Default'],'Predicted non-default':[tn,fn],'Predicted default':[fp,tp]}), 'Illustrative confusion matrix')
        st.caption('Assumes false-alarm cost 1, missed-default cost as selected, correct outcomes cost 0, and equal costs across accounts. Real intervention and revenue data are unavailable.')
    elif sub == 'Expected loss':
        st.caption('All inputs are assumptions, not observed bank loss parameters.')
        c1,c2,c3 = st.columns(3)
        pd_pct = c1.number_input('Assumed default probability (%)', 0.0, 100.0, 5.0, key='assumed_pd')
        lgd = c2.number_input('Assumed loss given default (%)', 0.0, 100.0, 45.0, key='assumed_lgd')
        ead = c3.number_input('Assumed exposure at default (NTD)', 0.0, 10000000.0, 100000.0, 1000.0, key='assumed_ead')
        st.metric('Illustrative expected loss (NTD)', f'{pd_pct/100*lgd/100*ead:,.2f}')
        st.write('Expected loss = default probability × loss given default × exposure at default.')
        st.caption('Simplified arithmetic excluding discounting, recovery timing and exposure changes. No actual approval profit is established.')
elif section == 'Prediction simulator':
    heading('Prediction simulator', 'Explore the fixed historical model with synthetic inputs. This is not a lending decision.')
    st.info('No saved model artifact exists. Submitting this form reconstructs the documented random forest from the original training split before prediction. Inputs default to training-set medians; they do not describe a real customer.')
    data = source()
    training = data.loc[data['split'].eq('train')]
    features = training.drop(columns=EXCLUDED)
    labels = {'LIMIT_BAL':'Credit limit (NTD)', 'AGE':'Age (years)', 'PAY_0':'Repayment status · September', 'PAY_2':'Repayment status · August', 'PAY_3':'Repayment status · July', 'PAY_4':'Repayment status · June', 'PAY_5':'Repayment status · May', 'PAY_6':'Repayment status · April'}
    with st.form('simulation'):
        st.subheader('Account and repayment history')
        values = {}
        columns = st.columns(2)
        for i, col in enumerate(features.columns):
            if col.startswith('BILL_AMT'):
                label = f'Bill amount · month {col[-1]} (NTD)'
            elif col.startswith('PAY_AMT'):
                label = f'Payment amount · month {col[-1]} (NTD)'
            else:
                label = labels.get(col, col)
            low, high = int(training[col].min()), int(training[col].max())
            default = int(training[col].median())
            values[col] = columns[i % 2].number_input(label, min_value=low, max_value=high, value=default, step=1, key='sim_'+col)
        submitted = st.form_submit_button('Reconstruct model and estimate probability')
    st.caption('Numeric bounds are observed training ranges, not verified business rules. Repayment codes 0 and −2 are not fully documented by the dataset source. Synthetic combinations may not represent realistic accounts.')
    if submitted:
        try:
            model = reconstructed_model()
            sample = pd.DataFrame([values], columns=features.columns)
            probability = float(model.predict_proba(sample)[0, 1])
            band = 'Low' if probability < 0.10 else ('Medium' if probability < 0.25 else 'High')
            st.metric('Estimated probability of next-month default', f'{probability:.1%}')
            st.write(f'Illustrative risk band: **{band}**')
            st.caption('Bands are hypothetical: low <10%, medium 10–25%, high ≥25%. They are not validated policy thresholds; probability calibration and external validity are limited.')
        except (OSError, ValueError, KeyError) as exc:
            st.error(f'Unable to reconstruct the model or score these inputs: {exc}')

st.divider()
st.caption('UCI Default of Credit Card Clients · Yeh (2009) · DOI 10.24432/C55S3H · CC BY 4.0. Historical research only.')
