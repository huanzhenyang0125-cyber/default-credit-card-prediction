"""Fixed final specification for the Credit Risk Decision Lab model.

Shared across research evaluation scripts, test suites, and the Streamlit dashboard.
"""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
TARGET = 'default payment next month'
EXCLUDED = ['ID', TARGET, 'split', 'SEX', 'EDUCATION', 'MARRIAGE']


def load_data() -> pd.DataFrame:
    """Load official UCI credit dataset and merge with deterministic split partition."""
    data_path = ROOT / 'data/raw/default of credit card clients.xls'
    split_path = ROOT / 'reports/data_split.csv'
    
    data = pd.read_excel(data_path, header=1, engine='calamine')
    split = pd.read_csv(split_path)
    
    assert data.ID.is_unique and split.ID.is_unique and set(data.ID) == set(split.ID)
    return data.merge(split, on='ID', validate='one_to_one')


def fit_fixed_model(data: pd.DataFrame) -> Pipeline:
    """Train the pre-registered 20-feature Random Forest pipeline on training split only."""
    X = data.drop(columns=EXCLUDED)
    train_mask = data['split'].eq('train')
    assert train_mask.sum() == 18000
    
    pipeline = Pipeline([
        ('preprocess', ColumnTransformer([('numeric', StandardScaler(), list(X.columns))])),
        ('forest', RandomForestClassifier(n_estimators=100, min_samples_leaf=25, random_state=42, n_jobs=-1)),
    ])
    pipeline.fit(X.loc[train_mask], data.loc[train_mask, TARGET])
    return pipeline
