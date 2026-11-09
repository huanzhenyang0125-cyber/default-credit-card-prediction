"""The fixed final specification, shared by research diagnostics and the app."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
TARGET = 'default payment next month'
EXCLUDED = ['ID', TARGET, 'split', 'SEX', 'EDUCATION', 'MARRIAGE']


def load_data():
    data = pd.read_excel(ROOT / 'data/raw/default of credit card clients.xls', header=1, engine='calamine')
    split = pd.read_csv(ROOT / 'reports/data_split.csv')
    assert data.ID.is_unique and split.ID.is_unique and set(data.ID) == set(split.ID)
    return data.merge(split, on='ID', validate='one_to_one')


def fit_fixed_model(data):
    X = data.drop(columns=EXCLUDED)
    train = data['split'].eq('train')
    assert train.sum() == 18000
    model = Pipeline([
        ('preprocess', ColumnTransformer([('numeric', StandardScaler(), list(X.columns))])),
        ('forest', RandomForestClassifier(n_estimators=100, min_samples_leaf=25, random_state=42, n_jobs=-1)),
    ])
    model.fit(X.loc[train], data.loc[train, TARGET])
    return model
