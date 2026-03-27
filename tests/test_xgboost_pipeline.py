import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

if 'xgboost' not in sys.modules:
    xgboost_stub = types.ModuleType('xgboost')
    xgboost_stub.XGBClassifier = object
    sys.modules['xgboost'] = xgboost_stub

from src import train_xgboost, tune_xgboost, xgboost_features


def test_extract_linguistic_features_adds_expected_columns():
    df = pd.DataFrame({
        'content': ['HELLO world!', 'two words?'],
        'title': ['Title', None],
    })


    result = xgboost_features.extract_linguistic_features(df.copy())

    assert all(col in result.columns for col in xgboost_features.FEATURE_COLS)
    assert result.loc[0, 'caps_ratio'] > 0
    assert result.loc[0, 'exclamation_density'] > 0
    assert result.loc[1, 'question_density'] > 0
    assert result.loc[0, 'content_word_count'] == 2
    assert result.loc[1, 'avg_word_length'] > 0
    assert result.loc[0, 'title_content_ratio'] > 0


def test_build_linguistic_sparse_matrix_shape_and_dtype():
    df = pd.DataFrame({
        'caps_ratio': [0.1, 0.2],
        'exclamation_density': [0.0, 0.1],
        'question_density': [0.0, 0.2],
        'content_word_count': [10, 20],
        'avg_word_length': [4.0, 5.0],
        'title_content_ratio': [0.3, 0.4],
    })

    matrix = xgboost_features.build_linguistic_sparse_matrix(df)

    assert matrix.shape == (2, 6)
    assert matrix.dtype == np.float32


def test_fit_tfidf_on_training_sample_returns_configured_vectorizer():
    series = pd.Series(
        ['shared token text' for _ in range(10)] +
        ['rare example one', 'rare example two']
    )

    tfidf = xgboost_features.fit_tfidf_on_training_sample(series)

    assert tfidf.max_features == 1500
    assert tfidf.ngram_range == (1, 2)
    assert tfidf.dtype == np.float32
    assert tfidf.min_df == 10
    assert tfidf.max_df == 0.90
    assert len(tfidf.vocabulary_) > 0


def test_transform_text_in_chunks_matches_direct_transform():
    series = pd.Series([
        'alpha beta gamma',
        'beta gamma delta',
        'gamma delta epsilon',
        'delta epsilon zeta',
    ])
    tfidf = xgboost_features.fit_tfidf_on_training_sample(pd.Series(['alpha beta'] * 10 + ['beta gamma'] * 2))
    tfidf.fit(pd.Series(['alpha beta gamma'] * 10 + ['delta epsilon zeta'] * 10))

    expected = tfidf.transform(series.fillna(''))
    actual = xgboost_features.transform_text_in_chunks(series, tfidf, chunk_size=2, label='test')

    assert actual.shape == expected.shape
    assert np.allclose(actual.toarray(), expected.toarray())


class FakeVectorizer:
    def __init__(self):
        self.saved = False


class FakeXGBClassifier:
    last_init_kwargs = None
    last_fit = None
    last_predict_input = None
    saved_model_path = None

    def __init__(self, **kwargs):
        type(self).last_init_kwargs = kwargs

    def fit(self, X, y):
        type(self).last_fit = (X.copy(), np.array(y, copy=True))
        return self

    def predict(self, X):
        type(self).last_predict_input = X.copy()
        return np.zeros(X.shape[0], dtype=np.int8)

    def save_model(self, path):
        type(self).saved_model_path = path


class FakeRandomizedSearchCV:
    init_kwargs = None
    fit_args = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs
        self.best_score_ = 0.91
        self.best_params_ = {'max_depth': 12}

    def fit(self, X, y):
        type(self).fit_args = (X.copy(), np.array(y, copy=True))
        return self


def test_train_xgboost_main_text_only_pipeline(monkeypatch, tmp_path):
    train_df = pd.DataFrame({
        'content_processed': ['alpha beta', 'beta gamma', 'gamma delta', 'delta epsilon'],
        'type': [0, 1, 0, 1],
    })
    val_df = pd.DataFrame({
        'content_processed': ['alpha beta', 'delta epsilon'],
        'type': [0, 1],
    })

    def fake_read_csv(path, usecols=None, dtype=None):
        if 'train' in str(path):
            return train_df.copy()
        if 'val' in str(path):
            return val_df.copy()
        raise AssertionError(f'unexpected path: {path}')

    def fake_fit_tfidf(series):
        assert list(series) == list(train_df['content_processed'])
        return FakeVectorizer()

    def fake_transform(series, tfidf, chunk_size=50000, label='dataset'):
        values = np.arange(len(series) * 3, dtype=np.float32).reshape(len(series), 3)
        return sp.csr_matrix(values)

    dumped = {}

    def fake_joblib_dump(obj, path):
        dumped['obj'] = obj
        dumped['path'] = path

    monkeypatch.setattr(train_xgboost.pd, 'read_csv', fake_read_csv)
    monkeypatch.setattr(train_xgboost, 'fit_tfidf_on_training_sample', fake_fit_tfidf)
    monkeypatch.setattr(train_xgboost, 'transform_text_in_chunks', fake_transform)
    monkeypatch.setattr(train_xgboost.xgb, 'XGBClassifier', FakeXGBClassifier)
    monkeypatch.setattr(train_xgboost.joblib, 'dump', fake_joblib_dump)
    monkeypatch.setattr(train_xgboost, 'accuracy_score', lambda y_true, y_pred: 1.0)
    monkeypatch.setattr(train_xgboost, 'classification_report', lambda *args, **kwargs: 'report')
    monkeypatch.setattr(train_xgboost, 'confusion_matrix', lambda *args, **kwargs: np.array([[1, 0], [0, 1]]))
    monkeypatch.chdir(tmp_path)

    train_xgboost.main()

    assert FakeXGBClassifier.last_init_kwargs['tree_method'] == 'hist'
    assert FakeXGBClassifier.last_init_kwargs['n_estimators'] == 400
    assert FakeXGBClassifier.last_fit[0].shape == (4, 3)
    assert FakeXGBClassifier.last_predict_input.shape == (2, 3)
    assert dumped['obj'].__class__ is FakeVectorizer
    dumped_path = Path(dumped['path'])
    saved_model_path = Path(FakeXGBClassifier.saved_model_path)
    assert dumped_path.parent.name == 'models'
    assert dumped_path.name == 'tfidf_vectorizer1500ot.joblib'
    assert saved_model_path.parent.name == 'models'
    assert saved_model_path.name == 'xgboost_model1500ot.json'


def test_tune_xgboost_main_builds_and_downsamples_before_search(monkeypatch):
    train_df = pd.DataFrame({
        'title': ['t1', 't2', 't3', 't4'],
        'content': ['A!', 'B?', 'C!', 'D?'],
        'content_processed': ['alpha beta', 'beta gamma', 'gamma delta', 'delta epsilon'],
        'type': [0, 1, 0, 1],
    })

    def fake_read_csv(path, usecols=None, dtype=None):
        return train_df.copy()

    def fake_extract(df):
        for idx, col in enumerate(xgboost_features.FEATURE_COLS):
            df[col] = np.float32(idx + 1)
        return df

    def fake_fit_tfidf(series):
        return FakeVectorizer()

    def fake_transform(series, tfidf, chunk_size=50000, label='dataset'):
        values = np.arange(len(series) * 2, dtype=np.float32).reshape(len(series), 2)
        return sp.csr_matrix(values)

    resample_calls = {}

    def fake_resample(X, y, n_samples, random_state, stratify):
        resample_calls['shape'] = X.shape
        resample_calls['y'] = np.array(y, copy=True)
        resample_calls['n_samples'] = n_samples
        resample_calls['random_state'] = random_state
        resample_calls['stratify'] = np.array(stratify, copy=True)
        return X[:3], np.array(y[:3], copy=True)

    monkeypatch.setattr(tune_xgboost.pd, 'read_csv', fake_read_csv)
    monkeypatch.setattr(tune_xgboost, 'extract_linguistic_features', fake_extract)
    monkeypatch.setattr(tune_xgboost, 'fit_tfidf_on_training_sample', fake_fit_tfidf)
    monkeypatch.setattr(tune_xgboost, 'transform_text_in_chunks', fake_transform)
    monkeypatch.setattr(tune_xgboost.xgb, 'XGBClassifier', FakeXGBClassifier)
    monkeypatch.setattr(tune_xgboost, 'RandomizedSearchCV', FakeRandomizedSearchCV)
    monkeypatch.setattr(tune_xgboost, 'resample', fake_resample)

    tune_xgboost.main()

    assert FakeXGBClassifier.last_init_kwargs['tree_method'] == 'hist'
    assert FakeXGBClassifier.last_init_kwargs['eval_metric'] == 'logloss'
    assert FakeXGBClassifier.last_init_kwargs['n_jobs'] == 8
    assert resample_calls['shape'] == (4, 8)
    assert resample_calls['n_samples'] == 150000
    assert resample_calls['random_state'] == 42
    assert np.array_equal(resample_calls['stratify'], train_df['type'].values)
    assert FakeRandomizedSearchCV.init_kwargs['n_iter'] == 15
    assert FakeRandomizedSearchCV.init_kwargs['cv'] == 3
    assert FakeRandomizedSearchCV.init_kwargs['scoring'] == 'f1'
    assert FakeRandomizedSearchCV.init_kwargs['n_jobs'] == 1
    assert FakeRandomizedSearchCV.fit_args[0].shape == (3, 8)
    assert np.array_equal(FakeRandomizedSearchCV.fit_args[1], np.array([0, 1, 0]))
