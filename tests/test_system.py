import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from pipeline import validate, COLUMNS, FEATURES, SOURCE, evaluate, profiles, train, data_fingerprint, build_rfm_from_raw
from storage import initialize, load_data, add_batch


def test_source_and_validation():
    data = validate(pd.read_csv(SOURCE))
    assert len(data) == 4338
    assert FEATURES == ['Recency', 'Frequency', 'Monetary']
    for col, bad_val in [('Recency', -1), ('Frequency', 0), ('Monetary', 0), ('CustomerID', 0)]:
        broken = data.copy()
        broken.loc[0, col] = bad_val
        with pytest.raises(ValueError):
            validate(broken)
    with pytest.raises(ValueError):
        validate(data.drop(columns='Monetary'))


def test_raw_transaction_cleaning():
    raw = pd.DataFrame({
        'InvoiceNo': ['536365', 'C536366', '536367', '536368', '536369'],
        'StockCode': ['85123A', '85123A', '84406B', '84029G', '84029E'],
        'Description': ['A', 'A', 'B', 'C', 'D'],
        'Quantity': [6, -2, 4, 0, 3],
        'InvoiceDate': pd.to_datetime(['2011-12-01', '2011-12-02', '2011-12-03', '2011-12-04', '2011-12-05']),
        'UnitPrice': [2.5, 2.5, 3.0, 4.0, 5.0],
        'CustomerID': [17850.0, 17850.0, np.nan, 13047.0, 17850.0],
        'Country': ['United Kingdom'] * 5,
    })
    rfm, stats = build_rfm_from_raw(raw)
    assert stats['missing_customer_id'] == 1
    assert stats['cancelled_invoices'] == 1
    assert stats['invalid_qty_price'] == 1
    assert stats['clean_transactions'] == 2
    assert len(rfm) == 1 and int(rfm.iloc[0]['Frequency']) == 2


def test_storage_is_atomic_and_batch_idempotent(tmp_path):
    db = tmp_path / 'test.sqlite3'
    assert initialize(db) == 4338
    assert initialize(db) == 0
    sample = pd.DataFrame([[99001, 10, 2, 300.0, 50, 'United Kingdom']], columns=COLUMNS)
    assert add_batch(sample, 'test', db) == 1
    assert add_batch(sample, 'test again', db) == 0
    invalid = pd.concat([sample, sample], ignore_index=True)
    invalid.loc[1, 'Monetary'] = -5
    with pytest.raises(ValueError):
        add_batch(invalid, 'bad', db)
    assert len(load_data(db)) == 4339


def test_noise_and_degenerate_clusters():
    X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]])
    assert np.isnan(evaluate(X, np.array([-1] * 4))['silhouette'])
    assert np.isnan(evaluate(X, np.array([0] * 4))['silhouette'])
    scores = evaluate(X, np.array([0, 0, 1, 1]))
    assert scores['silhouette'] > .8 and scores['noise_rate'] == 0


def test_full_training_artifacts(tmp_path):
    data = validate(pd.read_csv(SOURCE))
    folder = train(data, tmp_path)
    assert (tmp_path / 'latest.txt').read_text() == folder.name
    assignments = pd.read_csv(folder / 'assignments.csv')
    assert len(assignments) == 4338
    assert assignments.suggestion.notna().all()
    best = pd.read_csv(folder / 'best_per_algorithm.csv')
    assert set(best.algorithm) == {'KMeans', 'Hierarchical', 'DBSCAN'}
    assert best.eligible.all()
    assert (folder / 'model.joblib').exists()
    assert (folder / 'RUN_REPORT.md').exists()
    for algorithm in ['KMeans', 'Hierarchical', 'DBSCAN']:
        detail = folder / 'algorithms' / algorithm
        assert (detail / 'model.joblib').exists()
        assert len(pd.read_csv(detail / 'assignments.csv')) == 4338


def test_noise_profile_is_not_marketed():
    data = validate(pd.read_csv(SOURCE)).head(10)
    segments, *_ = profiles(data, np.array([-1] * 5 + [0] * 5))
    assert 'xem xét riêng' in segments.loc[segments.cluster == -1, 'suggestion'].iloc[0]


def test_streamlit_initial_render_and_new_customer(tmp_path, monkeypatch):
    import storage
    db = tmp_path / 'ui.sqlite3'
    originals = {name: getattr(storage, name) for name in ['initialize', 'load_data', 'add_batch', 'history']}
    monkeypatch.setattr(storage, 'initialize', lambda: originals['initialize'](db))
    monkeypatch.setattr(storage, 'load_data', lambda: originals['load_data'](db))
    monkeypatch.setattr(storage, 'history', lambda: originals['history'](db))
    monkeypatch.setattr(storage, 'add_batch', lambda raw, source, target=db: originals['add_batch'](raw, source, target))
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=120)
    assert not app.exception
    assert len(app.tabs) == 4

    before = len(storage.load_data())
    next(b for b in app.button if b.label == 'Lưu & Dự đoán phân khúc').click().run(timeout=120)
    assert not app.exception
    assert any('xác nhận' in item.value for item in app.warning)
    assert len(storage.load_data()) == before

    next(item for item in app.checkbox if 'khách hàng mới' in item.label).check()
    next(b for b in app.button if b.label == 'Lưu & Dự đoán phân khúc').click().run(timeout=120)
    assert not app.exception
    assert any('Monetary' in item.value for item in app.error)

    app.number_input(key='in_Monetary').set_value(500.)
    next(b for b in app.button if b.label == 'Lưu & Dự đoán phân khúc').click().run(timeout=120)
    assert not app.exception
    assert len(storage.load_data()) == before + 1


def test_canonical_hash_survives_sqlite_and_roundtrip(tmp_path):
    raw = pd.read_csv(SOURCE)
    db = tmp_path / 'hash.sqlite3'
    initialize(db)
    assert data_fingerprint(raw) == data_fingerprint(load_data(db))
    assert add_batch(raw.copy(), 'same values', db) == 0
    changed = raw.copy()
    changed.loc[0, 'Monetary'] += 10
    assert data_fingerprint(raw) != data_fingerprint(changed)


def test_iqr_outliers_and_scaler_comparison():
    from pipeline import detect_outliers_iqr, compare_scalers
    data = validate(pd.read_csv(SOURCE))
    mask, bounds = detect_outliers_iqr(data, method='iqr')
    assert 500 < mask.sum() < 800 and bounds is not None
    comp = compare_scalers(data, outlier_method='iqr')
    assert comp.scaler.nunique() == 2 and set(comp.algorithm) == {'KMeans', 'Hierarchical', 'DBSCAN'}


def test_segment_names_and_new_customer_prediction(tmp_path):
    from pipeline import predict_segments
    data = validate(pd.read_csv(SOURCE))
    folder = train(data, tmp_path, compare=False)
    segments = pd.read_csv(folder / 'segments.csv')
    assert segments.segment.is_unique and segments.suggestion.notna().all()
    sample = pd.DataFrame([
        [99001, 15, 2, 450.0, 220, 'United Kingdom'],
        [99002, 10, 80, 150000.0, 50000, 'United Kingdom'],
    ], columns=COLUMNS)
    result = predict_segments(sample, folder)
    assert len(result) == 2 and result.segment.notna().all()
    assert not bool(result.is_outlier.iloc[0])
    assert bool(result.is_outlier.iloc[1]) and 'Ngoại lai' in result.suggestion.iloc[1]
    assert predict_segments(sample, tmp_path / 'missing') is None


def test_retrain_policy(tmp_path):
    from datetime import datetime, timedelta, timezone
    from pipeline import retrain_due
    assert retrain_due(4338, tmp_path)[0]
    train(validate(pd.read_csv(SOURCE)), tmp_path, compare=False)
    assert not retrain_due(4338, tmp_path)[0]
    assert not retrain_due(4350, tmp_path, min_new=20)[0]
    assert retrain_due(4358, tmp_path, min_new=20)[0]
    later = datetime.now(timezone.utc) + timedelta(days=8)
    assert retrain_due(4339, tmp_path, max_days=7, now=later)[0]
    assert not retrain_due(4338, tmp_path, max_days=7, now=later)[0]
