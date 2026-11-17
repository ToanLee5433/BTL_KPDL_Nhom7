import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from pipeline import validate, COLUMNS, FEATURES, SOURCE, evaluate, search, profiles, train, data_fingerprint
from storage import initialize, load_data, add_batch


def test_source_and_validation():
    data = validate(pd.read_csv(SOURCE))
    assert data.shape == (440, 8)
    assert len(FEATURES) == 6
    for value in [-1, np.nan, np.inf, 'wrong']:
        broken = data.astype(object).copy()
        broken.loc[0, 'Fresh'] = value
        with pytest.raises(ValueError):
            validate(broken)
    with pytest.raises(ValueError):
        validate(data.drop(columns='Channel'))
    broken = data.copy()
    broken.loc[0, 'Region'] = 4
    with pytest.raises(ValueError):
        validate(broken)


def test_storage_is_atomic_and_batch_idempotent(tmp_path):
    db = tmp_path / 'test.sqlite3'
    assert initialize(db) == 440
    assert initialize(db) == 0
    sample = pd.DataFrame([[1, 1, 10, 20, 30, 40, 50, 60]], columns=COLUMNS)
    assert add_batch(sample, 'test', db) == 1
    assert add_batch(sample, 'test again', db) == 0
    invalid = pd.concat([sample, sample], ignore_index=True)
    invalid.loc[1, 'Milk'] = -1
    with pytest.raises(ValueError):
        add_batch(invalid, 'bad', db)
    assert len(load_data(db)) == 441


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
    assert len(assignments) == 440
    assert assignments.suggestion.notna().all()
    table = pd.read_csv(folder / 'search.csv')
    assert set(table.algorithm) == {'KMeans', 'Hierarchical', 'DBSCAN'}
    best = pd.read_csv(folder / 'best_per_algorithm.csv')
    assert set(best.algorithm) == {'KMeans', 'Hierarchical', 'DBSCAN'}
    assert best.eligible.all()
    assert (folder / 'model.joblib').exists()
    assert (folder / 'RUN_REPORT.md').exists()
    for algorithm in ['KMeans', 'Hierarchical', 'DBSCAN']:
        detail = folder / 'algorithms' / algorithm
        assert (detail / 'model.joblib').exists()
        assert len(pd.read_csv(detail / 'assignments.csv')) == 440
        assert (detail / 'channel_distribution.csv').exists()


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
    assert any('chi tiêu dương' in item.value for item in app.error)

    app.number_input(key='in_Fresh').set_value(12345.)
    next(b for b in app.button if b.label == 'Lưu & Dự đoán phân khúc').click().run(timeout=120)
    assert not app.exception
    assert len(storage.load_data()) == before + 1


def test_canonical_hash_survives_sqlite_and_roundtrip(tmp_path):
    raw = pd.read_csv(SOURCE)
    db = tmp_path / 'hash.sqlite3'
    initialize(db)
    assert data_fingerprint(raw) == data_fingerprint(load_data(db))
    assert add_batch(raw.astype(float), 'same values different dtype', db) == 0
    changed = raw.copy()
    changed.loc[0, 'Fresh'] += 1
    assert data_fingerprint(raw) != data_fingerprint(changed)


def test_new_customer_retraining_snapshot(tmp_path):
    db = tmp_path / 'new.sqlite3'
    initialize(db)
    sample = pd.DataFrame([[2, 3, 1001, 2002, 3003, 4004, 5005, 6006]], columns=COLUMNS)
    add_batch(sample, 'new customer', db)
    dataset = load_data(db)
    folder = train(dataset, tmp_path / 'models')
    import json
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    assert metadata['rows'] == 441
    assert metadata['data_sha256'] == data_fingerprint(dataset)
    snapshot = pd.read_csv(folder / 'training_snapshot.csv')
    assert data_fingerprint(snapshot) == metadata['data_sha256']
    assert len(pd.read_csv(folder / 'assignments.csv')) == 441


def test_zero_spending_and_duplicate_columns():
    data = pd.read_csv(SOURCE).head(10)
    zero = data.copy()
    zero.loc[0, FEATURES] = 0
    with pytest.raises(ValueError, match='chi tiêu dương'):
        validate(zero)
    with pytest.raises(ValueError, match='Tên cột'):
        validate(pd.concat([data, data[['Fresh']]], axis=1))


def test_campaign_demo_rules():
    from experience import campaign_draft
    customer = pd.Series({'row_id': 12, 'cluster': 0})
    profile = pd.Series({'segment': 'Test segment'})
    ratios = pd.DataFrame([[2., 1., .5, 3., .1, .2]], columns=FEATURES, index=[0])
    draft = campaign_draft(customer, profile, ratios, 'test-run')
    assert draft['product_categories'] == ['Hàng đông lạnh', 'Hàng tươi sống']
    assert draft['customer_ref'] == 'KH-0012'
    assert draft['status'].startswith('Bản nháp')
    assert campaign_draft(customer, profile, ratios, 'test-run', eligible=False) is None
    customer['cluster'] = -1
    assert campaign_draft(customer, profile, ratios, 'test-run') is None


def test_customer_demo_generate_and_switch():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=120)
    assert not app.exception
    next(b for b in app.button if b.label == 'Tạo đề xuất khuyến mãi').click().run(timeout=120)
    assert not app.exception
    assert app.session_state['offer_draft']['customer_ref'] == 'KH-0001'
    selector = next(s for s in app.selectbox if s.label == 'Chọn mã khách hàng')
    selector.select(2).run(timeout=120)
    assert not app.exception
    next(b for b in app.button if b.label == 'Tạo đề xuất khuyến mãi').click().run(timeout=120)
    assert app.session_state['offer_draft']['customer_ref'] == 'KH-0002'


def test_elbow_and_min_segments():
    from pipeline import elbow_k, MIN_SEGMENTS
    assert elbow_k([2, 3, 4, 5, 6], [100, 40, 30, 28, 27]) == 3
    assert MIN_SEGMENTS == 3


def test_iqr_outliers_and_scaler_comparison():
    from pipeline import detect_outliers_iqr, compare_scalers
    data = validate(pd.read_csv(SOURCE))
    mask, bounds = detect_outliers_iqr(data, method='iqr')
    assert 50 < mask.sum() < 150 and bounds is not None
    comp = compare_scalers(data, outlier_method='iqr')
    assert comp.scaler.nunique() == 2 and set(comp.algorithm) == {'KMeans', 'Hierarchical', 'DBSCAN'}


def test_segment_names_and_new_customer_prediction(tmp_path):
    from pipeline import predict_segments
    data = validate(pd.read_csv(SOURCE))
    folder = train(data, tmp_path, compare=False)
    segments = pd.read_csv(folder / 'segments.csv')
    assert segments.segment.is_unique and segments.suggestion.notna().all()
    assert (folder / 'scaler_comparison.csv').exists() is False
    sample = pd.DataFrame([
        [1, 3, 2000, 500, 800, 600, 200, 900],
        [1, 3, 5000000, 5000000, 5000000, 5000000, 5000000, 5000000],
    ], columns=COLUMNS)
    result = predict_segments(sample, folder)
    assert len(result) == 2 and result.segment.notna().all()
    assert not bool(result.is_outlier.iloc[0])
    assert bool(result.is_outlier.iloc[1]) and 'Ngoại lai' in result.suggestion.iloc[1]
    assert predict_segments(sample, tmp_path / 'missing') is None


def test_retrain_policy(tmp_path):
    from datetime import datetime, timedelta, timezone
    from pipeline import retrain_due
    assert retrain_due(440, tmp_path)[0]
    train(validate(pd.read_csv(SOURCE)), tmp_path, compare=False)
    assert not retrain_due(440, tmp_path)[0]
    assert not retrain_due(445, tmp_path, min_new=10)[0]
    assert retrain_due(450, tmp_path, min_new=10)[0]
    later = datetime.now(timezone.utc) + timedelta(days=8)
    assert retrain_due(441, tmp_path, max_days=7, now=later)[0]
    assert not retrain_due(440, tmp_path, max_days=7, now=later)[0]
