"""Pipeline phân cụm RFM cho bộ dữ liệu Online Retail (UCI 352).

Đề tài: Phân khúc khách hàng bán lẻ (mô hình RFM) để gợi ý khuyến mãi.
Luồng chuẩn: Làm sạch giao dịch (nếu từ raw Excel) -> Tổng hợp RFM (Recency, Frequency, Monetary)
             -> Loại ngoại lai (IQR) -> Chuẩn hóa (StandardScaler / MinMaxScaler)
             -> K-Means / Hierarchical (Ward) / DBSCAN -> Đánh giá (Elbow, Silhouette)
             -> Đặt tên phân khúc RFM & Gợi ý khuyến mãi -> Lưu phiên kết quả.
"""
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

FEATURES = ['Recency', 'Frequency', 'Monetary']
COLUMNS = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'TotalItems', 'Country']
FEATURE_VI = {
    'Recency': 'Số ngày từ lần mua cuối (Recency)',
    'Frequency': 'Số đơn hàng đã mua (Frequency)',
    'Monetary': 'Tổng chi tiêu £ (Monetary)',
}
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data' / 'online_retail_rfm.csv'
RAW_EXCEL = ROOT / 'data' / 'Online Retail.xlsx'

# Bắt buộc >= 3 cụm cho phân khúc RFM để tách rõ: Trung thành / Mới-Tiềm năng / Nguy cơ rời bỏ.
MIN_SEGMENTS = 3


# --------------------------------------------------------------------------- 1. Làm sạch & Tổng hợp RFM
def build_rfm_from_raw(raw_df):
    """Làm sạch 541,909 giao dịch thô và tổng hợp thành bảng RFM theo CustomerID."""
    stats = {'raw_rows': int(len(raw_df))}
    df = raw_df.dropna(subset=['CustomerID']).copy()
    stats['missing_customer_id'] = stats['raw_rows'] - int(len(df))
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    cancelled = df['InvoiceNo'].str.startswith('C')
    stats['cancelled_invoices'] = int(cancelled.sum())
    df = df[~cancelled].copy()
    invalid_qp = (df['Quantity'] <= 0) | (df['UnitPrice'] <= 0)
    stats['invalid_qty_price'] = int(invalid_qp.sum())
    df = df[~invalid_qp].copy()
    stats['clean_transactions'] = int(len(df))

    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    ref_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: int((ref_date - x.max()).days)),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalPrice', 'sum'),
        TotalItems=('Quantity', 'sum'),
        Country=('Country', lambda x: str(x.mode().iloc[0])),
    ).reset_index()
    rfm['CustomerID'] = rfm['CustomerID'].astype(int)
    stats['rfm_customers'] = int(len(rfm))
    return rfm, stats


def validate(raw):
    if raw.columns.duplicated().any():
        raise ValueError('Tên cột bị lặp; cần mỗi cột xuất hiện đúng một lần.')
    missing = set(COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f'Thiếu cột: {sorted(missing)}')
    frame = raw[COLUMNS].copy()
    if frame.empty:
        raise ValueError('Dữ liệu rỗng.')
    num_cols = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'TotalItems']
    frame[num_cols] = frame[num_cols].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(frame[num_cols].to_numpy(dtype=float)).all():
        raise ValueError('Dữ liệu số có ô thiếu, chữ hoặc giá trị vô hạn.')
    if (frame['CustomerID'] <= 0).any():
        raise ValueError('CustomerID phải là số nguyên dương.')
    if (frame['Recency'] < 0).any():
        raise ValueError('Recency (số ngày) không được âm.')
    if (frame['Frequency'] < 1).any():
        raise ValueError('Frequency (số đơn hàng) phải >= 1.')
    if (frame['Monetary'] <= 0).any():
        raise ValueError('Monetary (tổng chi tiêu) phải > 0.')
    if (frame['TotalItems'] < 1).any():
        raise ValueError('TotalItems (tổng số sản phẩm) phải >= 1.')
    frame['Country'] = frame['Country'].fillna('Unknown').astype(str).str.strip()
    frame[['CustomerID', 'Recency', 'Frequency', 'TotalItems']] = frame[['CustomerID', 'Recency', 'Frequency', 'TotalItems']].astype(int)
    frame['Monetary'] = frame['Monetary'].astype(float)
    return frame.reset_index(drop=True)


def data_fingerprint(frame):
    """Mã băm chuẩn hóa dùng chung giữa CSV, SQLite, ứng dụng và notebook."""
    clean = validate(frame)
    payload = clean.to_json(orient='split', index=False, double_precision=10)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


# --------------------------------------------------------------------------- 2. Tiền xử lý (IQR + Scaling)
def detect_outliers_iqr(frame, method='iqr'):
    """Phát hiện ngoại lai theo quy tắc IQR (1.5 * IQR) trên Recency, Frequency, Monetary."""
    if method == 'none':
        return np.zeros(len(frame), dtype=bool), None
    values = np.log1p(frame[FEATURES]) if method == 'iqr_log' else frame[FEATURES].astype(float)
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = ((values < lower) | (values > upper)).any(axis=1).to_numpy()
    if (~mask).sum() < 15:
        return np.zeros(len(frame), dtype=bool), None
    bounds = {'method': method, 'lower': lower.to_dict(), 'upper': upper.to_dict()}
    return mask, bounds


def make_scaler(scaler_type='standard', use_log=False):
    scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
    if use_log:
        return Pipeline([('log', FunctionTransformer(np.log1p, validate=True)), ('scale', scaler)])
    return Pipeline([('scale', scaler)])


def prepare(frame, scaler_type='standard', outlier_method='iqr', use_log=False):
    outlier, bounds = detect_outliers_iqr(frame, method=outlier_method)
    prep = make_scaler(scaler_type=scaler_type, use_log=use_log)
    prep.fit(frame.loc[~outlier, FEATURES])
    X = prep.transform(frame[FEATURES])
    return prep, X, outlier, bounds


# --------------------------------------------------------------------------- 3. Đánh giá & Thuật toán
def evaluate(X, labels):
    mask = labels != -1
    count = len(set(labels[mask]))
    coverage = float(mask.mean())
    result = dict(clusters=count, noise_rate=1 - coverage, silhouette=np.nan, davies_bouldin=np.nan, calinski_harabasz=np.nan)
    if 2 <= count < int(mask.sum()):
        result.update(
            silhouette=float(silhouette_score(X[mask], labels[mask])),
            davies_bouldin=float(davies_bouldin_score(X[mask], labels[mask])),
            calinski_harabasz=float(calinski_harabasz_score(X[mask], labels[mask])),
        )
    return result


def elbow_k(ks, inertias):
    """Điểm khuỷu (Elbow Method): điểm xa nhất so với đường thẳng nối hai đầu đường cong inertia."""
    x = np.asarray(ks, dtype=float)
    y = np.asarray(inertias, dtype=float)
    if len(x) < 3 or y[0] == y[-1]:
        return int(x[0])
    xn = (x - x[0]) / (x[-1] - x[0])
    yn = (y - y[-1]) / (y[0] - y[-1])
    return int(x[int(np.argmax((1 - xn) - yn))])


def make_model(algorithm, params):
    if algorithm == 'KMeans':
        return KMeans(n_clusters=int(params['k']), random_state=42, n_init=20)
    if algorithm == 'Hierarchical':
        return AgglomerativeClustering(n_clusters=int(params['k']), linkage='ward')
    if algorithm == 'DBSCAN':
        return DBSCAN(eps=float(params['eps']), min_samples=int(params['min_samples']))
    raise ValueError('Thuật toán không hợp lệ')


def dbscan_grid(scaler_type):
    if scaler_type == 'minmax':
        eps_values = [0.06, 0.08, 0.1, 0.12, 0.15]
    else:
        eps_values = [0.35, 0.4, 0.45, 0.5, 0.6]
    return [('DBSCAN', {'eps': eps, 'min_samples': m}) for eps in eps_values for m in [10, 20, 30, 50]]


@dataclass
class SearchResult:
    preprocessor: object
    X: np.ndarray
    outlier: np.ndarray
    bounds: dict
    table: pd.DataFrame
    candidates: dict = field(default_factory=dict)
    best: pd.DataFrame = None
    selected: int = 0
    elbow: int = 0
    scaler_type: str = 'standard'
    outlier_method: str = 'iqr'
    use_log: bool = False

    @property
    def X_fit(self):
        return self.X[~self.outlier]


def search(frame, scaler_type='standard', outlier_method='iqr', use_log=False):
    frame = validate(frame)
    if len(frame) < 20:
        raise ValueError('Cần ít nhất 20 khách hàng để khảo sát phân cụm.')
    prep, X, outlier, bounds = prepare(frame, scaler_type=scaler_type, outlier_method=outlier_method, use_log=use_log)
    X_fit = X[~outlier]
    rows, candidates = [], {}
    kmax = min(9, len(X_fit))
    configs = [(name, {'k': k}) for name in ['KMeans', 'Hierarchical'] for k in range(2, kmax)]
    configs += dbscan_grid(scaler_type)
    for i, (algorithm, params) in enumerate(configs):
        model = make_model(algorithm, params)
        start = time.perf_counter()
        labels = model.fit_predict(X_fit)
        elapsed = time.perf_counter() - start
        scores = evaluate(X_fit, labels)
        counts = pd.Series(labels[labels != -1]).value_counts()
        min_c = 2 if algorithm == 'DBSCAN' else MIN_SEGMENTS
        eligible = (
            scores['clusters'] >= min_c
            and scores['noise_rate'] <= 0.30
            and len(counts) > 0
            and counts.min() >= 5
            and np.isfinite(scores['silhouette'])
        )
        rows.append(dict(
            candidate=i, algorithm=algorithm, params=json.dumps(params), seconds=elapsed,
            inertia=float(model.inertia_) if algorithm == 'KMeans' else np.nan,
            min_cluster=int(counts.min()) if len(counts) else 0, eligible=bool(eligible), **scores,
        ))
        candidates[i] = (model, labels)
    table = pd.DataFrame(rows)
    ranked = table[table.eligible].sort_values(['silhouette', 'davies_bouldin'], ascending=[False, True])
    if ranked.empty:
        ranked = table[table.silhouette.notna()].sort_values(['silhouette', 'davies_bouldin'], ascending=[False, True])
    if ranked.empty:
        raise ValueError('Không có cấu hình phân cụm hợp lệ. Hãy kiểm tra dữ liệu.')
    representatives = []
    for name in ['KMeans', 'Hierarchical', 'DBSCAN']:
        options = ranked[ranked.algorithm == name]
        if options.empty:
            options = table[table.algorithm == name].sort_values(['silhouette', 'davies_bouldin'], ascending=[False, True], na_position='last')
        representatives.append(options.head(1))
    best = pd.concat(representatives, ignore_index=True)
    full_cov = ranked[ranked.algorithm.isin(['KMeans', 'Hierarchical'])]
    selected_id = int(full_cov.iloc[0].candidate) if not full_cov.empty else int(ranked.iloc[0].candidate)
    km = table[table.algorithm == 'KMeans']
    elbow = elbow_k(km.params.map(lambda p: json.loads(p)['k']), km.inertia)
    return SearchResult(prep, X, outlier, bounds, table, candidates, best, selected_id, elbow, scaler_type, outlier_method, use_log)


def centroids(X, labels):
    return {int(c): X[labels == c].mean(axis=0) for c in sorted(set(labels)) if c != -1}


def assign_nearest(X, cents):
    keys = sorted(cents)
    matrix = np.array([cents[k] for k in keys])
    dist = np.linalg.norm(X[:, None, :] - matrix[None, :, :], axis=2)
    return np.array(keys)[dist.argmin(axis=1)], dist.min(axis=1)


def full_labels(res, labels_fit):
    labels = np.full(len(res.X), -1, dtype=int)
    labels[~res.outlier] = labels_fit
    cents = centroids(res.X_fit, labels_fit)
    if res.outlier.any() and cents:
        labels[res.outlier] = assign_nearest(res.X[res.outlier], cents)[0]
    return labels


def compare_scalers(frame, outlier_method='iqr'):
    """So sánh 2 bộ chuẩn hóa theo đề cương: StandardScaler vs MinMaxScaler."""
    parts = []
    for scaler_type, label in [('standard', 'StandardScaler (Z-score)'), ('minmax', 'MinMaxScaler [0, 1]')]:
        res = search(frame, scaler_type=scaler_type, outlier_method=outlier_method, use_log=False)
        part = res.best.copy()
        part.insert(0, 'scaler', label)
        part['rows_used'] = int((~res.outlier).sum())
        part['elbow_k'] = res.elbow
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- 4. Hồ sơ RFM & Gợi ý khuyến mãi
def describe_segment(cluster, ratio_row, mean_row):
    """Đặt tên phân khúc RFM và gợi ý khuyến mãi dựa trên R, F, M tương đối so với trung bình."""
    r_ratio, f_ratio, m_ratio = float(ratio_row['Recency']), float(ratio_row['Frequency']), float(ratio_row['Monetary'])
    r_val, f_val, m_val = float(mean_row['Recency']), float(mean_row['Frequency']), float(mean_row['Monetary'])

    if r_ratio >= 1.35:
        kind = 'Khách hàng ngủ quên / Nguy cơ rời bỏ'
        offer = 'Gửi Email/SMS tái kích hoạt kèm Voucher giảm giá 15% có thời hạn 7 ngày để kéo khách quay lại.'
    elif f_ratio >= 1.35 or m_ratio >= 1.35:
        kind = 'Khách hàng trung thành / Chi tiêu cao'
        offer = 'Tặng thẻ thành viên VIP, tích điểm nhân đôi và ưu tiên giảm giá cho đơn hàng tiếp theo.'
    elif r_ratio <= 0.85 and f_ratio < 1.1:
        kind = 'Khách hàng mới / Tiềm năng phát triển'
        offer = 'Tặng mã miễn phí vận chuyển (Freeship) và ưu đãi combo cho đơn hàng thứ 2 để tạo thói quen mua.'
    else:
        kind = 'Khách hàng phổ thông / Mua sắm ổn định'
        offer = 'Ưu đãi theo ngưỡng giá trị đơn hàng (giảm 10% cho đơn đạt mức tối thiểu) và giới thiệu sản phẩm bán chạy.'

    rationale = f'Trung bình: mua cách đây {r_val:.0f} ngày (R={r_ratio:.2f}x), {f_val:.1f} đơn (F={f_ratio:.2f}x), chi £{m_val:,.0f} (M={m_ratio:.2f}x).'
    return f'Nhóm {cluster} — {kind}', offer, rationale


NOISE_TEXT = 'Điểm nhiễu (DBSCAN): xem xét riêng trước khi gửi khuyến mãi.'
OUTLIER_TEXT = 'Khách hàng VIP / Mua buôn đột biến (Ngoại lai IQR): chăm sóc riêng 1-1 & chiết khấu theo đơn hàng lớn. (Gần với {segment}).'


def profiles(frame, labels, outlier=None):
    outlier = np.zeros(len(frame), dtype=bool) if outlier is None else np.asarray(outlier, dtype=bool)
    data = frame.copy()
    data['cluster'] = labels
    core = data[~outlier]
    means = core.groupby('cluster')[FEATURES].mean()
    medians = core.groupby('cluster')[FEATURES].median()
    overall = core[FEATURES].mean().replace(0, np.nan)
    ratios = means.div(overall)
    rows = []
    for cluster in means.index:
        count = int((labels == cluster).sum())
        out_count = int(((labels == cluster) & outlier).sum())
        if cluster == -1:
            segment, offer, rationale = 'Ngoại lệ / nhiễu (DBSCAN)', NOISE_TEXT, 'Không nằm trong cụm mật độ nào.'
        else:
            segment, offer, rationale = describe_segment(int(cluster), ratios.loc[cluster], means.loc[cluster])
        rows.append({
            'cluster': int(cluster), 'customers': count, 'outliers_assigned': out_count,
            'segment': segment, 'suggestion': offer, 'rationale': rationale,
        })
    return pd.DataFrame(rows), means, medians, ratios


def build_assignments(frame, labels, outlier, coords, segments):
    out = frame.copy()
    out['cluster'] = labels
    out['is_outlier'] = np.asarray(outlier, dtype=bool)
    out[['PC1', 'PC2']] = coords
    out = out.merge(segments[['cluster', 'segment', 'suggestion']], on='cluster', how='left')
    mask = out.is_outlier.to_numpy()
    out.loc[mask, 'suggestion'] = [OUTLIER_TEXT.format(segment=s) for s in out.loc[mask, 'segment']]
    return out


# --------------------------------------------------------------------------- 5. Huấn luyện & Lưu phiên
def train(frame, output_root=None, scaler_type='standard', outlier_method='iqr', use_log=False,
          algorithm='Auto', allow_ineligible=False, compare=True):
    frame = validate(frame)
    res = search(frame, scaler_type=scaler_type, outlier_method=outlier_method, use_log=use_log)
    table, best, chosen = res.table, res.best, res.selected
    if algorithm != 'Auto':
        options = best[(best.algorithm == algorithm) & (best.eligible | allow_ineligible)]
        if options.empty:
            raise ValueError(f'{algorithm} không có cấu hình đạt tiêu chí.')
        chosen = int(options.iloc[0].candidate)
    selected = table.loc[table.candidate == chosen].iloc[0]
    model, labels_fit = res.candidates[chosen]
    labels = full_labels(res, labels_fit)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    root = Path(output_root or ROOT / 'artifacts')
    folder = root / run_id
    folder.mkdir(parents=True)
    segment, means, medians, ratios = profiles(frame, labels, res.outlier)
    pca = PCA(n_components=2).fit(res.X_fit)
    coords = pca.transform(res.X)
    assignments = build_assignments(frame, labels, res.outlier, coords, segment)

    algorithm_root = folder / 'algorithms'
    for rep in best.itertuples():
        target = algorithm_root / rep.algorithm
        target.mkdir(parents=True)
        rep_model, rep_fit = res.candidates[int(rep.candidate)]
        rep_labels = full_labels(res, rep_fit)
        rep_segments, rep_means, rep_medians, rep_ratios = profiles(frame, rep_labels, res.outlier)
        rep_assignments = build_assignments(frame, rep_labels, res.outlier, coords, rep_segments)
        for name, values in {
            'assignments': rep_assignments, 'segments': rep_segments,
            'means': rep_means.reset_index(), 'medians': rep_medians.reset_index(),
            'relative_spending': rep_ratios.reset_index(),
        }.items():
            values.to_csv(target / f'{name}.csv', index=False, encoding='utf-8-sig')
        joblib.dump({'preprocessor': res.preprocessor, 'model': rep_model, 'pca': pca,
                     'algorithm': rep.algorithm, 'params': json.loads(rep.params)}, target / 'model.joblib')

    extra = {}
    if compare:
        extra['scaler_comparison'] = compare_scalers(frame, outlier_method=outlier_method)
    for name, df in {
        'search': table, 'best_per_algorithm': best, 'assignments': assignments, 'segments': segment,
        'means': means.reset_index(), 'medians': medians.reset_index(), 'relative_spending': ratios.reset_index(),
        **extra,
    }.items():
        df.to_csv(folder / f'{name}.csv', index=False, encoding='utf-8-sig')
    frame.to_csv(folder / 'training_snapshot.csv', index=False, encoding='utf-8-sig')

    digest = data_fingerprint(frame)
    import sklearn
    metadata = dict(
        run_id=run_id, created_utc=datetime.now(timezone.utc).isoformat(), rows=len(frame), data_sha256=digest,
        algorithm=selected.algorithm, params=json.loads(selected.params),
        scaler_type=scaler_type, outlier_method=outlier_method, use_log=use_log,
        eligible=bool(selected.eligible), outliers=int(res.outlier.sum()), elbow_k=res.elbow,
        selection_policy=f'Silhouette cao nhất (Davies-Bouldin phá hòa); >={MIN_SEGMENTS} cụm (DBSCAN >=2), nhiễu <=30%',
        metrics=evaluate(res.X_fit, labels_fit), pca_explained_variance=pca.explained_variance_ratio_.tolist(),
        versions={'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__},
    )
    safe_metadata = json.loads(json.dumps(metadata), parse_constant=lambda _: None)
    (folder / 'metadata.json').write_text(json.dumps(safe_metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    joblib.dump({
        'preprocessor': res.preprocessor, 'model': model, 'pca': pca, 'metadata': metadata,
        'centroids': centroids(res.X_fit, labels_fit), 'outlier_bounds': res.bounds, 'segments': segment,
    }, folder / 'model.joblib')

    report = [
        '# Báo cáo phiên huấn luyện RFM', '',
        f"Phiên: {run_id}; UTC: {metadata['created_utc']}",
        f"Dữ liệu: {len(frame)} khách hàng (loại {int(res.outlier.sum())} ngoại lai bằng {outlier_method} khi huấn luyện).",
        f"Chuẩn hóa: {scaler_type}" + (' + log1p' if use_log else '') + '.', '',
        '## So sánh ba thuật toán', '',
        '```text', best[['algorithm', 'params', 'clusters', 'noise_rate', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'eligible']].to_string(index=False), '```', '',
        f"Mô hình chính: {selected.algorithm} {selected.params}; Elbow gợi ý k={res.elbow}.", '',
        '## Hồ sơ phân khúc RFM & gợi ý khuyến mãi', '',
        '```text', segment[['cluster', 'customers', 'segment', 'suggestion']].to_string(index=False), '```',
    ]
    (folder / 'RUN_REPORT.md').write_text('\n'.join(report), encoding='utf-8')

    pointer = root / f'.latest_{run_id}.tmp'
    pointer.write_text(run_id, encoding='utf-8')
    pointer.replace(root / 'latest.txt')
    return folder


# --------------------------------------------------------------------------- 6. Dự đoán cho khách mới & Lịch định kỳ
def latest_run(root=None):
    root = Path(root or ROOT / 'artifacts')
    pointer = root / 'latest.txt'
    if not pointer.exists():
        return None
    folder = root / pointer.read_text(encoding='utf-8').strip()
    return folder if (folder / 'metadata.json').exists() and (folder / 'model.joblib').exists() else None


def predict_segments(frame, folder=None):
    """Gán phân khúc RFM và gợi ý khuyến mãi ngay cho khách hàng mới."""
    folder = Path(folder) if folder else latest_run()
    if folder is None or not (folder / 'model.joblib').exists():
        return None
    bundle = joblib.load(folder / 'model.joblib')
    if 'centroids' not in bundle:
        return None
    clean = validate(frame)
    X = bundle['preprocessor'].transform(clean[FEATURES])
    labels, dist = assign_nearest(X, bundle['centroids'])

    bounds = bundle.get('outlier_bounds')
    if bounds:
        vals = np.log1p(clean[FEATURES]) if bounds['method'] == 'iqr_log' else clean[FEATURES]
        lower = pd.Series(bounds['lower'])
        upper = pd.Series(bounds['upper'])
        is_outlier = ((vals < lower) | (vals > upper)).any(axis=1).to_numpy()
    else:
        is_outlier = np.zeros(len(clean), dtype=bool)

    seg = bundle['segments'].set_index('cluster')
    result = clean.copy()
    result['cluster'] = labels
    result['segment'] = seg.loc[labels, 'segment'].to_numpy()
    result['suggestion'] = seg.loc[labels, 'suggestion'].to_numpy()
    result['is_outlier'] = is_outlier
    result.loc[is_outlier, 'suggestion'] = [OUTLIER_TEXT.format(segment=s) for s in result.loc[is_outlier, 'segment']]
    result['distance_to_centroid'] = dist
    return result


def retrain_due(rows_now, root=None, min_new=20, max_days=7, now=None):
    """Chính sách huấn luyện định kỳ: khi có >= N khách mới hoặc quá X ngày kể từ lần huấn luyện trước."""
    folder = latest_run(root)
    if folder is None:
        return True, 'Chưa có mô hình nào.'
    meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    new = rows_now - int(meta['rows'])
    age = (now or datetime.now(timezone.utc)) - datetime.fromisoformat(meta['created_utc'])
    if new >= min_new:
        return True, f'Đã có {new} khách hàng mới (ngưỡng {min_new}).'
    if new > 0 and age.days >= max_days:
        return True, f'Đã {age.days} ngày kể từ lần huấn luyện trước và có {new} khách hàng mới.'
    return False, f'Chưa đến hạn: {max(new, 0)} khách mới (ngưỡng {min_new}), {age.days}/{max_days} ngày.'
