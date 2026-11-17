"""Pipeline phân cụm dùng chung cho notebook Colab, giao diện Streamlit và lịch huấn luyện lại.

Đề tài: Phân khúc khách hàng siêu thị để gợi ý khuyến mãi (Bài toán Phân cụm).
Dữ liệu: Wholesale Customers (UCI, 440 dòng, 6 nhóm chi tiêu + Channel, Region).
Luồng chuẩn: validate -> loại ngoại lai (IQR) -> chuẩn hóa (StandardScaler / MinMaxScaler)
             -> K-Means / Hierarchical (Agglomerative) / DBSCAN -> đánh giá (Elbow, Silhouette)
             -> hồ sơ phân khúc & gợi ý khuyến mãi -> lưu phiên kết quả.
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

FEATURES = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
COLUMNS = ['Channel', 'Region'] + FEATURES
CATEGORY_VI = {
    'Fresh': 'Hàng tươi sống',
    'Milk': 'Sữa & chế phẩm',
    'Grocery': 'Tạp hóa',
    'Frozen': 'Hàng đông lạnh',
    'Detergents_Paper': 'Chất tẩy rửa & giấy',
    'Delicassen': 'Thực phẩm chế biến',
}
FOOD = ['Fresh', 'Frozen', 'Delicassen']
RETAIL = ['Milk', 'Grocery', 'Detergents_Paper']
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data' / 'wholesale_customers.csv'

# Bắt buộc >= 3 cụm để phân khúc có ý nghĩa thiết kế khuyến mãi (k=2 chỉ tách 2 kênh Channel có sẵn).
MIN_SEGMENTS = 3


# --------------------------------------------------------------------------- 1. Kiểm tra dữ liệu
def validate(raw):
    if raw.columns.duplicated().any():
        raise ValueError('Tên cột bị lặp; cần mỗi cột xuất hiện đúng một lần.')
    missing = set(COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f'Thiếu cột: {sorted(missing)}')
    frame = raw[COLUMNS].apply(pd.to_numeric, errors='coerce')
    if frame.empty:
        raise ValueError('Dữ liệu rỗng.')
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError('Dữ liệu có ô thiếu, chữ hoặc giá trị vô hạn. Hãy sửa trước khi lưu.')
    if (frame[FEATURES] < 0).any().any():
        raise ValueError('Chi tiêu không được âm.')
    if (frame[FEATURES].sum(axis=1) == 0).any():
        raise ValueError('Mỗi khách hàng cần có chi tiêu dương ở ít nhất một nhóm hàng.')
    if not frame.Channel.isin([1, 2]).all() or not frame.Region.isin([1, 2, 3]).all():
        raise ValueError('Channel chỉ nhận 1/2; Region chỉ nhận 1/2/3.')
    frame[['Channel', 'Region']] = frame[['Channel', 'Region']].astype(int)
    frame[FEATURES] = frame[FEATURES].astype(float)
    return frame.reset_index(drop=True)


def data_fingerprint(frame):
    """Mã băm chuẩn hóa dùng chung giữa CSV, SQLite, ứng dụng và notebook."""
    clean = validate(frame)
    payload = clean.to_json(orient='split', index=False, double_precision=15)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


# --------------------------------------------------------------------------- 2. Tiền xử lý (IQR + Scaling)
def detect_outliers_iqr(frame, method='iqr'):
    """Phát hiện ngoại lai theo quy tắc IQR (1.5 * IQR).
    - 'iqr': tính IQR trực tiếp trên dữ liệu chi tiêu gốc.
    - 'iqr_log': tính IQR trên thang log1p (giữ lại nhiều điểm hơn).
    - 'none': không loại ngoại lai.
    """
    if method == 'none':
        return np.zeros(len(frame), dtype=bool), None
    values = np.log1p(frame[FEATURES]) if method == 'iqr_log' else frame[FEATURES].astype(float)
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = ((values < lower) | (values > upper)).any(axis=1).to_numpy()
    if (~mask).sum() < 10:
        return np.zeros(len(frame), dtype=bool), None
    bounds = {'method': method, 'lower': lower.to_dict(), 'upper': upper.to_dict()}
    return mask, bounds


def make_scaler(scaler_type='standard', use_log=False):
    """Tạo bộ chuẩn hóa dữ liệu: StandardScaler hoặc MinMaxScaler (có thể kèm log1p)."""
    scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
    if use_log:
        return Pipeline([('log', FunctionTransformer(np.log1p, validate=True)), ('scale', scaler)])
    return Pipeline([('scale', scaler)])


def prepare(frame, scaler_type='standard', outlier_method='iqr', use_log=False):
    """Tiền xử lý: phát hiện ngoại lai bằng IQR trên tập sạch -> fit scaler trên dữ liệu không ngoại lai."""
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
    """Lưới tham số DBSCAN phù hợp với thang đo của StandardScaler và MinMaxScaler."""
    if scaler_type == 'minmax':
        eps_values = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    else:
        eps_values = [0.75, 1.0, 1.1, 1.2, 1.35, 1.5, 1.75]
    return [('DBSCAN', {'eps': eps, 'min_samples': m}) for eps in eps_values for m in [3, 4, 5, 6]]


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
    if len(frame) < 15:
        raise ValueError('Cần ít nhất 15 khách hàng để khảo sát phân cụm.')
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
        eligible = (
            scores['clusters'] >= MIN_SEGMENTS
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
        # Dự phòng nếu tập dữ liệu thử nghiệm nhỏ không đạt ngưỡng 3 cụm
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
    # Ưu tiên chọn phương pháp phân cụm toàn bộ (KMeans/Hierarchical) làm mô hình chính nếu ở chế độ Auto
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
    """Nhãn cho mọi khách hàng. Khách ngoại lai (bị loại lúc train) được gán vào cụm gần nhất để tham khảo."""
    labels = np.full(len(res.X), -1, dtype=int)
    labels[~res.outlier] = labels_fit
    cents = centroids(res.X_fit, labels_fit)
    if res.outlier.any() and cents:
        labels[res.outlier] = assign_nearest(res.X[res.outlier], cents)[0]
    return labels


def compare_scalers(frame, outlier_method='iqr'):
    """So sánh 2 bộ chuẩn hóa theo đúng đề cương: StandardScaler vs MinMaxScaler."""
    parts = []
    for scaler_type, label in [('standard', 'StandardScaler (Z-score)'), ('minmax', 'MinMaxScaler [0, 1]')]:
        res = search(frame, scaler_type=scaler_type, outlier_method=outlier_method, use_log=False)
        part = res.best.copy()
        part.insert(0, 'scaler', label)
        part['rows_used'] = int((~res.outlier).sum())
        part['elbow_k'] = res.elbow
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- 4. Hồ sơ phân khúc & Gợi ý khuyến mãi
def describe_segment(cluster, ratio_row, total_ratio, retail_share):
    """Đặt tên phân khúc và gợi ý khuyến mãi siêu thị dựa trên mức chi tiêu tương đối."""
    top = ratio_row.dropna().sort_values(ascending=False).head(2).index
    top_vi = ', '.join(CATEGORY_VI[t] for t in top)
    food, retail = float(ratio_row[FOOD].mean()), float(ratio_row[RETAIL].mean())
    if total_ratio < 0.78:
        kind = 'Khách mua sắm tiết kiệm / quy mô nhỏ'
        offer = f'Tặng Voucher giảm giá cho đơn hàng tiếp theo và Combo tiết kiệm nhóm {top_vi} để tăng tần suất mua.'
    elif total_ratio > 1.35 and abs(food - retail) < 0.45 * max(food, retail):
        kind = 'Khách hàng lớn đa ngành (VIP)'
        offer = 'Áp dụng thẻ thành viên VIP, chiết khấu bậc thang theo hóa đơn lớn và miễn phí giao hàng tận nơi.'
    elif retail > 1.25 * food:
        kind = 'Nhóm ưu tiên Tạp hóa & Gia dụng'
        offer = f'Khuyến mãi mua nhiều giảm giá (mua 2 tặng 1 / giảm 10% theo thùng) cho nhóm {top_vi}.'
    elif food > 1.25 * retail:
        kind = 'Nhóm ưu tiên Thực phẩm Tươi & Đông lạnh'
        offer = f'Ưu đãi Combo thực phẩm tươi sống + đông lạnh hàng tuần, giảm giá giờ vàng cho {top_vi}.'
    else:
        kind = 'Khách hàng cân bằng nhiều nhóm hàng'
        offer = f'Ưu đãi theo ngưỡng giá trị hóa đơn kèm quà tặng dùng thử ở nhóm {top_vi}.'
    channel = f'{retail_share:.0%} thuộc kênh bán lẻ' if retail_share == retail_share else ''
    rationale = f'Chi tiêu trung bình bằng {total_ratio:.2f} lần mức chung; nổi bật nhất: {top_vi}.' + (f' {channel}.' if channel else '')
    return f'Nhóm {cluster} — {kind}', offer, rationale


NOISE_TEXT = 'Điểm nhiễu (DBSCAN): xem xét riêng trước khi gửi khuyến mãi.'
OUTLIER_TEXT = 'Khách hàng chi tiêu đột biến (Ngoại lai IQR): chăm sóc riêng / ưu đãi VIP theo hợp đồng. (Gần với {segment}).'


def profiles(frame, labels, outlier=None):
    outlier = np.zeros(len(frame), dtype=bool) if outlier is None else np.asarray(outlier, dtype=bool)
    data = frame.copy()
    data['cluster'] = labels
    core = data[~outlier]
    means = core.groupby('cluster')[FEATURES].mean()
    medians = core.groupby('cluster')[FEATURES].median()
    overall = core[FEATURES].mean().replace(0, np.nan)
    ratios = means.div(overall)
    total_ratio = means.sum(axis=1) / overall.sum()
    retail_share = core.groupby('cluster').Channel.apply(lambda s: float((s == 2).mean()))
    rows = []
    for cluster in means.index:
        count = int((labels == cluster).sum())
        out_count = int(((labels == cluster) & outlier).sum())
        if cluster == -1:
            segment, offer, rationale = 'Ngoại lệ / nhiễu (DBSCAN)', NOISE_TEXT, 'Không nằm trong cụm mật độ nào.'
        else:
            segment, offer, rationale = describe_segment(int(cluster), ratios.loc[cluster], float(total_ratio[cluster]), float(retail_share[cluster]))
        rows.append({
            'cluster': int(cluster), 'customers': count, 'outliers_assigned': out_count,
            'segment': segment, 'suggestion': offer, 'rationale': rationale,
        })
    return pd.DataFrame(rows), means, medians, ratios


def build_assignments(frame, labels, outlier, coords, segments):
    out = frame.copy()
    out.insert(0, 'row_id', np.arange(1, len(frame) + 1))
    out['cluster'] = labels
    out['is_outlier'] = np.asarray(outlier, dtype=bool)
    out[['PC1', 'PC2']] = coords
    out = out.merge(segments[['cluster', 'segment', 'suggestion']], on='cluster', how='left')
    mask = out.is_outlier.to_numpy()
    out.loc[mask, 'suggestion'] = [OUTLIER_TEXT.format(segment=s) for s in out.loc[mask, 'segment']]
    return out


# --------------------------------------------------------------------------- 5. Huấn luyện và lưu phiên
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
            'channel_distribution': pd.crosstab(rep_assignments.cluster, rep_assignments.Channel, normalize='index').reset_index(),
            'region_distribution': pd.crosstab(rep_assignments.cluster, rep_assignments.Region, normalize='index').reset_index(),
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
        selection_policy=f'Silhouette cao nhất (Davies-Bouldin phá hòa); >={MIN_SEGMENTS} cụm, mỗi cụm >=5, nhiễu <=30%',
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
        '# Báo cáo phiên huấn luyện', '',
        f"Phiên: {run_id}; UTC: {metadata['created_utc']}",
        f"Dữ liệu: {len(frame)} khách hàng (loại {int(res.outlier.sum())} ngoại lai bằng {outlier_method} khi huấn luyện).",
        f"Chuẩn hóa: {scaler_type}" + (' + log1p' if use_log else '') + '.', '',
        '## So sánh ba thuật toán', '',
        '```text', best[['algorithm', 'params', 'clusters', 'noise_rate', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'eligible']].to_string(index=False), '```', '',
        f"Mô hình chính: {selected.algorithm} {selected.params}; Elbow gợi ý k={res.elbow}.", '',
        '## Hồ sơ phân khúc & gợi ý khuyến mãi', '',
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
    """Gán phân khúc và gợi ý khuyến mãi ngay cho khách hàng mới vừa nhập."""
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


def retrain_due(rows_now, root=None, min_new=10, max_days=7, now=None):
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
