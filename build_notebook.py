"""Tạo notebook Google Colab cho đề tài Phân khúc khách hàng bán lẻ (RFM - Online Retail UCI 352).

Chạy:  python build_notebook.py   (sinh ra Online_Retail_RFM_Clustering.ipynb)
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md('''# Bài tập lớn Khai phá dữ liệu — Bài toán Phân cụm
## Đề tài: Phân khúc khách hàng bán lẻ (Mô hình RFM) để gợi ý khuyến mãi

**Nhóm 7 — 5 thành viên** *(điền họ tên và mã sinh viên trước khi nộp)*

- **Bộ dữ liệu:** *Online Retail Data Set* (UCI Machine Learning Repository ID 352: **541,909 giao dịch** $\\rightarrow$ làm sạch & tổng hợp thành **4,338 khách hàng RFM**).
- **Tiền xử lý:** Làm sạch giao dịch thô, tổng hợp chỉ số **RFM (Recency - Frequency - Monetary)**, phát hiện & tách ngoại lai bằng **IQR**, chuẩn hóa bằng **`StandardScaler`** và đối chiếu **`MinMaxScaler`**.
- **3 thuật toán so sánh:** **K-Means**, **Hierarchical Clustering (Agglomerative - Ward)** và **DBSCAN**.
- **Đánh giá:** **Elbow Method** (Khuỷu tay) và **Silhouette Score** (cùng Davies–Bouldin, Calinski–Harabasz).
- **Ứng dụng:** Giao diện Streamlit thu thập khách hàng mới, tiền xử lý, huấn luyện lại định kỳ và gợi ý khuyến mãi.

> **Cách chạy trên Google Colab:** Chọn *Runtime → Run all* (CPU mặc định). Cuối notebook có ô khởi chạy giao diện Streamlit trực tiếp trên Colab.''')

code('''import io, sys, json, shutil, zipfile, subprocess, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from IPython.display import display

print('Python', sys.version.split()[0], '| scikit-learn', sklearn.__version__, '| pandas', pd.__version__)
WORK = Path.cwd()
PROJECT_DIR = WORK / 'retail_app'
FIGURES = WORK / 'figures'
FIGURES.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': .25})''')

md('''## 1. Tải dữ liệu, Làm sạch 541,909 giao dịch & Tổng hợp RFM
Nguồn: [UCI Online Retail (ID 352)](https://archive.ics.uci.edu/dataset/352/online%2Bretail) — Daqing Chen (2015), DOI `10.24432/C5BW33`.

Quy trình làm sạch dữ liệu giao dịch thô:
1. Loại bỏ các dòng thiếu `CustomerID` (không định danh được khách hàng).
2. Loại bỏ hóa đơn bị hủy (`InvoiceNo` bắt đầu bằng chữ `'C'`).
3. Loại bỏ các dòng có `Quantity <= 0` hoặc `UnitPrice <= 0`.
4. Tổng hợp theo từng `CustomerID` thành 3 chỉ số **RFM**:
   - **Recency (R):** Số ngày kể từ lần mua gần nhất đến ngày chốt dữ liệu.
   - **Frequency (F):** Tổng số hóa đơn mua hàng hợp lệ.
   - **Monetary (M):** Tổng số tiền khách hàng đã chi tiêu (£).''')

code('''# Đồng bộ mã nguồn từ GitHub (hoặc dùng thư mục hiện tại nếu chạy trên máy cá nhân)
REPO_URL = 'https://github.com/ToanLee5433/BTL_KPDL_Nhom7.git'
if (WORK / 'pipeline.py').exists() and (WORK / 'data' / 'online_retail_rfm.csv').exists():
    PROJECT_DIR = WORK
elif not (PROJECT_DIR / 'pipeline.py').exists():
    shutil.rmtree(PROJECT_DIR, ignore_errors=True)
    subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, str(PROJECT_DIR)], check=True)

sys.path.insert(0, str(PROJECT_DIR))
import importlib, pipeline
importlib.reload(pipeline)
from pipeline import *

# Thử tải file Excel gốc (541,909 dòng) từ UCI để chạy trực tiếp bước làm sạch; nếu mạng chậm dùng sẵn file RFM trong repo
RAW_URL = 'https://archive.ics.uci.edu/static/public/352/online+retail.zip'
rfm_csv = PROJECT_DIR / 'data' / 'online_retail_rfm.csv'
raw_xlsx = PROJECT_DIR / 'data' / 'Online Retail.xlsx'
if not raw_xlsx.exists() and 'google.colab' in sys.modules:
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'openpyxl'], check=True)
        req = urllib.request.Request(RAW_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=45) as resp:
            with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
                zf.extractall(PROJECT_DIR / 'data')
    except Exception as e:
        print('Không tải được file Excel gốc từ UCI, dùng file RFM đã tổng hợp trong repo:', type(e).__name__)

if raw_xlsx.exists():
    raw_tx = pd.read_excel(raw_xlsx)
    data, clean_stats = build_rfm_from_raw(raw_tx)
    print('Thống kê làm sạch từ giao dịch gốc:', clean_stats)
else:
    data = validate(pd.read_csv(rfm_csv))
    print('Đã nạp bảng RFM chuẩn:', len(data), 'khách hàng (tổng hợp từ 541,909 giao dịch UCI).')

data = validate(data)
display(data.head())
display(data[FEATURES].describe().T.round(2))''')

md('''## 2. Tiền xử lý: Loại bỏ ngoại lai (IQR) & Chuẩn hóa dữ liệu
1. **Ngoại lai (IQR):** Tách các khách hàng có giá trị R, F hoặc M vượt ngoài $[Q_1 - 1.5\\times IQR,\\; Q_3 + 1.5\\times IQR]$ khỏi bước huấn luyện (để không kéo lệch tâm cụm), nhưng vẫn giữ lại ở đầu ra với nhãn **Ngoại lai VIP (IQR)** để chăm sóc riêng.
2. **Chuẩn hóa:** Đưa `Recency`, `Frequency`, `Monetary` về cùng thang đo bằng **`StandardScaler`** và đối chiếu với **`MinMaxScaler`**.''')

code('''outlier_mask, bounds = detect_outliers_iqr(data, method='iqr')
clean_data = data.loc[~outlier_mask].reset_index(drop=True)
print(f'Tổng khách hàng: {len(data):,} | Ngoại lai (IQR): {outlier_mask.sum():,} | Tập chuẩn huấn luyện: {len(clean_data):,}')

fig, axes = plt.subplots(2, 3, figsize=(15, 6.5))
for i, col in enumerate(FEATURES):
    axes[0, i].boxplot(data[col]); axes[0, i].set_title(f'{col} — Trước IQR ({len(data):,})'); axes[0, i].set_xticks([])
    axes[1, i].boxplot(clean_data[col]); axes[1, i].set_title(f'{col} — Sau IQR ({len(clean_data):,})'); axes[1, i].set_xticks([])
plt.tight_layout(); plt.savefig(FIGURES / 'boxplot_rfm_outliers.png', dpi=140); plt.show()

scaler_comp = compare_scalers(data, outlier_method='iqr')
display(scaler_comp[['scaler', 'algorithm', 'params', 'clusters', 'noise_rate', 'silhouette', 'davies_bouldin', 'eligible']].round(3))''')

md('''## 3. Huấn luyện & Đánh giá 3 thuật toán (K-Means · Hierarchical · DBSCAN)
- **K-Means:** Khảo sát $k = 2 \\dots 8$, dùng **Elbow Method** và **Silhouette Score** để chọn $k$.
- **Hierarchical (Ward):** Vẽ **Dendrogram** và khảo sát $k = 2 \\dots 8$.
- **DBSCAN:** Vẽ biểu đồ **k-distance** để chọn `eps` và khảo sát lưới `(eps, min_samples)`.''')

code('''res = search(data, scaler_type='standard', outlier_method='iqr', use_log=False)
table = res.table.copy()
print('Điểm khuỷu (Elbow Method) gợi ý k =', res.elbow)

km = table[table.algorithm == 'KMeans'].assign(k=lambda d: d.params.map(lambda x: json.loads(x)['k']))
hc = table[table.algorithm == 'Hierarchical'].assign(k=lambda d: d.params.map(lambda x: json.loads(x)['k']))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
axes[0].plot(km.k, km.inertia, 'o-', color='#2563eb')
axes[0].axvline(res.elbow, color='#dc2626', ls='--', label=f'Elbow gợi ý k = {res.elbow}')
axes[0].set(xlabel='Số cụm (k)', ylabel='Inertia (WCSS)', title='Phương pháp Khuỷu tay (Elbow Method - KMeans)')
axes[0].legend()

axes[1].plot(km.k, km.silhouette, 'o-', label='K-Means')
axes[1].plot(hc.k, hc.silhouette, 's-', label='Hierarchical (Ward)')
axes[1].set(xlabel='Số cụm (k)', ylabel='Silhouette Score', title='So sánh Silhouette Score theo k')
axes[1].legend()
plt.tight_layout(); plt.savefig(FIGURES / 'elbow_and_silhouette.png', dpi=140); plt.show()''')

code('''# Biểu đồ Silhouette từng điểm cho K-Means k = 2..5
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
Xf = res.X_fit
fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
for ax, k in zip(axes, [2, 3, 4, 5]):
    lab = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(Xf)
    s = silhouette_samples(Xf, lab); y = 20
    for c in range(k):
        v = np.sort(s[lab == c])
        ax.fill_betweenx(np.arange(y, y + len(v)), 0, v, alpha=.75)
        ax.text(-0.08, y + len(v) / 2, str(c), fontsize=9)
        y += len(v) + 20
    ax.axvline(s.mean(), color='red', ls='--'); ax.set_yticks([]); ax.grid(False)
    ax.set(title=f'k = {k} (Sil = {s.mean():.3f})', xlabel='Silhouette')
plt.tight_layout(); plt.savefig(FIGURES / 'silhouette_per_k.png', dpi=140); plt.show()

# Dendrogram (Hierarchical) & k-distance (DBSCAN)
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.neighbors import NearestNeighbors
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
dendrogram(linkage(Xf, method='ward'), truncate_mode='lastp', p=25, ax=axes[0])
axes[0].set(title='Dendrogram (Hierarchical Ward, rút gọn 25 nhánh)', xlabel='Nhóm gộp', ylabel='Khoảng cách')
dists = np.sort(NearestNeighbors(n_neighbors=10).fit(Xf).kneighbors(Xf)[0][:, -1])
axes[1].plot(dists, color='#0d9488')
axes[1].set(title='Biểu đồ k-distance (DBSCAN, k=10)', xlabel='Khách hàng sắp xếp tăng dần', ylabel='Khoảng cách láng giềng thứ 10')
plt.tight_layout(); plt.savefig(FIGURES / 'dendrogram_and_kdistance.png', dpi=140); plt.show()''')

md('''## 4. So sánh kết quả 3 thuật toán & Trực quan hóa PCA 2D''')
code('''display(res.best[['algorithm', 'params', 'clusters', 'noise_rate', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'seconds', 'eligible']].round(3))

pca = PCA(n_components=2).fit(res.X_fit)
coords = pca.transform(res.X)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, name in zip(axes, ['KMeans', 'Hierarchical', 'DBSCAN']):
    row = res.best[res.best.algorithm == name].iloc[0]
    lab = full_labels(res, res.candidates[int(row.candidate)][1])
    for c in sorted(set(lab)):
        m = (lab == c) & ~res.outlier
        ax.scatter(coords[m, 0], coords[m, 1], s=10, alpha=.55, label='Nhiễu' if c == -1 else f'Cụm {c}')
    ax.scatter(coords[res.outlier, 0], coords[res.outlier, 1], s=16, c='k', marker='x', alpha=.35, label='Ngoại lai VIP')
    ax.set_title(f"{name} {row.params}\\nSilhouette = {row.silhouette:.3f}")
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIGURES / 'three_algorithms_pca.png', dpi=140); plt.show()
print(f'PCA 2D giải thích {pca.explained_variance_ratio_.sum():.1%} phương sai.')''')

md('''## 5. Hồ sơ phân khúc RFM & Gợi ý khuyến mãi''')
code('''folder = train(data, output_root=PROJECT_DIR / 'artifacts', scaler_type='standard', outlier_method='iqr', use_log=False)
metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
segments = pd.read_csv(folder / 'segments.csv')
relative = pd.read_csv(folder / 'relative_spending.csv').set_index('cluster')
medians = pd.read_csv(folder / 'medians.csv')

print('Phiên lưu:', folder.name, '| Thuật toán chọn:', metadata['algorithm'], metadata['params'])
pd.set_option('display.max_colwidth', 220)
display(segments[['cluster', 'customers', 'outliers_assigned', 'segment', 'suggestion', 'rationale']])
display(medians.round(1))

fig, ax = plt.subplots(figsize=(8, 3.8))
im = ax.imshow(relative[FEATURES], aspect='auto', cmap='YlGnBu'); ax.grid(False)
ax.set_xticks(range(3), FEATURES)
ax.set_yticks(range(len(relative)), [f'Cụm {c}' for c in relative.index])
for i in range(relative.shape[0]):
    for j in range(3):
        ax.text(j, i, f'{relative[FEATURES].iloc[i, j]:.2f}', ha='center', va='center', fontsize=10)
ax.set_title('Tỷ lệ R, F, M trung bình của từng cụm so với mức chung (1.0 = trung bình)')
fig.colorbar(im, ax=ax)
plt.tight_layout(); plt.savefig(FIGURES / 'rfm_heatmap.png', dpi=140); plt.show()

# Thử nghiệm dự đoán phân khúc cho khách hàng mới
demo_new = pd.DataFrame([
    [99001, 15, 2, 450.0, 220, 'United Kingdom'],
    [99002, 10, 8, 2800.0, 1500, 'United Kingdom'],
    [99003, 250, 1, 250.0, 110, 'United Kingdom'],
], columns=COLUMNS)
display(predict_segments(demo_new, folder)[['CustomerID', 'cluster', 'segment', 'is_outlier', 'suggestion']])''')

md('''## 6. Chạy giao diện ứng dụng (Streamlit) & Đóng gói kết quả''')
code('''RUN_STREAMLIT = 'google.colab' in sys.modules
if RUN_STREAMLIT:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'streamlit'], check=True)
    log_file = open(WORK / 'streamlit.log', 'w')
    server = subprocess.Popen(
        [sys.executable, '-m', 'streamlit', 'run', 'app.py', '--server.port', '8501',
         '--server.headless', 'true', '--server.enableCORS', 'false', '--server.enableXsrfProtection', 'false'],
        cwd=PROJECT_DIR, stdout=log_file, stderr=subprocess.STDOUT,
    )
    import time; time.sleep(10)
    from google.colab import output
    output.serve_kernel_port_as_window(8501)
    print('Streamlit đang chạy tại cổng 8501.')
else:
    print('Chạy trên máy cá nhân: `python -m streamlit run app.py` trong thư mục', PROJECT_DIR)

shutil.copytree(FIGURES, folder / 'figures', dirs_exist_ok=True)
zip_path = shutil.make_archive(str(WORK / 'online_retail_rfm_results'), 'zip', folder)
print('Đã đóng gói kết quả & biểu đồ tại:', zip_path)''')

nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        'kernelspec': {'display_name': 'Python 3', 'name': 'python3', 'language': 'python'},
        'language_info': {'name': 'python'},
        'colab': {'name': 'Online_Retail_RFM_Clustering.ipynb', 'provenance': []},
    },
)
nbf.validate(nb)
nbf.write(nb, ROOT / 'Online_Retail_RFM_Clustering.ipynb')
print('Notebook generated: Online_Retail_RFM_Clustering.ipynb')
