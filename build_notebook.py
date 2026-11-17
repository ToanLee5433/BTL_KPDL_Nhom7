"""Tạo notebook Google Colab tự chứa, bám sát sườn đề cương bài tập lớn.

Chạy:  python build_notebook.py   (sinh ra Wholesale_Customers_Clustering.ipynb)
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


REPO_URL = 'https://github.com/ToanLee5433/BTL_KPDL_Nhom7.git'

md('''# Bài tập lớn Khai phá dữ liệu — Bài toán Phân cụm
## Đề tài: Phân khúc khách hàng siêu thị để gợi ý khuyến mãi

**Nhóm 5 người** *(điền danh sách thành viên khi nộp bài)*

- **Bộ dữ liệu:** Wholesale Customers Data Set (UCI Machine Learning Repository, 440 dòng, 8 cột).
- **Tiền xử lý:** Kiểm tra dữ liệu, loại bỏ ngoại lai bằng quy tắc **IQR**, chuẩn hóa bằng **StandardScaler** và đối chiếu **MinMaxScaler**.
- **3 thuật toán so sánh:** **K-Means**, **Hierarchical Clustering (Agglomerative - Ward)** và **DBSCAN**.
- **Đánh giá:** **Elbow Method** (Khuỷu tay) và **Silhouette Score** (cùng Davies–Bouldin, Calinski–Harabasz).
- **Ứng dụng:** Giao diện Streamlit thu thập dữ liệu mới, tiền xử lý, huấn luyện lại định kỳ và hiển thị kết quả phân khúc + gợi ý khuyến mãi.

> **Cách chạy trên Google Colab:** Chọn *Runtime → Run all* (dùng CPU mặc định). Notebook tự tải dữ liệu từ UCI (có bản dự phòng nhúng sẵn nếu mạng lỗi) và có ô chạy trực tiếp giao diện Streamlit ở cuối.''')

code('''import io, sys, json, hashlib, shutil, zipfile, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from IPython.display import display

print('Python', sys.version.split()[0], '| scikit-learn', sklearn.__version__, '| pandas', pd.__version__)
WORK = Path.cwd()
PROJECT_DIR = WORK / 'wholesale_app'
FIGURES = WORK / 'figures'
(PROJECT_DIR / 'data').mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': .25})''')

md('''## 1. Giới thiệu dữ liệu & Khám phá (EDA)
Nguồn: [UCI Wholesale Customers](https://archive.ics.uci.edu/dataset/292/wholesale+customers) — Cardoso, M. (2013), DOI `10.24432/C5030X`.

Bộ dữ liệu gồm **440 khách hàng**, mỗi dòng là tổng chi tiêu trong năm theo 6 nhóm mặt hàng:

| Thuộc tính | Ý nghĩa |
|---|---|
| `Fresh` | Chi tiêu nhóm hàng tươi sống |
| `Milk` | Chi tiêu sữa và chế phẩm |
| `Grocery` | Chi tiêu hàng tạp hóa |
| `Frozen` | Chi tiêu hàng đông lạnh |
| `Detergents_Paper` | Chi tiêu chất tẩy rửa & sản phẩm giấy |
| `Delicassen` | Chi tiêu thực phẩm chế biến sẵn |
| `Channel` | Kênh khách hàng: `1` = Nhà hàng/Khách sạn/Café; `2` = Bán lẻ / Siêu thị |
| `Region` | Khu vực: `1` = Lisbon; `2` = Oporto; `3` = Khu vực khác |

6 cột chi tiêu được dùng để tính khoảng cách phân cụm; `Channel` và `Region` dùng để mô tả và đối chiếu đặc điểm các cụm.''')

code('''# Tải mã nguồn & dữ liệu từ GitHub (hoặc dùng thư mục hiện tại nếu chạy trên máy cá nhân)
import subprocess
REPO_URL = 'https://github.com/ToanLee5433/BTL_KPDL_Nhom7.git'
if (WORK / 'pipeline.py').exists() and (WORK / 'data' / 'wholesale_customers.csv').exists():
    PROJECT_DIR = WORK
elif not (PROJECT_DIR / 'pipeline.py').exists():
    shutil.rmtree(PROJECT_DIR, ignore_errors=True)
    subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, str(PROJECT_DIR)], check=True)

sys.path.insert(0, str(PROJECT_DIR))
import importlib, pipeline
importlib.reload(pipeline)
from pipeline import *

# Thử tải trực tiếp từ UCI; nếu mạng UCI chậm/lỗi thì dùng file CSV chuẩn đi kèm trong repo
URL = 'https://archive.ics.uci.edu/static/public/292/wholesale+customers.zip'
try:
    req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=15) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
            csv_name = next(name for name in archive.namelist() if name.endswith('.csv'))
            (PROJECT_DIR / 'data' / 'wholesale_customers.csv').write_bytes(archive.read(csv_name))
    print('Nguồn dữ liệu: Tải trực tiếp từ UCI')
except Exception:
    print('Nguồn dữ liệu: Dùng data/wholesale_customers.csv từ GitHub repo')

raw = pd.read_csv(PROJECT_DIR / 'data' / 'wholesale_customers.csv')
data = validate(raw)
print('Kích thước dữ liệu:', data.shape, '| Giá trị thiếu:', raw.isna().sum().sum(), '| Dòng trùng:', raw.duplicated().sum())
display(data.head())
display(data[FEATURES].describe().T)''')

md('''## 2. Tiền xử lý: Phát hiện ngoại lai (IQR) & Chuẩn hóa dữ liệu
1. **Phát hiện & loại bỏ ngoại lai (Outliers):** Dùng quy tắc **IQR** ($[Q_1 - 1.5\\times IQR,\\; Q_3 + 1.5\\times IQR]$). Vì các thuật toán dựa trên khoảng cách rất nhạy với giá trị cực đại, ta loại ngoại lai khỏi tập huấn luyện (nhưng vẫn gắn cờ để chăm sóc riêng ở bước khuyến mãi, không làm mất dữ liệu khách hàng).
2. **Chuẩn hóa dữ liệu:** Đưa 6 thuộc tính về cùng thang đo bằng **`StandardScaler`** (Z-score) hoặc **`MinMaxScaler`** ($[0, 1]$). Bắt buộc phải chuẩn hóa vì biên độ chi tiêu giữa các nhóm hàng chênh lệch rất lớn.''')

code('''# Trực quan hóa Boxplot trước và sau khi loại bỏ ngoại lai bằng IQR
outlier_mask, bounds = detect_outliers_iqr(data, method='iqr')
clean_data = data.loc[~outlier_mask].reset_index(drop=True)
print(f'Tổng số khách hàng: {len(data)} | Ngoại lai (IQR): {outlier_mask.sum()} | Tập sạch để huấn luyện: {len(clean_data)}')

fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
axes[0].boxplot([data[c] for c in FEATURES], tick_labels=FEATURES)
axes[0].set_title(f'Trước khi loại ngoại lai ({len(data)} dòng)'); axes[0].tick_params(axis='x', rotation=25)
axes[1].boxplot([clean_data[c] for c in FEATURES], tick_labels=FEATURES)
axes[1].set_title(f'Sau khi loại ngoại lai bằng IQR ({len(clean_data)} dòng)'); axes[1].tick_params(axis='x', rotation=25)
plt.tight_layout(); plt.savefig(FIGURES / 'boxplot_outliers.png', dpi=140); plt.show()

# Đối chiếu 2 bộ chuẩn hóa theo đề cương: StandardScaler vs MinMaxScaler
scaler_comp = compare_scalers(data, outlier_method='iqr')
display(scaler_comp[['scaler', 'algorithm', 'params', 'clusters', 'noise_rate', 'silhouette', 'davies_bouldin', 'eligible']].round(3))''')

md('''## 3. Huấn luyện & Đánh giá 3 thuật toán (K-Means · Hierarchical · DBSCAN)
- **K-Means:** Khảo sát $k = 2 \\dots 8$, dùng **Elbow Method** (Inertia/WCSS) và **Silhouette Score** để chọn $k$.
- **Hierarchical Clustering (Agglomerative - Ward):** Vẽ **Dendrogram** và khảo sát $k = 2 \\dots 8$.
- **DBSCAN:** Dựa vào biểu đồ **k-distance** để chọn `eps` và khảo sát lưới `(eps, min_samples)`.
- Lưu ý: Với bài toán gợi ý khuyến mãi, chọn $k \\ge 3$ để tránh chỉ tách lại 2 kênh `Channel` có sẵn.''')

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

code('''# Biểu đồ Silhouette chi tiết từng mẫu cho K-Means k = 2..5
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
Xf = res.X_fit
fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
for ax, k in zip(axes, [2, 3, 4, 5]):
    lab = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(Xf)
    s = silhouette_samples(Xf, lab); y = 10
    for c in range(k):
        v = np.sort(s[lab == c])
        ax.fill_betweenx(np.arange(y, y + len(v)), 0, v, alpha=.75)
        ax.text(-0.08, y + len(v) / 2, str(c), fontsize=9)
        y += len(v) + 10
    ax.axvline(s.mean(), color='red', ls='--'); ax.set_yticks([]); ax.grid(False)
    ax.set(title=f'k = {k} (Sil = {s.mean():.3f})', xlabel='Silhouette')
plt.tight_layout(); plt.savefig(FIGURES / 'silhouette_per_k.png', dpi=140); plt.show()

# Dendrogram (Hierarchical) & k-distance (DBSCAN)
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.neighbors import NearestNeighbors
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
dendrogram(linkage(Xf, method='ward'), truncate_mode='lastp', p=25, ax=axes[0])
axes[0].set(title='Dendrogram (Hierarchical Ward, rút gọn 25 nhánh)', xlabel='Nhóm gộp', ylabel='Khoảng cách')
dists = np.sort(NearestNeighbors(n_neighbors=4).fit(Xf).kneighbors(Xf)[0][:, -1])
axes[1].plot(dists, color='#0d9488')
axes[1].set(title='Biểu đồ k-distance (DBSCAN, k=4)', xlabel='Khách hàng sắp xếp tăng dần', ylabel='Khoảng cách láng giềng thứ 4')
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
        ax.scatter(coords[m, 0], coords[m, 1], s=18, alpha=.7, label='Nhiễu' if c == -1 else f'Cụm {c}')
    ax.scatter(coords[res.outlier, 0], coords[res.outlier, 1], s=32, c='k', marker='x', alpha=.5, label='Ngoại lai (IQR)')
    ax.set_title(f"{name} {row.params}\\nSilhouette = {row.silhouette:.3f}")
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIGURES / 'three_algorithms_pca.png', dpi=140); plt.show()
print(f'PCA 2D giải thích {pca.explained_variance_ratio_.sum():.1%} phương sai.')''')

md('''## 5. Hồ sơ phân khúc khách hàng & Gợi ý khuyến mãi
Huấn luyện phiên chính thức, lưu vào `wholesale_app/artifacts/` (dùng chung cho giao diện Streamlit) và trích xuất bảng hồ sơ chi tiêu + gợi ý khuyến mãi cho từng phân khúc.''')
code('''folder = train(data, output_root=PROJECT_DIR / 'artifacts', scaler_type='standard', outlier_method='iqr', use_log=False)
metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
segments = pd.read_csv(folder / 'segments.csv')
relative = pd.read_csv(folder / 'relative_spending.csv').set_index('cluster')
medians = pd.read_csv(folder / 'medians.csv')

print('Phiên lưu:', folder.name, '| Thuật toán chọn:', metadata['algorithm'], metadata['params'])
pd.set_option('display.max_colwidth', 220)
display(segments[['cluster', 'customers', 'outliers_assigned', 'segment', 'suggestion', 'rationale']])

fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(relative[FEATURES], aspect='auto', cmap='YlGnBu'); ax.grid(False)
ax.set_xticks(range(6), [CATEGORY_VI[f] for f in FEATURES], rotation=25, ha='right')
ax.set_yticks(range(len(relative)), [f'Cụm {c}' for c in relative.index])
for i in range(relative.shape[0]):
    for j in range(6):
        ax.text(j, i, f'{relative[FEATURES].iloc[i, j]:.2f}', ha='center', va='center', fontsize=9)
ax.set_title('Mức chi tiêu tương đối của từng cụm (1.0 = bằng mức trung bình chung)')
fig.colorbar(im, ax=ax)
plt.tight_layout(); plt.savefig(FIGURES / 'segment_heatmap.png', dpi=140); plt.show()

# Thử nghiệm gán phân khúc cho khách hàng mới ngay lập tức
demo_new = pd.DataFrame([
    [2, 1, 4000, 8500, 12000, 1200, 4500, 800],
    [1, 3, 18000, 2000, 2500, 5500, 500, 1200],
], columns=COLUMNS)
display(predict_segments(demo_new, folder)[['Channel', 'cluster', 'segment', 'is_outlier', 'suggestion']])''')

md('''## 6. Giao diện ứng dụng (Streamlit) & Đóng gói kết quả
Giao diện đáp ứng đủ 4 yêu cầu: thu thập dữ liệu mới (form + CSV), tiền xử lý, huấn luyện lại định kỳ và hiển thị kết quả phân khúc + khuyến mãi cho người dùng.''')
code('''RUN_STREAMLIT = 'google.colab' in sys.modules
if RUN_STREAMLIT:
    import subprocess, time
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'streamlit'], check=True)
    log_file = open(WORK / 'streamlit.log', 'w')
    server = subprocess.Popen(
        [sys.executable, '-m', 'streamlit', 'run', 'app.py', '--server.port', '8501',
         '--server.headless', 'true', '--server.enableCORS', 'false', '--server.enableXsrfProtection', 'false'],
        cwd=PROJECT_DIR, stdout=log_file, stderr=subprocess.STDOUT,
    )
    time.sleep(10)
    from google.colab import output
    output.serve_kernel_port_as_window(8501)
    print('Streamlit đang chạy tại cổng 8501. Nếu trình duyệt chặn pop-up, chạy: output.serve_kernel_port_as_iframe(8501)')
else:
    print('Chạy trên máy cá nhân: `python -m streamlit run app.py` trong thư mục', PROJECT_DIR)

shutil.copytree(FIGURES, folder / 'figures', dirs_exist_ok=True)
zip_path = shutil.make_archive(str(WORK / 'wholesale_results'), 'zip', folder)
print('Đã đóng gói toàn bộ kết quả & biểu đồ tại:', zip_path)''')

nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        'kernelspec': {'display_name': 'Python 3', 'name': 'python3', 'language': 'python'},
        'language_info': {'name': 'python'},
        'colab': {'name': 'Wholesale_Customers_Clustering.ipynb', 'provenance': []},
    },
)
nbf.validate(nb)
nbf.write(nb, ROOT / 'Wholesale_Customers_Clustering.ipynb')
print('Notebook generated')
