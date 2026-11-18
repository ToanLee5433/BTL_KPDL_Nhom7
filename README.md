# Phân khúc khách hàng bán lẻ (Mô hình RFM) để gợi ý khuyến mãi

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ToanLee5433/BTL_KPDL_Nhom7/blob/main/Online_Retail_RFM_Clustering.ipynb)

Bài tập lớn môn Khai phá dữ liệu (nhóm 5 người) — **bài toán Phân cụm** trên bộ dữ liệu **Online Retail (UCI 352: 541,909 giao dịch $\rightarrow$ 4,338 khách hàng RFM)** sử dụng **K-Means, Hierarchical Clustering (Ward) và DBSCAN**; thực nghiệm trên Google Colab và ứng dụng web Python + Streamlit.

## Kết quả chính (Online Retail — 4,338 khách hàng RFM)

| Thành phần | Thiết kế & Kết quả |
|---|---|
| **Làm sạch & RFM** | Từ **541,909** giao dịch thô, loại 135,080 dòng thiếu `CustomerID`, 8,905 hóa đơn hủy (`C*`) và 40 dòng `Quantity/UnitPrice ≤ 0` $\rightarrow$ còn **397,884 giao dịch hợp lệ**, tổng hợp thành **4,338 khách hàng RFM** (`Recency`, `Frequency`, `Monetary`). |
| **Tiền xử lý** | Tách **628 ngoại lai bằng quy tắc IQR** (giữ **3,710** khách hàng chuẩn để huấn luyện; 628 khách ngoại lai vẫn được giữ ở đầu ra với nhãn VIP), chuẩn hóa bằng **`StandardScaler`** (đối chiếu **`MinMaxScaler`**). |
| **K-Means** | Khảo sát $k = 2 \dots 8$; tốt nhất tại **$k = 3$** với **Silhouette = `0.460`**, Davies–Bouldin = `0.766` (Elbow gợi ý $k = 4$, đạt `0.413`). **Đạt tiêu chí**. |
| **Hierarchical (Ward)** | Khảo sát $k = 2 \dots 8$ + biểu đồ **Dendrogram**; tốt nhất tại **$k = 3$** với **Silhouette = `0.444`**, Davies–Bouldin = `0.787`. **Đạt tiêu chí**. |
| **DBSCAN** | Chọn `eps` từ **k-distance**; tốt nhất tại `eps = 0.5, min_samples = 50` $\rightarrow$ **2 cụm**, tỷ lệ nhiễu `12.9%` ($\le 30\%$), **Silhouette = `0.451`**. **Đạt tiêu chí**. |
| **Phân khúc RFM** | 3 phân khúc rõ rệt (*Nhóm 0: Khách hàng mới / Tiềm năng* — 1,968 KH; *Nhóm 1: Khách hàng ngủ quên / Nguy cơ rời bỏ* — 1,074 KH; *Nhóm 2: Khách hàng trung thành / Chi tiêu cao* — 1,296 KH) kèm chiến lược khuyến mãi riêng. |

## Chạy trên Google Colab (cho cả nhóm)

1. Bấm thẳng nút **Open in Colab** ở đầu trang này (hoặc upload file `Online_Retail_RFM_Clustering.ipynb` lên https://colab.research.google.com/).
2. Chọn *Runtime $\rightarrow$ Run all* (dùng CPU mặc định).
3. Notebook tự tải file Excel gốc (541,909 dòng) từ UCI để chạy trực tiếp bước làm sạch + RFM (nếu UCI chậm sẽ dùng sẵn `data/online_retail_rfm.csv` trong repo), vẽ đầy đủ Boxplot IQR, Elbow, Silhouette, Dendrogram, k-distance, PCA 2D và khởi chạy giao diện Streamlit ngay trên Colab.

## Chạy ứng dụng Streamlit trên máy cá nhân (Windows / macOS / Linux)

Sau khi `git clone` hoặc giải nén thư mục dự án, mở Terminal / PowerShell **tại thư mục gốc của dự án** rồi chạy:

**Trên Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe retrain.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**Trên macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python retrain.py
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`. Toàn bộ đường dẫn trong code đều dùng đường dẫn tương đối (`Path(__file__).resolve().parent`), nên đặt thư mục ở đâu trên máy nào cũng chạy bình thường.

## Huấn luyện định kỳ ngoài giao diện (Windows Task Scheduler)

Có thể đặt lịch chạy tự động bằng lệnh:
```powershell
python retrain.py --if-due
```
Chỉ huấn luyện khi có ít nhất 20 khách hàng mới (`--min-new`) hoặc quá 7 ngày (`--max-days`) kể từ lần huấn luyện trước.

## Cấu trúc dự án

- `pipeline.py`: Làm sạch giao dịch thô (`build_rfm_from_raw`), kiểm tra dữ liệu (`validate`), IQR, `StandardScaler`/`MinMaxScaler`, 3 thuật toán, đánh giá (Elbow, Silhouette), phân khúc RFM & dự đoán khách mới.
- `storage.py`: Lưu trữ SQLite (`data/retail_customers.sqlite3`), chống nhập lặp batch theo SHA-256.
- `app.py`, `experience.py`: Giao diện web Streamlit 4 tab.
- `retrain.py`: Script huấn luyện lại theo dòng lệnh / Task Scheduler.
- `build_notebook.py`: Tạo `Online_Retail_RFM_Clustering.ipynb` cho Google Colab.
- `docs/BAO_CAO.md`: Báo cáo chi tiết bài tập lớn (6 chương) & bảng phân công 5 thành viên.
- `tests/test_system.py`: Bộ kiểm thử tự động (`python -m pytest tests -q`).
- `data/online_retail_rfm.csv`: Bảng 4,338 khách hàng RFM đã tổng hợp từ 541,909 giao dịch UCI.
