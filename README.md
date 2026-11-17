# Phân khúc khách hàng siêu thị để gợi ý khuyến mãi

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ToanLee5433/BTL_KPDL_Nhom7/blob/main/Wholesale_Customers_Clustering.ipynb)

Bài tập lớn môn Khai phá dữ liệu (nhóm 5 người) — **bài toán Phân cụm** trên bộ dữ liệu **Wholesale Customers (UCI, 440 dòng)** sử dụng **K-Means, Hierarchical Clustering (Ward) và DBSCAN**; thực nghiệm trên Google Colab và ứng dụng web Python + Streamlit.

## Kết quả chính (phiên bản hoàn thiện)

| Thành phần | Thiết kế & Kết quả |
|---|---|
| **Tiền xử lý** | Kiểm tra dữ liệu (`validate`), phát hiện & tách **108 ngoại lai bằng quy tắc IQR** (giữ 332 dòng chuẩn để huấn luyện; ngoại lai vẫn được gắn nhãn chăm sóc VIP ở đầu ra), chuẩn hóa bằng **`StandardScaler`** (đối chiếu với **`MinMaxScaler`**). |
| **K-Means** | Khảo sát $k = 2 \dots 8$; **Elbow Method** gợi ý **$k = 4$**; Silhouette Score = `0.271`, Davies–Bouldin = `1.281` (**Đạt tiêu chí**). |
| **Hierarchical (Ward)** | Khảo sát $k = 2 \dots 8$ + biểu đồ **Dendrogram**; tốt nhất tại **$k = 3$** với Silhouette Score = `0.276`, Davies–Bouldin = `1.448` (**Đạt tiêu chí**). |
| **DBSCAN** | Chọn `eps` từ biểu đồ **k-distance**; tốt nhất tại `eps = 1.1, min_samples = 3` $\rightarrow$ **3 cụm**, tỷ lệ nhiễu `19.9%` ($\le 30\%$), Silhouette = `0.229` (**Đạt tiêu chí**). |
| **Phân khúc** | 3 phân khúc rõ rệt (*Nhóm 0: Khách mua sắm tiết kiệm*; *Nhóm 1: Ưu tiên Tạp hóa & Gia dụng*; *Nhóm 2: Khách lớn đa ngành VIP*) kèm chiến lược khuyến mãi riêng. |

## Chạy trên Google Colab (cho cả nhóm)

1. Mở https://colab.research.google.com/ $\rightarrow$ *File $\rightarrow$ Upload notebook* $\rightarrow$ chọn `Wholesale_Customers_Clustering.ipynb`.
2. Chọn *Runtime $\rightarrow$ Run all* (dùng CPU mặc định).
3. Notebook tự tải dữ liệu từ UCI (có bản nhúng sẵn dự phòng nếu mạng/UCI lỗi) và chạy đầy đủ từ EDA, Boxplot IQR, Elbow, Silhouette, Dendrogram, k-distance đến PCA 2D.
4. Ô cuối (Mục 6) tự khởi chạy **giao diện Streamlit ngay trên Colab** và đóng gói file `wholesale_results.zip`.

> Lưu ý cho nhóm: `Wholesale_Customers_Clustering.ipynb` được sinh tự động từ `build_notebook.py`. Nếu chỉnh sửa logic, hãy sửa trong `pipeline.py` / `build_notebook.py` rồi chạy `python build_notebook.py` để đồng bộ.

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

Mở trình duyệt tại `http://localhost:8501`. Toàn bộ đường dẫn trong code đều dùng đường dẫn tương đối (`Path(__file__).resolve().parent`), nên đặt thư mục ở đâu trên máy nào cũng chạy bình thường. Ứng dụng gồm 4 tab đúng theo yêu cầu:
1. **Dữ liệu & Khám phá (EDA):** Xem 440 dòng, thống kê mô tả, biểu đồ Boxplot phát hiện ngoại lai.
2. **Tiền xử lý & So sánh 3 thuật toán:** Chọn trực tiếp IQR / `StandardScaler` / `MinMaxScaler`, xem Elbow, Silhouette, Dendrogram, k-distance và bảng so sánh 3 thuật toán.
3. **Kết quả phân khúc & Khuyến mãi:** Biểu đồ PCA 2D, chi tiêu tương đối, tra cứu từng khách hàng và xuất phiếu khuyến mãi JSON/CSV.
4. **Nhập liệu mới & Huấn luyện lại:** Form nhập gọn nhẹ (có 3 nút điền mẫu thử nhanh 1-click), **in ngay phân khúc và khuyến mãi** sau khi lưu, upload CSV, huấn luyện lại thủ công hoặc tự động định kỳ ở thanh bên.

## Huấn luyện định kỳ ngoài giao diện (Windows Task Scheduler)

Có thể đặt lịch chạy tự động bằng lệnh:
```powershell
python retrain.py --if-due
```
(Trong Task Scheduler: *Program/script* trỏ tới file `python.exe` trong `.venv\Scripts\` của bạn, *Arguments* điền `retrain.py --if-due`, *Start in* điền đường dẫn thư mục dự án trên máy bạn).
Chỉ huấn luyện khi có ít nhất 10 khách hàng mới (`--min-new`) hoặc quá 7 ngày (`--max-days`) kể từ lần huấn luyện trước.

## Cấu trúc dự án (Sạch sẽ để Push Git)

- `pipeline.py`: Tiền xử lý (validate, IQR, StandardScaler/MinMaxScaler), 3 thuật toán, đánh giá (Elbow, Silhouette), phân khúc & dự đoán khách mới.
- `storage.py`: Lưu trữ SQLite (`data/customers.sqlite3`), chống nhập lặp batch theo SHA-256.
- `app.py`, `experience.py`: Giao diện web Streamlit 4 tab.
- `retrain.py`: Script huấn luyện lại theo dòng lệnh / Task Scheduler.
- `build_notebook.py`: Tạo `Wholesale_Customers_Clustering.ipynb` cho Google Colab.
- `docs/BAO_CAO.md`: Báo cáo chi tiết bài tập lớn (6 chương) & bảng phân công 5 thành viên.
- `tests/test_system.py`: Bộ kiểm thử tự động (`python -m pytest tests -q`).
- `data/wholesale_customers.csv`: Bộ dữ liệu gốc UCI (440 dòng).

*(Các file tạm như `*_executed.ipynb`, `*.sqlite3`, `artifacts/`, `*.zip`, `.venv/` đã được cấu hình trong `.gitignore` để không đẩy lên Git).*
