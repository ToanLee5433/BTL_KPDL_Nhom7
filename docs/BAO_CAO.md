# BÁO CÁO BÀI TẬP LỚN MÔN KHAI PHÁ DỮ LIỆU

**Đề tài:** Phân khúc khách hàng bán lẻ (Mô hình RFM) để gợi ý khuyến mãi (Bài toán Phân cụm)  
**Nhóm thực hiện:** Nhóm 7 — 5 thành viên *(điền họ tên và mã sinh viên trước khi nộp)*

| STT | Họ và tên | Mã sinh viên | Phần phụ trách chính |
|---|---|---|---|
| 1 | *(Thành viên 1)* | | Giới thiệu bài toán, làm sạch 541,909 giao dịch & khám phá dữ liệu RFM (EDA) — **Chương 1** |
| 2 | *(Thành viên 2)* | | Tiền xử lý: phát hiện ngoại lai (IQR) & chuẩn hóa dữ liệu — **Chương 2** |
| 3 | *(Thành viên 3)* | | Thuật toán K-Means & đánh giá bằng Elbow, Silhouette — **Chương 3 (3.1, 3.3) & 4.1** |
| 4 | *(Thành viên 4)* | | Thuật toán Hierarchical, DBSCAN & phân tích khuyến mãi RFM — **Chương 3 (3.2) & 4.2, 4.3** |
| 5 | *(Thành viên 5 - Nhóm trưởng)* | | Xây dựng Google Colab, giao diện Streamlit, SQLite & huấn luyện định kỳ — **Chương 5 & Tổng hợp** |

---

## CHƯƠNG 1. GIỚI THIỆU BÀI TOÁN, LÀM SẠCH VÀ KHÁM PHÁ DỮ LIỆU (EDA)
*(Phụ trách viết & thuyết trình: Thành viên 1)*

### 1.1. Đặt vấn đề và lý do chọn mô hình RFM
Trong hoạt động bán lẻ hiện đại, khách hàng có hành vi mua sắm rất đa dạng: có người mới mua một vài lần gần đây, có khách hàng trung thành mua đều đặn hàng tháng với giá trị đơn hàng cao, nhưng cũng có những khách hàng từng mua rồi bỏ hẳn nhiều tháng không quay lại. Nếu áp dụng một chương trình khuyến mãi chung cho tất cả mọi người, cửa hàng vừa tốn chi phí ưu đãi cho những khách vốn dĩ đã mua nhiều, vừa không kéo được những khách sắp rời bỏ quay lại.

Nhóm lựa chọn bài toán **Phân cụm (Clustering)** kết hợp với **mô hình RFM** (Recency – Frequency – Monetary), mô hình phân tích hành vi khách hàng phổ biến nhất trong ngành bán lẻ, để chia tập khách hàng thành các nhóm đặc trưng và gợi ý chiến lược khuyến mãi riêng cho từng nhóm.

### 1.2. Bộ dữ liệu gốc Online Retail (UCI ID 352) và bước làm sạch giao dịch
Nhóm sử dụng bộ dữ liệu **Online Retail Data Set** trên kho dữ liệu UCI Machine Learning Repository (tác giả Daqing Chen, 2015, DOI: `10.24432/C5BW33`). Dữ liệu gốc ghi nhận toàn bộ giao dịch thực tế của một hệ thống bán lẻ trực tuyến tại Anh trong vòng 1 năm (từ 01/12/2010 đến 09/12/2011), với quy mô **541,909 dòng giao dịch** và **8 cột**:
- `InvoiceNo`: Mã hóa đơn (nếu bắt đầu bằng chữ `C` là hóa đơn bị hủy).
- `StockCode`: Mã sản phẩm.
- `Description`: Tên sản phẩm.
- `Quantity`: Số lượng sản phẩm trong dòng giao dịch.
- `InvoiceDate`: Ngày và giờ phát sinh giao dịch.
- `UnitPrice`: Đơn giá sản phẩm (tính bằng bảng Anh `£`).
- `CustomerID`: Mã định danh duy nhất của khách hàng.
- `Country`: Quốc gia cư trú của khách hàng.

Vì đây là dữ liệu giao dịch thực tế nên chứa rất nhiều nhiễu. Trước khi tổng hợp thành hồ sơ khách hàng, nhóm thực hiện làm sạch theo 3 bước:
1. Loại bỏ **135,080 dòng bị thiếu `CustomerID`** (vì không xác định được khách hàng để phân khúc).
2. Loại bỏ **8,905 dòng thuộc hóa đơn hủy** (`InvoiceNo` bắt đầu bằng `'C'`).
3. Loại bỏ **40 dòng có `Quantity <= 0` hoặc `UnitPrice <= 0`** (lỗi nhập liệu hoặc hàng tặng không tính doanh thu).

Sau khi làm sạch, dữ liệu còn lại **397,884 giao dịch hợp lệ**. Từ đây, nhóm tính doanh thu từng dòng (`TotalPrice = Quantity * UnitPrice`) và gom nhóm theo `CustomerID` để tạo bảng **4,338 khách hàng** với 3 chỉ số **RFM**:
- **Recency (`R`):** Số ngày tính từ lần mua hàng gần nhất của khách hàng đến ngày chốt dữ liệu (càng nhỏ chứng tỏ khách vừa mới mua gần đây).
- **Frequency (`F`):** Tổng số hóa đơn mua hàng hợp lệ của khách hàng (càng lớn chứng tỏ khách mua càng thường xuyên).
- **Monetary (`M`):** Tổng số tiền (`£`) khách hàng đã chi tiêu trong kỳ (càng lớn chứng tỏ giá trị đóng góp doanh thu càng cao).

### 1.3. Thống kê mô tả dữ liệu RFM (4,338 khách hàng)
Phân bố của 4,338 khách hàng đến từ nhiều quốc gia (nhiều nhất là Vương quốc Anh với 3,920 khách, tiếp theo là Đức 94 khách, Pháp 87 khách, Tây Ban Nha 29 khách, Bỉ 24 khách...) có các chỉ số thống kê như sau:

| Chỉ số RFM | Nhỏ nhất (Min) | Trung vị (50%) | Trung bình (Mean) | Lớn nhất (Max) | Độ lệch chuẩn (Std) | Độ lệch (Skewness) |
|---|---:|---:|---:|---:|---:|---:|
| `Recency` (ngày) | 1.0 | 51.0 | 92.54 | 374.0 | 100.01 | 1.25 |
| `Frequency` (đơn) | 1.0 | 2.0 | 4.27 | 209.0 | 7.70 | 12.07 |
| `Monetary` (£) | 3.75 | 674.48 | 2,054.27 | 280,206.02 | 8,989.23 | 19.32 |

Từ bảng thống kê có thể thấy ngay hai đặc điểm lớn:
1. **Dữ liệu lệch phải rất nặng ở `Frequency` và `Monetary`:** Một nửa số khách hàng chỉ mua từ 1 đến 2 đơn (`Trung vị = 2.0`) và chi dưới `£674.48`, nhưng có những khách hàng mua sỉ đặt tới **209 đơn** và chi tới **£280,206** (làm hệ số lệch của `Monetary` vọt lên `19.32`).
2. **Đơn vị và biên độ chênh lệch lớn:** `Monetary` dao động tới hàng trăm nghìn bảng Anh, trong khi `Frequency` chỉ từ 1 đến vài trăm. Nếu không xử lý ngoại lai và chuẩn hóa, khoảng cách giữa hai khách hàng sẽ hoàn toàn bị chi phối bởi cột `Monetary`.

---

## CHƯƠNG 2. TIỀN XỬ LÝ DỮ LIỆU
*(Phụ trách viết & thuyết trình: Thành viên 2)*

### 2.1. Kiểm tra ràng buộc dữ liệu đầu vào (`validate`)
Để đảm bảo cả dữ liệu tổng hợp ban đầu lẫn dữ liệu khách hàng mới nhập trên giao diện đều sạch, hệ thống kiểm tra chặt chẽ các điều kiện: đủ 6 cột (`CustomerID`, `Recency`, `Frequency`, `Monetary`, `TotalItems`, `Country`), không chứa ô trống hay giá trị vô hạn, `CustomerID >= 1`, `Recency >= 0`, `Frequency >= 1` và `Monetary > 0`.

### 2.2. Phát hiện và xử lý ngoại lai bằng quy tắc IQR
Quan sát biểu đồ **Boxplot** của 3 biến `Recency`, `Frequency`, `Monetary` cho thấy có rất nhiều điểm nằm vượt xa râu trên của hộp. Đây là những khách hàng mua sỉ hoặc tổ chức có tần suất và số tiền chi tiêu gấp hàng chục đến hàng trăm lần khách hàng lẻ thông thường. Nếu giữ nguyên khi chạy phân cụm, các điểm này sẽ kéo lệch tâm cụm của K-Means ra xa đám đông, làm các nhóm khách hàng phổ thông bị dồn chung vào một cụm.

Nhóm sử dụng quy tắc **IQR (Interquartile Range)** cho từng chỉ số RFM:
- Tính $IQR = Q_3 - Q_1$ (khoảng giữa phân vị 75% và 25%).
- Những khách hàng có giá trị nằm ngoài khoảng $[Q_1 - 1.5 \times IQR,\; Q_3 + 1.5 \times IQR]$ ở bất kỳ chỉ số nào sẽ được tách thành tập ngoại lai.

Kết quả lọc trên 4,338 khách hàng: phát hiện **628 khách hàng ngoại lai**, giữ lại **3,710 khách hàng chuẩn** để huấn luyện mô hình.

**Cách xử lý hợp lý về mặt kinh doanh:** Khác với lỗi cảm biến, 628 khách hàng ngoại lai ở đây chính là **những khách hàng lớn (VIP / khách mua buôn)** mang lại doanh thu rất cao cho doanh nghiệp. Vì thế nhóm không xóa vĩnh viễn họ khỏi hệ thống: họ chỉ được tách ra lúc huấn luyện để tìm tâm cụm chuẩn cho số đông 3,710 khách hàng, sau đó vẫn được gán vào cụm gần nhất kèm cờ **`Ngoại lai VIP (IQR)`** để bộ phận kinh doanh có chế độ chăm sóc riêng.

### 2.3. Chuẩn hóa dữ liệu (`StandardScaler` và `MinMaxScaler`)
Vì K-Means, Hierarchical và DBSCAN đều tính khoảng cách Euclid trong không gian 3 chiều $(R, F, M)$, việc chuẩn hóa về cùng thang đo là bắt buộc. Nhóm thực nghiệm cả hai phương pháp chuẩn hóa chuẩn:
1. **StandardScaler (Z-score):** $z = \frac{x - \mu}{\sigma}$ (đưa các cột về trung bình 0 và độ lệch chuẩn 1).
2. **MinMaxScaler:** $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ (co giãn các giá trị về đoạn $[0, 1]$).

Bộ chuẩn hóa chỉ được khớp (`fit`) trên tập 3,710 khách hàng chuẩn để tránh bị 628 điểm ngoại lai làm méo tham số $\mu, \sigma$ hay $x_{\max}$. Kết quả so sánh ở Chương 4 cho thấy `StandardScaler` giúp cả 3 thuật toán cùng hội tụ ổn định và đạt tiêu chí phân cụm.

---

## CHƯƠNG 3. CÁC THUẬT TOÁN PHÂN CỤM VÀ PHƯƠNG PHÁP ĐÁNH GIÁ
*(Phụ trách viết & thuyết trình: Thành viên 3 & Thành viên 4)*

### 3.1. Thuật toán K-Means *(Thành viên 3)*
K-Means chia $N$ khách hàng thành $k$ cụm sao cho tổng bình phương khoảng cách từ mỗi khách hàng tới tâm cụm của họ (Inertia / WCSS) là nhỏ nhất:
$$J = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2$$
Nhóm thiết lập `n_init = 20` (chạy 20 lần khởi tạo tâm khác nhau để chọn nghiệm tốt nhất) và `random_state = 42`. K-Means xử lý rất nhanh trên 3,710 khách hàng và cho phép gán nhãn nhanh khách hàng mới qua tọa độ tâm cụm.

### 3.2. Thuật toán Hierarchical Clustering & DBSCAN *(Thành viên 4)*
1. **Hierarchical Clustering (Agglomerative — liên kết Ward):** Xây dựng cây phân cấp từ dưới lên (vẽ bằng biểu đồ **Dendrogram**). Ở mỗi bước, thuật toán gộp hai cụm làm tổng phương sai nội cụm (Ward) tăng lên ít nhất. Khi chọn số cụm $k$, ta cắt ngang cây Dendrogram ở mức tương ứng.
2. **DBSCAN:** Gom cụm dựa trên mật độ điểm lân cận thông qua bán kính `eps` và số điểm tối thiểu `min_samples`. Những điểm nằm ở vùng thưa sẽ được đánh dấu là nhiễu (`-1`). Nhóm sử dụng biểu đồ **k-distance** để quan sát vùng chuyển tiếp mật độ và khảo sát lưới tham số `eps ∈ [0.35, 0.4, 0.45, 0.5, 0.6]` với `min_samples ∈ [10, 20, 30, 50]`.

### 3.3. Phương pháp đánh giá *(Thành viên 3)*
- **Phương pháp Khuỷu tay (Elbow Method):** Vẽ đồ thị Inertia của K-Means theo $k = 2 \dots 8$ và xác định điểm gấp khúc nơi tốc độ giảm Inertia bắt đầu chậm lại.
- **Chỉ số Silhouette Score:** Đo độ gắn kết trong cụm so với độ tách biệt với cụm gần nhất ($s \in [-1, 1]$, càng gần $1$ càng tốt).
- **Chỉ số bổ trợ:** Davies–Bouldin Index (càng nhỏ càng tốt), Calinski–Harabasz Index (càng lớn càng tốt) và tỷ lệ nhiễu (với DBSCAN).
- **Tiêu chí chọn mô hình:** Để phân khúc RFM đủ chi tiết cho việc thiết kế khuyến mãi, K-Means và Hierarchical phải có từ **3 cụm trở lên ($k \ge 3$)**, DBSCAN có từ 2 cụm trở lên với tỷ lệ nhiễu **$\le 30\%$**, mỗi cụm có tối thiểu 5 khách hàng; mô hình chính được chọn theo **Silhouette Score cao nhất**.

---

## CHƯƠNG 4. KẾT QUẢ THỰC NGHIỆM VÀ GỢI Ý KHUYẾN MÃI
*(Phụ trách phân tích: Thành viên 3 mục 4.1; Thành viên 4 mục 4.2 & 4.3)*

### 4.1. So sánh bộ chuẩn hóa & Khảo sát K-Means (Elbow, Silhouette) *(Thành viên 3)*
Kết quả đối chiếu hai bộ chuẩn hóa trên tập 3,710 khách hàng sạch (sau khi tách 628 ngoại lai IQR):

| Bộ chuẩn hóa | Thuật toán | Tham số tốt nhất | Số cụm | Tỷ lệ nhiễu | Silhouette | Davies–Bouldin | Đạt tiêu chí |
|---|---|---|---:|---:|---:|---:|---|
| **StandardScaler** | **K-Means** | `k = 3` | 3 | 0.0% | **0.460** | **0.766** | Đạt |
| **StandardScaler** | **Hierarchical (Ward)** | `k = 3` | 3 | 0.0% | **0.444** | 0.787 | Đạt |
| **StandardScaler** | **DBSCAN** | `eps = 0.5, min_samples = 50` | 2 | 12.9% | **0.451** | 0.466 | Đạt |
| MinMaxScaler | K-Means | `k = 3` | 3 | 0.0% | 0.479 | 0.757 | Đạt |
| MinMaxScaler | Hierarchical (Ward) | `k = 3` | 3 | 0.0% | 0.446 | 0.783 | Đạt |
| MinMaxScaler | DBSCAN | `eps = 0.08, min_samples = 10` | 11 | 10.5% | 0.015 | 2.401 | Đạt (nhưng phân mảnh 11 cụm, Sil rất thấp) |

Dễ thấy `MinMaxScaler` làm DBSCAN bị phân mảnh thành 11 cụm nhỏ với Silhouette chỉ đạt `0.015`. Ngược lại, **`StandardScaler` giúp cả 3 thuật toán đều đạt chất lượng rất đồng đều (Silhouette từ `0.444` đến `0.460`)**, nên nhóm chọn `StandardScaler` làm phương án chuẩn hóa chính.

Khi khảo sát K-Means trên `StandardScaler`:
- **Tại $k = 3$:** Silhouette đạt đỉnh cao nhất trong vùng $k \ge 3$ là **`0.460`**, Davies–Bouldin đạt **`0.766`**, Calinski–Harabasz đạt **`4,036.36`**, thời gian chạy chỉ **`0.077s`**.
- **Theo phương pháp Elbow:** Đường Inertia giảm mạnh nhất từ $k=2$ xuống $k=3$ (`3,502.54`) và $k=4$ (điểm khuỷu hình học gợi ý $k=4$, Silhouette `0.413`). Kết hợp cả hai tiêu chí, **$k = 3$** cho sự cân bằng tốt nhất giữa độ tách biệt cụm (Silhouette cao hơn) và ý nghĩa phân khúc RFM rõ ràng.

### 4.2. So sánh 3 thuật toán *(Thành viên 4)*
1. **K-Means (`k = 3`):** Đạt Silhouette cao nhất (**`0.460`**), Davies–Bouldin thấp (`0.766`), thời gian chạy nhanh nhất (`0.077s`), bao phủ 100% tập dữ liệu và có sẵn tâm cụm để dự đoán nhanh khách hàng mới. Đây là mô hình chính được hệ thống lựa chọn.
2. **Hierarchical Clustering (`k = 3`, Ward):** Cho kết quả sát với K-Means với Silhouette đạt **`0.444`**, Davies–Bouldin `0.787`, thời gian chạy `0.679s` (chậm hơn K-Means khoảng 9 lần do phải tính ma trận khoảng cách $3,710 \times 3,710$).
3. **DBSCAN (`eps = 0.5, min_samples = 50`):** Tách được 2 cụm lớn (Silhouette `0.451`), bỏ lại `12.9%` điểm ở vùng biên thành nhiễu. Dù Silhouette trên các điểm lõi khá cao, việc không phân nhóm hết khách hàng và khó tách sâu hơn 2 cụm khiến DBSCAN kém phù hợp hơn K-Means cho bài toán khuyến mãi toàn tệp.

*(Biểu đồ PCA 2D giữ lại **91.1%** phương sai của dữ liệu — PC1 66.5% và PC2 24.7% — cho thấy 3 cụm tách biệt khá rõ ràng trên mặt phẳng).*

### 4.3. Phân tích 3 phân khúc RFM và Gợi ý khuyến mãi *(Thành viên 4)*
Sau khi gán 628 khách hàng ngoại lai VIP vào cụm gần nhất để theo dõi trọn vẹn **4,338 khách hàng**, mô hình K-Means ($k=3$) cho ra 3 phân khúc rất rõ nét:

| Phân khúc | Số lượng KH | Chỉ số RFM trung bình | Nhận diện hành vi | Gợi ý chương trình khuyến mãi |
|---|---|---|---|---|
| **Nhóm 0 — Khách hàng mới / Tiềm năng phát triển** | **1,968 KH** *(45.4%)* | Mua cách đây **49 ngày** (`R=0.54x`), **2.1 đơn** (`F=0.75x`), chi **£599** (`M=0.67x`) | Chiếm đông nhất; mới mua hàng gần đây nhưng số lần mua và giá trị đơn còn ở mức trung bình thấp. | Tặng **mã miễn phí vận chuyển (Freeship)** và **ưu đãi combo cho đơn hàng thứ 2** để tạo thói quen quay lại mua sắm. |
| **Nhóm 1 — Khách hàng ngủ quên / Nguy cơ rời bỏ** | **1,074 KH** *(24.8%, gồm 154 ngoại lai)* | Lâu chưa mua (**227 ngày**, `R=2.52x`), chỉ **1.5 đơn** (`F=0.53x`), chi **£414** (`M=0.47x`) | Đã lâu không phát sinh đơn mới (trung bình hơn 7 tháng), tần suất và mức chi thấp nhất. | Gửi **Email/SMS tái kích hoạt** kèm **Voucher giảm giá 15% có thời hạn 7 ngày** để tạo động lực quay lại ngay. |
| **Nhóm 2 — Khách hàng trung thành / Chi tiêu cao** | **1,296 KH** *(29.9%, gồm 474 ngoại lai VIP)* | Mới mua gần đây (**36 ngày**, `R=0.40x`), **6.0 đơn** (`F=2.13x`), chi **£2,108** (`M=2.38x`) | Nhóm khách hàng nòng cốt mang lại doanh thu chủ lực: mua thường xuyên, mới mua gần đây và chi tiêu rất cao. | Tặng **thẻ thành viên VIP**, chương trình **tích điểm nhân đôi** và ưu tiên **giảm giá cho đơn hàng tiếp theo**. |

*(Riêng **628 khách hàng ngoại lai IQR** — đa phần nằm trong Nhóm 2 — được hệ thống gắn cờ riêng để bộ phận CSKH liên hệ chăm sóc 1-1 và áp dụng chiết khấu theo đơn hàng lớn).*

---

## CHƯƠNG 5. ỨNG DỤNG DEMO STREAMLIT VÀ GOOGLE COLAB
*(Phụ trách xây dựng & demo trực tiếp: Thành viên 5 - Nhóm trưởng)*

Hệ thống được triển khai đồng bộ trên **Google Colab** (`Online_Retail_RFM_Clustering.ipynb`) và **ứng dụng web Streamlit + SQLite** (`app.py`, `pipeline.py`, `storage.py`, `experience.py`, `retrain.py`) đáp ứng đủ 4 yêu cầu của ứng dụng:
1. **Google Colab:** Tự động clone mã nguồn từ GitHub, tải bộ dữ liệu Online Retail từ UCI để chạy làm sạch 541,909 giao dịch $\rightarrow$ 4,338 khách hàng RFM, vẽ đầy đủ biểu đồ và hỗ trợ mở thẳng giao diện Streamlit trên Colab.
2. **Giao diện Streamlit 4 tab:**
   - **Tab 1 & Tab 2:** Hiển thị EDA 4,338 khách hàng RFM, Boxplot ngoại lai IQR, cho chọn `StandardScaler`/`MinMaxScaler`, vẽ Elbow, Silhouette, Dendrogram, k-distance và bảng so sánh 3 thuật toán.
   - **Tab 3:** Hiển thị bản đồ cụm PCA 2D, bảng tỷ lệ RFM, tra cứu từng `CustomerID` để xuất phiếu khuyến mãi JSON/CSV.
   - **Tab 4 (Thu thập dữ liệu mới & Huấn luyện lại):** Nhập khách hàng mới qua form (có 3 nút điền mẫu thử nhanh 1-click) hoặc upload CSV, lưu vào SQLite (`data/retail_customers.sqlite3`), **hiển thị ngay phân khúc RFM và gợi ý khuyến mãi ra màn hình**. Thanh bên hỗ trợ chính sách tự huấn luyện định kỳ khi đủ $\ge N$ khách mới (mặc định 20) hoặc quá $X$ ngày (mặc định 7), hoặc chạy qua Windows Task Scheduler (`python retrain.py --if-due`). Toàn bộ mã nguồn đạt **100% kiểm thử tự động (`pytest`)**.

---

## CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 6.1. Kết luận
Nhóm đã hoàn thành trọn vẹn quy trình khai phá dữ liệu trên tập **Online Retail (541,909 giao dịch $\rightarrow$ 4,338 khách hàng RFM)**: từ làm sạch giao dịch thực tế, xử lý ngoại lai IQR, chuẩn hóa dữ liệu đến đối chiếu 3 thuật toán K-Means, Hierarchical và DBSCAN, và xây dựng ứng dụng web Streamlit hoàn chỉnh. Kết quả cho thấy mô hình **K-Means (`k=3`) trên `StandardScaler`** đạt Silhouette **`0.460`**, tách được 3 phân khúc RFM rõ rệt để gợi ý khuyến mãi sát với thực tế kinh doanh.

### 6.2. Hạn chế và hướng phát triển
- **Hạn chế:** Mô hình RFM hiện tại mới xét hành vi mua tổng thể, chưa xét đến sở thích theo từng nhóm sản phẩm (`StockCode`/`Description`) và chưa có dữ liệu tỷ lệ phản hồi thực tế sau khi gửi khuyến mãi.
- **Hướng phát triển:** Kết hợp thêm thuật toán khai phá luật kết hợp (Apriori/FP-Growth) trên giỏ hàng để gợi ý kèm sản phẩm mua cùng, và triển khai A/B Testing để đo lường doanh thu tăng thêm của từng chiến dịch khuyến mãi.
