# BÁO CÁO BÀI TẬP LỚN MÔN KHAI PHÁ DỮ LIỆU

**Đề tài:** Phân khúc khách hàng siêu thị để gợi ý khuyến mãi (Bài toán Phân cụm)  
**Nhóm thực hiện:** Nhóm gồm 4 thành viên *(điền họ tên và mã sinh viên trước khi nộp)*

| STT | Họ và tên | Mã sinh viên | Vai trò chính |
|---|---|---|---|
| 1 | *(Thành viên 1)* | | Tiền xử lý dữ liệu & Khám phá (EDA) |
| 2 | *(Thành viên 2)* | | Thuật toán K-Means & Đánh giá (Elbow, Silhouette) |
| 3 | *(Thành viên 3)* | | Thuật toán Hierarchical, DBSCAN & So sánh |
| 4 | *(Thành viên 4)* | | Xây dựng ứng dụng Streamlit, Lưu trữ SQLite & Huấn luyện định kỳ |

---

## CHƯƠNG 1. GIỚI THIỆU BÀI TOÁN VÀ BỘ DỮ LIỆU

### 1.1. Đặt vấn đề và mục tiêu
Trong hoạt động kinh doanh bán lẻ / siêu thị, việc áp dụng một chương trình khuyến mãi đồng loạt cho tất cả khách hàng thường không mang lại hiệu quả cao và gây lãng phí ngân sách ưu đãi. Mỗi nhóm khách hàng có thói quen mua sắm và mức chi tiêu khác nhau (ví dụ: nhóm ưu tiên hàng tươi sống, nhóm mua nhiều hàng tạp hóa – chất tẩy rửa, nhóm mua số lượng lớn...).

Mục tiêu của bài toán là áp dụng **bài toán Phân cụm (Clustering — học không giám sát)** để:
1. Phân chia tập khách hàng thành các phân khúc có hành vi chi tiêu tương đồng.
2. Phân tích đặc điểm từng phân khúc để **gợi ý chương trình khuyến mãi phù hợp**.
3. Xây dựng ứng dụng web đơn giản cho phép thu thập thêm khách hàng mới, tiền xử lý, huấn luyện lại mô hình định kỳ và hiển thị kết quả phân khúc cho người dùng.

### 1.2. Mô tả bộ dữ liệu
- **Tên bộ dữ liệu:** *Wholesale Customers Data Set*  
- **Nguồn:** UCI Machine Learning Repository — Margarida Cardoso (2013), DOI: `10.24432/C5030X`, giấy phép CC BY 4.0.  
- **Quy mô:** **440 dòng** (tương ứng 440 khách hàng) và **8 cột**. Không có giá trị thiếu (`Missing = 0`), không có dòng trùng lặp hoàn toàn (`Duplicates = 0`).

| Tên cột | Kiểu dữ liệu | Ý nghĩa thuộc tính | Vai trò trong bài toán |
|---|---|---|---|
| `Fresh` | Số thực | Tổng chi tiêu năm cho nhóm hàng tươi sống | Thuộc tính đầu vào phân cụm |
| `Milk` | Số thực | Tổng chi tiêu năm cho sữa và chế phẩm | Thuộc tính đầu vào phân cụm |
| `Grocery` | Số thực | Tổng chi tiêu năm cho hàng tạp hóa | Thuộc tính đầu vào phân cụm |
| `Frozen` | Số thực | Tổng chi tiêu năm cho hàng đông lạnh | Thuộc tính đầu vào phân cụm |
| `Detergents_Paper` | Số thực | Tổng chi tiêu năm cho chất tẩy rửa & sản phẩm giấy | Thuộc tính đầu vào phân cụm |
| `Delicassen` | Số thực | Tổng chi tiêu năm cho thực phẩm chế biến sẵn | Thuộc tính đầu vào phân cụm |
| `Channel` | Phân loại (`1`, `2`) | `1`: Khách sạn/Nhà hàng/Café (HoReCa); `2`: Bán lẻ/Siêu thị | Mô tả, đối chiếu sau phân cụm |
| `Region` | Phân loại (`1`, `2`, `3`) | `1`: Lisbon; `2`: Oporto; `3`: Khu vực khác | Mô tả, đối chiếu sau phân cụm |

> **Lưu ý thiết kế:** Chỉ đưa 6 cột chi tiêu vào tính khoảng cách phân cụm. `Channel` và `Region` là biến danh mục (mã số định danh), không mang ý nghĩa khoảng cách hình học nên chỉ dùng để đối chiếu tỷ lệ kênh/khu vực của từng cụm sau khi phân cụm xong.

---

## CHƯƠNG 2. TIỀN XỬ LÝ DỮ LIỆU

### 2.1. Kiểm tra chất lượng dữ liệu (Validation)
Trước khi lưu trữ và đưa vào mô hình, dữ liệu (cả dữ liệu gốc và dữ liệu mới nhập từ form/CSV) đều đi qua hàm kiểm tra `validate()` trong `pipeline.py`:
- Kiểm tra đủ 8 cột và không trùng tên cột.
- Chặn giá trị thiếu (`NaN`), chữ, giá trị vô hạn (`Inf`), giá trị chi tiêu âm (`< 0`).
- Chặn bản ghi có tổng chi tiêu cả 6 nhóm bằng `0`.
- Kiểm tra miền giá trị hợp lệ của `Channel` (`1` hoặc `2`) và `Region` (`1`, `2`, `3`).

### 2.2. Phát hiện và loại bỏ ngoại lai (Outliers) bằng quy tắc IQR
Dữ liệu chi tiêu có phân phối lệch phải mạnh (một số ít khách hàng chi tiêu đột biến gấp nhiều lần mức trung bình). Vì các thuật toán phân cụm dựa trên khoảng cách Euclid rất nhạy với ngoại lai, nhóm áp dụng quy tắc **IQR (Interquartile Range)**:
- Tính tứ phân vị $Q_1$ (25%), $Q_3$ (75%) và $IQR = Q_3 - Q_1$ cho từng cột trong 6 nhóm hàng.
- Ngưỡng hợp lệ: $[Q_1 - 1.5 \times IQR,\; Q_3 + 1.5 \times IQR]$.
- Kết quả trên 440 khách hàng: phát hiện **108 khách hàng ngoại lai**, giữ lại **332 khách hàng chuẩn** để huấn luyện mô hình phân cụm.
- **Cách xử lý khéo léo:** 108 khách hàng ngoại lai chỉ được tách ra ở bước huấn luyện để không kéo lệch tâm cụm, sau đó vẫn được gán vào cụm gần nhất kèm nhãn **`Ngoại lai (IQR)`** và chiến lược **chăm sóc VIP riêng** nhằm không làm mất khách hàng lớn của siêu thị.

### 2.3. Chuẩn hóa dữ liệu (Feature Scaling)
Biên độ chi tiêu giữa các nhóm hàng chênh lệch lớn (ví dụ `Fresh` có giá trị lên tới hàng chục nghìn, trong khi `Delicassen` nhỏ hơn nhiều). Do K-Means, Hierarchical và DBSCAN đều sử dụng khoảng cách hình học, bắt buộc phải đưa 6 thuộc tính về cùng thang đo:
- **StandardScaler (Z-score):** $z = \frac{x - \mu}{\sigma}$ (đưa về trung bình 0, độ lệch chuẩn 1).
- **MinMaxScaler:** $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ (đưa về đoạn $[0, 1]$).

Nhóm thực nghiệm đối chiếu cả hai bộ chuẩn hóa (`StandardScaler` và `MinMaxScaler`) ở Chương 4.

---

## CHƯƠNG 3. CÁC THUẬT TOÁN PHÂN CỤM VÀ CHỈ SỐ ĐÁNH GIÁ

### 3.1. Ba thuật toán áp dụng
1. **K-Means:**
   - Nguyên lý: Phân chia $N$ điểm dữ liệu vào $k$ cụm sao cho tổng bình phương khoảng cách từ mỗi điểm đến tâm cụm (Inertia / WCSS) là nhỏ nhất.
   - Cấu hình: Khảo sát $k = 2 \dots 8$, `n_init = 20`, `random_state = 42`.
   - Ưu/nhược điểm: Tốc độ nhanh, có sẵn hàm `predict()` cho dữ liệu mới; nhược điểm là phải chọn trước số cụm $k$ và giả định cụm có dạng cầu.
2. **Hierarchical Clustering (Agglomerative — liên kết Ward):**
   - Nguyên lý: Phân cụm phân cấp tích tụ (từ dưới lên), ban đầu mỗi điểm là một cụm, ở mỗi bước gộp hai cụm sao cho mức tăng phương sai nội cụm (Ward) là nhỏ nhất.
   - Cấu hình: Khảo sát $k = 2 \dots 8$, trực quan hóa bằng biểu đồ cây **Dendrogram**.
   - Ưu/nhược điểm: Không cần chọn số cụm ngay từ đầu (có thể quan sát Dendrogram để chọn điểm cắt); nhược điểm là độ phức tạp tính toán cao hơn với dữ liệu lớn và không có hàm `predict()` gốc cho điểm mới.
3. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):**
   - Nguyên lý: Gom nhóm dựa trên mật độ điểm lân cận thông qua 2 tham số: bán kính vùng lân cận `eps` và số điểm tối thiểu `min_samples`. Các điểm nằm ở vùng thưa được gắn nhãn nhiễu (`-1`).
   - Cấu hình: Dựa vào biểu đồ **k-distance** để định hướng vùng `eps`, khảo sát lưới `eps` × `min_samples ∈ {3, 4, 5, 6}`.
   - Ưu/nhược điểm: Không cần chỉ định trước số cụm $k$, phát hiện được cụm có hình dạng bất kỳ và tự loại điểm nhiễu; nhược điểm là nhạy với tham số `eps` và `min_samples`.

### 3.2. Phương pháp đánh giá
- **Phương pháp Khuỷu tay (Elbow Method):** Vẽ đồ thị Inertia (WCSS) của K-Means theo $k = 2 \dots 8$ và tìm điểm gấp khúc (nơi tốc độ giảm Inertia bắt đầu chậm lại).
- **Chỉ số Silhouette Score:** Đo độ gắn kết nội cụm so với độ tách biệt với cụm lân cận gần nhất, giá trị trong $[-1, 1]$ (càng gần $1$ càng tốt).
- **Các chỉ số bổ trợ:** Davies–Bouldin Index (càng nhỏ càng tốt), Calinski–Harabasz Index (càng lớn càng tốt), và tỷ lệ nhiễu (`noise_rate` đối với DBSCAN).
- **Quy tắc chọn mô hình:** Để phân khúc phục vụ gợi ý khuyến mãi có ý nghĩa kinh doanh (tránh $k=2$ chỉ tách lại 2 kênh `Channel` có sẵn), mô hình hợp lệ cần đạt: **số cụm $\ge 3$**, **kích thước cụm nhỏ nhất $\ge 5$**, **tỷ lệ nhiễu $\le 30\%$**, ưu tiên **Silhouette cao nhất** (Davies–Bouldin phá hòa).

---

## CHƯƠNG 4. KẾT QUẢ THỰC NGHIỆM VÀ PHÂN TÍCH PHÂN KHÚC

*(Số liệu dưới đây trích xuất trực tiếp từ phiên chạy chính thức `20261117T082722_797602ec` trên 440 khách hàng, loại 108 ngoại lai IQR, còn 332 dòng huấn luyện).*

### 4.1. So sánh bộ chuẩn hóa: StandardScaler vs MinMaxScaler
| Bộ chuẩn hóa | Thuật toán | Tham số | Số cụm | Tỷ lệ nhiễu | Silhouette | Davies–Bouldin | Đạt tiêu chí |
|---|---|---|---|---|---|---|---|
| **StandardScaler** | **K-Means** | `k = 4` | 4 | 0.0% | 0.271 | **1.281** | Đạt |
| **StandardScaler** | **Hierarchical** | `k = 3` | 3 | 0.0% | **0.276** | 1.448 | Đạt |
| **StandardScaler** | **DBSCAN** | `eps = 1.1, min_samples = 3` | 3 | 19.9% | 0.229 | 0.807 | Đạt |
| MinMaxScaler | K-Means | `k = 4` | 4 | 0.0% | 0.270 | 1.318 | Đạt |
| MinMaxScaler | Hierarchical | `k = 4` | 4 | 0.0% | 0.259 | 1.396 | Đạt |
| MinMaxScaler | DBSCAN | `eps = 0.25, min_samples = 5` | 2 | 28.3% | 0.428 | 0.615 | Không đạt (`< 3` cụm) |

**Nhận xét:** `StandardScaler` giúp **cả 3 thuật toán đều đạt tiêu chí phân cụm ($\ge 3$ cụm, nhiễu $\le 30\%$)** và cho Silhouette ổn định hơn so với `MinMaxScaler`, vì vậy nhóm chọn `StandardScaler` làm phương án chính.

### 4.2. So sánh 3 thuật toán (trên StandardScaler)
- **Phương pháp Elbow (K-Means):** Gợi ý điểm khuỷu tại **$k = 4$** (Inertia = `920.60`), trùng khớp với cấu hình tốt nhất của K-Means (`k = 4`, Silhouette = `0.271`, Davies–Bouldin = `1.281`, thời gian chạy `0.052s`).
- **Hierarchical (Ward):** Đạt Silhouette cao nhất trong các cấu hình hợp lệ là **`0.276`** tại **$k = 3$** (Davies–Bouldin = `1.448`, Calinski–Harabasz = `108.39`, thời gian chạy `0.002s`), nên được hệ thống chọn làm mô hình chính của phiên.
- **DBSCAN:** Tại `eps = 1.1, min_samples = 3`, phân tách được **3 cụm**, tỷ lệ nhiễu **19.9%** ($\le 30\%$), Silhouette trên các điểm không nhiễu đạt **`0.229`**. Tuy nhiên DBSCAN bỏ lại gần 20% khách hàng ở dạng nhiễu chưa gán nhóm, nên không thuận tiện bằng Hierarchical/K-Means khi triển khai khuyến mãi đại trà.

*(Khi trình bày báo cáo Word/Slide, nhóm chèn các hình chụp từ Notebook: `boxplot_outliers.png`, `elbow_and_silhouette.png`, `silhouette_per_k.png`, `dendrogram_and_kdistance.png`, `three_algorithms_pca.png`, `segment_heatmap.png`).*

### 4.3. Hồ sơ phân khúc và Gợi ý khuyến mãi
Hệ thống chia 440 khách hàng (gồm 332 khách chuẩn và 108 khách ngoại lai gán theo cụm gần nhất) thành **3 phân khúc rõ rệt**:

| Phân khúc | Số KH (Ngoại lai) | Trung vị chi tiêu nổi bật | Đặc điểm nhận diện | Gợi ý chương trình khuyến mãi |
|---|---|---|---|---|
| **Nhóm 0 — Khách mua sắm tiết kiệm / quy mô nhỏ** | **207** (25 ngoại lai) | `Fresh`: 8,305; `Frozen`: 1,662; `Grocery`: 2,175 | Chi tiêu tổng bằng **0.75 lần** mức chung; nổi bật ở **Hàng đông lạnh** (`1.20x`) và **Hàng tươi sống** (`1.07x`). Chỉ 2% thuộc kênh bán lẻ. | Tặng **Voucher giảm giá cho đơn hàng tiếp theo** và **Combo tiết kiệm nhóm Hàng đông lạnh + Hàng tươi sống** để tăng tần suất quay lại. |
| **Nhóm 1 — Nhóm ưu tiên Tạp hóa & Gia dụng** | **146** (37 ngoại lai) | `Grocery`: 10,518; `Milk`: 7,027; `Detergents_Paper`: 4,239 | Chi tiêu tổng bằng **1.24 lần** mức chung; nổi bật vượt trội ở **Chất tẩy rửa & giấy** (`2.28x`), **Tạp hóa** (`1.86x`), **Sữa** (`1.78x`). **70% thuộc kênh bán lẻ**. | **Khuyến mãi mua nhiều giảm giá** (mua 2 tặng 1 / chiết khấu 10% theo thùng) cho nhóm **Chất tẩy rửa & giấy, Tạp hóa**. |
| **Nhóm 2 — Khách hàng lớn đa ngành (VIP)** | **87** (46 ngoại lai) | `Fresh`: 18,567; `Delicassen`: 2,233; `Grocery`: 6,235 | Chi tiêu tổng cao nhất (**1.49 lần** mức chung); nổi bật ở **Thực phẩm chế biến** (`2.22x`) và **Hàng tươi sống** (`1.92x`), mua đa dạng nhiều nhóm hàng. | Áp dụng **thẻ thành viên VIP**, **chiết khấu bậc thang** theo hóa đơn lớn và **miễn phí giao hàng tận nơi**. |

*(Riêng 108 khách hàng được gắn cờ `Ngoại lai IQR` được đánh dấu ưu tiên để chuyên viên chăm sóc riêng / ký hợp đồng chiết khấu riêng).*

---

## CHƯƠNG 5. THIẾT KẾ ỨNG DỤNG WEB (STREAMLIT)

Ứng dụng được xây dựng bằng **Python + Streamlit + SQLite** (`app.py`, `experience.py`, `storage.py`, `pipeline.py`, `retrain.py`) đáp ứng đầy đủ 4 yêu cầu của bài toán:
1. **Thu thập dữ liệu mới (Tab 4):**
   - Form nhập liệu gọn nhẹ 6 nhóm chi tiêu (kèm 3 nút điền mẫu nhanh 1-click phục vụ demo) và chức năng upload file CSV hàng loạt (có file mẫu tải về).
   - Lưu trữ an toàn vào SQLite (`data/customers.sqlite3`), kiểm tra chống nhập lặp cùng một lô dữ liệu bằng mã băm SHA-256.
   - **Hiển thị kết quả tức thì:** Ngay khi lưu khách hàng mới, hệ thống tính khoảng cách tới tâm cụm của mô hình hiện tại và in ngay ra màn hình khách hàng đó thuộc phân khúc nào, có phải ngoại lai không và gợi ý khuyến mãi tương ứng.
2. **Tiền xử lý dữ liệu (Tab 1 & Tab 2):**
   - Tự động kiểm tra dữ liệu đầu vào (`validate`), vẽ biểu đồ Boxplot phát hiện ngoại lai, hỗ trợ chọn trực tiếp phương pháp xử lý ngoại lai (IQR) và bộ chuẩn hóa (`StandardScaler` / `MinMaxScaler`).
3. **Huấn luyện lại định kỳ (Thanh bên & `retrain.py`):**
   - Hỗ trợ **3 chế độ**: bấm nút huấn luyện thủ công trên giao diện; tự động huấn luyện lại khi mở ứng dụng nếu có **$\ge N$ khách hàng mới** (mặc định 10) hoặc **quá $X$ ngày** (mặc định 7); hoặc chạy tự động ngoài giao diện qua Windows Task Scheduler với lệnh `python retrain.py --if-due`.
   - Mỗi lần huấn luyện lưu thành một phiên độc lập trong `artifacts/<run_id>/` (kèm `model.joblib`, `metadata.json`, bảng kết quả CSV) để không ghi đè lịch sử.
4. **In kết quả ra màn hình (Tab 2 & Tab 3):**
   - Hiển thị trực quan biểu đồ Elbow, Silhouette, Dendrogram, k-distance, bảng so sánh 3 thuật toán, bản đồ phân cụm PCA 2D, biểu đồ hồ sơ từng khách hàng và cho phép tải danh sách phân khúc CSV hoặc phiếu đề xuất khuyến mãi JSON.

---

## CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 6.1. Kết luận
Nhóm đã hoàn thành trọn vẹn quy trình khai phá dữ liệu từ khảo sát dữ liệu, tiền xử lý (loại ngoại lai IQR, chuẩn hóa StandardScaler/MinMaxScaler), thực nghiệm đối chiếu 3 thuật toán phân cụm (K-Means, Hierarchical, DBSCAN) đến xây dựng ứng dụng web hoàn chỉnh bằng Streamlit. Kết quả thực nghiệm cho thấy quy trình `IQR + StandardScaler` giúp cả 3 thuật toán tạo ra các phân khúc hợp lệ và có ý nghĩa kinh doanh rõ ràng cho bài toán gợi ý khuyến mãi siêu thị.

### 6.2. Hạn chế và hướng phát triển
- **Hạn chế:** Tập dữ liệu gồm 440 khách hàng và chỉ có tổng chi tiêu năm (thiếu thông tin thời gian, mã định danh khách hàng, biên lợi nhuận và lịch sử phản hồi khuyến mãi thực tế). Chỉ số Silhouette ở mức trung bình (~0.27) cho thấy dữ liệu chi tiêu thực tế phân bố liên tục và có sự giao thoa nhất định giữa các nhóm.
- **Hướng phát triển:** Thu thập dữ liệu giao dịch theo tháng có mã khách hàng để phân tích biến động hành vi (RFM); thực hiện thử nghiệm A/B test có nhóm đối chứng để đo lường tỷ lệ phản hồi và lợi nhuận thực tế của từng gói khuyến mãi.

---

## PHỤ LỤC: PHÂN CÔNG CÔNG VIỆC NHÓM 4 NGƯỜI & KỊCH BẢN DEMO

### A. Bảng phân công chi tiết cho 4 thành viên

| Thành viên | Phần việc trong Code & Notebook | Phần viết Báo cáo & Slide | Phần trình bày / Demo khi bảo vệ |
|---|---|---|---|
| **Thành viên 1** *(Dữ liệu & Tiền xử lý)* | Phụ trách `validate()`, khám phá dữ liệu (EDA), biểu đồ Boxplot, xử lý ngoại lai IQR và chuẩn hóa (`StandardScaler` vs `MinMaxScaler`). | Viết **Chương 1 & Chương 2**; chuẩn bị slide giới thiệu bộ dữ liệu và bảng thống kê trước/sau khi loại ngoại lai. | Trình bày lý do chọn đề tài, ý nghĩa 8 thuộc tính; mở **Tab 1** giải thích Boxplot ngoại lai và lý do bắt buộc chuẩn hóa dữ liệu. |
| **Thành viên 2** *(K-Means & Đánh giá)* | Phụ trách thuật toán K-Means, hàm `elbow_k()`, tính Silhouette Score, Davies–Bouldin, Calinski–Harabasz. | Viết **Mục 3.1 (K-Means), Mục 3.2** và phân tích kết quả Elbow + Silhouette ở **Chương 4**. | Mở **Tab 2** giải thích biểu đồ Elbow ($k=4$), Silhouette Score, và lý do chọn $k \ge 3$ thay vì $k=2$. |
| **Thành viên 3** *(Hierarchical, DBSCAN & Khuyến mãi)* | Phụ trách Hierarchical (Dendrogram), DBSCAN (k-distance, lưới `eps`/`min_samples`), so sánh 3 thuật toán và quy tắc gợi ý khuyến mãi (`describe_segment()`). | Viết **Mục 3.1 (Hierarchical, DBSCAN), Mục 4.2, Mục 4.3** (bảng hồ sơ 3 phân khúc) và **Chương 6**. | Giải thích **Dendrogram, k-distance, bảng so sánh 3 thuật toán** ở Tab 2; mở **Tab 3** phân tích 3 nhóm khách hàng và gợi ý khuyến mãi. |
| **Thành viên 4** *(Ứng dụng Streamlit & Hệ thống)* | Phụ trách giao diện `app.py`, `experience.py`, lưu trữ SQLite (`storage.py`), dự đoán khách mới (`predict_segments()`) và lịch huấn luyện định kỳ (`retrain.py`). | Viết **Chương 5**; chụp ảnh màn hình giao diện đưa vào báo cáo; phụ trách chạy thử nghiệm `pytest` và đóng gói Colab. | Trực tiếp **demo Tab 4**: bấm nút điền mẫu thử nhanh, lưu khách hàng mới để **in ngay phân khúc ra màn hình**, chỉ vào chính sách huấn luyện định kỳ ở thanh bên và bấm **Huấn luyện lại**. |

### B. Bộ câu hỏi vấn đáp thường gặp (Cả 4 bạn cần nắm)
1. **Tại sao phải chuẩn hóa dữ liệu trước khi phân cụm?**  
   $\rightarrow$ Vì K-Means, Hierarchical và DBSCAN đều dùng khoảng cách Euclid; nếu không chuẩn hóa, cột có đơn vị lớn như `Fresh` (hàng chục nghìn) sẽ lấn át hoàn toàn cột nhỏ như `Delicassen`.
2. **Tại sao lại loại bỏ ngoại lai khi huấn luyện mà vẫn giữ trong kết quả?**  
   $\rightarrow$ Loại ngoại lai (108 dòng theo IQR) giúp tâm cụm không bị kéo lệch và giúp cả 3 thuật toán hội tụ đẹp; nhưng trong kinh doanh siêu thị, ngoại lai thường là khách VIP chi tiêu rất lớn nên cần gán nhãn chăm sóc riêng chứ không xóa bỏ khách hàng.
3. **Tại sao không chọn $k=2$ dù Silhouette ở $k=2$ thường cao?**  
   $\rightarrow$ $k=2$ chỉ tách lại 2 nhóm kênh bán hàng (`Channel 1` HoReCa và `Channel 2` Bán lẻ) vốn đã biết sẵn trong dữ liệu. Chọn $k \ge 3$ giúp tách sâu hơn (ví dụ tách nhóm mua sắm tiết kiệm, nhóm tạp hóa/gia dụng và nhóm khách lớn đa ngành) để thiết kế khuyến mãi sát thực tế hơn.
4. **Khi có khách hàng mới nhập vào giao diện, làm sao gán được nhóm cho Hierarchical và DBSCAN khi hai thuật toán này không có hàm `predict()`?**  
   $\rightarrow$ Ứng dụng gán tạm thời dựa trên khoảng cách gần nhất tới tâm của các cụm đã học (centroid), đồng thời lưu dữ liệu mới vào SQLite để đến chu kỳ huấn luyện lại thì cập nhật toàn bộ mô hình.
