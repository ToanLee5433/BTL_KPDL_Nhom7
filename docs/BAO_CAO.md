# BÁO CÁO BÀI TẬP LỚN MÔN KHAI PHÁ DỮ LIỆU

**Đề tài:** Phân khúc khách hàng siêu thị để gợi ý khuyến mãi (Bài toán Phân cụm)  
**Nhóm thực hiện:** Nhóm 7 — 5 thành viên *(điền họ tên và mã sinh viên trước khi nộp)*

| STT | Họ và tên | Mã sinh viên | Phần phụ trách chính |
|---|---|---|---|
| 1 | *(Thành viên 1)* | | Tìm hiểu bài toán, thu thập & khám phá dữ liệu (EDA) — **Chương 1** |
| 2 | *(Thành viên 2)* | | Tiền xử lý: phát hiện ngoại lai (IQR) & chuẩn hóa dữ liệu — **Chương 2** |
| 3 | *(Thành viên 3)* | | Thuật toán K-Means & đánh giá bằng Elbow, Silhouette — **Chương 3 (3.1, 3.3) & 4.1** |
| 4 | *(Thành viên 4)* | | Thuật toán Hierarchical, DBSCAN & phân tích khuyến mãi — **Chương 3 (3.2) & 4.2, 4.3** |
| 5 | *(Thành viên 5 - Nhóm trưởng)* | | Xây dựng Google Colab, giao diện Streamlit, SQLite & huấn luyện định kỳ — **Chương 5 & Tổng hợp** |

---

## CHƯƠNG 1. GIỚI THIỆU BÀI TOÁN VÀ KHÁM PHÁ DỮ LIỆU (EDA)
*(Phụ trách viết & thuyết trình: Thành viên 1)*

### 1.1. Lý do chọn đề tài và mục tiêu bài toán
Trong kinh doanh siêu thị và bán lẻ, mỗi khách hàng có một giỏ hàng và mức chi tiêu rất khác nhau. Có người chủ yếu mua thực phẩm tươi sống hàng ngày, có cửa hàng tạp hóa nhập số lượng lớn chất tẩy rửa và hàng khô, lại có những khách hàng mua sắm tiết kiệm với tần suất thấp. Nếu siêu thị tung ra một chương trình khuyến mãi giống hệt nhau cho toàn bộ tệp khách hàng, ngân sách ưu đãi sẽ bị dàn trải mà hiệu quả chuyển đổi lại không cao.

Xuất phát từ thực tế đó, nhóm chọn giải quyết bài toán **Phân cụm (Clustering)** — một trong ba bài toán lớn của môn Khai phá dữ liệu — với mục tiêu chia tập khách hàng thành các nhóm có hành vi chi tiêu tương đồng. Từ đặc điểm riêng của từng nhóm, hệ thống sẽ gợi ý chương trình khuyến mãi sát với nhu cầu thực tế của họ, đồng thời tích hợp vào một ứng dụng web nhỏ để có thể nhập thêm khách hàng mới và cập nhật mô hình theo định kỳ.

### 1.2. Giới thiệu bộ dữ liệu Wholesale Customers (UCI)
Nhóm sử dụng bộ dữ liệu **Wholesale Customers Data Set** lấy từ kho lưu trữ nghiên cứu uy tín UCI Machine Learning Repository (tác giả Margarida Cardoso, 2013, DOI: `10.24432/C5030X`). Bộ dữ liệu gồm **440 bản ghi** (tương ứng 440 khách hàng) và **8 thuộc tính**, không có giá trị bị thiếu (`Missing = 0`) và không có dòng trùng lặp hoàn toàn (`Duplicates = 0`).

Ý nghĩa cụ thể của 8 cột trong dữ liệu như sau:
- **6 cột chi tiêu hàng năm (đơn vị tiền tệ gốc):**
  1. `Fresh`: Chi tiêu cho nhóm hàng tươi sống (rau củ, thịt, cá...).
  2. `Milk`: Chi tiêu cho sữa và các chế phẩm từ sữa.
  3. `Grocery`: Chi tiêu cho hàng tạp hóa, thực phẩm khô.
  4. `Frozen`: Chi tiêu cho thực phẩm đông lạnh.
  5. `Detergents_Paper`: Chi tiêu cho chất tẩy rửa và sản phẩm giấy gia dụng.
  6. `Delicassen`: Chi tiêu cho thực phẩm chế biến sẵn.
- **2 cột thông tin phân loại (ngữ cảnh):**
  7. `Channel`: Kênh mua hàng gồm 2 giá trị — `1` là nhóm Khách sạn/Nhà hàng/Café (HoReCa, chiếm **298** khách hàng) và `2` là nhóm Bán lẻ/Siêu thị (chiếm **142** khách hàng).
  8. `Region`: Khu vực địa lý gồm 3 giá trị — `1` là Lisbon (**77** khách), `2` là Oporto (**47** khách) và `3` là khu vực khác (**316** khách).

Một điểm quan trọng nhóm đã thống nhất ngay từ bước thiết kế là **chỉ đưa 6 cột chi tiêu vào tính khoảng cách phân cụm**. Hai cột `Channel` và `Region` mang bản chất là mã định danh danh mục (ví dụ khu vực 3 không có nghĩa là "lớn gấp 3 lần" khu vực 1), nên nếu đưa vào tính khoảng cách Euclid sẽ làm sai lệch hoàn toàn kết quả. Thay vào đó, nhóm giữ 2 cột này để đối chiếu tỷ lệ kênh và khu vực của từng cụm sau khi đã phân cụm xong.

### 1.3. Khám phá dữ liệu (EDA)
Khi chạy thống kê mô tả trên 440 khách hàng, nhóm nhận thấy sự chênh lệch rất lớn cả về quy mô lẫn độ phân tán giữa 6 nhóm mặt hàng:

| Nhóm hàng | Nhỏ nhất (Min) | Trung vị (50%) | Trung bình (Mean) | Lớn nhất (Max) | Độ lệch chuẩn (Std) | Độ lệch (Skewness) |
|---|---:|---:|---:|---:|---:|---:|
| `Fresh` | 3.0 | 8,504.0 | 12,000.3 | 112,151.0 | 12,647.3 | 2.56 |
| `Milk` | 55.0 | 3,627.0 | 5,796.3 | 73,498.0 | 7,380.4 | 4.05 |
| `Grocery` | 3.0 | 4,755.5 | 7,951.3 | 92,780.0 | 9,503.2 | 3.59 |
| `Frozen` | 25.0 | 1,526.0 | 3,071.9 | 60,869.0 | 4,854.7 | 5.91 |
| `Detergents_Paper` | 3.0 | 816.5 | 2,881.5 | 40,827.0 | 4,767.9 | 3.63 |
| `Delicassen` | 3.0 | 965.5 | 1,524.9 | 47,943.0 | 2,820.1 | 11.15 |

Từ bảng trên, có thể rút ra hai nhận xét trực tiếp quyết định bước tiền xử lý ở Chương 2:
1. **Dữ liệu bị lệch phải rất mạnh:** Ở cả 6 cột, giá trị trung bình (`Mean`) đều cao hơn hẳn trung vị (`50%`), đặc biệt ở `Detergents_Paper` (trung bình 2,881.5 trong khi trung vị chỉ 816.5) và `Delicassen` (hệ số lệch lên tới `11.15`, mức chi cao nhất đạt 47,943 trong khi 75% khách hàng chỉ chi dưới 1,820). Điều này chứng tỏ có một nhóm nhỏ khách hàng chi tiêu đột biến kéo đuôi phân phối lệch hẳn sang phải.
2. **Thang đo không đồng nhất:** Chi tiêu `Fresh` trung bình khoảng 12,000, gấp gần 8 lần so với `Delicassen` (1,525). Nếu giữ nguyên số liệu thô đưa vào phân cụm, nhóm hàng `Fresh` và `Grocery` sẽ áp đảo hoàn toàn các nhóm hàng còn lại.

---

## CHƯƠNG 2. TIỀN XỬ LÝ DỮ LIỆU
*(Phụ trách viết & thuyết trình: Thành viên 2)*

### 2.1. Kiểm tra tính hợp lệ của dữ liệu đầu vào
Để ứng dụng thực tế không bị lỗi khi người dùng nhập liệu mới hoặc tải file CSV lên, toàn bộ dữ liệu trước khi lưu trữ và phân tích đều phải đi qua bước kiểm tra ràng buộc (`validate`):
- Đảm bảo đủ 8 cột đúng tên và không có cột bị lặp.
- Loại bỏ hoặc chặn các bản ghi chứa ô trống (`NaN`), ký tự chữ, giá trị vô hạn (`Inf`) hoặc số tiền chi tiêu âm (`< 0`).
- Chặn bản ghi mà cả 6 nhóm hàng đều bằng `0` (khách hàng không phát sinh chi tiêu).
- Đảm bảo `Channel` chỉ nằm trong tập `{1, 2}` và `Region` chỉ nằm trong tập `{1, 2, 3}`.

### 2.2. Phát hiện và xử lý ngoại lai bằng quy tắc IQR
Quan sát biểu đồ **Boxplot** ở Chương 1 cho thấy có nhiều điểm nằm vọt hẳn lên phía trên râu trên của hộp. Đây là các khách hàng ngoại lai (outliers). Với các thuật toán dựa trên khoảng cách và tâm cụm như K-Means hay liên kết Ward, chỉ cần một vài điểm nằm quá xa sẽ kéo lệch vị trí tâm cụm hoặc khiến thuật toán DBSCAN coi phần còn lại là một cụm duy nhất.

Nhóm áp dụng quy tắc **IQR (Interquartile Range)** chuẩn thống kê trên từng cột chi tiêu:
- Tính khoảng tứ phân vị: $IQR = Q_3 - Q_1$ (trong đó $Q_1$ là phân vị 25%, $Q_3$ là phân vị 75%).
- Các điểm nằm ngoài đoạn $[Q_1 - 1.5 \times IQR,\; Q_3 + 1.5 \times IQR]$ ở bất kỳ cột chi tiêu nào sẽ bị gắn cờ là ngoại lai.

Kết quả lọc trên 440 khách hàng ban đầu: hệ thống phát hiện **108 khách hàng ngoại lai**, giữ lại **332 khách hàng chuẩn** để huấn luyện mô hình.

**Điểm xử lý khéo léo của nhóm:** Trong bài toán bán lẻ, những điểm ngoại lai có giá trị chi tiêu rất lớn thực chất lại là những **khách hàng lớn / khách hàng VIP**, không phải lỗi nhập liệu. Do đó, nhóm **chỉ tách 108 điểm này ra khỏi quá trình tìm tâm cụm và huấn luyện mô hình** để cấu trúc cụm của số đông không bị méo mó; sau khi mô hình học xong, 108 khách hàng này vẫn được gán vào cụm gần nhất kèm nhãn **`Ngoại lai (IQR)`** và được đề xuất chính sách chăm sóc riêng ở bước gợi ý khuyến mãi. Nhờ vậy, hệ thống vừa đảm bảo chất lượng phân cụm, vừa không bỏ sót bất kỳ khách hàng nào.

### 2.3. Chuẩn hóa dữ liệu về cùng thang đo
Vì cả K-Means, Hierarchical và DBSCAN đều sử dụng khoảng cách Euclid $d(x, y) = \sqrt{\sum_{j=1}^{6}(x_j - y_j)^2}$ để đo độ tương đồng giữa hai khách hàng, việc đưa 6 cột chi tiêu về cùng một thang đo là **bắt buộc**. Nhóm cài đặt và thử nghiệm đối chiếu cả hai phương pháp chuẩn hóa phổ biến trong thư viện `scikit-learn`:
1. **StandardScaler (Chuẩn hóa Z-score):** Biến đổi từng cột về trung bình $\mu = 0$ và độ lệch chuẩn $\sigma = 1$ theo công thức:
   $$z = \frac{x - \mu}{\sigma}$$
2. **MinMaxScaler:** Co giãn toàn bộ giá trị của từng cột về đoạn $[0, 1]$ theo công thức:
   $$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

Lưu ý kỹ thuật: bộ chuẩn hóa chỉ được khớp (`fit`) trên tập 332 khách hàng chuẩn (sau khi đã loại ngoại lai) để giá trị trung bình, độ lệch chuẩn hay `max` không bị các điểm ngoại lai làm sai lệch, sau đó mới biến đổi (`transform`) cho toàn bộ dữ liệu. Kết quả thực nghiệm so sánh giữa hai bộ chuẩn hóa được trình bày ở Chương 4 cho thấy `StandardScaler` cho kết quả ổn định hơn và giúp cả 3 thuật toán cùng đạt tiêu chí phân cụm.

---

## CHƯƠNG 3. CÁC THUẬT TOÁN PHÂN CỤM VÀ PHƯƠNG PHÁP ĐÁNH GIÁ
*(Phụ trách viết & thuyết trình: Thành viên 3 & Thành viên 4)*

### 3.1. Thuật toán K-Means *(Thành viên 3)*
K-Means là thuật toán phân cụm phân hoạch (partitional clustering) kinh điển. Mục tiêu của K-Means là chia $N$ điểm dữ liệu thành $k$ cụm sao cho tổng bình phương khoảng cách từ mỗi điểm tới tâm cụm (centroid) của nó là nhỏ nhất. Hàm mục tiêu (Inertia / WCSS — Within-Cluster Sum of Squares) được viết là:
$$J = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2$$
trong đó $\mu_i$ là tâm của cụm $C_i$. Thuật toán lặp qua 2 bước: gán mỗi điểm vào cụm có tâm gần nhất, sau đó cập nhật lại tọa độ tâm cụm bằng trung bình cộng các điểm trong cụm cho đến khi hội tụ.

Trong bài toán này, nhóm cấu hình `n_init = 20` (chạy khởi tạo ngẫu nhiên 20 lần và lấy kết quả có Inertia nhỏ nhất để tránh rơi vào cực tiểu cục bộ) và `random_state = 42` để kết quả tái lập được. K-Means chạy rất nhanh và có sẵn hàm `predict()` cho dữ liệu mới, nhưng đòi hỏi người dùng phải chọn trước số cụm $k$ và hoạt động tốt nhất khi các cụm có dạng hình cầu.

### 3.2. Thuật toán Hierarchical Clustering & DBSCAN *(Thành viên 4)*
1. **Phân cụm phân cấp — Agglomerative Hierarchical Clustering (liên kết Ward):**
   Khác với K-Means phải chọn $k$ ngay từ đầu, Agglomerative tiếp cận theo hướng từ dưới lên (bottom-up): ban đầu xem mỗi khách hàng là một cụm riêng lẻ (332 cụm), sau đó ở mỗi bước sẽ gộp hai cụm gần nhau nhất lại thành một cho đến khi tất cả gom thành một cụm duy nhất. Quá trình gộp được biểu diễn trực quan bằng biểu đồ cây **Dendrogram**.
   Nhóm sử dụng tiêu chuẩn liên kết **Ward (`linkage='ward'`)**, tức là ở mỗi bước sẽ chọn gộp hai cụm nào làm **tổng phương sai trong cụm tăng lên ít nhất**. Nhược điểm của phương pháp này là không có hàm `predict()` gốc cho khách hàng mới phát sinh.

2. **Phân cụm theo mật độ — DBSCAN:**
   DBSCAN (Density-Based Spatial Clustering of Applications with Noise) không chia cụm dựa vào tâm mà dựa vào mật độ các điểm lân cận thông qua hai tham số: bán kính lân cận `eps` và số điểm tối thiểu `min_samples` để tạo thành một vùng đặc (core point). Những điểm nằm ở vùng thưa, không thuộc vùng ảnh hưởng của bất kỳ điểm lõi nào sẽ được gán nhãn nhiễu (`-1`).
   Ưu điểm lớn của DBSCAN là không cần biết trước số cụm $k$ và tự phát hiện được nhiễu; tuy nhiên thuật toán khá nhạy với bộ đôi `(eps, min_samples)`, đặc biệt trong không gian 6 chiều. Để chọn khoảng `eps` hợp lý, nhóm vẽ biểu đồ **k-distance** (sắp xếp tăng dần khoảng cách tới láng giềng thứ $k$ của mỗi điểm) và khảo sát lưới `eps` quanh vùng đường cong bắt đầu gập lên.

### 3.3. Phương pháp đánh giá chất lượng phân cụm *(Thành viên 3)*
Vì phân cụm là học không giám sát (không có nhãn đúng để tính Accuracy hay F1-score), nhóm sử dụng các chỉ số đánh giá nội tại kết hợp:
1. **Phương pháp Khuỷu tay (Elbow Method):** Áp dụng cho K-Means bằng cách vẽ đồ thị giá trị Inertia (WCSS) khi tăng $k$ từ 2 đến 8. Khi $k$ tăng, Inertia luôn giảm; điểm "khuỷu tay" là nơi tốc độ giảm bắt đầu chững lại rõ rệt — cân bằng giữa độ gọn của cụm và số lượng cụm không quá nhiều.
2. **Chỉ số Silhouette Score:** Đo lường đồng thời độ gắn kết bên trong cụm và độ tách biệt với cụm gần nhất:
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
   với $a(i)$ là khoảng cách trung bình từ điểm $i$ tới các điểm cùng cụm và $b(i)$ là khoảng cách trung bình từ $i$ tới cụm gần nhất tiếp theo. Giá trị nằm trong $[-1, 1]$, càng gần $1$ chứng tỏ điểm đó được phân cụm càng chuẩn.
3. **Các chỉ số bổ trợ:** **Davies–Bouldin Index** (tỷ số giữa độ phân tán trong cụm và khoảng cách giữa các cụm — **càng nhỏ càng tốt**), **Calinski–Harabasz Index** (**càng lớn càng tốt**), và **tỷ lệ nhiễu** đối với DBSCAN.

**Quy tắc chọn mô hình hợp lệ:** Trước khi chạy thực nghiệm, nhóm đặt ra tiêu chí rõ ràng: mô hình được chọn để triển khai khuyến mãi phải có **từ 3 cụm trở lên ($k \ge 3$)**, mỗi cụm có ít nhất **5 khách hàng**, và tỷ lệ nhiễu (với DBSCAN) **không vượt quá 30%**. Lý do loại $k=2$ là vì qua kiểm tra, việc chia 2 cụm gần như chỉ tách lại hai kênh `Channel 1` (HoReCa) và `Channel 2` (Bán lẻ) vốn đã biết sẵn, chưa đủ chi tiết để thiết kế các chương trình khuyến mãi riêng biệt.

---

## CHƯƠNG 4. KẾT QUẢ THỰC NGHIỆM VÀ GỢI Ý KHUYẾN MÃI
*(Phụ trách phân tích: Thành viên 3 mục 4.1; Thành viên 4 mục 4.2 & 4.3)*

*(Toàn bộ số liệu dưới đây được ghi nhận từ lần chạy chuẩn hóa trên 440 khách hàng, loại 108 ngoại lai bằng IQR, còn 332 khách hàng đưa vào phân cụm).*

### 4.1. Kết quả khảo sát K-Means (Elbow & Silhouette) và so sánh Scaler *(Thành viên 3)*
Trước hết, nhóm chạy đối chiếu hai bộ chuẩn hóa `StandardScaler` và `MinMaxScaler` trên cùng một tập 332 khách hàng sạch:

| Bộ chuẩn hóa | Thuật toán đại diện | Tham số | Số cụm | Tỷ lệ nhiễu | Silhouette | Davies–Bouldin | Đạt tiêu chí ($\ge 3$ cụm, nhiễu $\le 30\%$) |
|---|---|---|---:|---:|---:|---:|---|
| **StandardScaler** | K-Means | `k = 4` | 4 | 0.0% | 0.271 | **1.281** | Đạt |
| **StandardScaler** | Hierarchical (Ward) | `k = 3` | 3 | 0.0% | **0.276** | 1.448 | Đạt |
| **StandardScaler** | DBSCAN | `eps = 1.1, min_samples = 3` | 3 | 19.9% | 0.229 | 0.807 | Đạt |
| MinMaxScaler | K-Means | `k = 4` | 4 | 0.0% | 0.270 | 1.318 | Đạt |
| MinMaxScaler | Hierarchical (Ward) | `k = 4` | 4 | 0.0% | 0.259 | 1.396 | Đạt |
| MinMaxScaler | DBSCAN | `eps = 0.25, min_samples = 5` | 2 | 28.3% | 0.428 | 0.615 | Không đạt (chỉ có 2 cụm) |

Nhìn vào bảng đối chiếu, `StandardScaler` cho thấy sự vượt trội về tính ổn định khi **cả 3 thuật toán đều đạt ngưỡng phân cụm hợp lệ ($\ge 3$ cụm, nhiễu $\le 30\%$)**, trong khi `MinMaxScaler` làm thuật toán DBSCAN bị gom về 2 cụm (chưa đủ sâu cho bài toán khuyến mãi). Vì thế nhóm chọn `StandardScaler` làm phương án tiền xử lý chính.

Đi sâu vào kết quả khảo sát K-Means và Hierarchical theo số cụm $k = 2 \dots 8$ trên `StandardScaler`:

| $k$ | K-Means Inertia | K-Means Silhouette | K-Means DB | Hierarchical Silhouette | Hierarchical DB |
|---:|---:|---:|---:|---:|---:|
| 2 | 1,303.11 | 0.355 | 1.226 | 0.281 | 1.453 |
| 3 | 1,077.04 | 0.259 | 1.381 | **0.276** | 1.448 |
| 4 | **920.60** | **0.271** | **1.281** | 0.184 | 1.467 |
| 5 | 816.07 | 0.256 | 1.272 | 0.180 | 1.498 |
| 6 | 742.14 | 0.245 | 1.230 | 0.203 | 1.365 |

- **Theo phương pháp Khuỷu tay (Elbow Method):** Đường Inertia giảm mạnh từ $k=2$ (`1,303.11`) xuống $k=3$ (`1,077.04`) và $k=4$ (`920.60`), sau đó mức giảm chậm hẳn lại từ $k=5$ trở đi. Thuật toán tìm điểm khuỷu tự động trên đường cong xác định điểm khuỷu nằm tại **$k = 4$**.
- **Theo chỉ số Silhouette:** Ở K-Means, nếu bỏ qua $k=2$ (do chỉ lặp lại biến `Channel`), giá trị Silhouette đạt đỉnh cục bộ tại **$k = 4$ (`0.271`)**, đồng thời Davies–Bouldin cũng giảm xuống thấp (`1.281`). Như vậy cả Elbow Method và Silhouette đều đồng thuận chọn **$k = 4$** cho K-Means.

### 4.2. So sánh 3 thuật toán trên cùng tập dữ liệu *(Thành viên 4)*
Đặt cấu hình tốt nhất đạt tiêu chí của cả 3 thuật toán cạnh nhau trên `StandardScaler`:
1. **Hierarchical Clustering (Ward, `k = 3`):** Đạt Silhouette cao nhất trong các mô hình hợp lệ (**`0.276`**), thời gian chạy cực nhanh (`0.002s`), bao phủ **100%** dữ liệu khách hàng sạch (cụm nhỏ nhất có 41 khách). Đây là thuật toán được hệ thống lựa chọn làm mô hình chính của phiên.
2. **K-Means (`k = 4`):** Đạt Silhouette gần tương đương (**`0.271`**), nhưng có chỉ số Davies–Bouldin tốt hơn Hierarchical (`1.281` so với `1.448`) và Calinski–Harabasz cao hơn (`127.24` so với `108.39`). Đặc biệt K-Means có sẵn công thức tâm cụm rất thuận tiện để gán khách hàng mới.
3. **DBSCAN (`eps = 1.1, min_samples = 3`):** Tách được **3 cụm** với tỷ lệ nhiễu là **19.9%** ($\le 30\%$) và Silhouette trên các điểm không nhiễu đạt **`0.229`**. Một số cấu hình khác như `eps = 1.2, min_samples = 3` cho Silhouette `0.395` với `16.3%` nhiễu nhưng cụm nhỏ nhất dưới 5 điểm nên không đạt tiêu chí kích thước cụm. Điểm yếu của DBSCAN trên dữ liệu chi tiêu liên tục này là bỏ lại gần 20% khách hàng dưới dạng nhiễu chưa gán nhóm, khiến bộ phận kinh doanh khó triển khai khuyến mãi đồng bộ.

*Nhận xét chung:* Chỉ số Silhouette quanh mức `0.27` cho thấy dữ liệu chi tiêu thực tế là một khối phân bố liên tục và có sự giao thoa nhất định giữa các nhóm khách hàng (không tách rời hoàn toàn như dữ liệu lý thuyết), nhưng hoàn toàn đủ cơ sở để phân tầng hành vi mua sắm.

### 4.3. Phân tích 3 phân khúc khách hàng và Gợi ý khuyến mãi *(Thành viên 4)*
Sau khi gán lại 108 khách hàng ngoại lai vào cụm gần nhất để theo dõi đầy đủ 440 khách hàng, mô hình chính cho ra **3 phân khúc khách hàng** với hồ sơ chi tiêu (tính trên trung vị tiền gốc và tỷ lệ chi tiêu trung bình so với mặt bằng chung) rất rõ nét:

| Phân khúc | Số lượng KH (gồm ngoại lai) | Đặc trưng chi tiêu thực tế | Tỷ lệ kênh bán lẻ (`Channel=2`) | Gợi ý chương trình khuyến mãi |
|---|---|---|---|---|
| **Nhóm 0 — Khách mua sắm tiết kiệm / quy mô nhỏ** | **207 KH** *(có 25 ngoại lai)* | Tổng chi tiêu chỉ bằng **0.75 lần** mức chung. Trung vị `Fresh`: 8,305; `Frozen`: 1,662; `Grocery`: 2,175. Chi tiêu nổi trội tương đối ở **Hàng đông lạnh (`1.20x`)** và **Hàng tươi sống (`1.07x`)**, mua rất ít chất tẩy rửa (`0.26x`). | 2% (chủ yếu khách hàng nhỏ / bếp ăn nhỏ) | **Kích cầu tần suất mua:** Tặng **Voucher giảm giá cho đơn hàng kế tiếp** và thiết kế **Combo tiết kiệm Hàng đông lạnh + Hàng tươi sống** giá ưu đãi để khuyến khích khách quay lại mua thường xuyên hơn. |
| **Nhóm 1 — Nhóm ưu tiên Tạp hóa & Gia dụng** | **146 KH** *(có 37 ngoại lai)* | Tổng chi tiêu bằng **1.24 lần** mức chung. Trung vị `Grocery`: 10,518; `Milk`: 7,027; `Detergents_Paper`: 4,239. Chi tiêu vượt trội ở **Chất tẩy rửa & giấy (`2.28x`)**, **Tạp hóa (`1.86x`)** và **Sữa (`1.78x`)**. | **70%** (chủ yếu cửa hàng bán lẻ / siêu thị mini) | **Khuyến mãi theo số lượng:** Áp dụng chương trình **mua nhiều giảm giá (mua 2 tặng 1 / chiết khấu 10% theo thùng)** cho nhóm hàng **Chất tẩy rửa & giấy, Tạp hóa** để giữ chân khách hàng nhập hàng định kỳ. |
| **Nhóm 2 — Khách hàng lớn đa ngành (VIP)** | **87 KH** *(có 46 ngoại lai)* | Tổng chi tiêu cao nhất, bằng **1.49 lần** mức chung. Trung vị `Fresh`: 18,567; `Delicassen`: 2,233; `Grocery`: 6,235. Nổi bật ở **Thực phẩm chế biến (`2.22x`)** và **Hàng tươi sống (`1.92x`)**, đồng thời chi cao đều ở nhiều nhóm hàng. | 37% | **Chính sách VIP giữ chân:** Áp dụng **thẻ thành viên VIP**, **chiết khấu bậc thang theo giá trị hóa đơn lớn** và **miễn phí giao hàng tận nơi**. |

*(Ghi chú: Đối với 108 khách hàng được đánh dấu **`Ngoại lai (IQR)`** do có mức chi tiêu đột biến cao ở một số mặt hàng, hệ thống tự động gắn thêm khuyến nghị **chăm sóc riêng / đàm phán chiết khấu theo hợp đồng** thay vì chỉ áp voucher phổ thông).*

---

## CHƯƠNG 5. XÂY DỰNG ỨNG DỤNG DEMO TRÊN STREAMLIT VÀ GOOGLE COLAB
*(Phụ trách xây dựng & demo trực tiếp: Thành viên 5 - Nhóm trưởng)*

Để hoàn thiện yêu cầu thứ 4 của bài tập lớn (thu thập dữ liệu mới, tiền xử lý, huấn luyện định kỳ và hiển thị kết quả cho người dùng), nhóm đã xây dựng trọn bộ hệ thống chạy đồng bộ trên cả **Google Colab** (`Wholesale_Customers_Clustering.ipynb`) và **ứng dụng web Streamlit + SQLite** (`app.py`, `pipeline.py`, `storage.py`, `experience.py`, `retrain.py`):

1. **Google Colab (`Wholesale_Customers_Clustering.ipynb`):** Tự động đồng bộ mã nguồn từ GitHub, tải dữ liệu UCI, chạy toàn bộ quy trình EDA, Boxplot ngoại lai, Elbow, Silhouette, Dendrogram, k-distance, trực quan hóa PCA 2D và cho phép khởi chạy thẳng giao diện Streamlit ngay trên Colab.
2. **Giao diện Streamlit 4 tab gọn nhẹ:**
   - **Tab 1 — Dữ liệu & Khám phá:** Hiển thị 440 khách hàng, thống kê mô tả và biểu đồ Boxplot ngoại lai.
   - **Tab 2 — Tiền xử lý & So sánh 3 thuật toán:** Cho phép đổi trực tiếp cách lọc ngoại lai (`IQR` / `IQR Log` / `Không lọc`) và bộ chuẩn hóa (`StandardScaler` / `MinMaxScaler`) để quan sát tức thì sự thay đổi của biểu đồ Elbow, Silhouette, Dendrogram, k-distance và bảng so sánh 3 thuật toán.
   - **Tab 3 — Kết quả phân khúc & Khuyến mãi:** Hiển thị bản đồ cụm PCA 2D, bảng chi tiêu tương đối, tra cứu từng khách hàng để xuất phiếu gợi ý khuyến mãi (JSON) và tải danh sách phân khúc (CSV).
   - **Tab 4 — Nhập liệu mới & Huấn luyện lại:** Cung cấp form nhập gọn nhẹ (kèm 3 nút điền mẫu nhanh 1-click để demo) và upload CSV. Dữ liệu mới được kiểm tra hợp lệ (`validate`), chống trùng lô bằng SHA-256 trước khi lưu vào SQLite (`data/customers.sqlite3`), đồng thời **in ngay phân khúc và khuyến mãi gợi ý ra màn hình** dựa trên khoảng cách tới tâm cụm gần nhất.
3. **Huấn luyện lại định kỳ:** Hỗ trợ 3 cách chạy linh hoạt — bấm nút huấn luyện thủ công ở Tab 4; tự động huấn luyện lại ngay khi mở ứng dụng nếu có đủ $\ge N$ khách hàng mới (mặc định 10) hoặc quá $X$ ngày (mặc định 7) ở thanh bên; hoặc chạy tự động ngoài giao diện qua Windows Task Scheduler bằng lệnh `python retrain.py --if-due`. Toàn bộ mã nguồn đã được kiểm thử tự động đạt **15/15 test (`pytest`)**.

---

## CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 6.1. Kết luận
Nhóm đã thực hiện đầy đủ các bước của một dự án khai phá dữ liệu hoàn chỉnh: từ tìm hiểu ngữ cảnh dữ liệu, phát hiện dữ liệu bị lệch và xử lý ngoại lai bằng IQR, chuẩn hóa thang đo, đến thực nghiệm đối chiếu 3 thuật toán phân cụm K-Means, Hierarchical và DBSCAN, cuối cùng đóng gói thành ứng dụng Streamlit có khả năng thu thập dữ liệu mới và tự huấn luyện lại. Kết quả cho thấy việc loại bỏ ngoại lai đúng cách và chuẩn hóa bằng `StandardScaler` giúp cả 3 thuật toán hoạt động hiệu quả và tách được 3 phân khúc khách hàng có ý nghĩa thực tế cho chiến lược khuyến mãi của siêu thị.

### 6.2. Hạn chế và hướng phát triển
- **Hạn chế:** Bộ dữ liệu gốc chỉ ghi nhận tổng chi tiêu năm của 440 khách hàng, chưa có lịch sử giao dịch theo ngày/tháng, chưa có mã định danh khách hàng riêng biệt và chưa có dữ liệu tỷ lệ phản hồi thực tế sau khuyến mãi (nên chưa đo lường được doanh thu tăng thêm).
- **Hướng phát triển:** Thu thập dữ liệu hóa đơn theo thời gian để kết hợp thêm mô hình RFM (Recency – Frequency – Monetary), và triển khai thử nghiệm A/B Testing trên từng phân khúc để đánh giá hiệu quả lợi nhuận thực tế của các gói khuyến mãi.

---

## HƯỚNG DẪN CHIA SLIDE & KỊCH BẢN THUYẾT TRÌNH CHO 5 NGƯỜI

1. **Thành viên 1 (2 phút):** Trình bày Chương 1 — Đặt vấn đề, giới thiệu 8 cột dữ liệu Wholesale Customers và bảng EDA (chỉ ra dữ liệu bị lệch phải và chênh lệch thang đo).
2. **Thành viên 2 (2 phút):** Trình bày Chương 2 — Giải thích bước `validate`, biểu đồ Boxplot loại 108 ngoại lai bằng IQR (nhấn mạnh lý do không xóa khách VIP ở đầu ra) và công thức chuẩn hóa `StandardScaler` / `MinMaxScaler`.
3. **Thành viên 3 (2 phút):** Trình bày Chương 3 (K-Means) & Mục 4.1 — Giải thích Elbow Method ($k=4$), Silhouette Score, lý do chọn `StandardScaler` và lý do bắt buộc $k \ge 3$.
4. **Thành viên 4 (2 phút):** Trình bày Chương 3 (Hierarchical, DBSCAN) & Mục 4.2, 4.3 — Dendrogram, k-distance, bảng so sánh 3 thuật toán và 3 phân khúc khách hàng kèm gợi ý khuyến mãi tương ứng.
5. **Thành viên 5 (2–3 phút):** Demo trực tiếp trên màn hình ứng dụng Streamlit (mở nhanh Tab 1–3, sau đó sang Tab 4 bấm điền mẫu thử nhanh để lưu khách mới và chỉ ra kết quả phân khúc in ngay ra màn hình + cơ chế tự huấn luyện lại định kỳ ở thanh bên).
