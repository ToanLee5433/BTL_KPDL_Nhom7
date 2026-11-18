import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.neighbors import NearestNeighbors
import streamlit as st

from pipeline import (
    FEATURES, COLUMNS, FEATURE_VI, train, validate, search,
    latest_run, predict_segments, retrain_due,
)
from storage import initialize, load_data, add_batch, history
from experience import render_results_tab

st.set_page_config(page_title='Phân khúc khách hàng bán lẻ (RFM)', layout='wide')
st.title('Phân khúc khách hàng bán lẻ (RFM) để gợi ý khuyến mãi')
st.caption('Bài toán Phân cụm · Bộ dữ liệu Online Retail (UCI 352: 541,909 giao dịch → 4,338 khách hàng RFM) · K-Means / Hierarchical / DBSCAN')
initialize()
data = load_data()

# ---- Thanh bên: Trạng thái kho dữ liệu & Chính sách huấn luyện định kỳ ----
st.sidebar.metric('Khách hàng trong kho (RFM)', f'{len(data):,}')
st.sidebar.subheader('Huấn luyện định kỳ')
auto_retrain = st.sidebar.checkbox('Tự huấn luyện lại khi đến hạn', value=True, key='policy_auto')
min_new = st.sidebar.number_input('Khi có ≥ N khách hàng mới', min_value=1, value=20, step=1, key='policy_min_new')
max_days = st.sidebar.number_input('hoặc sau X ngày có dữ liệu mới', min_value=1, value=7, step=1, key='policy_max_days')
due, reason = retrain_due(len(data), min_new=int(min_new), max_days=int(max_days))
folder_now = latest_run()
if folder_now is not None:
    last_meta = json.loads((folder_now / 'metadata.json').read_text(encoding='utf-8'))
    st.sidebar.caption(f"Phiên hiện tại: {last_meta['algorithm']} {last_meta['params']} ({last_meta['scaler_type']}) · {last_meta['rows']:,} KH")
st.sidebar.caption(('⏰ Đến hạn: ' if due else '✅ ') + reason)
if due and auto_retrain and not st.session_state.get('auto_train_failed'):
    try:
        with st.spinner(f'Đang tự huấn luyện lại: {reason}'):
            train(data)
        st.toast('Đã cập nhật mô hình phân khúc RFM mới.')
    except ValueError as error:
        st.session_state['auto_train_failed'] = True
        st.error(f'Huấn luyện tự động thất bại: {error}')

tabs = st.tabs([
    '1. Dữ liệu & Khám phá (EDA)',
    '2. Tiền xử lý & So sánh 3 thuật toán',
    '3. Kết quả phân khúc & Khuyến mãi',
    '4. Nhập liệu mới & Huấn luyện lại',
])


# ==============================================================================
# TAB 1: TỔNG QUAN & KHÁM PHÁ DỮ LIỆU (EDA)
# ==============================================================================
with tabs[0]:
    st.subheader('Tổng quan bộ dữ liệu Online Retail (UCI 352) & Mô hình RFM')
    st.write('Từ **541,909 dòng giao dịch gốc**, sau khi làm sạch (loại 135,080 dòng thiếu `CustomerID`, '
             '8,905 hóa đơn hủy `C*`, 40 dòng `Quantity/UnitPrice ≤ 0` → còn **397,884 giao dịch hợp lệ**), '
             'hệ thống tổng hợp thành **3 chỉ số RFM** cho từng khách hàng: '
             '**Recency** (số ngày từ lần mua cuối), **Frequency** (số đơn hàng), **Monetary** (tổng chi tiêu £).')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Tổng khách hàng (RFM)', f'{len(data):,}')
    c2.metric('Số quốc gia', int(data['Country'].nunique()))
    c3.metric('Giá trị thiếu (Missing)', int(data.isna().sum().sum()))
    c4.metric('Chi tiêu trung vị (£)', f"£{data['Monetary'].median():,.1f}")

    st.subheader('Phát hiện ngoại lai (Outliers) bằng quy tắc IQR — Boxplot')
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col in zip(axes, FEATURES):
        ax.boxplot(data[col], orientation='vertical')
        ax.set_title(FEATURE_VI[col])
        ax.set_xticks([])
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption('Các dấu tròn phía trên râu Boxplot là các khách hàng ngoại lai (vượt Q3 + 1.5×IQR, ví dụ khách mua buôn chi hàng chục nghìn bảng Anh).')

    with st.expander('Xem bảng dữ liệu RFM & Thống kê mô tả'):
        st.dataframe(data.head(200), width='stretch')
        st.dataframe(data[FEATURES].describe().T, width='stretch')


# ==============================================================================
# TAB 2: TIỀN XỬ LÝ & SO SÁNH 3 THUẬT TOÁN
# ==============================================================================
with tabs[1]:
    st.subheader('Cấu hình tiền xử lý & Khảo sát trực tiếp')
    col_p1, col_p2, col_p3 = st.columns(3)
    outlier_opt = col_p1.selectbox(
        'Loại bỏ ngoại lai',
        ['iqr', 'iqr_log', 'none'],
        format_func=lambda x: {'iqr': 'IQR trực tiếp (Chuẩn)', 'iqr_log': 'IQR trên thang Log1p', 'none': 'Không loại ngoại lai'}[x],
    )
    scaler_opt = col_p2.selectbox(
        'Chuẩn hóa dữ liệu',
        ['standard', 'minmax'],
        format_func=lambda x: {'standard': 'StandardScaler (Z-score)', 'minmax': 'MinMaxScaler [0, 1]'}[x],
    )
    use_log_opt = col_p3.checkbox('Biến đổi Log1p trước chuẩn hóa', value=False)

    @st.cache_data(show_spinner='Đang khảo sát tham số cho 3 thuật toán...')
    def cached_search(_hash, df, sc, out, log):
        return search(df, scaler_type=sc, outlier_method=out, use_log=log)

    res = cached_search(len(data), data, scaler_opt, outlier_opt, use_log_opt)
    st.info(f'Sau tiền xử lý: giữ **{int((~res.outlier).sum()):,}/{len(data):,}** khách hàng chuẩn để huấn luyện '
            f'(tách riêng **{int(res.outlier.sum()):,}** khách ngoại lai VIP). Phương pháp Khuỷu tay gợi ý **k = {res.elbow}**.')

    st.subheader('1. Đánh giá K-Means & Hierarchical: Elbow Method và Silhouette Score')
    km = res.table[res.table.algorithm == 'KMeans'].copy()
    hc = res.table[res.table.algorithm == 'Hierarchical'].copy()
    km['k'] = km.params.map(lambda x: json.loads(x)['k'])
    hc['k'] = hc.params.map(lambda x: json.loads(x)['k'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(km.k, km.inertia, 'o-', color='#2563eb')
    axes[0].axvline(res.elbow, color='#dc2626', linestyle='--', label=f'Điểm khuỷu gợi ý k = {res.elbow}')
    axes[0].set(xlabel='Số cụm (k)', ylabel='Inertia (WCSS)', title='Phương pháp Khuỷu tay (Elbow Method)')
    axes[0].legend()

    axes[1].plot(km.k, km.silhouette, 'o-', label='K-Means')
    axes[1].plot(hc.k, hc.silhouette, 's-', label='Hierarchical (Ward)')
    axes[1].set(xlabel='Số cụm (k)', ylabel='Silhouette Score (càng cao càng tốt)', title='So sánh Silhouette Score theo k')
    axes[1].legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader('2. Trực quan hóa Hierarchical (Dendrogram) & DBSCAN (k-distance)')
    c_den, c_kd = st.columns(2)
    with c_den:
        fig, ax = plt.subplots(figsize=(6, 4))
        dendrogram(linkage(res.X_fit, method='ward'), truncate_mode='lastp', p=20, ax=ax)
        ax.set(title='Dendrogram (Hierarchical Ward, rút gọn 20 nhánh)', xlabel='Nhóm gộp', ylabel='Khoảng cách')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with c_kd:
        fig, ax = plt.subplots(figsize=(6, 4))
        dists = np.sort(NearestNeighbors(n_neighbors=10).fit(res.X_fit).kneighbors(res.X_fit)[0][:, -1])
        ax.plot(dists, color='#0d9488')
        ax.set(title='Biểu đồ k-distance (chọn eps cho DBSCAN, k=10)', xlabel='Khách hàng sắp xếp tăng dần', ylabel='Khoảng cách láng giềng thứ 10')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader('3. Bảng so sánh 3 thuật toán: K-Means · Hierarchical · DBSCAN')
    best_show = res.best[['algorithm', 'params', 'clusters', 'noise_rate', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'seconds', 'eligible']].copy()
    best_show.columns = ['Thuật toán', 'Tham số tốt nhất', 'Số cụm', 'Tỷ lệ nhiễu', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'Thời gian (s)', 'Đạt tiêu chí']
    st.dataframe(best_show, width='stretch')

    if st.button('Áp dụng cấu hình này & Huấn luyện lưu phiên', type='primary', key='apply_tab2'):
        with st.spinner('Đang huấn luyện và lưu kết quả...'):
            folder = train(data, scaler_type=scaler_opt, outlier_method=outlier_opt, use_log=use_log_opt)
        st.success(f'Đã huấn luyện và lưu phiên mới: {folder.name}. Mở Tab 3 để xem chi tiết phân khúc!')

    with st.expander('Xem toàn bộ bảng khảo sát tham số (K-Means, Hierarchical, DBSCAN)'):
        st.dataframe(res.table, width='stretch')


# ==============================================================================
# TAB 3: KẾT QUẢ PHÂN KHÚC & GỢI Ý KHUYẾN MÃI
# ==============================================================================
with tabs[2]:
    render_results_tab(data)


# ==============================================================================
# TAB 4: THU THẬP DỮ LIỆU MỚI & HUẤN LUYỆN LẠI ĐỊNH KỲ
# ==============================================================================
with tabs[3]:
    col_form, col_train = st.columns([1.2, 1])

    with col_form:
        st.subheader('Nhập khách hàng mới (dự đoán phân khúc RFM ngay)')
        next_id = int(data['CustomerID'].max()) + 1
        presets = {
            'Khách mới tiềm năng': dict(Recency=15, Frequency=2, Monetary=450., TotalItems=220),
            'Khách trung thành VIP': dict(Recency=10, Frequency=8, Monetary=2800., TotalItems=1500),
            'Khách sắp rời bỏ': dict(Recency=250, Frequency=1, Monetary=250., TotalItems=110),
        }
        p_cols = st.columns(3)
        for idx, (label, vals) in enumerate(presets.items()):
            if p_cols[idx].button(f'Mẫu: {label}', key=f'preset_{idx}'):
                st.session_state['in_cid'] = next_id
                for k, v in vals.items():
                    st.session_state[f'in_{k}'] = v

        with st.form('customer'):
            c_a, c_b = st.columns(2)
            cid = c_a.number_input('Mã khách hàng (CustomerID)', min_value=1, step=1,
                                   value=int(st.session_state.get('in_cid', next_id)), key='in_cid')
            country = c_b.text_input('Quốc gia (Country)', value='United Kingdom', key='in_Country')
            f_cols = st.columns(2)
            recency = f_cols[0].number_input('Recency (số ngày từ lần mua cuối)', min_value=0, step=1,
                                             value=int(st.session_state.get('in_Recency', 30)), key='in_Recency')
            frequency = f_cols[1].number_input('Frequency (số đơn hàng)', min_value=1, step=1,
                                               value=int(st.session_state.get('in_Frequency', 1)), key='in_Frequency')
            monetary = f_cols[0].number_input('Monetary (tổng chi tiêu £)', min_value=0.0, step=50.0,
                                              value=float(st.session_state.get('in_Monetary', 0.0)), key='in_Monetary')
            items = f_cols[1].number_input('Tổng số sản phẩm (TotalItems)', min_value=1, step=10,
                                           value=int(st.session_state.get('in_TotalItems', 50)), key='in_TotalItems')
            confirmed = st.checkbox('Đã kiểm tra đây là khách hàng mới và số liệu hợp lệ')
            submit = st.form_submit_button('Lưu & Dự đoán phân khúc', type='primary')

        if submit:
            if not confirmed:
                st.warning('Cần đánh dấu xác nhận trước khi lưu.')
            else:
                try:
                    new = pd.DataFrame([dict(CustomerID=cid, Recency=recency, Frequency=frequency,
                                             Monetary=monetary, TotalItems=items, Country=country)])
                    count = add_batch(new, 'Nhập từ biểu mẫu')
                    st.session_state['save_notice'] = f'Đã lưu {count} khách hàng mới vào SQLite.' if count else 'Bản ghi giống hệt đã tồn tại; không thêm trùng.'
                    st.session_state['save_prediction'] = predict_segments(new) if count else None
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))

        if 'save_notice' in st.session_state:
            st.success(st.session_state.pop('save_notice'))
        if st.session_state.get('save_prediction') is not None:
            pred = st.session_state.pop('save_prediction')
            with st.container(border=True):
                st.markdown('**Kết quả phân khúc & khuyến mãi cho khách hàng vừa nhập:**')
                for row in pred.itertuples():
                    st.write(f'• KH-{row.CustomerID}: **{row.segment}**' + (' · ⭐ Ngoại lai VIP' if row.is_outlier else ''))
                    st.info(row.suggestion)

        st.divider()
        st.subheader('Nhập hàng loạt từ CSV')
        st.download_button('Tải file mẫu CSV', pd.DataFrame(columns=COLUMNS).to_csv(index=False).encode('utf-8-sig'), 'retail_rfm_template.csv', 'text/csv')
        uploaded = st.file_uploader('Chọn file CSV (đủ 6 cột RFM)', type='csv')
        if uploaded is not None:
            try:
                incoming = validate(pd.read_csv(uploaded))
                st.dataframe(incoming.head(10))
                agree = st.checkbox('Xác nhận bổ sung tập khách hàng từ CSV này')
                if st.button('Lưu CSV vào kho', disabled=not agree):
                    count = add_batch(incoming, uploaded.name)
                    st.session_state['save_notice'] = f'Đã lưu {count} khách hàng.' if count else 'File này đã được nhập trước đó.'
                    st.session_state['save_prediction'] = predict_segments(incoming.head(3)) if count else None
                    st.rerun()
            except (ValueError, pd.errors.ParserError, UnicodeError) as error:
                st.error(str(error))

    with col_train:
        st.subheader('Huấn luyện lại mô hình')
        st.write('Chọn cấu hình và bấm nút để huấn luyện lại trên toàn bộ kho dữ liệu hiện có:')
        alg_opt = st.selectbox('Thuật toán chính', ['Auto', 'KMeans', 'Hierarchical', 'DBSCAN'])
        sc_opt = st.selectbox('Bộ chuẩn hóa', ['standard', 'minmax'], format_func=lambda x: 'StandardScaler' if x == 'standard' else 'MinMaxScaler', key='tr_sc')
        out_opt = st.selectbox('Xử lý ngoại lai', ['iqr', 'iqr_log', 'none'], format_func=lambda x: {'iqr': 'IQR trực tiếp', 'iqr_log': 'IQR (Log1p)', 'none': 'Không loại'}[x], key='tr_out')
        if st.button('Huấn luyện lại ngay', type='primary', key='btn_retrain'):
            try:
                with st.spinner('Đang huấn luyện lại mô hình...'):
                    folder = train(load_data(), scaler_type=sc_opt, outlier_method=out_opt, algorithm=alg_opt)
                st.session_state['auto_train_failed'] = False
                st.success(f'Huấn luyện xong phiên {folder.name}')
            except ValueError as error:
                st.error(str(error))
        st.caption('Ngoài giao diện, có thể đặt lịch tự động chạy `python retrain.py --if-due` trên Windows Task Scheduler.')

        st.subheader('Lịch sử nhập dữ liệu')
        st.dataframe(history(), width='stretch')
