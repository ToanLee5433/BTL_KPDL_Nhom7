"""Màn hình hiển thị kết quả phân khúc khách hàng RFM và gợi ý khuyến mãi."""
import json
from html import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from pipeline import FEATURES, FEATURE_VI, latest_run, data_fingerprint

COLORS = ['#2563eb', '#ea580c', '#0d9488', '#9333ea', '#db2777', '#64748b']


def latest_result():
    folder = latest_run()
    required = ['metadata.json', 'assignments.csv', 'segments.csv', 'medians.csv', 'relative_spending.csv']
    return folder if folder is not None and all((folder / name).exists() for name in required) else None


def campaign_draft(customer, profile, ratios, run_id, eligible=True):
    """Tạo bản nháp gợi ý khuyến mãi dựa trên hồ sơ RFM của phân khúc."""
    cluster = int(customer['cluster'])
    if cluster == -1 or not eligible:
        return None
    outlier = bool(customer.get('is_outlier', False)) if hasattr(customer, 'get') else False
    if outlier:
        campaign = 'Chăm sóc VIP / Mua buôn đột biến (Ngoại lai IQR)'
        message = 'Chuyên viên CSKH liên hệ trực tiếp để áp dụng chính sách chiết khấu theo đơn hàng lớn.'
    else:
        campaign = str(profile['suggestion'])
        message = f"Kính gửi Quý khách (Mã KH {int(customer['CustomerID'])}), {campaign}"
    return {
        'customer_ref': f"KH-{int(customer['CustomerID'])}", 'run_id': run_id, 'cluster': cluster,
        'segment': str(profile['segment']), 'is_outlier': outlier,
        'rfm': {'Recency': int(customer['Recency']), 'Frequency': int(customer['Frequency']), 'Monetary': float(customer['Monetary'])},
        'relative_rfm': {f: float(ratios.loc[cluster, f]) for f in FEATURES},
        'campaign': campaign, 'message': message,
        'status': 'Bản nháp gợi ý — sẵn sàng triển khai',
    }


def render_results_tab(data):
    """Hiển thị kết quả phân cụm RFM, biểu đồ PCA, hồ sơ cụm và gợi ý khuyến mãi."""
    folder = latest_result()
    if folder is None:
        st.info('Chưa có mô hình. Mở tab **4. Nhập liệu mới & Huấn luyện lại** để huấn luyện mô hình đầu tiên.')
        return
    meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    customers = pd.read_csv(folder / 'assignments.csv')
    segments = pd.read_csv(folder / 'segments.csv')
    medians = pd.read_csv(folder / 'medians.csv').set_index('cluster')
    ratios = pd.read_csv(folder / 'relative_spending.csv').set_index('cluster')

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric('Tổng khách hàng', f"{len(customers):,}")
    k2.metric('Số phân khúc', meta['metrics']['clusters'])
    k3.metric('Thuật toán chính', meta['algorithm'])
    k4.metric('Ngoại lai VIP (IQR)', f"{int(customers.get('is_outlier', pd.Series(dtype=bool)).sum()):,}")
    k5.metric('Silhouette Score', f"{meta['metrics']['silhouette']:.3f}")

    if data_fingerprint(data) != meta['data_sha256']:
        st.warning('Kho dữ liệu đã có thêm khách hàng mới chưa đưa vào mô hình này. Vào Tab 4 để huấn luyện lại.')

    st.subheader('1. Cơ cấu phân khúc RFM & Chiến lược khuyến mãi')
    left, right = st.columns([1, 2])
    with left:
        counts = customers.groupby('cluster').size()
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.pie(counts, labels=[f'Nhóm {c}' if c != -1 else 'Nhiễu' for c in counts.index],
               autopct='%1.1f%%', colors=COLORS[:len(counts)], wedgeprops={'width': .38}, startangle=90)
        ax.set_title('Tỷ lệ khách hàng theo phân khúc')
        st.pyplot(fig)
        plt.close(fig)
    with right:
        for profile in segments.itertuples():
            cluster = int(profile.cluster)
            title = 'Điểm nhiễu (DBSCAN)' if cluster == -1 else str(profile.segment)
            st.markdown(
                f'<div style="padding:14px 16px;border:1px solid #cbd5e1;border-top:4px solid #0d9488;border-radius:10px;margin-bottom:10px;">'
                f'<h4 style="margin:0 0 6px 0;">{escape(title)} · {int(profile.customers):,} khách hàng</h4>'
                f'<div style="font-size:0.9rem;color:#475569;margin-bottom:4px;">{escape(str(profile.rationale))}</div>'
                f'<div><b>Khuyến mãi:</b> {escape(str(profile.suggestion))}</div></div>',
                unsafe_allow_html=True,
            )

    st.subheader('2. Trực quan hóa phân cụm (PCA 2D) & Chỉ số RFM tương đối')
    c_pca, c_heat = st.columns(2)
    with c_pca:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for label, group in customers.groupby('cluster'):
            core = group[~group['is_outlier']] if 'is_outlier' in group else group
            ax.scatter(core.PC1, core.PC2, s=12, alpha=.55, label=f'Nhóm {label}' if label != -1 else 'Nhiễu')
        if 'is_outlier' in customers and customers['is_outlier'].any():
            out = customers[customers.is_outlier]
            ax.scatter(out.PC1, out.PC2, s=18, facecolors='none', edgecolors='#111827', alpha=.45, linewidths=.7, label='Ngoại lai VIP (IQR)')
        ax.set(xlabel='PC1', ylabel='PC2', title='Phân bố cụm khách hàng trên không gian PCA 2D')
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"PCA 2D giữ lại {sum(meta['pca_explained_variance']):.1%} phương sai (trực quan hóa 3 chiều R, F, M).")
    with c_heat:
        st.write('Tỷ lệ R, F, M trung bình của từng cụm so với mức chung (1.0 = bằng trung bình):')
        st.dataframe(ratios[FEATURES].style.format('{:.2f}').background_gradient(cmap='YlGnBu'), width='stretch')
        st.write('Trung vị (Median) thực tế của từng nhóm:')
        st.dataframe(medians[FEATURES].style.format('{:,.1f}'), width='stretch')

    st.divider()
    st.subheader('3. Tra cứu từng khách hàng & Xuất phiếu khuyến mãi')
    options = customers.CustomerID.astype(int).tolist()
    cid = st.selectbox('Chọn CustomerID (gõ mã số để tìm nhanh)', options, key='demo_customer')
    customer = customers.loc[customers.CustomerID == cid].iloc[0]
    cluster = int(customer.cluster)
    profile = segments.loc[segments.cluster == cluster].iloc[0]
    is_outlier = bool(customer.get('is_outlier', False))

    a, b, c, d = st.columns(4)
    a.metric('Recency (ngày)', f"{int(customer.Recency)}")
    b.metric('Frequency (đơn)', f"{int(customer.Frequency)}")
    c.metric('Monetary (£)', f"£{customer.Monetary:,.1f}")
    d.metric('Quốc gia', str(customer.Country))
    st.write(f"**Phân khúc:** {profile.segment}" + (' · ⭐ *Ngoại lai VIP (IQR)*' if is_outlier else ''))

    chart_col, offer_col = st.columns(2)
    with chart_col:
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        positions = np.arange(len(FEATURES))
        med_vals = medians.loc[cluster, FEATURES].to_numpy(dtype=float)
        cust_vals = customer[FEATURES].to_numpy(dtype=float)
        # Vẽ tỷ lệ so với trung vị nhóm (1.0 = bằng trung vị) để R, F, M cùng nằm trên một biểu đồ dễ đọc
        ratios_cust = cust_vals / np.where(med_vals == 0, 1, med_vals)
        ax.barh(positions, ratios_cust, color='#2563eb', alpha=.85)
        ax.axvline(1.0, color='#dc2626', linestyle='--', label='Trung vị nhóm (= 1.0)')
        ax.set_yticks(positions, [FEATURE_VI[f] for f in FEATURES])
        ax.set_xlabel('Tỷ lệ so với trung vị của nhóm')
        ax.set_title(f'Hồ sơ khách hàng KH-{cid}')
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with offer_col:
        draft = campaign_draft(customer, profile, ratios, folder.name, meta.get('eligible', True))
        if draft is None:
            st.info('Khách hàng thuộc điểm nhiễu, cần xem xét thủ công.')
        else:
            if st.button('Tạo đề xuất khuyến mãi', type='primary', key='generate_offer'):
                st.session_state['offer_draft'] = draft
            saved = st.session_state.get('offer_draft')
            if saved and saved['run_id'] == folder.name and saved['customer_ref'] == draft['customer_ref']:
                with st.container(border=True):
                    st.markdown(f"**{saved['campaign']}**")
                    st.info(saved['message'])
                    st.download_button(
                        'Tải phiếu đề xuất (JSON)',
                        json.dumps(saved, ensure_ascii=False, indent=2).encode('utf-8'),
                        f"{saved['customer_ref']}_offer.json", 'application/json',
                    )
            else:
                st.write(f"**Gợi ý mặc định:** {profile.suggestion}")

    with st.expander('Danh sách khách hàng & Tải kết quả CSV'):
        cluster_filter = st.multiselect('Lọc theo phân khúc', sorted(customers.cluster.unique().tolist()))
        visible = customers[customers.cluster.isin(cluster_filter)] if cluster_filter else customers
        st.dataframe(visible.head(500), width='stretch')
        st.download_button('Tải toàn bộ kết quả phân khúc (CSV)', visible.to_csv(index=False).encode('utf-8-sig'), 'online_retail_segments.csv', 'text/csv')
