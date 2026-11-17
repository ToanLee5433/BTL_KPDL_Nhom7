"""Màn hình hiển thị kết quả phân khúc khách hàng siêu thị và gợi ý khuyến mãi."""
import json
from html import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from pipeline import FEATURES, CATEGORY_VI, latest_run, data_fingerprint

NAMES = CATEGORY_VI
COLORS = ['#2563eb', '#0d9488', '#9333ea', '#ea580c', '#db2777', '#64748b', '#ca8a04', '#0891b2']


def latest_result():
    folder = latest_run()
    required = ['metadata.json', 'assignments.csv', 'segments.csv', 'medians.csv', 'relative_spending.csv']
    return folder if folder is not None and all((folder / name).exists() for name in required) else None


def campaign_draft(customer, profile, ratios, run_id, eligible=True):
    """Tạo bản nháp gợi ý khuyến mãi theo hồ sơ phân khúc của khách hàng."""
    cluster = int(customer['cluster'])
    if cluster == -1 or not eligible:
        return None
    top = ratios.loc[cluster, FEATURES].dropna().sort_values(ascending=False).head(2)
    if top.empty:
        return None
    product_names = [NAMES[name] for name in top.index]
    strategy = profile.get('suggestion') if hasattr(profile, 'get') else None
    if not isinstance(strategy, str) or not strategy:
        strategy = 'Ưu đãi theo ngưỡng đơn hàng cho ' + ' và '.join(product_names).lower()
    outlier = bool(customer.get('is_outlier', False)) if hasattr(customer, 'get') else False
    if outlier:
        campaign = 'Chăm sóc VIP / Ngoại lai: Khách hàng có mức chi tiêu đột biến'
        message = 'Chuyên viên chăm sóc khách hàng liên hệ trực tiếp để đề xuất chính sách chiết khấu riêng.'
    else:
        campaign = strategy
        message = ('Kính mời Quý khách tham khảo chương trình ưu đãi siêu thị dành cho nhóm hàng '
                   + ' và '.join(product_names).lower() + '. Chi tiết áp dụng tại quầy hoặc đơn đặt hàng tuần này.')
    return {
        'customer_ref': f"KH-{int(customer['row_id']):04d}", 'run_id': run_id, 'cluster': cluster,
        'segment': str(profile['segment']), 'is_outlier': outlier, 'product_categories': product_names,
        'relative_spending': {NAMES[name]: float(value) for name, value in top.items()},
        'campaign': campaign, 'message': message,
        'status': 'Bản nháp gợi ý — sẵn sàng áp dụng',
        'approval_checks': [
            'Kiểm tra tồn kho các mặt hàng nổi bật',
            'Xác nhận ngưỡng hóa đơn áp dụng voucher / chiết khấu',
            'Gửi thông báo qua ứng dụng / SMS khách hàng thân thiết',
        ],
    }


def render_results_tab(data):
    """Hiển thị kết quả phân cụm, biểu đồ PCA, hồ sơ cụm và gợi ý khuyến mãi."""
    folder = latest_result()
    if folder is None:
        st.info('Chưa có mô hình. Mở tab **Thu thập dữ liệu & Huấn luyện lại** để huấn luyện mô hình đầu tiên.')
        return
    meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    customers = pd.read_csv(folder / 'assignments.csv')
    segments = pd.read_csv(folder / 'segments.csv')
    medians = pd.read_csv(folder / 'medians.csv').set_index('cluster')
    ratios = pd.read_csv(folder / 'relative_spending.csv').set_index('cluster')

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric('Tổng khách hàng', len(customers))
    k2.metric('Số phân khúc', meta['metrics']['clusters'])
    k3.metric('Thuật toán chính', meta['algorithm'])
    k4.metric('Ngoại lai (IQR)', int(customers.get('is_outlier', pd.Series(dtype=bool)).sum()))
    k5.metric('Silhouette Score', f"{meta['metrics']['silhouette']:.3f}")

    if data_fingerprint(data) != meta['data_sha256']:
        st.warning('Kho dữ liệu đã có thêm khách hàng mới chưa được đưa vào mô hình này. Vào tab **Thu thập dữ liệu & Huấn luyện lại** để cập nhật.')

    st.subheader('1. Cơ cấu phân khúc & Gợi ý khuyến mãi')
    left, right = st.columns([1, 2])
    with left:
        counts = customers.groupby('cluster').size()
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.pie(counts, labels=[f'Nhóm {c}' if c != -1 else 'Nhiễu' for c in counts.index],
               autopct='%1.0f%%', colors=COLORS[:len(counts)], wedgeprops={'width': .38}, startangle=90)
        ax.set_title('Tỷ lệ khách hàng theo cụm')
        st.pyplot(fig)
        plt.close(fig)
    with right:
        for profile in segments.itertuples():
            cluster = int(profile.cluster)
            title = 'Điểm nhiễu (DBSCAN)' if cluster == -1 else str(profile.segment)
            body = f'<b>Khuyến mãi:</b> {escape(str(profile.suggestion))}'
            st.markdown(
                f'<div style="padding:14px 16px;border:1px solid #cbd5e1;border-top:4px solid #0d9488;border-radius:10px;margin-bottom:10px;">'
                f'<h4 style="margin:0 0 6px 0;">{escape(title)} · {int(profile.customers)} khách hàng</h4><div>{body}</div></div>',
                unsafe_allow_html=True,
            )

    st.subheader('2. Trực quan hóa phân cụm (PCA 2D) & Mức chi tiêu tương đối')
    c_pca, c_heat = st.columns(2)
    with c_pca:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for label, group in customers.groupby('cluster'):
            core = group[~group['is_outlier']] if 'is_outlier' in group else group
            ax.scatter(core.PC1, core.PC2, s=20, alpha=.65, label=f'Nhóm {label}' if label != -1 else 'Nhiễu')
        if 'is_outlier' in customers and customers['is_outlier'].any():
            out = customers[customers.is_outlier]
            ax.scatter(out.PC1, out.PC2, s=36, facecolors='none', edgecolors='#111827', linewidths=1.0, label='Ngoại lai (IQR)')
        ax.set(xlabel='PC1', ylabel='PC2', title='Phân bố cụm trên không gian PCA 2D')
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"PCA 2D giữ lại {sum(meta['pca_explained_variance']):.1%} phương sai (chỉ dùng để vẽ hình; phân cụm dùng đủ 6 nhóm hàng).")
    with c_heat:
        st.write('Chi tiêu trung bình của từng cụm so với mức chung (1.0 = bằng trung bình):')
        st.dataframe(ratios[FEATURES].style.format('{:.2f}').background_gradient(cmap='YlGnBu'), width='stretch')
        st.write('Trung vị chi tiêu trên đơn vị gốc:')
        st.dataframe(medians[FEATURES].style.format('{:,.0f}'), width='stretch')

    st.divider()
    st.subheader('3. Tra cứu từng khách hàng & Xuất đề xuất khuyến mãi')
    options = customers.row_id.astype(int).tolist()
    customer_id = st.selectbox('Chọn mã khách hàng', options, format_func=lambda v: f'KH-{v:04d}', key='demo_customer')
    customer = customers.loc[customers.row_id == customer_id].iloc[0]
    cluster = int(customer.cluster)
    profile = segments.loc[segments.cluster == cluster].iloc[0]
    is_outlier = bool(customer.get('is_outlier', False))

    a, b, c = st.columns(3)
    a.metric('Tổng chi tiêu năm', f'{customer[FEATURES].sum():,.0f}')
    b.metric('Phân khúc', str(profile.segment) + (' (Ngoại lai)' if is_outlier else ''))
    c.metric('Kênh mua hàng', 'Bán lẻ / Siêu thị' if customer.Channel == 2 else 'Khách sạn / Nhà hàng / Café')

    chart_col, offer_col = st.columns(2)
    with chart_col:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        positions = np.arange(len(FEATURES))
        ax.barh(positions - .18, customer[FEATURES].to_numpy(dtype=float), height=.35, color='#2563eb', label='Khách hàng')
        ax.barh(positions + .18, medians.loc[cluster, FEATURES].to_numpy(dtype=float), height=.35, color='#0d9488', label='Trung vị nhóm')
        ax.set_yticks(positions, [NAMES[f] for f in FEATURES])
        ax.set_xlabel('Chi tiêu năm')
        ax.set_title('So sánh chi tiêu khách hàng với trung vị nhóm')
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
                    for name, ratio in saved['relative_spending'].items():
                        st.write(f'• **{name}**: cao gấp **{ratio:.2f} lần** trung bình chung')
                    st.download_button(
                        'Tải phiếu đề xuất (JSON)',
                        json.dumps(saved, ensure_ascii=False, indent=2).encode('utf-8'),
                        f"{saved['customer_ref']}_offer.json", 'application/json',
                    )
            else:
                st.write(f"**Gợi ý mặc định:** {profile.suggestion}")

    with st.expander('Danh sách toàn bộ khách hàng & Tải kết quả CSV'):
        cluster_filter = st.multiselect('Lọc theo nhóm', sorted(customers.cluster.unique().tolist()))
        visible = customers[customers.cluster.isin(cluster_filter)] if cluster_filter else customers
        st.dataframe(visible, width='stretch')
        st.download_button('Tải kết quả phân khúc (CSV)', visible.to_csv(index=False).encode('utf-8-sig'), 'customer_segments.csv', 'text/csv')
