"""Chạy tay hoặc bằng Windows Task Scheduler / cron. Không cần mở Streamlit.

Ví dụ:
  python retrain.py
  python retrain.py --scaler minmax --outlier iqr_log
  python retrain.py --if-due
"""
import argparse
from pipeline import train, retrain_due
from storage import initialize, load_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', choices=['Auto', 'KMeans', 'Hierarchical', 'DBSCAN'], default='Auto')
    parser.add_argument('--scaler', choices=['standard', 'minmax'], default='standard')
    parser.add_argument('--outlier', choices=['iqr', 'iqr_log', 'none'], default='iqr')
    parser.add_argument('--use-log', action='store_true', help='Biến đổi log1p trước khi chuẩn hóa.')
    parser.add_argument('--allow-ineligible', action='store_true', help='Cho phép lưu cấu hình chưa đạt tiêu chí.')
    parser.add_argument('--if-due', action='store_true', help='Chỉ huấn luyện khi đủ khách mới hoặc quá hạn ngày.')
    parser.add_argument('--min-new', type=int, default=20, help='Ngưỡng số khách hàng mới (dùng với --if-due).')
    parser.add_argument('--max-days', type=int, default=7, help='Ngưỡng số ngày (dùng với --if-due).')
    args = parser.parse_args()
    initialize()
    data = load_data()
    if args.if_due:
        due, reason = retrain_due(len(data), min_new=args.min_new, max_days=args.max_days)
        print(reason)
        if not due:
            return
    folder = train(
        data, scaler_type=args.scaler, outlier_method=args.outlier,
        use_log=args.use_log, algorithm=args.algorithm, allow_ineligible=args.allow_ineligible,
    )
    print(folder)
    print((folder / 'metadata.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
