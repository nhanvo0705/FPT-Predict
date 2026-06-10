# Dự Án Đầu Tư Định Lượng: Dự Đoán Xu Hướng Cổ Phiếu FPT Bằng Thuật Toán XGBoost

Dự án này triển khai một hệ thống giao dịch thuật toán (Algorithmic Trading) hoàn chỉnh trên môi trường Python (Jupyter Notebook), sử dụng mô hình học máy **XGBoost Classifier** kết hợp cùng bộ công cụ backtest **VectorBT** và **QuantStats**. Mục tiêu của mô hình là dự đoán xu hướng tăng giá của cổ phiếu FPT tại thị trường chứng khoán Việt Nam.
---

Quy Trình 4 Bước Triển Khai Trong Notebook

## BƯỚC 1: CÀI ĐẶT THƯ VIỆN
Hệ thống sử dụng các thư viện bổ trợ mạnh mẽ cho khoa học dữ liệu và tài chính định lượng bao gồm: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost`, `vectorbt`, và `quantstats`.

## BƯỚC 2: THU THẬP & XỬ LÝ DỮ LIỆU
* Dữ liệu lịch sử giá OHLCV của mã cổ phiếu `FPT` từ ngày 01/01/2018 đến ngày 31/12/2025 được tải trực tiếp thông qua thư viện `vnstock`.
* Dữ liệu được chuẩn hóa múi giờ, kiểm tra và làm sạch hoàn toàn các lỗi dữ liệu trùng lặp hoặc giá trị thiếu (NaN).

## BƯỚC 3: FEATURE ENGINEERING 
Để mô hình có thể học được các cấu trúc và biến động, đã trích xuất 15 feature cho mô hình dự đoán:
* **Biến Mục Tiêu (`target`):** Nhãn nhị phân ($0$ hoặc $1$) đại diện cho việc giá đóng cửa có sinh lời vượt ngưỡng **0.15%** (bù đắp hoàn toàn phí giao dịch thực tế tại Việt Nam) sau 5 phiên kế tiếp hay không.
* **Nhóm Độ Biến Động:** Độ biến động lịch sử 20 phiên (`realized_vol_20d`), Biến động nâng cao Garman-Klass (`gk_vol_20d`), Tỷ lệ biến động ngắn/dài hạn (`vol_ratio_sl`), và Chỉ số ATR chuẩn hóa (`atr_14d`).
* **Nhóm Động Lượng & Xu Hướng:** Điểm chuẩn hóa `zscore_20`, Chỉ số sức mạnh tương đối `rsi_14`, Chỉ số kênh hàng hóa `cci_20`, Sức mạnh xu hướng tích hợp `trend_strength` (ADX 14 $\times$ Độ dốc MA20).
* **Nhóm Cấu Trúc Giá:** Phần trăm băng Bollinger (`bb_percent`), Khoảng cách tương đối tới đỉnh 52 tuần (`dist_52w_high`).
* **Nhóm Lợi Nhuận & Khoảng Cách MA:** Tỷ suất sinh lời bình quân kỳ hạn (`ret_5d`, `ret_20d`), Khoảng cách tương đối từ giá tới các đường xu hướng (`ma_dist_5d`, `ma_dist_20d`, `ma_dist_200d`).


### BƯỚC 4: XÂY DỰNG MÔ HÌNH DỰ ĐOÁN & BACKTEST
* **Cross-Validation Chống "Look-Ahead Bias":** Sử dụng phương pháp chia dữ liệu chuỗi thời gian `TimeSeriesSplit` với 5 Folds (gap=5) để bảo đảm mô hình không "học trước" dữ liệu tương lai.
* **Huấn Luyện XGBoost:** Mô hình cấu hình tham số tối ưu (`n_estimators=300`, `learning_rate=0.03`, `max_depth=4`, `reg_alpha=0.5`, `reg_lambda=1.5`) kết hợp cơ chế dừng sớm (`early_stopping_rounds=20`) trên tập Validation (lấy 10% cuối của tệp huấn luyện). Tín hiệu mua được kích hoạt khi xác suất tăng giá vượt ngưỡng `0.54`. 
* **Mô Phỏng Giao Dịch Thực Tế (VectorBT):** Chiến lược giả lập giao dịch mua với số vốn ban đầu là 100.000.000 VND, áp mức phí/thuế cố định 0.15% cho mỗi lệnh. Hệ thống tích hợp hàm mô phỏng lệnh đóng theo các lý do: Chạm chặn lỗ (Stop Loss - SL), Chạm mục tiêu (Take Profit - TP), hoặc hết thời gian nắm giữ tối đa. Đồng thời bộ lọc thị trường với chỉ báo MA50 và MA200 chỉ cho phép hệ thống vào lệnh với điều kiện ` giá ` > MA50 và MA200 để xác nhận thị trường uptrend
* **Đánh Giá Hiệu Suất (QuantStats):** Thiết lập tần suất đồng bộ theo thị trường Việt Nam. Trích xuất đầy đủ các thông số  để đánh giá rủi ro và lợi nhuận của hệ thống: Tỷ suất sinh lời lũy tiến (Cumulative Return), Sharpe Ratio, Sortino Ratio, và Mức sụt giảm tài sản lớn nhất (Max Drawdown).
* **Hiệu suất chiến lược:** Mô hình đạt tổng lợi nhuận 313.82%. Dù lợi nhuận tuyệt đối thấp hơn chiến lược Mua & Nắm giữ (Buy & Hold) cổ phiếu FPT (393.47%), hệ thống lại tối ưu vượt trội về hiệu suất quản trị rủi ro với chỉ số Sharpe 1.54 (vs. 1.17) và Sortino 2.45 (vs. 1.78). Đặc biệt, mức sụt giảm tài sản lớn nhất (Max Drawdown) được giữ ở mức rất an toàn là -20.71% so với mức sập sâu -34.63% của Benchmark
