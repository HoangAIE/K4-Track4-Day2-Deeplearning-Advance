# Original User Request

## 2026-10-03T13:46:44Z

Thực hiện liên hoàn Bước 1 (So sánh backbone) và Bước 2 (Khảo sát công thức huấn luyện) trên tập dữ liệu DeepWeeds theo đúng quy chuẩn thực nghiệm nghiêm ngặt của Lab Day 2: cố định split fold 0, không rò rỉ tập test, đo đạc trade-off khách quan và phân tích có bằng chứng số liệu.

Working directory: d:/K4-Track4-Day2-Deeplearning-Advance
Integrity mode: development

## Requirements

### R1. Bước 1 — Huấn luyện & So sánh ≥ 5 Backbone (Công thức nền T00)
- Huấn luyện tuần tự ít nhất 5 backbone bao phủ 4 họ kiến trúc bắt buộc:
  1. Họ ResNet (mốc chuẩn): `resnet50` (exp_id: `B01`)
  2. Họ ConvNeXt / ResNeXt: `convnext_tiny` (exp_id: `B02`)
  3. Transformer: `swin_tiny_patch4_window7_224` hoặc `vit_small_patch16_224` (exp_id: `B03`)
  4. Mạng nhẹ: `efficientnet_b0` (exp_id: `B04`) và `mobilenetv3_large_100` (exp_id: `B05`)
- Dùng chung công thức nền T00: AdamW, LR backbone 1e-4, LR head 1e-3, weight decay 0.05 (bỏ qua norm và bias), Warmup 1 epoch + Cosine Annealing, Cross-Entropy loss, AMP on, batch size 64 (giảm nếu OOM), 12 epochs, seed = 0.
- Đo Params (M), GMACs, Val Macro-F1, Val Top-1 Acc, thời gian train/epoch, độ trễ suy luận batch 1 (ms).
- Lưu đồ thị `curves/B0x_<ten_backbone>.png` và ghi số liệu vào sheet `Backbones` của `results.xlsx`.
- Dựa trên trade-off định lượng giữa Macro-F1 vs Latency/Params, chọn 1 backbone tối ưu nhất để tiếp tục ở Bước 2.

### R2. Bước 2 — Khảo sát Công thức Huấn luyện (Ablation trên Backbone đã chọn)
- Lấy backbone đã chọn từ Bước 1 làm mốc chuẩn cố định.
- Khảo sát ít nhất 3 trục trong các trục sau (mỗi trục thử ít nhất 2 giá trị, có ít nhất một trục là Loss hoặc Augmentation):
  - **Trục A (Khởi tạo):** `init="scratch"` vs `init="frozen"` vs `init="finetune"` (chuẩn T00).
  - **Trục B (Augmentation):** `aug="basic"` vs `aug="color"` / `aug="trivial"` hoặc kết hợp `mix="cutmix"`.
  - **Trục C (Hàm loss):** `loss="ce"` vs `loss="ls"` (label smoothing $\epsilon=0.1$) vs `loss="focal"` ($\gamma=2.0$).
  - **Trục D (Cân bằng mẫu):** `sampler=None` vs `sampler="balanced"`.
  - **Trục F (Chính quy hóa):** `ema_decay=None` vs `ema_decay=0.999`.
- Tuân thủ nghiêm ngặt **Nguyên tắc 1** (mỗi lần chạy chỉ thay đổi đúng một yếu tố so với T00).
- Thử nghiệm **ít nhất một kết hợp** các yếu tố tốt nhất để đánh giá hiệu ứng cộng dồn hay triệt tiêu.
- Lưu đồ thị huấn luyện cho từng thí nghiệm: `curves/T0x_<mota>.png`.
- Ghi kết quả vào sheet `Training` của `results.xlsx`: ghi rõ trục thay đổi (A–G), điểm khác biệt so với T00, seed, Val Macro-F1, Val Top-1, và $\Delta$ so với T00.

### R3. Phân tích Khoa học và So sánh với Nhiễu
- So sánh mức cải thiện $\Delta$ với mức nhiễu ước lượng (std qua seed $\approx 0.1 - 0.3$ điểm). Nếu $\Delta < \text{std}$, đánh giá là "không phân biệt được".
- Trả lời các câu hỏi thực nghiệm cốt lõi:
  - Khởi tạo scratch trên DeepWeeds có hiệu quả không? Đóng băng backbone có đủ tốt không?
  - Augmentation hay CutMix giúp tăng độ tổng quát hóa hay làm chậm hội tụ?
  - Loss nào xử lý lớp `Negative` áp đảo tốt nhất và cải thiện F1 các lớp hiếm?
  - Hiệu ứng kết hợp có cộng dồn tuyến tính không?
- Xác định công thức tối ưu nhất (Best Recipe) làm tiền đề cho Bước 3 (Kỹ thuật suy luận).

### R4. Tính toàn vẹn Dữ liệu & Môi trường Thực thi
- Dữ liệu DeepWeeds tại `data/images` và `data/labels` (fold 0).
- Tuân thủ quy tắc S1–S6: Mọi lựa chọn mô hình và siêu tham số đều dựa trên `val_subset0.csv`. Tuyệt đối không dùng `test_subset0.csv`.
- Lưu giữ đầy đủ file cấu hình và checkpoint trong `runs/<exp_id>/seed0/`.

## Verification Resources
- `eval.py`: Công cụ tính toán Macro-F1 và Top-1 chuẩn của bài lab.
- `code/model.py`: Chứa định nghĩa kiến trúc và hàm đếm tham số, GMACs (`count_gmacs`, `count_params`).
- `code/benchmark.py`: Đo độ trễ suy luận có warmup và đồng bộ CUDA.
- `code/train.py`: Engine huấn luyện chính với khả năng cấu hình linh hoạt qua `Config`.

## Acceptance Criteria

### Sheet `Backbones` trong `results.xlsx`
- [ ] Chứa đủ ít nhất 5 backbone bao phủ 4 họ kiến trúc bắt buộc.
- [ ] Điền đủ các cột: exp_id (B01..B05), backbone, tag trọng số, #tham số (M), GMAC, độ phân giải (224), epoch (12), seed (0), macro-F1 val, top-1 val, thời gian train/epoch, độ trễ batch-1 (ms), ghi chú.

### Sheet `Training` trong `results.xlsx`
- [ ] Chứa dòng mốc T00 và các dòng thí nghiệm theo ít nhất 3 trục (mỗi trục $\ge 2$ giá trị, có ít nhất một trục là loss hoặc augmentation).
- [ ] Có ít nhất 1 dòng thí nghiệm kết hợp các yếu tố tốt nhất.
- [ ] Điền đủ các cột: exp_id (T01, T02...), backbone, trục thay đổi (A–G), khác T00 ở điểm nào, seed, macro-F1 val, top-1 val, $\Delta$ so với T00, ghi chú.

### Minh chứng và Artifacts
- [ ] Thư mục `curves/` có đầy đủ các file ảnh `curves/B0x_*.png` và `curves/T0x_*.png` rõ nét (có train/val loss và val macro-F1 theo epoch).
- [ ] Báo cáo / tài liệu phân tích chỉ ra rõ backbone được chọn từ Bước 1 và công thức tối ưu tìm được ở Bước 2.
- [ ] Không có bất kỳ truy cập hay rò rỉ nào từ tập test (`test_subset0.csv`).
