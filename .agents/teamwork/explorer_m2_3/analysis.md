# Phân Tích Chiến Lược Thiết Kế Công Thức Kết Hợp (T_combo) & Schema Bảng Tính results.xlsx (Sheet Training)

**Tác giả**: Explorer 3 (`explorer_m2_3`)  
**Nhiệm vụ**: Milestone 2 — M2 Combined Recipe Design & Excel Training Sheet Schema  
**Dự án**: DeepWeeds Deep Learning Day 2 Lab  
**Ngày thực hiện**: 2026-10-03  
**Chế độ**: Read-only Investigation (Không sửa trực tiếp mã nguồn, tuân thủ nghiêm ngặt Zero Test Leakage)

---

## Tóm Tắt Điều Hành (Executive Summary)

Báo cáo này cung cấp bản thiết kế chiến lược hoàn chỉnh cho hai nhiệm vụ trọng tâm của Milestone 2:
1. **Thiết kế công thức huấn luyện kết hợp (`T_combo`) trên backbone tối ưu `convnext_tiny`**:
   - Phân tích cơ chế toán học và tương tác đa biến giữa các trục A (Khởi tạo), B (Augmentation), C (Loss function), D (Lấy mẫu cân bằng) và F (Chính quy hóa EMA).
   - Xác định chính xác các điểm **triệt tiêu tiêu cực (negative interference)**: Xung đột giữa Label Smoothing và CutMix (gây hiện tượng over-smoothing / target dilution), xung đột giữa Balanced Sampler và Focal Loss (gây compound penalty lên lớp đa số Negative, gia tăng phương sai gradient và nguy cơ overfit lớp hiếm).
   - Thiết lập công thức kết hợp chủ đạo **`T09` (`T_combo_primary`)**: `convnext_tiny` + `init="finetune"` + `aug="color"` (lighting invariance) + `loss="focal"` ($\gamma=2.0$, hạ trọng số background dễ phân loại) + `sampler=None` + `ema_decay=0.999` (bộ lọc thông thấp khử dao động tham số).
   - Thiết lập công thức kết hợp bổ trợ **`T10` (`T_combo_cutmix`)**: Thay `aug="color"` bằng `mix="cutmix"`, loại trừ hoàn toàn Label Smoothing để tránh triệt tiêu.
   - Phát hiện lỗ hổng logic tiềm ẩn trong `code/train.py` liên quan đến việc lưu trọng số EMA tại checkpoint.
2. **Quy chuẩn hóa Schema và Cơ Chế Cập Nhật Sheet `Training` trong `results.xlsx`**:
   - Định nghĩa chính xác 9 cột theo chuẩn đề bài: `["exp_id", "backbone", "trục thay đổi (A–G)", "khác T00 ở điểm nào", "seed", "macro-F1 val", "top-1 val", "Δ so với T00", "ghi chú"]`.
   - Thiết lập vị trí hàng 2 là dòng mốc chuẩn `T00` kế thừa nguyên bản từ kết quả `B02` của Milestone 1 (`convnext_tiny`, Val Macro-F1 = `0.9604`, Top-1 = `0.9697`, $\Delta = 0.0000$).
   - Công thức tính Delta: $\Delta = \text{Macro-F1}_{\text{exp}} - \text{Macro-F1}_{\text{T00}}$, định dạng số `+0.0000;-0.0000;0.0000` với quy tắc đối chiếu ngưỡng nhiễu seed $\sigma_{\text{seed}} \approx 0.0030$ (Slide p. 59).
   - Cung cấp mã nguồn Python hoàn chỉnh, sẵn sàng tích hợp vào `code/update_excel.py` để Worker agent thực thi mượt mà.

---

## 1. Thiết Kế Công Thức Kết Hợp (T_combo) & Phân Tích Tương Tác Giữa Các Yếu Tố

### 1.1 Điểm Tựa Thực Nghiệm: Backbone `convnext_tiny` & Mốc Chuẩn T00

Từ kết quả Milestone 1 (xác nhận trong `worker_m1/handoff.md` và `runs/B02/seed0/summary.json`):
- **Backbone chiến thắng**: `convnext_tiny` (`in12k_ft_in1k`, 27.827M tham số, 4.455 GMACs).
- **Hiệu năng nền T00**:
  - Val Macro-F1: **96.04%** (`0.9604`)
  - Val Top-1 Accuracy: **96.97%** (`0.9697`)
  - Thời gian huấn luyện: ~**79.11 giây/epoch** (tổng thời lượng 12 epochs ~15.8 phút trên GPU RTX 5060 Ti).
  - Độ trễ suy luận batch-1: **11.44 ms** (nhanh nhất trong 5 backbone khảo sát).
- **Cấu hình T00 cố định**:
  - Trình tối ưu: AdamW ($\text{lr}_{\text{backbone}} = 10^{-4}$, $\text{lr}_{\text{head}} = 10^{-3}$, $\text{weight\_decay} = 0.05$ bỏ qua norm & bias).
  - Lịch học: 1 epoch linear warmup + 11 epochs cosine decay về $10^{-4}$.
  - Hàm mất mát: Standard Cross Entropy (`loss="ce"`).
  - Augmentation: `basic` (`RandomResizedCrop(224, scale=(0.8, 1.0))` + `RandomHorizontalFlip` + `Normalize`).
  - Lấy mẫu: Tự nhiên (`sampler=None`, giữ nguyên tỷ lệ tập train: ~52% Negatives, ~6% mỗi loài cỏ).
  - Chính quy hóa: Không EMA (`ema_decay=None`), `drop_rate=0.0`.
  - Precision: AMP GradScaler trên CUDA, `batch_size = 64`, `epochs = 12`, `seed = 0`.

---

### 1.2 Danh Mục Các Yếu Tố Khảo Sát Trong Milestone 2

Theo phân công giữa các Explorer:
- **Explorer 1 (`explorer_m2_1`)**:
  - **Trục A (Khởi tạo)**:
    - `T01`: `init="scratch"` (khởi tạo ngẫu nhiên từ đầu, không trọng số pretrained).
    - `T02`: `init="frozen"` (đóng băng backbone feature extractor, chỉ huấn luyện classifier head).
  - **Trục B (Augmentation)**:
    - `T03`: `aug="trivial"` (hoặc `aug="color"`, kiểm tra biến đổi hình học & màu sắc).
    - `T04`: `mix="cutmix"` (cắt dán patch giữa các ảnh, $\alpha=1.0$).
- **Explorer 2 (`explorer_m2_2`)**:
  - **Trục C (Hàm loss)**:
    - `T05`: `loss="ls"`, `label_smoothing=0.1` (làm mịn nhãn $\epsilon=0.1$).
    - `T06`: `loss="focal"`, `focal_gamma=2.0` (tập trung mẫu khó, giảm phạt mẫu dễ).
  - **Trục D (Cân bằng mẫu)**:
    - `T07`: `sampler="balanced"` (`WeightedRandomSampler` nghịch đảo tần suất lớp).
  - **Trục F (Chính quy hóa)**:
    - `T08`: `ema_decay=0.999` (Exponential Moving Average của trọng số mô hình).

---

### 1.3 Phân Tích Tương Tác Sâu: Hiệu Ứng Triệt Tiêu (Negative Interference) vs Cộng Dồn (Synergy)

Khi chuyển từ các thí nghiệm đơn biến (`T01`–`T08`) sang công thức kết hợp (`T_combo`), một lỗi phổ biến là gom tất cả các yếu tố có $\Delta > 0$ vào một cấu hình duy nhất. Trên thực tế, các kỹ thuật deep learning thường tương tác phi tuyến tính và có thể triệt tiêu lẫn nhau.

#### A. Triệt tiêu giữa Label Smoothing (`loss="ls"`) và CutMix (`mix="cutmix"`)
1. **Cơ chế toán học**:
   - CutMix nội suy tuyến tính nhãn của hai ảnh $A$ và $B$ theo tỷ lệ diện tích $\lambda$:
     $$\mathbf{y}_{\text{cutmix}} = \lambda \mathbf{y}_A + (1 - \lambda) \mathbf{y}_B$$
     Bản thân nhãn sau CutMix đã là một phân phối xác suất mềm (soft continuous target).
   - Label Smoothing biến đổi nhãn one-hot $k \in \{1 \dots K\}$ thành:
     $$q'_k = (1 - \epsilon) \mathbf{1}[k = y] + \frac{\epsilon}{K}$$
     Với $K=9$ và $\epsilon=0.1$, xác suất lớp đúng bị giảm từ $1.0$ xuống $0.9$, và $0.011$ được san sẻ cho 8 lớp còn lại.
2. **Hệ quả khi ghép đôi (Over-smoothing & Dilution)**:
   - Trong `code/train.py` (dòng 200), `mixed_loss` gọi:
     $$\mathcal{L} = \lambda \mathcal{L}_{\text{crit}}(\hat{y}, y_A) + (1-\lambda) \mathcal{L}_{\text{crit}}(\hat{y}, y_B)$$
   - Nếu $\mathcal{L}_{\text{crit}}$ là `LabelSmoothingCE(0.1)`, cả $y_A$ và $y_B$ tiếp tục bị phân tán $10\%$ xác suất sang toàn bộ 9 lớp.
   - Khi đó, mạng nhận được tín hiệu giám sát cực kỳ mờ nhạt (diluted gradient signal). Đặc biệt đối với ảnh thực địa DeepWeeds, nơi cỏ dại thường có diện tích nhỏ trên nền đất/cỏ khô, việc vừa cắt dán (mất một phần tán lá) vừa làm nhòe nhãn khiến mô hình mất khả năng phân biệt ranh giới tinh vi giữa các loài cỏ tương đồng (ví dụ: Chinee Apple vs Snake Weed).
3. **Quyết định thiết kế**: **KHÔNG kết hợp đồng thời Label Smoothing ($\epsilon=0.1$) và CutMix trong cùng một công thức kết hợp**.

#### B. Triệt tiêu giữa Balanced Sampler (`sampler="balanced"`) và Focal Loss (`loss="focal"`)
1. **Cơ chế toán học**:
   - `sampler="balanced"` can thiệp ở **tầng dữ liệu (data-level)**: Mỗi batch lấy mẫu đồng đều giữa 9 lớp ($1/9 \approx 11.1\%$ mỗi lớp), giảm tỷ lệ xuất hiện của lớp đa số Negative từ $52.0\%$ xuống $11.1\%$ (giảm ~4.7 lần), đồng thời tăng tần suất các lớp hiếm từ $6\%$ lên $11.1\%$ (tăng ~1.9 lần).
   - `loss="focal"` can thiệp ở **tầng tổn thất (loss-level)**: Nhân hàm mất mát với trọng số giảm dần theo độ tự tin:
     $$\text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t), \quad \gamma = 2.0$$
     Với các mẫu Negative dễ phân loại ($p_t \approx 0.95$), trọng số mất mát bị thu nhỏ $(1 - 0.95)^2 = 0.0025$ (giảm 400 lần).
2. **Hệ quả khi ghép đôi (Compound Majority Penalty & Rare Class Overfitting)**:
   - Lớp Negative trong DeepWeeds không phải là một loài đơn lẻ, mà đại diện cho toàn bộ bối cảnh không có cỏ dại: đất đỏ, sỏi đá, phân gia súc, cành khô, bóng râm, cỏ bản địa vô hại. Nó đòi hỏi không gian biểu diễn rất rộng.
   - Khi vừa cắt giảm 4.7 lần số lượng ảnh Negative trong batch vừa triệt tiêu 400 lần gradient của các mẫu Negative có độ tự tin cao, mô hình bị "bỏ đói" bối cảnh nền. Hệ quả nhãn tiền: mô hình trở nên quá nhạy cảm (hyper-sensitive), phát hiện cỏ giả tràn lan (False Positives tăng vọt), làm sụt giảm nghiêm trọng Precision của các loài cỏ.
   - Đồng thời, 8 loài cỏ hiếm chỉ có ~600 ảnh mỗi loài trong tập train. Việc oversample liên tục trong 12 epoch với replacement kết hợp với việc Focal Loss tập trung vào mẫu khó sẽ khiến mô hình bị quá khớp (overfit) vào vài chục ảnh nhiễu/gắn nhãn sai của các lớp hiếm.
3. **Quyết định thiết kế**:
   - Giữ nguyên phân bố tự nhiên (`sampler=None`), để Focal Loss tự động điều tiết động (dynamic down-weighting) trên các mẫu Negative dễ. Đây là hướng tiếp cận tinh tế, bảo toàn đầy đủ phương sai nền tảng của lớp Negative mà vẫn giải phóng áp lực chi phối gradient.

#### C. Cộng dồn tích cực (Synergy) giữa Data Augmentation (`aug="color"`) và EMA (`ema_decay=0.999`)
1. **Cơ chế toán học**:
   - `aug="color"` (ColorJitter: 0.2 brightness, 0.2 contrast, 0.2 saturation, 0.1 hue) mô phỏng biến thiên ánh sáng mặt trời tự nhiên tại Queensland (mùa khô, nắng gắt, bóng mây). Nó giúp mô hình không bị lệ thuộc vào sắc độ ánh sáng tức thời. Tuy nhiên, ColorJitter làm tăng phương sai gradient giữa các mini-batch liên tiếp.
   - EMA đóng vai trò như một bộ lọc thông thấp (low-pass filter) trên quỹ đạo tham số:
     $$\mathbf{W}_{\text{EMA}} \leftarrow 0.999 \cdot \mathbf{W}_{\text{EMA}} + 0.001 \cdot \mathbf{W}_t$$
     Nó trung bình hóa các dao động trọng số ngẫu nhiên do data augmentation tạo ra, giúp mô hình hội tụ vào vùng cực tiểu phẳng (flat minimum).
2. **Hệ quả**: **Cộng dồn hoàn hảo**. Augmentation mở rộng biên tổng quát hóa, trong khi EMA đảm bảo độ ổn định và giảm thiểu phương sai suy luận.

#### D. Quyết định về Trục A (Khởi tạo)
- `init="scratch"`: Huấn luyện mạng ConvNeXt 28M tham số từ đầu trên 10,501 ảnh trong 12 epochs chắc chắn cho kết quả rất thấp (~40–60% Macro-F1).
- `init="frozen"`: Đóng băng backbone chỉ cho phép học linear head, không thể thích ứng các bộ lọc tích chập sâu với hình thái gân lá và gai của thực vật.
- Vì vậy, công thức kết hợp bắt buộc phải kế thừa `init="finetune"` với differential learning rates ($10^{-4}$ cho backbone, $10^{-3}$ cho classifier head).

---

### 1.4 Thiết Kế Chi Tiết Hai Công Thức Kết Hợp Ứng Viên

Dựa trên phân tích tương tác trên, chúng tôi thiết kế 2 công thức kết hợp rõ ràng:

#### Công thức Kết hợp Chủ đạo: `T09` (`T_combo_primary` — Orthogonal Synergy Recipe)
- **Mã thí nghiệm**: `T09`
- **Mục tiêu**: Kết hợp 3 yếu tố độc lập trên 3 trục khác nhau (Augmentation + Loss + Regularization) mà không hề có sự chồng chéo cơ chế:
  - **Trục A (Khởi tạo)**: `init="finetune"` (chuẩn T00)
  - **Trục B (Augmentation)**: `aug="color"` (tăng khả năng chịu biến thiên ánh sáng, bảo toàn hình thái vi mô của lá cây)
  - **Trục C (Loss function)**: `loss="focal"`, `focal_gamma=2.0` (tự động triệt tiêu gradient của 52% mẫu Negative dễ mà không làm méo mó phân bố batch)
  - **Trục D (Lấy mẫu)**: `sampler=None` (giữ phân bố tự nhiên)
  - **Trục F (Chính quy hóa)**: `ema_decay=0.999` (bộ lọc phẳng hóa trọng số, khắc phục dao động do color augmentation)
- **Lệnh thực thi chính xác (PowerShell)**:
  ```powershell
  $env:PYTHONUTF8=1; python code/train.py --exp_id T09 --seed 0 --fold 0 --set backbone=convnext_tiny aug=color loss=focal focal_gamma=2.0 ema_decay=0.999 num_workers=0
  ```
- **Kỳ vọng định lượng**:
  - T00 đạt 96.04% Macro-F1.
  - Dự kiến `T09` cải thiện thêm $+0.40\% \to +0.70\%$ Macro-F1, đạt khoảng **96.44% – 96.74%**, vượt ngưỡng nhiễu seed $\sigma \approx 0.30\%$.
- **Đường dẫn artifact**:
  - Checkpoint: `runs/T09/seed0/best_checkpoint.pt`
  - Đồ thị huấn luyện: `curves/T09_convnext_tiny.png` (và alias `curves/T09_combo_primary.png`)
  - Log & dự đoán: `runs/T09/seed0/summary.json`, `history.csv`, `val_logits.npy`

#### Công thức Kết hợp Bổ trợ: `T10` (`T_combo_cutmix` — Spatial Regularization Recipe)
- **Mã thí nghiệm**: `T10`
- **Mục tiêu**: Kiểm tra hiệu ứng cộng dồn khi sử dụng phép tăng cường không gian mạnh CutMix thay vì ColorJitter:
  - **Trục A (Khởi tạo)**: `init="finetune"`
  - **Trục B (Augmentation)**: `mix="cutmix"`, `mix_alpha=1.0` (tăng cường không gian)
  - **Trục C (Loss function)**: `loss="focal"`, `focal_gamma=2.0` (áp dụng mixed loss trên hai mục tiêu)
  - **Trục D (Lấy mẫu)**: `sampler=None`
  - **Trục F (Chính quy hóa)**: `ema_decay=0.999`
  - *Lưu ý*: Tuyệt đối không thêm `loss="ls"` để tránh hiện tượng over-smoothing.
- **Lệnh thực thi chính xác (PowerShell)**:
  ```powershell
  $env:PYTHONUTF8=1; python code/train.py --exp_id T10 --seed 0 --fold 0 --set backbone=convnext_tiny mix=cutmix loss=focal focal_gamma=2.0 ema_decay=0.999 num_workers=0
  ```
- **Đường dẫn artifact**:
  - Checkpoint: `runs/T10/seed0/best_checkpoint.pt`
  - Đồ thị huấn luyện: `curves/T10_convnext_tiny.png` (và alias `curves/T10_combo_cutmix.png`)

---

### 1.5 Phát Hiện Quan Trọng Về Kiểm Định Code: Lỗi Lưu Checkpoint EMA Trong `code/train.py`

Trong quá trình điều tra mã nguồn `code/train.py`, chúng tôi phát hiện một chi tiết kỹ thuật cực kỳ quan trọng ảnh hưởng trực tiếp đến hiệu quả của `ema_decay`:
- **Vị trí**: `code/train.py`, các dòng 400–442:
  ```python
  # Đánh giá trên VAL
  if ema is not None:
      ema.apply_shadow(model)

  val_fnames, val_true, val_logits, val_loss = evaluate(model, val_loader, val_criterion, device)

  if ema is not None:
      ema.restore(model)   # <--- DÒNG 406: Khôi phục trọng số thường!
  ...
  if val_macro_f1 > best_macro_f1:
      best_macro_f1 = val_macro_f1
      best_epoch = epoch
      torch.save({
          "epoch": epoch,
          "model_state_dict": model.state_dict(),   # <--- DÒNG 437: Lưu trọng số thường, KHÔNG PHẢI EMA!
          "macro_f1": val_macro_f1,
          "top1": val_top1,
          "cfg": dataclasses.asdict(cfg),
      }, best_ckpt_path)
  ```
- **Hệ quả quan sát được**:
  1. Khi chạy validation, mô hình dùng trọng số `shadow` của EMA để tính `val_macro_f1`.
  2. Ngay sau khi hàm `evaluate` hoàn thành (dòng 406), `ema.restore(model)` được gọi, đưa các tham số của mạng trở lại trọng số huấn luyện chưa được làm mịn!
  3. Khi phát hiện `val_macro_f1 > best_macro_f1` ở dòng 432, câu lệnh `torch.save` lại lưu `model.state_dict()` (tức trọng số thường, chứ không phải trọng số EMA đã tạo ra kết quả xuất sắc đó)!
  4. Đến khi nạp checkpoint ở Bước 4 (dòng 465) để suy luận hoặc test, mô hình chỉ nạp lại trọng số thường, làm mất đi lợi thế của EMA!
- **Khuyến nghị hành động cho Worker Agent**:
  Khi chuyển sang pha thực thi code, cần áp dụng chỉnh sửa nhỏ nhưng then chốt: Chuyển lệnh `ema.restore(model)` xuống sau khối `torch.save`, HOẶC nếu `ema is not None` thì nạp `ema.apply_shadow(model)` trước khi gọi `model.state_dict()` để lưu đúng trọng số EMA tối ưu vào `best_checkpoint.pt`.

---

## 2. Quy Chuẩn Schema & Cơ Chế Cập Nhật Bảng Tính `results.xlsx` (Sheet `Training`)

### 2.1 Quy Cách 9 Cột Chuẩn Tuyệt Đối Của Sheet `Training`

Theo yêu cầu nghiêm ngặt từ USER_REQUEST và RUBRIC.md mục E:
Bảng `Training` phải chứa đúng 9 cột sau (không thừa, không thiếu, đúng tiếng Việt có dấu):

| Cột | Tên Cột Chuẩn | Kiểu Dữ Liệu | Căn Lề | Định Dạng Số (Number Format) | Mô Tả Ý Nghĩa |
|:---:|:---|:---:|:---:|:---:|:---|
| **A (1)** | `exp_id` | Chuỗi ký tự | Giữa | Text | Mã thí nghiệm chuẩn (`T00`, `T01` ... `T10`) |
| **B (2)** | `backbone` | Chuỗi ký tự | Trái | Text | Tên backbone kiến trúc (`convnext_tiny`) |
| **C (3)** | `trục thay đổi (A–G)` | Chuỗi ký tự | Giữa | Text | Tên trục phân loại theo GUIDE mục 3 (`Mốc chuẩn`, `Trục A`, `Trục B`, `Trục C`, `Trục D`, `Trục F`, `Kết hợp`) |
| **D (4)** | `khác T00 ở điểm nào` | Chuỗi ký tự | Trái | Text | Mô tả chính xác điểm khác biệt duy nhất so với T00 |
| **E (5)** | `seed` | Số nguyên | Giữa | General (`0`) | Random seed thực nghiệm (`0`) |
| **F (6)** | `macro-F1 val` | Số thực | Phải | `0.0000` | Chỉ số chính Macro-F1 trên `val_subset0.csv` |
| **G (7)** | `top-1 val` | Số thực | Phải | `0.0000` | Chỉ số phụ Top-1 Accuracy trên val |
| **H (8)** | `Δ so với T00` | Số thực / Formula | Phải | `+0.0000;-0.0000;0.0000` | Mức cải thiện hoặc suy giảm so với T00 |
| **I (9)** | `ghi chú` | Chuỗi ký tự | Trái | Text | Nhận xét phân tích, so sánh với ngưỡng nhiễu |

---

### 2.2 Vị Trí & Dữ Liệu Dòng Mốc Chuẩn T00 Ở Đầu Bảng

Dòng 2 (ngay dưới dòng tiêu đề dòng 1) **bắt buộc phải là dòng mốc T00**. Dữ liệu được kế thừa chính thức từ thực nghiệm `B02` của Milestone 1:

```python
T00_RECORD = {
    "exp_id": "T00",
    "backbone": "convnext_tiny",
    "trục thay đổi (A–G)": "Mốc chuẩn",
    "khác T00 ở điểm nào": "Công thức nền T00 chuẩn (finetune, basic aug, CE loss, sampler=None, ema=None)",
    "seed": 0,
    "macro-F1 val": 0.9604,
    "top-1 val": 0.9697,
    "Δ so với T00": 0.0000,
    "ghi chú": "Mốc chuẩn T00 (kế thừa từ B02: AdamW, lr=1e-4/1e-3, CE, 12 ep; F1=96.04%)",
}
```

---

### 2.3 Công Thức Tính Delta ($\Delta$) & Quy Tắc So Sánh Nhiễu

1. **Công thức định lượng**:
   $$\Delta = \text{Macro-F1}_{\text{exp}} - \text{Macro-F1}_{\text{T00}}$$
   với $\text{Macro-F1}_{\text{T00}} = 0.9604$.
2. **Định dạng hiển thị trong Excel**:
   - Sử dụng định dạng tùy chỉnh `+0.0000;-0.0000;0.0000`.
   - Kết quả: Các giá trị dương hiển thị kèm dấu cộng rõ ràng (ví dụ: `+0.0046`), giá trị âm kèm dấu trừ (ví dụ: `-0.4374`), và giá trị 0 hiển thị `0.0000`.
   - Trong Excel formula: Cell cột H tại hàng $r$ có thể gán giá trị số thực tính trước bởi Python (để đảm bảo tương thích tuyệt đối khi đọc qua `pandas` và `openpyxl` với `data_only=True`), hoặc gán công thức `=F{r}-$F$2`. Để đạt độ tin cậy tối đa, hàm cập nhật sẽ tính sẵn số thực `round(val_f1 - t00_f1, 4)` và gán định dạng số hiển thị tương ứng.
3. **Quy tắc khoa học đối chiếu ngưỡng nhiễu (Slide Day 2 p. 59)**:
   - Ngưỡng lệch chuẩn do seed ngẫu nhiên trên mô hình phân loại: $\sigma_{\text{seed}} \approx 0.0010 - 0.0030$ (0.1% – 0.3%).
   - Nếu $|\Delta| < 0.0030$: Ghi chú phải kết luận *"Không phân biệt được so với nhiễu seed (|Δ| < 0.0030)"*. Không được tự nhận là tốt hơn hay kém hơn.
   - Nếu $\Delta \ge +0.0030$: Ghi chú kết luận *"Cải thiện có ý nghĩa thống kê (Δ ≥ +0.0030)"*.
   - Nếu $\Delta \le -0.0030$: Ghi chú kết luận *"Suy giảm hiệu năng rõ rệt (Δ ≤ -0.0030)"*.

---

### 2.4 Mã Nguồn Hoàn Chỉnh Cập Nhật Sheet `Training` Cho `code/update_excel.py`

Dưới đây là phần mở rộng mã nguồn Python chuẩn, sẵn sàng để Worker agent đưa vào `code/update_excel.py`:

```python
# ==============================================================================
# BỔ SUNG CHO code/update_excel.py: QUẢN LÝ SHEET TRAINING
# ==============================================================================

DELTA_FORMAT = "+0.0000;-0.0000;0.0000"

def upsert_training_row(excel_path: Union[str, Path], record: Dict[str, Any]) -> None:
    """Thêm mới hoặc cập nhật một dòng thí nghiệm vào sheet 'Training' của results.xlsx.
    
    Đảm bảo:
    - Đúng 9 cột quy định.
    - Dòng T00 luôn được đặt hoặc duy trì ở vị trí hàng 2.
    - Tự động tính Δ so với T00 nếu chưa có.
    - Định dạng số chuẩn 0.0000 và +0.0000;-0.0000;0.0000.
    - Tự động tô màu nổi bật #E8F8F5 cho dòng đạt macro-F1 cao nhất.
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        init_results_workbook(excel_path)

    wb = openpyxl.load_workbook(excel_path)
    if "Training" not in wb.sheetnames:
        ws = wb.create_sheet("Training")
        format_header_row(ws, TRAINING_COLUMNS)
    else:
        ws = wb["Training"]

    exp_id = str(record.get("exp_id", "")).strip()
    if not exp_id:
        raise ValueError("record bắt buộc phải có trường 'exp_id'")

    # 1. Tìm hoặc xác định chỉ số hàng
    target_row = None
    t00_f1 = 0.9604  # Giá trị chuẩn từ B02

    # Tìm hàng T00 trước để lấy giá trị chuẩn chính xác từ bảng nếu có
    for r in range(2, ws.max_row + 1):
        if str(ws.cell(row=r, column=1).value or "").strip() == "T00":
            val = ws.cell(row=r, column=6).value
            if val is not None:
                try:
                    t00_f1 = float(val)
                except (ValueError, TypeError):
                    pass
            break

    # Nếu đang ghi T00, ưu tiên hàng 2
    if exp_id == "T00":
        target_row = 2
    else:
        for r in range(2, ws.max_row + 1):
            val = ws.cell(row=r, column=1).value
            if val is not None and str(val).strip() == exp_id:
                target_row = r
                break
        if target_row is None:
            # Chèn tiếp vào cuối (hoặc hàng 3 nếu bảng mới có T00)
            target_row = ws.max_row + 1 if ws.cell(row=ws.max_row, column=1).value is not None else ws.max_row
            if target_row < 2:
                target_row = 2

    # 2. Chuẩn bị giá trị
    macro_f1 = float(record.get("macro-F1 val", record.get("macro_f1", 0.0)))
    top1 = float(record.get("top-1 val", record.get("top1", 0.0)))
    
    if exp_id == "T00":
        delta = 0.0000
    else:
        delta = float(record.get("Δ so với T00", round(macro_f1 - t00_f1, 4)))

    row_values = [
        exp_id,
        record.get("backbone", "convnext_tiny"),
        record.get("trục thay đổi (A–G)", record.get("axis", "")),
        record.get("khác T00 ở điểm nào", record.get("diff", "")),
        int(record.get("seed", 0)),
        macro_f1,
        top1,
        delta,
        record.get("ghi chú", record.get("note", "")),
    ]

    # 3. Ghi vào ô và định dạng
    for col_idx, val in enumerate(row_values, start=1):
        cell = ws.cell(row=target_row, column=col_idx, value=val)
        cell.font = DATA_FONT
        cell.border = THIN_BORDER

        if col_idx in [1, 3, 5]:
            cell.alignment = Alignment(horizontal="center", vertical="center")
        elif col_idx in [6, 7]:
            cell.alignment = Alignment(horizontal="right", vertical="center")
            cell.number_format = "0.0000"
        elif col_idx == 8:
            cell.alignment = Alignment(horizontal="right", vertical="center")
            cell.number_format = DELTA_FORMAT
        else:
            cell.alignment = Alignment(horizontal="left", vertical="center")

    # 4. Tô màu nổi bật dòng tốt nhất trong sheet Training
    best_f1 = -1.0
    best_row_idx = None
    for r in range(2, ws.max_row + 1):
        f1_val = ws.cell(row=r, column=6).value
        try:
            f1_num = float(f1_val)
            if f1_num > best_f1:
                best_f1 = f1_num
                best_row_idx = r
        except (ValueError, TypeError):
            pass

    for r in range(2, ws.max_row + 1):
        is_best = (r == best_row_idx and best_f1 > 0)
        for c in range(1, len(TRAINING_COLUMNS) + 1):
            cell = ws.cell(row=r, column=c)
            cell.fill = BEST_ROW_FILL if is_best else PatternFill(fill_type=None)

    # 5. Tự động căn chỉnh độ rộng cột hợp lý
    col_max_widths = {1: 12, 2: 18, 3: 22, 4: 45, 5: 10, 6: 15, 7: 15, 8: 16, 9: 55}
    for col_idx in range(1, len(TRAINING_COLUMNS) + 1):
        col_letter = get_column_letter(col_idx)
        max_len = 0
        for r in range(1, ws.max_row + 1):
            val_str = str(ws.cell(row=r, column=col_idx).value or "")
            max_len = max(max_len, len(val_str))
        target_w = max(max_len + 4, col_max_widths.get(col_idx, 12))
        ws.column_dimensions[col_letter].width = min(target_w, 65)

    ws.freeze_panes = "A2"
    wb.save(excel_path)
    wb.close()


def populate_training_from_runs(excel_path: Union[str, Path] = "results.xlsx",
                                runs_dir: Union[str, Path] = "runs") -> None:
    """Quét toàn bộ kết quả thí nghiệm T00..T10 từ runs/ và cập nhật sheet 'Training'."""
    excel_path = Path(excel_path)
    runs_path = Path(runs_dir)

    experiment_metadata = {
        "T00": {
            "axis": "Mốc chuẩn",
            "diff": "Công thức nền T00 chuẩn (finetune, basic aug, CE loss, sampler=None, ema=None)",
            "note": "Mốc chuẩn T00 (kế thừa từ B02: AdamW, lr=1e-4/1e-3, CE, 12 ep; F1=96.04%)",
            "fallback_run": "B02",
        },
        "T01": {
            "axis": "Trục A (Khởi tạo)",
            "diff": "Khởi tạo ngẫu nhiên từ đầu (init='scratch', không dùng pretrained weights)",
            "note": "Khởi tạo scratch trên DeepWeeds; đánh giá khả năng học từ đầu với 10k ảnh",
        },
        "T02": {
            "axis": "Trục A (Khởi tạo)",
            "diff": "Đóng băng backbone (init='frozen', chỉ huấn luyện linear head)",
            "note": "Đóng băng backbone; đánh giá chất lượng đặc trưng zero-shot của ImageNet",
        },
        "T03": {
            "axis": "Trục B (Augmentation)",
            "diff": "Tăng cường dữ liệu nâng cao (aug='trivial')",
            "note": "TrivialAugmentWide; đánh giá tác động của biến dạng hình học/màu sắc mạnh",
        },
        "T04": {
            "axis": "Trục B (Augmentation)",
            "diff": "Trộn mẫu CutMix (mix='cutmix', mix_alpha=1.0)",
            "note": "CutMix patch; buộc mạng học các đặc trưng cục bộ phân tán",
        },
        "T05": {
            "axis": "Trục C (Hàm loss)",
            "diff": "Làm mịn nhãn (loss='ls', label_smoothing=0.1)",
            "note": "Label smoothing 0.1; chống tự tin thái quá, cải thiện hiệu chuẩn",
        },
        "T06": {
            "axis": "Trục C (Hàm loss)",
            "diff": "Focal Loss (loss='focal', focal_gamma=2.0)",
            "note": "Focal Loss; giảm trọng số các mẫu Negative dễ, tập trung vào mẫu khó",
        },
        "T07": {
            "axis": "Trục D (Cân bằng mẫu)",
            "diff": "Bộ lấy mẫu cân bằng lớp (sampler='balanced')",
            "note": "WeightedRandomSampler; cân bằng tần suất xuất hiện giữa các lớp",
        },
        "T08": {
            "axis": "Trục F (Chính quy hóa)",
            "diff": "Trọng số trung bình động (ema_decay=0.999)",
            "note": "EMA 0.999; làm phẳng vùng cực tiểu và giảm phương sai tham số",
        },
        "T09": {
            "axis": "Kết hợp (Combo Chính)",
            "diff": "Kết hợp tối ưu không triệt tiêu: aug='color' + loss='focal' + ema_decay=0.999",
            "note": "Best Recipe kết hợp các trục trực giao; cộng dồn lợi thế ổn định",
        },
        "T10": {
            "axis": "Kết hợp (Combo Phụ)",
            "diff": "Kết hợp CutMix: mix='cutmix' + loss='focal' + ema_decay=0.999",
            "note": "Ablation kết hợp không gian mạnh; loại bỏ Label Smoothing tránh over-smoothing",
        },
    }

    # Đảm bảo T00 được điền đầu tiên
    t00_meta = experiment_metadata["T00"]
    t00_sum = runs_path / "T00" / "seed0" / "summary.json"
    if not t00_sum.exists():
        t00_sum = runs_path / "B02" / "seed0" / "summary.json"

    if t00_sum.exists():
        with open(t00_sum, "r", encoding="utf-8") as f:
            t00_data = json.load(f)
        record_t00 = {
            "exp_id": "T00",
            "backbone": t00_data.get("backbone", "convnext_tiny"),
            "trục thay đổi (A–G)": t00_meta["axis"],
            "khác T00 ở điểm nào": t00_meta["diff"],
            "seed": t00_data.get("seed", 0),
            "macro-F1 val": t00_data.get("best_val_macro_f1", 0.9604),
            "top-1 val": t00_data.get("best_val_top1", 0.9697),
            "Δ so với T00": 0.0000,
            "ghi chú": t00_meta["note"],
        }
        upsert_training_row(excel_path, record_t00)
        print(f"Đã cập nhật T00 (Mốc chuẩn) vào {excel_path}")

    # Cập nhật các thí nghiệm T01..T10
    for exp_id in ["T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08", "T09", "T10"]:
        sum_p = runs_path / exp_id / "seed0" / "summary.json"
        if not sum_p.exists():
            continue

        with open(sum_p, "r", encoding="utf-8") as f:
            summary = json.load(f)

        meta = experiment_metadata.get(exp_id, {})
        val_f1 = summary.get("best_val_macro_f1", 0.0)
        t00_f1 = 0.9604
        delta = round(val_f1 - t00_f1, 4)

        # Đánh giá nhiễu tự động cho ghi chú
        base_note = meta.get("note", "")
        if abs(delta) < 0.0030:
            noise_assessment = f"|Δ|={abs(delta):.4f} < 0.0030 (không phân biệt được so với nhiễu seed)"
        elif delta >= 0.0030:
            noise_assessment = f"Δ={delta:+.4f} ≥ +0.0030 (cải thiện có ý nghĩa thống kê)"
        else:
            noise_assessment = f"Δ={delta:+.4f} ≤ -0.0030 (suy giảm hiệu năng rõ rệt)"

        full_note = f"{base_note}; {noise_assessment}"

        record = {
            "exp_id": exp_id,
            "backbone": summary.get("backbone", "convnext_tiny"),
            "trục thay đổi (A–G)": meta.get("axis", "Ablation"),
            "khác T00 ở điểm nào": meta.get("diff", ""),
            "seed": summary.get("seed", 0),
            "macro-F1 val": val_f1,
            "top-1 val": summary.get("best_val_top1", 0.0),
            "Δ so với T00": delta,
            "ghi chú": full_note,
        }
        upsert_training_row(excel_path, record)
        print(f"Đã cập nhật {exp_id} vào {excel_path}")
```

---

## 3. Bản Đồ Tổng Thể Kế Hoạch Thực Nghiệm Milestone 2

Bảng tổng hợp toàn bộ các thí nghiệm từ T00 đến T10, phục vụ việc điều phối và giám sát thực thi của Worker agent:

| Exp ID | Trục (A–G) | Cấu Hình Khác T00 | CLI Override Flags | Checkpoint Path | Curve Path |
|:---:|:---:|:---|:---|:---|:---|
| **T00** | Mốc chuẩn | Không đổi (chuẩn nền B02) | N/A (kế thừa B02) | `runs/B02/seed0/best_checkpoint.pt` | `curves/B02_convnext_tiny.png` |
| **T01** | Trục A | `init="scratch"` | `--set backbone=convnext_tiny init=scratch num_workers=0` | `runs/T01/seed0/best_checkpoint.pt` | `curves/T01_convnext_tiny.png` |
| **T02** | Trục A | `init="frozen"` | `--set backbone=convnext_tiny init=frozen num_workers=0` | `runs/T02/seed0/best_checkpoint.pt` | `curves/T02_convnext_tiny.png` |
| **T03** | Trục B | `aug="trivial"` | `--set backbone=convnext_tiny aug=trivial num_workers=0` | `runs/T03/seed0/best_checkpoint.pt` | `curves/T03_convnext_tiny.png` |
| **T04** | Trục B | `mix="cutmix"` | `--set backbone=convnext_tiny mix=cutmix num_workers=0` | `runs/T04/seed0/best_checkpoint.pt` | `curves/T04_convnext_tiny.png` |
| **T05** | Trục C | `loss="ls"`, $\epsilon=0.1$ | `--set backbone=convnext_tiny loss=ls label_smoothing=0.1 num_workers=0` | `runs/T05/seed0/best_checkpoint.pt` | `curves/T05_convnext_tiny.png` |
| **T06** | Trục C | `loss="focal"`, $\gamma=2.0$ | `--set backbone=convnext_tiny loss=focal focal_gamma=2.0 num_workers=0` | `runs/T06/seed0/best_checkpoint.pt` | `curves/T06_convnext_tiny.png` |
| **T07** | Trục D | `sampler="balanced"` | `--set backbone=convnext_tiny sampler=balanced num_workers=0` | `runs/T07/seed0/best_checkpoint.pt` | `curves/T07_convnext_tiny.png` |
| **T08** | Trục F | `ema_decay=0.999` | `--set backbone=convnext_tiny ema_decay=0.999 num_workers=0` | `runs/T08/seed0/best_checkpoint.pt` | `curves/T08_convnext_tiny.png` |
| **T09** | Combo Chính | `aug="color"` + `loss="focal"` + `ema=0.999` | `--set backbone=convnext_tiny aug=color loss=focal focal_gamma=2.0 ema_decay=0.999 num_workers=0` | `runs/T09/seed0/best_checkpoint.pt` | `curves/T09_convnext_tiny.png` |
| **T10** | Combo Phụ | `mix="cutmix"` + `loss="focal"` + `ema=0.999` | `--set backbone=convnext_tiny mix=cutmix loss=focal focal_gamma=2.0 ema_decay=0.999 num_workers=0` | `runs/T10/seed0/best_checkpoint.pt` | `curves/T10_convnext_tiny.png` |

---

## 4. Kết Luận & Khuyến Nghị

1. **Về Công thức Kết hợp `T_combo`**:
   - `T09` là công thức tối ưu khoa học nhất: không vấp phải bất kỳ rủi ro triệt tiêu nào (tránh hoàn toàn over-smoothing của LS+CutMix và double-penalty của Sampler+Focal).
   - T09 hội tụ 3 trục độc lập: Tăng cường tính bất biến ánh sáng (Color), giảm phạt mẫu dễ đa số (Focal), và ổn định trọng số hội tụ (EMA).
2. **Về Bảng tính `results.xlsx` (Sheet `Training`)**:
   - Schema 9 cột đã được cố định hoàn toàn và sẵn sàng cho việc ghi tự động.
   - Dòng T00 ở đầu bảng đảm bảo việc tính Delta trực quan và chính xác.
   - Logic đối chiếu ngưỡng nhiễu $\sigma_{\text{seed}} \approx 0.0030$ đáp ứng 100% tiêu chí chấm của RUBRIC.
3. **Về Mã nguồn Thực thi**:
   - Cần lưu ý Worker agent xử lý đoạn code EMA trong `code/train.py` (dòng 406 & dòng 437) để checkpoint lưu giữ đúng trọng số shadow tối ưu.
