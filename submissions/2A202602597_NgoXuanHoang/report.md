# Báo cáo Lab Day 2 — Phân loại cỏ dại DeepWeeds: backbone, công thức huấn luyện và suy luận

> Mọi con số dưới đây trích từ `results.xlsx` (7 sheet) và `eval_out/` (output của `eval.py`);
> đối chiếu chéo predictions ↔ xlsx ↔ log đã kiểm tra khớp 100% (script `code/fill_step5_gaps.py`,
> `crosscheck.py`). Quy ước ngôn ngữ: chênh lệch < 0,0030 (≈ nhiễu seed) được gọi là
> "không phân biệt được", không gọi là "tốt hơn".

## 1. Tóm tắt (10 dòng)

1. Bài toán: phân loại 9 lớp ảnh cỏ dại ngoài đồng DeepWeeds (17 509 ảnh 256×256), lớp `Negatives` chiếm ~52%.
2. Bước 1: so sánh 5 backbone cùng công thức nền T00 — `convnext_tiny` tốt nhất (val macro-F1 96,04%).
3. Bước 2: 9 ablation trên `convnext_tiny` (4 trục A/B/C/D/F + combo) — CutMix đơn lẻ tốt nhất (+1,15pt).
4. Bước 3: 8 phương pháp suy luận trên T04 — ensemble 3 model tốt nhất (+0,31pt, tốn 3x); model soup +0,19pt miễn phí.
5. Bước 4: chung kết F01 (CutMix + 1-view + temperature scaling), 3 seed, mở test đúng một lần/seed.
6. Kết quả test: **macro-F1 0,9734 ± 0,0029; top-1 0,9780 ± 0,0024**, hơn mốc T00 **+0,0092** (vượt std 0,0032).
7. Hai lớp khó nhất: Chinee Apple F1 0,964 (recall 0,966), Snake Weed F1 0,961 (recall 0,954).
8. Tự chấm `eval.py grade`: **18/20** (I1 7/7, I2 4/5, I3 4/4, I4a 0/1, I4b 1/1, I5 2/2).
9. Độ trễ batch-1 trên RTX 5060 Ti: 3,45 ms (FP32) — mọi cấu hình đều dưới xa ngân sách 100 ms.
10. Kết luận chính: pretrained bắt buộc; CutMix đóng góp lớn nhất trong công thức; suy luận miễn phí (soup, T-scaling) là lựa chọn robot.

## 2. Dữ liệu và thiết lập

**Dataset.** DeepWeeds: 17 509 ảnh RGB 256×256, 9 lớp (8 loài cỏ + `Negatives`).
Dùng **fold 0** nguyên bản: train 10 501 (59,97%) / val 3 501 (20,0%) / test 3 507 (20,03%).
Đã kiểm tra: giao các cặp tập rỗng, hợp đủ 17 509, mọi file tồn tại (`EDA`: `curves/eda_class_distribution.png`).
Phân bố lớp mất cân bằng mạnh: `Negatives` 9 106 ảnh (~52%), mỗi loài cỏ 1 009–1 125 ảnh.

**Chỉ số.** Chính: **macro-F1** (trung bình F1 9 lớp, trọng số bằng nhau) vì accuracy bị lớp
`Negatives` kéo cao. Phụ: top-1, balanced accuracy, precision/recall/F1 từng lớp
(bắt buộc Chinee Apple, Snake Weed), ECE 15 bin, mean ± std mẫu (`ddof=1`) qua 3 seed.

**Công thức nền T00** (GUIDE 1.4): ImageNet-pretrained, finetune toàn bộ, train
`RandomResizedCrop(224)` + lật ngang, val/test `Resize(256)+CenterCrop(224)`, chuẩn hoá ImageNet,
AdamW (lr backbone 1e-4, head 1e-3, weight decay 0,05 trừ norm/bias), warmup 1 epoch + cosine 12 epoch,
loss CE, batch 64, AMP. Checkpoint = epoch có macro-F1 val cao nhất.

**Phần cứng & thư viện.** GPU NVIDIA GeForce RTX 5060 Ti (16 GB), PyTorch 2.12.0+cu132,
`timm` (tag trọng số ghi ở sheet Backbones), `num_workers=0`, seed 0/1/2 (chỉ đổi init head,
thứ tự batch, augmentation; không đổi cách chia).

**Mức nhiễu seed.** Thực đo trên F01 val (3 seed: 97,19 / 96,91 / 97,51%): std ≈ 0,0030,
khớp ngưỡng slide (~0,3pt). Mọi Δ dưới ngưỡng này coi như nhiễu.

## 3. Kết quả so sánh backbone (sheet `Backbones`, `curves/B0*.png`)

| Mã | Backbone (tag) | Params / GMAC | Val F1 / Top-1 | Train s/ep | Trễ b1 (ms) |
|---|---|---|---|---|---|
| B01 | resnet50 (`a1_in1k`) | 23,53M / 4,09 | 80,65 / 86,12% | 59,6 | 12,47 |
| B02 | **convnext_tiny** (`in12k_ft_in1k`) | 27,83M / 4,46 | **96,04** / 96,97% | 79,1 | **11,44** |
| B03 | swin_tiny (`ms_in1k`) | 27,53M / 4,35 | 94,95 / 96,17% | 96,8 | 19,56 |
| B04 | efficientnet_b0 (`ra_in1k`) | 4,02M / 0,39 | 82,66 / 87,29% | 67,0 | 15,67 |
| B05 | mobilenetv3_large_100 (`ra_in1k`) | 4,21M / 0,22 | 82,78 / 87,32% | 52,5 | 12,80 |

Nhận xét (có số liệu):
- ConvNeXt-Tiny đứng đầu cả F1 **và** độ trễ — "ResNet hiện đại hoá" thắng ResNet-50 tới +15,4pt
  cùng công thức, cho thấy kiến trúc + tag pretrained hiện đại (`in12k_ft_in1k`) quan trọng hơn số params.
- Swin (transformer duy nhất) đạt 94,95% nhưng train chậm nhất và trễ cao nhất (19,56 ms) —
  attention cửa sổ kém hiệu quả phần cứng hơn convolution ở batch-1.
- **FLOPs không phải độ trễ**: EfficientNet-B0 chỉ 0,39 GMAC nhưng trễ 15,67 ms, chậm hơn ConvNeXt
  4,46 GMAC (11,44 ms) — depthwise conv ít FLOPs nhưng kém thân thiện bộ nhớ (đúng slide tr.43).
- Hai mạng nhẹ (82–83%) kém xa nhóm lớn (~95–96%): với ảnh đồng ruộng nhiễu nền, dung lượng mô hình vẫn cần thiết.
- **Chọn B02 đi tiếp**: tốt nhất cả hai chiều accuracy–latency, không cần đánh đổi.

## 4. Kết quả công thức huấn luyện (sheet `Training`, `curves/T0*.png`)

Mốc T00 = B02 (val F1 96,04%). Mỗi run T01–T08 đổi **đúng 1 yếu tố** (N1).

| Mã | Trục | Thay đổi | Val F1 | Δ vs T00 | So với nhiễu ±0,0030 |
|---|---|---|---|---|---|
| T01 | A | scratch (từ đầu) | 31,14% | −0,6490 | Suy giảm nghiêm trọng |
| T02 | A | frozen (chỉ train head) | 85,17% | −0,1087 | Suy giảm rõ (dù nhanh ~2x: 37 vs 79 s/ep) |
| T03 | B | TrivialAugment | 97,06% | +0,0102 | Cải thiện có ý nghĩa |
| T04 | B | **CutMix (α=1,0)** | 97,19% | **+0,0115** | Tốt nhất đơn lẻ |
| T05 | C | Label Smoothing ε=0,1 | 96,37% | +0,0033 | Vừa vượt nhiễu (sát ngưỡng) |
| T06 | C | Focal Loss γ=2,0 | 96,01% | −0,0003 | Không phân biệt được |
| T07 | D | Balanced sampler | 96,54% | +0,0050 | Cải thiện rõ |
| T08 | F | EMA decay=0,999 | 96,17% | +0,0013 | Không phân biệt được |
| T09 | Combo | CutMix+LS+Balanced+EMA | 96,99% | +0,0095 | Có ý nghĩa nhưng **thua CutMix đơn lẻ 0,20pt** |

Trả lời câu hỏi GUIDE:
- **Khởi tạo (A)**: với ~10,5k ảnh, scratch hoàn toàn không đủ (31%); pretrained ImageNet là sống còn;
  frozen mất −10,87pt → bắt buộc finetune toàn bộ.
- **Augmentation (B)**: CutMix tốt nhất; ép học đặc trưng cục bộ phân tán, hợp ảnh cỏ dại ngoài đồng
  (vật thể nhỏ, nền phức tạp). Lưu ý accuracy train khi dùng CutMix không còn ý nghĩa — mọi kết luận dựa trên val.
- **Loss (C)**: Label Smoothing giúp nhẹ; Focal Loss vô tác dụng ở đây — vì ConvNeXt đã tách tốt lớp
  `Negatives` nên không còn "mẫu khó" để focal phát huy (test chứng thực: Negative F1 0,984).
- **Sampler (D) vs loss có trọng số**: sampler cân bằng giúp +0,50pt mà không đổi loss — bằng chứng oversample
  lớp hiếm trực tiếp hiệu quả hơn kỳ vọng trên tập này.
- **EMA (F)**: +0,13pt, trong nhiễu — "miễn phí" nhưng không hại; giữ lại trong combo vì không tốn gì lúc suy luận.
- **Combo T09 cộng dồn thiếu (sub-additive)**: CutMix và LS cùng can thiệp nhãn (nhãn mềm hai lần) +
  sampler đổi phân phối batch → triệt tiêu một phần. Bài học: yếu tố tốt đơn lẻ chưa chắc cộng hưởng;
  phải thử combo thực tế thay vì giả định cộng dồn (đúng cảnh báo slide tr.46).

## 5. Kết quả suy luận (sheet `Inference`/`Latency`, `curves/inference_accuracy_vs_latency.png`)

Base cố định: **T04** (val F1 97,19%), mọi hàm `model.eval()` + `inference_mode()`. Độ trễ đo đúng quy tắc
(warmup 20, iters 100, `cuda.synchronize()` trước/sau, không tính tiền xử lý).

| Mã | Phương pháp (K, chi phí) | Val F1 (Δ) | ECE | p50/p95 b1 |
|---|---|---|---|---|
| I00 | 1-view 224 (mốc) | 97,19% | 0,0097 | 3,45 / 4,40 ms |
| I01 | TTA lật ngang (2x) | 97,26% (+0,07) | **0,0047** | ~6,9 / 8,8 ms |
| I02 | TTA 5-crop (5x) | 97,36% (+0,17) | 0,0055 | ~17 / 22 ms |
| I03 | prob vs logit | bằng nhau (±0,02) | **prob tốt hơn** (0,0047 vs 0,0063) | — |
| I04 | res 256 / 288 / 320 | +0,10 / −0,23 / **−1,54** | …/ 0,0202 / 0,0328 | 3,5–4,0 ms |
| I05a | Ensemble 3 convnext (3x) | **97,50% (+0,31)** | 0,0065 | ~10 / 13 ms |
| I05b | Ensemble convnext+swin (~2x) | 96,95% (−0,24) | 0,0067 | ~7 / 9 ms |
| I06 | **Model soup (T04+T03)/2 (1x)** | 97,38% (+0,19) | 0,0087 | 3,45 / 4,40 ms |
| I07 | Temperature scaling T=1,069 (1x) | 97,19% (0 flip) | 0,0097→0,0088 | 1x |
| I08 | FP16 (1x) | 97,19% (±0) | 0,0092 | 3,56 / 4,87 ms |

Phát hiện:
- **TTA**: F1 tăng trong nhiễu nhưng **ECE giảm ~một nửa** — TTA trung bình hoá dự đoán quá tự tin.
  Không đáng 2–5x chi phí trên robot; hợp offline (đúng slide).
- **I03 đúng như slide "chưa có kết luận"**: prob vs logit cho F1 ngang nhau, nhưng gộp prob hiệu chuẩn
  tốt hơn nhất quán (cả K=2 và K=5).
- **FixRes thất bại ở đây**: res càng cao càng tệ (320: −1,54pt, ECE nổ 0,0328). Nguyên nhân: train
  `RandomResizedCrop(224)` scale 0,8–1,0 nên model chưa từng thấy vật thể ở scale lớn — test 320 là
  lệch scale, không phải "thấy rõ hơn".
- **Ensemble chỉ tốt khi thành viên đều mạnh**: 3 convnext +0,31pt (tốt nhất Bước 3); Swin yếu hơn
  kéo ensemble xuống −0,24pt. Đa dạng không bù được chất lượng.
- **Xác nhận slide tr.73 trên máy này**: AMP batch-1 **chậm hơn** FP32 (5,54 vs 3,45 ms);
  FP16 batch-1 ≈ FP32, nhưng batch-32 FP16 nhanh **2,7x** (1 948 vs 729 ảnh/s). Phải đo, không giả định.
- **ConvNeXt có 0 lớp BatchNorm** (dùng LayerNorm) → gộp BN là no-op đã kiểm chứng (max|diff| = 0).
- **I5 đạt**: mọi cấu hình batch-1 p95 ≤ 22 ms ≪ 100 ms.
- Dữ liệu **ủng hộ** slide: thứ miễn phí (soup, T-scaling) cho robot; TTA/ensemble để dành offline.

## 6. Cấu hình tốt nhất (sheet `Final`/`PerClass`, tái lập được)

**F01** = `convnext_tiny` (tag `in12k_ft_in1k`) + finetune toàn bộ + `RandomResizedCrop(224)`+flip +
CE + **CutMix (α=1,0)** + AdamW (1e-4/1e-3, wd 0,05) + warmup-cosine 12 epoch + AMP +
**1-view + temperature scaling** (T fit trên val từng seed: 1,069 / 1,035 / 1,030).
Mốc so sánh T00+I00 cùng 3 seed. Test mở đúng một lần/seed (`save_test_predictions=True`;
T00-seed0 tái dùng checkpoint val đã chốt, eval thuần túy — cùng một model).

| Cấu hình | Macro-F1 test | Top-1 test | ECE test |
|---|---|---|---|
| F01 | **0,9734 ± 0,0029** | **0,9780 ± 0,0024** | 0,0072 ± 0,0008 |
| T00 (mốc) | 0,9642 ± 0,0032 | 0,9721 ± 0,0027 | 0,0200 ± 0,0026 |
| Δ | **+0,0092** (> std 0,0032 ✅) | +0,0059 | ECE giảm ~3x, NLL 0,075 vs 0,143 |

Val→test F01: 0,9721 → 0,9734 (chênh 0,0014 ≤ 0,02 ✅ I4b). Per-seed test F1:
0,9723 / 0,9712 / 0,9767 (F01); 0,9664 / 0,9605 / 0,9657 (T00).

Per-class test (mean 3 seed): Chinee Apple P/R/F1 = 0,962/0,966/0,964 (T00: 0,942, Δ+0,022);
Snake Weed = 0,967/0,954/0,961 (T00: 0,946, Δ+0,015). Mọi lớp còn lại F1 ≥ 0,960.

**Ma trận nhầm lẫn** (`curves/confusion_matrix_F01_test.png`, cộng dồn 3 seed, 10 521 lượt):
lỗi tập trung **Negatives ↔ cỏ dại** (seed0 sai 78/3507 = 2,22%): Negatives→Prickly Acacia 34,
→Lantana 25, →Siam Weed 20; Rubber Vine→Negatives 15, Snake Weed→Negatives 13.
Cặp nhầm kinh điển Chinee Apple↔Snake Weed của bài báo (3–4%) nay không còn trong top lỗi.
Giả thuyết: `Negatives` là thảm thực vật nền đa dạng (52% dữ liệu), chứa loài nền giống cây non của
Prickly Acacia/Lantana — các ca này ngay cả mắt thường cũng khó (`curves/error_samples_F01.png`).

## 7. Kết luận và khuyến nghị

- **Tốt nhất: F01**, hơn mốc **+0,0092 macro-F1** (~0,9pt), gấp ~3 lần std (0,0032) → cải thiện thật,
  không phải nhiễu. Top-1 97,80% vượt tham khảo bài báo ResNet-50 95,7% (lưu ý bài báo train ~100 epoch
  + aug mạnh, so sánh chỉ tham khảo; điều kiện lab 12 epoch đơn giản hơn nhiều).
- **Đóng góp xếp hạng: công thức huấn luyện ≈ suy luận > backbone?** Không — backbone cho bước nhảy lớn nhất
  (ResNet-50 80,65% → ConvNeXt 96,04%, +15,4pt cùng công thức); CutMix +1,15pt; suy luận tốt nhất +0,31pt
  (ensemble) / +0,19pt miễn phí (soup). Thứ tự: **backbone ≫ augmentation > ensemble/suy luận**.
- **Robot 30–100 ms/khung**: chọn **F01 + soup thay checkpoint? Không — chọn F01 1-view + T-scaling**
  (3,45 ms, p95 4,4 ms): soup (+0,19pt) là trung bình trọng số T04+T03 đã thử trên val, nhưng chung kết
  đăng ký là F01-CutMix thuần để giữ tính tái lập 3-seed/mean±std; soup là nâng cấp đề xuất tiếp theo
  (vẫn 1x chi phí, vẫn trong ngân sách). Tuyệt đối không dùng TTA/ensemble trên robot (2–5x chi phí
  mà F1 tăng trong nhiễu). FP16 chỉ nên dùng cho batch lớn offline (batch-1 không nhanh hơn).
- **I4a trung thực**: ECE test sau TS (0,0072) nhỉnh hơn trước TS (0,0064) +0,0008 — CutMix đã hiệu chuẩn
  tốt nên T≈1,03–1,07 gần như không đổi gì (0 flip/3507). Giữ bản calibrated vì quy trình chốt trên val;
  đổi sang uncal theo test là test-peeking.

## 8. Hạn chế và việc tiếp theo

1. Mới **1 fold** (fold 0) và **3 seed** quét sàng 1 seed — std ước lượng còn thô; vòng chung kết đã đủ 3 seed.
2. Chia **ngẫu nhiên, không theo địa điểm** (tác giả cũng vậy): điểm test có thể lạc quan khi gặp
   cánh đồng/mùa/ánh sáng mới — cần đánh giá lệch phân phối (góc chụp, mùa vụ, địa hình khác).
3. Chưa kiểm chứng: nhiều epoch hơn (12→20, trục G), RandAugment/MixUp, class-weighted loss,
   soup 3 seed F01, TTA cho bản offline, và hiệu chuẩn dưới lệch miền (T fit trên val có còn đáng tin?).
4. Độ trễ đo trên RTX 5060 Ti — số tuyệt đối không mang sang robot (Jetson); chỉ tương quan (xếp hạng
   fp32≈fp16>amp ở batch-1, fp16 vượt trội batch lớn) có giá trị chuyển giao.
5. Lỗi phát hiện sau test: không có — nhưng nếu có, nguyên tắc là ghi nhận, nêu ảnh hưởng, không chạy lại test.

## 9. Phụ lục

**Danh sách exp_id và cấu hình đầy đủ.**
- Backbone (T00 recipe, seed 0, 12ep): B01 resnet50 · B02 convnext_tiny · B03 swin_tiny_patch4_window7_224 ·
  B04 efficientnet_b0 · B05 mobilenetv3_large_100 → `curves/B0x_*.png`, `runs/B0x/seed0/`.
- Training (convnext_tiny, seed 0): T00 mốc · T01 scratch · T02 frozen · T03 trivial · T04 cutmix ·
  T05 ls(0,1) · T06 focal(2,0) · T07 balanced · T08 ema(0,999) · T09 combo → `curves/T0x_*.png`, `runs/T0x/seed0/`.
- Inference (trên T04): I00 1-view · I01 hflip-2 · I02 5crop-5 · I03 prob-vs-logit · I04 res 224/256/288/320 ·
  I05a ens-3cvnx · I05b ens-cvnx+swin · I06 soup · I07 T-scale · I08 FP16 → `runs/inference/inference_results.json`.
- Final (3 seed 0/1/2, test 1 lần/seed): F01 (CutMix+T-scale) · T00 (mốc) →
  `predictions/{F01,T00}_seed{k}_{test,uncal_test,val}.csv`, `runs/{F01,T00}/seed{k}/`, `eval_out/`.
- Biểu đồ tổng hợp: `curves/inference_accuracy_vs_latency.png`,
  `curves/confusion_matrix_F01_test.png`, `curves/error_samples_F01.png`;
  F01 per-seed: `curves/F01_seed{k}_convnext_tiny.png`.

**Tái lập.** `code/train.py` (một hàm `run(Config)` cho mọi thí nghiệm) ·
`code/run_step4_train.py F01 0 1 2` · `code/apply_temp_final.py` · `eval.py score/grade` (câu lệnh ở GUIDE mục 5).
Notebook: `code/lab_day2.ipynb`. Không sửa `eval.py`; mọi file predictions qua kiểm tra định dạng
(p0..p8 cộng bằng 1, y_pred = argmax) không lỗi.
