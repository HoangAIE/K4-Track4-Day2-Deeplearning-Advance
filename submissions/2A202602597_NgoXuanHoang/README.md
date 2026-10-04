# Bài nộp Lab Day 2 — 2A202602597_NgoXuanHoang

## 1. Liên kết notebook
- Notebook Colab/Kaggle chạy lại được: *(điền link khi upload `code/lab_day2.ipynb`)*
- Dataset: ảnh `images.zip` (Zenodo DOI 10.5281/zenodo.7939060, MD5 `b7b30f96d466fba86016aa5a26606e0f`),
  CSV fold có sẵn trong `data/labels/` (dùng fold 0, không sửa).

## 2. Môi trường
- GPU: NVIDIA GeForce RTX 5060 Ti (16 GB), Windows + PowerShell, `$env:PYTHONUTF8=1`
- torch 2.12.0+cu132 · timm 1.0.30 · scikit-learn 1.4.2 · numpy 1.26.4 · pandas 2.2.0
- Cài đặt: `pip install torch timm scikit-learn pandas openpyxl matplotlib pillow`

## 3. Thứ tự chạy lại
```powershell
$env:PYTHONUTF8=1
# B0: kiểm tra split + pipeline (seed, loss ban đầu ~2.197, overfit 1 batch)
python code/run_step0_checks.py
# B1: 5 backbone (T00 recipe, seed 0)
python code/train.py --exp_id B01 --seed 0 --fold 0 --set backbone=resnet50 num_workers=0
python code/train.py --exp_id B02 --seed 0 --fold 0 --set backbone=convnext_tiny num_workers=0
python code/train.py --exp_id B03 --seed 0 --fold 0 --set backbone=swin_tiny_patch4_window7_224 num_workers=0
python code/train.py --exp_id B04 --seed 0 --fold 0 --set backbone=efficientnet_b0 num_workers=0
python code/train.py --exp_id B05 --seed 0 --fold 0 --set backbone=mobilenetv3_large_100 num_workers=0
# B2: ablation T01..T09 trên convnext_tiny (mỗi run đổi đúng 1 yếu tố so với T00)
python code/run_step2_remaining.py
# B3: suy luận + đo trễ trên T04
python code/run_step3.py
# B4: chung kết F01 vs mốc T00, 3 seed, test đúng 1 lần/seed
python code/run_step4_train.py F01 0 1 2
python code/run_step4_train.py T00 1 2
python code/gen_t00s0_test.py            # test preds T00-seed0 từ checkpoint (eval thuần túy)
python code/apply_temp_final.py          # T fit trên val từng seed -> F01_seed{k}_test.csv
python eval.py score --pred predictions/F01_seed0_test.csv predictions/F01_seed1_test.csv predictions/F01_seed2_test.csv --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01 --out eval_out
python eval.py grade --final predictions/F01_seed0_test.csv predictions/F01_seed1_test.csv predictions/F01_seed2_test.csv --baseline predictions/T00_seed0_test.csv predictions/T00_seed1_test.csv predictions/T00_seed2_test.csv --uncal predictions/F01_seed0_uncal_test.csv predictions/F01_seed1_uncal_test.csv predictions/F01_seed2_uncal_test.csv --final-val predictions/F01_seed0_val.csv predictions/F01_seed1_val.csv predictions/F01_seed2_val.csv --test-csv data/labels/test_subset0.csv --val-csv data/labels/val_subset0.csv --labels data/labels/labels.csv --latency-p95-ms 4.4 --latency-method proper --out eval_out
# B5: curves per-seed F01 + sheet Summary + đối chiếu
python code/fill_step5_gaps.py
```

## 4. Seed đã dùng
- Quét sàng (B01–B05, T00–T09, I00–I08): seed 0 duy nhất; ngưỡng nhiễu ±0,0030 (thực đo std val F01 ≈ 0,0030).
- Chung kết (F01, T00): seeds 0, 1, 2; báo cáo mean ± std (mẫu, ddof=1).

## 5. Kết quả chính (từ `eval.py`, khớp `results.xlsx`)
- F01 test: macro-F1 **0,9734 ± 0,0029**, top-1 0,9780 ± 0,0024, ECE 0,0072
- T00 test: macro-F1 0,9642 ± 0,0032 (Δ = +0,0092 > std ✅)
- Grade phần I: **18/20**. Độ trễ batch-1 FP32: p50 3,45 ms / p95 4,40 ms.

## 6. Cấu trúc thư mục
```
├── README.md / results.xlsx (7 sheet) / report.md
├── curves/        # mỗi exp_id (B/T/F) một ảnh + scatter F1-latency + confusion matrix + ảnh lỗi
├── predictions/   # F01_seed{k}_{test,uncal_test,val}.csv + T00_seed{k}_{test,val}.csv (k=0,1,2)
└── code/          # toàn bộ .py + lab_day2.ipynb (dùng eval.py gốc, không sửa)
```
Không nộp dataset (`data/images`) và checkpoint (`runs/`, mỗi file ~110 MB).
