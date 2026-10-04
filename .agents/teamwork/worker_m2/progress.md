# Progress Log - Worker M2

**Current Time**: 2026-10-03T17:40:56Z
**Last visited**: 2026-10-03T17:40:56Z
**Status**: Initialized. Ready to inspect code, prepare train/excel scripts, and launch ablation runs.

## Completed Tasks
- [x] Initialized DISPATCH.md and BRIEFING.md.
- [x] Reviewed Explorer handoffs (explorer_m2_1, explorer_m2_2, explorer_m2_3) and ORIGINAL_REQUEST.md.

## Ongoing Task
- [ ] Code adjustments (EMA checkpoint saving in `code/train.py`, updating `code/update_excel.py`).

## Remaining Tasks
- [ ] Run ablation experiments T01..T08 on `convnext_tiny`.
- [ ] Run combined recipe experiment T09.
- [ ] Verify artifacts and training curves.
- [ ] Update `results.xlsx` sheet `Training`.
- [ ] Perform scientific analysis and write `handoff.md`.
- [ ] Notify parent.
- [2026-10-03T16:06:15Z] **T01**: Bắt đầu chạy Lệnh: code/train.py --exp_id T01 --seed 0 --fold 0 --set backbone=convnext_tiny init=scratch num_workers=0
- [2026-10-03T16:26:40Z] **T01**: HOÀN TẤT F1=0.3114, Top1=0.5304, Thời gian: 20.4 phút
- [2026-10-03T16:27:13Z] **T02**: Bắt đầu chạy Lệnh: code/train.py --exp_id T02 --seed 0 --fold 0 --set backbone=convnext_tiny init=frozen num_workers=0
- [2026-10-03T16:37:08Z] **T02**: HOÀN TẤT F1=0.8517, Top1=0.8817, Thời gian: 9.9 phút
- [2026-10-03T16:37:09Z] **T03**: Bắt đầu chạy Lệnh: code/train.py --exp_id T03 --seed 0 --fold 0 --set backbone=convnext_tiny aug=trivial num_workers=0
- [2026-10-03T16:51:19Z] **T03**: HOÀN TẤT F1=0.9706, Top1=0.9766, Thời gian: 14.2 phút
- [2026-10-03T16:51:19Z] **T04**: Bắt đầu chạy Lệnh: code/train.py --exp_id T04 --seed 0 --fold 0 --set backbone=convnext_tiny mix=cutmix num_workers=0
- [2026-10-03T17:04:29Z] **T04**: HOÀN TẤT F1=0.9719, Top1=0.9780, Thời gian: 13.2 phút
- [2026-10-03T17:04:29Z] **T05**: Bắt đầu chạy Lệnh: code/train.py --exp_id T05 --seed 0 --fold 0 --set backbone=convnext_tiny loss=ls label_smoothing=0.1 num_workers=0
- [2026-10-03T17:18:17Z] **T05**: HOÀN TẤT F1=0.9637, Top1=0.9720, Thời gian: 13.8 phút
- [2026-10-03T17:18:18Z] **T06**: Bắt đầu chạy Lệnh: code/train.py --exp_id T06 --seed 0 --fold 0 --set backbone=convnext_tiny loss=focal focal_gamma=2.0 num_workers=0
- [2026-10-03T17:30:33Z] **T06**: HOÀN TẤT F1=0.9601, Top1=0.9689, Thời gian: 12.3 phút
- [2026-10-03T17:30:34Z] **T07**: Bắt đầu chạy Lệnh: code/train.py --exp_id T07 --seed 0 --fold 0 --set backbone=convnext_tiny sampler=balanced num_workers=0
- [2026-10-03T17:40:55Z] **T07**: HOÀN TẤT F1=0.9654, Top1=0.9734, Thời gian: 10.4 phút
- [2026-10-03T17:40:56Z] **T08**: Bắt đầu chạy Lệnh: code/train.py --exp_id T08 --seed 0 --fold 0 --set backbone=convnext_tiny ema_decay=0.999 num_workers=0
