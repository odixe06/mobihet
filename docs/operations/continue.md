# Bàn giao sống — 24/11/2026 08:10 +07

Snapshot trước compact: [archive/handoffs/continue-2026-11-24-before-compact.md](../../archive/handoffs/continue-2026-11-24-before-compact.md)
(study LA, lịch sử L4 r50 20/11).

## Trạng thái

- **Owner 24/11: MobiHet = FB-L L4 bỏ logit adjustment (`la_mode=off`, A3 của [screening LA](../research/screening-la.md))**, vẫn tên
  `mobihet`. Trục LA đóng. L4 có LA là lịch sử: r50 ở [FB-L](../research/screening-fbl.md), số liệu chuyển từ experiment vào
  [archive/l4-la-20261124](../../archive/l4-la-20261124/README.md).
- Kết quả R50 9 kịch bản, ablation, compute/byte của MobiHet nằm ở repo `~/project/experiment`:
  `../experiment/docs/research/mobihet.md`, `../experiment/docs/research/mobihet-ablation.md`.
- **Không có job nào chạy từ repo này.** Batch MobiHet (3 seed thiếu + A1/A2/A4 + đếm FLOP) chạy từ experiment, xem
  `../experiment/docs/operations/continue.md` mục "Batch MobiHet" (PID hàng đợi push, collector, log).
- `methods/mobihet_budget.py` (SHA-256 `3c18cc34…`) được notebook BERT của batch nhúng và pin trong
  `../experiment/scripts/build_bert_l4_ablation.py`. **Không sửa** khi batch còn cần build notebook BERT.

## Git / DVC

- Git `main` = `6bf6e82` đã push GitHub (LA study + chuyển MobiHet + archive). Thay đổi chưa commit còn lại ở working tree
  (skill Kaggle, `AGENTS.md`, `scripts/{gate_fbl,gate_fbl_full,run_fbl_r50,kaggle_cli_bounded,analyse_core_bottleneck,audit_sim_batch}.py`,
  `docs/operations/kaggle.md`) là của phiên trước, không thuộc việc này.
- DVC 24/11: `dvc add` + `dvc push -j 16` sáu target (hop s43 `pull`, collect r50 `s4{1,2,3}-r50-h2`, `s43-r50`, `sim-kaggle-la`);
  log `runs/dvc-push-20261124.log`. Kết quả ghi ở [backup](backup.md).

## Việc tay

- Commit `6bf6e82` (mobihet), `803a84f`, `288628e` (experiment) có trailer `Co-Authored-By:` trái quy tắc owner 30/10. Sửa cần
  rewrite + force-push `main`: owner quyết.
- 6 account lỗi token: `chiencosieuhang`, `abbafan`, `gaming3`, `goilaaiudi`, `hoangdang20t`, `trietbackup`.
- (tuỳ, từ 20/11) Giải phóng 19 GB bản trùng: `rm -rf ~/project/mobihet/artifacts/fbl-hop/mh-fbl-l4-top-s4{1,2}-r50/upload`.

## Việc tiếp theo (không chặn batch)

1. Khi batch experiment xong: cập nhật bảng/tài liệu theo mục "Sau khi đủ" của bàn giao experiment; backup DVC các pull mới.
2. Owner 24/11 (chưa xóa): kiểm `~/project/fl_veremi_bert` trước khi xoá local — thay đổi FB-L chưa commit, `artifacts/fbl-hop`
   (~19 GB, chưa DVC), `runs/fbl-r50/*/audit.json` mà docs repo này trỏ tới; đưa danh sách cho owner duyệt.
3. (từ 20/11, chưa làm) dời `.dvc/cache` (~35 GB) sang ORICO rồi xoá workspace `pull/` đã DVC — chỉ khi owner yêu cầu dọn ổ.
