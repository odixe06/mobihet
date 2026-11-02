# Continue — bàn giao sống, 02/11/2026 — W20 đang chạy; backup + dọn N/O/P/Q xong

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [vận hành](lessons.md).
Bàn giao trước (P/Q/R/V chi tiết, callback cũ): [snapshot](../../archive/handoffs/2026-11-02-1700-before-w20-launch.md).

## Quyền owner đang có (02/11)

- Chạy cả batch W20 sau preflight local PASS; kernel builder cache npy **public**.
- Step floor hợp lệ khi bước thêm chỉ synthetic; 1 epoch trên dữ liệu local giữ như repo benchmark.
- Push GitHub/DVC; sau backup đã kiểm, **xóa local artifacts N/O/P/Q (`n20`, `o20`, `o20-early-20261101`, `p20`, `q20`)**,
  giữ R/V. DVC đẩy đầy đủ (gồm snapshot) dù OneDrive chỉ còn ~67 GB.
- Không đổi protocol/HP; không chọn round/HP bằng official test; không thêm run ngoài batch đã khóa.
- Credential Kaggle process-scoped, không đổi global login. W&B online, key chỉ trong notebook private. Commit không có trailer `Co-Authored-By`.

## Job sống

| job | việc | log |
|---|---|---|
| `kaggle-w20-pipeline-20m-20261102b` | builder `odixe0502/mobihet-v4npy-cache/1` → quota → 10 notebook → gates/push → poll 1200 s → pull → audit 20 round | `runs/w-study/pipeline.log` |

State: `/home/odixe/.codex/long-jobs/jobs/<job>/state.json`; delivery event-only (không đánh thức phiên nào).
Theo dõi: `tail -n 200 -F -- /home/odixe/project/mobihet/runs/w-study/pipeline.log`.
Pipeline W dừng ở stage lỗi và giữ receipt (`runs/w-study/builder*.json`, `quota.json`, `batch.json`, `*.push*.json`);
chạy lại `python scripts/run_w20.py` tiếp từ stage còn thiếu, `--collect-only` khi đã push. Không xóa intent để push lại mù.
Attempt đầu `kaggle-w20-pipeline-20m-20261102` failed trước push (guard credential bắt nhầm `--wandb`), log `pipeline-attempt1.log`.
DVC attempt đầu failed vì `.gitignore` ignore-all trong `o20-early`/`p20` (đã xóa), log `dvc-backup-attempt1.log`.

## Khi job xong

1. **DVC N/O/P/Q: xong** (job `mobihet-dvc-backup-nopq-20261102b` succeeded, in sync, cleanup +52,9 GB) — [backup](backup.md).
   Q chỉ còn CM/log (control cho W); không `dvc add` lại Q.
2. **W20:** khi `W20_CLOSED`: kiểm `closure.json`, chạy phân tích (mở rộng `scripts/analyse_r20_v20.py` thêm ô W và control),
   ghi kết quả vào `docs/research/results-w20.md`, áp quy tắc ở [proposal](../research/proposal.md#sau-w20--quy-tắc-chọn).
   Ô ERROR: pull log, phân loại lỗi train/vận hành, recovery riêng (slug mới), không restart cả batch.
3. Sau W: backup R/V/W khi OneDrive còn chỗ (cần owner giải phóng/mua thêm nếu thiếu).

## Kết quả mới nhất

[R20/V20](../research/results-r20-v20.md): đủ 4 repeat + 8 replay (R8 nối phiên) + 8 validation (V0/V1 qua V0R/V1R).
Commit `97cf51d` (GitHub main) chứa toàn bộ code P/Q/R/V/W trước đây chạy qua overlay, preflight W `runs/w-study/preflight.json`.
Tài liệu tổng hợp + metadata DVC: commit `c980871` (đã push).

## Việc tay cho owner

- Khi muốn backup R/V/W (~25 GB): giải phóng hoặc mở rộng OneDrive; sau N/O/P/Q còn ~15 GB.
