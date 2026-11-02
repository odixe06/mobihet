# Continue — bàn giao sống, 03/11/2026 — F50 (50 round) đang chạy

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [vận hành](lessons.md).
Bàn giao trước (P/Q/R/V chi tiết, callback cũ): [snapshot](../../archive/handoffs/2026-11-02-1700-before-w20-launch.md).

## Quyền owner đang có (02/11)

- Sau W20: chạy các kiểm nghiệm cần thiết cho phương pháp cuối → X20 (đã xong).
- 03/11: chạy phương pháp đề xuất 50 round (bank, ablation core), checkpoint mỗi round; chưa cần trộn kiến trúc.
- Step floor hợp lệ khi bước thêm chỉ synthetic; 1 epoch trên dữ liệu local giữ như repo benchmark.
- Chuẩn FL: không chia sẻ dữ liệu local; server chỉ dùng tổng toàn mạng ([proposal §Quyền riêng tư](../research/proposal.md#quyền-riêng-tư)).
- Không đổi protocol/HP; không chọn round/HP bằng official test; không thêm run ngoài batch đã khóa.
- Credential Kaggle process-scoped, không đổi global login. W&B online, key chỉ trong notebook private. Commit không có trailer `Co-Authored-By`.

## Job sống

| job | việc | log |
|---|---|---|
| `kaggle-f50-pipeline-20261103` | `scripts/run_f50.py`: push → poll 1200 s → session chạm cap 11 h thì tự push resume `<slug>-c2` cùng owner → pull → `audit_f50.py` → `F50_CLOSED` | `runs/f-study/pipeline.log` |

3 kernel push 03/11 từ `20dbc19` ([thiết kế](../research/screening-f50.md), manifest `docs/research/f50-launch.json`):
heykewtiie F0 bank s42, chuchimnon F1 bank s43, keocaosu F2 core s42. Session/resume: `runs/f-study/sessions.json`.
Mỗi session ≈11 h; nhiều khả năng cần một session resume ngắn (≈1–2 h). Session ERROR không tự resume: pull log, phân loại.
State: `/home/odixe/.codex/long-jobs/jobs/<job>/state.json`; delivery event-only. Chạy lại `python scripts/run_f50.py --collect-only`
nếu worker chết sau launch.

## Bước tiếp

1. Khi `F50_CLOSED`: kiểm `runs/f-study/closure.json`, đọc theo [screening-f50 §Đọc kết quả](../research/screening-f50.md#đọc-kết-quả-khóa-trước-launch),
   viết `docs/research/results-f50.md` (10 metric r50 cạnh baseline r50).
2. Sau quota reset 07/11: launch core s43 (F50 deferred, cùng recipe) — cần một launch riêng trong `run_f50.py` (PLAN hiện chỉ 3 arm).
3. Backup R/V/W/X/F khi OneDrive còn chỗ.

## Kết quả mới nhất

[X20](../research/results-x20.md): 8/8 audit PASS. core PASS, bank PASS (sát ngưỡng, lợi thu hẹp theo round),
nckd_local xác nhận cơ chế lớp vắng, full trượt gate → phương pháp = core + bank, không realKD.

## Việc tay cho owner

- Duyệt batch 50 round (quota/thời gian).
- Backup R/V/W/X: giải phóng hoặc mở rộng OneDrive; sau N/O/P/Q còn ~15 GB.
