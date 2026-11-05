Snapshot trước chốt C10; bàn giao sống: docs/operations/continue.md.

# Continue — bàn giao sống, 03/11/2026 — C10 compression screening

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [vận hành](lessons.md).
Bàn giao trước (P/Q/R/V chi tiết, callback cũ): [snapshot](../../archive/handoffs/2026-11-02-1700-before-w20-launch.md).

## Quyền owner đang có (02/11)

- Sau W20: chạy các kiểm nghiệm cần thiết cho phương pháp cuối → X20 (đã xong).
- 03/11: chạy phương pháp đề xuất 50 round (bank, ablation core), checkpoint mỗi round; chưa cần trộn kiến trúc.
- Step floor hợp lệ khi bước thêm chỉ synthetic; 1 epoch trên dữ liệu local giữ như repo benchmark.
- Chuẩn FL: không chia sẻ dữ liệu local; server chỉ dùng tổng toàn mạng ([proposal §Quyền riêng tư](../research/proposal.md#quyền-riêng-tư)).
- Không đổi protocol/HP; không chọn round/HP bằng official test; không thêm run ngoài batch đã khóa.
- Credential Kaggle process-scoped, không đổi global login. W&B online, key chỉ trong notebook private. Commit không có trailer `Co-Authored-By`.

## Quyết định mới owner (03/11, phiên compression)

- Mục tiêu đề xuất hiện tại: **MobiHET Core**, giữ performance với băng thông tiết kiệm hơn.
- Cho phép codec local và Kaggle **10–15 round, seed42 duy nhất**; khóa C10=10round/cosine50,4ô Core/boot64/Qdown8/combo.
- Báo riêng byte thường/SecAgg; chưa SecAgg/DP thực thi. [Thiết kế C10](../research/screening-c10.md).

## Job sống

| job | việc | log |
|---|---|---|
| `kaggle-c10-compression-20261103` | `scripts/run_c10.py`: prepare → push4kernel → poll1200s → pull → audit10round | `runs/c10-study/pipeline.log` |

Controller C10: supervisor PID108803; state `/home/odixe/.codex/long-jobs/jobs/kaggle-c10-compression-20261103/state.json`;
delivery `queue`, owner thread `01a1a21d-e2b9-7f02-ab52-a030091a4ccd`. Manifest/version/owner chính xác ở `runs/c10-study/batch.json`
và `docs/research/c10-launch.json` (đã push đủ4, version1; trạng thái ban đầu RUNNING cả4). Không retry khi có push-intent chưa rõ kết quả.
Files pipeline, notebook, C10artifact là read-only với phiên khác khi job sống.
Theo dõi: `tail -n 200 -F -- runs/c10-study/pipeline.log`.

F50 controller đã succeeded14:21 ngày03/11. `runs/f-study/closure.json`: F0/F1/F2 đều pulled, audited, last_round50,
`full50_pass=3`; log có F50_CLOSED3/3. Chưa viết results-f50 hoặc diễn giải performance trong phiên compression.
Core s43 vẫn deferred. Không còn coi F50 là job đang chạy.

## Bước tiếp

1. Khi `C10_CLOSED`: kiểm artifact/audit10round, viết results-c10 với10metric/cohort/delta cạnh Core và byte thường/SecAgg; không suy r10→r50.
2. F50 đã đóng: kiểm `runs/f-study/closure.json`, đọc theo [screening-f50 §Đọc kết quả](../research/screening-f50.md#đọc-kết-quả-khóa-trước-launch),
   viết `docs/research/results-f50.md` (10 metric r50 cạnh baseline r50).
3. Sau quota reset 07/11: launch core s43 (F50 deferred, cùng recipe) — cần một launch riêng trong `run_f50.py` (PLAN hiện chỉ 3 arm).
4. Backup R/V/W/X/F khi OneDrive còn chỗ.

## Kết quả mới nhất

[X20](../research/results-x20.md): 8/8 audit PASS. core PASS, bank PASS (sát ngưỡng, lợi thu hẹp theo round),
nckd_local xác nhận cơ chế lớp vắng, full trượt gate → phương pháp = core + bank, không realKD.

## Việc tay cho owner

- C10 hiện chưa cần thao tác tay. Nếu account/input/launch lỗi: đọc log, báo blocker cụ thể; không yêu cầu login từ suy đoán.
- Backup R/V/W/X: giải phóng hoặc mở rộng OneDrive; sau N/O/P/Q còn ~15 GB.
