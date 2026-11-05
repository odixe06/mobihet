# Continue — bàn giao sống, 05/11/2026 — sau L50, backup + dọn archive

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [vận hành](lessons.md).
Bàn giao trước: [05/11](../../archive/handoffs/2026-11-05-before-archive-cleanup.md), P/Q/R/V chi tiết [02/11](../../archive/handoffs/2026-11-02-1700-before-w20-launch.md).

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

## L50 — quyền mới owner04/11

- Chạy mới Core losslessCDF đủ50round,seed42,80train/20test như Core; không bootstrap64/Q8/bank.
- Ownerchọn2account/chuyểncheckpoint. Round1–25 emtrangtri;26–50 chuchimnon, mỗiaccountcap6,5h+15% quota.
- Cả2session dùng cache `odixe0502/mobihet-v4npy-cache` (SHA manifest như builder đãaudit), không dựngparquetcachelại.
- [Thiết kế L50](../research/screening-l50.md); local4round/resume/Corebitexact và generatedimportcell/tamperPASS.

## Job sống

Không còn job nào đang chạy.

L50 controller đã `L50_CLOSED` lúc 16:28:10 ngày 04/11. `runs/l50-study/closure.json`: 2/2 session (r1–25 `emtrangtri`, r26–50 `chuchimnon`) đều pulled, audited, last_round 50, `full50_pass=1`. Kết quả chi tiết tại [results-l50](../research/results-l50.md).

C10 đã đóng 4/4 audit 10 round, không còn watcher; [kết quả](../research/results-c10.md).

F50 controller đã succeeded 14:21 ngày 03/11. `runs/f-study/closure.json`: F0/F1/F2 đều pulled, audited, last_round 50, `full50_pass=3`.

## Bước tiếp

1. L50 đã đóng: đối chiếu 10 metric, cohort và byte giữa L0 Lossless và F2 Core tại [results-l50](../research/results-l50.md). Notion đã xong (owner 05/11).
2. F50 đã đóng: viết `docs/research/results-f50.md` (10 metric r50 cạnh baseline r50).
3. Sau quota reset 07/11: launch core s43 (F50 deferred, cùng recipe).

## Kết quả mới nhất

[L50](../research/results-l50.md): Core lossless CDF full 50 round PASS toàn diện. F1 macro 0.899129 (so với F2 0.899160, Δ = -0.000031); Acc 0.909285 (so với F2 0.909355, Δ = -0.000071); Client P10 0.752754 (so với F2 0.756214); Train loss 0.141436 (so với F2 0.141414). Tổng communication budget 50 round giảm từ 1,05 MB xuống 691 KB/xe (−34,45%, giảm 363.373 B/xe).

[C10](../research/results-c10.md): boot64 PASS (−22,9% byte r10). Q8/combo FAIL low256/absent. Lossless CDF256 đo codec khả thi, nay L50 đã chứng minh bit-exact và bảo toàn 100% chất lượng trên full 50 round.

## Việc tay cho owner

- Backup 05/11: rclone còn upload ~45 GB từ VFS cache; không xóa `~/.cache/rclone`, giữ mount. Xem [backup](backup.md).
- L50 hiện chưa cần thao tác tay; quota/account/dataset lỗi phải báo evidence cụ thể. Không yêu cầu login từ suyđoán.
