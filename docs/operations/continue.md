# Bàn giao sống — kiểm 13/11/2026 02:45 UTC+7

Đọc [CONTEXT](../../CONTEXT.md), [LESSONS](../../LESSONS.md), [operations lessons](lessons.md) trước.
Đây là **bàn giao hiện hành duy nhất**. Artifact/state/process thật ưu tiên tóm tắt.
Lịch sử B13–B19, A1, N1/N2, baseline seed 41/43 đến 13/11: [snapshot](../../archive/handoffs/2026-11-13-before-consolidation.md).

## Đang chạy — B20 giảm compute full-scale (Claude Code)

[Design/luật/kết quả](../research/screening-compute-b20.md), [phụ lục luật](../research/screening-compute-b20-addendum.md).
Luật r50 mỗi arm: macro F1 r50 **mỗi seed > .9048**; báo Δ so CBB cùng seed, 10 metric, guard + trend, FLOPs, `t_train`.

| arm | kernel (version 1) | push | cap | trạng thái 13/11 02:35 |
|---|---|---|---:|---|
| SD42 | `novahihi/mh-k-b20-sd42-r50` | 12/11 ~21:50 | 11,5 h | RUNNING |
| SD43 | `keocaosu/mh-k-b20-sd43-r50` | 12/11 ~21:50 | 11,5 h | RUNNING |
| SM42P | `emtrangtri/mh-k-b20-sm42p-r20` | 12/11 | 5,5 h | COMPLETE, pulled, audit PASS, **PASS_TRAJECTORY** |
| SM42 | `emtrangtri/mh-k-b20-sm42-r50-c2` (resume SM42P r20→r50) | 13/11 02:30 | 7,0 h | RUNNING |
| SM43 | `minhtran0601/mh-k-b20-sm43-r50` (fresh r1→r50) | 13/11 02:30 | 11,5 h | RUNNING |

- SM42P (pilot r20 cos50): macro .8962 vs CBB42 .9000, Δr20 −.0037, Δ TB r16–20 −.0039, slope Δ nửa sau +.00022 (closing);
  `t_train` 3,57 h vs CBB 3,61 h. [audit.json](../../runs/b20-full/audit.json).
- Batch SM r50: `scripts/run_b20_sm50.py` → `runs/b20-sm50/batch.json` (khóa, `launch-complete.json` có — **không rerun launch**).
  Resume = notebook SM42P + ô import kiểm 2 rank/round 20/method/seed + `--until 50` + budget 7 h (mẫu `run_codec_combo50.py`).
  SM43 = notebook giống SD43 trừ knob `synth_grad=mix16`. Base `be3b7d5`, overlay SHA = SM42P.
- **Watcher nền:** `scripts/watch_b20.py` (Bash background trong phiên Claude Code, PID 36880, task `b1jofw7yg`),
  poll 1200 s, tối đa 40 tick (~13 h). Mỗi tick `collect` cả hai batch; batch ALL_TERMINAL → `audit` một lần
  (`runs/b20-full/audit.json`, `runs/b20-sm50/audit.json`). Thoát sớm nếu kernel nào terminal khác COMPLETE.
  Log `runs/b20-watch/watch.log`, state `runs/b20-watch/state.json`.
  Theo dõi: `tail -n 100 -F -- /home/odixe/project/mobihet/runs/b20-watch/watch.log`.
- Watcher mất (đóng phiên/reboot): kiểm PID; không bật watcher trùng. Chạy tay
  `python scripts/run_b20_full.py collect && python scripts/run_b20_full.py audit` và tương tự `run_b20_sm50.py`.
- Sau audit: verdict từng arm vào [design](../research/screening-compute-b20.md), gộp phụ lục vào design, cập nhật
  bảng compute Notion Proposal 2 nếu arm PASS. Backup: rsync `runs/b20-full/`, `runs/b20-sm50/` (bỏ `.pt`) vào `_receipts/` từng run,
  `dvc add` per-run SD42/SD43/SM42/SM43 + `dvc push` ([backup](backup.md)); SM42P đã DVC 13/11. Nhiều arm PASS → owner chọn arm rẻ nhất; combo SM+SD chỉ sau verdict từng arm.

## Đã xong gần đây

- **Baseline 3 seed** (`../fl_dagsnet_veremi_v2/runs/seeds/`): 15 baseline × seed 41/43 audit PASS/PASS_BOUNDED,
  NILM s41/s43 p2 đủ 50 round (audit 21:51 12/11). SCAFFOLD loại khỏi Notion (phân kỳ r2–5, NaN r15).
  Chưa xác minh hàng NILM-FL 3 seed trên Notion đã cập nhật.
- **CBB s41** `thuyanh316/mh-k-cbb-s41-r50/1`: audit 10/10, macro .9077; 3 seed .9078 ± .0004 — [report](../../runs/cbb-s41/report.md).
- B13–B19, A1, N1, N2: đóng, FAIL — [sổ tay](../research/method-building-brief.md) §4–5. Không job/collector nào của các batch này còn sống;
  state controller cũ `running` (B14, B18) là stale sau reboot, không restart.

## Việc kế

1. Chờ watcher B20 (completion đánh thức phiên). Không model polling.
2. Owner quyết hướng sau B20: giữ CBB và việc khác, hoặc nới một ràng buộc ([sổ tay §5](../research/method-building-brief.md)).
3. Backlog: mô hình trộn kiến trúc full-scale; privacy bootstrap (hoãn 09/11).

## Quyền / việc tay

Owner: đổi mọi module/compute, hỗ trợ khác kiến trúc; experiment chỉ DAGSNet W1; thang sim → pilot r20 → r50 hai seed đã ủy quyền
tự lên khi PASS. Credential process-scope; không đổi global credential, không xóa artifact, không hủy job khỏe.
Git push chỉ khi owner yêu cầu (13/11: đã yêu cầu push GitHub). Chưa cần thao tác tay.
