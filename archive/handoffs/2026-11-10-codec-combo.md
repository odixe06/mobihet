> **Snapshot 10/11/2026 21:30** của docs/operations/continue.md trước khi rút gọn (combo codec r20/r50 xong, owner chốt CBB). Trạng thái sống: [continue](../../docs/operations/continue.md).

# Bàn giao sống — kiểm 10/11/2026 11:05 UTC+7

Đọc [CONTEXT](../../CONTEXT.md), [LESSONS](../../LESSONS.md), [operations lessons](../../docs/operations/lessons.md) trước. Đây là **bàn giao hiện hành duy nhất**;
state/log/receipt thật ưu tiên hơn tài liệu. Lịch sử 07–09/11 sáng: [snapshot 09/11](2026-11-09-km-lite-b9-b10.md).

## Đang làm

Owner 09/11: giảm băng thông trước rồi chốt → **xong 10/11: owner chốt CBB (KM16 + lite6 + QQ8) làm phương pháp chính.**
Combo r20 (6 arms) đã hoàn tất chạy, pull và audit lúc 04:41 10/11:
- CBA (lite6+QQ8+LY0) **FAIL cả 2 seed** (trượt guard benign; bỏ Lloyd 4 lượt làm suy giảm biểu diễn cụm khi ghép cùng QQ8).
- CBB (lite6+QQ8, Lloyd 4 lượt) **PASS cả 2 seed** (bootstrap unicast 1,15 MB −68%, round 319 B −71%).
- C0 (Core sim) **PASS** (khớp F2 s42 r20, Δmacro −0.0000).
- CQ8 (Core+QQ8) **PASS** (Δmacro TB −0.0005 vs C0, round 317 B).
Kết quả chi tiết đã ghi vào [screening-codec](../../docs/research/screening-codec.md#kết-quả-combo-r20-10112026-0441--66-audit-pass-runs後codec-combo20auditverdict-r20json).

## Trạng thái worker và Kaggle

**Combo r50 đang chạy** (push 10/11 ~10:58, xác nhận `RUNNING` cả 4 lúc ~11:00; luật khóa trước push:
[screening-codec "Combo r50"](../../docs/research/screening-codec.md#combo-r50--luật-khóa-10112026-trước-push)).
`scripts/run_codec_combo50.py` → `runs/codec-combo50/batch.json`:

| arm | kernel | kiểu | cap | ước tính |
|---|---|---|---:|---|
| CBB42 | `annhannh331/mh-k-cbb-s42-r50-c2/1` | nối r20→r50 | 7,0 h | ≈6,0 h + setup (headroom mỏng) |
| CBB43 | `annhannh331/mh-k-cbb-s43-r50-c2/1` | nối r20→r50 | 7,0 h | như trên |
| CQ8 | `trietbackup/mh-k-cq8-s42-r50-c2/1` | nối r20→r50 | 8,0 h | ≈6,5 h |
| CQ8s43 | `minhtrit06/mh-k-cq8-s43-r50/1` | mới r1→r50 | 11,5 h | ≈9,9 h |

Watcher Claude Code: `runs/codec-combo50/watch.sh` (PID 6315, `setsid`, poll 1200 s, tối đa 60 vòng), log `runs/codec-combo50/watch.log`,
xong ghi `runs/codec-combo50/DONE`. Chỉ collect (pull → `artifacts/codec-combo50/<kernel>/pulled.ok`); **chưa có analyser r50**.
Waiter auto-audit (10/11 20:23, Claude Code bg task `betffglup`): `runs/codec-combo50/analyse-after-done.sh` chờ local file `DONE`
(không gọi Kaggle) rồi chạy `scripts/analyse_codec_combo50.py --no-query`, log `runs/codec-combo50/analyse-after-done.log`, xong ghi `ANALYSED`.
Kiểm: `tail runs/codec-combo50/watch.log`; collect tay: `python scripts/run_codec_combo50.py collect`. Không bật watcher thứ hai.
CBB hết budget trước r50 → partial, audit round đã commit, nối tiếp cùng owner sau reset quota 14/11 (không restart r1).

## Việc tiếp

1. ✅ 10/11 ~20:20: `scripts/analyse_codec_combo50.py` đã viết + chạy trên 3 arm pull xong (CBB42/CBB43/CQ8 s42 audit PASS, rule PASS;
   **CBB PASS cả 2 seed**). CQ8s43 RUNNING (ước tính ~r49 lúc 20:20, cap 11,5 h đủ). Kernel đang chạy không lộ output/log trên Kaggle
   → script chỉ báo status + ước tính tiến độ (`runs/codec-combo50/progress-live.json`). Watcher không gọi analyser: sau khi
   `pulled: 4 / 4` chạy `/home/odixe/Tools/miniconda3/envs/nckh/bin/python scripts/analyse_codec_combo50.py`.
   ✅ 21:02 waiter chạy analyser 4/4: **audit PASS cả 4; CBB PASS 2 seed, CQ8 PASS 2 seed** — bảng ở screening-codec "Kết quả combo r50".
   Watcher 6315 + waiter đã kết thúc (`DONE`, `ANALYSED`). Không job sống.
2. ~~Khi cả 4 pull xong:~~ (xong) audit + verdict, ghi kết quả vào screening-codec, bảng Core+QQ8 vs KM lite6+QQ8 (10 metric, byte) → ✅ **owner 10/11 chốt CBB** (F1 .9078 vs CQ8 .8991), ghi ở [proposal](../../docs/research/proposal.md).
3. Việc kế: backlog owner (seed 41, BERT, model trộn) — **chưa được duyệt chạy**, hỏi owner trước.

## Quyền và ràng buộc hiện hành

- Owner 09/11: thử hết codec, lên thẳng full-scale; luật xét F1 toàn hệ thống, guard lớp nhỏ nới; quota dư dùng ngay.
- Owner 09/11: **PASS → tự chạy bậc scale kế, không hỏi lại** (thang ở CONTEXT). Hướng a (gộp phiên) và b (tăng compute) **đã đóng**.
- Credential theo process (`kaggle_as.py`), global login giữ nguyên; không nhúng W&B.
- Không Git/DVC push, không xóa artifact. Workspace dirty, chưa commit (mọi thay đổi 07–09/11).
- Owner loại: trao model/proxy DAGSNet. Compute tại xe: không tụt hạng (Core 4,00×F; ngưỡng V2X-HtFL 4,11×F).

## Backup 10/11

Git `a9eff54` + DVC 43 manifest in sync trên OneDrive; local dọn +92,5 GB. Chi tiết: [backup](../../docs/operations/backup.md). `artifacts/codec-combo20` chưa DVC.

## Việc tay

Không có.
