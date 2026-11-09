# Bàn giao sống — kiểm 10/11/2026 ~04:30 UTC+7

Đọc [CONTEXT](../../CONTEXT.md), [LESSONS](../../LESSONS.md), [operations lessons](lessons.md) trước. Đây là **bàn giao hiện hành duy nhất**;
state/log/receipt thật ưu tiên hơn tài liệu. Lịch sử 07–09/11 sáng: [snapshot 09/11](../../archive/handoffs/2026-11-09-km-lite-b9-b10.md).

## Đang làm

Owner 09/11: **giảm băng thông (mỗi round + bootstrap) trước, rồi owner chốt phương pháp chính** (Core hay Core + KM16).
Codec study + G1 (12 pilot) đã audit xong: 9 codec PASS, QQ4/SR48/SR32 FAIL ([screening-codec](../research/screening-codec.md), [catalog](../research/improvement-proposals.md) §G1).
Owner 10/11 duyệt **combo r20**: CBA (lite6+QQ8+LY0) và CBB (lite6+QQ8) × seed 42/43, thêm C0 (Core sim) + CQ8 (Core+QQ8).
Luật khóa trước push: [screening-codec §Combo r20](../research/screening-codec.md#combo-r20--luật-khóa-10112026-trước-push).
Smoke local 200 xe 2 round `--compile` PASS cả CBA và CQ8 (`runs/sim/smoke-{cba,cq8}`; Q ≈200+207 B/round, bootstrap CBA ≈1,08 MB unicast sim).

## Worker sống

| job | kernel (`owner/slug/1`) | push | trạng thái lúc push |
|---|---|---|---|
| CBA42 | `annhtnt/mh-k-cba-s42-p20` | 10/11 00:15 | QUEUED |
| CBA43 | `annhtnt/mh-k-cba-s43-p20` | 10/11 00:15 | QUEUED |
| CBB42 | `annhannh331/mh-k-cbb-s42-p20` | 10/11 00:15 | QUEUED |
| CBB43 | `annhannh331/mh-k-cbb-s43-p20` | 10/11 00:15 | RUNNING |
| C0 | `trietbackup/mh-k-core-s42-p20` | 10/11 00:15 | RUNNING |
| CQ8 | `trietbackup/mh-k-cq8-s42-p20` | 10/11 00:15 | RUNNING |

Cap 7,5 h/kernel, r20 ≈ 4 h. Watcher nền (Claude Code, owner phiên này): `runs/codec-combo20/watch.sh`, PID 9572 (setsid),
log `runs/codec-combo20/watch.log`, poll 1200 s, tự `collect` + `scripts/analyse_codec_combo20.py 20`, ghi `runs/codec-combo20/DONE`.
Theo dõi: `tail -f runs/codec-combo20/watch.log`. Máy tắt → phiên sau chạy `python scripts/run_codec_combo20.py collect` rồi analyser.

## Việc tiếp (theo thứ tự)

1. Khi `DONE`: đọc `runs/codec-combo20/{audit,verdict-r20}.json`, ghi kết quả vào screening-codec.
2. Theo luật khóa (tự động): CBA PASS cả 2 seed → nối CBA42/43 lên r50 từ state r20 (CBB dừng); CBA FAIL + CBB PASS → nối CBB;
   CQ8 PASS → nối CQ8 s42 r50 + push CQ8 s43 r50 (đối chứng F2/F3 nếu C0 ≈ F2, ngược lại nối cả C0).
3. Sau r50: báo owner số liệu Core vs KM combo để **owner chốt phương pháp chính**.
4. G1 đóng: giữ 64 dòng synthetic.

## Quyền và ràng buộc hiện hành

- Owner 09/11: thử hết codec, lên thẳng full-scale; luật xét F1 toàn hệ thống, guard lớp nhỏ nới; quota dư dùng ngay.
- Owner 09/11: **PASS → tự chạy bậc scale kế, không hỏi lại** (thang ở CONTEXT). Hướng a (gộp phiên) và b (tăng compute) **đã đóng**.
- Credential theo process (`kaggle_as.py`), global login giữ nguyên; không nhúng W&B.
- Không Git/DVC push, không xóa artifact. Workspace dirty, chưa commit (mọi thay đổi 07–09/11).
- Owner loại: trao model/proxy DAGSNet. Compute tại xe: không tụt hạng (Core 4,00×F; ngưỡng V2X-HtFL 4,11×F).

## Backup 10/11

Git `a9eff54` + DVC 43 manifest (y50, km*, codec/g1 pilot, f3, sim-kaggle*) in sync trên OneDrive; local dọn +92,5 GB. Chi tiết: [backup](backup.md). `artifacts/codec-combo20` chưa DVC (kernel còn chạy).

## Việc tay

Không có. Máy có thể tắt: watcher dừng theo, kernel vẫn chạy trên Kaggle; phiên sau chạy lại collect.
