# Collect/audit sau reboot và crash — 14/10/2026, UTC+7

**Đã hoàn tất batch mới:** 4 kernel Hybrid full terminal, 7 arm/18 round đã pull/audit.
Không chạy lại training, không push notebook mới, không Git/DVC add/push/backup/cleanup.
Các quyền và cổng cũ giữ nguyên; ghi chú chuẩn bị/RUNNING ngày13/10 là lịch sử.

## Kernel, source và checkpoint

| kernel version1 | status terminal | arm: round cuối đã audit |
|---|---|---|
| minhtrit06/mobihet-hybf1-s42-20261013-cal-v1 | ERROR | local:2, DML:2 |
| odixeuit/mobihet-hybf2-s42-20261013-cal-v1 | COMPLETE | Gaussian:3, Hybrid:3 |
| trietbackup/mobihet-hybf3-s42-20261013-cal-v1 | ERROR | Gaussian-own:3, DML-own:2 |
| odixe0502/mobihet-hybf4-s42-20261013-cal-v1 | COMPLETE | Hybrid-own:3 |

Mỗi folder `papers/hybrid-full/kaggle/<slug>/` có `AUDITED.json`, `source/`,
`runs/<arm>_s42/config.json`, `complete/`, `weights/`, `resume/`, `predictions/` và payload nếu có.
Source chính `papers/hybrid-full/proj/` vẫn frozen; hash trong source manifest/executed notebook đã đối chiếu.
Target50/LR50, 20 client/100% train80%/seed42; không sửa source/manifest/notebook đã PUSHED.

Collector `scripts/kaggle/collect_hybrid_full.py` dùng helper `.agents/`, Range resume/6 connections,
376file/0thiếu/pull exit0. [Log](../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/logs/collection-20261014.log).
Collector đã ghi đủ4marker trước crash. Sau reboot không còn process collector;
không khởi động trùng/tải lại phần đã đủ. Không còn output mới cần chờ trong batch này.
Status snapshot lúc14:05UTC+7:56latest-version kernel,48COMPLETE/8ERROR, khôngRUNNING trong danh mục.
[Snapshot](kernel-status-current.json). Status không thay artifact proof và không chứng minh mọi session remote đều tắt.

Executed notebook chứa key chỉ ở ignored `private_push/` mode0600; bản redacted nằm trong public notebookfolder.
Giữ chi phí private version history theo ngoại lệ đã chốt; không đưa key vào source dataset/report/chat.

## Kết quả và giới hạn

- [Tính khả thi Hybrid full](../../methods/gaussian-proxy-hybrid/builds/hybrid-full/results/feasibility-20261014.md): tín hiệu tốt đầu full-data;
  chưa r50. ERROR do budget/parentassert3, worker exit0, checkpoint vẫn hợp lệ.
- [10 metrics cuối Hybrid](../../methods/gaussian-proxy-hybrid/builds/hybrid-full/results/latest-rounds.md),
  [report54evaluation](../../methods/gaussian-proxy-hybrid/builds/hybrid-full/results/report.md),
  [CSV](../../methods/gaussian-proxy-hybrid/builds/hybrid-full/results/metrics-all.csv), [per-class](../../methods/gaussian-proxy-hybrid/builds/hybrid-full/results/per-class-f1.csv).
- [10 metrics cuối H1/H2](../../methods/shared-baselines/builds/federated-baselines-full/results/latest-rounds.md): DML/own r7,
  Gaussian full/diag/own r10, lr4 r9, local r11/50. Không lấy terminal round khác nhau làm chênh lợi FL.
- Raw-test/client0 bounded inference mới: Hybrid42check, H1/H2 64check PASS,
  16.384prefix row, r1/round cuối từng run, margin>0,02. PeakRSS≈1.902/1.909MiB qua localguard;
  không full-inference hoặc leakage proof. [Hybrid](../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/probes/inference-proof.json),
  [H1/H2](../../../methods/shared-baselines/builds/federated-baselines-full/probes/inference-proof.json).
- BoundaryCells calibration hai seed đã audit/cost PASS; utility20round chưa chạy.
  [Ledger](../../methods/boundary-cell-counts/builds/boundary-cell-pilot/results/rebuild.md). Không notebook utility mới đã PUSHED.
- V3–V10/P2/S-a lịch sử đã có report/audit hoặc proof DVC; thiếu workspace sau cleanup
  không tự là output mới thiếu. Không restore hàng loạt hoặc re-audit weights đã ở remote trong phiên này.

R1 contact/deadline/ACK và R2 khác kiến trúc chưa được xác minh bằng full StageU này.
SecureSum/synthetic statistic không tự là crypto/DP. Officialtest chỉ mô tả, không chọn HP/best round.

## Quota và việc tiếp theo

Quota14/10: khoảng77,23h còn lại từ bảng đã làm tròn; refresh string service17/10.
[Snapshot](../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/probes/quota-20261014.txt). Đây không reservation.
Hybrid giữ pairing hiện tại/worst-round+dư15% cần khoảng314,70h phiên còn lại tới50,
chưa repeated preparation/import/compile. [Dự toán điều kiện](../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/probes/budget-remaining-20261014.json).
Không dùng budget H1/H2 hoặc pilot2,5% thay cost Hybridfull; max hai worker để đọc giờ phiên account.

Thứ tự nối việc đã được phép:

1. H1/H2 cont02 lấy parent cont01 đã audit, round riêng7/9/10/11; Hybrid lấy parent calibration trên,
   last-round2/3. Giữ50roundschedule, import/SHA/markerlast/require-resume trước data decode.
   Controller chấp nhận boundary theo budget, không đòi đồng loạt3round/session. Chạy theo quota,
   nối sau refresh17/10 theo quyết định đã chốt; chưa notebook continuation mới được tạo/push.
2. BoundaryCells paired r20 đủ direct/teacher P2/P4 +matchedlocal/mean/Gaussian +own/shuffle/marginal.
   FL/single có thể nối calibrationr2; controls fresh. Scientific `proj/` không đổi.
   `scripts/kaggle/boundary_logging.py` và `boundary_resume.py` mới có code hỗ trợ,
   chưa đủ controller/import/W&B-resume preflight, chưa tích hợp vào notebook utility.
3. ResponseRisk quyết định mới nhất: chỉ local augmentation trước. λJS0,1/swap9, probe g%20=1
   khôngSGD, matchedcontrols/holdout cũ; không mở shared/own response weights hoặc gọi local gain là FL.
4. EMA cả Gaussian/DML: backbonehalf-life2, noheadEMA, BN4096own/head16, gp cũ chính.
   R20 mỗi seed: FL+EMA−plain≥0,010; −own+EMA≥0,015; benign giảm≤0,005;
   cost≤2×plain cùng nền. Source/notebook chưa triển khai; interaction báo riêng.
5. Không mở dài adaptation FAIL V2–V7/V9/V10 y nguyên. Bài học cập nhật tại
   [LESSONS](../../../LESSONS.md), [trạng thái26nhóm](../overview/status-summary.md).

Không cần thao tác tay cho batch collect/audit này. W&B health polling đã ngừng khi các run ổn định;
không monitor tiếp vì batch đã terminal. Không còn job collect/report/inference local của phiên này sau khi hoàn tất.
