# BoundaryCells — ledger tái dựng/kiểm nghiệm

Đề xuất: [theo phương pháp](../../../README.md).
Scientfic protocol: [contract](contract.md); [dự báo utility](../../../../../archive/research/performance-feasibility-2026-10-13.md).

## Trước train

**Cập nhật 13/10, 15:24 UTC+7:** hai notebook đã PUSHED version1 lúc 14:30,
terminal COMPLETE, pull/audit đủ 5 arm×2 round mỗi seed. Cả 4 candidate mỗi seed
cost PASS (1,49–1,60× single-DAGS). [Calibration và 10 metrics](calibration-report.md).
Thông tin “chưa PUSHED/đang upload” bên dưới là mốc chuẩn bị cũ, đã được thay bằng
marker `notebook/<kernel>/PUSHED` và `kaggle/<kernel>/AUDITED.json`.
20 nhánh đã qua bounded branch smoke; chưa utility r20. [Proof](../../../../../../methods/boundary-cell-counts/builds/boundary-cell-pilot/probes/branch-proof-initial2.json).
Pilot paired r20 đủ controls đang chuẩn bị; FL/single nối r2 đã audit, controls mới bắt đầu r1.
Không sửa scientific `proj/` đã PUSHED. Logging adapter/import nằm ngoài scientific source.

- Bounded probe40.960train/4.096group row, không SGD/test. P2/P4 target hơn own/shuffle,
  nhưng entropy cao; chưa chứng minh DAGSNetutility. [JSON](../../../../../../methods/boundary-cell-counts/builds/boundary-cell-pilot/probes/bounded-probe.json).
- Producer probe banđầu giữ exactSHA ở `probe-source-v1/`.
- Smokeinitial1 trước sửa paid-prior decoder giữ producer/sourceSHA ở `smoke-source-initial1/`;
  không xóa output cũ. Smokeinitial2 nguồn cuối:5arm×3round,20client×256row,
  4corruptionrefusals/strictDAGSNetrebuild/exactresume weights/state/pred PASS;
  peak2.834MiB. [Proof](../../../../../../methods/boundary-cell-counts/builds/boundary-cell-pilot/probes/local-proof-initial2.json), [log](../../../../../../methods/boundary-cell-counts/builds/boundary-cell-pilot/logs/smoke-initial2.log).
- Wire sanitize: ineligible cell không phát target rare qua maskfalse; client dùng decodedFP32prior,
  không đọc trực tiếp server counts. Own nhận paidcommonmask, không population targetvectors.
  Đây là giao thức statistic, chưa crypto/DP.
- Notebook hai seed banđầu chưa push; giữ source chuẩn bị. Hai notebook mới tách seed,
  khóa GPU cảJSON vànotebookmetadata;8controllerfailure/budgetcases PASS.
  [Proof](../../../../../../methods/boundary-cell-counts/builds/boundary-cell-pilot/probes/notebook-proof-initial2.json).
- Private source owner42 READY/114file; owner43 đang upload. Scientific source/hash giữ nguyên giữa hai owner.

## Kernel IDs đã chạy

- `minhtran0601/mobihet-bc-cal-s42-20261013-initial2-v1`
- `dinhan25/mobihet-bc-cal-s43-20261013-initial2-v1`

Mỗi kernel chỉcalibration2round/20schedule, không tự nối r3 hoặcutility.
PUSHED version1 lúc 07:30:02/07:30:30 UTC; audit calibration cả hai seed đã hoàn tất.
R20 utility còn chưa đo; notebook calibration không tự nối quá r2.

Quota/budget dự kiến2h/session+15%reserve2,3h mỗiaccount. Snapshot/reservation cũ của notebook chung
không dùng làm reservation cho hai notebook mới; lấy freshquota và ghi allocation riêng trướcpush.
Theo dõi banđầu W&B online/applied/skipped, metric đủ10khi đánh giá; lỗi thật đangtrain thì báo owner dừng.
Stable → ngừnghealthpoll; collector `scripts/kaggle/collect_boundary_calibration.py` chỉpull/audit.

## Cổng

Cost từ2commits `algorithm_seconds` trướceval/commit so freshsingle-DAGS;≤2× từngseed/P/mode.
CostPASS chưa utilityPASS; muốn mở20round cần audit source/config/commit/weights/BN/pred→metrics trước.
R20 pairedutility/own/shuffle/strongbaseline/benign guard tại contract; không đổi headline/bestround để cứu.
R1contact và R2 khác kiến trúc chưa kiểm nghiệm.
