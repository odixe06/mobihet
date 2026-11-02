# P20 — closure audit, 01/11/2026

Đã pull và phân loại đủ32kernel exactversion1 theo [manifest](../../../docs/research/p20-launch.json):
**31run PASS_BOUNDED đủ20round; P21 lỗi trước round1;0auditfailure chưa xử lý**.
Đây là trạng thái artifact, không kết luận phương pháp thắng hoặc đạt fairness.

## Bằng chứng

- Job `p20-final-artifact-audit-20261101` succeeded22:21:31+07/exit0, hai marker audit hợp lệ.
- Closure `runs/p-study/final-audit/closure.json`, reportSHA khớp `audit.json` và `diagnostic-audit.json`.
- 620testCM/620snapshot,186trainCM; test support9.347.679 và train support37.361.273 mỗi lần đánh giá.
- Sourceprovenance/8overlay/config/LRcosine50/seed/batch64/full5343/W1/395408params đã kiểm;
  31run có compilegatePASS cả2rank/TeslaT4, log không traceback.
- CM tái tạo đủ10metric, perclass, mean/p10/median/min theo đúng quy ước lớp có mặt ở từng xe.
- Snapshot hữu hạn; teacher tiêu thụ trễ1round/signature, aggregate/packetbody, BN/weights của proxy replica,
  bank refresh và lớp cân bằng đã đối chiếu với artifact.
- Diagnostic34xe tại1/2/5/10/15/20 và external tại1/10/20 đủ ID/support; evaluator-only.
- Tổng payload scientific local3613file/25.283.541.660B. Đây là artifact được collector chọn,
  không phải toàn bộ resume/privatebank trên Kaggle.

6run từng dừng ở checkerQ: P15/P17/P18/P19/P20/P29. Giữ log/checker cũ ở `runs/p-study/recovery-v1/`.
CPU/CUDA reductionfloat32 có thể rơi hai phía ranh giớifp16. Checker mới truyền khoảng sai số theo precision/số hạng
qua quantizer và normalization;220snapshotQ phù hợp.3regressiontest dùng P15r5 thực, chặn sửaQ/classidentity.
Audit31run đã chạy lại đầy đủ sau sửa checker; **không sửa artifact hoặc source train, không rerun training**.

## Run thiếu và giới hạn

P21 `accanniez/mh-p21-proxy-hybrid-s43/1`: source/config/log đầy đủ nhưng không CM/snapshot/round hoặc resume checkpoint.
Xác suất upload không hữu hạn trước aggregation/fitproxy r1; [sự cố](p20-numerical-incident.md).
Probe T4 sau đó qua1round không thay P21benchmark; C10 vẫn chỉ có seed42, không nội suy seed43 hoặc gọi pairedPASS.

PASS_BOUNDED không gồm prediction→CM replay, reconstruction/fullresume privatebank, fullcontentfingerprint dataset,
exactCUDAreduction, DP/secureaggregation, mixedarchitecture hoặc contact/dropout/transport proof.
Snapshot nhỏ giữ server/teacher tensors, không checkpoint riêng mọi xe. Byte là payload protocol đã ghi,
không toàn bộ mạng thực tế; walltime2T4 không latency một xe.

Gate pull/audit toàn bộ đã đóng; được chuyển sang phân tích các [đối chứng khóa](screening-p20.md).
