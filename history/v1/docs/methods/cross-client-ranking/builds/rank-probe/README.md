# CrossClientRank — locked train-only probe

Hai GPU probes đã push15/10, sau CPU mount/SHA/20 strict CE-r2 models và controller fixtures PASS.
[Seed42](https://www.kaggle.com/code/minhtriethihi/mobihet-ccr-probe-s42-20261015-v1) ·
[Seed43](https://www.kaggle.com/code/khanhmay0304/mobihet-ccr-probe-s43-20261015-v1).
Hai seed đã terminal/audit: locked probe PASS. Diagnostic owner chốt đã terminal/audit và tái hiện energies;
residual inflation được phân tích, chưa utility. [Kết quả và giới hạn](results/report.md).

## Objective và phạm vi

`p=softmax(logits)`, T1, margin1, lambda0,25, 16 one-v-rest classes.
Pooled rank loss là trung bình squared positive/negative score gaps của joint collection private models;
local block surrogate dùng cohort means stop-gradient. Cả positive và negative side nhận gradient.
Own thay cohort counts/means/N bằng local; cùng support mask khi so gradients.
[Phương trình, nguồn primary, hạn chế](../../../../archive/research/new-directions-2026-10-15-afternoon.md).

Probe chỉ forward trên weights CE-r2 đã audit từ ClassTail calibration local arm; không optimizer.
Không đọc group/old/official test. Dữ liệu input gồm toàn bộ T của pilot; đây không phải full43M-record run.
Per-record logits/gradients chỉ nằm trong memory xe; output chỉ moments, counts và scalar diagnostics.
Không secure aggregation/DP hoặc contact/deadline R1 proof.

## Khóa trước thực thi

- Seeds42/43; dùng toàn T để scan score moments FP64, DAGSNet BN-eval/FP32, không AMP/TF32.
- Global mỗi phía positive/negative: ≥128record, ≥5contributors. Own mỗi phía≥32record.
- Tối đa256record/client/true-class, SeedSequence[1729,client,class], random không thay thế;
  nửa đầu fit, nửa sau test; record lẻ thêm vào test. Không dùng T split này làm holdout utility.
- Cả pooled/own rank terms dùng common support. Mỗi true-class cần≥5valid clients;
  mean energy theo client ngang trọng số, rồi macro-class. Energy≤1e−20 đánh dấu degenerate.
- Projection riêng mỗi client:16true-class CE directions,15centered class biases,
  1centered-logit direction. Fit-column RMS từ fit-half; least squares rcond1e−10;
  drop-column RMS≤1e−12. Chỉ đo residual ở test-half, không dùng projection update model.
- Mỗi seed cần≥12/16supported và nondegenerate classes; ≥12classes có
  RMS(pooled-rank−own-rank)/RMS(own-rank)≥0,05; eligible scan coverage≥0,90;
  macro-class residual-energy/delta-energy≥0,20. Cả hai seed PASS mới đủ điều kiện bàn utility.
- Primary utility nếu mở: private/group/gp theo true-class prior cố định cho mọi arm/freshGaussian/DML;
  head diagnostic. Owner đã chốt thêm controls/utility sau diagnostic; xem build rank-utility, không sửa probe này.
- Budget0,5h/seed. Old-owner quota đã đọc; không query ba account mới.

## Payload, bằng chứng và audit

Up928B/client: header32 +2uint64 counts +4FP64 sums/sumsq +2uint32 contributor flags mỗi class.
Down746B/client: header32 +2FP64 means +2uint64 counts +2uint32 contributors mỗi class,
eligibility2 +prior64 +N8. Bootstrap protocol packet trả actual bytes; sum cả20clients trong result.
Không gọi số này là toàn bộ channel overhead của utility chưa triển khai.

Local explicit all-pairs/autograd, block surrogate, finite difference, missing support,
wire/CRC corrupt refusal và projection fixtures PASS. Max FP64 gradient error3,47e−18.
Actual saved-r2 strict reconstruction/predict64rows PASS. CPU mounted inputs kiểm88SHA +20warm models.
Các bằng chứng này không thay kết quả production hai T4 hoặc utility.

Collector kiểm exact executed notebook sau redact key, source/input/protocol/warm-weight provenance;
tái dựng độc lập aggregate wire và scalar gates từ saved client energies. Không tái chạy toàn model scan/
gradient inference ở local, không gọi đây là independent full-gradient recomputation.

[Source và notebooks](../../../../../methods/cross-client-ranking/builds/rank-probe) ·
[Local algebra proof](../../../../../methods/cross-client-ranking/builds/rank-probe/probes/local-preflight.json) ·
[Continuation và quyết định owner](../../../../archive/operations/continuation-2026-10-15-afternoon.md).

Private inline W&B key chỉ ignored0600; private Kaggle version history vẫn giữ key. Artifact mới chưa DVC backup.

Diagnostic0,25h/seed giữ nguyên weights/sample/basis/rcond/gate, chỉ bổ sung SVD/coefficient/fit-test/class influence.
Normalized design condition≈5,2 bác giả thuyết SVD ill-conditioning. Class5 gần-zero delta và coefficient fit lớn
khuếch đại vài heldout T records. Gate không đổi; residual sensitivity báo riêng, không utility proof.
Source diagnostic riêng trong scripts, không sửa source/proj hoặc notebook probe đã PUSHED.

Sau diagnostic owner đã chốt [utility protocol](utility-proposal.md) và [build riêng](../rank-utility/README.md).
Probe/diagnostic vẫn không là utility PASS.
