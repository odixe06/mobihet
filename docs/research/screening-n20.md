# N20 — tách realKD và synthetic fallback, khóa01/11/2026

Owner yêu cầu bắt đầu kiểm chứng sau [phân tích M20](analysis-m20-mechanisms.md).
Đây là phép đầu trong [diagnostic-next](diagnostic-next.md), không chạy đồng thời mọi đề xuất.

## Câu hỏi, giả thuyết, dấu hiệu bác bỏ

M có3/4ô của factorial, thiếu realKDoff+fallback. Fallback có dao động timeDelay khi Q đổi argmax,
nhưng hai kênh cùng đổi; realKD làm giảm mạnh timeDelay trong khi cải thiện15lớp còn lại.
Kiểm tổng ảnh hưởng của realKD, synthetic fallback và interaction trong feedback FL. Không giả định đã chứng minh gatechatter.

| cell | seed42 / seed43 | realKD | synthetic | đối chiếu |
|---|---|---|---|---|
| A | N0 / N1 | off | softQ | control M2/M3 |
| B | N2 / N3 | argmax gate | softQ | control M4/M5 |
| C | N4 / N5 | off | fallback | ô còn thiếu |
| D | N6 / N7 | argmax gate | fallback | control M6/M7 |

RealKD chia mean toàn bộ dòng real, gate nhân0 khi argmaxQ[y]≠y. SoftQ dùng Q[y] trên synthetic dù teacher sai.
Fallback dùng Q[y] nếu argmax đúng, onehot(y) nếu sai; không đổi gate ở realKD. Outer coefficient1, không supportmixture.
Q thu online và dùng lag1. R1 mọiô CEreal+CEsynthetic, chưateacher; phải khớp control R1 trong cùng điều kiện.

Hiệu ứng per-seed: real khi soft=B−A; real khi fallback=D−C; fallback khi noReal=C−A;
fallback khi real=D−B; interaction=(D−C)−(B−A). Báo cả chênh scalar và từng client/lớp.
Teachertrajectory mỗi ô có thể khác: đây là total effect, không gọi là isolate gradient dưới cùng Q.
Bác bỏ giải thích “chỉ realKD gây dao động” nếu C vẫn dao động tương tự D hoặc tương tác cho kết quả ngược.
Giảmdaođộng không đủ utility nếu F1/xe nhỏ/15lớp còn lại giảm.

## Cấu hình và đánh giá khóa

- Full5.343xe, DAGSNetW1, mọi xe mỗi round; r1–20 mới, cosine theo50, seed42/43.
- [Protocol chung](protocol.md): batch64,1epoch, AdamWreset/round, clip1,AMPfp16; không thay LR/optimizer/data split.
- Cùng copula64synthetic/bước, LAreal, meanfp16encode/decode, server countaggregation; group_rows4096.
- Không neuralproxy, không rawrows/features/per-rowlabels/privateweights trong thông điệp FL; schema output không phụ thuộc kiến trúc.
  Batch này chung DAGSNet, chưa mixedarchitecture/DP/secureaggregation/contactdrop proof.
- Own-test từng xe là chính, endpoint20; đủ10metric/16lớp, mean/p10/median/std/p90−p10, traincountcohort/absent/rare,
  tỷ lệ pairedclient tăng/giảm. Macro trên các lớp có trong ytrue mỗi xe; pooled đủ16lớp.
- Δr20−r15, slope11–20; số flip teacherargmax theo lớp, jump target/metric theo đúng lag. Không chọn best test round hoặc HP theo test.
- Cùng probe34xe của M tại1/2/5/10/15/20: shadowfp32/dropout0/trainBN, không actualAMPgradient; fulltrainCM descriptive.
  Externaldiagnostic34xe tại1/10/20 giữ nguyên evaluator/pool/source-exclusion, secondary, không vào train/Q/bootstrap/FLmessages.
- Packet544B/chiều,1088B/xe/round; copulabootstrap3311744B;20round=3333504B/xe. Báo thời gianwall2×T4, compile/eval/diagnostic.
- Source5trainingfile byte-identical M; pin Gitbase mới +5overlay cùng SHA để reuse auditor. Không thay code model/loss/speed.
  Các controls rerun đồng thời; không dùng sessionM cũ làm pairedcontrol duy nhất.

## Trước push và vận hành

54tests; realdata preflight4ô W1/batch64/synth64,2round, codec/CM/state strictrebuild; localgroup_rows128/2xe để vừa3050 và không có slot rỗng ở compilegate,
production4096 và2×T4compilegate phải kiểm từ mỗi kernel. Không gọi fixture utility hoặc localsm86 là T4proof.
Nguồn, metadata notebookprivate/image pin, W&B online, quota/owneridentity phải PASS trước launch.
8account riêng, cap6h/run (48hcap), quota≥6,9h/account và reservation0 khi khóa; ưu tiên account nhiều quota, không đổi login dinhan25.
Ước lượng từ M2–M7 gồm train/eval/diagnostic; thêm setup/compile/stateIO và margin. Partial<20 phải báo partial, không tự r1/rerun.

Poll exactowner/slug/version1 nền1200s qua codex-long-jobs, không modelpollwaiting. Initialhealth kiểm2×T4/compile/W&B,
terminal thì collect+audi20round trước kết luận. ERROR được pull để giữ bằng chứng; không tự cancel healthy kernel.
Manifest `runs/n-study/batch.json`, runtime/jobownership duy nhất ở [continue](../operations/continue.md).
Reference chỉ gắn phần cơ chế đã chạy tại [method](method.md); N là adaptation/ablation của dự án, không benchmark reproduce bài gốc.
