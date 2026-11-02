# O20 — teacher replay, khóa01/11/2026

Owner yêu cầu tiếp tục kiểm chứng sau [N20](results-n20.md). Chạy đúng phép replay ở
[diagnostic-next](diagnostic-next.md), chưa triển khai mọi đề xuất cùng lúc.

## Evidence → mechanism → falsifier

N20 realKD làm giảm timeDelay từr2 dù gate13đóng; đồng thời nâng other15/mean/p10. Qtrajectory khác nhau giữa cell,
chưa tách tiếp nhận client khỏi feedbackteacher. FullQ truyền cả confusion offdiagonal nhưng utility nội dung này chưa isolate.

Giữ cùng Q1…Q19 train-only từ N0(seed42)/N1(seed43), dùng lag1; Q20 chỉ snapshot/cuối, không train r21.
Mỗi seed chạy4cell từr1, cùng init/generator/sampler/protocol:

| cell | arm seed42/43 | realKD | synthetic target |
|---|---|---|---|
| R0 | O0/O1 | off | fullQ |
| R1 | O2/O3 | argmaxgate theo rawQ | fullQ |
| R2 | O4/O5 | off | giữ diagonal, chia đều15offdiagonal |
| R3 | O6/O7 | off | giữ diagonal, hoán vị15offdiagonal cố định |

R1−R0: realKD còn hại13 khi teacher được giữ chung hay không? Còn hại thì không cần feedbackQ để gây tác động;
nếu yếu/mất thì feedbacktrajectory quan trọng. Đây vẫn total effect clienttraining, chưa isolate từng classgradient/BN.
DirectKD13 phải bằng0 ởr2 của R1 vì rawQ1 reject13; không hardcode13 vào training.

R0−R2: soft confusion hơn smoothing đồng đều không? Uniformization đổi entropy, chưa chứng minh quan hệ tên lớp.
R0−R3: giữ entropy/diagonal/multiset xác suất, phá quan hệ tên lớp; nếu fullQ không hơn permutation thì chưa có
bằng chứng utility của offdiagonalidentity trong cấu hình này. Không suy vôích mọi cấu hình hoặc dự báo bytecompression thắng.
Không mở realgate cho R2/R3 sau biến đổi argmax. Không cộng gains vào server√n/mixture/generator chưa kiểm.

## Teacher identity, target và thông điệp

Nguồn SHA decodedbundle `79e26c53646fb7b5d7cc45b2d60690939d3a3cde5e885ad0696fff4e1d35acc8`;
40snapshot nguồn N0/N1 đã đối chiếu collectorreceipt. Bundle chỉ Q16×16 tổng hợp từ train-only outputs,
không rawsample/feature/per-rowlabel/privateweight. Chưa DP/secureaggregation.

`scripts/freeze_o20_teacher.py` khóa nguồn vào `methods/mobihet_replay_data.py`, kèm snapshotSHA.
Phục hồi packetfp16 tương đương sao cho decodedQ trùng nguồn trên localCUDA; không tuyên bố đúng packet gốc.
RuntimeGPU phải decode trùng từng bit nguồn trước bootstrap; CPU reduction cho phép≤2e−7 và luôn dùng nguyên Q nguồn.
T4 là cổng riêng, fail thì không train. Không reencode Q đãnormalized tùy tiện vì có thể lệch~1e−4.

Phép biến đổi chạy local sau decode rawQ. Uniform giữ nguyên diagonal, meanoffdiagonal tínhfp64 rồi castfp32;
permutation giữ nguyên các giá trịoffdiagonal. Rowmass kiểm bằng fp64 để tránh nhầm reductionorder thành thay mass.
Permutation RNG độc lập `numpy.default_rng(20261101)`,16row, không dùng test/trainstats chọn hoán vị;
matrixSHA và inputmoduleSHA nằm trong manifest trước launch.

Client vẫn collect/encode/aggregate ownoutputs đúng code N; lưu Q_live chỉ chẩn đoán, không feedback xuống client.
Q/Q_raw snapshot là target/rawQ của lịch replay cho round kế; driver lấy teacher trước round, probe đúng lag.
`methods/mobihet.py` và4trainingfile còn lại giữ nguyên SHA N; subclass riêng `mobihet_replay` thay downlink,
không sửa model/loss/optimizer/LR. Không neuralproxy. Schema không phụ thuộc kiến trúc; batch này vẫn chung DAGSNetW1.

## Protocol, endpoint và quyết định

Giữ [protocol](protocol.md):5343xe, W1,1epoch/batch64,AdamWreset/round,clip1/fp16;
20round mới/cosine50, seed42/43; copula64/step, LAreal, fixedweight1, supportoff, online/count/meanfp16.
Own-test từng xe endpoint20, pooled10metric/16class, mean/p10/std/cohort/absent/rare và pairedclientdelta.
Δr20−r15/slope11–20;34shadowclients tại1/2/5/10/15/20, external34 tại1/10/20 chỉ evaluator, không vào train/Q.
Không chọn best-round, HP hoặc permutation bằng officialtest; nhiều batch đã đọc test, không freshconfirmatorytest.

R0 là replaycontrol, không thêm livecontrol tốn2session vì N0/N1 đã có source/receipt.
Trước nhìn O, khóa tripwire tái lập R0 vs N0/N1: |ΔpooledF1r20|≤.001,
|ΔtimeDelayF1|≤.01, |Δp10|≤.01, |Δmeanxe<256|≤.01. Đây là giới hạn vận hành phép so,
không CI/noisebound/significance hoặc utilitythreshold. Vượt thì audit nguyên nhân và hạn chế kết luận,
không tune/rerun để tìm pass. SoR1/R2/R3 trực tiếp với R0 cùng batch là chính.
Hai seed chưa significance; vẫn phải đọc other15/weakclient, không gọi pooledgain là fairness hoặc winner tổng.

## Byte/thời gian và phạm vi replay

Packet-equivalent giữ544B/chiều/xe/r, bootstrap3311744B, tổng20round3333504B/xe.
**Đây là counterfactual**, teacherhistory được preload trong source riêng của nghiên cứu; không gọi nó liveFL cải tiến.
Báo thêm actualsize toàn `mobihet_replay_data.py` từ manifest; decodedbundle61440B gồm cả2seed,
referencefloat32+fp16bodies. Nếu tính preload như download cho từng xe: `3311744 + 20×544 + modulebytes`,
không cộng lần nữa downlink đã cache. Không cộng raw61440 và modulebytes như hai lần tải cùng dữ liệu.
Không tuyên bố cost-saving từ replay. Source/runtime/transport/ACK/setup có phạm vi khác; chưa end-to-endcontactproof.

Báo train/eval/diagnostic/setupcompile/notebookwall; thời gian2×T4 không là latency mỗi xe.
N đo train~642–655s/r, notebook~4,1–4,5h/run. Khóa cap6h/run,8account riêng,48sessionhourcap,
quota≥6,9h/account/reserved0; kiểm lại ngay trước push. W&B online và assert2×T4/compile2rank.
Không đổi globalCLI dinhan25. Inlinekey chỉ notebookprivate theo quyền có sẵn; lịch sử remote giữ key vĩnh viễn.

## Cổng và hoàn tất

61tests; fixture thật2xe173train/46test, W1/group128,3round×4cell compiledAMP;
R1privateweights/BN bitexact4cell; mỗi bước đối chiếu chèn teacher thủ công qua code gốc;
state roundtrip/strictrebuild/CM/finite/544B PASS local. Không T4/exactresume/utilityproof.
Nguồn/modelbank nguồn không tải lại từ archive. Notebookbase là HEAD đãpush, overlay giữ đúng filebytes mới.

Workercodex-long-jobs poll1200s, không modelpollwaiting; startup W&B/logs → terminal → collect/audit.
`audit_o20.py` giữ fullsource/CM/finite/probe/lag checks M/N, thêm rawQ/target từng snapshot vs frozeninput,
giữ Q_live tách khỏi Q_used và báo cảhai cách tính byte. COMPLETE/exit0 không thay audit đủ8×20round.
ERROR vẫn collect terminal để giữ bằng chứng, không cancel kernel khỏe hoặc launchtrùng khi push chưa rõ.
Manifest/runtime/ownership ở [continue](../operations/continue.md). Reference FD/KD vẫn như [method](method.md),
replay/hoán vị là chẩn đoán dự án, không thêm paper chưa triển khai hoặc gọi reproduce phương pháp gốc.
