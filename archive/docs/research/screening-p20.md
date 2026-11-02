# P20 — nội dung teacher, carrier và bottleneck ngoài KD

Owner01/11 ủy quyền nhiều kiểm nghiệm song song, gồm đối chứng, giữ clientmodel chung repo benchmark.
V1 chỉ tham khảo66feature/20–100client, không chuyển sốF1/ngưỡng/kết luận nguyên nhân sang v4.
[Audit nguồn](prior-evidence-p20.md), [bằng chứng O](results-o20.md), [protocol](../../../docs/research/protocol.md).

## Quy tắc chung và cổng trước launch

32run =16cell×seed42/43. Mọi cell:5343xe, DAGSNetW1/395408parameter,77input/16class,20round từr1,
cosine50,1localepoch/batch64,AdamWreset/round/clip1/AMPfp16; own-test endpoint20, không besttest/HPsweep.
Source DAGSNet/grouped/protocol byte-identical `../fl_dagsnet_veremi_v2` đã ghi SHA trong audit nguồn.
Không gọi screening20round là so thắng benchmark50round. Modelheterogeneity là khả năng giao thức, chưa kiểm batch này.

Ngoại trừ hai realonly, giữ copula64/bước, LAreal, fixedweight1, countaggregation, meanfp16 và supportoff.
Chỉ thay loss/messageteacher/support đúng cell bên dưới; không đổi optimizer/epoch/feature order/scaler.
Teacher lag1: postroundr tạo teacher cho r+1; synthetic r1 dùng hardCE.

Trước launch: fulltests + W1/realfixture/compiledAMP3round mọi cell, state/eval/strictrebuild,
actualpacket encode/decode/CRC/shape/version, seededbank/rank/RNG và diagnostic không mutate trainstate.
Control class/gate/augment phải khớp code cũ trên fixture; bankcells có r1privateweights/BN chung.
Local sm86 không thay T4compile/2rank/full5343startup; startup phải kiểm onlineW&B đủ10metric và actual2T4.

## Cell khóa trước nhìn kết quả

| cell | arm42/43 | teacher / input / thay đổi | đối chiếu chính |
|---|---|---|---|
| C0 real | P0/P1 | realCE+LA, không synthetic/KD; chỉ bootstrapprior | C1−C0: augmentation cùng LA |
| C1 augment | P2/P3 | streamingcopula hardCE, không learnedteacher | C2−C1: softQ utility |
| C2 class | P4/P5 | streamingcopula, classQ, noRealKD | currentcontrol |
| C3 gate | P6/P7 | C2 + binary argmax realKD | C3−C2: live realKDtotal effect |
| C4 margin | P8/P9 | C3 realKD coefficient `max(0,Qcc−maxother)` | C4−C3 và C4−C5 |
| C5 strength | P10/P11 | binarygate × scalar/client khớp coefficientmass C4 | tách selection khỏi giảm KDstrength |
| C6 bank_class | P12/P13 | freshbank1024/r, aggregate inputtargets→classmeanQ | matchedbankcontrol C7 |
| C7 bank_input | P14/P15 | cùng bank/uplink C6, nhận target riêng từng syntheticinput | C7−C6: inputdependence |
| C8 proxy_bank | P16/P17 | serverDAGSNet học aggregate C7; gửi proxylogits trên bank | C8−C7: neuralserverfit |
| C9 proxy_stream | P18/P19 | servernhưC8, gửi fp16model; client query streamingcopula | C9−C14: extrapolation/support |
| C10 proxy_hybrid | P20/P21 | C8 servertarget=.5ensemble+.5onehot syntheticlabel | C10−C8: supervisedprior ở server |
| C11 bank_hard | P22/P23 | freshbank nhưC6–C8 nhưng hardCE, không teacher | C11−C1: finitebankeffect; C7−C11 |
| C12 project_augment | P24/P25 | C1 + đúng6binaryNaNflags/zero missingfeatures | cùng C13 kiểm teacher×support2×2 |
| C13 project_class | P26/P27 | C2 + cùng projectionmissingness | C13−C2, interaction với C12−C1 |
| C14 proxy_weights_bank | P28/P29 | cùngserver/bank C8 nhưng down fp16model, cachetargets tại client | C14−C8: carrier/compute/bytecontrol |
| C15 real_no_la | P30/P31 | localCE không LA/globalprior/synthetic/KD | C0−C15: LA/prior effect |

Không addgain từ ablation. LiveFL feedback khác giữa cell: các contrast là totaleffect,
không tự phân rã causal riêng target/BN/teachertrajectory. Save aggregate/teacher để replay sau nếu cần.

### Tiếp nhận realKD

C4: `a_c=max(0,Qcc−maxother)`; C5: `keep_c=1[argmaxQc=c]`,
`s_k=sum_c n_kc a_c / max(1,sum_c n_kc keep_c)` và coefficient `s_k keep_c`.
Cùng Q/counts thì `sum n_kc coefficient` khớp từng xe; không đảm bảo gradientnorm hoặc KLmass khớp.
Qtrajectory khác qua FL nên không gọi đây là isolate tuyệt đối continuity. Không hardcode class13/testthreshold.
Synthetic luôn softQ, không đổi fallback đồng thời.

### Teacher input-dependent và server

Mỗi round tạo **bank mới**1024copularow cân16lớp, seed=`seed*100003+round` độc lập realshuffle/streaminggenerator.
Hai rank và client dựng cùngbank, không gửi feature/record. Client học roundr trên bank của teacher r−1 (r1 dùng bank0).
RuntimeallgatherSHA phải xác nhận bank và downlinkteacher giống nhau giữa hai rank; lệch thì dừng trước commit/aggregation tương ứng.
Mỗi bước vẫn64synthetic,16lớp cân; rút ngẫu nhiên row trong lớp từ bank, không tăng localsteps.
Postepoch client score bankr ở evalBN/dropoutoff, gửi1024×16probabilitiesfp16, server gộp theo n_kc của nhãn sinh.
Bootstrap counts đã có; thiếu lớp→weight0. Không gửi realinputs/predictions từng realrow/privateweights/BN.

C6 và C7 chung scorer/codec/weight/support; C6 downclassmean còn C7 downperinput.
Đây là kiểm lại giả thuyết khác anchorJ: bank refresh mỗi round, fixedweight1 và fp16decode thực;
bankhard/streamhard ngăn gán finitebankeffect cho teacher. Chưa bảo đảm hỗ trợ input mới tốt.

Serverproxy C8/C9/C10/C14 là **DAGSNetW1 riêng tại server**, không nhận clientweights hoặc clientBN.
Warm-start qua round;200step/r, batch128, lr.001, AdamWwd1e−4/clip1/fp32, dropout0;
BN train trên syntheticbank. Objective softCE tới ensemble, C10 thêm .5onehot theo nhãn sinh, không giả định nhãn đó đúng thực tế.
Server giữ fp32state, teacher replica fp16params+BN, evalBN/dropoutoff cố định trong round.
C8/C10 downoutputs trênbank từ đúng replica này; C14 downmodel và client cache cùng outputsfp16;
C9 query model trên streaminginputs. Không realKD proxy trong batch này: cô lập syntheticteacher trước.
Proxy riêng tại server không buộc client cùng kiến trúc; hiện giữ chung clientDAGSNet để so benchmark.

Carriercontrol C8/C14 kỳ vọng gầnkhớp, không đòi bitexact T4sessions; nếu khác nhiều phải audit RNG/precision trước utilityclaim.
Khóa tripwire endpoint cùng O: |Δpooled|≤.001, |ΔtimeDelay/p10/meanxe<256|≤.01.
Đây là operationaltripwire, không CI/significance/noisebound. Không rerun/tune để tìm PASS.

### V4 missingness và localobjective

Projection chỉ dùng publictrain-fittedscaler mapping `nan_indicator`:6flag threshold.5, feature báo thiếu→0.
Không normalize lại, không thêm/loại input, không fit hoặc hỏi officialtest. Cờ hợp lệ chưa bảo đảm jointmanifold.
Đo trước utility: tỉ phần cờ liên tục và vi phạm flag1→feature0 trên synthetic; realtrainaudit bounded đã PASS.
C0 chỉ count/prior bootstrap64B/chiều, C15 không bootstrap. Không sinh covariance/histogram không dùng ở realonly.
Không tăng epoch cho xe nhỏ hoặc giữ AdamWmoment trái protocol. Bước/round/exposure/gap là diagnostic, không causalproof.
V1 BNsplit/head-only đã FAIL; P chưa mở lại BNrecalibration/BNsplit.

## Falsifier và tiêu chí đọc

- Teacherinput/proxy: teacherfit/labelagreement tăng mà own-test/weakclient không tăng→bác bỏ đủ mạnh để nhận method;
  không kéo dài hoặc thay mục tiêu sau nhìn test. ProxyfitKL thấp không là realteacherquality.
- C4 nếu hơn C3 nhưng không hơn C5→chưa có bằng chứng selection tốt hơn giảmstrength.
- Projection sửa constraint nhưng classifier không tốt hơn→constraint đó chưa là bottleneck utility.
- C0/C15 giúp kiểm LA; không gọi mọi lỗi là teacher. Control vẫn tăng15→20 thì không gọi plateau toàn hệ thống.

Báo đủ10metric mỗi round/16lớp, pooled/mean/p10/std, pairedclientdelta, xe<256/rare/absent,
trainCM matchedsameclient/class và slope15→20. Không nhận fairness nhờ kéo xe mạnh xuống.
Không hardcode lớp13 hoặc chọn besttest/HP. Officialtest đã dùng nhiều batch, không freshconfirmatorytest;2seed chưa significance.
Proxy có evaluator-only train/test cùng34client, max512row/fold/xe, lưu teacher/studentCM và sửa/hỏng paired;
không dùng kết quả đó làm confidenceweight/gate/train. Qlabelconditioned không có inputprediction,
không so "Qtestaccuracy biếttrueclass" với proxytestaccuracy như cùng nhiệm vụ.
Shadowgradient là fp32/dropout0 với teacher mới sản xuất cuối round, không gradientAMPactual hoặc clippingrate toàn mạng.

## Giao tiếp và runtime

Copula bootstrap3311744B/xe; C2–C5/C13 Q544B/chiều/r như control.
Newbankuplink32800B; downC6=544B, C7/C8/C10=32800B; C9/C14=`32+2*(P+BNfloats)` theo layout thực.
C1/C11/C12 không learnedbytes/r nhưng vẫn trả copulabootstrap. C0 prior128B một lần, C15zeroFLmessage.
P bank/proxy serialize header32B+fp16body, validate kind/round/shape/length/CRC, **decoded body mới vào server/client**.
Modelcarrier gồm params+runningmean/var, cấu trúc/layout đã khóa; counterBN không dùng trong groupedmodel.
Seed/index/round trong protocolheader; chưa transportACK/retransmit/crypto/contactdrop/end-to-endproof.
Giữ nguyên source/controlcodec cũ, không gọi toàn protocol đã transport-wireproof chỉ vì codec mới có CRC.

Khóa cap7h/notebook,32run cap224sessionhour, dự phòng1.15→257.6h, phân2run/account×16account nếu livequota/slot đủ.
HistoricalN/O~4.1–4.6h; proxy scoring/server/extrapolation có overhead mới, chưa đo fullT4.
Không tự tăng round hoặc kéo cap khi thiếu; callback audit actualround rồi quyết recovery.
GPUgiây train/scoring/server/eval/diagnostic/setup/total và bytebootstrap+hai chiều đều báo; không gọi wall2T4 là latency từng xe.
Poll1200s bằng codex-long-jobs, không modelpollwaiting; earlyterminal được pull/audit trong worker, lỗi thì giữ kernel khác sống.

## Reference đúng thành phần

ClassQ/KL kế thừa [method](../../../docs/research/method.md). Perinputconsensus và serverensemble→proxy là adaptation của
[FedMD §2.2](https://arxiv.org/html/1910.03581v1#S2.SS2) và [FedDF §3](https://arxiv.org/html/2006.07242v3#S3),
khác bài gốc: syntheticcopula seed thay publicdataset, không uploadprivateweights, weightedprobabilities thay logitmean,
16class/5343client và protocolDAGSNet của dự án. Không gọi reproduce nguyên bài hoặc chuyển gain paper sang VeReMi.
Reference chỉ gắn code đã kiểm local/cell tương ứng; remotecompletion/utility chỉ sau collectaudit.
Projection/margin/strength/hybridtarget là giả thuyết dự án, không gán tên paper chưa triển khai.

Source/batch/notebookSHA và ownership runtime ở [continue](../../../docs/operations/continue.md); file manifest khóa sẽ ghi sau preflight.
