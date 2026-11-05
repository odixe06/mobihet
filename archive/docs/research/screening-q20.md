# Q20 — streaming, inputtarget, serverweight và clientreception

Owner yêu cầu bắt đầu các kiểm nghiệm theo [đề xuất sau P](proposed-method-p20.md).
Batch đầu16run/8cell×seed42/43,20round fresh/cosine50/full5343/DAGSNetW1/batch64/1epoch,
AdamWreset/AMPfp16/clip1 như [protocol](../../../docs/research/protocol.md). Own-test từngxe, đủ10metric mỗi round.
Giữ train+val37.361.273/test9.347.679; không dùng test chọn HP hoặc sourceexpert.
P21R1 giữ worker/source riêng; Q không chờ hoặc suy gain từ C10 thiếu seed43.

## Cell và giả thuyết đã khóa trước launch

| cell | arm seed42/43 | nội dung | đối chiếu chính |
|---|---|---|---|
| S0 | Q0/Q1 | control C2 streamingclassQ, giữ code học cũ | nền mới và referenceP4/P5 |
| S0query | Q2/Q3 | cùng streamingclassQ, thêm score/upload/downinputbank nhưng không dùng bank để học | S0query−S0 kiểm tác dụng phụ của scoring/carrier |
| S1 | Q4/Q5 |32stream+32bank synthetic, cảhai dùng onlineclassQ | S1−S0query tách support |
| S2 | Q6/Q7 | cùng32+32input; stream dùng onlineclassQ, bank dùng target theoinput | S2−S1 tách targetresolution trên cùng support |
| S2sqrt | Q8/Q9 | nhưS2, chỉ inputensemble dùng sqrt(n_kc); onlineclassQ vẫn count | S2sqrt−S2 kiểm serverinputweight |
| S2real | Q10/Q11 | S2 thêm realKD theo argmaxgate classQ | S2real−S2 kiểm realKD trong teacherpipeline mới |
| S2margin | Q12/Q13 | S2 thêm realKD với margin Qcc−maxQcj dương | margin−real và margin−strength |
| S2strength | Q14/Q15 | binarygate nhân scale theo từngxe, tổng countweightedmass bằng margin | controlstrength cho selection của margin |

Không gọi margin/countsqrt là kỹ thuật mới: P/M đã thử trong classQ/bank khác;
Q kiểm liệu dấu tradeoff có giữ khi thêm inputteacher/mixedsupport. Chưa triển khai gradientprojection hoặc qualityweight.
Các contrast là live total effects gồm teacherfeedback; không phân rã causal. Replay cùngteacher sẽ là bước sau khi có history.

## Thông tin và phép học

Bootstrap/copula/prior giữ P. ClassQ là meanfp16 online trên realtrain trong localepoch, gộp n_kc,
được nhận từ round trước; không phải classmean của syntheticbank. Giữ offdiagonalutility O20.
M=1024input/16lớp cân, bankr sinh lại theo seed*100003+r,64input/lớp. Input mới được score sau localepoch,
outputfp16 encode/decode trước gộp; weights n_kc hoặc sqrt(n_kc) của nhãn synthetic, không confidenceoracle.
Target syntheticlabel có thể không đúng trên realmanifold; không dùng agreement để nhận phương pháp.

S0/S0query giữ64streamingrows như C2. S1–S2strength dùng32streamingcopula +32row từ bankr−1,
mỗi nửa đủ16lớp cân. S1/S2 dùng cùng input/RNG schedule; cảhai64synthetic/step, không tăng epoch/steps.
Round1 chưa có Q/target → hardlabel synthetic. S1–strength privateweights r1 phải trùng nhau.
Sau round1, S1 dùng Q[y] cho cảhai nửa; S2 trở đi dùng Q[y] cho stream và perinputtarget cho bankhalf.
RealKD chỉ Q[y] trên realtrain, không neuralproxy hoặc access realrows từ xe khác.

Không proxyserver fit/downmodel ở batch này. Server giữ aggregateclassQ và inputtargets; teacher cố định trong một round,
cập nhật cho round sau. Không privateweights/BN/rawrows/per-rowreallabel trong FLmessages.
Schemaoutput không phụ thuộc kiến trúc client; hiện toànDAGSNet, chưa heterogeneousproof hoặc formalDP/SecAgg.

S0query kỳ vọng weights/BN/Q/RNG/metric trùng S0 trên local3round; thêm packet/scoring không được sửa privateupdate.
Remote tripwire như P/O: |ΔpooledF1|≤.001, |ΔtimeDelayF1/p10/meanxe<256|≤.01.
Đây là operationalcheck, không CI/significance; không rerun/tune để tìm PASS. Nếu fail phải audit trước nhận utility.

## Gate, chi phí và artifact

- Giữ8overlay P đóng băng, thêm module Q/entrypoint; không sửa scientificsource đang chạy P21R1.
- Preflight W1/realv4/seed42+43/3round, compiledAMP; check S0 vs P/C2, S0query vsS0 bitexact,
  S1–strengthr1bitexact, classQonline/inputaggregate/decodedpacket/teacherlag, diagnosticstate và state/load strictrebuild.
- Runtime dừng cảhai rank nếu compilegateFAIL; nonfiniteprob bị chặn trước packet, metadata ở recovery/rank*.jsonl.
- S0:544B/chiều/r,3333504B/xe/20r gồm bootstrap3311744B. Các cell còn lại:33344B/chiều/r,
 4645504B/xe/20r; gồm hai packetclassQ544 và perinput32800. Không gọi S0query overhead miễn phí.
- Header32B+bodyfp16; input/query regenerated từ seed/version, không gửi inputbody. Chưa transportACK/crypto/contactproof.
  S0/S0query giữ classcodec cũ: bodyfp16 đã decode, headerbyte theo quy ước. Literalheader/CRCroundtrip áp dụng
  inputpacket và classpacket S1 trở đi; không gọi toàn protocol đã wireproof.
- Source/config/batch/notebookSHA,320testCM/320snapshot/96trainCM, shadow/external34xe evaluator-only,
  aggregateclass/input sums/weights, teacherchain/packetbytes và compileguard theo từngrank.
  Small snapshot không fullprivatebank/resume hoặc rawpredictionreplay/fullcontentfingerprint.
- Báo pooled/mean/p10/std, pairedclientdelta/fractionharmed, xe<256/rare/absent/xe>=4096, perclass và slope15→20.
  Tăng mean không tự đáp ứng mọixe; không chọn besttest hoặc policy riêng từngxe bằng test.
- Cap6h/run,96sessionhour; reserve15%→110.4h. LogP classcontrol tối đa4.534h, inputbank4.178h;
  cap6 cao hơn worstcase đo~32%, Q không serverproxyfit và vẫn giữ reserve0.4h của notebook.
  Điều chỉnh cap vận hành từ7h trước prepare vì livequota9.6–14h/account; không đổi sốround/HP để khớp quota.
  Chỉ push nếu liveownerquota/slots đủ toànbatch,
  dành ngân sách run đang active, không dựa reservedAPI0 để coi slot rảnh.
  Poll1200s workerlocal, collect cảERROR; failure cần diagnose/fix/verify, không bỏ các run khác hoặc relaunch mù.

Serverqualityweight từ localvalidation chưa sạch không vào batch. Nhánh originalfoldtrain→val cần controls/statfit riêng;
không tự đổi headlineprotocol. Không mở lại GMM/MADE/BNsplit hoặc trộn hard/softρ cũ dưới tên mới.

## Reference đúng thành phần

ClassQ/KL kế thừa [method](../../../docs/research/method.md): FD/KD adaptations đã dùng. Inputconsensus tiếp nối C7/P20,
tham khảo [FedMD](https://arxiv.org/abs/1910.03581); copulasharedqueries/countweightedprobability khác publicdata/logitmean của bài.
Q không có serverproxy nên không gán FedDFproxyfit cho các cell Q. Mixedsupport/sqrtinputweight/reception là giả thuyết dự án;
không reference PCGrad/FedAUX như thành phần đã triển khai.

Trạng thái/runtime ownership tại [bàn giao](../../../docs/operations/continue.md). Manifest sẽ khóa sau preflight/quota.
