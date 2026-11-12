# Khảo sát nguồn teacher / server — 11/11/2026

Phạm vi: nguồn sơ cấp, đối chiếu CBB và lịch sử đã audit; không train hoặc sửa phương pháp.
Đọc cùng [history audit](history-audit-20261111.md), [transfer review](knowledge-transfer-review-20261111.md),
[temporal feasibility/quality](crossfit-feasibility-20261111.md), [B17](../../../docs/research/screening-macro-b17.md).
Các hướng dưới là **giả thuyết adaptation**, chưa bằng chứng tăng macroF1 trên CBB.

## Phát hiện hữu ích nhất từ paper

**Có cơ sở trực tiếp cho nguồn ngoài fit; không có định lý chuyển thẳng sang classQ→KM.**
Dao et al., *Knowledge Distillation as Semiparametric Inference*, §4.2: teacher mỗi fold fit ngoài fold,
student dùng mọi row, nhưng predictionteacher được đánh giá tại **chính input student**.
§2 giả định mẫu độc lập cùng distribution; phần kết luận còn để analogue cho syntheticaugmentation làm việc tiếp.
Do đó đổi thành `mean_y p_teacher(x_heldout)` rồi ép vào KMrow khácx không thừa hưởng guarantee.
[Paper](https://arxiv.org/html/2104.09732v1),
[code tác giả](https://github.com/microsoft/semiparametric-distillation).

**Đừng hiểu nhầm “precision” của CRAD thành PPV.** CRAD dùng classaccuracy trên validation,
tức recall của trueclass. “Precision weighting” là inversevariance của ước lượngaccuracy:
`ã=(an+2)/(n+4); w=1/(ã(1−ã)/(n+4)+epsilon)`.
Paper trao **peer modelsnapshots**, chạy teacher trên inputrecipient; 300round, một seed và chọn checkpoint qua validation.
Paper thừa nhận weight thưởng decisiveness, consensus cần giả định expertkém là thiểu số;
appendix nêu trường hợp filter loại specialist tốt nhất.
[CRAD v2 §3.2–3.3, §4.1, AppendixA–B](https://arxiv.org/html/2609.00446v2).

Hệ quả đại số riêng: cùngn, `a=.05` và `.95` có weight bằng nhau. Đo được accuracy chính xác
không đồng nghĩa accuracy cao. Không copy inversevariance thành reliabilityweight rồi tuyên bố có bảo đảm.
Không copy modelsnapshots: trái schema CBB hiện hành và đắt hơn nhiều so với classQ.

**Personalized logits-prototype đã có tiền lệ gần hơn KT-pFL.** FedLDP (2026) mô tả
logitsprototypes, tách collaborative/local knowledge và servercollaborationgraph tạo prototype riêng từngclient;
đánh giá graphlearning sáu dataset, lưu ý extreme longtail chưa tối ưu.
Chỉ đọc được abstract/introduction/conclusion do fulltext403; **chưa xác minh công thức graph/codec/code**.
[Bản tác giả tại publisher](https://www.sciencedirect.com/science/article/pii/S0950705126008683).
Không gán gain graphclassification cho VeReMi hoặc gọi adapter CBB là reproductionFedLDP.

## Bốn estimand khác nhau, diagnostic trước utility

### 1. Nguồn Q ngoài fit, giữ nguyên cách dùng Q — ưu tiên

**Evidence:** B14confidence tăng không tăngmacro; B16 đổi reductionFAIL; B17prior/realNCKDFAIL.
V20hardCM đã có train→futuregap và rankingquality nhạy support. Đây chưa là softQ của CBB.

**Cơ chế cần kiểm:** producer cùngweights đo Q tại row đãfit và row chưafit, thay nguồn target;
không đồng thời thay loss, qualityweights hoặc syntheticmain.

**Phép kiểm rẻ và phân biệt được:** một shadowDAGSNetW1 train earlyA; generator/Q/BN/optimizer chỉA.
Sau cùng round, evalmode cố định, đo cảA vàheldoutH. Match sốrow theo client×trueclass
để so source khỏi confound count; ghi coverage tất cả16class và fallback.
Đo softmean, offdiagonal, Brier/NLL, hardCM và masssupport; không chỉentropy.
Giữ mainfulltrain. Control cùngshadow/evalmode quan trọng hơn so trực tiếp heldoutQ với online/trainmodeQ.
Fixedpreprocessing vẫn chứa H theo protocol: gọi model-out-of-fit conditional on preprocessing.

**Dấu bác bỏ:** chỉ Q mềm đi tương tựtemperature nhưng không tín hiệu lỗi bổ sung; support lớp11 quáhẹp;
hoặc sim2seed không đạt Δmacro≥+.002 từngseed. Metricquality tăng riêng không cho PASS.

**Chi phí:** singletemporalshadow hiện lowerbound main+shadowsteps≈1.811×, chưa gồm inference/bootstrap/I/O.
Chỉ packetstats rời xe; phải đếm bootstrapshadow và giao tiếpnội bộ, không chỉ teacherfile cuối.
Không gọi thiết kế này mới hoàn toàn: đã ở feasibility; mới bổ sung kiểm giới hạn lý thuyết và controlmatchedsource.

### 2. Teacher theo recipientdomain — ưu tiên sau diagnostic nhóm

**Evidence:** Q CBB toànmạng nhưnhau cho mọiclient; sourcecode `_client_clusters()` chỉ nhóm cho generator.
Teacherrecipient chưa exacttrial trong lịch sử audit; QK/K10 lại là target theo vùnginput, đã FAIL.

**Cơ chế:** `Q_k[y] = sum_j n_jy a(k,j,y) q_j[y] / sum_j n_jy a(k,j,y)`;
`a` chọn quầnthểexperts phù hợp recipient, giữ architecturebasis chỉ labelspace.
Không lấy cosinehiddenvectors khácbasis hoặc coi confidence là reliability.

**Diagnostic:** metadata publicscenario/run đã có, khóa grouping từ metadata trước chấm;
kiểm group có giải thích futureconfusion **sau standardized classprior và count** haykhông.
Dùng nội bộ temporalval retained làm diagnostic, không optimizeweights theo nó rồi gọi freshconfirmation.
Đồng thời đếm full/sim support từnggroup×class, effectiveexpertcount và fallbackglobal.
Nhóm cófewrareclass hoặc heterogeneity chỉdo prior → dừng đường grouping này.
Trên heldoutgroup kiểm sourcegroup predictionerror khácrecipientgroup thế nào;
ownclientCM riêng chỉ cho evidence domainshift, chưa đo cross-recipienttransfer.

**Dấu bác bỏ:** khác biệt biến mất sau prior/countcontrol, không ổn qua seed hoặc groupkhông có đủ lớp;
utilitymacro không qua gate thì đóng adaptation, không tìm sốgroup/topk mới qua test.

**Chi phí:** có thể giữ Quplink hiện tại và một Q16×16downlink/xe.
Tổng mạng unicast/nhóm khác một globalbroadcast; tính thật, không gọi communicationunchanged chỉvì mỗi xe nhận cùngsize.
Chưa có cơ sở chọn dynamicgraphN² hoặc sốcluster từ abstractFedLDP.

### 3. Weight expert bằng confusion ngoài fit — khác với đổi nguồn Q

**Evidence:** hardCM tươnglai V20 cho thấy trainquality không xếp hạng future đều cácclass;
benign→attack FP là lỗi quan trọng. Recall riêng của expert không đo falsepositive vào classđích.
CountweightedQ là populationmean đúng theo estimand hiện tại; weightquality **đổi estimand**, không sửa bug.

**Cơ chế:** sau khi có measurement ngoài fit hợp lệ, gửi classconditional confusion/count hoặc
`TP,FP,FN` tổng để ước lượng precision và support tại priorreference train cố định.
So **cùng source Q** giữa countweight và qualityweight; không ghép sourceOOD + qualityweight trong mộtarm.
Ownheldoutquality vẫn chưa bảo đảm transferquality sang recipientkhác.

**Diagnostic trước khi chọn rule:** đối chiếu ranking ngoài fit với trainranking và count bằng cùng
supporteligibility; ghi uncertainty và negativeclassmixture. Thiếu classsupport là unknown, không quality0.
Ví dụ precisionstandardizedbinary không loại hoàn toàn shift giữa15negativeclasses.
Không chọn inverseerror/entropy/medianfilter grid chỉvì CRAD báo gain.

**Dấu bác bỏ:** qualitymeasurement không thêm tín hiệu ngoài count; precision tăng chỉdo prior;
hoặc tradeoffbenign/TD khiến macro không qua gate. Nếu chỉ giữ specialistconfidence mà không biết correctness, chưađủ.

**Chi phí:** TP/FP/FN 16×3×int32 =192B/xe/round trướcheader/nén; raw16×16CMint32=1024B.
Đây là upperbody đầyđủ, cóthể sparse; softQ ngoài fit tính riêng. Extra inference/auxtrain thường đắt hơnstats.
Không dữliệuthật đi server; không hứa DP hoặc bảo mật tuyệtđối.

### 4. Tách disagreement giữa expert khỏi biến thiên trong class — dự phòng

**Evidence:** Qmean không phân biệt confidenceổn định với mixtureprediction.
Betweenclientvariance cóthể tính ngay từ Qclientđãcó; withinclient cần thêm `E[p²|y]`.
LegacyP30 khớp `Cov(p,x)` khi tối ưuinput, không directoutputmoment; nhưng chưathử không đủ lý do train.

Tiền lệ *Knowledge Distillation of Uncertainty* (NeurIPS2025) học phânphối teacherensemble qua
mean/covariance **function theo input**, nhằm giữ uncertainty khi nénensemble.
Đó không phải classconditionalmoment trên cácinput khácnhau và không chứng minh macroF1 tăng.
[Paper tác giả](https://proceedings.neurips.cc/paper_files/paper/2025/file/7bad4edcac6b2f314d6aa25dd1446272-Paper-Conference.pdf).

**Diagnostic:** tính betweenexpertvariance trên góiQclient retained trước. Chỉ thêm diagonalM2 nếu
cần tách `Var_total = E_client Var_row + Var_client mean` với **cùng countweights**.
Kiểm variance còn liên quan lỗi future sau meanconfidence/count/classcontrol không;
nếu chỉtrainvariance hoặc không có matchedfutureprobabilities thì báo chưađịnhdanh, không launchvarianceKD.

**Chi phí:** thêm16×16fp16=512B/xe/round trướcheader/nén, hơn QuplinkCBB rấtđángkể;
fulltriangle16×136fp16=4352B. Không cần modelalignment; không gửirow.
Hạ ưu tiên vì giữmoment không đồng nghĩa tạo decisionboundary tốt hơn.

## Những paper đọc thêm nhưng không làm nhánh mới

[Selective-FD](https://pmc.ncbi.nlm.nih.gov/articles/PMC10774276/) dùng sharedproxysamples,
densityratio clientselector và entropyserverselector; thresholdclient lấy từ localvalidation.
Trong thí nghiệm benchmark, proxy là10–20% trainrow mỗiclass; không phù hợp yêu cầu không gửi realrows nếu copy cách dựngpool.
Biến thành seededKMproxy sẽ thành query/bankadapter, họ đã có nhiều FAIL. Không đềxuấtentropy/densitygate như ý tưởng chưathử.

[FedDKC](https://arxiv.org/abs/2204.07028) chủđề confidence/entropycongruence không đáp ứng
evidence cầncorrectness ngoài fit; không ưu tiên thêm sharpening/temperature sauB14 và lịch sử QT.
[Does Knowledge Distillation Really Work?](https://proceedings.neurips.cc/paper_files/paper/2021/hash/376c6b9ff3bedbbea56751a84fffc10c-Abstract.html)
báo fidelityteacherstudent không luôn đồng hành generalization; chỉ hỗtrợ lựa utility làmgate, không cho recipeCBBmới.

## Quyết định khảo sát

Ưu tiên **matchedsourceQ ngoài fit** và **diagnosticrecipientdomain**; làm hai việc riêng.
Weightquality chờ nguồnmeasurement; moment2 chờ bằngchứng bổsung ngoàimean/count.
Không đủ nguồn để khẳngđịnh một hướng chắcvượt APFed/CANPerFL hoặc tựpushbatchmới từ báo cáo này.
Thiết kế utility tiếp vẫn CBB controlkhôngSAM, DAGSNetW1, macro-first≥+.002/từngseed,
sim200×2 → pilot20cos50 → r50×2; gate/source/count/codec/byte/budget phải khóa trướctrain.

## Diagnostic recipient đã thực hiện trong phiên

[Script](../../../scripts/analyse_recipient_domains.py),
[kết quả](../../../runs/recipient-domain-review-20261111/report.md),
[số đầy đủ + SHA](../../../runs/recipient-domain-review-20261111/analysis.json).
V2/V3augmentation, V4/V5mixedinput, r20cos50; tái tính đủ10metric train/internalval và support/ID,
không officialtest. Chia client bằng SHA256 cố định thành2647/2696;
donor/recipient khôngtrùng. Metadata từ clients.parquet, không world/sender/nhãn để gán group.

Ước lượng `p(output=j | trueclass=c)` của **ownmodel** từ donor ở global hoặc group,
dự báo phân phối output trên recipientfuture. Tính Brier kỳ vọng theo từngtrueclass rồi mean16class;
Δ=global−group, dương là forecast tốt hơn. Đây không là predictionloss của classifier.
Conditionđủ16hàng giúp không lẫn thay đổi prior trueclass vào trọng số macro;
không loại confound từ donortrainingprior/covariates/count/temporalshift hoặc chứng minh transfer.

| sourceforecast → recipientfuture | group | V4 ΔmacroBrier | V5 ΔmacroBrier |
|---|---|---:|---:|
| train đãfit | scenario4 | +.00010108 | +.00008106 |
| train đãfit | scenario/run8 | **−.00013197** | **−.00013335** |
| future của donor khácclient, chỉdiagnostic | scenario4 | +.00211813 | +.00203803 |
| future của donor khácclient, chỉdiagnostic | scenario/run8 | **+.00263183** | **+.00261167** |

Futurepeer cùnggroup tốt hơn global ở16/16class, cả4arm;
V4/V5c11 riêng+.025171/.025031, TD+.006357/.006405, benign+.000534/.000503.
Ngưỡng donorsupport8/32 cho kết quả nhưnhau, không fallback group×class tại full5343.
Kiểm algebra Brier so bruteonehot, single-groupidentity và thiếu-donorglobalfallback PASS.

Countdiagnostic riêng trên **mainpopulation**: sim200 có nhóm `highway_2:test` chỉ3xe,
**không dòng class11**; groupfull tươngứng có53expert class11≥8row, effectiveexperts50,36.
Đây là `run` metadata nguồn, không split test huấn luyện/đánhgiá của protocol.
Tách8group có thể tạo thiếuclass ngay trong sim dù full cósupport; bắt buộc globalfallback và báo
hiệuứng populationchange, không gán groupmissing làquality0. Fullgroup c11maxexpertshare tới5,3%,
lớn hơn globalmaxselfshare0,46%; grouping cógiá về concentration, không miễnphí.

**Đổi thứ tự ưu tiên:** domain có association ngoài fit nhưng trainconfusion đãfit chưa khai thác được
đồngđều; không đủ để gửi8Qteacher ngay. Futurepeer có label/prediction tươnglai donor,
**không dùng làm teacher huấn luyện**, không phải upperbound chính thức hoặc gainmacroF1.
Ưu tiên nguồnđo hợp lệ trước, rồi kiểm personalizedteacher từ **cùng source** nếu còn tínhiệu.
TrainCM là postepoch `evaluate.own_cm` trêntrain, không onlineQ trainmode củaCBB;
không suy diagnostic này đã đo trực tiếp ranking/hạnchế của softQonline.

## Thí nghiệm tiếp đề xuất — matchedsource, chưa khóa launch

Ba mainarm, mỗiseed: **C**=CBB; **E**=Q đo ở earlyrow đãfit của shadow;
**H**=Q đo ở heldoutrow chưafit của **cùng shadow**. E là mechanisticcontrol,
không chỉ so H với C rồi gán toàngain cho ngoài fit. Hai seed cần2shadow+6main phase;
chưa có code/budget/quotagate hoặc pushB18.

Main dùng fulltrain, optimizer/forward/per-rowKD/KM/Q8 giữnguyên.
Shadow chỉearlyA, bootstrapKM/Qinternal/BN/weights không thấyH. Sau roundr,
evalmode/rawlogits nhưnhau cho hai nguồn; Q chỉ dùng ở mainr+1.
Trong mỗi ôeligible có≥8row cảA/H, match sốprobe `m=min(nA,nH)` bằng rowhash cốđịnh,
không match bằng confidence hoặc labelpredicted. Cùng rowcount, codec, countweight `n_main` cho E/H;
ôineligible fallback **cùng quy tắc onlineCBB của arm**. Đây là hybrid, không gọi pureOOF.
Tỷ trọngpopulationeligible98,3685%, c11 chỉ81,8160% trong sim; phải báo riêng thay vì ẩn fallback.

Điều kiện nhận H cần Δmacro≥+.002 so C **từngseed**, đồngthời H hơn E từngseed;
E/H khác eval/producer/countsupport hoặc sourceauditFAIL thì không diễn giải outsidefit.
Chưa chọn E/H bằng xem entropy/officialtest; nếu macroFAIL đóng đúng adaptation.
Gate này là **đề xuất trướctrain**, chưa là camkết launch hoặc proofutility.

Main+shadowsteps lowerbound1.811×, thêm matchedprobeinference tới589.968row/round,
hai bootstrap chính/phụ và Qinternalshadow + Qmeasurement + Qmainfallback.
Số589.968 là upperbound `2×heldoutrows`, không actualruntime/byteproof;
rowselection/countidentity/codec/livebudget vẫn cần triểnkhai và kiểm trướclaunch.
Chỉ stats/probability rời xe; auxmodel ở xe, schema labelspace hỗ trợ khácarchitecture,
thửnghiệm vẫnDAGSNetW1. Fixedscaler cóH nên ngoài fit chỉ conditional onfixedpreprocessing.
