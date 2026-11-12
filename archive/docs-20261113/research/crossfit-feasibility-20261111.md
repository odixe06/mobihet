# Cross-fit uplink — feasibility 11/11/2026, chưa launch

Owner đồng ý thử cải tiến mới11/11. Hướng khác [SAMb15](../../../docs/research/screening-sam-b15.md), không tự ghép combo.
Mục tiêu: thêm tín hiệu ngoài fit, thay vì coi confidence train là quality.
Đây là thiết kế khả thi đang hoàn thiện, **chưa có classifierutility hoặc launchgate**.

## Nguồn bằng chứng

[B14](../../../docs/research/screening-uplink-b14.md): LG teacherdiagonal≈.994 vẫn macroFAIL; Q hiện thu trên realrow mà model đã học qua nhiều round.
Postepoch/preupdate/entropytrain không biến thành ngoài fit. Modeltrộn vẫn phải hỗ trợ theo protocol.

Đếm trên `cache/fold_counts.npy` (sim dùngfoldtrain, panel200seed7) và
`evidence/research-data/train_counts.npy` (fulltrain+val); SHA và số đầy đủ ở
[support.json](../../../runs/crossfit-feasibility-20261111/support.json).
Giả định phân đôi **cân bằng theo (client,trueclass)**: A=floor(n/2), B=n−A.
Đây là countfeasibility, chưa chọn/chia row thật hoặc fit model.

| | sim200 | full5343 |
|---|---:|---:|
| realrows | 1.476.064 | 37.361.273 |
| ôclient×class có n≥16 (mỗi nửa≥8) | 2.501/2.767 | 60.700/69.993 |
| tỷ trọng realrows trong ôeligible | 99,860% | 99,798% |
| eligible riêng suddenConstantSpeed | 96,652% | 97,000% |
| activeclientsteps main/round (floor16) | 23.692 | 607.110 |
| activeclientsteps hai shadow/round cộng lại | 25.229 | 652.278 |
| tỷ lệ shadowsteps/mainsteps | 1,065× | 1,074× |

Hai shadow nhìn tổng cùng số realrows như main, nhưng floor làm thêmsteps; còn một forward đánhgiá
ngoài fit mỗi row/round. Main+shadow không chỉ “thêm16entropy miễn phí”. Không đổi floor main để ghép gain.

## Thiết kế nguồn ngoài fit có thể triển khai

- Chia cố định A/B từ **train của chính protocol**, không officialtest hoặc foldval của sim.
  Dùng stablehash/seed/rowindex, cân bằng theo client/class; artifact identity/rowset phải audit.
- ShadowA là DAGSNet của client, train **chỉ A**, generatorKMA từ pooledA, QinternalA chỉ từA;
  đánhgiá B trong evalmode, không update BN/weight, không dùng B cập nhật bất kỳ state/generator/QinternalA.
  ShadowB đối xứng. Không dùng mainmodel hoặc mainbootstrap trong shadow vì chúng đã chứa holdout.
- Main vẫn train **toàn bộ realtrain**, không bỏ holdout để làm đẹpvalidation.
  Prediction trueclass trên hai nửa ngoài fit có thể gộp thành QCF16×16 và error/count theo lớp;
  architecture của model riêng không cần khớp parameterbasis của clientkhác.
- Không gọi qualityshadow là qualitymain. Cần arm riêng kiểm **teacher cross-fit**, trước khi thêm cơ chế
  inverse-error weighting hoặc gatequality. Không dùng confidence thay error ngoài fit, không gán gain từ lý thuyết.
- Với ôn<16, chưa đủ8mẫu mỗi nhánh: phải khóa fallback hỗn hợp với Q8baseline/counts hoặc đánhgiá chỉ tín hiệu
  pooled ngoài fit. Chưa quyết semantics; không launch khi chưa rõ. Không gọi hybrid là pureout-of-fit.
- QCF đo tại roundr chỉ có thể dùng ở mainr+1 nhưCBB; không dùng tương lai của shadowr+1 hoặc best-roundselection.

## Vì sao tách pha shadow thay vì ba model thường trú

Fullmainparameters fp32≈7,870GiB/toànmạng =3,935GiB/rank2T4.
Ba modelclient (main+2shadow)≈11,805GiB/rank chỉweights; trainshard≈2,679GiB và testshard≈.670GiB,
tổng≈15,15GiB **trước** gradient/optimizer/activation/temporary. Không đủ headroom T4.
Các số là phép tính lowerbound, chưa đo peakruntime; không thử nhét ba model đồng thời rồi giảm batch để vừa.

Phương án khả thi về dependency:

1. ShadowA chạy20round với source/generator/Qinternal chỉA, lưu statsoutofitB từnground.
2. ShadowB chạy20round độc lập và có thể đồng thời trên kernelkhác, lưu statsoutofitA.
3. Sau source/split/count/codec audit, mainkernel đọc các bảngteacher đã khóa theo round;
   không cần mang shadowweights sang main hoặc gửi chúng cho clientkhác.

Hai shadow có trạng thái riêng từinit, Qinternal riêng, teacher không nhận mainfeedback;
vì vậy có thể tính trước, không có vòng phụ thuộc vào main. MainQuplink sẽ không còn cùng vai trò CBB;
phải tính bootstrap hai shadow + uplink/downlink nội bộ hai nhánh + truyền statsoutofit/main, không chỉ fileQcuối.
Tối đa≈3bootstrap nếu giữKM16/lite6 mỗi nhánh; không được gộp cả3schema rồi gọi communicationbaseline.
Đây là tham gia thêm training phụ tại client; deployment có thể chạy tuần tự, không mặc định người dùng có ba GPU.

## Còn phải chốt trước code/train

1. Làm rowpartition thật, kiểm disjoint/completeness/client/class/fold/SHA và không đường holdout→shadowbootstrap/Q.
2. Chốt ôsmall/fallback, codec/count/qualityschema; viết packetreplay audit.
3. Chốt estimand: QCF ngoài fit thay Qonline; giữ loss/generator main. Nếu muốn reliabilityweight phải là arm khác.
4. Tính walltime/quota **toàn bộ các phase và seed**, local khả thi rồi mới notebooksource/statdataset.
   Shadowstate không cần tải về local, các bảngaggregate có thể mount sau audit.
5. Utilitygate vẫn b13 từngseed; không xem officialtest để chọn weighting/temperature.

Chưa khẳng định cross-fit sẽ tốt hơn: train ít hơn mỗi shadow, QCF có thể mềm và gây tradeoff nhưQT2 cũ;
khác biệt là nguồnngoài fit, không đổi tên temperature. Chỉ chạy sau khi source/semantics/compute/byte đã khóa.

## Sau B16 — partition thời gian thật đã đếm, chưa train

[B16](../../../docs/research/screening-mean-b16.md) MK/HK đềuFAIL; không mở lại loss/dose qua nhánh này.
Đã chạy probe metadata trên410parquet **train/**, chỉ chọn200xe/foldtrain hiện hành;
không dùng row foldval hoặc officialtest. Source/data hash/count ở
[probe](../../../runs/temporal-feasibility-20261111/probe.py),
[support](../../../runs/temporal-feasibility-20261111/support.json),
[partition](../../../runs/temporal-feasibility-20261111/partition.npz).

Probe engineering cố định cutoff tại rowceil(.8n) của receive-time từngxe;
giữ toàn bộ timestamp ties phía sớm, phía muộn làm heldout. Không stratify theo nhãn.
Đây là **đo feasibility**, chưa khóa80% làm hyperparameter cho utility hoặc train teacher.
Counts A+H khớp từngclient/class với cachefoldtrain, mọi xe cócả2phía,
max(timeA)<min(timeH). Twin cùng receive-time của cùngxe không bị chia hai phía.
Không bảo đảm độc lập mọi sự kiện khácxe hoặc feature-history; chưa có embargo.

| số đo sim200 | kết quả |
|---|---:|
| early / heldout realrows | 1.181.080 / 294.984 |
| ôclient×class có≥8row cả2phía | 1.993 / 2.767ômặt |
| populationmass trong ôeligible | **98,3685%**, không99,8602% balancedhalf |
| heldoutmass trong ôeligible | 99,1040% |
| suddenConstantSpeed eligiblepopulation | **81,8160%**, không96,652% balancedhalf |
| feignedBraking / accMult eligiblepopulation | 91,0351% / 92,7861% |
| classpriorTV A↔H perclient median / p90 / max | .15931 / .38093 / .82308 |
| earlyshadowsteps / mainsteps mỗi round | 19.211 / 23.692 |

Single early→late shadow là phương án có thể khảo sát rẻ hơn hai complementaryshadow;
modelaux vẫnDAGSNetW1, main vẫnfulltrain và giữ per-rowKD. Main+shadowsteps lowerbound≈1,811×main,
còn heldoutinference294.984row/round, bootstrap/probe/packet/I/O, không FLOPs/elapsed proof.
Hai complementaryshadow cũ và singletemporalshadow là thiết kế khác; chưa chọn bằng utility.

**Bài học chặn launch mù:** classprior thay mạnh; Sybil từ6,65%early lên13,13%heldout.
Precision thô không tách producerquality với prior. Nếu dùng precision làmweights,
cần chuẩn hóa theo priorreference khóa từ **train**, classconditional CM/probabilities và missing-classfallback;
không chọnweights qua simval/officialtest hoặc coi ôkhôngsupport làquality0.
Lớp11 chỉ58xe đủ8row/cảphía: phải bảo vệ support/uncertainty vàfallback trước launch.

**Giới hạn preprocessing:** v4 scaler fit từ originalfoldtrain, trong đó có nestedheldout H.
Giữ77scaledfeatures chung theo protocol; không tự fit scaler riêng. Teacheraux chỉ được gọi
**model-out-of-fit conditional on fixed preprocessing**, không end-to-end independent/out-of-fit toànpipeline.
Generator/Q/BN/weights của shadow vẫn phải tách H; fixedscaler không là lý do cho phép H vào classconditionalbootstrap.
V20 train→originalval khác nested split này và đã chạy; không gọi probe mới là phát hiện temporalvalidation chưa từng có.

Tại mốc probe sauB16 chưa có score chất lượng/utility. B17 sauđó thửprior/real-NCKD vàFAILmacro;
việc tiếp cho nhánh ngoài fit: khóa estimand
(teacher từ nguồnngoàifit hay weight expert), control cùngshadow/source/evalmode,
fallback/precisionreferenceprior/codec/byte/cost và localaudit trước bất kỳbatchtrain.

## Diagnostic quality từ artifact V20 — sauB17, chưa train mới

Đã dùng CM **train32.685.166row / internaltemporalval4.676.107row**, full5343DAGSNetW1,
V2/V3augmentation vàV4/V5mixed-input, hai seed. Source/config/CM/SHA ở
[analysis](../../../runs/temporal-quality-review-20261111/analysis.json),
[bảng](../../../runs/temporal-quality-review-20261111/report.md).
`scripts/analyse_temporal_quality.py` tái tính r20 **đủ10metric cảtrain/val**, support/IDkhớp;
không officialtest, không fitweights hoặc launchtraining.
Đây là diagnostic mới trên thửnghiệm V20 **đãchạy**, không gọi temporalvalidation chưa từng thử.

Để tách bớt binaryprior, dùng precisionone-vs-rest tại **cùng globalprior từtrain**:
`P*=πc·recall / (πc·recall + (1−πc)·FPR)`.
Không chuẩn hóa được mixture các classâm; thay covariate/negative-mixture vẫn ảnh hưởngP*.
Chỉ so ô có cảpositive/negative≥8 hoặc≥32 trong **cảhai split**, báo sensitivity cảhai threshold,
không chọnthreshold để lấy correlation đẹp. Khôngpredictedpositive gánP*=0 theo diagnosticconvention,
không loại xe dựa trênprediction: giữeligibility chỉtheotruesupport. P* là diagnostic, không headlineprecision thực.

Ví dụ V4/V5, ngưỡng32:

| class | medianP*train / future s42 | medianP*train / future s43 | SpearmanP*train↔future s42 / s43 |
|---|---:|---:|---:|
| benign | .9706 / .8243 | .9718 / .8260 | .2104 / .2445 |
| accMult | .9962 / .9168 | .9960 / .9134 | .0549 / .0271 |
| feignedBraking | .9965 / .9475 | .9962 / .9462 | .0445 / .0024 |
| timeDelay | .9229 / .4358 | .9230 / .4358 | −.1127 / −.1274 |

TDcorrelation ởthreshold8 lại+.0420/+.0504 với2687xe/cảseed, so1492xe/cảseed ởthreshold32.
Đây là **support/population sensitivity**, không prooftrainquality luôn tươngquanâm.
Count↔futureP* cũng tùyclass: accMult≈.374/.358, benign≈.027/.012 ởthreshold32;
không kết luận countluônvôích hoặc weightquality chắc hơncount.

**Evidence→mechanism→falsifier:** trainconfidence/quality không xếp hạng futureexpert tốt đồngđều,
vàB17 đổi reception/prior không đủmacro → nguồnđo ngoài fit là estimand cócơsở khảo sát.
Nhưng countweightedQ đang ước lượngpopulationmean, không phải đang hứa expertquality;
qualityweight sẽ đổi estimand, không gọi sửa“bugaggregation”. Cần kiểm teacheroutsidefit/qualityrank thêm gì,
giữ per-rowKD/maintraining, matchedteacher/evalmode/countsupport control; chỉutilitymacro mới choPASS.

Giới hạn: V20khácKM/lite6/Q8 CBB; hardCM không khôi phụcsoftQ vàownclientquality không chứng minh
cross-recipient transfer. **Chưa có B18/source/design/budget khóa hoặc utility mới**;
không launch gridconfidence/entropy/inverse-error từ correlation này.
