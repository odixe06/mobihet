# B15 — SAM local trên CBB, khóa 11/11/2026 trước train

Owner11/11 đồng ý thử các cải tiến mới sau [phân tíchb14](../../archive/docs-20261113/research/b14-analysis-20261111.md).
Giữ quyền10/11: performance trước, compute được tăng, byte giảm tối đa chưa capcứng,
schema hỗ trợ khác kiến trúc; **experiment chỉ DAGSNet W1**. CBB vẫn baseline.

## Evidence → mechanism → falsifier

Gap full-scale còn ở benign tương lai bị dự đoán attack; b13loss và b14interpolation/logpool đềuFAIL.
Teacher LG quá tự tin vẫn không đạt macro. B15 kiểm một cơ chế khác: giảm nhạy loss với perturb tham số
**riêng từng client**, không thay generator hoặc gộp model giữa client.

[SAM gốc](https://arxiv.org/abs/2010.01412), Algorithm1/AppendixC.3; [code tác giả](https://github.com/google-research/sam/blob/main/sam_jax/training_utils/flax_training.py)
`get_sam_gradient`: gradient đầu tạo perturbL2, gradient sau tại modelperturbed, commit BN innerstate;
stochastic key dùng lại. AppendixC.3 thử cùngrho.05 xuyên dataset; **không khẳng định đây là rho tối ưu cho VeReMi**.
Đây là adaptation SAM+AdamW+CBB, không reproduction CIFAR/SGD hoặc claim novelty.

Lợi cần thấy: macro≥+.002 từngseed, giữ guard, giảm FPbenign trên lớp1/6/7/11.
Bác bỏ nếu chỉ trainloss/sharpness tốt hơn, hoặc chỉ đổi benign↔TD/rare/xeítmẫu.
Không tunefloor/rho/temperature/outerweight sau nhìn utility; nhánhFAIL dừng.

## Cơ chế đã chốt

Chỉ2arm: **C** `sam_rho=0` delegate CBB exact; **SAM** `sam_rho=.05`.
Mỗi activeclient step:

1. Sinh64synthetic như CBB. Một forward chung64real+64synthetic (tailmask/floor như cũ),
   lossL=CEreal(z+LA)+KL(Q[y]||p_synth); r1 chưaQ dùng hardCEsynthetic.
2. g=∇L tại w, trướcclip/AdamW, unscale4096. Với norm>0,
   epsilon=.05*g/||g||2 theo **toàn bộ tham số của riêng modelclient**, gồm bias/BNaffine;
   gradient0→epsilon0; paddedinactive→epsilon0. Không perturb BNbuffer hoặc gửi epsilon/gradient.
3. Lặp loss tại detached(w+epsilon) trên **cùng rows/synthetic/dropout** và BNinputbanđầu.
   Không autodiff qua epsilon; clip1/AdamW thực hiện **một lần** ở w với gradientpass2.
   AdamWweightdecay/LR/resetmoment giữ protocol, không thêm L2 vào hướngperturb.
4. Q chỉ thu pass1, BN chỉ commit newBNpass1 (và chỉ khi applied+hasreal như core).
   Pass2 Q dùng scratch, BNdiscard; RNG sau bước đúng trạng thái saupass1.
   Paramsperturbed nằm bộ nhớ riêng, không add/sub tại w tránh lỗi restorefp32.
   Giữ flatparameter strides để compiledforward dùng cùng graphsignature.
5. Firstgradientnonfinite vẫn reject update, kể cả pass2 hữu hạn. Không chữa overflow bằng bỏclient.
   Ghi counter/finite/maxnormerror theo round; mọi realrow phải đếm đúng1lần.

Không thêm epoch/round, không đổi batch, synthetic64/floor16, prior, teacher, codec, kiến trúc hoặc đánh giá.
Hai gradient thay vìmột tăng gradientcompute xấp xỉ2×, không là speedoptimization.
0byte mới trong schema; byte nén thực có thể đổi vì student/Q đổi. Bootstrap giữ lite6/KM16w0.5;
báo đầy đủ bootstrap+uplink/downlink+tổng. Kiến trúc khác chỉ cần tự tính perturb trong parameterspace riêng;
không chứng minh heterogeneityfull-scale vì batch này chỉW1.

## Thiết lập và utilitygate

**4kernel = C/SAM ×42/43**, panel200xe seed7, foldtrain→val, cosine20/r20;
protocol/data nhưb13/b14, không officialtest. `SIM_BATCH=b15` dùng `methods/mobihet_sam.py`.
Control mới cùngsource/method nhưngrho0; không dùng controlb14 thay runmới.
Cap**3,75h/kernel**, cảbatch15kernelh+15%margin; 2T4/kernel.
Baseline b14t_train≈3000s/20r; dự trù SAM≈6000s +setup/compile/eval, cap cònmargin.
Đây là estimate, chưa đo SAMT4; quota tính sống/slot/exactowner trướcpush, không tự vượtquotalive.

Giữ **nguyên gateb13** từngseed endpoint20:
Δmacro≥+.002; Δp10≥−.006; Δbenign≥−.0005; ΔTD≥−.0035;
Δabsent≥−.01 nếu cảarm/control≥2000dòng. Low256/rare/c11/P/R/F1 từnglớp/TB16–20/byte/compute báo đầy đủ.
Audit source/config/LR/all20CM→10metric/finite/skip≤1%/mọiclientapplied + SAMcounter PASS trướcverdict.
PASS **hai seed** mới pilot; FAIL không tune/extend/ghépcombo.

Pilotfullr20cosine50s42: endpointmacro≥+.002; guardTB16–20 p10≥−.006, low256≥−.003,
TD≥−.0035, benign≥−.0005, absent≥−.01; báoabsentendpoint vìfullchỉ1555dòng.
PilotPASS→resume đúng2rankstate tới50 +seed43r50đồngthời.
R50từngseed: endpointmacro≥+.002, guardTB46–50 nhưpilot, so CBB cùngseed/round.
Hai seedr50PASS mới đổi phươngphápchính. Khôngchọn bestround/HPquaofficialtest.

## Localgate và actualaudit

- CPUunit: rho0hai round weights/BN/Q/loss/byte/RNG exactCBB; rho>0gradient so explicit2passreference
  cho hard/softteacher, realtail/floor-only/padded; firstoverflowkhông hồi sinh; resume đúng state+RNG.
- `check_sam_cuda.py`: realfoldtrain256rows, W1AMP/compiled/dropout, replaycùngweights+stride/RNG:
  logits/gradient/BN exact lầnhai. Chỉ proof computationalprerequisite, không wholetrajectory.
- Guardedlocal200xe W1/compile2round: r2tiêuthụQ, finitemodelstate/BN/teacher,
  CMsupport/10metric, ≤1%skip/mọi clientapplied; khôngutilitytuningtừsmoke.
- `audit_sam.py`: actualcounter mỗi round realrows=Σn, activesteps=Σmax(ceil(n/64),16),
  nonfinitefirst/second≤1%steps, errornormperturb bound<1e-6. Đây là counter+frozen-source audit,
  không replay toàn bộ privategradient. Source/config/CM/compile2rank audit trên kernelthật vẫn cần.
- Sourcefreeze → unit/CUDA/local → freshquota → privatepinned2T4notebook → push4run → collect1200s → audit/report.
  Giữ intent/receipt; ERRORpull; vận hànhlỗi thì sửa đúngstage, không restarttrainvìaudit/collector.

Evidence ở `runs/sim-kaggle-b15/` và `artifacts/sim-kaggle-b15/` saucollect;
job/owner/sourceownership ở [continue](../operations/continue.md).

## Reliability ngoài fit

Được owner đồng ý khảo sát; tách khỏi SAM để không ghép hai thay đổi.
Phải thiết kế nguồn out-of-fit và bootstrap/Q phụ táchholdout; không dùng entropytrain làmreliability.
Feasibility/nguồn/cost ở [kế hoạch cross-fit](../../archive/docs-20261113/research/crossfit-feasibility-20261111.md); chưa gán utility trước chạy.

## Kết quả đóng 11/11/2026 07:28 — SAM FAIL cả hai seed

Controller `mobihet-b15-20261111` succeeded07:28:38, artifact thật4/4kernel đủ20round.
Chạy lại source/config/CM/LR/10metric audit và actualSAMcounter PASS; compile2T4/2rank PASS,
skip0,23.692activeclientstep và1.476.064realrows/round đúngcount, normerror trong bound.
Audit không replay toàn bộ privategradient/modelweights. 14sourcefiles banđầu được kiểmSHA và giữ
`runs/sim-kaggle-b15/frozen-source/` trước cập nhật tài liệu/chuẩn bị batchsau; source-lock gốc giữ nguyên.

| | seed42 | seed43 |
|---|---:|---:|
| CBB macro | .914625 | .913287 |
| CBB+SAM macro | .915156 | .914483 |
| Δmacro | +.000531 | +.001196 |
| Δp10 | −.007145 | −.010200 |
| Δlow256 (9xe, mô tả) | −.007102 | −.019823 |
| Δbenign F1 | +.005452 | +.005052 |
| ΔtimeDelay F1 | −.006887 | −.001443 |
| Δmacro TB16–20 (mô tả) | +.000539 | +.001206 |
| trainseconds20r CBB / +SAM | 3086 / 4863 | 3121 / 4925 |

**FAIL macro+p10 cả2seed, thêmTD seed42. Không pilot, không tune rho hoặc extend.**
Trainelapsed≈1,576×/1,578× control, không phải FLOPsequivalent; bootstrap giữ1.196.032B/xe,
round hai chiềuSAM≈315/314B vsCBB≈306/305B do nội dungQ nén khác, không thêm field.

Benign recall+.01497/+.01160 nhưng precision−.00227/−.00023.
TDprecision+.05821/+.05649, recall−.02163/−.01424; TD→benign thêm280/143row.
FPbenign→attack ngoàiTD chỉ giảm111/24row. Có tác động lên boundary, chưa giải quyết đồng thời utility/cohort.
MeanclientF1 low256 giảm cả2seed; pooledaccuracy/weightedF1 tăng không thay macro+guard đãkhóa.
Không gọi flatness đã được chứng minh là nguyên nhân của tradeoff hoặc genericSAM vôích ở mọi rho/dataset.

Báo riêng **CBB+SAM**, không gán gain tối ưu cho cơ chếFL; CBB vẫn phươngphápchính.
[10metric/byte](../../runs/sim-kaggle-b15/report-r20.md),
[CM/cohort](../../runs/sim-kaggle-b15/diagnostics.md),
[verdict](../../runs/sim-kaggle-b15/verdict-r20.json), [SAMcounter](../../runs/sim-kaggle-b15/sam-audit.json).
