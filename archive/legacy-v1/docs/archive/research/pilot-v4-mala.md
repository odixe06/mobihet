# V4-M — MALA tại xe, hợp đồng kiểm khả thi

2026-10-12 UTC+7. Chủ dự án đã chốt τ=0,05, cap=10, eta=0,01 trong tọa độ chuẩn,
20 burn-in/xe/round rồi1transition/privatebatch; không record nào rời xe.
Manifest khóa trước Kaggle; chưa kết quả F1. [C đãtrượt](probes/pilot-v2.md) là lý do kiểm adaptation mới,
không kéo dài/tune C để cứu cổng. Mọi mạng học được DAGSNet.

## Bằng chứng → cơ chế → điều bác bỏ

C weighting thua Gaussian hard-label và shuffle; ESS fraction≈0,007–0,024 từnglớp.
Weighting chỉ chọn lại điểm đã draw từ Gaussian, có thể ít điểm mang trọng số hữu ích.
MALA thay sampling tại xe: chuỗi chuyển điểm Gaussian về vùng có ratio lớn, rồi CE synthetic khôngimportanceweight.
Đây là giả thuyết, chưa chứng minh ESS là nguyên nhân duy nhất, chưa bảo đảm20burn-in trộn chuỗi.
Nếu không hơn weighted-Gaussian/own/shuffle và không qua guard thì dừng, không tăngburn-in để cứu pilot.

[Roberts–Tweedie1996](https://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/Langevin/RobertsTweedieBernoulli1996.pdf)
nghiên cứu khuếch tán Langevin/discretization và Metropolis correction. Nguồn không phải paperFL;
ratio DAGSNet + Gaussian và cấu hình ở đây là adaptation dự án, không mượn bảo đảm hội tụ ngoài giả định nguồn.
JEM thuần `sum exp(logits)` trên R66 đã loại: headLayerNorm khiến logits bounded, integral vôhạn.
Gaussian-base dưới đây có chuẩn hóa hữu hạn; không cần thay architecture để cứu JEM.

## Mật độ và bước chuyển

`q_y=N(mu_y,sd_y²)` từ thống kê train gộp qua codec V2 thực, cùng floor1e-4 và fp16decoded như C.
Ratio DAGSNet học NCE trên local perturbed real với τ.05 và negative Gaussian, class-normalize như C.
Raw/perturbed record, score từngrecord, chain và latent chỉ tồn tại tại xe.

Với `z=(x−mu_y)/sd_y`, đặt

`ell_y(z)=−||z||²/2 + clamp(r_y(mu_y+sd_y*z),−10,log(10))`.

`exp(clamp(r))∈[exp(−10),10]`, nên Gaussian-base tích phân hữu hạn/dương.
Gradient theo z tính FP32, teacher eval/BN frozen/dropoutoff; không backprop vào tham sốteacher.

`z'=z+eta*grad ell(z)+sqrt(2eta)*eps`, eps standardnormal, eta.01.

`log alpha=ell(z')−ell(z)+ (||z'−z−eta*g(z)||²−||z−z'−eta*g(z')||²)/(4eta)`.

Squarednorm/acceptance ratio FP64; so với loguniform, không exponentiate để tránhoverflow.
Proposal/score nonfinite fail; không âm thầm skip hoặc thay sampler.
Pool256chain, nhãn luân phiên16class; mỗi xe/round khởi tạo noise chuẩn,20burn-in,
1transition trước mỗi privatebatch; sxdetach trước syntheticCE. R1 private chỉ Gaussian nhưC,
từr2 sử dụng ratio version trước đó. Chain không xuyên round: ratio snapshot đổi cần burn-in mới.

Shuffle control dùng permutation công khai của66tọa độ z: evaluate ratio tại
`mu_y+sd_y*Pi*z`; prior standardnormal bất biến vớiPi. Khi nhận chain vẫn dùng địa chỉ x gốc.
Giữ phân bốpotential dưới q và compute, phá địa chỉ feature; không shuffle record thật.

## Arm, protocol, cổng

FamilyM: `diag`, `c_fl` weightedGaussian fresh, `c_mala_fl`, `c_mala_own`, `c_mala_shuffle`.
Fresh controls cùng wrapper/source/runtime/fixture; C cũ chỉ lịch sử, không thay freshcontrol.
Own giữ ratio chỉxe, chung prior/q moments nhưC để cô lập trao đổi ratio, không gọi own là local-only hoàn toàn.
Không reuse weight teacher đãtrain trên holdout; mọi arm bắt đầu đúng seed/init.

20client,2,5% fixture,alpha.5,66features; seed42/43,20round×1localepoch;
privatebatch512/aux256, AdamWwd0 reset/round, lr1e-3→1e-5 cosine,clip1;
AMP private/NCE nhưV2, FP32 inputgradient khôngAMP, FP64 acceptance norms.
Group/private/gp r20 headline; old/head/gp/benign/perclass và10metric mọi evalr5/10/15/20.
Không official-test tuning. W&B online/key bắt buộc trước train.

Hai seed độc lập đều cần:

- MALA_FL−weightedGaussian ≥0,015;
- MALA_FL−MALA_own ≥0,010 và−MALA_shuffle ≥0,010;
- MALA_FL−diag ≥0,015;
- group/gp−localV2 ≥0,03, old/gp>localV2, benignF1≥localV2−0,03;
- proposal/gradient/chain/weights hữu hạn, resume đúng; báo acceptance/ESJD mỗixe/round,
  tỷlệ acceptance ngoài[0,1] là lỗi implementation. Không dùng threshold acceptance để chọneta sauF1.

Không lấy ESS của weights làm gate samplerMALA: không importanceweights trongCE mới.
Đo chain autocorrelation/displacement và histogram radial chỉ diagnostic, không chứng minh density fit/mixing.
Nếu quality qua mới tách λ/cost và kiểm contact; chưa mở fullscale hoặc nhiều client.

## Byte, FL, giới hạn

U stage dùng aggregate ratioDAGSNet dense nhưC: uploadfp32weightedupdate395024weights+header/count,
downloadfp16snapshot790080B/xe/round. Prior/moments thực trả phí; không seed miễn phíBN/weights.
MALA không thêm uplink record/chain, nhưng **không tự giải R1**: dense teacher vượt B0/B2 mộtcontact.
Chỉ sauqualityPASS mới ghép carrier đãquaV3 và kiểm contact/deadline/bootstrap/ACK/catch-up riêng.
Teacher ratio fullDAGSNet không phải classifier triển khai; private DAGSNet mới là IDS report.
R2 khácarchitecture chỉ khả thi giao diện, chưa thực nghiệm.

TrustedSecureSum simulation cohort≥5 nhưV2, chưacrypto/DP. Gửi tham sốratio/gộpmoments,
không gửi raw/perturbed/syntheticchain. Lowquota account ưu tiên khi cósession/quota thật;
không chen vào account đã2kernelV3. Không Git/DVC/pushartifact.

## Preflight bắt buộc

- Null standardGaussian MALA: kiểm dấu MH bằng biểu thức exactGaussian, stationarymoments,finite.
- Inputscore DAGSNet so với finite-difference; restore requires_grad/BN/dropout.
- 5arm×3round nhỏ, strictweight/payload/pred→10metric; exactresume cả3MALAarm vàchains.
- Actual2T4 Torch2.10 preflightscore/transition trướcworker; metadata/sourceimmutable/privatekeyboundary.
- Artifactcollect + exactexecutedsource/fixture/fingerprint/weights/chain/diagnostics sauterminal.

Local preflight không F1feasibility hoặc T4speed. [Ledger](../methods/gaussian-sampling/builds/support-mala/results/rebuild.md).
