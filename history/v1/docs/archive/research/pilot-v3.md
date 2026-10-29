# V3 — manifest carrier, khóa2026-10-11

Chủ dự án xác nhận **cả4carrier, fp32 hai chiều, private DAGSNet+KD ngay**.
[Thiết kế/nguồn](v3-carriers-sources.md), [tri thức bổ sung](v3-knowledge-sources.md).
Không sửa V2 đã PUSHED. Đây là đợt mới, không retry/tune A CountSketch đã trượt.

## Cố định

Fixture/protocol/numeric/BN/private CE+KD T2λ1 từr2 như [V2](pilot-v2.md),20round,seed42/43.
Mọi learned model DAGSNet; trusted aggregate≥5, không raw/perturbed/latent/per-record gradients export.
Local chỉ math/3round×256rows smoke; train20round Kaggle, W&B online trướctrain bắt buộc.
Optimizer AdamW reset mỗi client/round, wd0; clip1 sau gradient mask nếu rolling train.
Prior codec hiện hành được trả phí, giữ như đối chứng A. Không mở official test/fullscale/R2 khác kiến trúc.

## Carrier và arm

Tên bắt đầu `a_` để dùng chính CE+KD/BN-policy A. Private+proxy sau local epoch đều được lưu.

| family | small | medium | arm/cỡ |
|---|---|---|---|
| R random-coordinate |k512|k24576|noEF,EF,ownEF|
| T rolling-coordinate |k512|k24576|communication-only,train-only,own train-only|
| O fresh orthogonal right projection |rank1|rank32|noEF,EF,ownEF|
| P adaptive two-stage projection |không fitB0|rank8|noEF,EF,ownEF + randomO rank8 EF/ownEF|

R uniform subset seed recipe `SeedSequence([2401,round])`, không scaleD/k.
T mỗi tensor permutation seed2402, phân bổ k theo tensor size, tối thiểu1/tensor;
Hamilton remainder theo tên/tensor-order cố định. Quay block modulo tensor length, không pad message.
T communication-only train full rồi chọn cùngmask; train-only gradient ngoài mask0, assert delta ngoài mask0.
O/P packing395024→512×772,240paddingzero; public Gaussian Q(seed2403,round), reducedQR,
canonical sign R-diagonal; Q fresh từnground, không warm-start. Rank8 O control dùng cùngseed P.
P stage1sum A_iQ→QR(P)→broadcastdecodedleftfactor; stage2sum A_iᵀleft→broadcastrightfactor.
Không thay hai stage bằng hai factor local rồi cộng. Cohort cố định20 ở U.

EF **tại xe, parameter space**, residual unweighted local delta. U counts n_i cố định:
target_i=delta_i+residual_i; upload n_i×compressed(target_i), packetcount=n_i;
RSU sum/n_total, fp32 decode. Residual_i=target_i−decompress(decoded own upload/n_i).
Đây là bookkeeping tương đương weighted residual với fixed n_i; finite-contact đổi cohort phải thiết kế ownership riêng.
own dùngcodec cùngcỡ tại xe, không trao đổi proxy/residual; counts/prior chung giữ như A.
Mọi residual/own snapshot/model version commit cùng marker. P partialstage không commit.

Payload dùng32Bheader V2 coefficient vectorfp32, uploadcount8B và16Bcontext mỗi message;
context băm configchung,baseweights,mask/basis,round/stage; sai context bị từ chối trước sum.
Không gửi indices vìmask/seed public.
P có4message thật (Pup,Pdown,Qup,Qdown), không giấu factor đồng bộ.
U ledger chỉ application bytes; chưa ACK/frame/crypto. Byte R/Osmall4,200B/xe/round;
prior200B một lần→4,400B trước ACK/bootstrap. P rank8 có82,384B/xe/round trướcACK.
**Không claim B0 đạt** trước contactstage.
Metadata/layout/seed/hash/catch-up/ACK phải trảphí trong manifest C riêng, không miễn vì seed.

## Đối chứng và cổng

A đã verify có cùng driver/private recipe/fixture/runtime: local,a_own là reference có provenance;
**a_dense32 mới chạy cùng R**, truyền densefp32 để cùng precision với carrier, không dùng densefp16 A làm
retention denominator chính. Densefp16A chỉ đối chiếu lịch sử; khácquantization được ghi rõ.
không lấy metric từ P2 thiếucheckpoint. V3 thêm owncodec để tách lợi FL khỏi local codec/training.
Local smoke kiểm tương thích shared CE/KD path; nếu baseline recipe khác thì phải chạy baseline mới trước nhận cổng.

Hai seed r20: teacher dense32head−localgp≥0,05; dense32private qua common.
Carrierprivate−a_own≥0,01, giữ≥80%positivegain a_dense32−a_own;
carrier−owncodec≥0,01; groupgp−localgp≥0,03; oldgp>localgp; benignF1≥local−0,03.
EF−noEF≥0,005 để claimEF; T train−comm≥0,005 để claimmasked optimization;
P EF−O rank8EF≥0,005 để claimadaptivebasis. Thiếucontrast không tự bác bỏutility nếucổngchung qua.
Không chọncỡ/head/bestround sau điểm. Numericfail/không đủr20→FAIL theo cấu hình; operationalstartupfail→thu trước xétretry.

M-gate: real-update projection/residual identity,norm không khuếch đại với sốhọc tolerance đã khóa;
cohort/base/schema/codec/framecut; T outside-mask đúng0; P two-stage/abort; actualbytes;
allarm3roundsmoke,exactresume đại diện4carrier,tamper; T4 production-rank preflight cả2GPU.
QR orthogonality Frobenius≤1e−5, residual identity atol2e−5/rtol1e−5; không nhậnuniformcontraction finitefloat.
Teacher/private metric từpredictionconfusion mỗi client, đủ10metric/perclass, checkpoint/hash/source audit.

Chỉcỡ/hướng U qua mới contactstage với cổngV2:cohort≥5,timeout5round,1contact/round,
≥16/20xe activate,retention≥70%,gain≥0,01; P phải mô phỏng2RTT vàdeadline thật.
