# V5-H — K2b label-kernel soft target, teacher DAGSNet tại xe

2026-10-12 UTC+7, khóa trước Kaggle. Chủ dự án cho phép soft target làthống kê gộp;
mọi model học DAGSNet. Đã chốt bandwidth train-only: h∈{1,2,4,√66}, nhỏnhất qua
coverage≥90%tổng/≥80%từnglớp; khôngqua dừng trước teacher. Không sửa V3/V4 đãpush.

## Cơ chế, giới hạn

K2 cũ residualhàm cần đồngbộ probeDAGSNet790080B/snapshot. **K2b làadaptation khác**:
không gradient/residualcủaθ, khôngteacher-vote. Client tính `H_imy=sum_{local label y} exp(−distance²/(2h²))`;
secure-sumH/countclasses rồi `t_my=(H_my/n_y)/sum_c(H_mc/n_c)`. T là thống kê target trênpublicanchor,
khôngclassifier triểnkhai. ChỉDAGSNetcompanion học/đánhgiá. Khônggọi bảnnày FFGB nguyênbản hoặcnoveltyproof.
[Nguồn/bảncũ](v3-knowledge-sources.md), [probe](../../papers/v5/anchor-probe.json).

Gaussianbootstrap nhưV2, actualdecodefp16mu/sd. M32small và512medium, labelstratified16class;
anchors từ noise PCG64seed4902+M, rồi **encode/decode anchorfp16** thật trướcquery/train.
Distance Euclidean theo D=class-balanced mixtureSD từdecodedmu/sd, floor.01.
Khôngfitbandwidth bằngholdout. Coverage `max_m kernel>=1e-3`, fulltrain tại mỗiworker trướcteacher.
Wire hfp32 được ánhxạ optiongầnnhất trong4candidatepublic cốđịnh; không dùngmột continuoush khácsauquantization.
4hcoveragequery đồngcohort/canonicalanchors; đếm theo lớp gộp, không perrecord/kernelweight rời xe.
Client cóthểcache4Hlocal nhưng chỉH củaselectedh gộp. H/chunk/querygradient khôngexport.
Mọi real/label được truyvấn tại xe; RSU chỉđọc tổngcohort≥5 trongemulator.

Teacher local20DAGSNet, init chung nhưV2; không traođổi parameters/optimizerstate/BN.
Mỗiround recalibratecanonicalBN bằng4096sharedGaussian, freezeBN/dropout;20full-anchorsteps,
AdamWwd0reset,cosine lr1e-3→1e-5,clip1,AMPfp16/FP32loss. SoftCE `−sum t*logsoftmax(logits)` T=1,
hardGaussiancontrol cùngloss onehot. Đâyteacherfitting, futureprivateKD giữT2 nhưA nhưng chưa mở.
Decode targetfp16 rồi renormalizerows; all-zeroHrow dùnguniform16, ghi rủi ro không thayanchor.
Own targets chỉlocalH/countlocal, cùngGaussian/publicanchors/selectedh; không gọi own làkhônggiao tiếp hoàn toàn.

## Arm và cổng

FamilyH: `b_dense` fresh teacher upperreference, mỗiM `fl`,`own`,`shuffle`,`gauss`.
Shuffle publicpermseed4903+M đổiđịa chỉ targetrows; khôngrecords/labels thật.
Gauss hardlabels theo classkhởitạoanchor, không labelvotecủateacher yếu.
9arm×2seed42/43,20round,20client/2,5%fixture/alpha.5/66features; evalr5/10/15/20 old/group,head,
đủ10metric/perclass/benign. Role**teacher/proxy**, privateweights chưatrain khôngreportprivateIDS.
Realstatrows=n_i tách khỏioptimizerrows=20*M; applied+skipped=20, không giảcountquery thànhgradientsteps.

MỗiM/hai seed finalr20: teacherFLhead≥localV2privategp+0,05 và≥gausshead+0,01;
FL−own≥0,01,FL−shuffle≥0,01. Numeric/coverageFAIL hoặc khôngđủr20 đóngcổng;
operationalstartupfail thu/log trướcxétretry. Không chọnM/head/bestround/h khác đểcứu.
TeacherPASS mới privateKD vớiFL−own/shuffle/diag,commonold/benign guards trongmanifest mới.
Dense reference đểthấyK2b mấtquality haytri thứcglobal khôngcóích, khôngdenominatorretention K2b mới.

## Byte/coldstart

Một lần/xe:Gaussianmoments8552up+4256down,prior200; anchors4256(M32)/67616(M512)down;
coverage360up+100selectedhdown; H2088/32808up; targets1056/16416down.
**FL/shuffle totals20868/130308B/xe**, trướcframe/ACK/schema/security; own vàgauss khônggửiH/target.
Coverage→selectedh làphase riêng: coldstart3logicalphases(moments/anchors,coverage/h,H/targets),
không giấuRTT. M32 B0 khôngfitcoldstart1contact; M512B2 applicationfit chưa deadline/cohortproof.
20teacherrounds khôngnewknowledgeupload/down; khôngdenseθsync miễnphí hayseedBNreconstruction claim.
Publiclayout/hgrid/seed/schema/bootstraphash/catch-up phải trảphí contactstage riêng.

## Preflight và bằngchứng

Local40k sampleh4 coverage≈98,8/99,2%, nhưngtargetentropy≈0,983/0,982: gầnuniform.
CoveragePASS khôngteacher/F1PASS; chưa fulltraingate. Localdriver3round chỉsmoke.
9armtiny, exactsmallFL/mediumownresume; actualanchors/targetpacket/countsteps/source/weights/pred→10metric.
Kaggle2T4 pinnedTorch2.10 actualmatmul/codec/teacherbackwardpreflight; W&Bonline/key trướctrain.
Pullterminal/executedsource/fixture/fingerprint/hashchain/sampledinference; chỉsauaudit nhậncổng.
TrustedaggregatesimulationchưaDP/crypto; group/oldholdoutinternal cógiới hạn scaler/dataset kếthừa.
