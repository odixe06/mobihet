# Kiểm nghiệm mở rộng — 2026-10-12 UTC+7

Chủ dự án yêu cầu mở rộng hướng pilot có lợi và triển khai hướng chưa kiểm chứng;
không tạo run trùng nếu kernel tương ứng đang chạy.

## Quyết định mới đã chốt

- H1/H2:20client,100%train80%,50round,**chỉseed42**,7arm(local/full/diag/lr4/own/DML/DMLown). Batch512,cosine1e−3→1e−5,
  DAGSNet-only; test20% descriptive/full mỗi round, headline r50, không chọn HP từ test.
- Nyström:rank4/8,explicitWupper có tính byte,FP32upload/FP16factor-download;
  freshfull/lr4/lr8/diag +own/sign-shuffle; cổng tại design§7 đã được chủ dự án đồng ý.
  Pilotseed42/43; seed42-only chỉ áp dụng đợt full H1/H2.
- Hybrid:pilot2×2 local/Gaussianfull/DML/DML+Gaussian, thêmown controls;
  syntheticCE vào cảprivate/proxy, KD trênreal, BN batchtrộn. Seed42/43,20round.
- Mọi notebookW&B online trướctrain; kiểmroundđầu đến khiổnđịnh rồi **ngừnghealthpoll** theoquyếtđịnhmới.
  Collect/auditterminal tiếp; không đánhgiácổng chỉtừW&B. Có lỗi/bất ổn đang chạy:
  báokernel/link/bằngchứng và yêu cầu chủ dự án dừng; không tựcancelkernel.
  Numerical/deadline guard trong notebook vẫn dừng theo manifest.
- KhôngGit/DVCadd/push/cleanup; localchỉboundedprobe/smoke/collect/report.

## Run đã mở, không mở trùng

V6đã qua15smoke/4exactresume/strictinference;4privateinputREADY.
Pushversion1 lúc11:11UTC+7 ngày2026-10-12:

| family/seed | kernel | quota trước push h |
|---|---|---:|
| A42 | [minhtran0601/mobihet-v6a-s42-20261012-v1](https://www.kaggle.com/code/minhtran0601/mobihet-v6a-s42-20261012-v1) | 5,77 |
| A43 | [dinhan25/mobihet-v6a-s43-20261012-v1](https://www.kaggle.com/code/dinhan25/mobihet-v6a-s43-20261012-v1) | 6,24 |
| B42 | [trietbackup/mobihet-v6b-s42-20261012-v1](https://www.kaggle.com/code/trietbackup/mobihet-v6b-s42-20261012-v1) | 13,95 |
| B43 | [odixe0502/mobihet-v6b-s43-20261012-v1](https://www.kaggle.com/code/odixe0502/mobihet-v6b-s43-20261012-v1) | 14,94 |

Tài khoảnítquota ưu tiênpilot. CollectorfamilyA/B jobs6, tổng12connections;
log `papers/v6/collection-{a,b}-20261012.log`, GPUauditlock dùngchung.
W&Bregistry30identity tại [wandb-runs](../../../archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/probes/wandb-runs.json);
monitor [live-status](../../../archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/probes/live-status.json). Snapshot11:12UTC+7 chưa cóW&Bhistory,
không chứng minh failure. Source/version đã push giữ nguyên.

## Phần đang triển khai

1. H1/H2full:driver50round, actualcodec, source/datahash, checkpointresumable,
   fulltest/10metrics/clientCM/perclass. Đo3roundT4/budget trước launch chuỗirun dài.
2. Nyström:source/smoke/resume/pilotU theo cổng đãchốt; chưacontact trướcU.
3. Hybrid:contract2×2/own, smoke/resume/pilot; không đưa lênfull trướcutility.
4. K2/K4:đãchốt residualteacher-first/Gaussian-baseMH; khóa cost≤2×freshdenseCE T4 trướcpilot.
   K4unbasedR66 đãFAIL vềnormalizability; khôngrerun y nguyên.

W&Binline key nằm ignoredprivate_push0600 và private Kaggle versionhistory lâu dài;
không vào publicsource/log/chat. Không cần thao tácmanual cho bước đã thực hiện.

## Collect lúc12:51UTC+7

APIowner-scoped: A42/A43ERROR, B42/B43COMPLETE; khôngV6đangchạy cầnmanualstop.
W&B30identity cóhistory:18complete20round; G59trướccommit hoặc1round, chưaheadlineartifact.
Oldcollectorprocess khôngcòn sau phiên bịtạmngắt; khởi động lạifamilyA/B jobs6,
Range-resume qua helper; logs `papers/v6/collection-{a,b}-20261012-terminal.log`.
Seed42A333file/0missing đãthu; source/payload/model/prediction auditđanghoàntất,
seed43 vàfamilyB đangthu. Khôngrerunkernel.

Seed42G59cả6arm lỗi `V6 synthetic fp16 input overflow`; parentERROR bởiassertallcomplete.
Bafull66control đềucomplete20round. Giữbằngchứngnumericalpartial vàcontrol hợp lệ.
Đã có `scripts/report_v6.py` tínhquality/compressionriêng vàjointinteraction;
chỉAUDITED được vào report. W&Bfinished không thay complete-roundgate.

## Audit và launch tiếp lúc15:57UTC+7

V6 cả4kernel/30arm đãaudit,0missing.18control/G61 đủ20round;12G59numericalFAIL.
G61quality/compressionpairedFAIL; jointmask lợi tươngtựfull66, khônggeometryproof.
[Report](../methods/geometry-primitive-moments/builds/geometry-g59-g61/results/report.md), [cổng](../../../archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/probes/gates.json). Khôngrerun y nguyên.
Khôi phụcfixture67file vàmetadataV2/V6 chọnlọc610file exactMD5; khôngbackup/add/cleanup.

V7Nyström local10arm×3round/4exactresume/strictpayload/inferencePASS,
controllernormal/deadline/keyboundaryPASS.2privateinputđãupload; kiểmREADY trướcpush.
Owner minhtran0601 quota5,00h, dinhan25 quota5,47h; parent≤3,8h+dựphòng15%.
W&Bregistry20identity tại `papers/v7/wandb-runs.json`.
HybridV8 đãcósource/manifest; localboundedpreflightđangchạy.
K2/K4 cost≤2×freshdenseCE đãđượcchủdựán chốt trongphiên này.

V7pushhai notebookversion1 thànhcông15:58UTC+7; API16:00UTC+7 RUNNINGcảhai.
W&B chưa cóhistory trongsnapshot15:59; liveness/trainhealth cònUNKNOWN. Khôngfailureclaim.
CollectorV7 family n jobs6 đãmở, log `papers/v7/collection-n-20261012.log`.

HybridV8source local7arm/3exactresume/strictBN/payload/inferencePASS;2inputREADY.
Push16:10UTC+7:trietbackup/mobihet-v8h-s42-20261012-v1 và
odixe0502/mobihet-v8h-s43-20261012-v1,version1;W&B14identity.
V7W&B16:06UTC+7:8control đangr2,loss1,40–1,47,skips0–1/1965steps;
finite/tiến độ cóhistory, chưagates. NewNyströmarm chưađếnslot.

## Teacher/full source chuẩn bị16:38UTC+7

K2V9:13boundedarm/3exactresume PASS trênsource trướcloggingfix; V9final3 đangkiểm
source cuối. K4V10:7arm/3exactresume PASS; final2 queued cùngGPUlock.
Đãchốt samplerprobe/cổngV10 tại r1/5/10/20; khóa trướcpilot.
TeacherW&B preflight dùngcommit=False đểkhôngadvance step trướcroundsauresume;
clientresidualapplied0 táchpublicteacher_applied20/400. Costcontroller actual2round
algorithm_seconds/freshdenseCE, checkpoint tiếp đúng20r nếu≤2×; khác thìcostFAILgiữ2commit.

V9prepared ownerodixeuit/minhtrit06, W&B26identity; input đangupload.
V10prepared ownertrietbackup/odixe0502; cầnfinalsmoke+notebookcost/deadlinechecks/inputREADY.
Fullsource trong `papers/full-validation/proj/`,7arm/50r/s42, localboundedcheck queued.
Fullcache chỉKaggle/temp, khôngrawpreparedarray ởworkingoutput; testscaleonce.
ChưaK2/K4/fullPUSHED; khônggọisourceprepared làtrainingđãbắtđầu.

V9push16:48UTC+7:odixeuit/mobihet-v9a-s42-20261012-v1 và
minhtrit06/mobihet-v9a-s43-20261012-v1; API16:55UTC+7QUEUED, khôngduplicatepush.
V10push16:56UTC+7:trietbackup/mobihet-v10e-s42-20261012-v1 và
odixe0502/mobihet-v10e-s43-20261012-v1,version1.
Userchốt: đãổnđịnh thìngừnghealthpoll, collect/auditterminal tiếp.
V8seed42 chưaW&Bhistory; thửexecutednotebook lúc16:55 trả404 khiRUNNING,
khôngtrainingfailure hoặcproofnotebooklỗi. Seed43 đãroundcommit/finite, ngừnghealthpollseed43.

## Full push17:18–17:27UTC+7, V10 terminal

Full1catbaochau(local/full),full2minhtriethihi(diag/lr4),full3khanhmay0304(own/DML),
full4catbaochau(DMLown) đãpushv1.7arms/100%train80%/50schedule/stopafter3 đểcalibrate/resume.
Local7arms/4exactresume/strictBN/codec/pred→10metric/inferencePASS;4notebookcontrollerPASS.
[Fullledger](../../methods/shared-baselines/builds/federated-baselines-full/results/rebuild.md).
V10 hai kernelCOMPLETE,14arm đãpull/audit,0missing. EnergyFL cost2,969/3,107×freshCE;
12candidate giữ2commit/costFAIL,2CEraw đủ20round. Chưacandidatequalityr20. Khôngrerun y nguyên.
[Report](../methods/dagsnet-joint-energy/builds/gaussian-base-energy/results/report.md).
Userđãchốt stablekhônghealthpoll:V7/ V8seed43 stable; V8seed42nohistory vẫnunknown.
V9/V10baseline córoundcommit/finite/applied; V10terminal khôngcầntheodõi.
Fullmới cầnkiểmpreprocessing/compile/initialtraining, chưaclaimhealthy từRUNNING.

## Kết quả/healthcheck17:59UTC+7

V10 đãaudit: energyFL r1sampler **FAIL cảhai seed**, khôngphảiPASS nhưghi chúnội bộ trướcđó.
Movementqua5% nhưngcóchainfrozen; seed43between-initmeanenergy vượt0,1SD.
Cost2× cũngFAIL, energycandidatequalityr20 chưađo; report/gates lànguồnsố.
KhôngMHnegative cũngcostFAIL: chưacôlậpMH/FP32/extra-forward nhưnguyênnhân riêng.

V8seed42 đãcóW&Bhistory/commit r1 Gaussian/hybrid, finite/AMPskip0 và2/3930;
trướckiaAPI RUNNING/nohistory khônglàfailure. Ngừnghealthpoll kernel này.
V9seed42API COMPLETE lúc17:59; collectorđangpull1396file rồiexactsource/checkpoint audit.
Seed43RUNNING theoAPI nhưng snapshotW&B chưađủidentityhistory; causeUNKNOWN, khôngrerun.

Full7arm cóW&Bonline/fingerprint/heartbeatexactidentity tại
[registry](../../../methods/shared-baselines/builds/federated-baselines-full/probes/wandb-runs.json), [snapshot](../../../methods/shared-baselines/builds/federated-baselines-full/probes/live-status.json).
Diag/lr4/Gaussianown đãcommit r1/full10metric,applied84053/84058/84056,
skipped30/25/27 (≈0,03%), lossfinite; ngừnghealthpoll cácarm này.
Local/fullGaussian/DML/DMLown cófiniteheartbeat qua nhiềuclient, r1commit chưatrongsnapshot.
Compile/source/full-datahash cònterminalaudit; khôngđọc W&B làartifactproof hoặcqualitygate.
Sau3roundcalibration tínhremaining50roundbudget+dư15%, tiếpcheckpoint đãkhóa;
khônglaunch lại từr1/đổiHP/lịch từtest. Chưacầnthao tácmanual.

## K2 seed42 audit18:14UTC+7

1396file/0missing;13arm đủ20round/source+executed-redacted/weights/BN/cohort/targets/payload/
pred→10metric/inference đãPASS. Report từAUDITED, khôngW&Bsummary.
M64/M512costPASS nhưngqualityFAIL; hơnown nhưnggầnclassmean/hardlabel/shuffle, thualocal.
RefreshM64PASS/M512FAIL; khôngđồngnghĩa teacherutilityPASS.
[Report](../methods/anchor-function-residual/builds/anchor-residual/results/report.md), [gates](../../../archive/methods/anchor-function-residual/builds/anchor-residual/probes/gates.json).
Seed43 chưaterminalaudit; khôngpairedconfirmation/privateKD/contact hoặc rerun y nguyên.

18:19UTC+7 fullinitialcheck chỉ4run chưacommit ởsnapshot17:59:local/fullGaussian
đãcommitr1, finite/AMPskipthấp; ngừnghealthpoll. DML/DMLown cófiniteSGD tớiclient19,
chưar1commit. Sourcefulltesteval20private+20proxy khôngheartbeat trongeval;
silence chưaprooffailure/mixing/quality hoặcworkerphase, khôngmanualstop khichưacóerror.
V7seed42API COMPLETE, collectorđangpull806file; khôngrerun, đợiAUDITED trướcutilityclaim.
