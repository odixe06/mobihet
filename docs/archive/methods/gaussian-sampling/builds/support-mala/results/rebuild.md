# V4 — ledger tri thức mới

2026-10-12 UTC+7. K2b/K3/MALA đã được chủ dự án cho phép; K2b bandwidth/K3 lịch80step đanghỏi chốt.
Mọi model học được DAGSNet, FL không localraw/perturbedrecord rời xe.

MALA [manifestkhóa](../../../../../research/pilot-v4-mala.md), implementation `mh/v4_mala.py`,
adapter `mh/v4_run.py` giữ nguyên V2driver/CE/NCE/metric/atomiccommit.
`scripts/check_v4_mala.py --tag mala-final`: mathnull/sign/inputgradient,5arm smoke,3exactresume/inference PASS,
peak3879MiB. Smoke `mala` cũ guardchặn do thêmmodule khi đangcheck; không numericalfailure.
Support `scripts/check_v4_support.py --tag support-final` PASS5arm/2exactresume/inference,
peak3257MiB; `support-math.json`:65536draw/arm,structural/shuffle0binary/profile/firstconstraintinvalid,
marginal1847firstconstraintinvalid/5943firstrow. Đây fidelity, khôngF1feasibility.
Byteone-shot12776up/10592down+200prior=23568B/xe. Ownsupport chỉcommonprior, không gộpsupportstats.
Đã dựng4notebookprivate,compile/metadata/exactorchestration/keyboundaryPASS;4datasetprivateREADY trướcpush.
**PUSHED cả4kernel v1 lúc2026-10-11 17:22UTC (2026-10-12 00:22UTC+7)**; chưaF1.
Không sửa cácmoduleV3/V4/sourceinput đãpush. K2b/K3 code tiếp theo dùngwrapper/datasetV5 riêng.

| family | seed42 | seed43 |
|---|---|---|
| MALA | minhtran0601/mobihet-v4m-s42-20261012-v1 | dinhan25/mobihet-v4m-s43-20261012-v1 |
| Support | odixeuit/mobihet-v4s-s42-20261012-v1 | minhtrit06/mobihet-v4s-s43-20261012-v1 |

`wandb-runs.json`20exactfingerprints sourceinput/fixture/config, projectmobihet-v4/entity21522798-uit.
`live-status.json` chỉprogress; [artifactreport](report.md)/`gates.json` PENDING khi chưaAUDITED.
Collector `scripts/kaggle/collect_v2.py --build v4 --family m|s --watch`, logscollection-m/s;
`scripts/report_v4.py` chỉ lấyAUDITED, đủ10metric/perclass/byte/time/pairedgates.
Key chỉprivate_push0600, publicsecret-free; privateversionhistory giữkey nhưworkflow đãnhắc.
`quota-before.json`2026-10-11 17:16UTC: minhtran0601≈10.585h,dinhan25≈10.844h,
odixeuit≈19.887h,minhtrit06≈20.172h. MALA ưu tiên2ownerítquota(mỗiownerR-V3 còn1slot),
support2ownercóP-V3 còn1slot. ƯớcMALA≤3h, support≤1,5h/owner chưađoT4;
giới hạn4h/arm và3hnhậnjob nhưV3, không dùng ướclượnglàmeasurement.

K3 local `scripts/probe_v4_gradient.py` PASS:32trainrow/FP32evalDAGS/IPC1/4×4step,
peakRSS1732MiB. `gradient-probe.json` actualTorch2.13/RTX3050, warmstep≈0,054/0,060s;
không dùng thay đo T4Torch2.10 hoặc gate2×baseline. Đạo hàm đãkhảthi,compute cònphảiđo.

HistoricalCscoreprobe `scripts/probe_v4_ratio_score.py` PASS64freshGaussianpoint/lớp×2seed,
peak1371MiB; clipfloor≈57,8%/63,6%, medianratio score0 ở13/16 và15/16lớp.
Đây risk overlap/localguidance, không MALAoutcome/mixingproof, khôngthay hyperparameter đãkhóa.
`scripts/plot_v2_ratio.py` nốiESS/clippedscore/perclassgain thànhPNG/SVG diagnostic, khôngcausalclaim.

C đủ8arm×20roundverify,FAIL; freshweightedGaussian/diag trongMALA để tránh lệchwrapper/source.
V3 đangqueue8kernel; không gọi queued/run/W&B trạngthái là kết quả đãaudit.

Git/DVC khôngthựchiện; artifactchưa backup. Local chỉ probe/smoke/collect/report.

## Collect saubackup2026-10-12

KernelAPI đãCOMPLETE; audit/collect đangtiếptục, khôngrerun.
Oldauditmetadata khôiphục từDVCobject, MD5PASS; khônggọikiểmweights lạichooldkernel.
Newkernelpull/verifyđủsource/checkpoint/payload/prediction trướcreport.
[Phiêncollect](../../../../../operations/live-2026-10-12.md),
[restoreproof](../../../../../../operations/restored-metadata-2026-10-12.json).
DVCmanifest cũkhôngbao hàmoutputmới; khôngadd/push/cleanup lúcnày.
