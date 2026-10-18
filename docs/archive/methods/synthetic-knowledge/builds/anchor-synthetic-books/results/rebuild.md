# V5 — ledger kiểm nghiệm K2b/K3

2026-10-12 UTC+7. Chủ dự án xácnhận softtargetstats/DAGSNet-only,
bandwidthtrain-only4options/cổngcoverage, IPC1/4,80slotround-robin/refresh5round/2×baseline.
[Hmanifest](../../../../../research/pilot-v5-anchors.md), [G/Jmanifest](../../../../../research/pilot-v5-books.md).

## Source/preflight

`mh/v5_anchor.py`,`mh/v5_books.py`,`mh/v5_run.py`; không sửa MH V3/V4 đãpush.
23arm×3roundsmoke +6exactresumes (bookrefreshsmallperiod2) +model/state/packet/pred→10metric/inference PASS,
`driver-check-final2.log`, peak3333MiB. Productionbook80/period5, smoke16/period2 ghi rõ.
Smoke `final` weights/books/packets đúng nhưngstatewalltime mismatch; sửa khônglưucondensation_seconds
vàostate thuậttoán, giữ trongrounddiagnostic; giữartifactcũ, khôngnumericalfailure.
Full80slot IPC1/4 bounded512producerrecords FP32 PASS, `book80-probe.json`,peak2101MiB;
Torch2.13/RTX3050, khôngthayactualT4/currenttrained/F1/costproof.
Hcoverage sample40k h4PASS nhưngsofttargetgầnuniform, chưafulltraingate.

6notebookcompile/metadata/onlinekeyboundary/exactorchestrationPASS;
4booknotebook×reject/terminatefaulttestsPASS (`budget-check-final.log`).
Budgetreject branch dùngmutablebaselinequeue bịIndexError, sửa trướcpush sangMETHODS[0]; khôngtrainingfault.
Hpreflight độc lập kernel/teacher, khôngphụthuộcK3second-derivativegate.
G/J actualfull80bothobjectives/IPC/cả2T4; freshdiag+noise2workerGPU0trướccases,
maxhours2×baseline +parentcostguard, sourcecost_reference/observed lưuactualbytes/time.

## Kaggle

**6kernelv1PUSHED2026-10-11 18:52UTC/2026-10-12 01:52UTC+7**,3privateinputREADY trướcpush.

| family | seed42 | seed43 |
|---|---|---|
| H K2b | minhtriethihi/mobihet-v5h-s42-20261012-v1 | catbaochau/mobihet-v5h-s43-20261012-v1 |
| G IPC1 | khanhmay0304/mobihet-v5g-s42-20261012-v1 | minhtriethihi/mobihet-v5g-s43-20261012-v1 |
| J IPC4 | catbaochau/mobihet-v5j-s42-20261012-v1 | khanhmay0304/mobihet-v5j-s43-20261012-v1 |

`quota-survey.json`18:09UTC: selectedowners≈21,854/21,871/22,067h, noreservations beforepush;
2kernels/owner. Lowquota9owner khác đangV3/V4 giữreserve≈11,5–11,7h, khôngcoileft0làactualspend.
ƯớcH≤1,5h,G/J≤3h/kernel chưameasurement; actualcost/baseline làgate.
ActiveCLIlogin khôngđổi, scopedOAuth/nativeAPI; noMCPcapability, noGit/DVC/backup.
46exactW&Bidentities ở`wandb-runs.json`, entity21522798-uit/projectmobihet-v5;
live-status chỉprogress, khôngF1proof. Keyprivate_push0600, publicsecret-free; historyprivate giữkey đãnhắc.
Collectorh/g/j `scripts/kaggle/collect_v2.py --build v5 --family h|g|j --watch`, localGPUverifyfilelock.
TerminalCOMPLETE/ERROR phảipull/audit từngarm;0roundsource/configproof khôngmodelproof.

## Còn làm

ActualT4preflight/logs; thu/verify/report10metric/perclass/pairedgates +cost gates; chỉPASS mới privateKD/contact riêng.
K5 [algebra/primitive](../../../../../research/v5-algebra-sources.md) mớisource/design, chưacode/pilot;
không épimpliedspeed=claim/residual0/onroad đểxóaattack, cần6chi tiết chốt saupreflight.

## Collect saubackup2026-10-12

KernelAPI đãCOMPLETE; audit/collect đangtiếptục, khôngrerun.
Oldauditmetadata khôiphục từDVCobject, MD5PASS; khônggọikiểmweights lạichooldkernel.
Newkernelpull/verifyđủsource/checkpoint/payload/prediction trướcreport.
[Phiêncollect](../../../../../operations/live-2026-10-12.md),
[restoreproof](../../../../../operations/restored-metadata-2026-10-12.json).
DVCmanifest cũkhôngbao hàmoutputmới; khôngadd/push/cleanup lúcnày.
