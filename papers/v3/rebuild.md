# V3 — ledger triển khai

2026-10-11. Chủ dự án muốn thêm phương án/kiểm mọi hướng khả thi trong lúc V2B/C chạy.
Đã xác nhận: cả4carrier,fp32 hai chiều,private+KD ngay nhưA.
[Manifest](../../docs/research/pilot-v3.md), [nguồn](../../docs/research/v3-carriers-sources.md).

## Trạng thái

Đã push8kernel v1 lúc2026-10-11 17:00UTC, input6owner private READY trước push.
Math trên `a_dense_s42/resume/round_001.pt` PASS, peakRSS536MiB:
projection linearity,norm/residual,QR,two-stagePowerSGD,actualpacketbytes.
Không dùng probe này làm F1feasibility. `math-probe.json` ghi kết quả/saved-update provenance.
R-smallEF3round×20client×256train PASS,peakRSS2196MiB; chỉ driver smoke.
23arm×3round,exact4-carrier resume,tamper vàsample inference đã PASS.

23compressedarm×3round,exact4-carrier resume,strict/sampledinference/tamper PASS ở sourcecontext cuối,
peakRSS2714MiB. Bảnpre-context PASS2727MiB nhưng chưa kiểmmodel/mask mismatch; source lưu `pre-context-source/`.
`scripts/check_v3_abort.py` PowerSGDrank8 actualsaved-denseupdate,5clientCPU stageQfinish fault PASS:
model/version/residual khôngcommit, retrywhole-round bằngfreshround; peak675MiB.
`abort-probe.json` ghi288160Bapplication đãphí trướcfault. Udriver crash trướccheckpoint;
durablefailed-frameaccounting ởfinitecontact vẫnphảithiếtkế riêng, chưaR1proof.
Probeđầu sofreshmodelkhácinit bịassert; test đãkhóainitseed, khôngcarrierbug.
Thêm16Bcontext hash mỗi message (scienceconfig/baseweights/mask/basis/stage), sai context từchối trướcsum.
EF chỉ chuyểnpending→committed saucohort đủ; P có2stage thật. Byte đãcậpnhậtmanifest.
Control độc lập`a_dense32` cùngprecision +4representatives/resume trênwrappercuối PASS,peak2720MiB.
Local/a_own A giữreference doCE/KD/fixture/runtime giống; denseAfp16 chỉhistorical, khôngretention denominator chính.
K1schema sample40960row/7337firstrowPASS; JEMunbasedR66FAILnormalizability (LayerNorm headbound), khôngheavytrain.
Chủ dự án đã chốt K2b thống kêsoft target (learnedmodel đềuDAGSNet), K3cảIPC1/4 với2×baseline,
MALA τ.05/cap10/eta.01/20burn-in/1transition mỗibatch. V4manifest/preflight trước pilot riêng.

Code `mh/v3_math.py`, `mh/v3_methods.py`, adapter `mh/v3_run.py` dùng nguyên driverV2,
CE/KD/BN/metrics/atomiccommit không sửa. V2source đãpush giữ nguyên.
P cóPup/Pbasisdown/Qup/Qdown thật; không giấu second phase. Residual/delta local lưu trong resume đểaudit.
8notebook compile/metadata/keyboundary/exactorchestration PASS; onlineW&B trước train bắt buộc.
Key chỉ private_push0600, bảnpublic secret-free. Đã nhắc chi phí lịch sử phiên bản private trước push.

| family | seed42 owner/kernel | seed43 owner/kernel |
|---|---|---|
| R | minhtran0601/mobihet-v3r-s42-20261011-v1 | dinhan25/mobihet-v3r-s43-20261011-v1 |
| T | trietbackup/mobihet-v3t-s42-20261011-v1 | odixe0502/mobihet-v3t-s43-20261011-v1 |
| O | trietbackup/mobihet-v3o-s42-20261011-v1 | odixe0502/mobihet-v3o-s43-20261011-v1 |
| P | odixeuit/mobihet-v3p-s42-20261011-v1 | minhtrit06/mobihet-v3p-s43-20261011-v1 |

`quota-before.json`: minhtran0601≈11.05h,dinhan25≈11.36h;4ownerkhác≈17.57–20.17h trước push.
R ưu tiên2ownerítquota;T/O mỗiowner2kernel;P mỗiowner1kernel. Không đổi active CLI account.
`wandb-runs.json`48identity từsourceinput immutable/fixture/config; `live-status.json` không bằng chứng artifact.
Collector `scripts/kaggle/collect_v2.py --build v3 --family r|t|o|p --watch`, log từngfamily;
GPU local inference cófilelock, tránh chạy đồng thời. Nhận COMPLETE/ERROR vẫn phải pull/audit.

## Còn làm

Productionrankpreflight2T4 trongnotebook; chờ actual log. Pull/audit/report/cổng cả4family×2seed.
K1–K4knowledge vẫn đề xuất riêng: supportaudit/coverage/compute/samplingpreflight trước mởtraining.
Git/DVC không commit/push/backup; artifactchưabackup. Localkhôngtrain20round.

## Collect saubackup2026-10-12

KernelAPI đãCOMPLETE; audit/collect đangtiếptục, khôngrerun.
Oldauditmetadata khôiphục từDVCobject, MD5PASS; khônggọikiểmweights lạichooldkernel.
Newkernelpull/verifyđủsource/checkpoint/payload/prediction trướcreport.
[Phiêncollect](../../docs/operations/live-2026-10-12.md),
[restoreproof](../../docs/operations/restored-metadata-2026-10-12.json).
DVCmanifest cũkhôngbao hàmoutputmới; khôngadd/push/cleanup lúcnày.
