# Marginal transport sau direct-generator — feasibilityPASS, 06/11

Job gốc lỗi xuấtJSON numpy.bool_, không phải lỗifit. Phục hồi bằng `python -m scripts.recover_marginal_probe`: verify source/plan/sampleSHA,3CMsupport vàCM→10metric khớp loggốc. Không train lại. Jobstatefailed giữnguyên, artifactrecoveryPASS riêng.

[Plan](screening-marginal-direct-generator.md). PairTV .199355→.066747, singletonguardPASS cả16class, TSTRmacroPASS. CPUboundedtrain→val, khôngofficialtest/FL.

| metric | copula | forest+transport |
|---|---:|---:|
| accuracy | 0.712450732 | 0.793137784 |
| precision_macro | 0.759363527 | 0.833928353 |
| precision_micro | 0.712450732 | 0.793137784 |
| precision_weighted | 0.792543689 | 0.831532455 |
| recall_macro | 0.816017819 | 0.847167398 |
| recall_micro | 0.712450732 | 0.793137784 |
| recall_weighted | 0.712450732 | 0.793137784 |
| f1_macro | 0.761808674 | 0.831997139 |
| f1_micro | 0.712450732 | 0.793137784 |
| f1_weighted | 0.721902902 | 0.804388574 |

Khoảng80.7% tổngmacrodelta đến từclass15 zeroSpeedReport (baselineF1=0→.906469); class11 .386870→.653775. Class1 .867551→.736356, class13 .236768→.210829. Không coi aggregatePASS làutilityFL hoặcmọiclassđượclợi. RealHGBmacro .909274 chỉmốcdiagnostic.

Bước tiếp đãkhóa: [attribution copula-only marginaltransport](screening-marginal-copula.md). Táchhiệuứngsửamarginal khỏihistogram2D; khôngthêmHP/đổigate, khônglaunchFLtừgaincóthểbị1classchi phối.

## Torchport và preflight07/11

[Thiết kế/codec](../../docs-20261109/research/screening-forest-t4.md).8testsPASS. Artifact kiểm lại từdisk: seed42/43 localGPU sm86/world1,
2xe613/2029,4round: straight vàstop2→resume3→restore2→4 khớpbit cảfullstate/history/CM/snapshot; sourceSHA khôngđổi.
Static samplerSHA khớp state/recovery, bootstrap1704951B/xe. Reports:
`runs/forest-production/{preflight,preflight-artifact-audit}.json`. ChưaT4/2rank/fullnetwork/FLutilityproof.

GPU sampler trênpaneltrain→val cũ16class: pairTV .199354639→.067298627 (−66.24%), singletonguard≤+.02/classPASS.
KhôngfitHGB lại, khôngofficialtest. `runs/forest-production/sampler-gate.json` giữsource/sampleSHA/perclassstatistics.
Đây làstatisticalportgate, không thay TSTRhoặcutilityFL.

Owner02:39 cậpnhật: nối sessioncũ giữưu tiên, cho thửnghiệmkhảthingay bằngquota dư. Timer0700 đãcancel trướcpush;
workerngay vẫnkiểm freshquota/slots/inputaccess trước2kernelruntime. Trạngtháilive tại [continue](../../../docs/operations/continue.md).
