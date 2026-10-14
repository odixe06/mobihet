# Bằng chứng và bài học theo phương pháp

Cập nhật2026-10-14; [danh mục phương pháp](README.md), [26hướng/triển vọng](status-summary.md), [quyết định](../docs/operations/decisions-2026-10-12.md).
Số đo gốc duy nhất ởreport/CSV; mục này giải thích tác động đến thiết kế. Không dùng officialtest để chọn.

## Bằng chứng đã đủ hai seed

| phương pháp | kết luận hẹp | bài học dùng cho thiết kế tiếp | nguồn số |
|---|---|---|---|
| Gaussian moment lớp / proxyDAGSNet DML | Có lợi trong P1; R1/R2 chưa đủ | Fullcov/teacher đủsứcchứa làđối chứng mạnh; prior/KD cầnđối chứngown | [P1](../docs/research/probes/pilot-p1.md) |
| Fourier/KME vàlatent-mean tĩnh | PairedFAIL | Fit synthesis khôngđồng nghĩa hữu ích; chưacôlập bank/encoder như nguyênnhân | [S-a](../papers/new-fl/report.md) |
| CountSketch proxy | PairedFAIL; EFnumericalpartial | Sketch-spaceEF cóthể khuếchđại collision; chuyểnparameter-space cầntest mới | [V2](../papers/v2/report.md) |
| Seeded-subspace proxy | 512nonfinite r1;1024teacherFAIL | Ít hệsố hạnchếsứcchứa, khôngmặcđịnhfine-tuneevidence ápchoIDSfromscratch | [V2](../papers/v2/report.md) |
| Weighted Gaussian densityratio | PairedFAIL | Clip/classnormalize khôngtự tăngESS hoặcF1; supportmismatch cònmở | [V2](../papers/v2/report.md) |
| Bốncarrier R/T/O/P | Đủ48run, tấtcảpairedFAIL | EF/basis/trainmask có contribution nhưng không giữ≥80% lợi dense32; componentPASS khônglàutilityPASS | [V3](../papers/v3/report.md) |
| Gaussian support atom/profile/first | PairedFAIL | Fidelity0invalid cóthể đi kèmF1thấp hơndiag; own/shufflepositive chưađủ | [V4](../papers/v4/report.md) |
| Gaussian MALA | PairedFAIL | Finite/acceptance cao không chứng minh mixing hoặc tri thức hơnown/shuffle | [V4](../papers/v4/report.md) |
| Kernel anchors teacher | Small/mediumpairedFAIL | Coverageđạt cóthể buộcbandwidthlớn,targetgầnuniform; nhãnhardGaussian tốt hơn | [V5](../papers/v5/report.md) |
| Gradientbooks IPC1/4 vàlatentmean refreshed | QualityFAIL, cost2×PASS | Đạo hàm/fit/exactresume đúng khônglàIDSutility; 2×2 khôngcho objective/encoder contribution đủcổng | [V5](../papers/v5/report.md) |
| Geometry G59/G61 | G59numericalFAIL; G61quality/compressionpairedFAIL | Giữprimitive sửafidelity khôngtựtăngutility; jointmaskbenefit cầnfull66jointcontrol | [V6](../papers/v6/report.md) |
| Gaussian-base jointenergy/MH | PairedcostFAIL vàr1samplerFLFAIL; qualityr20chưađo | Normalizable khôngđồngnghĩa samplehiệuquả; q-negativekhôngMH cũngcostFAIL, khônggánmọicost choMH | [V10](../papers/v10/report.md) |
| Covariance Nyström rank4/8 | CompressionPASS, qualitypairedFAIL | Giảmbyte/cócontribution không giữutility exactrank; không cứu bằng tăngprecision khi chưa côlậpnguyênnhân | [V7](../papers/v7/report.md) |
| Gaussian+proxy Hybrid | Pilot utility paired PASS | Hơn từng thành phần chưa chứng minh interaction dương; full-data dài cần controls hai DAGSNet riêng | [V8](../papers/v8/report.md) |
| Anchor residual teacher M64/512 | Cost PASS, quality paired FAIL | Refresh/fit/contribution không tự mang ranh giới hơn class-mean/hard/local; không mở KD từ cost PASS | [V9](../papers/v9/report.md) |

## Audit hoàn tất

- **MALA seed42:** đãaudit5arm đủ20round; tăngnhẹ sovsweightedGaussian, thuaown/shuffle/diag; seed42FAIL.
  Đãauditđủseed43: pairedFAIL. Acceptance/mixing chưađược chứngminh bởi finitechain.
- **Gradient books IPC1/4:**28run đủ20round; cả4kernel cost2×actualbaselinePASS,
  cảhai cỡ pairedqualityFAIL. Grad-trainedthuaGaussiancontinuous vàkhônghơnlatentmean/frozen đủcổng.
  CPUmixedderivativePASS khôngphảiqualityPASS.
- **V3R/O/T/P:**48run đủ20round vàsource/state/prediction audit; dense32 reference đủ2seed,
  tấtcả7sizegate pairedFAIL. Khôngchọnhead hoặcbestround để cứu cổnggp.

Đủ18kernel V3–V5:12kernel newlycollect/audit,6kernel oldproof khôiphụcmetadataexactMD5;
khôngrerun. Ba hướng ưu tiên có bằng chứng pilot: Gaussian/DML/Hybrid;
Hybrid cả hai seed đã audit/utility PASS. Nyström/K5 adaptation đã đóng cổng.

[Livecollect](../docs/operations/live-2026-10-12.md); finalheadline ởreport khi đủAUDITED.
Old6kernel audit táidùng metadataexactMD5; weights vẫnDVCremote, khôngclaim kiểmweightslại trongphiên.

## Từ bài học đến hướng đã chốt

**K5 nominalG51/G53 bịfidelitytoy bácbỏ trướctrain:** impute0 làmbranchmissing khôngidentifiable
 từprimitive66. Giữ8derived bịảnhhưởng⇒haiđềxuất cùngG59.
Khôngtrain haiarmđồngnhất hoặc hạbytegate: fullcov59 khôngđạtcarrier65%beforemask.
Chủdựánchọn kiểmG59/G61 **quality** và đo marginalmask vsjoint32.
[Thiết kế](geometry-primitive-moments/design.md), [toyproof](geometry-primitive-moments/fidelity-missing.json),
[quyết định](geometry-primitive-moments/decisions.md).

Giảthuyếttrướcpilot: moment cóđiều kiện first/profile giữquan hệreceiver, analyticderived giữ mộtphần
redundancy nonlinear màGaussian66 phá. V6 đãkiểmfull66plain/support,own,shuffle vàsamecompute;
G61 khôngđủquality/compressiongate, jointbenefit xuất hiệncảởfull66control.
Khôngép impliedspeed bằngclaims hoặc residual0: các dấuhiệu bấtthường cóthể chínhlàattack.

## Đợt mới đã triển khai

V7Nyström/V8hybrid/V9residualteacher cả hai seed đã audit đủ controls.
V9 đủ26arm-seed×20round:costPASS nhưngteacherqualitypairedFAIL cảM64/M512;
hơnown khôngđủhơnclassmean/hard/shuffle, refreshM64PASS khônglàutilityPASS.
Coverage/fit tốt vẫn thua local cả hai seed.
[Report hai seed](../papers/v9/report.md), [cổng](../papers/v9/gates.json).
V10normalizableGaussian-base đãaudit: dừngadaptation bởi lockedcost vàsamplerFLr1;
không lấy source/smokePASS thayquality. KhôngđổiHP/kéodài cứu cổng.
H1/H2full7arm/100%train80%/seed42 đãlaunch4calibrationnotebook theo50roundschedule;
3roundđầu đểđoT4/budget vàchuyểncheckpoint, khônggọifull50complete.
[Manifest/run/budget](../docs/operations/expansion-2026-10-12.md), [fullledger](../papers/full-validation/rebuild.md).

## Audit full-data ngày14/10

H1/H2 tới r7–r11/50; DML còn lợi so own cùng round, Gaussian còn lợi nhưng không tăng đơn điệu.
Hybrid full đủ4kernel/7arm/18round, last r2/r3. Hybrid có tín hiệu tốt so matchedDML/own;
chưa r50, một seed full không paired confirmation. Hai ERROR do budget/controller, checkpoint hợp lệ,
không rerun từr1. Cùng sốmodel/SGDstep không bảo đảm cùngwalltime.
[Kết luận](../papers/hybrid-full/feasibility-20261014.md),
[10 metrics Hybrid](../papers/hybrid-full/latest-rounds.md),
[10 metrics H1/H2](../papers/full-validation/latest-rounds.md).

BoundaryCells mới calibration cost PASS/chưa utility. ResponseRisk chỉ localaugmentation trước;
EMA cả Gaussian/DML đã chốt cổng/chưa source. Không nâng dự báo hiệu năng từ code/smoke/cost.

## Giới hạn bắt buộc

StageU khôngđủR1. Input/publicschema/prior/bootstrap/cohort/RTT/ACK vẫntrảbyte/deadline ởcontact.
DAGSNet-only khônglàmixedarchitectureproof. SecureSum môphỏng khôngcrypto/DP.
Two-seed khôngCI; localprobe chỉđiềukiệncần. ScientificFAIL khôngtựđộngruntimefailure; COMPLETE cũngcóthểfailcổng.
