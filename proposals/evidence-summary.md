# Bằng chứng và bài học theo phương pháp

Cập nhật2026-10-12; [danh mục phương pháp](README.md), [24hướng/triển vọng](status-summary.md), [quyết định](../docs/operations/decisions-2026-10-12.md).
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

## Audit hoàn tất

- **MALA seed42:** đãaudit5arm đủ20round; tăngnhẹ sovsweightedGaussian, thuaown/shuffle/diag; seed42FAIL.
  Đãauditđủseed43: pairedFAIL. Acceptance/mixing chưađược chứngminh bởi finitechain.
- **Gradient books IPC1/4:**28run đủ20round; cả4kernel cost2×actualbaselinePASS,
  cảhai cỡ pairedqualityFAIL. Grad-trainedthuaGaussiancontinuous vàkhônghơnlatentmean/frozen đủcổng.
  CPUmixedderivativePASS khôngphảiqualityPASS.
- **V3R/O/T/P:**48run đủ20round vàsource/state/prediction audit; dense32 reference đủ2seed,
  tấtcả7sizegate pairedFAIL. Khôngchọnhead hoặcbestround để cứu cổnggp.

Đủ18kernel V3–V5:12kernel newlycollect/audit,6kernel oldproof khôiphụcmetadataexactMD5;
khôngrerun. Chỉ3hướngthiếukếtquảutility mới trong5hướngưu tiên: hybrid/Nyström/K5.

[Livecollect](../docs/operations/live-2026-10-12.md); finalheadline ởreport khi đủAUDITED.
Old6kernel audit táidùng metadataexactMD5; weights vẫnDVCremote, khôngclaim kiểmweightslại trongphiên.

## Từ bài học đến hướng đã chốt

**K5 nominalG51/G53 bịfidelitytoy bácbỏ trướctrain:** impute0 làmbranchmissing khôngidentifiable
 từprimitive66. Giữ8derived bịảnhhưởng⇒haiđềxuất cùngG59.
Khôngtrain haiarmđồngnhất hoặc hạbytegate: fullcov59 khôngđạtcarrier65%beforemask.
Chủdựánchọn kiểmG59/G61 **quality** và đo marginalmask vsjoint32.
[Thiết kế](geometry-primitive-moments/design.md), [toyproof](geometry-primitive-moments/fidelity-missing.json),
[quyết định](geometry-primitive-moments/decisions.md).

Cơ chế đáng kiểm: moment cóđiều kiện first/profile giữquan hệreceiver, analytic7derived giữ mộtphần
redundancy nonlinear màGaussian66 phá. Cầnfull66plain/support,own,shuffle vàsamecomputeđểtáchhai hiệuứng.
Khôngép impliedspeed bằngclaims hoặc residual0: các dấuhiệu bấtthường cóthể chínhlàattack.

## Giới hạn bắt buộc

StageU khôngđủR1. Input/publicschema/prior/bootstrap/cohort/RTT/ACK vẫntrảbyte/deadline ởcontact.
DAGSNet-only khônglàmixedarchitectureproof. SecureSum môphỏng khôngcrypto/DP.
Two-seed khôngCI; localprobe chỉđiềukiệncần. ScientificFAIL khôngtựđộngruntimefailure; COMPLETE cũngcóthểfailcổng.
