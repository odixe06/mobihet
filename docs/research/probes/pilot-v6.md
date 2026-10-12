# V6 — kết quả K5 geometry primitive moments

2026-10-12. [Manifest khóa](../pilot-v6-geometry.md),
[report đầy đủ10metric](../../../papers/v6/report.md), [cổng](../../../papers/v6/gates.json),
[hình](../../../papers/v6/figures.md). Headlineprivate/group/gp r20, không chọn head/round tốt nhất.

## Bằng chứng hoàn tất

4kernel/30arm đãcollect vàauditđúng source đã chạy, 0missing mỗi pull.
18arm đủ20round;12G59arm numericalFAIL, trongđó2arm có1roundcommit,10arm chưa commit.
Đã kiểmexecutedsource redacted, source/fixturehash,Torch2.10/2×T4/W&Bonline,
strictmodel395.024parameters/BN/state/staticpackets/eligibility/samplerrows,
prediction→clientCM→10metric vàsampleddecisiveinference r5/r20.
API:A42/A43ERROR bởiparentassertallcomplete; B42/B43COMPLETE.
Khôngrerun: output đủ kết luận cổng và giữ đượccontrol trongkernelERROR.

## G59: sampler không qua productionfinitegate

Cả12arm gặp `V6 synthetic fp16 input overflow`, khôngW&B/auth/OOM.
Log chỉ định vị guard synthetic, chưa ghi feature/value gâyoverflow nên không gán nguyên nhân duy nhất.
Inverse(dist>0) không có bound gần0; Gaussianprimitive độc lập không giữ mọi quan hệ geometry,
là cơ chế khả dĩ đã dự báo trướcpilot. Khôngclampdenominator/clipinput hoặc đổiprecision cứu cổng.
Toyfidelity/probe4096draw/smoke3roundPASS không chứng minh finite tail khi10triệudraw/arm.

## G61: finite và có tri thức quần thể, đóng góp geometry chưa đủ

| mask | seed | F1 | −strongerfull66 | −own | −shuffle | byte/control |
|---|---:|---:|---:|---:|---:|---:|
| marginal | 42 | 0.629138 | +0.004010 | +0.131665 | +0.001668 | 0.844949 |
| marginal | 43 | 0.627786 | -0.002368 | +0.134284 | +0.000567 | 0.844949 |
| joint32 | 42 | 0.648883 | +0.007507 | +0.147636 | +0.001327 | 0.846150 |
| joint32 | 43 | 0.650102 | +0.007884 | +0.152830 | +0.004448 | 0.846150 |

CommonprivateguardsPASS; **qualitypairedFAIL**, **compressionpairedFAIL** cảM/J.
Chênhstrongerfull66 chưa≥0,015, shuffle chưa≥0,010; byte chưa≤0,65.
Own thấp cho thấy lợi từpopulationknowledge; chưa chứng minh analyticderived là phần tạo lợi đó.

Joint−marginal:G61 +0,019745/+0,022316, qua contribution0,005 cảhai seed.
Nhưngfull66 cũng +0,018644/+0,017573; interaction chỉ+0,001101/+0,004743.
Không quy toàn bộ joint gain cho geometry. Moment cóđiều kiện đáng được đo riêng trong H2;
không dùng đóng góp joint để đổiqualitygate củaK5.

JointbootstrapG61=3.572.700B/xe hai chiều so marginal252.856B; full66supportjoint4.222.300B
so marginal299.256B. Conditionalmoment tăng khoảng14lần byte, dùô khôngđủcohort đãfallback.
Đâyactualapplicationbytes gồmprior, chưaframe/ACK/RTT/retry/security; khôngR1proof.
Không so vớiP1full lịch sử đểgọi gaingeometry: controller/RNG/precision khác vàfreshcontrols
đã cho thấy phần lớnF1 có thể đạt khi giữfull66.

## Quyết định sau pilot

Dừng adaptationG59/G61 hiện tại; khôngcontact/fullscale/rerun y nguyên hoặc tunecứu cổng.
Giữ control/jointcontribution vàfinite-tail lesson để thiết kế mới có đối chứng riêng.
ĐợtfullH1/H2, Nyström, Hybrid đã đượcchủdựáncho phép vẫn tiếp tục theocontract riêng.
Hai seed khôngCI; toànmodelDAGSNet, chưamixedarchitecture hoặccrypto/DP.
