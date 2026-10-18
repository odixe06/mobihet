# Contractive EF — sửa cơ chế, giữ cổng V2-A

## Nguyên nhân tái hiện

Collision tối giản: size4/depth1/width1/topk3, accumulated=1.
Original decode→re-encode cho residual norm2, tức2×input. CPU FP32, không model/BN/AMP.
Vì vậy FP16 hoặc BN không cần thiết để gây overshoot; BN overflow là hậu quả trong run cũ.
Đối chiếu [artifact cũ](../../../../../archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/probes/sketch-diagnostics.json).

## Sửa và dự báo

Giữ cùng hash/sketch/k/sparsity và producer local epoch. Với proposal δ từ decoder:
γ=clip(<A,Sδ>/||Sδ||²,0,1). Serialize/decode γδ trước kiểm ||A−Sδ_wire||²≤||A||².
Nếu lượng tử hóa làm mất điều kiện, giảm γ một nửa tối đa16 lần, cuối cùng patch0.
Residual trừ đúng wire patch; commit/version/cohort/ACK giữ nguyên. Không retune LR/clip/model.

Dự báo: chặn amplification do phép trừ sketch, có thể giữ proxy học được; chưa dự báo chắc utility.
γ≈0 kéo dài, update không mang ranh giới hoặc F1 không giữ gain dense đều bác hướng.
Norm không tăng so A không có nghĩa residual co đều qua round, fidelity tốt hoặc IDS tốt.

[80 ca kích thước production](../../../../../methods/proxy-countsketch-repair/builds/contractive-ef-pilot/probes/repair-algebra.json)
với generated updates, hai seed, small/medium×20round PASS; collision ratio sau sửa <0,001.
[Exact resume](../../../../../methods/proxy-countsketch-repair/builds/contractive-ef-pilot/probes/exact-resume.json)
bounded local medium PASS. Không thay full-epoch/T4 hoặc quality proof.

## Trial đã chuẩn bị

Seed42/43, original fixture/lịch20/AdamW/reset/batch512/T2/λKD1, gp cũ chính.
7arm: local/own/dense/small-noEF/small-repairedEF/medium-noEF/medium-repairedEF.
Chỉ sửa aggregate cho hai repaired arms; không rerun unstable original EF.
NoEF controls giữ original aggregate. Source riêng; không sửa40module mh hoặc source full đang chạy.

Giữ [cổng cũ](../../../../archive/research/pilot-v2.md):
private group−local≥0,03; old>local; benign≥local−0,03;
repaired−own≥0,01 và giữ≥80% gain dense−own; EF−noEF≥0,005 để gán lợi riêng residual.
Dense cũng phải qua common gate và proxy teacher group/head−local private/group/gp≥0,05 như cổng gốc.
Small/medium xét riêng, hai seed; byte/cost/contact báo riêng.
Nếu numerical/cost/quality FAIL: không kéo dài adaptation này để tìm round thắng.

Budget3h/seed, owner minhtriethihi/khanhmay0304; cùng tài khoản với ClassTail nhưng session riêng.
Combined worst6h×1,15=6,9h mỗi owner, nằm trong snapshot9,47h/7,19h trước launch.
Không dùng quota3account mới cho hai pilot này.

Collector [collect_contract.py](../../../../../methods/proxy-countsketch-repair/builds/contractive-ef-pilot/scripts/collect_contract.py)
pull terminal COMPLETE/ERROR, audit exact source/version, mọi checkpoint/BN/payload/10metrics,
kiểm decoded residual không tăng ở mọi saved round kể cảr1. Chỉ tính quality gates cho từng candidate và controls aligned đủr20;
hai seed/R1/cost vẫn kiểm riêng. Local collector đã kiểm saved medium3round thật, không T4 utility.

[Source/scripts](../../../../../methods/proxy-countsketch-repair/builds/contractive-ef-pilot/README.md) ·
[Bàn giao](../../../../archive/operations/continuation-2026-10-15.md).

Cập nhật chiều15/10: hai seed đã terminal/pull/audit. Smallr20 qualityFAIL và không có EF contribution;
mediumr17/r14 chưa strict-r20, nhưng kém aligned own0,2486/0,2577, last5 phẳng/giảm.
Không nối để tìm round thắng. Sửa numerical overshoot không sửa được teacher/representation retention;
actual patch chỉ chạm phần nhỏ tọa độ. [Report/gates](results/report.md). Không bác cả họ CountSketch/EF.
