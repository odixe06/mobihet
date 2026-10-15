# ClassTail pilot — contract và checkpoint

20 client, fixture pilot gốc từ train80%, group/old nội bộ; không official test.
Model DAGSNet, batch512, AdamW/reset, cosine20 1e-3→1e-5, clip1, AMP.

Tail arms r1/r2 CE; từ r3 dùng threshold tính trên scan T ở cuối round trước, BN eval/no-grad.
Scan không dùng group/old. Local CE không scan mỗi round; probe r2 tách thời gian scan khỏi
denominator cost chuẩn. Reference Gaussian/DML chạy recipe của chúng từ r1.

8 arm: local, fixed ζ=log16, own, pooled, class-shuffle, class-mass CE, Gaussian-full, DML.
Class-shuffle xoay chỉ số lớp một bước; không giữ class mass. Class-mass CE khớp mean weight
trên scan T trước SGD; mass thực tế sau SGD vẫn có thể khác. Báo riêng, không gọi exact-matched.
Mỗi applied step lưu weighted target mass; gp chính từ mass đó. X-dependent weights không tạo
bảo đảm label-shift; head/gp cũ là diagnostic, không đổi headline để cứu gate.

Khóa trước run tại [JSON](../../../../../methods/class-tail-risk/builds/class-tail-pilot/probes/locked-contract.json):

- Probe r2 cả hai seed: ≥12/16 lớp có pooled tail fraction [0,10;0,40], membership pooled/own
  khác ≥0,05 macro-class; ≥5 contributors/class, pooled N≥128, own N≥32. Overflow invalid → CE fallback.
- R20 group/private/gp: pooled−own/shuffle≥0,010, −local≥0,015, benign giảm≤0,005 so local và
  strongest Gaussian/DML như proposal, mỗi seed.
- Chỉ mở dài nếu hơn reference mạnh nhất Gaussian/DML≥0,015 mỗi seed, cost≤2×plain CE.
- Probe/codec/cost PASS không thay utility PASS. Không tune bin/α/λ sau FAIL.

Codec thực: up2.272B, down162B, dictionary164B/client; header32 gồm schema/version/CRC/hash16.
Counts uint32, aggregate uint64; eligibility16bit; không raw record. Chưa contact/DP proof.

Local [preflight](../../../../../methods/class-tail-risk/builds/class-tail-pilot/probes/local-preflight.json):
gradient/tie/codec/overflow, 4-arm exact resume và applied mass PASS; bounded fixture, không T4 utility/cost.
Timing scan chỉ ở diagnostics, loại khỏi algorithm state để exact-resume không phụ thuộc walltime.

Calibration đã push version1:
[seed42](https://www.kaggle.com/code/minhtriethihi/mobihet-ct-cal-s42-20261015-v1),
[seed43](https://www.kaggle.com/code/khanhmay0304/mobihet-ct-cal-s43-20261015-v1).
Session budget3h/seed; dừng mỗi arm ở r2 của lịch20. Cả hai đã terminal/pull/audit source/state/predictions.
Paired probe PASS:16/16lớp mỗi seed; membership diff0,145720/0,136078.
[Probe và toàn10metrics](results/report.md). Utility hai seed đã đủ8arm×r20 và audit: FAIL.
Pooled−local−0,002683/−0,002533; −own−0,001644/−0,002840;
−shuffle−0,001509/−0,001034; −classmass−0,000713/−0,000164.
Kém DML0,196729/0,191719, benign FAIL; cost PASS. Không nối hoặc retune sau FAIL.
[Failure diagnostics](results/failure-analysis.json) mô tả late bin-floor/tail, không cô lập nguyên nhân duy nhất.

Utility continuation đã dựng từ exact r2,8arm/seed, giữ frozen40mh+4proj/LR20;
seed42 owner catbaochau, seed43 khanhmay0304, budget2,5h/session. Dataset riêng và CPU production-import
SHA/fingerprint/last-round/strict-model gate PASS trước GPU push. Cả hai utility đã PUSHED version1.
catbaochau2,9526h đủ2,875h gồm15%; khanhmay0304 snapshot6,4918h trừ thêm toàn3h repair và15%
cho cả repair+utility vẫn đủ6,325h. Đây không là bằng chứng đủ mọi armr20 trong một session.

Collector terminal utility: thêm `--utility` vào `collect_tail_calibration.py`, rồi chạy `report_tail.py`.
ERROR vẫn pull/audit committed rounds; không retry r1. Cost báo algorithm và full round r3–20 so CE không scan;
source/import/budget trả phí riêng. Class-mass CE ngang pooled trong≤0,005 cả hai seed: chưa quy lợi cho hard-example mechanism.

[Source/scripts/notebooks](../../../../../methods/class-tail-risk/builds/class-tail-pilot/README.md) ·
[Bàn giao phiên](../../../../operations/continuation-2026-10-15.md).
