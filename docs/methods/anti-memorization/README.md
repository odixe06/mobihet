# AntiMem + QProxy — chống học thuộc receiver, EMA triển khai, nén proxy

Pilot r20 hai seed PUSHED 15/10 12:00Z trên `nguyenminhtran060113` (14 arm/seed, 7h/session).
**Kết quả audited:** [results.md](results.md) — Gaussian private PASS mọi kỹ thuật, proxy FAIL, QProxy q4 PASS. [Thiết kế/cổng](design.md) · [Chẩn đoán gốc](../../archive/research/plateau-diagnosis-2026-10-15.md)

| kernel | arms |
|---|---|
| [am-s42-20261015-v1](https://www.kaggle.com/code/nguyenminhtran060113/mobihet-am-s42-20261015-v1) | am_{dml,hybrid,gaussian} × {gốc, m4b, swap, swappos} + am_dml_q8/q4 |
| [am-s43-20261015-v1](https://www.kaggle.com/code/nguyenminhtran060113/mobihet-am-s43-20261015-v1) | như trên |

Preflight local: `am_<base>` bit-exact v8 `h_<base>` (smoke r1/r3); exact full-vs-resume r3 PASS cho q4 và swap;
emitted controller với worker giả PASS (queue, cô lập worker lỗi, prune giữ r10/r20); dataset 110 names/sizes PASS.
W&B project `mobihet-antimem`. Source/notebook/artifact: [build](../../../methods/anti-memorization/builds/antimem-pilot/).
