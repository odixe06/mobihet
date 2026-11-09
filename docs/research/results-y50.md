# Y50 — kết quả đã audit, 06/11/2026

Prior: Y1/Y2 đủ r20, pull + audit PASS; luật utility FAIL, dừng đúng thiết kế. Oracle đủ r50, pull + audit PASS.
**noq/nofloor đủ r50 (audit `runs/y-transfer/Y{3..6}/audit-r50`), luật ablation: Q `needed`, floor `needed` → giữ cả hai trong Core** (mục cuối).

Nguồn: `runs/y-study/wave1/collection/Y{1,2}-r20/{audit,diagnostic-audit}.json`, `runs/y-study/wave1/prior-r20.json`. Phạm vi: source/config/CM/support/snapshot/teacher/finite; chưa prediction replay hoặc dựng lại full private bank.

| metric r20 | Y1 seed42 | Y2 seed43 |
|---|---:|---:|
| accuracy | 0.916049321 | 0.916349716 |
| precision_macro | 0.915146375 | 0.916145143 |
| precision_micro | 0.916049321 | 0.916349716 |
| precision_weighted | 0.907885717 | 0.908261098 |
| recall_macro | 0.889590211 | 0.889780688 |
| recall_micro | 0.916049321 | 0.916349716 |
| recall_weighted | 0.916049321 | 0.916349716 |
| f1_macro | 0.896694048 | 0.897176879 |
| f1_micro | 0.916049321 | 0.916349716 |
| f1_weighted | 0.906976458 | 0.907295947 |

So Core cùng seed: macro +.003345/+.002890, nhưng timeDelay −.026514/−.027439, absent −.015434/−.030225; low256 seed43 −.003425. Không nhận prior từ gain pooled. Hai seed, official test đã dùng lặp; không là kiểm định significance.

## Oracle seed42 — r50 đã đóng

`outlookcheck/mh-y0-oracle-s42-r50/1`: 50/50 rounds, 50 snapshot, 50 own-test CM; source/config/LR/support/teacher/finite audit PASS.
Nguồn: `runs/y-study/wave1/collection/Y0-r50/{audit,diagnostic-audit}.json`, `runs/y-study/analysis/analysis.json`.

| metric | oracle r50 |
|---|---:|
| accuracy | 0.914073643 |
| precision_macro | 0.911329603 |
| precision_micro | 0.914073643 |
| precision_weighted | 0.914147077 |
| recall_macro | 0.924295864 |
| recall_micro | 0.914073643 |
| recall_weighted | 0.914073643 |
| f1_macro | 0.917581486 |
| f1_micro | 0.914073643 |
| f1_weighted | 0.914031019 |

| chỉ số | oracle | Δ so Core seed42 r50 |
|---|---:|---:|
| f1_macro | 0.917581486 | +0.018421755 |
| accuracy | 0.914073643 | +0.004718391 |
| p10 | 0.786666667 | +0.030452639 |
| low256 | 0.868910025 | +0.026737697 |
| timeDelay | 0.376334719 | +0.034327410 |
| absent | 0.755627010 | +0.046302251 |
| benign | 0.835772630 | -0.004969597 |
| other15 | 0.953664604 | +0.017361378 |

Luật khóa: **generator_worth_it**, vì macro tăng ≥.01. Oracle dùng dòng thật từ pool rank để sinh synthetic, không là cơ chế FL triển khai được; chỉ định vị dư địa của generator. Một seed, official test đã đọc lặp; không chứng minh generator mới nào có utility hoặc tương thích ngân sách. Full private state không pull mặc định, audit không prediction replay/full-bank reconstruction.

## Ablation noq / nofloor — r50 đã đóng (08/11)

Nguồn: `runs/y-study/analysis/analysis.json` (`scripts/analyse_y50.py`, luật `y50_common.ablation_rule`), audit r50 `runs/y-transfer/Y{3,4,5,6}/audit-r50/`.
Δ so Core cùng seed tại r50 (noise: macro/accuracy .0005, p10 .006, low256 .003, timeDelay .0035).

| arm | seed | Δaccuracy | Δmacro | Δp10 | Δlow256 | ΔtimeDelay | Δbenign | Δabsent |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| noq (Y3) | 42 | **−.0038** | −.0001 | −.0036 | −.0008 | +.0113 | −.0096 | −.0006 |
| noq (Y4) | 43 | **−.0038** | −.0001 | −.0054 | **−.0049** | +.0131 | −.0098 | −.0193 |
| nofloor (Y5) | 42 | −.0001 | −.0003 | **−.0178** | **−.0276** | −.0014 | −.0001 | −.0913 |
| nofloor (Y6) | 43 | .0000 | −.0001 | **−.0157** | **−.0281** | −.0007 | +.0001 | −.1080 |

- **Q needed**: accuracy giảm quá ngưỡng cả hai seed (benign −.010; Q giữ ranh giới benign, đổi lại timeDelay +.012 khi bỏ Q).
- **Floor needed**: p10/low256 giảm cả hai seed, absent −.09/−.11 — floor bảo vệ xe nhỏ/lớp vắng, không ảnh hưởng pooled macro.
- Hai seed, test đã dùng lặp; không là significance. Core giữ nguyên Q + floor.
