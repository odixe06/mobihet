# L50 — Core lossless CDF, full 50 round, seed42

04/11/2026: Đủ 50 round qua 2 session cross-account (`emtrangtri` r1–25, `chuchimnon` r26–50); audit toàn diện source/CM/checkpoint/codec PASS.
Cấu hình chuẩn: Cosine50, 5.343 xe, DAGSNet W1, floor16, copula 256 bin, lossless zlib-6 uint16 CDF.
Thiết kế: [screening-l50](screening-l50.md). Artifact: `runs/l50-study/{closure,collection/L0/audit,collection/L0/diagnostic-audit}.json`.

## Kết luận

1. **Bảo toàn 100% chất lượng biểu diễn (bit-exact decode & trajectory match)**:
   - Lossless CDF lượng tử hóa `uint16` little-endian rồi nén `zlib` mức 6 kèm header 32B CRC32; client giải mã bit-exact (`bitexact_cdf: true`, `bitexact_mass: true`).
   - Điểm cuối r50: macro F1 đạt **0.899129** (so với F2 Core gốc 0.899160, chênh lệch $\|Δ\| = 0,000031$, tức 0,003%); Accuracy đạt **0.909285** (so với 0.909355, $\|Δ\| = 0,000071$); Train loss **0.141436** (so với 0.141414, $\|Δ\| = 0,000022$).
   - Phân phối nhóm xe yếu: Client P10 đạt **0.752754** (so với 0.756214); Client mean đạt **0.863278** (so với 0.863357).
2. **Tiết kiệm băng thông thực tế (verified payload)**:
   - Phân bố biên downlink (CDF 256 bin): 630.784 B → **267.411 B** (tiết kiệm −57,6%).
   - Bootstrap 2 chiều/xe: 1.000.439 B → **637.066 B** (tiết kiệm −36,3%).
   - Tổng communication budget 50 round/xe: 1.054.839 B → **691.466 B** (tiết kiệm −34,45%, giảm 363.373 B/xe).
   - Toàn mạng 5.343 xe qua 50 round: 5,63 GB → **3,69 GB** (giảm ~230 lần so với FedAvg 852 GB).

## Đủ 10 metric pooled r50

| STT | Metric | L0 (Lossless CDF r50) | F2 (Core trước r50) | Chênh lệch (Δ) |
|:---:|---|---:|---:|---:|
| 1 | `accuracy` | 0.909285 | 0.909355 | -0.000071 |
| 2 | `precision_macro` | 0.898569 | 0.898246 | +0.000323 |
| 3 | `precision_micro` | 0.909285 | 0.909355 | -0.000071 |
| 4 | `precision_weighted` | 0.903027 | 0.903132 | -0.000105 |
| 5 | `recall_macro` | 0.901863 | 0.902222 | -0.000360 |
| 6 | `recall_micro` | 0.909285 | 0.909355 | -0.000071 |
| 7 | `recall_weighted` | 0.909285 | 0.909355 | -0.000071 |
| 8 | `f1_macro` | 0.899129 | 0.899160 | -0.000031 |
| 9 | `f1_micro` | 0.909285 | 0.909355 | -0.000071 |
| 10 | `f1_weighted` | 0.905337 | 0.905430 | -0.000093 |

## Phân phối client cohort r50

| Chỉ số | L0 (Lossless CDF) | F2 (Core trước) | Chênh lệch (Δ) |
|---|---:|---:|---:|
| Client F1 mean | 0.863278 | 0.863357 | -0.000079 |
| Client F1 p10 | 0.752754 | 0.756214 | -0.003460 |
| Client F1 median | 0.874883 | 0.875518 | -0.000635 |
| Client F1 min | 0.150000 | 0.153846 | -0.003846 |
| Client accuracy mean | 0.880626 | 0.880852 | -0.000226 |
| Train loss | 0.141436 | 0.141414 | +0.000022 |
| Max skipped frac | 0.000000 | 0.000000 | 0.000000 |

## F1 từng lớp tại round 50

| Lớp tấn công | L0 (Lossless) | F2 (Core) | Chênh lệch (Δ) |
|---|---:|---:|---:|
| benign | 0.840604 | 0.840742 | -0.000139 |
| accelerationMultiplication | 0.880357 | 0.879524 | +0.000833 |
| constantPositionOffset | 0.922666 | 0.922969 | -0.000303 |
| constantSpeedOffset | 0.950155 | 0.950151 | +0.000004 |
| dataReplay | 0.944695 | 0.944299 | +0.000396 |
| dosAttack | 0.995441 | 0.995471 | -0.000031 |
| feignedBraking | 0.940136 | 0.940029 | +0.000107 |
| positionMirroring | 0.852055 | 0.852538 | -0.000482 |
| randomPositionOffset | 0.987155 | 0.987230 | -0.000075 |
| randomSpeedOffset | 0.977155 | 0.977205 | -0.000049 |
| reversedHeading | 0.978882 | 0.979011 | -0.000130 |
| suddenConstantSpeed | 0.830619 | 0.830358 | +0.000261 |
| suddenStop | 0.986490 | 0.986366 | +0.000124 |
| timeDelayAttack | 0.340694 | 0.342007 | -0.001313 |
| trafficCongestionSybil | 0.995547 | 0.995485 | +0.000062 |
| zeroSpeedReport | 0.963412 | 0.963170 | +0.000242 |

## Communication budget hai chiều/xe

| Khoản mục | Core gốc (uncompressed) | L0 (Lossless CDF) | Tiết kiệm |
|---|---:|---:|---:|
| Bootstrap uplink (histogram) | 271.031 B | 271.031 B | 0 B |
| Bootstrap downlink (CDF + Cholesky) | 729.408 B | 366.035 B | −363.373 B (−49,8%) |
| *Trong đó CDF packet* | *630.784 B* | *267.411 B* | *−363.373 B (−57,6%)* |
| **Tổng bootstrap 1 lần** | **1.000.439 B** | **637.066 B** | **−363.373 B (−36,3%)** |
| 50 round Q matrix (up + down) | 54.400 B | 54.400 B | 0 B |
| **Tổng communication budget 50 round** | **1.054.839 B (~1,05 MB)** | **691.466 B (~675 KB)** | **−363.373 B (−34,45%)** |
| **Tổng toàn mạng (5.343 xe)** | **5,63 GB** | **3,69 GB** | **−1,94 GB** |

## Giới hạn bằng chứng

- Đã audit 50 CM riêng từng round của cả hai rank, 50 snapshot nhỏ, log sessions, full resume checkpoint verification SHA, teacher chain và CRC32 codec.
- Một seed duy nhất (seed 42); split v4 chuẩn hóa 80% train+val / 20% test timeline.
- Không prediction→CM replay; snapshot nhỏ không chứa đầy đủ private weights toàn mạng.
