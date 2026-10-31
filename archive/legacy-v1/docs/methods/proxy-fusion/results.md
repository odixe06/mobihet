# ProxyFuse — kết quả probe hậu kiểm (15/10/2026)

Kernel `nguyenminhtran060113/mobihet-pf-probe-20261015-v1` COMPLETE; input SHA manifest PASS trong kernel.
Weights ct_dml r20 ClassTail (DVC MD5 PASS). Tái hiện argmax private head/gp/gp_legacy và proxy head:
**0 dòng lệch** trên 2 seed × 2 holdout × 20 client × 65.536 dòng. [Contract khóa](design.md).

## Group (primary), mean20client, F1 macro

| seed | proxy head | private gp | private gp_legacy | **fuse_gp (primary)** | fuse_prob_gp (diag) | Δ primary − proxy | benign Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | **0,7735** | 0,6584 | 0,6215 | 0,7415 | 0,7483 | −0,0320 | −0,0207 |
| 43 | **0,7719** | 0,6502 | 0,6183 | 0,7382 | 0,7448 | −0,0337 | −0,0198 |

Old holdout cùng chiều: proxy 0,7579/0,7565, fuse_gp 0,7340/0,7313, private gp 0,6564/0,6478.
Đủ10metrics/per-class/per-client tại `methods/proxy-fusion/builds/fuse-probe/artifacts/kaggle/pf-probe-20261015-v1/proxyfuse/proxyfuse_results.json`.

Bổ sung (mô tả, group, tổng 20 client): private gp đúng mà proxy sai 26.642/29.662 dòng (≈2,0–2,3%);
proxy đúng mà private sai 137.012/149.318 (≈10,5–11,4%).

## Kết luận có phạm vi

**FAIL cổng khóa** cả hai seed. Ở r20 pilot, private DML không mang tín hiệu bổ sung đáng kể ngoài proxy trên
receiver chưa thấy và đoạn thời gian cuối; trộn private làm proxy kém đi. Không quét β/T để cứu.
Hệ quả owner chốt15/10: **proxy head làm headline DML/Hybrid**, private gp báo kèm.
Không suy cho private khác kiến trúc (R2), proxy bị nén, hoặc metric theo receiver riêng của từng xe.
