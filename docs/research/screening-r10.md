# Sàng lọc 10 round — S0–S4 (30/10/2026)

Full 5.343 xe, W = 1, seed 42, lịch LR 50 round chạy r1–r10, Kaggle 2×T4, SHA `6c3536d`, W&B `mobihet-v4`. Cổng: [improvement-plan §3](improvement-plan.md).
**S0 dừng budget 4 h ở r9** (1.066 s/round) ⇒ cổng so ở **r9** (deviation: plan ghi r10). Một seed.

| arm | private F1 r9 | r10 | mean/xe r9 | p10 r9 | proxy r9 | s/round | byte lên / xuống | FLOPs client / dòng (n̄ = 6.993) |
|---|---|---|---|---|---|---|---|---|
| S0 `dml` | **0,8584** | — | 0,7619 | 0,5416 | 0,8672 | 1.066 | 117 KB / 112 KB | 8,97 × fwd |
| S1 `srv` (n_{k,c}) | 0,8520 | 0,8514 | 0,7645 | 0,5655 | 0,7724 | 698 | 44 KB / 106 KB | 5,98 + 4.096/n_k ⇒ 6,57 × |
| S2 `srv` (n_k) | 0,8493 | 0,8451 | 0,7649 | 0,5704 | 0,7813 | 669 | 44 KB / 108 KB | như S1 |
| S3 `cheap` | 0,7818 | 0,7778 | 0,6713 | 0,4483 | 0,2999 | 768 | 114 KB / 111 KB | 5,98 + 16.384/n_k ⇒ 8,32 × |
| S4 `none` | **0,8662** | **0,8679** | 0,7624 | 0,5391 | — | 529 | 544 B / 544 B | 4,48 × |
| FedAvg (repo v2) | 0,8724 (global) | 0,8749 | | 0,6834 | | 224 | 1,6 MB / 1,6 MB | 2,99 × |
| FedProto (repo v2) | 0,8613 (client) | 0,8629 | | 0,4882 | | 231 | 16 KB / 16 KB | 2,99 × |

FLOPs: FlopCounterMode trên toy (`scripts/cost.py` env, K = 8, fit n = 64 / 512 ⇒ phần theo dòng + phần cố định theo xe).
Server distill của `srv` (200 × 512 × 3 fwd mỗi round ≈ 57 fwd-dòng/xe) không tính vào client. Xe nhỏ hơn n̄ chịu phần cố định nặng hơn.

## Cổng

| | điều kiện | đo | kết luận |
|---|---|---|---|
| S1 vs S0 | Δ private ≥ −0,005; Δ p10 ≥ −0,010; FLOPs ≤ 0,7 × | −0,0064; +0,024; 0,73 × | **FAIL** (sát cả hai ngưỡng) |
| S3 vs S0 | như S1 | −0,0766; −0,093; 0,93 × | **FAIL** |
| S2 vs S1 | mô tả | −0,0027 r9 / −0,0063 r10 | gộp theo lớp n_{k,c} không thua n_k |
| S4 vs S0 | mô tả | **+0,0078**; −0,0025; 0,50 ×; byte 214 × ít hơn | tốt nhất + rẻ nhất trong 5 arm |

## Đọc

- Proxy chưng cất ở server (S1) chỉ 0,77, thấp hơn proxy DML 0,87: teacher yếu ⇒ private thua S0 0,006. Phần cố định 4.096 dòng/xe
  ăn lợi FLOPs. Tri thức qua proxy không còn bù được chi phí.
- Proxy KD chỉ trên Gaussian ở xe (S3) sụp (0,30), kéo private đi xuống từ r3 ⇒ synthetic **không** đủ để học proxy (khớp v1: synthetic
  tốt làm tập chuyển giao, kém làm dữ liệu học).
- Bỏ hẳn proxy (S4, KD về xác suất trung bình theo lớp kiểu FedDistill) mạnh nhất ở r9–r10 và còn tăng đều (+0,0017/round r9→r10).
  Thua FedAvg global 0,006–0,007, thua xa ở p10 (0,54 vs 0,68): xe yếu chưa được lợi từ gộp.
- S0 có cú tụt r2 (0,708), từ r3 hồi phục — đặc điểm DML đã biết.
