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

# Sàng lọc bộ sinh synthetic trên S4 `none` — G0–G4 (30/10/2026)

Cùng protocol, SHA `f86e137`, hp `synth` (docstring [methods/mobihet.py](../../methods/mobihet.py)). Đối chứng S4 (`gauss`). Audit: 10/10 round,
không có giá trị không hữu hạn, skipped 0, HP đúng protocol. Một seed.

| arm | synth | private F1 r10 | Δ vs S4 | mean/xe | p10 | Δ p10 | s/round |
|---|---|---|---|---|---|---|---|
| S4 | gauss (Gaussian gộp) | 0,8679 | — | 0,7724 | 0,5671 | — | 529 |
| G0 | off (không synthetic) | 0,8497 | **−0,0182** | 0,7289 | 0,4891 | −0,0780 | 449 |
| G1 | oracle (dòng thật client khác, không triển khai được) | **0,8802** | **+0,0123** | 0,7879 | 0,5896 | +0,0225 | 537 |
| G2 | short (ưu tiên lớp xe thiếu) | 0,8669 | −0,0010 | 0,7647 | 0,5423 | −0,0248 | 522 |
| G3 | mix (FedMix) | 0,8667 | −0,0012 | 0,7775 | 0,5796 | +0,0125 | 543 |
| G4 | copula | 0,8732 | **+0,0053** | 0,7793 | 0,5738 | +0,0067 | 546 |

## Cổng

| | điều kiện | đo | kết luận |
|---|---|---|---|
| G1 trần | G1 − S4 ≥ +0,005 thì còn chỗ cho bộ sinh | +0,0123 | **còn chỗ** ⇒ tiếp tục mảng bộ sinh |
| G2 | Δ ≥ +0,010 và Δp10 ≥ −0,010 | −0,0010; −0,0248 | **FAIL** |
| G3 | như G2 | −0,0012; +0,0125 | **FAIL** (p10 và mean/xe tăng, pooled không) |
| G4 | như G2 | +0,0053; +0,0067 | **FAIL** cổng gain, nhưng lấy được ~43 % khoảng trần (0,0053 / 0,0123) |
| G0 | mô tả | −0,0182; −0,0780 | synthetic **cần** cả khi không có proxy, nhất là cho xe yếu (p10) |

## Đọc

- Không proxy vẫn cần synthetic: bỏ đi mất 0,018 pooled và 0,078 ở p10 — dòng Gaussian là input duy nhất của lớp xe không có.
- Dòng thật của client khác hơn Gaussian 0,012 ⇒ chất lượng input còn đáng cải thiện. Copula (biên thật hơn) lấy gần một nửa;
  khớp v1 (copula tăng độ đồng ý nhãn 0,47 → 0,67). Ở v1 với proxy teacher, copula FAIL utility; ở đây (teacher Q[y], không phụ
  thuộc mẫu) nó có ích hơn.
- Đổi phân phối nhãn (G2) không giúp và hại p10. Mixup (G3) chỉ áp dụng cho lớp xe đã có ⇒ không chạm lớp thiếu, chỉ đỡ trung bình/xe.


## Sửa kế toán byte 30/10 ~17:10

544 B là **mỗi chiều**, tức 1.088 B/xe/round. Các bảng lên/xuống ở trên đúng nhưng lời tóm tắt cũ bỏ bootstrap.
S4 bootstrap hai chiều 394.368 B/xe; G4 3.311.744 B/xe. So chi phí end-to-end phải cộng khoản này, xem [H0–H4](screening-h10.md).
Không thay kết quả F1 hoặc cổng đã khóa.
