# Vì sao Gaussian/DML/Hybrid chững sớm — chẩn đoán 15/10/2026

Owner hỏi15/10: các phương pháp tốt nhất chững từ round đầu, metrics chưa đạt. Số dưới đây từ artifact full đã audit
(official test, **chỉ mô tả**; không dùng để chọn HP) và pilot nội bộ group. Dùng cùng [LESSONS](../../../LESSONS.md).

## 1. Bằng chứng

| arm (full, seed42) | train CE r1→cuối | test F1 macro r1 → cuối | đọc |
|---|---|---|---|
| local | 0,169→0,016 (r16) | 0,562 → 0,513 | đỉnh r1, giảm đơn điệu |
| Gaussian | 0,207→0,030 (r14) | 0,644 → 0,600 | đỉnh r1, giảm đơn điệu |
| DML private gp / proxy head | 0,200→0,058 / 0,279→0,124 | 0,610→0,732 / r5 0,776 → r20 0,788 | proxy bão hòa từ r5 |
| Hybrid private gp / proxy head | 0,234→0,080 | 0,676→0,733 / r3 0,776 → r15 0,776 | proxy bão hòa từ r3 |

- Full: 43M dòng/20 client, mỗi round = 1 epoch ≈4.200 bước/client giữa hai lần tổng hợp. Pilot2,5%: ≈100 bước.
  Pilot **không** tái hiện suy giảm (local bão hòa r3, Gaussian r8, DML còn tăng tới r20).
- Proxy DML r5→r20 theo lớp: benign 0,714→0,508, positionMirroring 0,062→0,273, timeDelay 0,137→0,227;
  12/16 lớp ≥0,9. Lỗi **dịch chuyển trong khối nhầm lẫn** {benign, dataReplay, positionMirroring, timeDelay}.
  timeDelay có trần thông tin (centralized 0,23); centralized DAGSNet macro 0,8532 ([kế thừa](../legacy/kien-thuc-ke-thua.md)).
- LR cosine T50: r20 vẫn 6,8e-4. Pilot T20 cho lợi lớn ở cuối lịch (DML private gp r15 0,624 → r20 0,658,
  head 0,509→0,591); repo anh em NILM-FL proxy r50 0,8228.
- ProxyFuse: private đúng mà proxy sai chỉ ≈2% dòng group; trộn private làm proxy kém −0,032
  ([kết quả](../../methods/proxy-fusion/results.md)). CCR/ClassTail/class-mass đều trong ±0,002 của local.

## 2. Nguyên nhân (có phạm vi)

1. **Private học thuộc receiver local**: mỗi round full quá dài nên Gaussian/local đạt điểm tổng quát tốt nhất sau r1,
   các epoch sau chỉ fit thêm vân tay receiver (shortcut η²≈0,8–0,95). Gaussian synthetic không chặn được.
2. **Model triển khai sai**: tri thức quần thể nằm trong proxy chung; private chỉ nhận KD trên x local nên trễ 0,05–0,12.
3. **Proxy bão hòa** dưới LR cao và client drift (4.200 bước/round); phần còn lại của macro F1 nằm ở khối 4 lớp nhầm lẫn,
   một phần là trần thông tin của 66 feature.

Chưa cô lập tỉ lệ đóng góp của drift so với trần thông tin; full giữ protocol, chờ r50 (owner chốt15/10).

## 3. Quyết định và kiểm tiếp

- Owner chốt: proxy head làm headline DML/Hybrid, private gp báo kèm. Full không đổi protocol §1.
- Pilot r20 hai seed: receiver-swap, M4b, swap+vị trí, EMA+BN local (role) và QProxy q8/q4
  ([contract](../../methods/anti-memorization/design.md)).
- Pilot chỉ kiểm shortcut receiver, không kiểm drift 4.200 bước. Kỹ thuật PASS pilot mới đề xuất kiểm full.
