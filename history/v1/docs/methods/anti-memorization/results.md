# AntiMem + QProxy — kết quả pilot r20 hai seed (audited 15/10 ~19:00Z)

Contract khóa trước push: [design](design.md). Group holdout, mean20client, r20; đủ 10 metrics/per-class:
[report](results/report.md), [gates.json](results/gates.json), [metrics-all.csv](results/metrics-all.csv).
Sessions: `nguyenminhtran060113/mobihet-am-s4{2,3}-20261015-v1` (s42: 10/11 arm r20, `am_hybrid_swappos` r19 thay
bằng tail), tail `odixeuit/…s42…tail-v2`, `minhtrit06/…s43…tail-v2` (4/4). Audit: executed source, input SHA,
marker chain, weights r10/r20 strict/finite, predictions → confusion → 10 metrics.

## Cổng (Δ so `am_<base>`, seed42 | seed43)

| kỹ thuật | DML proxy head | Hybrid proxy head | Gaussian private gp | kết luận |
|---|---:|---:|---:|---|
| swap | −0,004 \| +0,001 | −0,006 \| −0,005 | **+0,013 \| +0,013** | PASS Gaussian; FAIL proxy |
| m4b | −0,002 \| +0,006 | −0,002 \| −0,006 | **+0,025 \| +0,025** | PASS Gaussian (lớn nhất); FAIL proxy |
| swappos | −0,012 \| −0,003 | −0,006 \| −0,009 | **+0,016 \| +0,018** | PASS Gaussian; FAIL proxy |
| EMA role (gốc) | proxy_ema −0,030 \| −0,034 | −0,028 \| −0,028 | private_ema **+0,039 \| +0,035** | PASS Gaussian; proxy EMA hại |

Benign Gaussian tăng +0,021…+0,046 ở mọi kỹ thuật PASS. Kết hợp tốt nhất Gaussian: swappos + private_ema
0,6287/0,6297 (gốc 0,5711/0,5740, +0,058/+0,056). DML private gp (không primary) cũng tăng với m4b +0,021/+0,028,
swap +0,016/+0,021; Hybrid private gp m4b +0,009/+0,011.

**QProxy PASS hai seed:** q4 − dense proxy head +0,001/−0,000, private gp −0,000/−0,001, benign ≥ −0,001;
q8 sanity ≤0,001. q4 = 208.260B/chiều/round, nhỏ hơn dense fp32 7,65×.

## Đọc

- Shortcut receiver hại **model chỉ thấy dữ liệu local** (Gaussian/private): bỏ hoặc phi tương quan khối receiver
  và trung bình quỹ đạo giúp rõ. Proxy chung đã được tổng hợp qua 20 client nên shortcut không còn là nút thắt;
  can thiệp input làm proxy hơi kém, EMA proxy kém rõ (trung bình qua các round đầu kém hơn).
- Pilot không kiểm drift full; Gaussian swappos/private_ema đang được kiểm thẳng trên full trong FullDrift.
- Tái lập: `am_hybrid_swappos` s43 chạy hai lần (nguyenminhtran060113 và minhtrit06) cho weights/predictions SHA
  giống hệt r1/r10/r20.

## Đề xuất kế tiếp (chưa launch)

- QProxy q4 cho DML/Hybrid full: giảm byte proxy 7,65× không mất utility ở pilot; cần owner duyệt kiểm full.
- Gaussian m4b (+private_ema) trên full nếu FullDrift xác nhận Gaussian swappos/EMA có lợi ở full.
