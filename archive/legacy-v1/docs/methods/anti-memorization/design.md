# AntiMem + QProxy — chống học thuộc receiver và nén proxy DML

Owner chốt15/10 tối: thử mọi kỹ thuật chống học thuộc (receiver-swap, M4b, EMA+BN local, kết hợp) và QProxy
trên pilot r20 hai seed, account `nguyenminhtran060113`. Proxy head là headline DML/Hybrid (báo kèm private gp).
Full giữ protocol, chờ r50. Cổng dưới đây khóa **trước** khi push.

## Vấn đề đã đo

- Full mỗi round = 1 epoch ≈4.200 bước/client. Gaussian/local đạt đỉnh test ở r1 rồi giảm
  (Gaussian 0,644→0,597; local 0,562→0,504) trong khi train CE 0,21→0,03: private học thuộc receiver local.
  Vân tay receiver (9 feature noise/profile, η²≈0,8–0,95) là shortcut đã biết ([dataset](../../../knowledge/dataset.md)).
- DML/Hybrid: proxy chung bão hòa ~0,78–0,79 từ r5, private 0,732/0,733; ProxyFuse cho thấy private không thêm
  tín hiệu ngoài proxy ([probe](../proxy-fusion/results.md)).
- Pilot2,5% (~100 bước/round) không tái hiện suy giảm; pilot chỉ kiểm được shortcut receiver, không kiểm drift full.
- DML dense trao đổi fp32 1.593.440B/chiều/round, xa B2≈231kB/contact.

## Cơ chế từng kỹ thuật

| arm suffix | thay đổi | vì sao có thể giúp | dấu hiệu sai |
|---|---|---|---|
| `swap` | 50% dòng real: thay khối 9 feature receiver bằng dòng khác receiver cùng xe (mọi lớp), generator riêng | cắt liên kết vân tay receiver ↔ nhãn, giữ phân phối feature thật | group không tăng ≥0,010 |
| `m4b` | đặt 9 feature receiver + 4 vị trí tuyệt đối (`f_rcv/snd_x/y_rel`) về0 ở train/synthetic/eval | bỏ hẳn shortcut và feature lệch train/test; Build8 +0,074 full | không tăng hoặc benign giảm |
| `swappos` | `swap` + đặt0 4 vị trí | kết hợp: giữ feature receiver nhưng phi tương quan, bỏ vị trí lệch | không hơn `swap`/`m4b` |
| role `*_ema` | EMA trọng số (half-life2 round) của private và proxy chung, BN tính lại trên dữ liệu local mỗi xe | trung bình dọc quỹ đạo giảm dao động/học thuộc cuối round; kế thừa +0,037 | EMA − gốc < +0,010 |
| `q8`/`q4` | proxy DML: up = delta int-b stochastic theo hàng, down = int-b nearest so replica, server giữ model chính xác (EF), BN buffer fp16 | giữ đủ tọa độ (khác sketch/subspace đã FAIL), lỗi lượng tử bị chặn từng tọa độ | q4 kém dense >0,010 |

`am_<base>` tái hiện **bit-exact** v8 `h_<base>` (smoke r1/r3); kỹ thuật chỉ thêm hook input. Mỗi arm có private/proxy
riêng; EMA không đổi training. Không gửi record/activation; swap dùng dòng của chính xe.
q4 actual 208.260B/chiều/round (7,65× nhỏ hơn dense fp32).

## Cổng khóa (pilot r20, group primary, mean20client, mỗi seed42/43)

Primary: DML/Hybrid = proxy head; Gaussian = private gp. Báo đủ10metrics/per-class, old, private gp/head.
- Kỹ thuật PASS nếu technique − `am_<base>` ≥ +0,010 primary **cả hai seed** và benign ≥ −0,005.
- EMA PASS nếu role_ema − role gốc ≥ +0,010 primary cả hai seed (proxy_ema cho DML/Hybrid, private_ema cho Gaussian).
- QProxy: q4 − dense ≥ −0,010 trên proxy head **và** private gp, benign ≥ −0,010, cả hai seed; q8 ≥ −0,005 là sanity.
- Không chọn round/HP theo holdout; không quét SWAP_P/β/bit sau khi thấy kết quả. FAIL ghi phạm vi rồi dừng.
- PASS pilot chỉ đủ đề xuất kiểm full; không là bằng chứng full/R1 contact/R2.

## Giới hạn

M4b danh sách13 feature dựng lại từ định nghĩa kế thừa (không còn code Build8). Swap không chứng minh
counterfactual vật lý. QProxy chưa gồm ACK/retransmission/contact; bootstrap dense ban đầu vẫn tính.
Không W&B resume qua session: controller giữ weights r10/r20/round cuối để tiết kiệm đĩa.
