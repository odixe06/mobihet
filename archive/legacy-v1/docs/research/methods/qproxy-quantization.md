# QProxy q4 — lượng tử int4 theo hàng cho update proxy

**Chốt 20/10/2026 (headline frz/gkd/la/kdpx): q3 + LZMA2 (`q3_zl`) PASS cả hai seed**, ep15 private head Δ −0,0024/+0,0032 so dense;
byte proxy 66,8/66,7 MB cho 20 client × 15 round. Q4_zl giữ làm phương án dự phòng; 2-bit scale max, uplink-only và RMS đều FAIL trên headline mới.
Q2 PASS phía dưới chỉ thuộc gpool cũ, không chuyển sang frozen teacher. [Vòng 20–23](../../methods/full-drift/results.md).

**Trạng thái (17/10 ~14:00Z):** Utility noninferior PASS (m4b_q4 s42, s43 −0,003); **codec wire thật `QPacket`** (fd.py, source sau r11):
byte = `len(packet)` = 208.268 B/chiều, train bit-exact với q4 đã audit (test `--part r11`, `checks`).

## Cơ chế
Up: delta proxy so replica, int4 theo hàng (dim0), scale fp16, làm tròn ngẫu nhiên; BN buffers fp16, integer thô.
Down: server giữ model chính xác (error feedback), gửi int4 nearest của (exact − replica).

## Trao đổi
`QPacket` (17/10): header 40 B (magic MHQ4, schema, bits, base/version, độ dài, context, CRC32) + theo thứ tự state_dict:
counter int64, BN fp16, tham số = scale fp16 mỗi hàng + mã q+8 gói hai mã/byte. Server chỉ cộng tensor decode từ bytes
(lên) và client dựng proxy từ bytes xuống. **208.268 B/chiều** (dense fp32 1.593.440 B ⇒ 7,65×); bộ đếm cũ 208.260 B.
Decode trả đúng tensor `quantize()` (mã −0,0 đọc lại +0,0, không đổi giá trị).

## Bằng chứng (audited)
Pilot r20 hai seed: proxy/private ±0,001. FullDrift: q4 noninferior (dml, m4b ±0,000; m4b s43 −0,003).

## Việc tiếp
Ghép headline: `fd_dml_m4b_gpool_q4` (noninferior so `gpool`; vòng 12 qa).

## q2 — ternary 2-bit (C2, đề xuất owner 17/10; vòng 13 `fd_dml_m4b_gpool_q2`, từ đầu ep1→6)
q4 (208.268 B/chiều = 416.536 B up + down/client/round) vượt mốc B2 231.000 B ≈ 1,80× trước moment/prior/ACK/retry ⇒ thử
precision thấp hơn, **giữ mọi tham số/tầng** (không pruning/slim/subspace). `quantize(·, 2)`: q ∈ {−1, 0, 1}, scale fp16 theo
hàng, làm tròn ngẫu nhiên lên, nearest + EF phía server xuống; `QPacket(bits=2)` (magic MHQ2) gói mã q+2 bốn mã/byte, byte
= `len(packet)` (đo ở smoke trong test `r13`). Dự toán owner ≈ 109.512 B/chiều, 219.024 B/round — còn ≈ 12 KB dưới B2.
Cổng noninferior so gpool (Δ ≥ −0,010, benign ≥ −0,010; báo client tệ nhất, so kèm gpool_q4). q4 giữ utility **không chứng
minh** q2 giữ. Chỉ khi lỗi tập trung ở tensor cụ thể mới mở mixed q2/q4 (chọn trên calibration, byte nâng q4 phải nằm trong
packet). Nếu q2 fail hoặc overhead vượt B2: ghi **DML chưa đạt R1 tại cap này**, ưu tiên thay kênh hoặc chốt cap thực tế bằng
throughput/contact evidence; không lặp sketch/rolling subnetwork đã FAIL. Ngân sách chung: [communication budget](../communication-budget.md).

**Kết quả q2 (audited):** ep6 private 0,7175 (+0,0001 vs gpool), proxy 0,688 (−0,124); **ep15 private 0,7334 (−0,0008 vs gpool@e15,
noninferior PASS), test 0,7499; proxy 0,725 (−0,118)**. 109.512 B/chiều (219.024 B/round, dưới B2 231.000 B trước overhead). Private
không phụ thuộc chất lượng proxy (khớp probe 15) ⇒ q2 dùng được nếu target là private; proxy không còn là model triển khai tốt.

## Nguồn
Nội bộ: [AntiMem](../../methods/anti-memorization/results.md), [FullDrift](../../methods/full-drift/results.md), [plan §3.1](../../archive/research/algorithm-improvement-plan-2026-10-17.md).
