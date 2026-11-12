# B19 — cụm biên theo nhầm lẫn của generator (A2'), khóa 12/11/2026

Owner 12/11 duyệt A2' với yêu cầu **không lặp sai lầm của các hướng cũ**; nếu generator hết khả thi thì được đổi
định hướng, miễn giữ **đa kiến trúc** và **băng thông hạn chế**. Bối cảnh/trần: [sổ tay](method-building-brief.md) §2–§5.
CBB giữ nguyên là phương pháp chính cho tới khi một bậc thang PASS.

**Owner 12/11 (giữa phiên): tạm không dùng lượng tử hóa** để train tốt nhất; giảm băng thông thử sau. Vì vậy nền của **cả hai arm**
là KM16 w0 .5 **không lượng tử** (= công thức KM50: moment fp32, mean/Cholesky xuống fp16 như bản gốc, Q fp16; không lite6, không QQ8).
Thành phần biên dùng thống kê fp32, xuống fp16 như cụm KM. Số đối chiếu A1 (generator CBB) không dùng làm baseline gate;
diagnostic tính lại cả hai generator trong cùng lượt.

## Bằng chứng → cơ chế → lợi kỳ vọng → dấu hiệu bác bỏ

- **Bằng chứng.** A1 ([diag](../../runs/a1-gen-posterior/diag.json)): dùng generator CBB làm Bayes-classifier trên dòng thật fold val,
  recall thấp nhất đúng ở các lớp yếu của CBB — TD .31, benign .76, lớp7 .78, lớp11 .80, lớp2 .86. Bottleneck 08/11: synthetic lệch
  ranh giới thật đúng các lớp 1/7/4/10/3/13; oracle (dòng thật) +.010 macro so Core ⇒ dư địa generator có, nhưng K32/EM/whitening
  (tăng cụm/LL đều mọi lớp) đã FAIL.
- **Cơ chế.** Generator không mô tả vùng mà lớp y chồng lên lớp c'. Thêm thành phần mixture cho lớp y **chỉ ở vùng dòng thật lớp y
  bị chính generator đoán là c'**, đo trên dòng thật toàn mạng. Lớp và vùng nhận thêm cụm do dữ liệu quyết định, không định sẵn benign.
- **Lợi kỳ vọng.** Synthetic phủ đúng vùng biên ⇒ model riêng bớt FP/FN ở các cặp nhầm của lớp yếu; trần là oracle (+≤.010).
- **Bác bỏ.** (1) Gate local dưới đây trượt; (2) sim Δmacro < +.002 ở một seed; (3) chỉ đổi benign↔TD mà macro không tăng.

## Thuật toán (một lượt thêm ở bootstrap, round không đổi)

1. Sau KM16 w0 .5 (không lượng tử), mỗi xe dùng **bảng như xe nhận** (CDF/mass lớp, km_p, km_mu/km_L fp16) chấm mọi dòng train thật một lần:
   `log f_c(x) = logsumexp_k[log p_ck + log N(z_c; μ_ck, L_ck)] + Σ_j [log(m_cj/Δ) + z_cj²/2 + log√(2π)]`, `z_c` = **mid-rank** của lớp c
   (đúng cách fit thống kê; A1 cho thấy PIT chính xác tại atom sai hoàn toàn), Δ = 8/256, mass sàn 1e-7, quyết định
   `argmax_c log f_c(x) + log π_c` với π toàn mạng (count đã có trong bootstrap, Laplace +1).
2. Dòng nhãn y bị đoán c' ≠ y: theo cặp (y, c') xe tính count, mean và covariance population của `z_y` (mid-rank lớp y); **ô < 8 dòng
   tại một xe không gửi**. Gửi count + Σz + Σzzᵀ fp32 (không lượng tử, owner 12/11).
3. Server cộng; giữ cặp ≥ 20 dòng toàn mạng (như luật giữ cụm KM). Mỗi cặp = một thành phần lớp y (tối đa 15/lớp):
   mean = s1/n, cov = s2/n − mmᵀ chiếu PSD, co về covariance tổng của mixture lớp y với w0 = .5·D (như KM), +1e-4 I.
4. Trọng số: thành phần n/N_y, cụm KM cũ nhân (1 − Σn/N_y). **Số dòng synthetic mỗi lớp giữ 4/bước (64 cân bằng)**; chỉ đổi thành phần
   trong lớp. Xấp xỉ đã biết: cụm cũ vẫn chứa dòng biên (đếm hai lần một phần), gate local bắt nếu méo.
5. Downlink broadcast: p fp32, mean và Cholesky fp16 (như cụm KM gốc).
   Byte lên/xuống đo thật và cộng vào bootstrap; Q, loss, floor, optimizer, số bước, eval **không đổi**.

## Sai lầm cũ và cách khóa tránh

| sai lầm đã gặp | nguồn | khóa trong B19 |
|---|---|---|
| Fidelity/LL tăng nhưng utility giảm (EM1/EM2/whitening, standard/forest TSTR tốt) | [generators](generators.md), LESSONS | Gate local chỉ là **bộ lọc loại** đo khả năng phân biệt trên dòng thật held-out, không LL; quyết định chỉ bằng sim utility |
| Tăng cụm đều mọi lớp (K32 FAIL) | b3/b9 | Chỉ thêm ở cặp nhầm đo được, ≤15/lớp, ngưỡng 8/20 dòng cố định |
| Đổi tỷ lệ/ưu tiên synthetic giữa lớp (absent +.084 nhưng macro −.0045; benign share; temp) | LESSONS, b10 | Giữ 4 dòng/lớp/bước; trọng số thành phần theo count thật trong lớp |
| Q đo trên synthetic FAIL | b10 QS | Q vẫn đo trên dòng thật, không đụng |
| Chứng minh trên đường không triển khai (codec/fp64 server) | LESSONS codec | Chấm bằng bảng fp16 như xe nhận; thành phần mới qua đúng fp16 trước khi dùng |
| PIT chính xác tại atom (A1 acc .02) | A1 | Mọi chấm dòng thật dùng mid-rank |
| Ô moment nhỏ lộ dòng | privacy 09/11 | Ô < 8 dòng/xe không gửi; giữ ≥ 20 dòng toàn mạng |
| Chọn ngưỡng/HP sau khi xem số | protocol | Một cấu hình duy nhất khóa ở đây, không lưới, không chỉnh sau gate |
| Gain r20 chỉ là tăng tốc (bank F50) | F50 | Thang sim → pilot r20 cos50 → r50 hai seed, so control cùng round |
| Gói thiếu file khi deploy (B18 `core_hp.txt`) | ops lessons | Overlay khai đủ `methods/mobihet_boundary.py` + chuỗi kế thừa + `runs/sim/core_hp.txt`; smoke compile 200 xe 2 round trước quota |
| Đọc benign↑ TD↓ như thắng | B17/B18 | Báo đủ CM/10 metric; macro là gate duy nhất, guard phụ report-only |

## Gate

**Local (trước mọi push), `scripts/diag_gen_posterior.py --boundary`:** sim 200 xe panel seed7, cùng bootstrap; Bayes-classifier
(mid-rank, prior toàn mạng) trên 211.118 dòng thật fold val: macro(augmented) − macro(KM16 không lượng tử, cùng lượt) **≥ +.010** **và**
ít nhất 3/4 lớp TD/benign/7/11 không giảm recall. Trượt ⇒ dừng B19, không sim.

**Sim (B19, `SIM_BATCH=b19`):** C (`km_boundary=off`, exact KM16 w0 .5 không lượng tử) và BD (`km_boundary=on`) × seed 42/43, r20 cosine20, 200 xe fold val,
cap 2,75 h. **PASS khi Δmacro r20 ≥ +.002 ở từng seed**; p10/low256/TD/benign/absent report-only. PASS ⇒ tự lên pilot full-scale r20
cos50 seed42 so CBB cùng round (luật như B17), rồi r50 hai seed. FAIL ⇒ dừng, không tune/extend/combo.

**Chi phí báo:** bootstrap lên/xuống thêm (đo thật), compute thêm = một lượt chấm generator trên train local (×F so một epoch forward),
byte/round và compute/round không đổi.

## Nếu FAIL

A2' là thử cuối cho hướng “sửa generator theo bằng chứng”. FAIL ở local hoặc sim ⇒ coi kênh generator (data-space) đã hết dư địa
trong ngân sách hiện tại; bước kế là đề xuất định hướng mới giữ đa kiến trúc + băng thông thấp (owner 12/11), không thử biến thể generator khác.

## Kết quả gate local — 12/11/2026: **FAIL → dừng B19, không sim**

[diag.json](../../runs/b19-boundary-diag/diag.json), `scripts/diag_boundary.py` (gọi đúng `GenScore`/bootstrap của module train).
Bootstrap: 1.476.064 dòng train chấm, generator train-acc .874; 2.729 ô gửi, 56 cặp giữ (lớp benign 10, lớp7 7, TD 7 thành phần);
tỷ trọng biên TD .584, benign .224, lớp7 .155. Thêm bootstrap ≈ 168 KB/xe lên, 346 KB xuống (fp32/fp16).

| Bayes-classifier generator, fold val 211.118 dòng | macro | acc | recall TD | benign | lớp7 | lớp11 |
|---|---:|---:|---:|---:|---:|---:|
| KM16 w0 .5 không lượng tử (nền) | .8926 | .8892 | .331 | .804 | .805 | .792 |
| + thành phần biên | .8956 | .8953 | .302 | .835 | .801 | .806 |

Δmacro **+.0030 < +.010**; TD/lớp7 giảm (2/4 lớp yếu không giảm < 3/4). Benign/lớp11 tăng nhưng TD tụt: cùng kiểu đổi benign↔TD đã thấy ở
B17/B18. Theo mục “Nếu FAIL”: **kênh generator coi như hết dư địa** trong ngân sách hiện tại; không thử biến thể generator khác.

Số phụ (độ khớp, không phải utility): cùng thước đo, generator CBB lượng tử (lite6/int8, A1) .8802 vs không lượng tử .8926.
Lượng tử làm giảm khả năng phân biệt của generator; r50 CBB vs KM50 chỉ −.0003 macro nên chưa suy ra utility.
