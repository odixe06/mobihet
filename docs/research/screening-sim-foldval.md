# Simulator fold-train → fold-val — luật khóa 08/11/2026 (trước khi xem kết quả)

Mục đích: chọn ngưỡng T cho A và sàng ứng viên CF bằng FL thật (không judge offline), không đọc official test.
Nguồn chẩn đoán: [bottleneck review](bottleneck-review-20261108.md). Ràng buộc owner: compute ≤ Core, không trao model, giữ R1/R2.

## Thiết lập

- `scripts/sim_fold_val.py`: 200 xe phân tầng theo số dòng fold train (10 tầng quantile, seed 7, xe có ≥32 dòng val);
  ids sha256[:16] `41d3ea2e19a6e703`; 1.476.064 dòng train, 211.118 dòng val. Train = fold train, "test" = fold val của train split.
- Copula/LA/Q tính trên 200 xe (khác 5343 xe của protocol, như nhau mọi arm). Cosine20 đủ 20 round (endpoint ngắn), batch 64,
  AdamW reset, compile, 2×T4/arm. HP Core = F50 Core trừ diagnostics (`runs/sim/core_hp.txt`).
- Launcher `scripts/run_sim_kaggle.py`; code `methods/mobihet_sim.py` (overlay trên HEAD `be3b7d5`, mọi knob tắt = Core).

| arm | thay đổi | compute so Core | byte |
|---|---|---|---|
| S0 | Core seed42 | = | = |
| S1 | Core seed43, đo nhiễu seed | = | = |
| A256/A1024/A4096 | synthetic lớp c có trọng số 0 tại xe có n_kc ≥ T (xe không cần sinh) | ≤ (bỏ dòng sinh) | = |
| CF | twin phản thực: lớp tấn công c∈{1,2,3,6,7,10,11,12,15} = dòng benign thật của batch với họ feature S_c lấy mẫu lại từ copula có điều kiện lớp c; synthetic benign = dòng tấn công thật của batch với S_{c'} lấy mẫu từ điều kiện benign; còn lại như Core | +1 matvec 77×77/dòng synthetic (≈+1,5% F) | = (bảng S_c cố định) |

Cơ chế/dấu hiệu: FP lớp c xảy ra ở xe có c (bottleneck §1); A bỏ synthetic lệch manifold nơi dòng thật đủ → precision lớp 1/7/6/3/10 tăng.
CF dạy "chỉ họ trường bị sửa quyết định nhãn", ngữ cảnh lấy từ dòng thật của xe → benign gần ranh giới c được gán benign.
Bác bỏ: precision các lớp đó không tăng hoặc macro không vượt nhiễu.

## Luật (fold val, r20, private own-model)

Δ = arm − S0. Nhiễu N = |S1 − S0| từng metric. PASS khi đồng thời:
1. Δ f1_macro pooled ≥ max(+.003, N_macro);
2. client F1 p10 ≥ −.006; mean xe <256 dòng train ≥ −.003; F1 timeDelay ≥ −.0035; F1 benign ≥ −.0005;
3. accuracy dòng lớp vắng (lớp có ở val, không có ở train xe đó) ≥ −.01 khi ≥2.000 dòng.

Chọn T: trong A PASS, Δmacro lớn nhất; chênh ≤ .001 thì lấy T nhỏ hơn (ít compute hơn). Không A nào PASS → không chọn T,
không pilot A. CF PASS → preflight + pilot Kaggle r20/cosine50 seed42 full 5343 xe so Core F2 theo gate cũ.
Không tune lại T/arm sau khi xem số; một seed, panel fold val đã dùng cho probe trước → không là xác nhận độc lập.

## Kết quả 08/11 (đã pull, `artifacts/sim-kaggle/`, `runs/sim-kaggle/verdict-r{12,14}.json`)

Cả 6 notebook COMPLETE nhưng **dừng budget trước r20** (cap 1,25 h; setup verify npy + nạp ≈35 phút; T4 ≈150 s/round, chi phối bởi xe lớn nhất):
S0/S1/A1024 r15, A256/A4096 r14, CF r12. Luật được áp ở round chung r12 (đủ 6 arm) và r14 (thiếu CF). **Không phải endpoint r20 đã khóa**:
verdict là tạm, nhưng chênh lớn gấp nhiều lần nhiễu và cùng dấu ở cả r12 lẫn r14.

| arm | Δmacro r12 | Δmacro r14 | ΔP lớp1 / lớp6 r12 | verdict |
|---|---:|---:|---:|---|
| S1 (nhiễu seed) | −.0007 | −.0020 | +.007 / +.007 | — |
| A256 | −.0305 | −.0326 | −.030 / −.074 | FAIL |
| A1024 | −.0131 | −.0155 | −.055 / −.049 | FAIL |
| A4096 | −.0015 | −.0019 | −.007 / −.009 | FAIL |
| CF | −.0087 | — | +.046 / −.101 | FAIL |

S0 macro r12 .9049, r14 .9070. **Không chọn T.** Không pilot A, không pilot CF.

Đọc:
- Bỏ synthetic ở lớp xe đã có nhiều dòng thật làm xấu **đơn điệu theo lượng bỏ**, và precision lớp1/6 *giảm*: giả thuyết "synthetic lệch manifold
  gây FP ở xe có lớp c" **bị bác**. Synthetic lớp c có ích cả khi dòng thật đủ: nó đóng vai regularizer/tri thức toàn mạng chống overfit local.
- CF tăng precision lớp1 (+.046) nhưng làm sụp lớp6 (−.101), timeDelay −.025: lớp1/6 dùng chung họ acl (3/9/11/15 chung spd, 2/7 chung pos),
  nên twin phản thực chỉ khác nhau ở phân phối điều kiện của cùng một họ → lẫn cặp lớp chung trường. Judge offline đã cảnh báo đúng chiều.
- Hệ quả: nút thắt nghiêng về **thiếu** tri thức toàn mạng ở xe lớn, không phải thừa synthetic. Hướng tiếp phải tăng tỷ trọng tín hiệu toàn mạng
  mà không tăng compute (ví dụ đổi tỷ lệ thật/synthetic ở xe lớn), không phải cắt synthetic.

## Đợt b2 — trọng số synthetic + cấu trúc oracle (đủ r20, `runs/sim-kaggle-b2/verdict.json`)

Cap 2,25 h, cả 9 arm đủ 20/20 round. Control S0 macro .9081, nhiễu |S1−S0| .0003 (low256 nhiễu tới .045: chỉ ~20 xe <256 dòng, guard này quá ồn ở 200 xe).

| arm | thay đổi | Δmacro r20 | ghi chú |
|---|---|---:|---|
| L2 / L4 | λ synthetic 2 / 4 mọi xe | −.0027 / −.0043 | timeDelay −.013 / −.023; FAIL |
| L2big / L4big | λ chỉ xe ≥1024 dòng | −.0028 / −.0036 | FAIL |
| O1 | oracle (dòng thật pool rank, chẩn đoán) | **+.0103** | P lớp1 +.070, lớp7 +.053; lặp trần full-scale (+.018 r50) |
| O2 | oracle, mỗi feature từ dòng thật độc lập | −.0216 | thấp hơn copula: phụ thuộc giữa feature đáng ≈.02 |
| O3 | oracle, mỗi khối vật lý từ dòng độc lập | −.0112 | P lớp1 −.161: phụ thuộc **giữa khối** quyết định lớp1 |

Đọc: lượng/tỷ trọng synthetic đã ở gần tối ưu (bỏ bớt hại, tăng trọng số hại) → chỉ còn **chất lượng** bộ sinh. Bộ mô phỏng tái tạo trần oracle,
nên đủ dùng để sàng generator. Bộ sinh phải giữ phụ thuộc toàn 77 chiều (gồm giữa khối); block-wise độc lập không đủ.

## Đợt b3 — KM-copula (đủ r20, control = b2 S0/S1, `runs/sim-kaggle-b3/verdict.json`)

Bộ sinh mới `cop_kmeans=K` (`methods/mobihet_sim.py`): mỗi lớp là mixture K Gaussian trong không gian normal-score của copula,
tâm khởi tạo mu_c ± sqrt(eig)·v từ covariance gộp (không cần dòng nào rời xe), 4 vòng Lloyd federated (tổng/đếm theo cụm), covariance
cụm co về covariance lớp (trọng số 2D dòng), bỏ cụm <20 dòng; marginal và nghịch đảo CDF giữ như Core. Compute sinh = Core
(+1 lần chọn cụm/dòng); bootstrap thêm 4+1 lượt qua dữ liệu local (khoảng cách tới K tâm, ≪ 1 epoch DAGSNet).

| arm | Δmacro | Δp10 | ΔtimeDelay | Δbenign | ΔP lớp1 | byte bootstrap thêm/xe (lên / xuống) | luật |
|---|---:|---:|---:|---:|---:|---|---|
| KM8 | +.0051 | +.019 | +.007 | +.000 | +.056 | 1,74 MB / 0,87 MB | FAIL (chỉ low256 −.006) |
| **KM16** | **+.0057** | **+.031** | **+.012** | **+.001** | **+.062** | ≈3,47 MB / 1,74 MB | **PASS** |
| KM32 | +.0052 | +.019 | +.007 | +.002 | +.065 | 6,95 MB / 3,47 MB | FAIL (chỉ low256 −.009) |

KM16 lấy lại ≈55% trần oracle sim (+.0103), là generator đầu tiên thắng Core trên utility FL. low256 guard ồn (S1−S0 −.045).
Byte: một lần bootstrap, vẫn ≪ FedAvg 159,5 MB nhưng gấp vài lần tổng Core ≈1,05 MB → cần codec gọn hơn (fp16 s2, chỉ lớp có mặt) nếu giữ.
Theo luật: KM16 → pilot full-scale `scripts/run_km_pilot20.py` (5343 xe, seed42+43, r20/cosine50, so Core F2/F3 r20, gate pilot20 cũ);
`takids1909/mh-k-km16-s{42,43}-p20/1`, cap 7,5 h, probes off (khôi phục RNG nên không đổi quỹ đạo).

## Đợt b4 — ô R1 trộn kiến trúc (đủ r20, `runs/sim-kaggle-b4/r1.json`, `scripts/analyse_r1.py`)

Lần đầu dự án chạy model khác kiến trúc trong một run Core. Mỗi rank một width DAGSNet (`--widths`), xe chia theo rank (100/100, xen kẽ theo
kích thước); Core chỉ allreduce thống kê lớp/Q/CM (không tensor cỡ model), nên trộn không cần đổi giao thức. Hai chiều hoán đổi M1/M2
để mỗi xe chạy cả hai kiến trúc; đồng nhất W1 = b2 S0, đồng nhất W0.25 = H25.

| nhóm | macro trộn M1 / M2 | đồng nhất | Δ M1 / M2 |
|---|---:|---:|---:|
| W1 (395.408 tham số) | .9067 / .9075 | .9077 / .9089 | −.0010 / −.0014 |
| W0.25 (26.864 tham số) | .8829 / .8799 | .8800 / .8794 | +.0029 / +.0006 |

Không nhóm nào sụp; W1 mất ≈.001 (Q gộp từ model yếu hơn), W0.25 không mất. Chênh kiến trúc (.908 vs .880) lớn hơn nhiều chênh do trộn.
Giới hạn: cùng họ DAGSNet; chưa có họ model khác, chưa full-scale, một seed; nhóm 100 xe nên p10 ồn (±.025).

## Đợt b5 — độ bền KM16 + KM16 trong ô R1 (đủ r20, `runs/sim-kaggle-b5/{verdict,r1_km}.json`)

Control b2 S0/S1. KS43 so S1 (cùng seed 43), các arm khác so S0. Thứ tự watcher r20→r16→r18; verdict.json đã chạy lại ở r20.

| arm | thay đổi | Δmacro r20 | Δp10 | Δlow256 | ΔtimeDelay | Δbenign | ΔP lớp1 | luật r20 | Δmacro r16 / r18 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| KS43 (vs S1) | KM16 seed 43 | **+.0058** | +.016 | +.038 | −.0004 | +.003 | +.059 | **PASS** | — |
| KI8 | 8 vòng Lloyd | +.0055 | +.013 | +.012 | +.003 | +.001 | +.067 | PASS | +.0055 / +.0057 |
| **KW05** | co covariance w0=0,5D | **+.0070** | **+.032** | +.006 | +.012 | +.002 | +.064 | **PASS** | +.0070 / +.0067 |
| KR1 | KM16 trong ô trộn W1/W0.25 | xem dưới | | | | | | | |

KS43 macro .9136, ten: accuracy .9232, precision macro/micro/weighted .9255/.9232/.9164, recall macro/micro/weighted .9081/.9232/.9232,
f1 macro/micro/weighted .9136/.9232/.9169. Lợi KM16 lặp lại ở seed 43 (+.0057 seed 42 → +.0058 seed 43). 8 vòng Lloyd không thêm gì
(+0,3 MB byte) → giữ 4. Co yếu hơn (w0 .5D) tốt hơn đều ở r16/r18/r20 → b6 kiểm lặp seed và w0 nhỏ hơn.

KR1 theo nhóm width (cùng chia xe như b4 M1):

| nhóm | KR1 macro | vs Core trộn M1 | vs Core đồng nhất | vs KM16 đồng nhất W1 |
|---|---:|---:|---:|---:|
| W1 (100 xe) | .9124 | +.0057 | +.0047 | −.0014 |
| W0.25 (100 xe) | .8864 | +.0035 | +.0064 | — |

Pooled KR1 .8990 vs Core trộn M1 .8945 (+.0045). KM16 giữ lợi trong ô dị kiến trúc ở **cả hai** nhóm; W0.25 p10 −.011 (nhóm 100 xe ồn ±.025).
(So KR1 với S0 trong verdict.json là so khác kiến trúc → bỏ qua dòng KR1 ở đó.)

Byte: full covariance K=16 là 3,47 MB lên/xe; gửi chỉ lớp xe có chỉ còn 2,80 MB trung bình (xe có trung bình 12,9 lớp) → sparse
không đủ. b6 thử covariance gọn: "within" (covariance trong cụm gộp = cov lớp − phân tán giữa tâm, chỉ từ đếm/tổng) 0,40 MB và
"diag" (phương sai cụm theo vector riêng lớp) 0,48 MB.

## Đợt b6 — w0, K24, covariance gọn (đủ r20, `runs/sim-kaggle-b6/verdict.json`)

| arm | thay đổi | byte lên/xe | Δmacro r20 | Δp10 | Δlow256 | ΔtimeDelay | ΔP lớp1 | luật r20 | Δmacro r16 / r18 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| **KW05S43 (vs S1)** | w0 .5, seed 43 | 3,47 MB | **+.0069** | +.026 | +.025 | +.001 | +.058 | **PASS** | — |
| KW01 | w0 .1 | 3,47 MB | +.0060 | +.029 | +.009 | −.0003 | +.065 | PASS | +.0054 / +.0061 |
| K24W05 | K24, w0 .5 | 5,21 MB | +.0057 | +.022 | −.006 | −.0007 | +.069 | FAIL (low256) | +.0053 / +.0060 |
| KWI | within, w0 .5 | 0,40 MB | +.0001 | +.019 | +.013 | +.001 | +.019 | FAIL | +.0002 / +.0002 |
| KWI0 | within, w0 0 | 0,40 MB | +.0000 | +.015 | −.011 | −.0055 | +.027 | FAIL | +.0004 / +.0002 |
| KDG | diag, w0 .5 | 0,48 MB | +.0012 | +.014 | −.013 | −.0031 | +.050 | FAIL | +.0010 / +.0011 |

KW05S43 ten: accuracy .9234, precision macro/micro/weighted .9256/.9234/.9166, recall macro/micro/weighted .9098/.9234/.9234,
f1 macro/micro/weighted .9147/.9234/.9172.

Đọc:
- **KM16 w0 .5 là cấu hình tốt nhất, lặp lại ở 2 seed** (+.0070 / +.0069). w0 .1 và K24 không hơn → không tune thêm hai chiều này.
- Tâm cụm một mình (within/diag) chỉ lấy lại ≈0–20% lợi: lợi của KM nằm ở **tương quan riêng từng cụm**, không ở vị trí tâm.
  Cùng hướng với O2/O3 (phụ thuộc giữa feature quyết định).
- b7 thử mức giữa: covariance riêng cụm chỉ trong R hướng riêng lớn nhất của lớp (`cop_kmeans_cov=pcaR`, phần còn lại = within).

## Đợt b7 — covariance cụm trong top-R (đủ r20, `runs/sim-kaggle-b7/verdict.json`)

| arm | byte lên/xe | Δmacro r20 | Δp10 | Δlow256 | ΔtimeDelay | Δbenign | luật r20 | Δmacro r16 / r18 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| KP8 | 0,44 MB | +.0007 | +.017 | −.014 | −.005 | −.000 | FAIL | +.0007 / +.0006 |
| KP16 | 0,54 MB | +.0014 | +.031 | +.003 | −.005 | −.002 | FAIL | +.0020 / +.0018 |
| KP32 | 0,94 MB | +.0033 | +.036 | −.004 | +.001 | −.0006 | FAIL (benign sát) | +.0031 / +.0032 |
| KP48 | 1,55 MB | +.0051 | +.028 | −.004 | +.003 | −.000 | FAIL (low256) | +.0047 / +.0047 |
| K8W05 | 1,74 MB | +.0044 | +.035 | +.010 | +.007 | +.001 | PASS | +.0027 / +.0043 |

Đường byte–utility đơn điệu (within 0,40 MB +.000 → pca16 +.0014 → pca32 +.0033 → pca48 +.0051 → full 3,47 MB +.0070): lợi tăng theo
lượng thông tin bậc hai mỗi cụm, không có điểm gãy rẻ. Hướng còn lại: giữ full moment nhưng lượng tử hoá upload mỗi xe (b8).

## Đợt b8 — codec lượng tử hoá theo xe (đủ r20, `runs/sim-kaggle-b8/verdict.json`)

`cop_kmeans_q`: mỗi xe gửi count int32 + mean/covariance tâm hoá từng cụm ở fp16 hoặc int8 + 1 scale fp32/vector-ma trận; server cộng
n·(q(cov)+q(mean)q(mean)ᵀ) rồi chiếu PSD (int8 không PD nếu không chiếu). Mô phỏng đúng từng xe. Xuống vẫn fp16 1,74 MB.

| arm | byte lên/xe | Δmacro r20 | Δp10 | Δlow256 | ΔtimeDelay | Δbenign | luật r20 | Δmacro r16 / r18 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| KQ16 fp16 | 1,74 MB | +.0060 | +.031 | +.008 | +.012 | +.002 | PASS | +.0053 / +.0060 |
| **KQ8 int8** | **0,88 MB** | **+.0067** | +.024 | +.005 | +.012 | +.001 | **PASS** | +.0059 / +.0063 |
| **KQ8S43 int8 (vs S1)** | 0,88 MB | **+.0065** | +.013 | +.008 | +.002 | +.003 | **PASS** | — |
| K8Q8 K8 int8 | 0,44 MB | +.0041 | +.023 | +.017 | +.012 | +.001 | PASS | +.0038 / +.0041 |

KQ8S43 ten: accuracy .9232, precision macro/micro/weighted .9249/.9232/.9164, recall macro/micro/weighted .9093/.9232/.9232,
f1 macro/micro/weighted .9143/.9232/.9172.

Đọc: int8 theo xe giữ ≈95% lợi full fp32 (+.0067/+.0065 vs +.0070/+.0069) với ¼ byte lên → khác với within/diag/pca, lượng tử hoá
không bỏ thông tin bậc hai mà chỉ thêm nhiễu nhỏ, bị co w0 hấp thụ. Thời gian bootstrap local 400 xe: int8 42 s vs fp32 71 s.
**Cấu hình chọn cho full-scale: KM16 w0 .5 int8** → [screening-km50](screening-km50.md), `catbaochau/mh-k-km16q8-s{42,43}-r50/1`.
