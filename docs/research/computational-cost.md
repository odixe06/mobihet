# Chi phí tính toán — MobiHet so với các phương pháp trong `fl_dagsnet_veremi`

Cập nhật 17/10/2026. Owner: so computational cost (Big-O, **tách client và server**) của phương pháp hiện tại với
[`odixe06/fl_dagsnet_veremi`](https://github.com/odixe06/fl_dagsnet_veremi). Dữ liệu hai bên **giống hệt**: VeReMi 80/20,
train 43.045.415 dòng, test 10.761.343 dòng, Dirichlet α 0,5, 20/50/100 client, 50 round × 1 epoch local, batch 512,
DAGSNet cho mọi model (trừ LwFedNIDS cắt tỉa). Phương pháp hiện tại = **DML-m4b + `gpool`** (headline client).

## Ký hiệu và đại lượng đo

| ký hiệu | nghĩa | giá trị |
|---|---|---|
| N, n_k | số dòng train toàn bộ / của client k (Σ n_k = N) | 43.045.415; n̄_k = 2,15M (20c) |
| K | số client | 20 / 50 / 100 |
| P | tham số DAGSNet | 395.024 |
| F | FLOPs một forward DAGSNet / mẫu | **4,44 MFLOPs**; một bước train (fwd+bwd) = 13,28 MFLOPs ≈ 3F |
| B, a | batch; tỷ lệ mẫu Gaussian phụ của `gpool` (256/512) | 512; 0,5 |
| C, D | số lớp; số feature | 16; 66 |

F, P đo bằng [`scripts/count_dagsnet_flops.py`](../../scripts/count_dagsnet_flops.py) (torch `FlopCounterMode`, CPU).

## 1. Big-O mỗi round

Mọi phương pháp đều O(n_k·F) ở client và O(K·P) (hoặc nhỏ hơn) ở server; khác nhau ở **hằng số c = số lượt
"forward-tương-đương" mỗi mẫu** (train một model = 3, forward teacher = 1). Hằng số này quyết định chi phí thật.

| phương pháp | client / round | c | server / round | một lần (setup) |
|---|---|---:|---|---|
| AFPHA (FedAvg + FedProx + cụm 2 cấp) | 1 model + prox O(n_k/B·P) | 3 | O(K·P): trung bình cụm → toàn cục, d_i = ‖w_i − w‖ | — |
| FD-IDS (FedProx + KD từ global) | student + forward teacher (đóng băng) | 4 | O(K·P) | — |
| LwFedNIDS (cắt 70 % kênh) | 1 model nhỏ, F' ≈ 0,09F (ước theo tỷ lệ tham số 9,09 %) | 0,27 | O(K·P'), P' = 35.891 | cắt tỉa O(P) trên server |
| NILM-FL (DML: w_s + proxy w_r) | 2 model | 6 | O(K·P) | — |
| PerFed-SKD (self-KD, chọn client) | 1 model + forward teacher; + đo a_m (bản dựng: trên test 10,76M dòng) | 4 (+eval) | O(K) ngưỡng τ + O(\|S\|·P) | — |
| pFedES (F_k + extractor G 407.874 tham số, 2 pha) | ① G fwd + F_k train trên batch ghép 2×; ② G train + F_k fwd/bwd tới input | ≈12 | O(K·P_G) | — |
| TinyProto (FedProto + CPS/APS) | 1 model + cộng prototype O(n_k·d) | 3–4 | O(K·C·s), s = 50 | mask lớp O(C·d) |
| **MobiHet DML-m4b + `gpool`** | proxy 3F + private 3F·(1+a) trên dòng thật + Gaussian; lấy mẫu O(a·n_k·D²); m4b O(n_k·D) | **7,5** | **O(K·P)** (FedAvg proxy + BN) | client: moment lớp O(n_k·D²); server: cộng O(K·C·D²) + Cholesky O(C·D³) |

- Lấy mẫu Gaussian: a·D² = 2.178 FLOPs/dòng thật = 0,05 % của F. KD (KL hai chiều) O(n_k·C), m4b đặt 0 13 feature
  O(n_k·D): không đáng kể. Moment `gpool` gửi một lần (+2,78 MiB lên, +1,39 MiB xuống cho cả 20 client).
- Server của mọi phương pháp đều không đáng kể: K·P = 7,9M phép cộng/round (20c) ≈ 1,4·10⁻⁸ chi phí client.
  Server MobiHet bằng FedAvg; tăng tuyến tính theo K như các phương pháp trao đổi model.
- Tổng client/round tỷ lệ c·N·F, **không phụ thuộc K** (Σ n_k = N); K lớn chỉ chia nhỏ phần mỗi client.

## 2. Thay số (dữ liệu VeReMi 80/20)

| phương pháp | c | client tổng / round | mỗi client / round (20c) | 50 round |
|---|---:|---:|---:|---:|
| LwFedNIDS | 0,27 | 0,05 PFLOP | 2,6 TFLOP | 2,6 PFLOP |
| AFPHA / FedAvg | 3 | 0,57 PFLOP | 28,7 TFLOP | 28,7 PFLOP |
| TinyProto | 3–4 | 0,57–0,77 PFLOP | 29–38 TFLOP | 29–38 PFLOP |
| FD-IDS, PerFed-SKD (chưa tính eval chọn client) | 4 | 0,77 PFLOP | 38,3 TFLOP | 38,3 PFLOP |
| NILM-FL | 6 | 1,15 PFLOP | 57,4 TFLOP | 57,4 PFLOP |
| **MobiHet DML-m4b + `gpool`** | **7,5** | **1,43 PFLOP** | **71,7 TFLOP** | **71,7 PFLOP** |
| pFedES | ≈12 | ≈2,3 PFLOP | ≈115 TFLOP | ≈115 PFLOP |

Suy luận trên thiết bị (mỗi message): private DAGSNet F = 4,44 MFLOPs + luật gp O(C) — bằng các phương pháp DAGSNet khác;
LwFedNIDS ≈ 0,4 MFLOPs. Bộ nhớ train ở client: 2 model + AdamW ≈ 8P số thực (12,6 MB) + C·D² cho Gaussian, như NILM-FL;
FedAvg 4P.

## 3. Đo thực tế (train, không gồm eval; GPU-phút T4 mỗi round, quy về 43,05M dòng)

Repo kia chạy một run trên 2×T4 (GPU-phút = 2 × wall). MobiHet chạy một arm trên một T4; FullDrift train trên 40,0M dòng
(đã bỏ holdout) nên nhân 1,076.

| phương pháp | GPU-phút / round | so FedAvg cùng harness | nguồn |
|---|---:|---:|---|
| LwFedNIDS | 8,3 | 0,76× AFPHA | `history.csv` 20c, train_sec trung vị |
| AFPHA | 10,9 | 1 | `history.csv` 20c, t_train_s trung vị |
| PerFed-SKD | 11,6 (+8,2 eval chọn client) | 1,06× | `history.csv` 20c |
| TinyProto | 13,6 | 1,25× | đo 3 round 20c (`tests.md`) |
| FD-IDS | 8,3 + 7,7 teacher = 16,0 | 1,47× | round 520 s − teacher 230 s − eval ~40 s |
| NILM-FL | 18,9 | 1,74× | report §train/round 568 s |
| pFedES | 34,9 | 3,2× | report §train/round 1.046 s |
| MobiHet FedAvg (`fd_fedavg`) | 17,4 | 1 | FullDrift s42 |
| MobiHet DML (`h_dml`, full 43M) | 32,3 | 1,86× | full-validation, 50 round |
| MobiHet DML-m4b | 32,5–32,8 | 1,87× | FullDrift s42 (vòng 3, 10) |
| **MobiHet DML-m4b + `gpool`** | **36,6–39,2** | **2,1–2,25×** | FullDrift s42/s43 (vòng 9–10) |

Đọc:
- **Thuật toán:** MobiHet tốn ≈2,2× FedAvg, hơn NILM-FL (cùng cấu trúc DML) khoảng 1,25× do `gpool` (+0,5 lượt train private),
  thấp hơn pFedES (≈3,2×). FD-IDS/PerFed-SKD/TinyProto rẻ hơn ~1,5–2×; LwFedNIDS rẻ nhất.
- **Hiện thực:** cùng thuật toán, harness MobiHet chậm hơn repo kia **1,6–1,8×** (FedAvg 17,4 so AFPHA 10,9 dù AFPHA thêm
  FedProx; DML 32,3 so NILM-FL 18,9). Đây là chi phí kỹ thuật, không phải của phương pháp. Nếu đạt hiệu suất của NILM-FL,
  DML-m4b + `gpool` ước ≈21–23 GPU-phút/round (≈1,2× NILM-FL, khớp tỷ lệ c 7,5/6).
- **FLOPs không phải wall-time:** T4 chạy DAGSNet ở ~0,65 TFLOPS hiệu dụng (MobiHet) – ~1,0 TFLOPS (NILM-FL), ≈1–1,5 %
  đỉnh fp16 — bị giới hạn bởi launch/bộ nhớ, không bởi FLOPs. LwFedNIDS ít hơn 11× FLOPs nhưng mỗi bước train compiled chỉ nhanh 1,16× (5,85 so 6,78 ms).
  Vì vậy **số lượt model mỗi mẫu (c)** dự báo wall-time tốt hơn tổng FLOPs.
- **Eval theo protocol** (không phải chi phí phương pháp): model cá nhân hóa phải chấm N+1 model trên test mỗi round
  (MobiHet `h_dml` ≈22 GPU-phút, NILM-FL ≈9,8, pFedES ≈9,1), model toàn cục chỉ một.

## 4. Truyền tin mỗi client mỗi round (để đủ bức tranh)

| phương pháp | lên + xuống |
|---|---|
| MobiHet, NILM-FL, AFPHA, FD-IDS | 2 × ≈1,59 MB (proxy/model fp32 + BN); MobiHet q4: 2 × 0,21 MB (bộ đếm codec) |
| pFedES | 2 × 1,63 MB (G) |
| PerFed-SKD | 2 × 1,58 MB × \|S\|/K (0,37 ở 20c) |
| LwFedNIDS | 2 × 0,14 MB |
| TinyProto | O(C·s) số thực: ~32.000 tham số tổng cho 20 client mỗi round |

## Giới hạn

- Hằng số c đếm lượt forward/backward; bỏ qua optimizer O(P) mỗi bước (P/B ≈ 772 FLOPs/mẫu ≪ F) và tải dữ liệu.
- F' của LwFedNIDS ước theo tỷ lệ tham số (cắt kênh hai phía của lớp ⇒ ≈0,3²); pFedES coi FLOPs của G ≈ F.
- Số đo hai repo khác harness/nền chạy (2 arm song song trên một máy 4 vCPU ở MobiHet); TinyProto từ 3 round đo.
- `gpool` mới có ep6 (holdout nội bộ); chưa có run r50 của DML-m4b + `gpool`.

## 5. Tối ưu mã chạy (17/10 chiều, bit-exact — không đổi phương pháp)

Owner 17/10: computational cost không phải mục tiêu, nhưng code nhanh hơn mà không đổi phương pháp thì làm (Kaggle 2×T4).
Profile `private_epoch` (`gpool`, RTX 3050 local): ~674 kernel lẻ/bước (≈500 `fill_` do deterministic mode lấp NaN vào
bộ nhớ chưa khởi tạo) và ~6 lần đồng bộ host/bước (`assert isfinite(loss)`, `isfinite(norm)`, `get_scale()` ×2, mask
index-put, kiểm `draw`). Đã gỡ (kiểm hoãn tới cuối epoch, đếm skip AMP trên GPU, `masked_fill`, `fill_uninitialized_memory
= False`): **34,0 → 26,9 ms/bước (−21 %)**, state/packet/log bit-exact. T4 production đo 26–27 ms/bước nhưng compute GPU chỉ
~12 ms ⇒ CPU là nút thắt; thêm `--workers W` (client chia cho W tiến trình, GPU xoay vòng, bit-exact — test `par`).
Benchmark T4 `accanniez/mobihet-fd-bench-t4-20261017-v1` (COMPLETE 17:13Z; 2048 bước gpool client 17, không train thật):

| cấu hình | ms/bước/tiến trình | tổng bước/s (2×T4) |
|---|---:|---:|
| v0 (harness r11), 1 tiến trình | 25,4 | 39,4 |
| v1 (sync-free), 1 tiến trình | 21,7 | 46,1 |
| v0, 1 tiến trình/GPU (×2) | 25,4 | 78,8 |
| **v1, 1 tiến trình/GPU (×2)** | 22,0 | **90,9** (+15 %) |
| v1, 2 tiến trình/GPU (×4) | 42,8 | 93,4 (+3 %) |
| v1, 3 tiến trình/GPU (×6) | 63,2 | 95,1 (+5 %) |

⇒ Chọn `--workers 2` (một tiến trình/GPU): thêm tiến trình chỉ +3–5 % vì 4 vCPU là trần; vòng 12–13 dùng W = 2.
Một arm gpool ep6 trên cả hai GPU ≈ nửa thời gian một arm/GPU cũ (vòng 13 đo ~46 s/2048 bước mỗi worker).

## 6. Tìm cách viết code nhanh nhất (18/10, skill `federated-training-speed`)

Harness r18, bước gkd thật (2 DAGSNet train + 2 forward teacher), T4 production 26,3 ms/bước/tiến trình. `torch.cuda.set_sync_debug_mode`: trong bước
chỉ còn 1 sync (`GradScaler.step` đọc `found_inf`). Benchmark local RTX 3050 (1 client fixture 33.858 dòng, 67 bước, sau 1 epoch warm-up):

| biến thể | ms/bước | lớp |
|---|---:|---|
| base (deterministic, AdamW foreach) | 30,4 | bit-exact (hiện tại) |
| `fused` AdamW (bỏ sync scaler, 1 kernel optimizer) | 24,9 (−18 %) | tolerance |
| `nondet` (cudnn.benchmark, bỏ deterministic) | 27,2 (−11 %) | tolerance |
| `fused` + `nondet` | **22,9 (−25 %)** | tolerance |

Chưa đo trên T4; chưa thử capture cả bước. Lớp tolerance cần owner duyệt và chạy lại đối chứng trong harness mới.

**Owner duyệt 18/10 ~09:15Z** cho kernel build từ vòng 19: cờ `fd.py --fast` (fused + nondet); kernel cũ giữ bit-exact và đối chứng cũ, arm `--fast` chỉ so đối chứng `--fast` ([design §Code nhanh](../methods/full-drift/design.md)). Test `test_fd_local --part fast` PASS: không cờ = bit-exact r18 (5 arm, 2 epoch); 1 bước từ cùng trạng thái, logit private lệch fused 2,8–3,2e-5, fused+nondet 4,1–6,0e-3 của max|logit|, 0 argmax đổi trên dòng phân định rõ, cùng số bước applied/skipped; sai khác tương đối theo tensor tới 0,5–0,7 chỉ ở bias BN gradient ≈ 0 (Adam bước đầu ±lr theo dấu nhiễu) — không phải tiêu chí. ms/bước T4 đo từ run đối chứng `--fast` đầu tiên (`ms_per_step.py`).

**Đo T4 production (18/10 ~18:30Z, `ms_per_step.py`, cùng arm la_kdpx s42 ep1→15, `--workers 2`):** code cũ (rc) 25,19 ms/bước → `--fast` (tb) **18,75 ms/bước (−25,6 %)**, cùng 1.172.460 bước; khớp benchmark local −25 %. Private head 0,8225 → 0,8269 (+0,004, trong chênh seed). gkd s43: 24,54 → 19,34 ms/bước (−21 %), head 0,7953 → 0,7984 (cổng equiv PASS).

## 7. Run cuối train/chấm tách (20/10)

Owner ưu tiên seed42; session 20c đầu 4 round, 50/100c 2 round. Mỗi round chấm bằng một notebook hai GPU trên account khác, đồng thời train continuation sau khi evaluator có heartbeat. [Pipeline và ước lượng](../operations/final-pipeline.md).

Đếm lại local bằng FlopCounterMode: forward **4443648 FLOPs/mẫu**, forward+backward **13280256**, 395024 tham số. `round.cost` ghi operator estimate riêng với walltime, bước SGD và VRAM; ước lượng không bao gồm BN/KL/optimizer/calibration/probes, không gọi là toàn bộ chi phí runtime. `runtime.json` ghi setup/compile/hardware. Communication tính byte gói serialize, bootstrap và up/down riêng, host time encode/decode; byte/algorithm-second là tốc độ tạo payload, không phải bandwidth link vật lý.
