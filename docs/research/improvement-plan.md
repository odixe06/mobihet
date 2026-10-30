# MobiHet v2 — phân tích và kế hoạch cải tiến (30/10/2026)

Owner 30/10: (1) train proxy như một model thứ hai tốn tài nguyên/thời gian; (2) synthetic chưa thực sự hiệu quả cho việc học;
(3) cải thiện distill proxy ↔ private; (4) cố gắng **bỏ việc train proxy**. Protocol: [protocol](protocol.md) (5.343 xe, 50 round, W = 1).

## 1. Hiện trạng đo được

| | giá trị | nguồn |
|---|---|---|
| Arm hiện hành | `gkd + la + kdpx`, frz teacher, gpool, proxy q3_zl ([methods/mobihet.py](../../methods/mobihet.py)) | v1 vòng 17, final 20/50/100 client |
| Mỗi bước local | private fwd+bwd trên b + b/2 dòng (real + Gaussian), proxy fwd+bwd trên b, teacher fwd trên 1,5 b | code |
| FLOPs client / dòng / round | **8,97 × forward** (FedAvg 2,99) ⇒ 3,0× FedAvg | [cost](cost.md) |
| Thời gian Kaggle v4 (W = 1) | r1 1.335 s + eval 398 s; r2 > 36 phút (FedAvg 224 s); GPU 100 %, bộ nhớ 96,8 % | run `fl2-mobihet` (repo baseline) |
| Thời gian local compile (100 xe) | r2 52 s, r3 44 s ⇒ chậm ở Kaggle do bộ nhớ đầy (allocator), không phải FLOPs | profile 30/10 |
| Byte / xe / round | ~117 KB lên, ~114 KB xuống (dense 1,6 MB) | r1 v4 |
| F1 r1 v4 | private 0,801 (pooled own-test), proxy 0,441 | W&B |
| Baseline v4 cùng protocol (đang chạy) | FedAvg r18 0,887; FedProto client r11 0,863 | repo baseline |

Tách công trạng thái v1 (đều đã audit, ep6/ep15, 20 client; [results v1](../../history/v1/docs/methods/full-drift/results.md)):

| thành phần | Δ private | đọc |
|---|---|---|
| frz teacher (proxy snapshot đầu round, đóng băng) | +0,037 | KD từ model **gộp** là kênh tri thức chính |
| gkd (KD teacher trên dòng Gaussian) | **+0,031** trên frz | synthetic **làm tập chuyển giao KD** thì hiệu quả |
| gpool (CE trên dòng Gaussian) | +0,016 ep6, **FAIL horizon** ep15 | synthetic **làm dữ liệu học nhãn** thì yếu |
| la (logit adjustment về prior toàn cục) | +0,016 | sửa lệch prior local |
| kdpx (proxy ← private, cap 1) | +0,006 private, +0,011 proxy | proxy học từ private là cần |
| `own` (proxy không gộp) | −0,043 | lợi FL thật |
| NTD, hidden-KD, FedDyn, SCAFFOLD, kdcap, prox, avgm, EMA | FAIL | không mở lại |
| synthetic P23–P37 (copula, GMM, MADE, CCVR, atom, cvp/cop…) | FAIL; trần oracle dòng thật **+0,010** | **không mở lại** (v1 LESSONS) |

**Kết luận phân tích.** Proxy hiện làm hai việc: (a) *vật mang* tri thức gộp xuống xe (teacher của frz + gkd — phần lợi lớn),
(b) *bộ học* trên dữ liệu thật của xe để tri thức đi lên server (DML + kdpx). (b) chiếm ~1/3 FLOPs và một nửa bộ nhớ nhóm
(2 model / bước), trong khi tri thức nó mang lên đã có sẵn trong private (proxy chỉ học CE + KL từ private). Có thể chuyển (b)
thành **truyền dự đoán của private trên một tập chuyển giao chung** và để **server** chưng cất proxy — proxy không còn được
train trên xe. Synthetic không bị bỏ: v1 cho thấy nó tốt ở vai trò tập chuyển giao KD (gkd), kém ở vai trò dữ liệu nhãn.

## 2. Hướng thiết kế

**A. Proxy chưng cất ở server (khuyến nghị chính; FedDF — Lin et al., NeurIPS 2020; FedMD — Li & Wang 2019).**
Tập chuyển giao chung X_s (M dòng) sinh **bằng seed** từ moment Gaussian gộp (bootstrap gpool có sẵn) ⇒ không truyền X_s.
Xe: train **chỉ private** (CE + la, KD frz trên real, gkd trên Gaussian — giữ các thành phần PASS); cuối epoch forward private
trên X_s, upload xác suất lượng tử (M × 16, int8 ⇒ M = 4.096: 64 KB; hoặc top-k). Server: gộp ensemble theo lớp
(w_{k,c} ∝ n_{k,c}: chỉ tin xe trên lớp nó thật sự thấy), chưng cất proxy S bước trên X_s, gửi Q3 delta như hiện tại.
Kỳ vọng: FLOPs client ~6 × fwd (−33 %), một model/bước ⇒ nhóm nhẹ một nửa bộ nhớ (gỡ nút Kaggle); byte lên giảm hoặc giữ.
Rủi ro: proxy chỉ thấy synthetic ⇒ teacher yếu hơn. Đo trực tiếp bằng F1 proxy và lợi frz/gkd.

**B. Proxy train rẻ ở xe (dự phòng nếu A làm teacher yếu).** Proxy không học trên dòng thật; cuối epoch chỉ KD private → proxy
trên M dòng Gaussian (vài trăm bước), upload Q3 delta như cũ. FLOPs proxy ∝ M thay vì n_k.

**C. Không proxy (cận dưới băng thông; FedDistill — Jeong et al. 2018; FedProto — Tan et al. 2022).** Xe upload logit trung bình
theo lớp (16 × 16) của private trên dữ liệu thật; KD về logit gộp. Rẻ nhất; FedProto v4 đã 0,863 nên là đối chứng bắt buộc.

**D. Tập chuyển giao tốt hơn (sau khi A hoặc B qua cổng; mỗi lần một yếu tố).**
D1 mixup dòng thật × Gaussian cùng lớp cho KD ở xe (FedMix — Yoon et al., ICLR 2021) — mẫu gần manifold thật, không rời xe.
D2 tinh chỉnh X_s ở server bằng đảo ngược BN (DeepInversion — Yin et al., CVPR 2020): vài bước gradient trên X_s khớp running
stats BN của proxy + CE về lớp đích; dữ liệu không rời xe, chỉ dùng BN của model đã gộp.
D3 khai thác mẫu khó ở server (FedFTG — Zhang et al., CVPR 2022): chọn/sinh X_s nơi ensemble và proxy bất đồng nhiều nhất.
Không mở lại copula/GMM/MADE/CCVR/cvp/cop (v1 FAIL).

**E. Distill tốt hơn (trên arm thắng; mỗi lần một yếu tố).** E1 logit standardization KD (Sun et al., CVPR 2024): chuẩn hóa
z-score logit thầy/trò trước softmax — gỡ chênh thang logit. E2 DKD (Zhao et al., CVPR 2022): tách KD lớp đúng / lớp không đúng,
trọng số riêng (NTD thuần đã FAIL ở v1 ⇒ chỉ dạng tách trọng số). E3 trọng số KD theo độ tin của thầy trên từng lớp (n_{·,c}
gộp) thay cho w = 1/(CE_p + CE_r) — xe ít/không có lớp c thì tin thầy nhiều hơn.

## 3. Thí nghiệm (đề xuất; cổng khóa trước khi chạy)

Sàng lọc: full 5.343 xe, **10 round** (lịch LR của 50 round, chỉ chạy r1–r10), seed 42, Kaggle 2×T4; headline sàng lọc r10.
Primary = **private F1-macro pooled trên test của chính xe**; kèm trung bình theo xe, p10, benign, F1 proxy, FLOPs/dòng
(`scripts/cost.py`), byte/xe/round, giây GPU/round. Một yếu tố mỗi arm so với đối chứng cùng seed.

| # | arm | so với | cổng |
|---|---|---|---|
| S0 | `mobihet` hiện hành, sửa hạ tầng (`expandable_segments`, `group_rows` 4.096) — thuật toán giữ nguyên | — | đối chứng; đo tốc độ |
| S1 | A, w_{k,c} = n_{k,c} | S0 | **noninferior** private (Δ ≥ −0,005, p10 ≥ −0,010) **và** FLOPs/dòng ≤ 0,7 × S0 |
| S2 | A, w_k = n_k | S1 | mô tả (cô lập trọng số theo lớp) |
| S3 | B | S0 | như S1 |
| S4 | C | S0 | mô tả; cận dưới băng thông |
| S5 | thắng(S1, S3) + D1 | thắng | gain Δ ≥ +0,010, p10 không giảm > 0,010 |
| S6 | thắng + D2 | thắng | như S5 |
| S7 | thắng + E1 / E2 / E3 (3 arm) | thắng | như S5 |
| F | arm cuối, 50 round | FedAvg, FedProto, mọi baseline repo v2 (cùng protocol) | headline r50 |

Không chọn HP/round theo test (protocol không có val riêng ⇒ M, S, nhiệt độ cố định trước: M = 4.096, S = 200 bước Adam 1e-3,
batch 512, T = 1). Lặp seed 43 cho arm cuối nếu quota cho phép.

## 4. Owner duyệt (30/10)

- Sàng lọc 10 round full-K (cổng §3 khóa trước khi chạy). Thử **mọi** hướng (A, B, C, rồi D/E) để tìm phương án tối ưu.
- Code: `methods/mobihet.py` `arm ∈ {dml, srv, cheap, none}` (+ `agg_soft`, `m_transfer` 4.096, `server_steps` 200).
- 30/10 (sau S0–S4): nền = **S4 `none`** (không proxy); chạy cả 5 bộ sinh G0–G4 song song. Không proxy vẫn giữ được model khác nhau
  mỗi xe (chỉ trao đổi xác suất 16 × 16 + thống kê input, không tham số) — owner chỉ xét về phương pháp, **không cần chạy kiểm chứng**;
  nếu cần: arm trộn độ rộng W theo xe (k mod 3 → W 1 / 0,5 / 0,25), báo F1 theo nhóm.

## 5. Kết quả tới 30/10 và đề xuất tiếp (chưa chạy; chờ owner duyệt)

Kết quả: [screening-r10](screening-r10.md). Tóm tắt r10 (private F1 pooled / p10): S0 `dml` 0,8584 (r9) · S1 `srv` 0,8514 FAIL ·
S3 `cheap` 0,7778 FAIL · **S4 `none` 0,8679 / 0,567** · G0 off 0,8497 · G1 oracle 0,8802 (trần) · G2 short 0,8669 · G3 mix 0,8667 /
0,580 · **G4 copula 0,8732 / 0,574** (+0,0053, FAIL cổng +0,010, lấy ~43 % khoảng trần). FedAvg r10 0,8749, FedProto r10 0,8629.

Đọc: bỏ proxy là hướng đúng (rẻ nhất, tốt nhất, 544 B/round); synthetic vẫn cần (G0 −0,018, p10 −0,078); chất lượng input còn
+0,012 để lấy; copula đi đúng hướng. Rủi ro lớn nhất: mọi Δ ≤ 0,006 là **một seed** và 10 round (v1: lợi ep6 từng biến mất ở ep15).

Đề xuất (nền = G4 copula trên S4; một yếu tố/arm; 10 round full K; cổng như §3: Δ ≥ +0,010 vs nền, p10 không giảm > 0,010):

| # | arm | đổi gì | lý do / câu hỏi |
|---|---|---|---|
| H0 | S4 và G4, **seed 43** | lặp seed | đo nhiễu giữa seed; G4 − S4 cùng dấu ở 2 seed mới tin +0,005. Ưu tiên cao nhất |
| H1 | copula hỗn hợp theo (lớp × cụm scenario·run, 8 cụm như AFPHA) | biên + tương quan theo cụm, chọn cụm ∝ n_{c,cụm} gộp | input phủ đúng các "chế độ" dữ liệu của xe khác (oracle lấy ngẫu nhiên dòng toàn cục) |
| H2 | copula + mix (G3 trên dòng copula) | mixup dòng copula × dòng thật cùng lớp | G3 tăng p10/mean/xe, G4 tăng pooled ⇒ thử cộng |
| H3 | copula, liều ×2 (64 dòng/bước) | số dòng synthetic | v1 `gw2` −0,005 với proxy; với teacher Q[y] chưa biết |
| H4 | copula, đích CE one-hot thay Q[y] trên dòng synthetic | loại đích | Q[y] chỉ phụ thuộc nhãn ⇒ gần label smoothing; đo đích nào tốt hơn |
| F | arm thắng (hoặc G4 nếu không arm nào qua), **50 round**, + S4 50 round | headline | so FedAvg 0,9042 / FedProto 0,8698 / APFed 0,9119 ([results baseline](https://github.com/odixe06/fl_dagsnet_veremi_v2/blob/main/docs/research/results.md)) |

Sau đó (từ §2, áp lên arm thắng): E1 logit standardization, E3 trọng số KD theo độ tin lớp; D2 (BN inversion) không áp được khi
không có model chung. Chi phí: H0–H4 ~1,6 h/run (6 run song song đủ quota); F ~8 h/run (529–546 s/round × 50 + eval).
