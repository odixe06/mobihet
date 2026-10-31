# P37 — synthetic làm trọng số private/proxy đổi thế nào (full model, không frozen-head)

23/10/2026 ~03:30Z; owner: "phân tích kỹ sự ảnh hưởng của synthetic data đến sự thay đổi các trọng số của private cũng như proxy",
tra tài liệu, thử thêm một lần cuối trước khi đóng hướng; "kiểm tra thật kỹ các bản build trước". **Protocol, arm và luật đọc khóa trước khi chạy.**

## Bằng chứng → câu hỏi

- Vòng 26: harness frozen-head (P32–36) PASS hai seed nhưng online Δ +0,0003. Harness đóng băng backbone **và 31 lớp BatchNorm** của DAGSNet,
  chỉ train head. Online, mỗi batch private = 512 dòng thật + 256 dòng synthetic (⅓ batch) đi chung BN train-mode ⇒ synthetic đổi (a) thống kê
  BN mà dòng thật được chuẩn hóa theo, (b) backbone qua gradient KL, (c) head. Proxy **không** thấy synthetic (`paux` chỉ private); nó chỉ chịu
  ảnh hưởng gián tiếp qua KL proxy←private trên dòng thật (`kdpx`). Chưa phép đo nào tách ba kênh này.
- Đã có (không lặp): BN tính lại **hậu kỳ** trên dòng thật −0,03/−0,04 (vòng 13; BN trộn là một phần hàm đã học) — P37 đo can thiệp **lúc train**,
  không recalibrate sau. Tăng liều synthetic (gw2 −0,005, g512, gk2 vòng 18) không lợi. P1 FedGen adaptation (pilot Gaussian, extractor local)
  ≈ LG — khác bối cảnh teacher cố định hiện tại. Danh sách đóng: [audit build cũ §8](synthetic-build-history-audit-2026-10-22.md).

## Thiết lập

- Code production: `fd.Run`/`fd.Method`/`Run.private_epoch` của source r31 (eager, deterministic), arm của xa
  `fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl`, `--fast` (fused AdamW), aux_batch 256, batch 512. Chỉ monkeypatch: đường fixture + bỏ cổng manifest
  full (fixture pilot 2,5 %, cap 32.768 dòng/client, group/old 65.536), và các can thiệp dưới. Bảng Gaussian gộp dựng bằng `make_tables`
  production **trên fixture** (ước lượng mẫu của moment full).
- Trạng thái đầu: weights đã audit **xa ep6** (chính) và **xa ep15** (mô tả): private k, proxy = teacher = snapshot gộp; `version` > 0.
  Mô phỏng round kế (r = 7 / 16, LR cosine T50 đúng lịch). Mỗi arm: **4 lần `private_epoch` liên tiếp** (4 × 64 = 256 bước/client; Adam reset
  mỗi lần như production), cùng seed thứ tự dòng ⇒ dòng thật giống hệt giữa arm. Không aggregate q3; proxy gộp mô tả = trung bình fp32 20 client.
- Không official test. Chấm private (luật head) và proxy gộp trên group/old fixture, 10 metric + per-class từ confusion.

| arm | synthetic rows | BN của dòng thật | gradient synthetic tới | vai trò |
|---|---|---|---|---|
| `none` | không | thật | — | không synthetic (≈ frz) |
| `gauss` | Gaussian gộp, KL snapshot | trộn ⅓ synthetic | mọi lớp | **production** |
| `gauss_bn` | Gaussian, trọng số loss 0 | trộn | — | chỉ kênh BN |
| `gauss_sep` | Gaussian, forward riêng, BN eval (running) | thật | mọi lớp | tách BN |
| `gauss_head` | Gaussian, đặc trưng no-grad BN eval | thật | chỉ head | head-only |
| `cop` | copula (vòng 18) thay Gaussian | trộn | mọi lớp | support vòng 26 |
| `real` | dòng thật client khác (oracle R3-ineligible) | trộn | mọi lớp | trần support |
| `real_sep` | như real, forward riêng BN eval | thật | mọi lớp | oracle tách BN |
| `real_head` | như real, chỉ head | thật | chỉ head | oracle head-only |
| `gauss_b` | Gaussian, luồng rút khác | trộn | mọi lớp | **sàn nhiễu** (thêm sau smoke) |

*Sửa trước khi chạy đủ (~03:35Z), sau smoke 1 client × 1 pass:* hai run chỉ khác luồng rút đã lệch trọng số cỡ cả cập nhật epoch, nên mọi
cosine/hoán đổi head phải đọc so `gauss|gauss_b` (sàn nhiễu); gradient đo với dropout tắt (BN vẫn train-mode) để chênh mixed vs real-only chỉ
do BN. Không đổi arm/cổng/luật đọc khác.

Real oracle: mỗi nhãn của chu kỳ lớp lấy ngẫu nhiên một dòng fixture cùng lớp từ 19 client còn lại (chỉ chẩn đoán, không thành kênh FL).

## Đo

1. **Utility sau 4 lần epoch** (mean 20 client, benign, client tệ nhất, per-class) của private; proxy gộp.
2. **Phân rã đổi trọng số** private và proxy: D = θ_arm − θ_none theo nhóm (stems, dense, google, alex, squeeze, BN affine, BN running stats,
   head LN, head.2, head.5): ‖D‖ / ‖θ_none − θ_đầu‖, cosine(D_gauss, D_real), cosine(D_cop, D_real), cosine(D_gauss_head, D_real_head).
3. **Hoán đổi head:** backbone arm X + head `none` và ngược lại ⇒ phần lợi/hại nằm ở backbone(+BN) hay head.
4. **Gradient tại trạng thái đầu** (fp32, mixed-BN như production, 8 batch/client): norm từng số hạng theo nhóm (CE own có la, KL teacher own,
   KL synthetic, KL real-oracle — đều nhân trọng số `1/(CE+CE_proxy)` thật); cosine g_syn với g_ce, g_kd, g_real; và gradient CE own tính trong
   batch trộn vs batch chỉ thật (kênh BN trên gradient dòng thật).
5. Đặc trưng: đổi tương đối c544 của private trên 4.096 dòng group (arm vs none); lệch running stats BN so thống kê batch thật own.

## Luật đọc (khóa) ⇒ lần thử cuối

Δ so `gauss` (production) trên **xa ep6**, mean private group; ep15 phải cùng dấu (≥ 0) mới nhận:
- `gauss_sep` hoặc `gauss_head` ≥ **+0,005**, benign ≥ −0,005, worst ≥ −0,010 ⇒ ứng viên online (0 byte, R2 giữ): chọn arm Δ lớn hơn;
  hòa trong 0,002 ⇒ chọn `gauss_sep` (đổi ít hơn). Online: xa recipe + token mới, s42 ep15 `--fast` vs xa, cổng như vòng 26
  (Δ ≥ +0,010, benign ≥ −0,005, worst ≥ −0,010) — cần owner duyệt push.
- `gauss_bn` − `none` ≤ −0,005 ⇒ ghi BN trộn là kênh hại (mô tả), kể cả khi không ứng viên nào qua.
- Nếu `real_head` giữ ≥ 70 % lợi `real − none` ⇒ lợi support ở tầng quyết định (khớp P33–36); < 30 % ⇒ lợi ở backbone/BN, head-only không đủ.
- Không ứng viên nào qua ⇒ báo cơ chế đo được, **không** tự mở biến thể khác; owner quyết đóng hướng.
- Mọi số là fixture 256 bước, không thay online ep15. Không đổi số bước/arm/seed sau khi thấy kết quả.

## Kiểm tra trước khi tin

- Arm `gauss` = gọi thẳng `Run.private_epoch` không patch; arm `none`/`gauss_bn` chỉ đổi cờ (`syn_teacher=None`+`paux=False`; `gk2` với GK2=0).
- `gauss_bn` trước bước đầu: gradient synthetic = 0 chính xác; `gauss_sep`/`gauss_head` với batch không synthetic cho logits dòng thật = forward thường.
- Predictions → confusion → 10 metric tính lại khớp; mọi weights hữu hạn; source/fixture/weights SHA trong JSON.

## Tài liệu tham khảo — đối chiếu cơ chế (tra 23/10, chưa phải bằng chứng trên dữ liệu này)

| công trình | cơ chế | liên hệ với P37 / build cũ |
|---|---|---|
| [AdvProp, Xie et al. CVPR 2020](https://arxiv.org/abs/1911.09665); [Sandwich BN](https://arxiv.org/pdf/2102.11382); DSBN | dữ liệu phụ lệch phân phối dùng BN phụ riêng; BN chung trộn thống kê hai phân phối | `gauss_sep` (BN eval cho synthetic) là dạng 0 tham số của ý này; khác BN tính lại hậu kỳ đã FAIL vòng 13 |
| [FedGen, Zhu et al. ICML 2021](https://proceedings.mlr.press/v139/zhu21b.html) | generator ở không gian latent, mẫu chỉ điều chỉnh classifier | `gauss_head`; pilot P1 FedGen adaptation ≈ LG (extractor local, không teacher cố định) — bối cảnh khác |
| [CCVR, Luo et al. NeurIPS 2021](https://neurips.cc/virtual/2021/poster/27280) | hiệu chỉnh classifier bằng đặc trưng ảo Gaussian theo lớp | đã dùng cho teacher (vòng 25/26 FAIL); chưa dùng cho head private |
| [VHL, Tang et al. ICML 2022](https://proceedings.mlr.press/v162/tang22d.html) | dữ liệu ảo sinh từ nhiễu (StyleGAN chưa train), CE ảo + supervised-contrastive kéo đặc trưng thật về đặc trưng ảo cùng lớp (ảo detach); ảnh thật (Tiny-ImageNet) không thắng nhiễu | vai trò synthetic là **neo** phân bố đặc trưng, không cần giống thật — khớp P32–36: tăng độ thật không chuyển lợi online |
| [DeSA, Huang et al. ICML 2024](https://proceedings.mlr.press/v235/huang24v.html) ([code](https://github.com/ubc-tea/DESA)) | FL phi tập trung, model khác kiến trúc: CE + REG (căn embedding với anchor tổng hợp) + KD trên anchor | `gkd` ≈ phần KD; chưa có REG. R2 giữ được vì mỗi model tự căn đặc trưng của mình |
| [FedFTG, Zhang et al. CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Fine-Tuning_Global_Model_via_Data-Free_Knowledge_Distillation_for_Non-IID_Federated_CVPR_2022_paper.pdf) | hard sample nơi local/global bất đồng, fine-tune global ở server | gần `xo` vòng 24 (FAIL) |
| [FedLGD](https://arxiv.org/html/2303.02278v3), [FedDM](https://arxiv.org/pdf/2207.09653) | dataset distillation (distribution/gradient matching); FedLGD ghi dữ liệu ảo local có thể **làm tăng** lệch đặc trưng giữa client | cùng họ KME/DM/V5 đã FAIL; cảnh báo ủng hộ đo backbone/BN |
| [FedFed, Yang et al. NeurIPS 2023](https://arxiv.org/pdf/2310.05077) | chia sẻ đặc trưng "nhạy hiệu năng" + DP | gửi đặc trưng từng record ⇒ vi phạm R3, không dùng |

Smoke 1 client × 1 pass (~03:30Z, chỉ để kiểm code, không phải kết quả): trộn ⅓ synthetic vào BN đổi gradient CE dòng thật ~60 % độ lớn
(cos 0,83 so batch chỉ thật); gradient synthetic là số hạng lớn nhất của loss private; cos(D_gauss, D_real) ~0,45 dưới sàn nhiễu ~0,6–0,8.

## Kết quả — Kaggle `minhtran0601/mobihet-fd-probe37-20261023-v1` (23/10, 2×T4, ~82 phút/checkpoint)

Chạy local dừng sau 2 client (owner: việc nhiều giờ lên Kaggle). [ep6 JSON](../../methods/full-drift/results/probe37-weight-dynamics-xa-e06.json),
[ep15 JSON](../../methods/full-drift/results/probe37-weight-dynamics-xa-e15.json), [generator](../../../methods/full-drift/builds/fd-probe/scripts/gen_probe37.py).
**Audit:** SHA output khớp `COMPLETE.json`; `fd.py`/`fd_probe37.py` chạy = local; weights SHA = marker xa đã audit; 840 confusion → 10 metric
tính lại khớp. Sàn nhiễu `gauss_b` − `gauss` +0,0001 / +0,0004 mean ⇒ chênh dưới ~0,001 là nhiễu.

| arm | ep6: F1 group / Δ vs gauss (benign, worst) | ep15: F1 group / Δ vs gauss (benign, worst) | client hơn gauss (ep6/ep15) |
|---|---|---|---|
| `none` | 0,7969 / −0,0037 (−0,002, −0,004) | 0,8195 / −0,0012 (+0,002, −0,003) | 5 / 8 |
| **`gauss`** (production) | 0,8006 | 0,8207 | — |
| `gauss_b` (sàn nhiễu) | +0,0001 | +0,0004 | 7 / 10 |
| `gauss_bn` (chỉ BN) | −0,0025 | −0,0017 | 4 / 6 |
| `gauss_sep` (tách BN) | −0,0023 (+0,000, +0,000) | −0,0009 (+0,001, −0,003) | 6 / 8 |
| `gauss_head` (head-only) | −0,0025 (−0,002, −0,001) | −0,0010 (+0,001, −0,002) | 6 / 7 |
| `cop` | +0,0021 (+0,001, +0,004) | +0,0024 (+0,002, +0,002) | 14 / 16 |
| `real` (oracle) | +0,0034 (+0,008, +0,011) | **+0,0099** (+0,009, +0,017) | 13 / 19 |
| `real_sep` | −0,0053 | +0,0052 | 6 / 15 |
| `real_head` | −0,0054 | +0,0015 | 4 / 11 |

Proxy gộp group: mọi arm 0,8185–0,8220 (ep6), 0,8531–0,8556 (ep15) — trong nhiễu.

**Đọc theo luật khóa:** `gauss_sep` và `gauss_head` đều **dưới** production ở cả hai checkpoint ⇒ **không ứng viên online**.
`gauss_bn` − `none` +0,0012 / −0,0005 ⇒ BN trộn **không** là kênh hại. `real_head` giữ −23 % (ep6) / 24 % (ep15) lợi `real − none` ⇒ < 30 %:
lợi support thật nằm ở **backbone/BN**, head-only không đủ. Theo khóa: báo cơ chế, không mở biến thể; owner quyết đóng hướng.

**Cơ chế đo được:**
- **Synthetic có tác dụng thật nhưng nhỏ:** `gauss − none` +0,0037 / +0,0012; copula thêm +0,0021 / +0,0024 (khớp vòng 18 `cop` +0,003 và
  vòng 26 online). Copula đạt ~80 % (ep6: +0,0058 vs +0,0071) / ~32 % (ep15: +0,0036 vs +0,0111) lợi oracle so `none`.
- **Trần oracle ≈ cổng online:** ngay cả dòng thật của client khác (R3-ineligible) chỉ +0,0099 so production ở ep15 (+0,0034 ở ep6) trên
  fixture 256 bước ⇒ cổng +0,010 nằm ở mức trần mà mọi bộ sinh synthetic chỉ tiến gần. Đây là lý do mọi ứng viên synthetic online (vòng 18,
  24–26) chỉ +0,000…+0,008.
- **Hướng trọng số:** cos(D_gauss, D_real) 0,40–0,54 mọi nhóm, dưới sàn nhiễu cos(D_gauss, D_gauss_b) 0,56–0,94 ⇒ Gaussian đẩy private theo
  hướng khác hệ thống so support thật; copula gần hơn (0,44–0,69, nhất là BN stats và head). Kênh tách BN/head-only còn xa real hơn (0,11–0,47).
- **Gradient tại trạng thái đầu (ep6):** KL synthetic là số hạng lớn nhất (‖g‖ 4,88 vs CE 3,76, KD 3,17), gần trực giao với CE own (cos 0,14)
  và KD own (0,18), chỉ trùng 0,45 với gradient support thật; ở `stems` gradient synthetic gấp 3× CE. Trộn synthetic vào batch BN xoay gradient CE
  dòng thật (cos 0,89, đổi 48 % độ lớn) nhưng can thiệp `gauss_sep`/`gauss_bn` cho thấy điều này không đổi utility.
- **Proxy:** synthetic không đổi proxy — ‖D_proxy‖/cập nhật ~0,6 như nhau cho gauss/cop/real (phân kỳ do logit private khác), F1 proxy trong nhiễu.
- Giới hạn: fixture 2,5 %, 256 bước, proxy không aggregate q3; mô tả cơ chế, không thay online ep15.
