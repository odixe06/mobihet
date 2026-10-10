# Đề xuất hướng mới — cộng tác không chia sẻ dữ liệu, model khác loại, ngân sách contact

Cập nhật 2026-10-10. **Đề xuất, chưa triển khai.** Mỗi hướng gồm: cơ chế, cách đáp ứng R1–R3, bằng chứng thuận/nghịch từ archive, phép thử đầu tiên có thể bác bỏ.
Bằng chứng và số gốc: [kiến thức kế thừa](../legacy/kien-thuc-ke-thua.md). Protocol: [protocol.md](protocol.md).

## 1. Yêu cầu

| # | yêu cầu | kiểm bằng |
|---|---|---|
| R1 | Giao tiếp xe–RSU nằm trong ngân sách contact: deadline thật, tính đủ byte | simulator archive, kế toán byte, assert deadline/byte cap |
| R2 | Client được dùng model khác kiến trúc | registry `dagsnet/mlp/cnn1d/gru` (+TCN); không averaging tham số của model riêng |
| R3 | Không dữ liệu thô hoặc record `(x, y)` nào rời xe; không client nào thấy local data của client khác (tiêu chuẩn FL), kể cả trong đối chứng thí nghiệm | assert payload; tri thức hợp lệ = tham số khối chung, prototype/thống kê, logit trên dữ liệu proxy, generator/dữ liệu tổng hợp |
| R4 | 66 feature gốc, cùng cấu hình `fl_dagsnet_veremi` | protocol §1 |

## 2. Ràng buộc thiết kế rút từ archive

- **C1.** Tri thức phải chở được **ranh giới đa receiver**, không chỉ trung bình lớp. Mọi biến thể prototype Build3–6 chỉ được ±0,006.
- **C2.** Thống kê trong latent bị cũ đi khi backbone còn học (N3 offline +, online −).
  Chỉ dùng chúng để calibration trên snapshot đóng băng (kiểu D3); không dùng làm loss liên tục.
  Thống kê trong **không gian input** thì không bị trôi theo representation.
- **C3.** Backbone yếu cần tri thức tác động toàn model (M2: chỉnh head ≈ 0 ở kịch bản khác kiến trúc).
- **C4.** Trên chính dữ liệu này, **chia sẻ tham số** mang được tri thức đa receiver (0,77–0,82 test 20c khi không giới hạn truyền).
  NILM-FL cho thấy học tương hỗ private↔proxy đưa được tri thức đó vào model private: 0,514 → 0,696, tăng suốt 50 round.
- **C5.** Công thức local dùng chung cho mọi arm: `gp`, EMA + tính lại BN, LR cosine. So sánh phải cùng nền này.
- **C6.** Ngân sách quyết định dạng tri thức nào khả thi (bảng §3).

## 3. Ngân sách contact và kích thước tri thức

Phân bố byte/contact đo bằng simulator archive (R=800 m, 5–20 m/s, rate 20–80 kB/s, 50 round × 20 xe; không trừ thời gian train):

| mức | `contact_scale` / `rate_scale` | trung vị | p10 | p90 |
|---|---|---:|---:|---:|
| B0 stress (Build9–21, cap 32 KiB) | 0,01 / 0,2 | 4,5 kB | 0,7 kB | 12,8 kB |
| B1 | 0,1 / 0,2 | 46 kB | 8 kB | 129 kB |
| B2 | 0,1 / 1,0 | 231 kB | 42 kB | 645 kB |
| B3 không scale | 1,0 / 1,0 | 2,2 MB | 0,34 MB | 6,3 MB |

Kích thước payload một chiều (chưa tính header/ACK):

| tri thức | kích thước | B0 | B1 | B2 | B3 |
|---|---:|---|---|---|---|
| π_g (histogram 16 lớp) | 64 B | ✓ | ✓ | ✓ | ✓ |
| Head chung d=64 (64×16+16) fp16 | 2,1 kB | ✓ | ✓ | ✓ | ✓ |
| Logit int8 trên K=256 mẫu proxy | 4,1 kB | ~1 contact | ✓ | ✓ | ✓ |
| Gaussian theo lớp, input 66 chiều, mean+var fp16 | 4,3 kB | ~1 contact | ✓ | ✓ | ✓ |
| … + hạng thấp r=4 | 12,7 kB | 2–3 contact | ✓ | ✓ | ✓ |
| Prototype 16×256 fp16 (DAGSNet latent 256) | 8,3 kB | 2 contact | ✓ | ✓ | ✓ |
| Proxy MLP 66-64-64-16 (9.488 tham số) int8 | 9,5 kB | 2–3 contact | ✓ | ✓ | ✓ |
| Proxy MLP 66-128-128-16 (27.152) int8 | 27 kB | ~6 contact | ✓ | ✓ | ✓ |
| Head DAGSNet đầy đủ (144.720) fp16 | 289 kB | ✗ | ~6 contact | 1–2 contact | ✓ |
| DAGSNet đầy đủ (395.024) int8 / fp16 | 395 / 790 kB | ✗ | ✗ | 2–4 contact | ✓ |

⇒ Ở B0 chỉ tri thức cỡ vài kB/round là khả thi. Chia sẻ model cỡ DAGSNet chỉ khả thi từ B2–B3. **Phải chốt mức ngân sách trước khi chọn kiến trúc proxy (Q1).**

## 4. Các hướng đề xuất

### H1 — Proxy dùng chung + học tương hỗ (ưu tiên 1)

**Cơ chế.**
- Mỗi xe giữ model private (kiến trúc tùy ý) và một **proxy** có kiến trúc chung, nhỏ.
- Ngoài contact, xe train hai model bằng deep mutual learning: CE nhãn thật + KL hai chiều private↔proxy.
- Trong contact, xe tải xuống bản proxy toàn cục mới nhất (delta so với version xe đang có) và tải lên delta proxy của mình. Cả hai chiều đều nén và chia fragment có cursor.
- RSU tổng hợp **bất đồng bộ** theo staleness: mỗi xe đến vào thời điểm khác nhau, không có round đồng bộ thật.
- Model private không bao giờ rời xe; suy luận dùng private (có `gp`).

**Đáp ứng.**
- R2: chỉ proxy có kiến trúc chung, private có kiến trúc bất kỳ.
- R3: chỉ tham số proxy rời xe. Có thể thêm DP-SGD cho riêng proxy như ProxyFL, private vẫn train không cần DP.
- R1: chọn kích thước proxy + nén (int8, top-k, SVD hạng thấp như FedKD) theo ngân sách §3.

**Bằng chứng thuận.**
- Repo anh em NILM-FL, trên đúng dữ liệu này với proxy = DAGSNet và không giới hạn truyền: proxy 0,823, private 0,696, tăng suốt 50 round.
- pFedES (chia sẻ proxy extractor nhỏ nhưng tri thức không đi vào nhánh suy luận) chỉ đạt 0,486. ⇒ cách đưa tri thức vào model private quyết định kết quả.
- Literature: [ProxyFL](https://www.nature.com/articles/s41467-023-38569-4) (Nat. Commun. 2023: private khác kiến trúc + proxy dùng chung + DML, có DP),
  [FML](https://arxiv.org/abs/2006.16765), [FedKD](https://www.nature.com/articles/s41467-022-29763-x) (mutual distillation + nén gradient SVD, giảm tới 94,89% byte),
  [FedMRL](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7a9ee756fb71d401bcb5230db02c11db-Abstract.html) (NeurIPS 2024: model phụ nhỏ dùng chung + projector cá nhân).

**Bằng chứng nghịch / rủi ro.**
- Proxy nhỏ (kiểu MLP) có thể không đủ sức chứa. Ở M2, MLP64 là model yếu nhất (0,37 → 0,47 dù có peer). Chưa biết MLP proxy FedAvg đạt bao nhiêu trên 16 lớp này.
- KD từng không thêm gì ở Build7. Nhưng khi đó teacher học trên cache nhỏ, khác ở đây (proxy đã học từ mọi client).
- Tổng hợp bất đồng bộ + non-IID α=0,5 có thể trôi (staleness).
- Ở B0, proxy phải rất nhỏ hoặc chia nhiều contact.

**Phép thử đầu tiên.**
- H1-T0, không kênh. Chạy FedAvg thuần trên proxy ứng viên (MLP nhỏ, CNN1d nhỏ) và local-only DAGSNet, 20c, ngắn.
  Nếu proxy global < local-only DAGSNet trên test thì proxy nhỏ khó mang thêm tri thức ⇒ cần proxy lớn hơn (kéo theo ngân sách ≥ B2) hoặc bỏ dạng nhỏ.
- H1-T1, không kênh. Học tương hỗ với 2–3 cỡ proxy, đo private − local-only. Cổng đề xuất: ≥ +0,03 cả hai seed.
- H1-T2. Bật kênh (bất đồng bộ, nén, deadline). Giữ được ≥ 70% lợi của T1.
- H1-T3. Private khác kiến trúc: không kiến trúc nào âm, kiến trúc yếu lợi nhiều nhất.

### H2 — Thống kê lớp trong không gian input → dữ liệu tổng hợp cho head triển khai (ưu tiên 2)

> **Tín hiệu đầu (2026-10-10, [H2-T0-lite](probes/h2-input-stats.md), MLP, 4 client × 2 seed):** Gaussian hiệp phương sai đầy đủ gộp từ 19 client khác
> +0,102/+0,106 macro-F1 so với replay cân bằng của chính client (8/8 dương); diag +0,050/+0,058; điểm trung bình lớp −0,013/−0,022.
> Cổng PASS. Chưa phải DAGSNet, chưa có ngân sách contact/DP ⇒ cùng hạng ưu tiên với H1, chạy S1a trước S2.

**Cơ chế.**
- Mỗi xe gửi lên **tổng có thể cộng dồn** theo lớp trong không gian 66 feature: `n_c`, `Σx`, `Σx²` (đường chéo); tùy chọn sketch hạng thấp hoặc GMM ít thành phần.
- RSU cộng các tổng này, rồi phát xuống phân phối có điều kiện theo lớp ở mức **quần thể**.
- Xe sinh mẫu tổng hợp `x̃ ~ p(x|c)` và dùng chúng ở đúng chỗ record cache từng thắng: fit lại **head triển khai** mỗi cycle trên backbone EMA đóng băng (D3), và/hoặc replay có trần.

**Đáp ứng.**
- R2: mẫu ở không gian input, nên mọi kiến trúc dùng được giống record cũ. Mỗi xe tự encode bằng backbone của mình. Không cần latent chung.
- R3: chỉ thống kê cộng dồn rời xe. Tương thích secure aggregation (chỉ RSU thấy tổng); có thể thêm nhiễu DP vào tổng.
- R1: diag 4,3 kB, hạng thấp r=4 12,7 kB.
- Thống kê input không bị trôi theo representation, tránh được lỗi N3 (C2).

**Bằng chứng thuận.**
- Record replay + head triển khai là đòn bẩy lớn nhất của archive. M1: 32 mẫu thật/lớp ≈ 70% lợi tối đa.
- N3 offline cho thấy thông tin bậc hai có ích khi encoder đóng băng.
- Hướng này kiểu CCVR, nhưng đặt ở không gian input thay vì latent.

**Rủi ro.**
- Gaussian/GMM có thể mô tả kém các lớp đa mode (benign, các scenario khác nhau) và các feature rời rạc/điền 0 (cụm Sybil).
- Mẫu tổng hợp không tách được lớp thì chỉ dời ranh giới (U15: gửi có chủ đích các lớp giống benign làm benign recall −0,05/−0,10).
- Thống kê input của một client lộ nhiều hơn thống kê latent. Cần tổng hợp qua ≥ k client trước khi phát xuống, và cân nhắc DP.

**Phép thử đầu tiên, offline, rẻ.** Không client nào thấy dữ liệu của client khác, kể cả trong đối chứng (quy tắc FL, chủ dự án nhắc 2026-10-10).
Đại lượng duy nhất đi qua client là tổng cộng dồn theo lớp.
- H2-T0-lite (đang chạy, [kết quả](probes/h2-input-stats.md)): MLP nhỏ, 4 client đích × 2 seed. So:
  - local;
  - local + replay cân bằng của chính client (đối chứng prior);
  - local + mẫu Gaussian diag / full gộp từ 19 client còn lại;
  - local + điểm trung bình lớp (đối chứng kiểu prototype).
- H2-T0 (sau S0): backbone local-only DAGSNet đóng băng, fit lại head 16 bước trên local / local cân bằng / mẫu tổng hợp.
- Cổng đề xuất: tổng hợp − local cân bằng ≥ +0,01 và tổng hợp − điểm trung bình > 0, cả hai seed, benign F1 không giảm quá ngưỡng nhiễu.
- Không đạt thì dừng H2 dạng tham số, cân nhắc generator (H5).

### H3 — Head/classifier chung trên giao diện latent (FedGH / LG-FedAvg)

**Cơ chế.**
- Mỗi model private thêm projector cục bộ về d chiều chung (ví dụ 64).
- Head chung (64×16) được tổng hợp bằng FedAvg, hoặc RSU train head từ prototype đã nhận ([FedGH](https://arxiv.org/abs/2303.13137), ACM MM 2023).
- Xe dùng head chung thay head riêng hoặc kết hợp với head riêng.

**Đáp ứng.** Payload 2 kB, hợp B0. Khác kiến trúc được nhờ projector.

**Bằng chứng nghịch.**
- FedGH train head từ prototype trung bình, tức bị giới hạn bởi C1.
- P3 projection head từng −0,026.
- Head chung trên các backbone khác nhau buộc backbone thích nghi theo head, có thể kéo lùi model mạnh.

**Vai trò.** Rẻ, thuộc HtFLlib, nên dùng làm **baseline HtFL** và thành phần phụ. Không ưu tiên làm phương pháp chính.

### H4 — Họ prototype làm baseline bắt buộc (FedProto, FedTGP, TinyProto)

- TinyProto-FP và pFedES đã có bản dựng trong `fl_dagsnet_veremi` (20c 0,621 / 0,486). Paper MobiHet cần các baseline này.
- Cải tiến duy nhất có cơ sở từ archive: dùng thống kê latent (mean + hạng thấp) **chỉ** cho head triển khai trên snapshot EMA đóng băng, không làm loss liên tục (C2).
- Không đầu tư thêm vào biến đổi trung bình lớp (C1).
- Code baseline: [HtFLlib](https://github.com/TsingZ0/HtFLlib) (KDD 2025; có FedProto, FedTGP, FedGH, FML, FedKD, LG-FedAvg, FedGen).

### H5 — Distillation bằng logit trên tập proxy không riêng tư, và generator

**Logit trên proxy (kiểu [DS-FL](https://arxiv.org/abs/2008.06180)).**
- Xe tính logit trên K điểm proxy dùng chung, gửi K×16 int8 (4 kB với K=256).
- RSU gộp logit (entropy reduction), phát lại soft label đồng thuận; xe distill.
- Hoàn toàn không phụ thuộc kiến trúc và byte không tăng theo cỡ model.
- **Cần nguồn proxy hợp lệ.** Ba lựa chọn, đều phải chủ dự án chấp nhận giả định:
  - (i) mẫu sinh từ thống kê H2 với seed chung — 0 byte cho input;
  - (ii) tin nhắn RSU tự nghe được (không nhãn);
  - (iii) mô phỏng công khai riêng bằng bộ sinh VeReMi NextGen.
- Rủi ro: KD/teacher trước đây yếu (Build7, S13 ≈ làm mượt theo lớp). Phải có cổng chất lượng teacher và đối chứng class-mean.

**Generator tại RSU (kiểu [FedGen](https://proceedings.mlr.press/v139/zhu21b.html), ICML 2021).**
- RSU học generator nhỏ từ các proxy/head đã nhận để sinh mẫu cho xe.
- Dữ liệu dạng bảng 66 chiều nên generator nhỏ.
- Chỉ mở nhánh này nếu H2 dạng tham số thiếu độ trung thực.
- Rủi ro: generator có thể ghi nhớ mẫu; phải đánh giá rò rỉ.

### H6 — Lớp "MobiHet": lập lịch theo contact (đóng góp riêng của dự án, chồng lên H1/H2)

- Lập lịch gửi gì trong contact ngắn: ưu tiên fragment theo giá trị trên byte (thống kê hay delta proxy), dựa trên `tau_hat`, không nhìn tương lai.
- Có cursor tiếp tục giữa các contact; chỉ kích hoạt snapshot coherent.
- RSU tổng hợp bất đồng bộ có trọng số staleness.
- Đây là phần giữ Gap D của `MobiHet_IDS.md` §1.2 (heterogeneity + contact deadline + chọn tri thức). Chỉ đánh giá **sau khi** H1/H2 có tín hiệu.

## 5. Thứ tự và kịch bản kiểm chứng

| bước | việc | đầu ra | cổng/dừng |
|---|---|---|---|
| S0 | Driver theo protocol §1, dựng từ code `fl_dagsnet_veremi` + simulator archive. Mốc **local-only** 20c (50 round, test mọi round) | số mốc dưới chưa có | tái lập được một baseline anh em (ví dụ FedAvg/AFPHA 20c) trong sai số 1 seed |
| S1a | H2-T0 offline: head refit trên mẫu tổng hợp vs local cân bằng và điểm trung bình (máy local hoặc 1 phiên Kaggle) | bảng lợi so đối chứng | tổng hợp − local cân bằng ≥ +0,01 cả hai seed |
| S1b | H1-T0: sức chứa proxy (FedAvg proxy nhỏ, không kênh) | proxy global F1 so với local-only | proxy ≥ local-only DAGSNet |
| S2 | H1-T1 học tương hỗ, 2 cỡ proxy, 20c, 2 seed, ngắn | private − local | ≥ +0,03 cả hai seed |
| S3 | Bật kênh (mức ngân sách theo Q1) cho nhánh qua cổng | lợi giữ lại, byte, số contact hoàn tất | giữ ≥ 70% lợi |
| S4 | Khác kiến trúc (DAGSNet/CNN1d/GRU/MLP), lập lịch H6, 50/100 client | bảng theo kiến trúc, Pareto byte–F1 | không kiến trúc nào âm |
| S5 | Run chính 50 round theo protocol, baseline HtFL (TinyProto, FedProto/FedTGP, FedGH, FML/FedKD), hai seed | bảng so sánh với `fl_dagsnet_veremi` | — |

Mỗi bước: cổng khóa trước khi chạy, cặp seed, một yếu tố/arm, đủ 10 metric. Thất bại thì sửa theo nguyên nhân hoặc bỏ, không kéo dài run để tìm điểm thắng (LESSONS §1).

## 6. Câu hỏi cần chủ dự án chốt

**Đã chốt 2026-10-10:**
- Q1: báo nhiều mức, chính B0 + B2.
- Q2: protocol anh em + holdout pilot.
- Q4: không dữ liệu thô + secure aggregation mô phỏng, DP sau.
- Bước tiếp: dừng để chủ dự án đọc tài liệu.

Còn mở: Q3, Q5, Q6, Q7.

1. **Q1 — ngân sách contact.** Giữ B0 stress của archive (trung vị 4,5 kB), hay dùng B1/B2/B3, hay báo nhiều mức?
   Mức chọn quyết định proxy H1 có được dùng DAGSNet hay phải là model nhỏ.
2. **Q2 — protocol.** Theo đúng `fl_dagsnet_veremi` (50 round × 1 epoch, test mọi round, headline r50), cộng holdout nội bộ cho pilot như [protocol §3](protocol.md)?
3. **Q3 — số contact mỗi round.** Mỗi round mỗi xe có đúng một contact mô phỏng, hay dùng multi-contact (M15)?
   Repo anh em giả định mọi client tham gia mọi round.
4. **Q4 — mức riêng tư.** Đủ ở mức "không dữ liệu thô rời xe", hay cần DP (ε) / secure aggregation có số liệu?
5. **Q5 — dữ liệu proxy cho H5** (nếu đi nhánh này): chấp nhận nguồn nào trong (i)/(ii)/(iii)?
6. **Q6 — thứ tự.** Đồng ý S0 → S1a ∥ S1b → S2 không? Hay muốn đi thẳng một hướng?
7. **Q7 — M4b.** Xác nhận tắt M4b (giữ nguyên 66 feature) để khớp repo anh em.
