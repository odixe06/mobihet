# Hướng cải thiện model của client — 16/10/2026

**Owner 16/10 ~03:00Z:** mục tiêu là cải thiện **khả năng dự đoán của các client**, không phải model proxy.
Tài liệu này tổng hợp: trạng thái train lúc đổi mục tiêu, bằng chứng nội bộ, các phương pháp đã tìm hiểu
(mỗi ý ghi nguồn `[n]`, danh sách ở §7) và đề xuất kiểm. Chưa launch gì; mọi đề xuất cần owner duyệt.
Đối chiếu [LESSONS](../../../LESSONS.md), [danh mục phương pháp](../../proposals.md), [protocol](../../research/protocol.md).

Mức đọc nguồn: phần lớn chỉ đọc abstract/trang hội nghị/README code, chưa đọc toàn văn.
Trước khi port thuật toán nào phải đọc phần method của paper gốc; chỗ còn nghi ghi "(cần đối chiếu)".

## 1. Trạng thái train lúc đổi mục tiêu (W&B 16/10 ~03:30Z, chưa audit)

| run | trạng thái | proxy | private gp | ghi chú |
|---|---|---:|---:|---|
| `h_dml` (full q3, test) | r33, running | 0,793 | 0,744 | thuyanh316, xong ~04:30Z |
| `h_dml_own` (test) | r31, running | 0,558 | 0,675 | |
| `h_hybrid` (test) | r27, running | 0,771 | 0,741 | |
| `h_hybrid_own` (test) | r27, **finished** | 0,571 | 0,692 | hết giờ session |
| FullDrift vòng 2 (group) | `fd_dml@e15` r2, `fd_dml_sub4@e15` r2, `fd_dml_sub4_q4` r2, `fd_dml_sub16_q4` r2, `fd_fedavg@e15` r4, `fd_fedavg_sub4@e15` r4 | — | r2: 0,597 / 0,645 / 0,646 / 0,680 | 6 arm đầu đều có heartbeat 03:29Z |

Vòng 2 khóa cổng theo proxy ([design](../../methods/full-drift/design.md)); mọi arm DML vẫn ghi `private/*` nên
cho bằng chứng về client mà không tốn thêm quota.

## 2. "Dự đoán của client" đang đo thế nào

Code hiện tại ([v2_run.evaluate](../../mh/v2_run.py)): mỗi model private của client k chạy trên **cùng một**
official test toàn cục (10,76M dòng), rule `gp` cộng `log π_g − log π_k` (logit adjustment hậu nghiệm [11]),
rồi lấy trung bình 20 client. Partition Dirichlet α=0,5 theo nhãn, test không chia theo client
([dataset §8](../../../knowledge/dataset.md)).

Hai cách hiểu cho kết luận khác nhau. Bảng dưới tính lại từ confusion matrix từng client của artifact đã audit
([scripts/client_view.py](../../../scripts/client_view.py)). Cột "local" đánh trọng số lại hàng confusion theo
π_k/π_test (π_k = tỉ lệ lớp train của client k), tức giả lập phân bố nhãn riêng của client. Cách này chỉ
mô phỏng lệch nhãn, chưa mô phỏng lệch feature/receiver.

| artifact (đã audit) | model/rule | global macro F1 (min client) | local macro F1 | local accuracy (min) |
|---|---|---:|---:|---:|
| `h_dml` r23 test | private head | 0,611 (0,431) | 0,677 | **0,818** (0,666) |
| | private gp | 0,733 (0,657) | 0,661 | 0,767 (0,571) |
| | proxy head | **0,787** | 0,675 | 0,771 (0,564) |
| `h_hybrid` r18 test | private head | 0,619 | **0,683** | **0,827** |
| | private gp | 0,735 | 0,664 | 0,774 |
| | proxy head | 0,773 | 0,662 | 0,755 |
| `h_local` r23 test | private head | 0,493 | 0,556 | 0,754 |
| `fd_dml_sub4` ep6 group | private gp | 0,700 (0,632) | 0,656 | 0,796 |
| | proxy head | 0,789 | 0,690 | 0,794 |
| `fd_gaussian_swappos` ep6 group | private head | 0,591 | 0,674 | **0,877** (0,759) |

Đọc bảng:
- **(A) client = model riêng, đánh giá trên phân bố toàn cục** (protocol hiện tại). Private gp kém proxy
  0,054 ở r23 và client tệ nhất chỉ 0,657. Nếu client được dùng chính proxy thì đã đạt 0,787, nên
  "cải thiện client" theo nghĩa này là thu hẹp khoảng private–proxy khi private phải là model riêng (R2).
- **(B) client = dự đoán trên phân bố của chính client** (personalized). Private head đã ngang proxy về macro F1
  (0,677 so 0,675) và hơn về accuracy (+0,047). Gaussian swappos có accuracy local cao nhất (0,877).
  Theo nghĩa này, nhóm phương pháp personalized FL mới đúng mục tiêu.
- Đang đo ở ep6 group (pilot nội bộ) và r23 test (mô tả), một seed.

Hướng proxy đã có bằng chứng truyền sang private. FullDrift ep6 group: sub4 làm private gp tăng +0,035
(0,666→0,700) cùng lúc proxy tăng +0,026 ([kết quả](../../methods/full-drift/results.md)). Hướng chống học thuộc cũng
có bằng chứng: AntiMem pilot r20 cho DML private gp tăng với m4b (+0,021/+0,028) và swap (+0,016/+0,021)
([AntiMem](../../methods/anti-memorization/results.md)).

## 3. Owner đã chốt (16/10 ~04:00Z)

- **Cả hai định nghĩa, A là primary:** private gp trên holdout/test toàn cục; báo kèm cột local và client tệ nhất.
- **Giữ R2:** private độc lập, chỉ trao đổi qua distillation/ensemble; loại §4.5.
- **Vòng 2 giữ nguyên + cổng phụ private gp** khóa ~04:00Z ([design](../../methods/full-drift/design.md)).
- **Owner nhắc (16/10 sáng):** vẫn giữ **R1** (trao đổi ngắn trong contact hạn chế, ngân sách byte hạn chế) và **R2**
  (mỗi client có thể dùng model riêng; hiện chỉ DAGSNet để dễ so). Short-quota account được dùng cho pilot ngắn.

Câu hỏi gốc:

1. **Định nghĩa** "dự đoán của client": (A) private trên test toàn cục (giữ protocol), hay (B) trên phân bố
   riêng của từng client (cần thêm holdout theo client)? Hay báo cả hai, một cái làm primary?
2. **Ràng buộc kiến trúc:** private có phải giữ khác kiến trúc với proxy (R2) không? Nếu được dùng chung
   kiến trúc, có thể khởi tạo/trộn tham số private từ proxy ([13][7][14][15]). Khi đó (A) gần như tầm thường
   (client dùng proxy).
3. **Vòng 2 đang chạy:** giữ nguyên (đề xuất: giữ, vì arm DML đều ghi private) và **thêm cổng phụ private gp
   khóa ngay bây giờ** (đã thấy W&B r2, chưa thấy ep6)?

## 4. Phương pháp đã tìm hiểu

Cột "ý lấy từ" ghi nguồn của **từng ý**. Cột "dấu hiệu sai" là điều kiện bác bỏ khi kiểm.

### 4.1 Học tương hỗ và distillation (khung hiện tại)

| ý | lấy từ | áp vào MobiHet | bằng chứng/rủi ro nội bộ | dấu hiệu sai |
|---|---|---|---|---|
| Hai model học lẫn nhau qua KL hai chiều | DML [1] | Recipe hiện tại: private ↔ proxy cùng batch | Nền của mọi arm DML | — |
| Model chung nhỏ làm trung gian giữa model riêng và global | FML [2] | Proxy = meme model; private = model riêng của client | Private gp < proxy 0,054 ở full r23 | — |
| Cường độ KD theo độ đúng (KL chia tổng CE hai model) | FedKD [3] (cần đối chiếu eq.) | Code đã chia KL cho (CE_private + CE_proxy) | Đã có, không phải hướng mới | — |
| KD cả hidden state/attention, không chỉ logits | FedKD [3] | Thêm loss lớp giữa proxy→private (cần projector nếu khác kiến trúc) | Private thiếu ranh giới mà proxy có; ProxyFuse cho thấy private ít thông tin riêng | private gp Δ < +0,010 |
| Thầy tốt hơn ⇒ trò tốt hơn | quan sát nội bộ (FullDrift) | sub4/sub16 tăng tần suất tổng hợp proxy | sub4 private gp +0,035 ep6 | private gp ep15 sub4 − đối chứng < +0,010 |

### 4.2 Lệch phân bố nhãn khi train private (α=0,5)

| ý | lấy từ | áp vào MobiHet | bằng chứng nội bộ | dấu hiệu sai |
|---|---|---|---|---|
| Logit adjustment: cộng log prior vào logits (loss hoặc hậu nghiệm) | Menon et al. [11] | Rule `gp` hiện là bản hậu nghiệm | head→gp +0,12 global ⇒ lệch prior rất lớn | — |
| Hiệu chỉnh logits theo tần suất lớp ngay trong loss local | FedLC [10] | **Prior-consistent DML:** CE trên z+log π_k, KL với proxy trên z+log π_g; triển khai z+log π_g (A) hoặc z+log π_k (B) | [LESSONS](../../../LESSONS.md): gp sửa quá tay khi KD kéo private về global; benign gp giảm | private (z+log π_g) group ep6 < private gp +0,010, hoặc benign < −0,005 |
| Chỉ distill phần lớp "không đúng" để giữ tri thức global, không cãi CE local | FedNTD [9] | Thay KL proxy→private bằng NTD | Lệch nhãn gây quên lớp hiếm local | private gp Δ < +0,010 |
| Hai head trên một thân: head chung (loss cân bằng) + head riêng (loss local) | FedRoD [8] | Private có head A và head B; một model phục vụ cả hai định nghĩa | Head-only không chở đủ ranh giới (LESSONS) ⇒ phải giữ train thân | head A global < private gp hoặc head B local < private head |

### 4.3 Chống học thuộc receiver/domain trong private

| ý | lấy từ | áp vào MobiHet | bằng chứng nội bộ | dấu hiệu sai |
|---|---|---|---|---|
| Swap/đặt 0 khối feature receiver (swappos, m4b) | AntiMem nội bộ | Áp cho private DML trên full | Pilot DML private gp +0,016…+0,028; Gaussian full swappos +0,052 | private gp group ep6 < +0,010 |
| EMA trọng số private + BN tính lại | AntiMem nội bộ | Chỉ có lợi cho Gaussian | DML private_ema −0,026…−0,040 ở full | đã FAIL với DML, không thử lại |
| Tách bias biểu diễn theo domain client (bộ nhớ bias riêng + regularize mean) | FedDBE [16] | Bias receiver của client vào module riêng, thân dùng chung sạch hơn | Receiver shortcut là nút thắt private (LESSONS) | global và local đều < +0,010 |

### 4.4 Kết hợp private và proxy khi dự đoán (personalized, hợp với B)

| ý | lấy từ | áp vào MobiHet | bằng chứng nội bộ | dấu hiệu sai |
|---|---|---|---|---|
| Trộn model local và global bằng hệ số học được cho mỗi client | APFL [4] | Hệ số α_k giữa private và proxy | [ProxyFuse](../../methods/proxy-fusion/results.md) pilot r20 group: fuse 0,742 kém proxy −0,032 nhưng hơn private gp +0,083; chưa đo theo phân bố riêng của client | local F1 mix < max(private, proxy) + 0,010 |
| Học trọng số ensemble private+shared trên phần giữ lại của dữ liệu local, có mutual learning | FedAPEN [5] | Như trên, α_k học trên split local, hợp model khác kiến trúc | Theo local, private head ≈ proxy F1 và hơn accuracy ⇒ hai model bổ sung nhau | như trên |
| Ensemble thích nghi local+federated có regret bound | Fed-POE [6] | Phương án online của α_k | — | như trên |
| kNN trên embedding của model chung, nội suy với output | kNN-Per [12] | Bộ nhớ kNN local trên embedding proxy (dữ liệu không rời xe) | Có lợi cho B; hại A vì học thuộc receiver | local Δ < +0,010 hoặc RAM/latency quá giới hạn |
| Model chung nhỏ + model riêng, **trộn representation** qua projector riêng, head nhiều mức (Matryoshka) | FedMRL [17] | Trộn embedding proxy + private thay vì trộn logits | Đúng cấu trúc MobiHet; HtFLlib [21] xếp mạnh (nguồn thứ cấp) | client (fused) < private + 0,010 |
| Mutual learning có regularize cho personalized FL | PFML [22] | Biến thể của DML hiện tại | — | — |

### 4.5 Chia sẻ tham số (chỉ khi private được cùng kiến trúc proxy)

| ý | lấy từ | áp vào MobiHet | ràng buộc |
|---|---|---|---|
| Trộn global vào local theo từng phần tử, trọng số học trên mục tiêu local, trước mỗi round | FedALA [13] | Khởi tạo private từ trộn private/proxy | Phá R2 |
| Model riêng regularize về global bằng ‖v−w‖² | Ditto [7] | Private có số hạng prox về proxy | Phá R2; FedProx trên proxy không tăng F1 |
| Chỉ tổng hợp thân, head cố định; fine-tune head khi dùng | FedBABU [14] | Proxy body chung, head private | Phá R2; head-only yếu (LESSONS) |
| Thân chung, head riêng mỗi client | FedRep [15] | Như trên | Phá R2 |

### 4.6 Prototype/heterogeneous header (ưu tiên thấp)

FedProto (chưa xác minh nguồn), FedTGP [18], FedORGP [19], FedGH [20]: trao đổi prototype lớp hoặc head chung,
hợp với model khác kiến trúc. Ưu tiên thấp vì LESSONS ghi head-only/prototype trung bình lớp không chở đủ ranh giới
đa receiver. Tham khảo khi R2 bắt buộc và kênh logits không đủ.

### 4.7 IDS xe cộ có personalization (bối cảnh)

CANPerFL [23] (CAN IDS, model chung rồi chuyển tri thức cho từng hãng) và DFF-FL [24] (fusion feature theo node).
Khác dữ liệu (CAN bus, không phải VeReMi), chỉ dùng làm bối cảnh.

## 5. Lọc theo R1/R2 và đề xuất kiểm

| hướng | byte/contact (R1) | model riêng (R2) | xếp |
|---|---|---|---|
| m4b/swappos (§4.3), FedNTD (§4.2) | 0 byte thêm, 1 contact/epoch | chỉ đổi phía client, mọi kiến trúc | **vòng 3**: m4b/swappos PASS, NTD FAIL |
| QProxy q4 | giảm byte proxy 7,65× | proxy cố định, private tự do | **vòng 3** (noninferior ở 1 contact/epoch) |
| Hidden-state KD (§4.1, FedKD [3]) | 0 byte | cần projector local | vòng 4: **FAIL** (+0,006) |
| FedRoD hai head (§4.2, [8]) | 0 byte | chỉ đổi private | sau vòng 3 |
| FedAPEN/APFL α_k, kNN-Per, FedMRL (§4.4) | 0 byte (local) | hợp R2 | nhánh B, cần holdout theo client |
| sub4/sub16 (thầy tốt hơn) | **4×/16× contact mỗi epoch**; q4 giảm byte nhưng không giảm số contact | hợp R2 | chỉ khi lịch contact cho phép; vòng 2 đo sẵn |
| FedALA, Ditto, FedBABU, FedRep (§4.5) | — | **phá R2** | loại |

Prior-consistent DML (FedLC [10]/[11]) tạm hoãn: khi proxy cũng train CE trên dữ liệu local thì đổi tham số
z = u + log π_k cho cùng hàm loss; chưa tách được cơ chế khác hiện tại nếu không đo prior thực của proxy.

**Vòng 3 (nhánh A, cổng khóa trong [design §Vòng 3](../../methods/full-drift/design.md)):** đối chứng `fd_dml` vòng 1,
primary private gp group ep6, gain ≥ +0,010 và benign ≥ −0,005.
1. `fd_dml_m4b`: bằng chứng pilot DML private +0,021/+0,028. Cơ chế: bỏ shortcut receiver. Sai nếu Δ < +0,010.
2. `fd_dml_swappos`: pilot +0,019/+0,025, Gaussian full +0,052. Sai nếu Δ < +0,010.
3. `fd_dml_ntd` ([9]): head→gp +0,12 cho thấy private lệch prior local. Cơ chế: KD chỉ trên lớp không đúng, không
   cãi CE local ở lớp đúng. Sai nếu Δ < +0,010.
4. `fd_dml_q4`: R1, noninferior private gp và proxy (≥ −0,010).

**Kết quả vòng 3 (audited):** m4b private gp +0,036 (proxy +0,052), swappos +0,033 (+0,049) PASS; FedNTD −0,042 FAIL;
q4 noninferior. [Bảng](../../methods/full-drift/results.md).

**Vòng 4 (PUSHED 06:32Z, odixe0502):** `fd_dml_hkd`: private khớp tầng ẩn 256-d của proxy qua projector local
(FedKD [3]); proxy không đổi. Bằng chứng: private là bản chép kém của proxy (ProxyFuse) và logit chỉ chở 16 số/dòng.
`fd_gaussian_m4b`: Gaussian (client model thuần, hợp R2 nhất) + m4b, pilot +0,025 hai seed. Cổng gain như trên.

**Kết quả vòng 4–5 (audited 16/10 ~17:00Z, [bảng](../../methods/full-drift/results.md)):** hidden-state KD +0,006 **FAIL**
(cùng NTD ⇒ kênh KD không phải nút thắt, shortcut receiver mới là); Gaussian m4b +0,067 PASS; m4b@e15 private
+0,036/proxy +0,086 PASS (benign proxy không sụp); m4b_q4 noninferior; m4b_sub4_q4 private +0,031 PASS.
**Vòng 6** (seed 43, PUSHED 16:52Z): dml, m4b, sub4_q4, m4b_sub4_q4. Owner 16/10: R1 cần **mô phỏng contact thực tế**
trước khi chọn số tổng hợp/epoch (sub16 chờ).

**Nhánh B (báo kèm, chưa chạy):** fixture holdout theo client (2% cuối theo thời gian của từng client×lớp,
bổ sung `fd_prepare.py`), rồi α_k [4][5], FedRoD [8], kNN-Per [12], FedMRL [17] theo chi phí tăng dần.

Chi phí: DML ep6 đo vòng 1 = 217 phút/arm + setup 5 phút. Quota đọc live ~04:00Z (reset 17/10 00:00Z):
nguyenminhtran060113 16,6h, annhannh331 19,0h (f đang chạy), anhtruong1909 12,3h (10,3h giữ cho d/e),
thuyanh316 8,4h, odixe0502 4,0h, account khác ≤2,5h (không đủ một arm ep6).

## 6. Đã loại hoặc không ưu tiên

- FedFomo/peer ensemble: upload model riêng, trái ràng buộc RSU chỉ thấy tổng cohort ([tài liệu 15/10](new-directions-2026-10-15-afternoon.md)).
- Private EMA cho DML: đã FAIL ở full (§4.3).
- Kênh thống kê vô hướng (ClassTail, CCR…): ≈ local (LESSONS).

## 7. Tài liệu tham khảo

1. Zhang, Xiang, Hospedales, Lu. *Deep Mutual Learning*. CVPR 2018. [arXiv:1706.00384](https://arxiv.org/abs/1706.00384)
2. Shen et al. *Federated Mutual Learning*. 2020. [arXiv:2006.16765](https://arxiv.org/abs/2006.16765)
3. Wu, Wu, Lyu, Huang, Xie. *Communication-efficient federated learning via knowledge distillation* (FedKD). Nature Communications 2022. [PMC9018897](https://pmc.ncbi.nlm.nih.gov/articles/PMC9018897), [arXiv:2108.13323](https://arxiv.org/abs/2108.13323)
4. Deng, Kamani, Mahdavi. *Adaptive Personalized Federated Learning* (APFL). 2020. [arXiv:2003.13461](https://arxiv.org/abs/2003.13461)
5. Qin et al. *FedAPEN: Personalized Cross-silo Federated Learning with Adaptability to Statistical Heterogeneity*. KDD 2023. [doi:10.1145/3580305.3599344](https://doi.org/10.1145/3580305.3599344), [code](https://github.com/zhenqincn/FedAPEN)
6. *Personalized Federated Learning with Mixture of Models for Adaptive Prediction and Model Fine-Tuning* (Fed-POE). 2024. [arXiv:2410.21547](https://arxiv.org/abs/2410.21547)
7. Li, Hu, Beirami, Smith. *Ditto: Fair and Robust Federated Learning Through Personalization*. ICML 2021. [arXiv:2012.04221](https://arxiv.org/abs/2012.04221)
8. Chen, Chao. *On Bridging Generic and Personalized Federated Learning for Image Classification* (FedRoD). ICLR 2022. [arXiv:2107.00778](https://arxiv.org/abs/2107.00778)
9. Lee et al. *Preservation of the Global Knowledge by Not-True Distillation in Federated Learning* (FedNTD). NeurIPS 2022. [proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/fadec8f2e65f181d777507d1df69b92f-Abstract.html), [arXiv:2106.03097](https://arxiv.org/abs/2106.03097)
10. Zhang et al. *Federated Learning with Label Distribution Skew via Logits Calibration* (FedLC). ICML 2022. [PMLR v162](https://proceedings.mlr.press/v162/zhang22p.html), [arXiv:2209.00189](https://arxiv.org/abs/2209.00189)
11. Menon et al. *Long-tail learning via logit adjustment*. ICLR 2021. [arXiv:2007.07314](https://arxiv.org/abs/2007.07314)
12. Marfoq, Neglia, Kameni, Vidal. *Personalized Federated Learning through Local Memorization* (kNN-Per). ICML 2022. [arXiv:2111.09360](https://arxiv.org/abs/2111.09360)
13. Zhang et al. *FedALA: Adaptive Local Aggregation for Personalized Federated Learning*. AAAI 2023. [doi:10.1609/aaai.v37i9.26330](https://doi.org/10.1609/aaai.v37i9.26330), [arXiv:2212.01197](https://arxiv.org/abs/2212.01197)
14. Oh, Kim, Yun. *FedBABU: Toward Enhanced Representation for Federated Image Classification*. ICLR 2022. [arXiv:2106.06042](https://arxiv.org/abs/2106.06042)
15. Collins, Hassani, Mokhtari, Shakkottai. *Exploiting Shared Representations for Personalized Federated Learning* (FedRep). ICML 2021. [arXiv:2102.07078](https://arxiv.org/abs/2102.07078)
16. Zhang et al. *Eliminating Domain Bias for Federated Learning in Representation Space* (FedDBE). NeurIPS 2023. [proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2e0d3c6ad1a4d85bef3cfe63af58bc76-Abstract.html), [code](https://github.com/TsingZ0/DBE)
17. Yi et al. *Federated Model Heterogeneous Matryoshka Representation Learning* (FedMRL). NeurIPS 2024. [proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7a9ee756fb71d401bcb5230db02c11db-Abstract.html), [arXiv:2406.00488](https://arxiv.org/abs/2406.00488)
18. Zhang et al. *FedTGP: Trainable Global Prototypes with Adaptive-Margin-Enhanced Contrastive Learning for Data and Model Heterogeneity in Federated Learning*. AAAI 2024. [arXiv:2401.03230](https://arxiv.org/abs/2401.03230)
19. *FedORGP: Guiding Heterogeneous Federated Learning with Orthogonality Regularization on Global Prototypes*. 2025. [trang tóm tắt](https://lacuna.tiptreesystems.com/work/fedorgp-guiding-heterogeneous-federated-learning-with-orthogonality/wrk_a22b500140cdb1a2c304a75a96a98765) (chưa xác minh bản gốc)
20. Yi et al. *FedGH: Heterogeneous Federated Learning with Generalized Global Header*. [arXiv:2303.13137](https://arxiv.org/abs/2303.13137)
21. Zhang et al. *HtFLlib: A Comprehensive Heterogeneous Federated Learning Library and Benchmark*. KDD 2025. [arXiv:2506.03954](https://arxiv.org/abs/2506.03954), [code](https://github.com/TsingZ0/HtFLlib)
22. Yang et al. *Regularized Mutual Learning for Personalized Federated Learning* (PFML). ACML 2021. [mlanthology](https://mlanthology.org/acml/2021/yang2021acml-regularized)
23. *CANPerFL* — personalized FL cho CAN IDS. Applied Sciences 13(11):6369. [MDPI](https://www.mdpi.com/2076-3417/13/11/6369) (chỉ đọc abstract)
24. *A Lightweight Intrusion Detection System with Dynamic Feature Fusion Federated Learning for Vehicular Network Security* (DFF-FL). Sensors 2025, 25(15):4622. [PMC12349077](https://pmc.ncbi.nlm.nih.gov/articles/PMC12349077/)

Nội bộ: [FullDrift](../../methods/full-drift/results.md) · [AntiMem](../../methods/anti-memorization/results.md) ·
[ProxyFuse](../../methods/proxy-fusion/results.md) · [chẩn đoán chững](plateau-diagnosis-2026-10-15.md).
