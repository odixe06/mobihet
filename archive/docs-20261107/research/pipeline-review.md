# Rà pipeline Core sau F50 — còn cải tiến được ở đâu

06/11/2026, sau F3. Đây là phân tích và đề xuất. [Y50](../../../docs/research/results-y50.md): oracle r50 đã audit, generator_worth_it; prior trượt r20; ablation Q/floor đang nối phiên. các mục khác chưa có run, chưa có quyền launch. Đối chiếu đủ M→X, K/L, H/J, R/V và v1.
Kết quả: [F50](../../../docs/research/results-f50.md). Tái tạo: `scripts/analyse_core_gaps.py`, `scripts/analyse_core_bias.py`, `scripts/probe_information_gap.py`.
Nguồn tham khảo: [papers](../../../papers/README.md).

## 1. Core chưa tối ưu: số đo

Core = trung bình F2/F3 nếu không ghi khác; baseline một seed ở repo benchmark; official test đã đọc nhiều lần nên chỉ để định vị.

| chỉ số r50 | Core | FedAvg | APFed client | centralized |
|---|---:|---:|---:|---:|
| accuracy | .9096 | .9228 | .9278 | .9580 |
| f1_macro | .8995 | .9042 | .9119 | .9593 |
| timeDelay F1 | .3413 | .1767 | .2742 | .6164 |
| benign F1 (F3) | .8415 | .8686 | .8754 | .9208 |
| client mean | .8640 | .8667 | .8663 | — |
| client mean bỏ timeDelay (F3) | .9060 | .9227 | .9196 | — |
| p10 bỏ timeDelay (F3) | .8180 | .8360 | .8314 | — |

- SCAFFOLD có p10 .7738 và CANPerFL .7684, cao hơn Core .7564. Nhóm xe <256: Core .844, CANPerFL .866, SCAFFOLD .855.
- 59,2% lỗi pooled F3 là cặp benign↔timeDelay (timeDelay→benign 37,9%, ngược lại 21,3%).
- Xe yếu: decile dưới có timeDelay F1 trung bình .10 và lớp này có mặt ở 81% xe, nên timeDelay kéo p10.
- 65,2% dòng test thuộc 600 xe ≥16.384 dòng; ở nhóm này Core .8997, APFed .9108.
- Core kém precision ở lớp hiếm: accelerationMultiplication .841 so với FedAvg .980, positionMirroring .812 so với .901, suddenConstantSpeed .867 so với .967.
- Byte L50: bootstrap 92,1% (637.066 B), Q 7,9% (54.400 B).

## 2. Probe train fold → val fold

150/410 file, 1,89 triệu dòng fit, 1,72 triệu dòng val, không đọc official test. Mô hình gộp, không phải model từng xe; HGB chỉ học 1,5 triệu dòng.

| bộ phân loại | macro F1 | timeDelay F1 | other15 | AUC benign/timeDelay |
|---|---:|---:|---:|---:|
| Gaussian QDA | .8198 | .2783 | .8559 | — |
| Bayes của copula, không train | .8420 | .3243 | .8765 | .6794 |
| HGB gộp | .8539 | .2468 | .8944 | .7902 |
| Bayes copula + gộp sender | .8542 | .3547 | .8875 | .6992 |
| HGB + gộp sender | .8698 | .2415 | .9117 | .8053 |

Đọc: (1) copula bỏ mất thông tin benign/timeDelay (AUC .68 so với .79), nên chất lượng generator còn dư địa; (2) gộp sender nhân quả có lợi cho 15 lớp còn lại, không sửa timeDelay; (3) Bayes copula yếu hơn học sinh, không dùng làm teacher, chỉ là mốc tham chiếu 0 train.
Val phân đoạn ngắn (trung vị 1 dòng/cặp) nên gộp sender có thể bị đánh giá thấp. Val khác official test, không so trực tiếp với cột Core ở mục 1.

## 3. Từng khâu

| khâu | hiện tại | đã thử, không lặp | còn mở |
|---|---|---|---|
| Đặc trưng | 77 cột mỗi dòng | m4b là đồng nhất trên v4; projection cờ NaN không ổn định | ngữ cảnh chuỗi theo sender (repo features) |
| Bootstrap | histogram 256 bin + Σz, Σzzᵀ theo lớp | Gaussian kém hơn; copula_cluster, copula_mix, GMM/MADE/CCVR/atom/MALA, sketch, latent book; boot64 PASS r10; lossless PASS r50 | moment thưa theo lớp có mặt; boot64 ở r50 |
| Sinh synthetic | 64 dòng cân bằng lớp, dòng mới mỗi bước | short ưu tiên lớp thiếu hại p10; bank hữu hạn kém streaming; liều 32→64 | **prior của synthetic chưa nhất quán với LA** |
| Loss xe | CE+LA thật, KL(Q[y]‖p) synthetic, hệ số 1 | adaptive kém; realKD/NCKD/margin/strength/gate/fallback/ρ; synth NCKD chỉ dời ngưỡng | chiếu gradient khi real/synthetic xung đột (cosine ≈ .03, ~1/4 âm, chưa thử) |
| Teacher Q | online trên dòng thật, gộp n_kc | postepoch ≈ control; sqrt(n) không nâng xe nhỏ; Q8 hại; bucket 8 không lợi | Q sụp về one-hot; Q từ dòng chưa học; bỏ Q ở r50 |
| Tối ưu | 1 epoch, reset AdamW, floor 16 | EMA private không ổn định theo tổ hợp; không đổi protocol riêng | floor 32; floor chưa kiểm ở r50 |
| Suy luận | logit thô từng dòng | luật prior trộn gpe v1 −.09; BN recalibration/split FAIL | gộp sender nhân quả, cần đổi protocol |
| Đánh giá | own-test, 10 metric | — | báo thêm metric không timeDelay, AUC; ablation r50 |
| Tiêu chí owner | băng thông: có; đa mô hình: **chưa có run nào** | — | ô trộn kiến trúc |

## 4. Ứng viên, theo thứ tự

Mọi mục giữ protocol trừ khi ghi. Ngưỡng đọc: pooled .0005, p10 .006, low256 .003, timeDelay .0035.

| # | việc | bằng chứng → cơ chế | dấu hiệu | bác bỏ khi | chi phí |
|---|---|---|---|---|---|
| 1 | Ablation r50: Core−Q, Core−floor, 2 seed | bank mất lợi từ r20 tới r50; floor và Q cũng chỉ có bằng chứng r20; Q entropy .05 ở r50 | Core−Q trong spread thì Core thành one-shot, 0 byte/round | Core hơn rõ ở cả hai seed thì giữ nguyên | 4 kernel ~10,2 h |
| 2 | Trần oracle r50 (`synth=oracle`, không triển khai được) | AUC .68 so với .79; G1 r10 còn +.0123 so với Gaussian | oracle − Core lớn thì đáng làm generator | chênh trong spread thì dừng hướng generator | 1 kernel, cần kiểm code arm với Core |
| 3 | Synthetic nhất quán prior: thêm shift log(1/16) − log π_g vào loss synthetic | thật dùng prior toàn cục, synthetic prior đều; precision lớp hiếm thấp | other15 và benign tăng vượt spread, 0 byte | timeDelay giảm >.02 hoặc absent giảm >.01 | r20 2 seed, xác nhận r50 |
| 4 | Ô trộn kiến trúc: xe nhỏ model nhỏ, thêm một họ khác DAGSNet | tiêu chí owner chưa có bằng chứng; W.25 val .9296 so với W1 .9648 | mỗi nhóm kiến trúc giữ low256/p10 trong spread | nhóm nào sụp so với Core đồng nhất | cần sửa engine grouped; owner 02/11 hoãn |
| 5 | Q từ dòng synthetic mới, cùng byte (chỉ khi #1 giữ Q) | Q[13,13] .938 so với recall test .288 | Q giữ mềm tới r50, other15 tăng | không vượt spread; trần thấp vì Q chỉ +.002 ở r20 | r20 |
| 6 | Floor 32 (chỉ khi #1 giữ floor) | low256 thua CANPerFL .022 | low256 tăng ở r50 | chỉ tăng ở r20 như bank | r50 |
| 7 | Moment thưa theo lớp có mặt | xe trung bình có 13,1 lớp; lossless, không đổi train | 691.466 → ≈655.737 B (−5,2%, tính từ count, chưa đo wire) | — | không tương thích SecAgg |

## 5. Hướng khác ngoài copula

Phải giữ băng thông thấp và model tùy ý. Thứ tự theo bằng chứng hiện có.

| hướng | vì sao còn đáng xét | rủi ro | quyết định cần |
|---|---|---|---|
| Model phụ siêu nhỏ dùng chung, ví dụ W.25 26.864 tham số, q4 ≈13–14 KB/chiều/round (kiểu FedMRL) | mang ranh giới học từ dòng thật, đúng phần Core thiếu (other15 −.017); model riêng vẫn tùy ý; S0 cũ thua nhưng dùng hệ số adaptive và Gaussian, C14 chỉ học bank | ≈1,3 MB/50 round, gấp ~2 lần Core; KD từ model gộp có thể hại timeDelay (FedAvg F1 .18) | **trái quyết định "không trao đổi trọng số"**, owner phải chọn |
| Generator học federated (flow/diffusion bảng) | chỉ mở nếu #2 cho thấy trần cao; v1 GMM/MADE FAIL nhưng v1 cũng từng FAIL copula | cần nhiều round trao trọng số generator, byte lớn hơn bootstrap | sau #2 |
| Gộp nhân quả theo sender hoặc model chuỗi tại xe | 0 byte, không phụ thuộc kiến trúc; probe other15 +.017 | đổi protocol, phải đánh giá lại baseline; không sửa timeDelay | owner chọn đổi cách chấm |
| Dữ liệu mô phỏng công khai ở server làm tập chuyển giao | VeReMi vốn từ mô phỏng; có joint thật mà không đụng dữ liệu xe | bank hữu hạn từng kém streaming (−.005…−.009); cần giả định dữ liệu công khai | owner chọn giả định |

**Đã loại, không thử lại nếu không có cơ chế mới:** prototype (FedProto/FedTGP/TinyProto/V2X-HtFL p10 .55–.59); proxy server, FedDF/FedMD (S1, C8–C10, C14); anchor/input-bank (J, P, Q, F50); bucket/ô input (K6/K7, BoundaryCells); ridge RFF (.38–.48); kênh vô hướng (ClassTail, CrossClientRank); sketch/subspace/PowerSGD; realKD mọi biến thể; BN split.

## 6. Giới hạn và điểm cần ghi khi viết bài

- Fed-TDA đã tổng hợp bảng từ phân bố từng cột + covariance toàn cục, nên đóng góp của Core phải đặt ở phần theo lớp, step floor, class Q, lossless và quy mô 5.343 xe. Mới đọc tóm tắt, cần đọc toàn văn.
- Centralized là một model toàn cục, có thể dùng ngữ cảnh chéo xe nhận; .6164 là bằng chứng thông tin tồn tại, không phải mục tiêu đạt được ở từng xe.
- Can thiệp #3 prior đã trượt gate r20 cả hai seed: gain pooled nhưng hại timeDelay/absent. Oracle seed42 r50 macro +.018422 so Core: còn dư địa generator, chưa chứng minh hướng học generator cụ thể.
- Hai seed không phải significance. Chưa có DP/SecAgg thực thi; bootstrap một lần cũng chưa kiểm khả năng bị xe độc hại đầu độc.

Phân tích sau oracle: [generator từ thống kê gộp](generator-after-oracle.md). Ưu tiên kiểm joint histogram2D residual trên nền copula; hypothesis, chưa launch.

Pairhist finitecandidateIPF đã [trượt feasibility](results-pair-generator.md), không chạyFL. [Forestconditionalhybrid](generator-direct-conditional.md) là đềxuấtmới dựa trênESSfailure, chưatriểnkhai.
