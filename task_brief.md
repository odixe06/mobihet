# Task brief — xây dựng lại MobiHet cho Fable 5.2

Ngày tổng hợp: 16/11/2026. Ngôn ngữ làm việc: tiếng Việt. Đây là bản tổng hợp phục vụ reasoning,
không thay protocol, kết quả gốc hoặc bàn giao sống. Số lịch sử phải đọc đúng method/model/seed/endpoint.
Thông tin mới owner chốt trong phiên này ưu tiên hơn yêu cầu nghiên cứu cũ.

## 1. Nhiệm vụ và yêu cầu owner đã xác nhận

Thiết kế lại MobiHet cho federated IDS V2X trên VeReMi NextGen v4. Tối ưu đồng thời chất lượng,
communication và compute, với **khả năng client dùng kiến trúc model khác nhau là yêu cầu quan trọng nhất**.

| Trục | Yêu cầu hiện tại |
|---|---|
| Performance | Primary = pooled Macro-F1 16 lớp. Giữ khoảng top 5; chấp nhận thấp hơn baseline tốt nhất tối đa khoảng **0.015 điểm tuyệt đối**, không phải 0.015% |
| Communication | Xếp hạng **riêng**, tổng byte càng thấp càng tốt, cần top 5. Owner xác nhận **không bắt buộc xấp xỉ phương pháp ít byte nhất** |
| Compute | Tối ưu thấp nhất có thể trong miền performance đạt yêu cầu. Owner chưa đặt trần 1.2×/1.5×/2×; không tự tạo trần hoặc yêu cầu top 5 compute |
| So FL trao weights | Macro-F1 ngang hoặc gần; communication tiết kiệm rõ; compute có thể xấp xỉ |
| So FL trao thông tin khác | Mong muốn Macro-F1 tốt hơn; communication cạnh tranh/top 5; compute không phình quá lớn, giảm được càng tốt |
| Heterogeneity | Cơ chế trao đổi phải dùng được khi client dùng model hoàn toàn khác nhau; cùng số lớp/ý nghĩa nhãn và input schema |
| Thử nghiệm mới | **Seed 42**, chạy riêng DAGSNet và riêng FeatureBERT, **sim → pilot → full**. Owner xác nhận **chưa chạy mixed-model** |
| Công bằng | Đồng nhất dữ liệu, split, local epoch, round, batch, optimizer, LR, evaluation và phạm vi đo cost |

Không dùng Pareto rank thay hai hạng performance/communication độc lập. Báo cả chênh lệch so baseline tốt nhất
và thứ hạng: đạt khoảng cách 0.015 chưa tự bảo đảm top 5. T3 cũ là control riêng khi thử phương pháp mới;
không đếm hai biến thể MobiHet như hai baseline độc lập để làm lệch hạng.

Phạm vi phiên này: tổng hợp brief; owner bổ sung pull/audit pFedES cuối ở `fl_veremi_bert`, cập nhật Notion,
tài liệu và push GitHub; owner sau đó yêu cầu push mọi thay đổi hiện tại **cả hai repo**. **Chưa có lệnh launch training phương pháp mới**.

## 2. Bản đồ repo và nguồn ưu tiên

Workspace gốc `/home/odixe/project/`:

| Repo / nguồn | Vai trò |
|---|---|
| `mobihet/` | Phương pháp MobiHet, lịch sử thử nghiệm, generator/codec, T3 DAGSNet, audit và lesson |
| `fl_veremi_bert/` | FeatureBERT, 14 baseline, port T3, kết quả hai model và comparison builder |
| `fl_dagsnet_veremi_v2/` | Baseline DAGSNet chuẩn; protocol, implementation từng bài, cost và kết quả 3 seed |
| `mobihet-features/` | Dựng dataset từ JSON raw, split, sửa artifact, feature engineering, scaler. Tên thư mục thực tế dùng dấu `-`, không phải `mobihet_features` |

Thứ tự tin cậy: quyết định owner mới → executed source/config + audited artifact → results/method/protocol
hiện hành → catalog lịch sử → snapshot archive. Không lấy một câu "đang chạy" trong design cũ làm trạng thái hôm nay.

Điểm vào: [CONTEXT](CONTEXT.md), [LESSONS kỹ thuật](LESSONS.md),
[LESSONS vận hành](docs/operations/lessons.md), [bàn giao sống](docs/operations/continue.md).
Nguồn so sánh: [comparison.md](docs/research/comparison.md), [comparison.json](docs/research/comparison.json),
[builder BERT](../fl_veremi_bert/scripts/build_project_comparison.py).
JSON chứa `evidence` theo seed/view/source, `performance`, `communication`, `client_compute`, `server_compute`.

Các HEAD khi bắt đầu đọc: MobiHet `234afd2f227e631a4a50c77d03f896bb694a7579`,
BERT `d41b05610f9d0af7828804cf493c010bb5e6f228`,
features `f6d7d9ef9fb243c419ba8ddb8a6c54dfbd3305cc`.
Đây là mốc tra cứu; không phải source SHA chung của mọi run.

## 3. Dataset: thông tin cần cho thiết kế cơ chế

Nguồn: [dataset canonical](knowledge/dataset.md), [input order](knowledge/input_order.json),
[label mapping](knowledge/label_mapping.json), [dataset builder context](../mobihet-features/CONTEXT.md),
[feature processing v4](../mobihet-features/docs/research/feature-processing-v4.md),
[README bản scaled](../mobihet-features/data/v4-scaled/README.md).

- Kaggle: `odixe0502/veremi-nextgen-v4-scaled-per-vehicle`.
- **5.343 client = xe nhận**, khóa `scenario × run × receiver_id`; 4 scenario, 8 run. Partition tự nhiên, **không Dirichlet**.
- Tổng 46.708.952 dòng. Fold train 32.685.166; val 4.676.107; test 9.347.679.
- FL dùng **train + val = 37.361.273 dòng**, official test = 20% cuối timeline mỗi xe.
  Fold val đã nhập vào train ở full-scale; không được coi đó là holdout độc lập của run full.
- Mọi client có train/test và benign. Train/client: min 76, p10 178, median 1.368, p90 18.177, max 95.287.
  Test/client median 343, min 20. Thiếu lớp và mất cân bằng số dòng giữa xe rất mạnh.
- **77 input = 71 feature + 6 cờ NaN**, float32 trước lưu fp16 trên GPU.
  `msg`4, `rel`8, `ses`14, `trf`6, `win`9, `nbr`2, `map`4, `p4_std`24.
- Đã scale: `z = Phi^-1(CDF_midrank(x))`, fit **fold train**, NaN→0 + cờ, clip khoảng ±3.72.
  Không normalize lại; mọi phương pháp dùng đúng thứ tự 77 cột.
- Scaler dataset dùng histogram 16.384 bin trên `sign(x)*log1p(abs(x)/1e-6)`.
  Đây **khác** histogram 256 bin có nhãn của bootstrap MobiHet trên đầu vào đã scale.
- Định danh/nhãn không là input: `scenario, world, run, client_id, fold, label, receiver_id, sender_id, message_id, rcv_time_ns`.

16 lớp: `0 benign`, `1 accelerationMultiplication`, `2 constantPositionOffset`, `3 constantSpeedOffset`,
`4 dataReplay`, `5 dosAttack`, `6 feignedBraking`, `7 positionMirroring`, `8 randomPositionOffset`,
`9 randomSpeedOffset`, `10 reversedHeading`, `11 suddenConstantSpeed`, `12 suddenStop`,
`13 timeDelayAttack`, `14 trafficCongestionSybil`, `15 zeroSpeedReport`.
Lớp11 ít nhất: 225.294 dòng train+val, 63.314 test. Benign 9.773.821 / 2.172.713;
TD 1.924.480 / 482.549; Sybil 3.287.699 / 1.652.147.

Các caveat ảnh hưởng reasoning:

1. Raw JSON có `rcv_pos` sai ở 6 lớp. V3 sửa trạng thái xe nhận qua quỹ đạo `ego_*`,
   bỏ 7.071.455 dòng tấn công ngoài vòng đời xe nhận. Không suy insight từ v2/final5132 sang v4.
2. Benign lấy một lần theo lần nhận, world chọn bằng hash trong các world không sửa nội dung.
   13/15 loại attack có twin benign cùng sự kiện. Random row split có thể leak; row-disjoint chưa event-independent.
3. `ses_dmsgid` đã bỏ: `message_id` là bộ đếm simulator, shortcut mạnh.
   Không dùng sender thật để nối pseudonym Sybil; không dùng world làm nhãn phụ.
4. TD dời cả send/receive time 2–4s cố định trong phiên; `rcv-send` và động học trong phiên không tự lộ attack.
   Tín hiệu nằm ở tương tác/bối cảnh. Centralized tốt hơn chứng tỏ **không được kết luận 77 feature bất lực tuyệt đối**.
5. Mỗi world chỉ có một loại attack; benign giữa Sybil/DoS dễ hơn world yên tĩnh: shortcut bối cảnh.
   Sybil prior train+val **8.80%**, test **17.67%**. Ghi chú 8.0% ở repo features là fold train riêng.
6. Feature phiên/cửa sổ dùng lịch sử nhân quả; suy luận vẫn một quyết định/message.
   Gộp hậu nghiệm xác suất cả phiên là thay đổi evaluation, từng bị owner đóng.
7. Session đúng để phân tích = `(client, world, pseudonym)`; `(client,pseudonym)` trộn world/twin.
   Metadata đó chỉ dùng diagnostic theo hợp đồng, không tự đưa vào model.
8. Scaled cờ NaN/quan hệ vật lý không thể đọc như raw values. Nội suy cùng nhãn chưa chắc giữ manifold.

Code dataset: [build.py](../mobihet-features/src/v2/build.py), [features.py](../mobihet-features/src/v2/features.py),
[ego.py](../mobihet-features/src/v2/ego.py), [map.py](../mobihet-features/src/v2/map.py),
[p4.py](../mobihet-features/src/v2/p4.py), [scale.py](../mobihet-features/src/v2/scale.py).
Phiên này đọc code/docs; `data/v4-scaled/clients.parquet` local chưa có, chưa đếm lại toàn dataset.

## 4. Model và protocol công bằng

Nguồn: [protocol DAGSNet](docs/research/protocol.md), [protocol BERT](../fl_veremi_bert/docs/research/protocol.md),
[DAGSNet](knowledge/dagsnet.md), [FeatureBERT](../fl_veremi_bert/knowledge/bert.md),
[BERT parity](../fl_veremi_bert/docs/research/v2-parity.md).

| Chung | Cấu hình khóa |
|---|---|
| Round / local data | 50 round, mỗi xe đúng 1 epoch dòng thật/round, không bỏ tail batch/client |
| Batch | 64; xe nhỏ dùng hết số dòng có |
| Optimizer | AdamW, beta=(0.9,0.999), eps=1e-8, decay=1e-4; reset moment mỗi round |
| LR | Cosine theo round, 1e-3 r1 → 1e-5 r50; LR cố định trong round |
| Số học | Clip norm1 theo client, dropout0.1, AMP fp16; runtime Kaggle 2×T4 đã pin |
| Participation | Mọi xe mỗi round; ngoại lệ method gốc có selection phải ghi rõ và tính selected fraction |
| Init | Cùng theta0 trong cùng model/seed. Không thể yêu cầu DAGSNet/BERT có cùng tensor theta0 |
| Aggregation weights | Theo số dòng n_k; prototype giữ luật bài gốc |
| Endpoint | Round cuối, không best-round/test tuning; CANPerFL có ngoại lệ fine-tune dưới đây |

- **DAGSNet W1**: 395.408 tham số, BN buffers; input `x.view(B,11,7).transpose(1,2)`.
  Bốn nhánh Dense/Google/Alex/Squeeze, head hidden256. W0.75 từng đạt gate centralized nhưng owner giữ W1.
- **FeatureBERT**, không phải BERT ngôn ngữ pretrained: 77 token feature + CLS; d64, 2 encoder layer,
  4 attention head, FFN128, post-LN, dropout0.1; **77.904 tham số, không BN buffer**, CLS hidden64.
  Code [bert.py](../fl_veremi_bert/core/bert.py), [grouped.py](../fl_veremi_bert/core/grouped.py).
  Ít tham số hơn DAGSNet nhưng FLOPs forward cao hơn: **10.225664 vs 4.452096 MFLOP/dòng**.
- Baseline DAGSNet nằm repo `fl_dagsnet_veremi_v2`; BERT port kiểm AST20 gate giữ data/train/method semantics.
- Baseline production `group_rows=8192`, T3 frozen build `4096`. Đây là khác biệt hạ tầng phải công khai khi so wall time.
  Không dùng latency hai cấu hình để khẳng định thuật toán nhanh hơn nếu chưa benchmark chung.
- CANPerFL = FedAvg50 + **10 epoch fine-tune**. Báo rõ thêm compute; không lấy ngoại lệ này để cho MobiHet thêm epoch.
- LwFedNIDS dùng model tỉa; compute rất thấp không phản ánh cùng full architecture.

Đánh giá: global model chấm toàn test; private model xe k chấm test xe k rồi gộp CM.
Pooled Macro-F1 tính trên **16 lớp**, khác trung bình Macro-F1 theo xe (chỉ lớp có y_true ở xe đó).
Báo đủ10 metric: accuracy; precision/recall/F1 macro, micro, weighted; thêm per-class F1,
client mean/p10/median/min, low256, rare/absent, benign↔TD confusion. Guard phụ hiện report-only;
không bỏ lớp yếu khỏi Macro-F1 hoặc chọn view tốt nhất sau khi thấy test.

## 5. MobiHet hiện tại = T3, cơ chế chính xác

Nguồn: [method](docs/research/method.md), [proposal](docs/research/proposal.md),
[T3 results](docs/research/results-t3.md), [B21](docs/research/screening-dose-b21.md).
**Core, KM50, CBB, T3 là các phiên bản khác nhau; không tráo số.**

Bootstrap một lần trên train+val local:

1. Histogram theo lớp/feature, 256 bin → CDF gộp toàn mạng.
2. Normal-score mid-rank; moment lớp; federated k-means **16 cụm/lớp, 4 lượt Lloyd**.
   Covariance mỗi cụm shrink về covariance lớp, `w0=0.5`; bỏ cụm <20 dòng toàn mạng.
3. Lite6: moment/tâm int8, sparse+zlib, Cholesky phần dưới6bit, đường chéo/scale fp16, CDF lossless.
4. Xe tự sinh synthetic từ mixture copula; server không gửi dòng thật hoặc model.

CBB dùng64 synthetic/bước + floor16. **T3 đổi liều theo số dòng xe và nửa lịch**, giữ codec/Q/loss nền:

| Tầng n_k(train+val) | r1–25 | r26–50 |
|---|---:|---:|
| S <4096 | 64 | 64 |
| M 4096–16383 | 64 | 32 |
| L ≥16384 | 48 | 16 |

Mỗi lớp synthetic cân bằng; floor16 bước/round cho xe ít bước thật, bổ sung **synthetic-only**.
Không lặp thêm dòng thật. Loss synthetic lấy mean riêng, hệ số1; giảm số dòng **không giảm hệ số loss**.

```text
pi_k[c] = (n_kc + 1)/(n_k + 16)
pi_g[c] = (sum_k n_kc + 1)/(sum_k n_k + 16)
L_real = mean CE(z_real + log(pi_k) - log(pi_g), y_real)
round1: L_syn = mean CE(z_syn, y_syn)
round>=2: L_syn = mean KL(Q_previous[y_syn] || softmax(z_syn))
L = L_real + L_syn
Q_r[c] = sum_k sum_{i:y_i=c} softmax(f_k(x_i)) / sum_k n_kc
```

Q thu **online trên dòng thật khi train**, không thêm forward. Q r chỉ dùng từ r+1.
QQ8 lượng tử8bit, uplink chỉ lớp hiện diện; downlink bảng chung16×16.
Q[y] là teacher **theo true class**, không là classifier điều kiện theo x.
Private model và BN giữ tại xe. Cơ chế schema độc lập kiến trúc; DAGSNet/BERT đã chạy **riêng**, chưa full-scale mixed.

Code: [mobihet_compute.py](methods/mobihet_compute.py) → [mobihet_refine.py](methods/mobihet_refine.py)
→ [mobihet_sim.py](methods/mobihet_sim.py), [mobihet_wstudy.py](methods/mobihet_wstudy.py),
[mobihet.py](methods/mobihet.py), [core/local.py](core/local.py).
HP T3: `synth_tier_min=[4096,16384]`, `synth_tier_rows=[[64,64],[64,32],[48,16]]`,
`synth_late_from=25`, `synth_step_floor=16`, `q_bits=8`, KM16/lite6.

Hợp đồng nghiên cứu hiện có: chỉ thống kê/count/xác suất, không real rows/model/gradient/BN,
không shared head/adapter hoặc hidden-space prototype bắt buộc cùng representation.
Không mặc định có public real data/teacher pretrained. Nới hợp đồng phải nêu rõ để owner quyết.
**Chưa có SecAgg/DP**; moment ô1dòng có thể lộ dữ liệu. Không đồng nhất "không gửi row" với bảo đảm privacy.

## 6. Kết quả hiện tại, baseline và ngưỡng mới

Các số dưới là bản r50 đã audit3seed41/42/43, đã bổ sung pFedES cuối16/11.
[Comparison canonical](docs/research/comparison.md) giữ bảng đầy đủ và nguồn cập nhật.

| Method / kênh trao đổi | Macro DAGSNet | Macro BERT | MB/xe/50r DAGSNet | MB/xe/50r BERT |
|---|---:|---:|---:|---:|
| APFed / weights | .9130 | .9076 | 159.4688 | 31.1616 |
| CANPerFL / weights + fine-tune | .9097 | .9117 | 159.4688 | 31.1616 |
| **MobiHet T3 / bootstrap + Q** | **.9080** | **.9148** | **1.164697** | **1.168699** |
| FedAvg / weights | .9048 | .8990 | 159.4688 | 31.1616 |
| FedProx / weights | .9005 | .8886 | 159.4688 | 31.1616 |
| AFPHA / weights | .8993 | .8868 | 159.4688 | 31.1616 |
| FD-IDS / weights + KD | .8990 | .8814 | 159.4688 | 31.1616 |
| LwFedNIDS / pruned weights | .8892 | .8823 | 16.2000 | 4.0776 |
| NILM-FL / proxy weights | .8850 | .8821 | 159.4688 | 31.1616 |
| FedTGP / prototypes | .8719 | .8826 | 1.6384 | .4096 |
| PerFed-SKD / selected weights | .8713 | .8922 | 45.6177 | 11.8769 |
| FedProto / prototypes | .8693 | .8876 | 1.6416 | .4128 |
| V2X-HtFL / prototypes | .8666 | .8770 | 1.6416 | .4128 |
| TinyProto / sparse prototypes | .8541 | .8857 | .166912 | .044928 |
| pFedES / shared proxy extractor weights | .8523 | .8740 | .1404 | .1404 |

pFedES không phải kênh chỉ-xác-suất; trao tham số extractor nhỏ. Prototype/representation khác chiều không tự là
giải pháp cho arbitrary model heterogeneity. Danh mục và deviation từng baseline:
[methods BERT](../fl_veremi_bert/docs/research/methods.md), docstring `methods/<alias>.py` ở repo baseline.

T3 DAGSNet seed41/42/43: **.907866023 / .907678902 / .908324032**;
mean ± sampleSD **.907956319 ± .000331909**.
BERT T3 seed42 **.914532369**, mean3seed **.914770238 ± .000211920**.
Audit CM/source/HP/support PASS; state audit thêm finite banks/strict rebuild/RNG khi có. Ba seed không là significance.

Seed42 baseline tốt nhất sau bổ sung pFedES: DAGSNet APFed **.911900144**,
BERT CANPerFL **.910056154**. Suy ra floor theo khoảng cách0.015 lần lượt
**.896900144** và **.895056154**. Đây là **phép tính tham chiếu trên seed42 r50**, không gate sim/pilot,
không thay điều kiện top5, và phải tính lại nếu baseline tốt nhất thay đổi.
Không so candidate1seed với mean3seed để khóa PASS.

**Ảnh hưởng pFedES tới mục tiêu communication:** bảng đầy đủ đã bổ sung pFedES sau audit phiên cuối; trước đó T3 hạng2 DAGSNet/hạng5 BERT.
Payload pFedES đã có trong cost.json: 1.404B lên + 1.404B xuống/xe/round → .1404MB/50r.
Với đầy đủ14baseline, T3 hiện **hạng3 DAGSNet/hạng6 BERT**: BERT chưa đạt mục tiêu communication top5 mới.
Với BERT, nhóm rẻ gồm TinyProto(.044928), pFedES(.1404), FedTGP(.4096), FedProto/V2X(.4128).
Để vào top5 vững khi xếp ordinal, nên tổng byte **<.4128MB/xe/50r**, tức giảm khoảng **65%** so T3 hiện tại.
Đây là mốc suy ra từ bảng cost, **không phải cap owner tự chốt**; xử lý đồng hạng phải thống nhất trước đo.
Không loại pFedES khỏi roster chỉ để giữ top5.

## 7. Chi phí: cách đo, điểm phình và giới hạn bằng chứng

Communication dùng unicast từng xe, MB=10^6B, tổng = bootstrap + tất cả round(up+down).
Báo tách bootstrap/online/up/down/tổng xe/tổng mạng. Baseline weights mặc địnhfp32, T3 codec theo stored packets.
Chưa tính header/retransmission/SecAgg; không gộp broadcast miễn phí vào bảng unicast.
Scaler dataset là preprocessing chung; nếu tính end-to-end preprocessing phải tính cho mọi method cùng phạm vi.

- Bootstrap T3 **1.151278MB/xe**, online DAGSNet .013420MB/50r, BERT .017421MB/50r.
  Bootstrap chiếm **98.85% / 98.51%** tổng byte. Cắt thêm Q ít giúp tổng50round.
- T3 ít byte hơn FedAvg khoảng **137× DAGSNet / 26.7× BERT**, đã gồm bootstrap.
- CBB từng bị báo nhầm4×F; phép đo sửa thành **6.09×F**, T3 khoảng **4.94×F DAGSNet**.
  Không dùng hệ số DAGSNet để tính BERT.

| Client compute, MFLOP/dòng thật/round | DAGSNet | BERT |
|---|---:|---:|
| FedAvg/FedProx/FedProto/AFPHA | 13.297 | 40.022 |
| CANPerFL, phân bổ thêm10epoch | 15.957 | 48.026 |
| FedTGP/TinyProto/FD-IDS | 17.749 | 50.248 |
| V2X-HtFL | 18.290 | 50.285 |
| **T3, ước lượng useful work** | **21.964** | **66.108** |
| APFed/NILM-FL | 26.594 | 80.044 |
| pFedES | 35.524 | 100.521 |
| LwFedNIDS, model tỉa | 1.288 | 5.880 |

T3 ≈**1.652× FedAvg** ở mỗi model; thấp hơn APFed nhưng cao hơn prototype methods.
Nguồn: [comparison](docs/research/comparison.md), [cost BERT](../fl_veremi_bert/docs/research/cost.md),
[cost.py BERT](../fl_veremi_bert/scripts/cost.py), `../fl_dagsnet_veremi_v2/scripts/cost.py`.
Baseline đo round2 toyCPU bằng FlopCounterMode, gồm matmul/conv/bmm/SDPA, không đếm đầy đủ
elementwise/optimizer/codec/eval/bootstrap. PerFed-SKD compute hiện chuẩn hóa theo dòng xe được chọn;
so total-network phải nhân đúng participation/data, không chỉ đặt FLOP/dòng cạnh nhau.

T3 dùng `C_train(model)*(1+rho)`,
`rho=sum_k max(ceil(n_k/64),16)*mean_dose_k/(64*sum_k ceil(n_k/64)) = .651785455`,
mean_dose S/M/L=64/48/32. Đây là quy ước theo bước của B21, không đếm tất cả padded GPU work.
Engine grouped sinh số dòng lớn nhất trong nhóm rồi mask; `mask=0` không chắc loại phép toán khỏi kernel.
T3 giảm FLOPs hữu ích19% so CBB nhưng `t_train` s42 +1%, s43 −4%: chưa có wall-time speedup rõ.
Không suy tiết kiệm năng lượng/CPU/edge latency khi chưa đo.

Server T3 steady-state khoảng2.822M operation/round theo `2*K*C^2 + K*C + 3*C^2`,
chưa codec/bootstrap. Bootstrap gồm histogram/moment/Lloyd/factorization; **chưa đo đầy đủ**, không ghi0.
So compute mới cần tách client/server, bootstrap/train/eval, measured/estimated, useful/padded work,
warmup/steady-state và peak memory; ưu tiên một harness đo chung.

## 8. Nút thắt khoa học và bài học không nên lặp

Nguồn: [method-building brief](docs/research/method-building-brief.md), [diagnosis](docs/research/diagnosis.md),
[LESSONS](LESSONS.md). Các chẩn đoán sau chủ yếu **CBB/KM50 DAGSNet**, chưa tự chuyển sang T3 BERT.

- CBB r50 thiếu hụt Macro-F1: TD44%, benign11%, class11 10%, class7 9%.
  TD F1 khoảng.351; centralized DAGSNet TD khoảng.616, macro toàn cục.9593.
- Oracle thay synthetic bằng pooled real rows: macro.9176 seed42, ngoài hợp đồng FL.
  Đây là trần chẩn đoán của carrier/cấu hình cụ thể, không trần mọi phương pháp heterogeneous FL.
- Gap tới APFed chủ yếu benign tương lai → FP attack; private train precision≈.99, test≈.84.
  V20 TD recall train.886 → future val.267 ở xe lớn. Personalization có lợi nhưng dễ học thuộc local context.
- Khoảng59% lỗi đếm dòng benign↔TD. Prior/threshold/tỷ lệ/temperature thường dời lỗi, không thêm phân tách.
- Curve phẳng cuối50round do cosine LR; không chứng minh model hết khả năng học.
  Thêm epoch/LR cho riêng MobiHet không công bằng với baseline.
- Synthetic mang kiến thức toàn mạng cả ở xe lớn. 600 xe≥16k dòng giữ65% dòng thật/63% bước synthetic;
  3.820 xe<4096 chỉ14.6% bước. Compute tập trung xe lớn, thiệt hại cắt liều tập trung xe nhỏ.
- Q train gần one-hot hoặc confidence cao không chứng minh generalization/reliability.
  MeanQ không khôi phục per-row variance/Brier/NLL; muốn moment2 phải thực sự thu/gửi thêm.
- LL/TSTR/pairTV/teacher agreement/codec exactness là diagnostic, không thay classifier utility FL.
- p10 sim200 dao động lớn; absent full chỉ1.555dòng. Báo guard/trend nhưng không suy significance hoặc CI từ2seed.

## 9. Những gì đã thử: tra theo cơ chế, không theo tên mới

Index chính: [tried-methods-index](docs/research/tried-methods-index.md),
[generator catalog](docs/research/generators.md), [improvement catalog](docs/research/improvement-proposals.md).
**FAIL dưới đây là verdict theo gate/control/recipe lúc đó**, không định lý mọi biến thể cùng họ vô ích.
Mục tiêu mới cho phép trade-off rộng hơn; không sửa ngược verdict cũ. Muốn xem lại adaptation cũ cần
chỉ ra lý do mục tiêu/cơ chế/cost thay đổi, evidence mới và gate mới trước run; không tự restart nhánh đã đóng.

| Study | Cơ chế / kết quả cần nhớ | Nguồn |
|---|---|---|
| b1–b8 | Bỏ synthetic ở xe đủ dòng hại đơn điệu; tăng hệ sốsynthetic hại; KM16 ra đời. Sim mixed W1/W0.25 không sụp, chưa mixed DAGSNet/BERT full | [sim fold val](docs/research/screening-sim-foldval.md) |
| KM50 | KM16 w0.5 int8 tăng macro +.0085/+.0086 vs Core r50; benign giảm khoảng.0015, owner nhận | [KM50](docs/research/screening-km50.md) |
| b9/b10 | K32/K8/K24, EM1/2, whitening, pop/partial stats, temp/share/gap-fill/Q từsynthetic, covariance variants FAIL utility; LL tăng chưa dự báo gain | [KM-lite](docs/research/screening-km-lite.md) |
| Codec/combo | Lite6+QQ8 nhận đếnr50; QQ4 FAIL xe nhỏ. Bỏ Lloyd riêng PASS chưa bảo đảm combo: LY0+QQ8 FAIL benign | [codec](docs/research/screening-codec.md) |
| b11 | EMA beta.5/.8, prior inference, Q theo KMregion, softenQ đều FAIL; Qregion cũng có tiền thân K10 bucket8 | [catalog](docs/research/improvement-proposals.md) |
| b12 | Floor24–64, synthetic80–128 tăng compute mà gain nhỏ; gộp session có gain≈+.002 nhưng guard/policy loại | [catalog](docs/research/improvement-proposals.md) |
| G1 | Cắt đều48/32synthetic từr1: full pilot FAIL, low256−.012/−.030; không đồng nghĩa T3 cắt theo cohort FAIL | [catalog](docs/research/improvement-proposals.md) |
| B13 | Sqrt-countQ, reverse/symmetricKL,4JS, realCEsmoothing.05: 5arm FAIL2seed. Có biến thể đổi benign lấyTD; JS gain<.001 | [B13](docs/research/screening-module-b13.md) |
| B14 | Same-class interpolation.5 macro−.0158; geometric/meanlogQ thêm byte không gain≥.002. Teacher rất sắc vẫn FAIL | [B14](docs/research/screening-uplink-b14.md) |
| B15 | SAMrho.05 gain+.00053/+.00120, elapsed+58%, p10/TD giảm; generic optimizer gain không chứng minh FL tốt hơn | [B15](docs/research/screening-sam-b15.md) |
| B16 | KL-of-classmean prediction, half-dose: MK−.001855/−.000390; HK≈0/+.000365, không qua gain gate | [B16](docs/research/screening-mean-b16.md) |
| B17 | Prior-coordinate, realNCKD trên CBB: prior−.000658/+.000192, NR−.000164/+.001630, FAIL gain2seed | [B17](docs/research/screening-macro-b17.md) |
| B18 | Shadow/out-of-fit teacher matched controls: H−C−.001066/−.001132; benign↑, TD−.047/−.038. Train≈1.92×, bootstrap≈2.38MB | [B18](docs/research/screening-source-b18.md) |
| A1 | Generator posterior soft labels: chỉ.86%synthetic non-onehot; BayesTDrecall.36<.42; FAIL local, chưatrain | [sổ tay §5](docs/research/method-building-brief.md) |
| B19/A2' | Thêm mixture ở vùng generator nhầm; Bayesmacro+.003<gate+.010, TD giảm; FAIL diagnostic local | [B19](docs/research/screening-boundary-b19.md) |
| N1 | Server proxy DAGSNet trên synthetic trùng P20; proxy≈.79, student≈.856 ở recipe cũ; chưa có cơ chế mới đểlaunch | [sổ tay §5](docs/research/method-building-brief.md) |
| N2 | Centralized HGB trên sim làm trần cấu hình GBDT liên kết: macro.8065<.924; dense hist byte rất lớn; FAIL local, không mọiGBDT | [sổ tay §5](docs/research/method-building-brief.md) |
| B20 SH/SM/SD | SHhead-only macro−.016, absent−.21/−.25: backbone cần tri thức synthetic. SMfull16+head48 r50≈.9057/.9058,4.65×F. SD64→32 từr26 r50≈.9072/.9081,5.32×F | [B20](docs/research/screening-compute-b20.md), [gate addendum](docs/research/screening-compute-b20-addendum.md) |
| B21 | T1/T3/T4F r50PASS2seed, T3chi phíthấpnhất trongarmPASS→đượcchọn. T4pilotFAIL; T5simFAIL. T4Ffloor48 tăngp10+.012/.013 nhưng compute/time tăng | [B21](docs/research/screening-dose-b21.md) |
| F50/L50 | Input-bank lợi r20 mất r50; giữ streaming/Core, CDF lossless | [F50](docs/research/results-f50.md), [L50](docs/research/results-l50.md) |
| Y50 | Oracle; bỏQ/bỏfloor; prior gain Core chưa sangCBB. Q/floor needed ởr50 | [Y50](docs/research/results-y50.md) |
| Generator khác | PairIPF, forest+Gaussian, marginal transport/copula, standardlatent, forest+quantile: local hoặc pilotFAIL | [catalog](docs/research/generators.md) |

Lịch sử trước Core — mở đúng hồ sơ khi cần, không nạp toàn archive:

| Study | Cần tránh gọi là chưa thử | Link |
|---|---|---|
| M20 | Teacher + personalized distillation, sqrt-count trên recipe cũ | [M20](archive/docs/research/results-m20.md) |
| N20/O20 | RealKD × syntheticfallback factorial; teacherreplay | [N20](archive/docs/research/results-n20.md), [O20](archive/docs/research/results-o20.md) |
| P20/P21 | Ensemble/bank/stream/hybrid/serverproxy/sharedproxyweights | [P20](archive/docs/research/results-p20.md), [P21](archive/docs/research/results-p21-recovery.md) |
| Q20/R20/V20 | Streaming/inputtargets/serverweight/reception; matchedteacherreplay; temporal localvalidation | [Q20](archive/docs/research/results-q20.md), [R/V20](archive/docs/research/results-r20-v20.md), [V20](archive/docs/research/screening-v20.md) |
| W20/X20/C10 | Decoupled/NCKD, stepfloor, bootstrapcompression, inputbankcombos | [index07/11](archive/docs-20261107/README.md) |
| K10/L10/H10/J10/R10 | BucketQ, KDtarget/weight/gate, synthetic10roundscreens | [tried index](docs/research/tried-methods-index.md) |
| Legacyv1 | 66feature, Dirichlet20/50/100client; NTD/hiddenKD/FedDyn/SCAFFOLD/prox/EMA/GMM/MADE/CCVR… | [kế hoạch lịch sử](archive/docs/research/improvement-plan.md) |

Các gain cũ không chuyển tự động sang v4/5.343xe/T3/BERT; không cộng gain ablation để đoán combo.
Đối chiếu novelty cần đọc [papers index](papers/README.md),
[generator primary sources](docs/research/generator-primary-sources.md): Fed-TDA gần bootstrap copula,
federated distillation theo lớp gần Q. Chưa đánh giá novelty khoa học mới trong phiên này.

## 10. Thang thử mới và điều kiện chốt trước launch

Owner đã chọn **sim → pilot → full, seed42, đơn model**. Thang lịch sử để tái dùng cấu trúc,
**chưa phải bộ gate mới được owner duyệt**:

| Bậc | Cấu hình lịch sử | Kiểm trước lên bậc |
|---|---|---|
| Local preflight | Fixture200xe, compile forward+backward,2round; chặnNaN/skip/rebuild/packet lỗi | Knoboff=control; source/config đúng; cost accounting hoạt động |
| Sim | Panel200xe seedpanel7, foldtrain1.476.064 → foldval211.118;20round cosine20 | So control/baseline cùngmodel/seed/panel/endpoint. Panel đã dùng nhiều lần, khôngfreshconfirmation |
| Pilot | Full5.343xe, train+val, r1–20 của **cosine50**, checkpoint nối được | So cùnground/cùngschedule, audit rồi mớiutility. Không so simcos20 với pilotcos50 như cùngđiểm |
| Full | Nối đúngcheckpoint tới50round, vẫncosine50 | Finalgate: khoảngcáchbestbaseline≤.015, performance/communicationtop5, báocompute vàguard |

Thiết kế BERTsim/control phải xác minh: đã có BERTfull không có nghĩa đã có đầy đủ baselineBERTsim/pilot
trên cùngpanel/cosine. Khi thiếu, ghi thiếu evidence và dự toán matchedcontrol; không lấy r50full làm gatesim.
Mốc late phải theo nửa lịch (sim>10, full>25). Knob có tác dụng saur26 không thể được pilotr20 xác nhận đầyđủ.

Gate cũ: B13–B18 thường yêu cầu gain≥+.002 **cả2seed**; B21 noninferiority TB5round≥−.0015,
endpoint≥−.003. Gate mới owner chấp nhận performance−.015 với mục tiêu giảmcost nên **không tự bê gatecũ**.
Phải chốt cách tính ranking/tie, comparator từng bậc và budget trước launch.
Seed42 chỉ là screening theo yêu cầu, không kết luận robust/statisticalsignificance.

## 11. Hạ tầng, artifact và thao tác cần giữ đúng

- Python dự án: `/home/odixe/Tools/miniconda3/envs/nckh/bin/python`; localRTX3050sm86/4GiB,
  RAM≈15GiB; Kaggle2×T4sm75/15GBmỗiGPU. Thông số local không chứng minh hành viT4.
- Hai GPU chứa hai shard client; không đồng bộ gradient giữa client. Groupedmodel độc lập.
- Checkpoint resume cần privatebank/BN/teacher/method-driverRNG/torchCPU-CUDARNG **cả2rank cùnground**;
  baseline lịch sử có thể thiếu torchRNG: báo scope, không gọi exactresume.
- Audited CM/support→10metric không thay storedprediction→CMreplay. Dataset chưa full-contentfingerprint.
  Không gọi PASS_BOUNDED là toàn bộmodel/datasetđãchứngminh.
- Source notebookđãpush giữ frozen; fix checker không trainlại hoặc đổi tolerance để tìmPASS.
  COMPLETE/exit0 có thể partial; phải đọc round/state/CM thực.
- [scripts index](scripts/README.md) chỉ entrypoint theo study; không chạy launcher nghiên cứuđóng từ tênfile.
- Kagglepull dùng exact `owner/slug/version`, CLIaccount-scoped; poll1200s bằng localworkerqua
  `codex-long-jobs`; tránhcollectortrùng. Job sống/owner chỉ trong hai `continue.md`, không trong brief.
- Git/DVCpush/cleanup không suy ra từ quyền training; không xóaartifact/cache. DVCworkspacepartial
  không `dvc add` ghiđè manifestđủ. Không đưa credential/notebookinlinekey vào Git/docs.

## 12. Đầu ra mong muốn từ Fable 5.2

1. Đọc yêu cầu mới, xác định constraint cứng và mục tiêu tối ưu; đối chiếu T3 với đủbaseline theo từngmodel.
2. Đưa tối đa2–3hướng có căn cứ, mỗi hướng có **bằng chứng → cơ chế → khác nghiên cứu đãthử ởđâu
   → lợi/cost kỳvọng → dấuhiệubácbỏ**. Không chỉ liệtkê thuậttoán hoặc đổi tênnhánhFAIL.
3. Với hướngưu tiên: mô tả message/schema, bootstrap/round, aggregation, lossvàschedule,
   cách giữ arbitrarymodelheterogeneity, compute/memory/client/server và tổngbyte50round.
4. Phân biệt giảm thuậttoán/FLOPs với tốiưu engine/walltime; chỉ rõ khoảnchi phí chưađo.
5. Đề xuất ablation tối thiểu và matchedcontrol cho seed42 từngmodel, sim→pilot→full,
   gate trước run theo mục tiêu0.015/top5; không đổi protocol riêngMobiHet.
6. Nêu giới hạn: data/contextshortcut, lớpTD/benign, testđãđọclặp, mixedmodelchưađo,
   privacychưatriểnkhai; chỉ yêu cầu owner chốt điểm thực sự thiếu.

**Việc tay hiện tại:** không cần để đọc/tổnghợp repo. Nếu Fable không truy cập filesystem,
đính kèm file này; khi cần bằng chứngchi tiết, đính kèm tài liệu mà từnglinktrỏtới
(ưu tiên comparison.json, method, protocol, tried-methods-index, LESSONS).
Link relative không tự mang nội dung repo sang chatkhác.

Đã chốt pFedES cuối: seed41 BERT Macro-F1 .873799410; mean3seed .873955730±.001220923.
[Audit receipt](../fl_veremi_bert/docs/research/pfedes-final-audit.json) giữ full10metric/source/parent/state và giới hạn.
Vận hành cũ đã [archive](archive/operations-20261116/README.md); scripts còn dùng theo [index](scripts/README.md).
