# Đo compute chung — useful vs executed (18/11/2026)

Owner 18/11: đo compute **cùng một cách cho MobiHet và mọi baseline**, làm song song với FB-L. Hợp đồng đo gốc: [task brief](../../task_brief.md) §8.

## Hai thước, cùng đơn vị

- **Useful** = dòng method thật sự train (thật + synthetic + floor) / dòng thật × FLOP/dòng FedAvg đo bằng `FlopCounterMode`
  (13,297 / 40,022 MFLOP DAGSNet / BERT). Baseline đo trên toy không pad (`../fl_veremi_bert/scripts/cost.py`) là cùng thước này.
  Gate ≤ 1,01× hiện khóa trên thước useful.
- **Executed** = dòng engine grouped thực thi, gồm client pad và dòng bị mask (`core/local.py`: mỗi bước chạy Gfix client ×
  (slot + số synthetic lớn nhất của round)), chia cho FedAvg executed trong **cùng engine** (1,087 dòng/dòng thật do pad nhóm).
  Đây là thứ `t_train`/GPU-giây phản ánh.

`scripts/compute_executed.py` (cỡ xe full-scale, batch 64, group_rows 4096, nửa lịch ở cột muộn):

| arm | useful | executed, engine hiện tại | executed nếu engine bỏ dòng mask theo nhóm |
|---|---:|---:|---:|
| FedAvg | 1,000 | 1,000 | 1,000 |
| T3 | 1,655 | 2,065 | 1,681 |
| Lite2 B+U1 | 1,365 | 1,807 | 1,394 |
| FB N4 / FB-L ×4 | 1,000 | **2,065** | 1,029 → **0,997** với `compact` |
| FB N2 / FB-L ×2 | 1,003 | **1,807** | 1,034 → **1,001** với `compact` |

Cột cuối từ 19/11 = đúng knob `compact` (nén nhóm + bước floor không slot thật; script đã sửa theo). T3/Lite2 không có knob này.

## Hệ quả

- Engine hiện **không hiện thực** phần tiết kiệm của FB: dòng thật ngoài ngân sách và synthetic vượt liều xe vẫn chạy rồi bị mask
  (synthetic rộng `aux` = max của round cho mọi xe). Khớp `t_train` FB ≈ T3. Nói "1,0× FedAvg" chỉ đúng trên thước useful.
- Engine nén theo nhóm (nhóm chỉ chạy số dòng thật/synthetic lớn nhất mà xe trong nhóm cần) đưa FB về ≈ 1,03× executed;
  phần dư còn lại là bước floor của xe nhỏ chạy cả client pad. Bản `compact` đang sửa (dưới); thay đổi engine cần test tương đương (dropout RNG
  đổi theo shape ⇒ tolerance, không bit-exact) và skill `federated-training-speed`.
- Công bằng: xếp hạng compute trên **useful** cho cả roster (baseline đo không pad), báo kèm **executed cùng engine** cho MobiHet và FedAvg.
  Không trộn useful của method này với executed của method khác.

## Kiểm chứng 18/11

- FlopCounterMode một bước FB N4 xe M/L (`_loss_tier` fwd+bwd) / bước FedAvg cùng slot = **2,000×** (useful (37+16)/64 = 0,828).
- Wall-clock train r1–20 BERT: FedAvg 3,26 h, FB N2 6,17 h (1,89×), T3 5,95 h.
- Biến thể engine (`runs/sim-kaggle-fbl/debug/variants.py`), so FedAvg cùng engine: nén nhóm N4 1,029 / N2 1,034;
  + bước floor không slot thật **0,997 / 1,001**; + bỏ client hết dòng (so FedAvg lý tưởng 1,005 dòng/dòng thật) 1,006 / 1,009.
- `compact` (knob `mobihet_budget`) đã code mức "nén nhóm + floor không slot thật". "Lệch gradient 3,5 %" 18/11 **không phải bug**
  (19/11): 1 phần tử ReLU sát 0 lật dấu ở fp32 khi shape đổi; fp64 hai đường trùng 1e-16 ([LESSONS](../../LESSONS.md)).
  Test tương đương từng bước ở fp64: `random`/`top`/`prop`/`top`+q, mỗi bước có floor lẫn dòng thật đổi thứ tự.

## Đo trong engine thật — 19/11/2026 (cùng bộ đếm cho FedAvg và MobiHet)

`scripts/count_executed.py` bọc `grouped.TRAIN` chỉ trong `local_epoch` (loại compile gate, eval, server/bootstrap), đếm mỗi round
dòng thực thi (Gfix × độ rộng, gồm client pad và dòng mask) và dòng train (mask đúng). Panel sim 200 xe, `group_rows=4096`, 2 round
compiled, GPU local RTX 3050 (`runs/compute-executed-20261119/`, `run.sh`). Không là kết quả performance.

| arm | executed / dòng thật | × FedAvg executed | useful / dòng thật | `t_train` r2 |
|---|---:|---:|---:|---:|
| FedAvg | 3,7684 | 1,000 | 1,000 | 57 s |
| FB N4, mask | 7,6034 | **2,0177** | 0,9742 | 100 s |
| FB N4, `compact` | 3,4499 | **0,9155** | 0,9742 | 61 s |
| FB-L L4 (top), `compact` | 3,4499 | **0,9155** | 0,9742 | 62 s |

- Số đếm engine **trùng harness** `compute_executed.py --panel 200` tới 4 chữ số (FedAvg 3,768; N4 mask 2,0177; compact 0,9155; useful 0,974)
  ⇒ harness full-scale (0,997 / 1,001) là số executed của engine thật, không chỉ ước lượng.
- Panel 200 xe pad rất nặng (FedAvg 3,77 dòng/dòng thật vs 1,087 full-scale) nên tỷ lệ panel không thay số full-scale.
- `t_train` compact ≈ 1,07–1,09× FedAvg dù dòng 0,92×: phần dư là việc ngoài model (sinh synthetic, top-k, Q, Python), không phải FLOP model.
  Mask → compact giảm `t_train` 39 %. FeatureBERT: dòng thực thi giống hệt (cùng vòng engine, cùng method quyết định độ rộng; FLOP/dòng hằng).
- FeatureBERT cùng panel, compiled (`bert.sh`; method byte-identical `a4147a1b…`): `t_train` r2 FedAvg **121 s**, FB-L Q4 `compact` **119 s**
  (0,98×), Q4 mask 247 s (smoke 18/11) ⇒ BERT compute-bound, wall-clock bám số dòng.

## Tổng compute toàn roster — xong 20/11/2026 00:45 (owner 19/11)

Owner chốt: **FLOP mở rộng + GPU-time**, toàn roster (FedAvg + 13 baseline + MobiHet T3 / FB N4c, N2c / FB-L L4, P4, Q4, L2, P2, Q2)
× DAGSNet + FeatureBERT, panel 200 thật, eval báo riêng (không cộng). Công cụ: `scripts/op_counter.py` (mm = công thức
`torch.utils.flop_counter`, khớp FlopCounterMode 13,297 MFLOP/dòng; op khác = 1/phần tử, reduction theo phần tử đọc), `scripts/total_compute.py`
(bootstrap = constructor; train = `local_epoch`; ngoài train = phần còn lại của round: server, codec, packet, prototype; eval),
`scripts/total_compute_report.py`. Hàng đợi `runs/total-compute-20261119/queue.py` (count eager + time compiled, 2 round, round 2 = steady);
baseline DAGSNet ở `../fl_dagsnet_veremi_v2 --panel 200`, BERT qua `sim_fold_val.py`. Bảng đầy đủ: `runs/total-compute-20261119/report.md`.

**93/95 job đo; `real_rows` = 1.476.064 mọi job.** Top = 1e12 op; tổng FL = bootstrap + 50 × (train + ngoài train); GPU-s 50r =
cùng công thức theo wall-clock compiled trên RTX 3050 4 GB local (group_rows 4096); time × FedAvg so FedAvg cùng repo/engine.
Count eager OOM 4 GB ⇒ đếm ở group_rows 2048/1024 (+ `eval_rows` 8192, chunk `class_sums` theo nó), op train quy đổi về dòng executed
engine 4096 (op/dòng không phụ thuộc shape): toàn bộ BERT, DAGSNet NILM-FL.

| method | DAGSNet FL × FedAvg | DAGSNet time × | BERT FL × FedAvg | BERT time × | ghi chú |
|---|---:|---:|---:|---:|---|
| LwFedNIDS | 0,107 | 0,32 | 0,205 | 0,76 | model pruned |
| **FB-L L4** / P4 / Q4, FB N4c | **0,920** | 1,08–1,09 | **0,917** | 0,98 | chọn **L4** (owner 19/11) |
| FB-L L2 / P2 / Q2, FB N2c | 0,963 | 1,10–1,11 | 0,960 | 1,02 | |
| FedAvg | 1,000 | 1,00 | 1,000 | 1,00 | DAGSNet: repo MobiHet và v2 trùng FLOP (2835 / 2813 GPU-s) |
| FedProto / FedProx / AFPHA | 1,000–1,001 | 1,02–1,11 | 1,000 | 1,00–1,01 | |
| TinyProto / FedTGP / V2X-HtFL | 1,10–1,14 | 1,14–1,43 | 1,10 | 1,12–1,15 | pass prototype ngoài train ≈ 8 / 23 Top/round |
| CANPerFL | ≈ 1,20 | ≈ 1,20 | ≈ 1,20 | ≈ 1,20 | **suy ra**, không đo: FedAvg 50r + 10 epoch fine-tune `local_epoch` giống hệt |
| FD-IDS | 1,325 | 1,33 | 1,319 | 1,29 | forward teacher (`aux_executed`) |
| PerFed-SKD | 1,426 | 1,37 | 1,415 | 1,40 | |
| APFed / NILM-FL | 1,99–2,00 | 2,01–2,15 | 2,000 | 1,99–2,01 | |
| MobiHet T3 (mask) | 2,008 | 1,92 | 2,019 | 2,13 | |
| pFedES | 2,638 | 2,08 | 2,751 | 2,79 | |

- Hạng FLOP: FB-L L4 **thứ 2 / 16** (FedAvg + 13 baseline + T3 + L4) cả hai model (sau LwFedNIDS, model nhỏ hơn); thấp hơn FedAvg 8 %. Bootstrap + server MobiHet ≈ 0,1 % một round train.
- Wall-clock local: DAGSNet FB-L 1,08× (launch-bound ở panel nhỏ: compact giảm FLOP không giảm thời gian), BERT 0,98×. Panel pad nặng
  (FedAvg DAGSNet 3,77 dòng executed/dòng thật) ⇒ tỉ lệ panel ≠ full-scale; claim full-scale dùng executed engine 0,997× / 1,001× (mục trên).
- CANPerFL không đo trực tiếp: method yêu cầu state FedAvg r50 thật (`hp.init`, assert round 50); tạo state giả = provenance giả.
- Sửa harness trong lúc đo (19/11): forward không mask → `aux_executed` (fd_ids, pfedes); chunk `class_sums` (fedtgp OOM);
  `knowledge/clients.parquet` từ DVC (afpha). BERT N2c trước ghi 0,819× vì chưa có time run để quy đổi group_rows → nay 0,960×.
  `bert-nilm time` trùng pytest 19:57 ⇒ đo lại (JSON cũ `*.perturbed-by-pytest-1957.json`). BERT T3 count dùng chunk `class_sums` 32 768
  (trước bản sửa) — chỉ khác padding chunk.

## Việc còn lại

1. Baseline executed trong engine thật (pad nhóm, client mất bước) cho roster BERT — cùng hàm đếm, đọc hp từng baseline.
2. ~~Sửa `compact`, đo `t_train`~~ xong 19/11 (trên). Owner chốt có chạy FB-L với `compact` + gate executed không.
3. Server/bootstrap: đo trong roster (ngoài train/round, bootstrap) — ≈ 0,1 % train với MobiHet; brief §7 xong cho FLOP.
