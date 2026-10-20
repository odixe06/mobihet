# So sánh với `fl_dagsnet_veremi` — truyền tin, tính toán, performance

Cập nhật 20/10/2026. So MobiHet với 7 phương pháp dựng lại trong [`odixe06/fl_dagsnet_veremi`](https://github.com/odixe06/fl_dagsnet_veremi)
(commit `f9014e1`, bản clone `../fl_dagsnet_veremi`, số đọc từ `report/<method>/report.md` và CSV của repo đó) theo ba tiêu chí.
Cùng dữ liệu VeReMi 80/20 (train 43.045.415 / test 10.761.343 dòng), Dirichlet α 0,5, DAGSNet, seed 42; kịch bản **20 client**.

## Đọc trước mọi con số

- **Phương pháp hiện tại của MobiHet (FullDrift `gkd_la_kdpx`) CHƯA chạy cùng điều kiện.** Số của nó là **probe**: train 80 % trừ holdout
  nội bộ (40,0M dòng, 93 %), 15 round, test20 chỉ chấm ở round cuối. Repo kia: train toàn bộ 80 %, 50 round, test20 mọi round, headline r50.
  Probe chỉ dùng để chọn cấu hình (cổng trên holdout nội bộ). Owner 20/10: vét xong mọi khả năng thì chạy các kịch bản cùng điều kiện,
  không holdout (`fd.py --final`, [design §Run cuối](../methods/full-drift/design.md)).
- **Cùng điều kiện hiện chỉ có `h_dml`** (DML proxy + private, phương pháp MobiHet cũ, 50 round, toàn bộ train80).
- **Model được chấm khác nhau:** AFPHA, FD-IDS, LwFedNIDS chấm **một model toàn cục**. NILM-FL, PerFed-SKD, pFedES, TinyProto và MobiHet
  private chấm **từng model client trên toàn bộ test rồi lấy trung bình** 20 client (cùng cách tính với MobiHet).
- Repo kia chạy một seed; chênh lệch dưới ~0,01 F1 macro không phân biệt được với nhiễu seed (MobiHet đo s42/s43 lệch 0,008).

## 1. Performance — đủ 10 metric, 20 client

Thứ tự cột: accuracy, precision macro / micro / weighted, recall macro / micro / weighted, F1 macro / micro / weighted.
F1 tệ nhất = F1 macro của client kém nhất (chỉ phương pháp có model client).

| phương pháp | model chấm | round | acc | P mac | P mic | P wtd | R mac | R mic | R wtd | **F1 mac** | F1 mic | F1 wtd | F1 tệ nhất |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AFPHA | toàn cục | 50 | 0,7487 | 0,7781 | 0,7487 | 0,7986 | 0,8015 | 0,7487 | 0,7487 | **0,7752** | 0,7487 | 0,7481 | — |
| FD-IDS | toàn cục | 50 | 0,7327 | 0,7837 | 0,7327 | 0,7971 | 0,7887 | 0,7327 | 0,7327 | **0,7655** | 0,7327 | 0,7297 | — |
| LwFedNIDS | toàn cục | 50 | 0,7791 | 0,8080 | 0,7791 | 0,8085 | 0,7950 | 0,7791 | 0,7791 | **0,7917** | 0,7791 | 0,7853 | — |
| NILM-FL | TB client | 50 | 0,6805 | 0,7842 | 0,6805 | 0,7966 | 0,7311 | 0,6805 | 0,6805 | **0,6963** | 0,6805 | 0,6674 | 0,5257 |
| PerFed-SKD | TB client | 50 | 0,6229 | 0,7022 | 0,6229 | 0,7446 | 0,7114 | 0,6229 | 0,6229 | **0,6504** | 0,6229 | 0,6077 | 0,5578 |
| pFedES | TB client | 50 | 0,5276 | 0,5921 | 0,5276 | 0,6805 | 0,5656 | 0,5276 | 0,5276 | **0,4856** | 0,5276 | 0,5054 | 0,2817 |
| TinyProto (luật proto) | TB client | 50 | 0,5639 | 0,6852 | 0,5639 | 0,7283 | 0,6349 | 0,5639 | 0,5639 | **0,6212** | 0,5639 | 0,5972 | 0,5428 |
| MobiHet `h_dml` private (luật gp) — cùng điều kiện | TB client | 50 | 0,7097 | 0,8011 | 0,7097 | 0,8057 | 0,7399 | 0,7097 | 0,7097 | **0,7363** | 0,7097 | 0,7177 | — |
| MobiHet `h_dml` proxy — cùng điều kiện | toàn cục | 50 | 0,7932 | 0,8076 | 0,7932 | 0,8225 | 0,8143 | 0,7932 | 0,7932 | **0,7994** | 0,7932 | 0,7976 | — |
| *Probe* `gkd` (rd) private | TB client | 50 | 0,8392 | 0,8647 | 0,8392 | 0,8559 | 0,8013 | 0,8392 | 0,8392 | **0,8063** | 0,8392 | 0,8284 | 0,7771 |
| *Probe* `gkd_la_kdpx` s42 private | TB client | 15 | 0,8690 | 0,8867 | 0,8690 | 0,8632 | 0,7987 | 0,8690 | 0,8690 | **0,8229** | 0,8690 | 0,8506 | 0,8108 |
| *Probe* `gkd_la_kdpx` s43 private | TB client | 15 | 0,8663 | 0,8891 | 0,8663 | 0,8635 | 0,7887 | 0,8663 | 0,8663 | **0,8145** | 0,8663 | 0,8466 | 0,8003 |
| *Probe* `gkd_la_kdpx` + `q4_zl` s42 private | TB client | 15 | 0,8700 | 0,8861 | 0,8700 | 0,8627 | 0,7979 | 0,8700 | 0,8700 | **0,8226** | 0,8700 | 0,8510 | 0,8154 |
| *Probe* `gkd_la_kdpx` proxy s42 | toàn cục | 15 | 0,8763 | 0,8982 | 0,8763 | 0,8724 | 0,8062 | 0,8763 | 0,8763 | **0,8327** | 0,8763 | 0,8590 | — |

Nguồn MobiHet: [`h_dml` r50](../methods/shared-baselines/builds/federated-baselines-full/results/latest-rounds.md);
probe [FullDrift metrics-all.csv](../methods/full-drift/results/metrics-all.csv) (test, luật head); F1 tệ nhất tính từ
`confusion_clients` trong `result.json`. Tham chiếu tập trung (không FL): EDL-CMSO F1 macro 0,853 (`../fl_dagsnet_veremi/edl_cmso`).

Đọc:
- Probe `gkd_la_kdpx` (0,82 F1 macro sau 15 round, client tệ nhất 0,80–0,81) cao hơn mọi phương pháp của repo kia ở r50 — tốt nhất bên kia là
  LwFedNIDS 0,792 (model toàn cục) và NILM-FL 0,696 (model client). So model client với model client: +0,13 F1 macro so NILM-FL; client tệ nhất 0,81 so 0,54 (tốt nhất bên kia, TinyProto).
  **Chưa phải kết luận**: khác điều kiện (15 round, 93 % dữ liệu train, cấu hình chọn bằng holdout nội bộ).
- Precision micro = recall micro = accuracy ở mọi dòng (bài toán đơn nhãn) — vẫn báo đủ 10 cột.

## 2. Communication cost — 20 client, 50 round

Repo kia **tính** byte từ giao thức (số tham số × 4 B fp32), không đo trên dây. MobiHet **đo** byte trên dây (độ dài gói thật, gồm header),
cộng cả bootstrap và moment Gaussian một lần. Số MobiHet ghi "ước" = trung bình byte/round đo trong 15 round × 50.

| phương pháp | trao đổi mỗi round | mỗi client / round (lên + xuống) | tổng 50 round, 20 client | so FedAvg |
|---|---|---:|---:|---:|
| AFPHA, FD-IDS (FedAvg/FedProx) | model DAGSNet fp32 (395.024 tham số + BN) | 3,19 MB | 3.187 MB | 1× |
| NILM-FL | proxy w_r | 3,19 MB | 3.187 MB | 1× |
| pFedES | extractor G (407.874 tham số) | 3,26 MB | 3.263 MB | 1,02× |
| PerFed-SKD | model, chỉ client được chọn (\|S\|/M ≈ 0,38) | ≈0,61 MB (TB) | 608 MB (580 MiB, báo cáo) | 0,19× |
| LwFedNIDS | model đã tỉa 70 % kênh (35.891 tham số) | 0,30 MB | 295 MB (281,5 MiB, báo cáo) | 0,093× |
| TinyProto | prototype thưa (s = 50) | 6,4 kB | 6,4 MB | 0,002× |
| MobiHet `h_dml` / `gkd_la_kdpx` dense | proxy fp32 + bootstrap + moment Gaussian | 3,19 MB | ≈3.220 MB (ước) | 1,01× |
| MobiHet `gkd_la_kdpx` + `q4_zl` | proxy int4 + LZMA2 | 0,33 MB | **≈368 MB** (ước) | 0,12× |
| MobiHet `gkd_la_kdpx` + `q3_zl` | proxy 3-bit + LZMA2 | 0,22 MB | **≈259 MB** (ước) | 0,081× |

- Phần một lần của MobiHet: bootstrap proxy 31,9 MB (state ban đầu gửi mỗi client) + moment Gaussian 4,4 MB. Bootstrap là init từ seed,
  client tự dựng được ⇒ có thể bỏ, trùng bit ([design §Vòng 22](../methods/full-drift/design.md)); khi đó q3_zl ≈ 227 MB.
- Vòng 23 đang thử 2-bit với bước RMS (byte proxy dự kiến −32 % so q3). Byte của các kênh nén đã kiểm không giảm performance trên probe.
- TinyProto và PerFed-SKD rẻ hơn MobiHet về byte nhưng F1 macro client thấp hơn 0,20 và 0,17.

## 3. Computational cost

Chi tiết Big-O, FLOPs và cách đo: [computational-cost.md](computational-cost.md). c = số lượt forward-tương-đương mỗi mẫu (train một model = 3).
GPU-phút/round = train, không gồm eval, 2×T4, quy về 43,05M dòng.

| phương pháp | c | GPU-phút train / round | so AFPHA | round trung vị gồm eval (s) | tổng 50 round (h) |
|---|---:|---:|---:|---:|---:|
| LwFedNIDS | 0,27 | 8,3 | 0,76× | 254 | 3,6 |
| AFPHA | 3 | 10,9 | 1 | 343 | 4,8 |
| PerFed-SKD | 4 (+eval chọn client) | 11,6 (+8,2) | 1,06× | 593 | 8,3 |
| TinyProto | 3–4 | 13,6 | 1,25× | — | — |
| FD-IDS | 4 | 16,0 | 1,47× | 510 | 7,2 |
| NILM-FL | 6 | 18,9 | 1,74× | 862 | 12,2 |
| pFedES | ≈12 | 34,9 | 3,2× | 1.320 | 18,2 |
| MobiHet `h_dml` (code cũ) | 6 | 32,3 | 2,96× | — | — |
| **MobiHet `gkd_la_kdpx` `--fast`** | 7,5 + forward teacher | **26,4** | 2,4× | 772 (probe, eval holdout) | ước 16 (run cuối, gồm eval test20) |

- MobiHet `--fast`: 18,75 ms/bước/tiến trình, train trung vị 736 s/round wall trên 2×T4 cho 40,0M dòng ⇒ 2 × 736 / 60 × 1,076 = 26,4 GPU-phút.
- Server mọi phương pháp không đáng kể (O(K·P)). Suy luận trên xe: một DAGSNet (4,44 MFLOPs/mẫu) như mọi phương pháp DAGSNet; LwFedNIDS ≈ 0,4.
- Eval theo protocol (không phải chi phí phương pháp): model client phải chấm N model trên test mỗi round; run cuối MobiHet chấm private, private_ema
  và proxy ⇒ ước ~7 phút/round.

## 4. Tổng hợp và việc còn

| | performance (F1 macro client) | communication (50 round) | computation |
|---|---|---|---|
| tốt nhất repo kia | LwFedNIDS 0,792 (toàn cục), NILM-FL 0,696 (client) | TinyProto 6,4 MB | LwFedNIDS 8,3 GPU-phút/round |
| MobiHet hiện tại (probe) | 0,82 (s42), 0,81 (s43) | 259–368 MB | 26,4 GPU-phút/round |

MobiHet đổi tính toán (≈2,4× AFPHA, thấp hơn pFedES) lấy performance client; truyền tin bằng ~8 % FedAvg nhờ kênh nén. Việc còn để so
đúng: run cuối `--final` (toàn bộ train80, test20 mọi round, 50 round) cho cấu hình chốt + đối chứng — owner duyệt kịch bản trước khi chạy.
