# Run cuối (final) — contract train/chấm tách, source, test, cost

[Handoff hiện tại](continue.md) · [Kết quả chọn cấu hình](../methods/full-drift/results.md) · [Protocol](../research/protocol.md).

## Contract đã chốt

Hai arm: `fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl` và đối chứng `fd_dml_m4b_gpool_frz_gkd`.
q3_zl PASS cả seed42/43; 2-bit loại. Seed42 trước, seed43 chờ hoàn tất seed42.
Mỗi kịch bản 50round × 1epoch toàn train80%, cosineT50; toàn test20% chấm mỗi round, headline r50.
Giữ DAGSNet/dataset/scaler/partition, `--fast`; không dùng official test chọn HP/round.

| clients | batch | session train cap | train/round dự kiến | chấm/round dự kiến, 2GPU |
|---|---:|---:|---:|---:|
| 20 | 512 | 4round | 13–17phút | 11–15phút |
| 50 | 512 | 2round | 14–20phút | 27–37phút |
| 100 | 256 | 2round | 27–38phút | 54–74phút |

Ước lượng từ bốn run probe đã audit: train median736–837s, max810–959s trên~40M dòng; test ep15 cũ949–981s.
Ngoại suy43,05M train và parallel EMA; **chưa benchmark final50/100c**. Setup5–15phút cũ đã bị bác:
artifact r24 đo prepare20c1797s (~30phút),100c2712s (~45phút), chưa gồm mọi compile/startup.
[Timing đã audit](evidence/final-2026-10-20/r24-prepare-timing.json). ETA toàn chiến dịch200–300session-hours cũ chưa tính đúng
repeated preparation cho hàng trăm notebook chấm; cần tính lại, không coi đủ quota/đã tối ưu.
Giữ source r25 đã push; phiên sau ưu tiên đo/giảm repeated decode bằng fixture cache có provenance, kiểm tương đương
trước tag vận hành mới. Query quota live/reset trước mỗi batch.

Train commit weights/BN/resume từng round, không chấm inline. Terminal ⇒ pull/audit, một round/notebook chấm trên account khác;
2GPU chia client và EMA, proxy rank0. Evaluator có heartbeat tiến triển ⇒ được train tiếp song song từ parent committed.
Batch đầu16notebook chấm; tối đa2GPU session/account, ưu tiên round seed42 chưa chấm. Báo riêng train completion và eval completion.

## Source và kiểm chứng

Source mới `20261015-r25`; giữ nguyên r24 đã push. So r24, chỉ source `fd.py` đổi;40module legacy không đổi.
[Local proof](evidence/final-2026-10-20/final-recovery-local-2026-10-20.json).

- [fd.py](../../methods/full-drift/builds/fd-probe/source/proj/fd.py): final fixture context, K20/50/100, defer_eval/session cap,
  weights/resume/codec mỗi round, fingerprint; [fd_prepare.py](../../methods/full-drift/builds/fd-probe/source/proj/fd_prepare.py): partition đúngK, toàn train80%.
- [fd_eval.py](../../methods/full-drift/builds/fd-probe/source/proj/fd_eval.py): strict395024params, weights_only/SHA, private/EMA/proxy,
  prediction→confusion→10metrics; giữ đúng EMA BN calibration.
- [test_final_startup.py](../../methods/full-drift/builds/fd-probe/scripts/test_final_startup.py): tái hiện legacy crash trên final-only filesystem,
  adapter PASS, pilot context/vector exact; actual notebook cell childrc7 phải fail, rc0 pass.
  Non-smoke hai arm×K20/50/100×rank0/1 PASS, batch production và tail; tiny fixture1023/client,test256,
  chỉ thay full-row coverage gate trong test. Không sửa gate production.
- [test_final_pipeline.py](../../methods/full-drift/builds/fd-probe/scripts/test_final_pipeline.py): compiled20c PASS,
  weights khớp r23; q3 resume giống uninterrupted; detached eval1/2worker predictions/10metrics exact; SHA sai bị từ chối.
  Local một RTX3050; chưa thay bằng chứng physical2T4.
- [gen_fd.py](../../methods/full-drift/builds/fd-probe/scripts/gen_fd.py): notebook private0600, public không secret, source/hash guard;
  childrc lỗi phải raise notebook. [gen_eval.py](../../methods/full-drift/builds/fd-probe/scripts/gen_eval.py),
  [gen_continue.py](../../methods/full-drift/builds/fd-probe/scripts/gen_continue.py) chỉ chuẩn bị, không tự push.
- [collect_fd.py](../../methods/full-drift/builds/fd-probe/scripts/collect_fd.py),
  [collect_eval.py](../../methods/full-drift/builds/fd-probe/scripts/collect_eval.py): exact executed source/input/fingerprint/checkpoint/metrics;
  [report_final.py](../../methods/full-drift/builds/fd-probe/scripts/report_final.py): chỉ join artifact đã audit, báo missing rounds và completion.

## Lỗi đã xử lý

r24 zc/zi crash trước round 1 (Gaussian hash cứng pilot manifest trong runtime final-only; notebook không propagate child rc nên Kaggle báo COMPLETE).
Source r25 sửa và test PASS. [Recovery r24](../archive/operations/final-recovery-2026-10-20.md) · [sổ lỗi](notebook-errors.md).
Kernel, trạng thái live và lệnh tiếp theo chỉ ghi ở [continue](continue.md).

## Communication và computational cost

`round.cost`: serialize up/down/total bytes, codec host encode/decode seconds, applied/skipped steps,
server aggregation seconds, sum client epoch walltime và rank0 peakVRAM. Bootstrap báo riêng.
`payload_bytes_per_algorithm_second` là payload/time trong harness, không phải bandwidth đường truyền vật lý.
Checkpoint/prediction I/O Kaggle là vận hành, không cộng vào FL wire cost.

FLOPs operator estimate DAGSNet: forward4443648/mẫu, train13280256/mẫu; Gaussian matvec tách riêng.
Không gồm BN/KL/optimizer/calibration/probes; walltime báo riêng. `runtime.json` ghi torch/devices/setup/compile gates.
[Chi tiết cost](../research/computational-cost.md). Mỗi round/model role có đủ10metric từ full confusion, không batch-average.

## Lộ trình nhanh nhất (đề xuất 20/10 ~19:30Z, chờ owner duyệt)

### Số đo r25 (đã audit, 6 kernel)

| kịch bản | arm | chuẩn bị dữ liệu | setup `fd.py` | round 1 / round sau | kết quả |
|---|---|---:|---:|---|---|
| 20c | q3 (za) | 1.019 s | 126 s | 943 / 838–919 s | 4 round |
| 20c | gkd (zc) | 1.566 s | 110 s | 857 / 775–784 s | 4 round |
| 50c | q3 (ze) | 4.420 s | — | — | ERROR, hết budget 2h |
| 50c | gkd (zg) | 1.114 s | 124 s | 1.032 / 810 s | 2 round |
| 100c | q3 (zi) | 8.792 s | — | — | ERROR, hết budget 3h |
| 100c | gkd (zk) | 5.090 s | 99 s | 1.515 / 1.300 s | 2 round |

Chuẩn bị dữ liệu dao động 17–147 phút, không theo số client (theo máy); session 2–3h mất 40–100 % cho chuẩn bị. 20c và 50c cùng
~84k bước/round (batch 512) nên round gần bằng nhau; 100c batch 256 ⇒ 168k bước, ~1,65×. Chưa có số đo chấm: ước 20c 11–15, 50c 27–37,
100c 54–74 phút/round trên 2 GPU (ngoại suy probe). **Đo 20/10 (za r1–4, 20c, audit PASS 10 metric): 772–805 s/round** (~41 lượt forward test20 ⇒ ~19 s/lượt ⇒ ngoại suy 100c ~63 phút, chờ zk xác nhận); **50c (zg r1–2): 1.866–1.911 s/round**; 20c gkd (zc r1–4) 792–820 s/round nhưng chuẩn bị trong session mất 7.392 s ⇒ notebook chấm luôn gắn cache; predictions ~98 MB/round 20c, ~313 MB 50c. W&B: khởi động worker ~25 s/round, mỗi client ~65 s/worker (private + EMA trên 10,76M dòng), gộp + nén 60–100 s. Tăng tốc bit-exact chỉ ~10–15 %; nhanh hơn nữa (batch lớn, cuDNN autotune) đổi số học ⇒ cần owner duyệt.

### Ba thay đổi (không đổi phương pháp, source r25, fingerprint)

1. **Session train dài ~11,5h, bỏ cap round.** `fd.py` đã tự dừng trước round không kịp (`elapsed + 1,15 × round chậm nhất + 60 s > max_hours`),
   `session_rounds` không thuộc fingerprint ⇒ chỉ đổi generator. Chuỗi train là đường găng; mỗi lần ngắt session tốn chuẩn bị + setup + hàng đợi + collect/upload.
2. **Cache fixture bằng kernel CPU (không tốn quota GPU).** Mỗi account cần dùng chạy một lần `prepare_fd(final=True, clients=K)` trên CPU, output
   ~7,3 GB; notebook GPU gắn output đó (`kernel_sources`), kiểm SHA từng file theo `manifest.json` và manifest trùng byte với r25 đã audit ⇒ fingerprint
   không đổi, nối tiếp được từ parent r25. Bỏ 17–147 phút chuẩn bị khỏi mỗi session GPU (≈ 1h × ~45 notebook).
3. **Chấm nhiều round mỗi notebook.** Notebook gọi `fd_eval.py` lần lượt cho 4–20 round (20c ~20, 50c ~8, 100c ~4–5), lấp 4–6h; không đổi `fd_eval.py`.

### Lịch

| bước | việc | thời gian (wall) |
|---|---|---|
| 0 | Sinh + test local generator mới (session dài, cache, chấm nhiều round); push kernel CPU cache cho các account sẽ dùng | ~2–3h (CPU kernel chạy song song, miễn phí GPU) |
| 1 | Train S1 cả 6 chuỗi song song (11,5h): za/zc/zg/zk nối từ round đã audit, ze/zi chạy lại từ r1. Cùng lúc chấm r1–4/r1–2 đã có (đo tốc độ chấm thật) | +11,5h |
| 2 | Collect S1 (≤ 8 GB/chuỗi 100c) → parent → train S2; chấm các round S1 trên account khác song song | +~1h, rồi +3–11,5h |
| 3 | Chấm các round S2 trải nhiều account; `report_final.py` | +~3h sau khi S2 xong |

Round mỗi session S1 (chuẩn bị ≤ 15 phút nhờ cache): 20c q3 ~41, 20c gkd ~46 (đủ tới r50), 50c gkd ~47, 50c q3 ~44, 100c gkd ~28, 100c q3 ~25
⇒ 20c/50c xong trong 1–2 session, 100c 2 session. **Train cả 6 chuỗi xong ≈ 24–26h sau bước 1**; không cache thì 100c q3 cần 3 session (~30h).

### Quota

Cần ≈ 95–110h train (gồm margin) + ≈ 180h chấm (ước, chưa đo) ≈ 280–290h. Còn 263h (20/10 19:20Z, reset 24/10 00:00Z), trong đó 60h ở odixeuit/minhtrit06
không tạo được dataset ⇒ chỉ dùng được nếu owner share dataset nguồn/checkpoint bằng web. Train S1 cần 6 account ≥ 13,2h (1,15 × 11,5h): có 10 account đủ.
⇒ Train xong trước reset; phần chấm còn lại (~70–90h nếu ước đúng) chạy sau reset 24/10 00:00Z (480h) ⇒ đủ r1–50 cả 6 chuỗi ≈ sáng 24/10.
Chấm nhanh hơn ước (đo ở bước 1) có thể kéo về trước reset. Seed 43 (×2 toàn bộ) chỉ chạy được từ tuần quota sau.

### Lịch chấm (bộ nhanh r27, số đo T4 21/10)

| kịch bản | chấm/round ổn định (2 T4) | round đầu notebook | round còn phải chấm | giờ GPU | round/notebook (~4–5h) |
|---|---:|---:|---:|---:|---:|
| 20c | 301–357 s | ~480 s | 92 (za, zc r5–50) | ~8,5h | ≤ 46 (một chuỗi/notebook) |
| 50c | 843 s | ~1.120 s | 98 (zg r3–50, ze r1–50) | ~23h | ~20 |
| 100c | 1.449 s | ~2.075 s | 98 (zk r3–50, zi r1–50) | ~40h | ~11–12 |

Tổng ≈ 72h (bộ cũ ≈ 181h) + train S2 ≈ 16h ⇒ vừa quota trước reset 24/10 00:00Z. Thứ tự: r50 (headline) mỗi chuỗi trước, rồi round tăng dần;
round có ngay khi chuỗi S1/S2 collect + audit xong, account chấm khác account train và có cache K tương ứng (marker `artifacts/fixture-cache/`).
Bảng cũ (bộ chấm `fd_eval.py`, 20/10 23:30Z): 20c 792–820 s, 50c 1.866–1.911 s, 100c 3.979–4.064 s/round.

Kéo về trước reset chỉ có 2 cách: (a) tăng tốc chấm lớp tolerance (batch lớn/cuDNN autotune, cần owner duyệt và đo trên T4);
(b) 60h của odixeuit/minhtrit06 — không tạo được dataset, mỗi dataset checkpoint phải được owner share tay ⇒ không thực tế.
_Sửa 21/10:_ cả hai tạo được dataset lại; owner chọn thêm luồng song song ⇒ minhtrit06 dựng cache K50 làm luồng chấm; odixeuit owner dùng cho dự án khác.
**21/10 owner chọn (a): "chấm nhanh nhất có thể"** ⇒ §Bộ chấm nhanh.

Công cụ nối tiếp 21/10: `gen_continue.py --hours H` (đuôi ngắn vừa quota theo luật 1,15×; `launch_continue.sh … [giờ]`); `gen_fd.FROZEN` để nối tiếp tag r25 khi source sống đã sang tag mới (stage kiểm theo manifest của chính nó).

### Bộ chấm nhanh (owner 21/10, lớp tolerance)

[fd_eval_fast.py](../../methods/full-drift/builds/fd-probe/source/proj/fd_eval_fast.py), nguồn chấm tag **r26** (= 49 file r25 + file này,
`artifacts/input/fd-src-20261015-r26`, upload `<owner>/mobihet-fulldrift-src-20261015-r26`); train/nối phiên vẫn r25 (fingerprint không đổi).
Cùng kiểm SHA/config/fixture và cùng output (`eval/`, `metrics/`, `predictions/`) như `fd_eval.py`; `collect_eval.py` kiểm SHA evaluator theo contract
và `fast_gate` (mọi model qua gate). Ba thay đổi:
1. **Forward GEMM.** DAGSNet viết lại trên (dòng, vị trí, kênh), BN eval fold vào trọng số, conv k = một GEMM ra k×out rồi cộng dịch vị trí,
   các conv 1×1 cùng đầu vào chung GEMM; compile một lần (CUDA graph, batch 16.384). Mỗi model được gate với chính module gốc trên 8.192 dòng
   đầu (sai số logit ≤ 0,1, argmax trùng ở dòng margin > 0,1). EMA recalibration, prior, luật head/gp/gpe giữ code cũ.
2. **Worker bền nhiều round**, client nhận từ hàng đợi chung (file claim) ⇒ không GPU nào chờ GPU kia (bản cũ lệch tới 6 phút/round 100c).
3. **Gộp/nén round r chạy song song khi GPU chấm round r+1** (parts không nén ở `/kaggle/temp/eval-parts`, tối đa 2 round trên đĩa).

Bằng chứng: toán học khớp module gốc ở fp32 (≤ 3,6e-7, `test_fast_eval.py`); smoke 20c/100c lệch ≤ 0,08 % dự đoán, mọi metric
< 0,02 (`test_final_pipeline.py`). Bench T4 (`odixe0502/mobihet-fd-evalbench-20261021-v1`, za r1, 10,76M dòng, đường "hiện tại" trùng bit
dự đoán đã audit):

| forward | s/lượt (model A / B) | dòng đổi argmax | margin lớn nhất của dòng đổi |
|---|---:|---:|---:|
| module compiled hiện tại (16k) | 29,9 / 34,3 | — | — |
| GEMM im2col (16k/64k) | 22,3–24,6 | 2.648 / 1.810 | 0,027 |
| **GEMM + cộng dịch (16k)** | **19,2 / 17,8** | 2.588 / 1.839 (0,02 %) | 0,043 |

Chấm lại r1–4 za/zc, r1–2 zg/zk bằng bộ nhanh (tag r26) rồi `compare_eval.py` so với bản đã audit ⇒ `EQUIVALENCE.json` mỗi notebook; các
round này dùng kết quả bộ nhanh để cả chiến dịch cùng một evaluator. Kết quả T4 (collect + audit PASS, 21/10):

| notebook (`…-f1`) | round đầu (gồm khởi động) | round sau (chạy chồng) | bản cũ | khớp dự đoán | lệch metric lớn nhất | sai số gate lớn nhất |
|---|---:|---:|---:|---:|---:|---:|
| za 20c q3 r1–4 (odixe0502) | 471 s | 301–343 s | 772–805 s | 99,973 % | 2,7e-4 | 0,065 |
| zc 20c gkd r1–4 (minhtran0601) | 485 s | 333–357 s | 792–820 s | 99,972 % | 5,3e-4 | 0,077 |
| zg 50c gkd r1–2 (dinhan25) | 1.121 s | 843 s | 1.866–1.911 s | 99,964 % | 1,6e-3 | 0,062 |
| zk 100c gkd r1–2 (trietbackup) | 2.075 s | 1.449 s | 3.979–4.064 s | 99,964 % | 9,2e-5 | 0,047 |

Lệch dự đoán dồn ở EMA/`gp`, dòng sát nút. Lệch metric lớn nhất (zg r2, private `gp` precision_macro) là bước nhảy của precision lớp không được
dự đoán: client 22 đổi đúng 1 dòng thành lớp 15 (0 → 1 dự đoán đúng) ⇒ precision lớp đó 0 → 1, +1/16 ÷ 50 client. Không phải sai số tích lũy.
**Tag r27** (chấm đại trà): chỉ đổi ngưỡng gate 0,1 → 0,25 (sai số và margin dòng chắc chắn) vì đo được tới 0,077 trên vài trăm model mà chiến dịch
có ~20 nghìn lượt model; nạp sai trọng số cho lệch O(1) vẫn bị chặn. Dự đoán giống r26 (cùng code tính).

**Tag r29** (owner duyệt 21/10 ~10:30Z): gate tuyệt đối 0,25 vỡ ở round muộn — `dinhan25/…-e29-48-s42ze…` ERROR r34 50c, sai số 0,2508.
Logit tăng ~3× r5 → r48 (max|logit| 15–22 → 37–58) nên sai số fp16 tăng theo; đo local 255 model ze r5/r20/r33/r34/r48: sai số tương đối
max 0,21 → 0,48 %, 0 dòng chắc chắn đổi argmax (T4 r34 ≈ 0,45–0,73 %). Gate mới: `max_error / max|logit gốc| ≤ 2 %` + argmax trùng ở dòng
margin > 0,25 (giữ); `fast_gate` ghi thêm `max_rel`. Nạp sai trọng số lệch O(10 %) ⇒ vẫn bị chặn. Dự đoán giống r27 (cùng code tính).
`collect_eval.py` nhận r27 (tuyệt đối ≤ 0,25) và r29 (tương đối ≤ 2 %), `--partial` audit các round đã commit của notebook ERROR
(`rounds` = round đã audit, `planned_rounds` = kế hoạch). Test: `test_fast_eval.py` (fp32 3,6e-7), `test_final_pipeline.py` 20c (max_rel 0,13 %).
`report_final.py` sửa cùng ngày: đọc proof nhiều round (`rounds`) — trước đó chỉ nhận proof một round nên bỏ sót mọi kết quả bộ nhanh;
round chấm cả bộ cũ và bộ nhanh lấy bộ nhanh.
