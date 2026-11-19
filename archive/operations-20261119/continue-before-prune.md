> **Snapshot 19/11/2026 ≈ 07:10 +07 trước khi rút gọn bàn giao — lịch sử, không phải trạng thái sống.** Trạng thái sống: [continue](../../docs/operations/continue.md).

# Bàn giao sống — 19/11/2026 (phiên Claude mới: sửa xong `compact`; trước đó 18/11 ≈ 11:00 +07)

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [bài học vận hành](../../docs/operations/lessons.md).

## FB-L sim compact — XONG 19/11 (FBL_DONE; watcher đã thoát; 17/17 audit PASS; 6/6 PASS hai model, chọn P4; pilot chờ owner)

17 kernel version 1, tất cả RUNNING lúc 04:48 (r20 cosine20, seed42, panel 200, `compact=true`, method `a4147a1b…`):
- DAGSNet (`runs/sim-kaggle-fbl/batch.json`, base `a6e77a1`): N4c+N2c `dinhan25`, L4+P4 `novahihi`, Q4+L2 `heykewtiie`, P2+Q2 `annhannh331`;
  slug `mh-sim-fb-<tag>-s42-r20` (tag `n4c-37-16`, `n2c-40-16`, `l4-top`, `p4-prop`, `q4-top-q18`, `l2-top`, `p2-prop`, `q2-top-q20`).
- BERT (`../fl_veremi_bert/runs/sim-fbl/*.pushed`, base `49b2d88`): S0B+N4cB `odixeuit`, N2cB+L4B `annhtnt`, P4B+Q4B `accanniez`,
  L2B `thuyanh316`, P2B `minhtrit06`, Q2B `minhtriethihi`; slug `flb-sim-fbl-<tag>-s42-r20` (S0B tag `s0-t3`).
- Quota trước push `runs/sim-kaggle-fbl/quota.json` (04:41, 22 account sạch, ≈118 h). Launch log `runs/sim-kaggle-fbl/launch-20261119.log`,
  `../fl_veremi_bert/runs/sim-fbl/launch-20261119.log`.
- **Watcher** `runs/sim-kaggle-fbl/watch.sh`, PID **31627** (setsid, `watch.pid`), log `runs/sim-kaggle-fbl/watch.log`; poll 1200 s → collect
  hai repo → audit `fbl`/`fbl-bert` → `scripts/gate_fbl.py` → in `FBL_DONE`. Theo dõi: `tail -f runs/sim-kaggle-fbl/watch.log`.
  Sau reboot: kiểm PID còn sống trước khi chạy lại watcher; không chạy launcher lại (marker/push log đã có).
- Gate khóa trước push: [FB-L](../../docs/research/screening-fbl.md) (thêm executed ≤ 1,01×, nền N4c/N2c). Khi FBL_DONE: đọc `verdict-r20.json`,
  audit thật, cập nhật screening-fbl/tried-index/LESSONS; PASS ⇒ báo owner trước pilot (không tự nối r50).

## Job đang chạy — đo tổng compute toàn roster (local GPU, owner 19/11)

Owner chốt: FLOP mở rộng (`scripts/op_counter.py`: mm theo công thức torch + op khác theo phần tử) + GPU-time theo pha; toàn roster
(FedAvg + 13 baseline + MobiHet T3/N4c/N2c/L4/P4/Q4) × DAGSNet + FeatureBERT; panel 200 thật, 2 round; eval báo riêng.
- Hàng đợi `runs/total-compute-20261119/queue.py all`, PID **63321** (setsid, `queue.pid`), log `queue.log`; mỗi job `<model>-<method>.{count,time}.json`.
  Resumable: chạy lại cùng lệnh sẽ bỏ job `status=done`. Ước 8–10 h tuần tự. Runner `scripts/total_compute.py`, bảng `scripts/total_compute_report.py <dir>`.
- Baseline DAGSNet chạy trong repo `../fl_dagsnet_veremi_v2` (engine trùng MobiHet khi floor tắt) với `--panel 200`; BERT qua `sim_fold_val.py` (local_t3).
- Khi xong: chạy report, kiểm `real_rows`=1.476.064 mọi job, job lỗi/OOM ghi rõ; cập nhật [compute harness](../../docs/research/compute-harness.md).

## FB-L + engine `compact` — chuẩn bị (xong 19/11)

Owner 18/11: thử hướng mới trước, **tạm không nối r50** N2/N4; phương pháp phải đạt trên mọi model (DAGSNet + BERT);
thử mọi arm khả thi; sim đòi hơn N4 ≥ +.001 TB; đo compute cùng thước cho mọi method. Thiết kế + luật khóa: [FB-L](../../docs/research/screening-fbl.md);
đo compute: [compute harness](../../docs/research/compute-harness.md).

Đã xong (local, chưa commit):
- `methods/mobihet_budget.py`: knob `real_select=top|prop`, `real_select_q`, `row_score`/`row_gen` trong state; knob `compact` (xong 19/11, dưới).
- Smoke local PASS (không là kết quả): DAGSNet L4/P2/Q4 và BERT Q4, 200 xe 2 round compiled; `real_frac` = estimator, skip 0,
  panel BERT = DAGSNet (`41d3ea2e19a6e703`), r02 `t_train` FB-L = FB. Log `runs/sim-kaggle-fbl/smoke-*.log`.
- Launcher: `scripts/run_sim_fbl.py` (DAGSNet, tái dùng `run_sim_budget.py`, quota gộp nhiều lượt đọc sạch),
  `../fl_veremi_bert/scripts/{sim_fold_val.py,run_sim_fbl.py}` (BERT sim, dùng wrapper bounded của MobiHet), `scripts/gate_fbl.py` (gate 2 model),
  `scripts/audit_sim_batch.py fbl-bert`, watcher `runs/sim-kaggle-fbl/watch.sh` (poll 1200 s → audit → gate → `FBL_DONE`). Chưa chạy launch nào.

**Phát hiện compute (owner hỏi 18/11):** engine hiện tại *thực thi* FB N4 ≈ **2,07× FedAvg** (FlopCounter một bước M/L 2,000×;
wall-clock BERT FedAvg 3,26 h vs FB N2 6,17 h = 1,89×); estimator 1,000× chỉ là useful-work. Engine nén nhóm + bỏ slot thật ở bước floor
ước 0,997× (N4) / 1,001× (N2) so FedAvg cùng engine (`scripts/compute_executed.py`, `runs/sim-kaggle-fbl/debug/variants.py`).

**19/11 — `compact` xong (không job/kernel nào chạy, chưa push).**
- "Lệch gradient bước floor" **không phải bug**: 1 phần tử ReLU sát 0 lật dấu ở fp32 khi shape bmm đổi; fp64 hai đường trùng 1e-16,
  có bước đường *mask* mới là đường lệch so fp64 (`LESSONS` mục FB pilot). Test bước-so-bước chuyển sang fp64, phủ `random`/`top`/`prop`/`top`+q.
- Sửa thật trong method: `compact` đọc `ctx.rows` cả khi `random` ⇒ 4 test FB cũ FAIL; nay chỉ đọc khi chọn theo loss. Thuật toán không đổi.
- Test: `tests/test_mobihet_budget.py` **19/19 PASS** (`runs/compute-executed-20261119/pytest-budget.txt`). `without_floor` chỉ giữ `random`
  (top đổi thứ tự cộng BN ⇒ state cuối round lệch 1e-2 do Adam, đã phủ bằng test fp64 từng bước).
- Method copy BERT byte-identical: sha256 `a4147a1b…` cả hai repo.
- Compute đo trong engine thật (`scripts/count_executed.py`, `runs/compute-executed-20261119/`): panel 200 xe, × FedAvg executed:
  N4 mask 2,0177, N4 compact 0,9155, L4 compact 0,9155; trùng `compute_executed.py --panel 200` ⇒ full-scale **0,997× / 1,001×** (N4/N2).
  `t_train` r2: DAGSNet FedAvg 57 s / N4 mask 100 s / compact 61–62 s; BERT FedAvg 121 s / Q4 compact 119 s / Q4 mask 247 s.
  Không recompile warning; compile gate PASS cả hai model.

## Bước tiếp (theo thứ tự)

1. ~~Sửa `compact`~~, 2. ~~copy BERT + test + smoke~~ — xong 19/11 (trên).
3. Chốt với owner (đã đề xuất, **chưa duyệt**): chạy mọi arm FB-L **với `compact=true`** + rerun control N4c/N2c DAGSNet (cùng engine) và
   S0B/N4B/N2B BERT; thêm gate compute executed ≤ 1,01× bên cạnh useful. Sửa `screening-fbl.md`/`run_sim_fbl.py` theo đó **trước** push.
4. Quota: đọc live lại trước push (receipt 18/11 sáng đã cũ; mạng Kaggle chập chờn ⇒ `MOBIHET_KAGGLE_API_VIA_WWW=1`).
   Ước batch (mask, 18/11): DAGSNet 8 × 2,75 h, BERT 9 × 3,5 h (×1,15) ≈ 61 h; với `compact` arm FB/FB-L ngắn hơn ~40–50 %
   (đo local 19/11), cần ước lại từ `t_train`. Roster còn ≈ 120 h tới reset 21/11 (receipt cũ).
5. Launch DAGSNet trước (`run_sim_fbl.py launch`, tạo `batch.json` mà BERT build đọc), rồi `cd ../fl_veremi_bert && run_sim_fbl.py launch <owners>`,
   rồi watcher nền; ghi kernel/owner/PID vào file này.

## Pilot FB N4 + N2 r20 — xong, audit 18/11 (không chạy lại)

DAGSNet N2/N4 PASS, BERT N2/N4 FAIL gate pilot (ΔTB −.0104/−.0116 < −.010). Phân tích (trễ hội tụ ∝ real_frac, lớp 13 timeDelayAttack,
chiếu r50): [handoff Opus](../../runs/sim-kaggle-fb/opus-pilot-handoff-20261117.md), [LESSONS](../../LESSONS.md) mục FB pilot.
Watcher pilot đã kết thúc; r50 **tạm hoãn theo owner 18/11**.
[Snapshot khôi phục Claude](../operations-20261117/continue-before-codex-recovery.md),
[snapshot trước đủ8push](../operations-20261117/continue-before-fb-push-complete.md).

## FB DAGSNet sim — Đã hoàn tất 8/8 pull, audit PASS, chọn N4

Owner xác nhận FB = **Fixed-row-budget**, không phải FeatureBERT. Seed42, panel200xe, r20cos20.
[Luật khóa](../../docs/research/screening-budget.md): compute≤1,01× trung bình toàn mạng; thay dòng thật xe lớn bằng synthetic.
Artifact đã pull đủ 8/8 về `artifacts/sim-kaggle-fb/`, kiểm tra tính toàn vẹn và audit 100% PASS (`runs/sim-kaggle-fb/audit.json`).
Đánh giá gate theo luật khóa hoàn tất tại `runs/sim-kaggle-fb/verdict-r20.json`, marker `runs/sim-kaggle-fb/DONE` (2026-11-17T19:46:52+07:00).

| Arm | Exact kernel | Status | r20 macro | ΔTB r16–20 | cost (×FedAvg) | Verdict |
|---|---|---|---:|---:|---:|---|
| S0 | `mh-sim-lite2-t3-s42-r20` | COMPLETE | 0,9138 | +0,0000 | 1,655× | CONTROL |
| S7 | `mh-sim-lite2-b-u1-s42-r20` | COMPLETE | 0,9090 | −0,0047 | 1,365× | CONTROL |
| F0 | `novahihi/mh-sim-fb-b-u1-s42-r20/1` | COMPLETE | 0,9089 | −0,0050 | 1,365× | RERUN |
| R48 | `novahihi/mh-sim-fb-r48-s42-r20/1` | COMPLETE | 0,9097 | −0,0038 | 1,144× | ISOLATE |
| N1 | `annhannh331/mh-sim-fb-n1-48-8-s42-r20/1` | COMPLETE | 0,9054 | −0,0084 | 1,003× | FAIL |
| N2 | `annhannh331/mh-sim-fb-n2-40-16-s42-r20/1` | COMPLETE | 0,9071 | −0,0066 | 1,003× | PASS |
| N3 | `annhtnt/mh-sim-fb-n3-45-8-s42-r20/1` | COMPLETE | 0,9047 | −0,0090 | 0,999× | FAIL |
| **N4** | `annhtnt/mh-sim-fb-n4-37-16-s42-r20/1` | COMPLETE | **0,9072** | **−0,0064** | **1,000×** (0,9996×) | **PASS (SELECTED)** |
| Z0 | `accanniez/mh-sim-fb-z0-nosynth-s42-r20/1` | COMPLETE | 0,8800 | −0,0334 | 1,000× | REFERENCE |
| ZK | `accanniez/mh-sim-fb-zk-nosynth-qreal-s42-r20/1` | COMPLETE | 0,8845 | −0,0284 | 1,000× | REFERENCE |

- **Kết luận:** N4 được chọn (macro TB 0,907047, macro r20 0,907154, cost 0,9996×, p10 r20 0,7862).
- Máy reboot lúc 19:36 làm supervisor 38803 dừng; đã chạy tiếp chu trình kéo output, audit 8/8 PASS và gate chốt N4 trong phiên. Hiện không có watcher nào chạy nền.

## Pilot FB — hồ sơ launch (tham chiếu)

Kernel: N4P `dinhan25/mh-fb-n4-37-16-s42-r20/1`, N2P `minhtrit06/mh-fb-n2-40-16-s42-r20/1`, N4B `takids1909/flb-fb-n4-37-16-s42-r20/1`,
N2B `anhtruong1909/flb-fb-n2-40-16-s42-r20/1` (bản `novahihi` owner cancel, không dùng). Spec `runs/fb-full/`, `../fl_veremi_bert/runs/fb-pilot/`.
Launcher `scripts/run_fb_full.py`, `../fl_veremi_bert/scripts/run_fb_pilot.py` chưa commit.

## Cấu hình N4 (tham chiếu)

- N4 tự động lên bậc pilot full-scale (5.343 xe, r1–20 cosine50, seed 42) trên hai kiến trúc: DAGSNet và FeatureBERT.
- Cấu hình N4:
  - Tầng S (< 4.096 dòng): dòng thật 64/bước, synthetic 64,64 (giữ liều T3 đầy đủ + floor 16).
  - Tầng M (4.096–16.383 dòng): dòng thật 37/bước, synthetic 16,16.
  - Tầng L (≥ 16.384 dòng): dòng thật 37/bước, synthetic 16,16.
  - Compute toàn mạng đo đạt: 0,9996× FedAvg (≤ 1,01×).
- Quota đã kiểm 17/11 21:5x (`runs/fb-full/quota.json`); 4 kernel đã push (bảng trên).

## Recovery đã đóng — không làm lại

- Khôi phục Claude `810671d5-dff9-4284-b80d-28eb447cf9b9`, interrupt11:21, từ lịch sử jsonl owner cung cấp.
  Lúc tiếp quản chưa FB push; preflight/smoke cũ được tái dùng, không train/test lại phần hoàn thành.
- **Lỗi Kaggle do mạng chập chờn/timeout.** Gate cache giữ từng readPASS600s; failure N3 cuối cùng xảy ra ở
  `GenerateAccessToken` trước childCLI vì wrapper cũ đặt sau auth. Bản sửa bọc trước auth; regression3/3PASS.
  Không sửa SDK/credential/helper skill frozen; chỉ retry connect trước gửi request, không retry lỗi read sau push.
- 6kernel đầu giữ nguyên; N3 chỉ retry sau chứng cứ auth chưa gọiCLI + owned-list không có ref; giữ intent cũ và retryintent riêng.
  N4 push lần đầu. Không version2, không kernel/watcher trùng. Bằng chứng local `n3-before-submit-failure.json`, các pushlog/receipt.
- Job `mh-fb-finish-n3n4-20261117` supervisor38695 **succeeded**, log `finish-launch.log`, deliveryqueue đã nhận, completion đã xử lý.
  Các launcher gate cũ đã terminal; không rerun từ notification cũ. Chi tiết lỗi trong snapshot trên.

## Lite2 — xong / hoãn

- Sim12/12auditPASS chọnB+U1; pilot DAGSNet/BERT r20PASS. Không collect/audit lại.
- **DAGSNet r50 auditPASS gate Lite2 cũ:** `dinhan25/mh-lite2-bu1-s42-r50/1`, macro.905876977,
  tổng354116,304B/xe, estimator1,364×≤1,37×; chưa đạt đíchFB1,01×.
  [Kết quả](../../docs/research/screening-lite2.md), [10metric/giới hạn](../../docs/research/lite2-dagsnet-r50-audit.json).
  Recovery watcher `mh-lite2-r50-recover-20261117` succeeded; DONE/audit có thật, completion đã xử lý.
- **Owner hoãn push lại Lite2 BERT r50 trên mọi account; ưu tiên FB, không giữ quota riêng cho Lite2.**
  Pilot `annhannh331/flb-lite2-bu1-s42-r20/1` đã PASS, không collect lại.
- T3 DAGSNet/BERT đủr50/3seed, pFedES BERT đầy đủ; [comparison](../../docs/research/comparison.md).

## Khi watcher completion

1. Đọc state/log + terminal.json + pulled.ok; kiểm đủ8arm/r20 trước verdict. ERROR vẫn thu log/output.
   COMPLETE/exit0 không thay artifact audit. Query lỗi = UNKNOWN, không restart training.
2. Gate N1–N4: ΔTB r16–20≥−.007 và Δr20≥−.008 vs S0, cost≤1,01×; chọn macroTB cao nhất.
   F0/R48/Z0/ZK report-only, Z0/ZK không tự lênpilot. FAIL dừng arm, không tune/kéo dài.
3. PASS candidate → pilot full-scale DAGSNet + FeatureBERT theo luật đã duyệt; kiểm quota live/compatible source,
   port ba file method byte-identical sang BERT; không dùng Lite2 BERT r50 thay pilotFB. Mixed-model chưa được duyệt.
4. Cập nhật screening-budget/triedindex/lessons từ artifact thật. Ghi bàn giao trước dừng phiên.

Owner đã yêu cầu commit/push `main` cảhai repo. Receipt GitHub/HEAD/Gitbundle: `runs/maintenance-20261117/publish.json` mỗi repo.
Artifact mới giữ local, chưa DVCbackup; không xóa cache/output. [Backup](../../docs/operations/backup.md). Không cần thao tác tay hiện tại.
