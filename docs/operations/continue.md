# Bàn giao sống — kiểm 13/11/2026 11:05 UTC+7

Đọc [CONTEXT](../../CONTEXT.md), [LESSONS](../../LESSONS.md), [operations lessons](lessons.md) trước.
Đây là **bàn giao hiện hành duy nhất**. Artifact/state/process thật ưu tiên tóm tắt.
Lịch sử B13–B19, A1, N1/N2, baseline seed 41/43 đến 13/11: [snapshot](../../archive/handoffs/2026-11-13-before-consolidation.md).

## Đang chạy — B21 full-scale: pilot r20 + T1 r50 phiên 1 (Claude Code)

[Design + luật + kết quả sim](../research/screening-dose-b21.md). Sim 13/11 audit PASS 12/12: ứng viên **T1, T3, T4, T4F**; **T5 FAIL** (dừng).
Push 13/11 ~10:55–11:03, version 1, base `be3b7d5`, hp = CBB42 + knob tầng, `synth_late_from=25`, W&B off.

| arm | kernel | cap | ghi chú |
|---|---|---:|---|
| T4P | `dinhan25/mh-k-b21-t4p-r20` | 5,5 h | pilot r20 cos50 seed 42 |
| T3P | `emtrangtri/mh-k-b21-t3p-r20` | 5,5 h | pilot r20 cos50 seed 42 |
| T4FP | `keocaosu/mh-k-b21-t4fp-r20` | 6,0 h | pilot r20, floor 48 |
| T142 | `novahihi/mh-k-b21-t142-r50` | 5,5 h | T1 r50 **phiên 1**, dừng theo budget (state mỗi 5 round) |
| T143 | `odixe0502/mh-k-b21-t143-r50` | 4,4 h | T1 r50 seed 43 **phiên 1**, dừng theo budget |

- Launcher `scripts/run_b21_full.py <quota|launch|collect|audit> [t1]` → `runs/b21-full/` (pilot), `runs/b21-t1/` (T1); cả hai có
  `launch-complete.json` — **không rerun launch**.
- Waiter: Bash background phiên Claude Code, task `b0cy07077`, collect cả hai batch 1200 s (tối đa 27 lượt) → `audit` pilot khi tất cả terminal.
  Log: `tail -n 20 -F -- /home/odixe/project/mobihet/runs/b21-watch/watch.log`. Chết theo phiên → chạy tay
  `python scripts/run_b21_full.py collect`, `... collect t1`, rồi `python scripts/run_b21_full.py audit`. Không bật waiter trùng.
- Luật pilot (khóa): Δmacro TB r16–20 ≥ −.0015 và Δ r20 ≥ −.003 so CBB42 cùng round, không điều khoản quỹ đạo. `audit` in dòng `B21 PASS/FAIL`.
- Việc kế sau pilot/phiên 1 (cần quota mới, reset tuần thứ Bảy 14/11 ~07:00): pilot PASS → nối r20→r50 cùng owner + seed 43 r50;
  T142/T143 → nối từ round state cuối (bội 5) tới r50 cùng owner. Mẫu resume: `scripts/run_b20_sm50.py` (`resume_row`, ô import kiểm 2 rank/round).
  **Chưa viết launcher nối phiên B21.** Audit T1 chỉ chạy khi đủ r50 (`audit t1` trên phiên 1 sẽ báo rounds_exact FAIL — đúng).
- Sim b21: pipeline `b2r2kfzle` đã thoát (audit FAIL giả ở check `config`, đã sửa `audit_sim_batch.norm`, chạy tay audit/analyse/report);
  bản FAIL giữ ở `runs/sim-kaggle-b21/audit-config-false-fail.{json,log}`.
- Chưa ghi: bài học sim B21 vào LESSONS / operations lessons, hàng B21 ở tried-methods-index và bullet B21 trong CONTEXT (còn ghi “chạy sim”).

## B20 đã hoàn tất — audit 13/11 19:15 VN

[Design/luật/kết quả](../research/screening-compute-b20.md), [phụ lục luật](../research/screening-compute-b20-addendum.md).
Luật r50 mỗi arm: macro F1 r50 **mỗi seed > .9048**; báo Δ so CBB cùng seed, 10 metric, guard + trend, FLOPs, `t_train`.

| arm | kernel (version 1) | push | cap | trạng thái 13/11 09:10 |
|---|---|---|---:|---|
| SD42 | `novahihi/mh-k-b20-sd42-r50` | 12/11 ~21:50 | 11,5 h | COMPLETE, audit PASS, **PASS** .9072 vs CBB .9074 |
| SD43 | `keocaosu/mh-k-b20-sd43-r50` | 12/11 ~21:50 | 11,5 h | COMPLETE, audit PASS, **PASS** .9081 vs CBB .9081 |
| SM42P | `emtrangtri/mh-k-b20-sm42p-r20` | 12/11 | 5,5 h | COMPLETE, pulled, audit PASS, **PASS_TRAJECTORY** |
| SM42 | `emtrangtri/mh-k-b20-sm42-r50-c2` (resume SM42P r20→r50) | 13/11 02:30 | 7,0 h | COMPLETE, pulled 09:10; **chưa audit** (r50 macro .9057 đọc từ `rounds.jsonl`) |
| SM43 | `minhtran0601/mh-k-b20-sm43-r50` (fresh r1→r50) | 13/11 02:30 | 11,5 h | RUNNING (cap hết ≈ 14:00) |

- **SD r50 PASS hai seed** (audit 07:40, [audit.json](../../runs/b20-full/audit.json)): Δ −.0002/−.0001, p10 −.0068/−.0041, 5,32×F.
  Chưa ghi vào design (source-lock tới khi SM43 xong), chưa DVC (`artifacts/b20-full/mh-k-b20-sd4{2,3}-r50` untracked).
- **Watcher gốc PID 36880 đã chết 08:21** (đóng phiên). Thay bằng waiter chỉ batch SM: Bash background phiên Claude Code 13/11 09:10,
  task `bypmmrltr`, lặp `run_b20_sm50.py collect` 1200 s (tối đa 30 lượt) → `audit` khi ALL_TERMINAL, ghi cùng `runs/b20-watch/watch.log`.
  Waiter này cũng chết theo phiên: khi mất, chạy tay lệnh collect/audit bên dưới. Không chạy lại `watch_b20.py` (state mới sẽ audit lại b20-full).
- Diagnostic cohort 13/11: `scripts/analyse_b20_cohort.py` → [cohort.txt](../../runs/b20-cohort/cohort.txt); bài học ở [LESSONS](../../LESSONS.md).

- SM42P (pilot r20 cos50): macro .8962 vs CBB42 .9000, Δr20 −.0037, Δ TB r16–20 −.0039, slope Δ nửa sau +.00022 (closing);
  `t_train` 3,57 h vs CBB 3,61 h. [audit.json](../../runs/b20-full/audit.json).
- Batch SM r50: `scripts/run_b20_sm50.py` → `runs/b20-sm50/batch.json` (khóa, `launch-complete.json` có — **không rerun launch**).
  Resume = notebook SM42P + ô import kiểm 2 rank/round 20/method/seed + `--until 50` + budget 7 h (mẫu `run_codec_combo50.py`).
  SM43 = notebook giống SD43 trừ knob `synth_grad=mix16`. Base `be3b7d5`, overlay SHA = SM42P.
- **Watcher gốc (đã chết):** `scripts/watch_b20.py` (Bash background trong phiên Claude Code, PID 36880, task `b1jofw7yg`),
  poll 1200 s, tối đa 40 tick (~13 h). Mỗi tick `collect` cả hai batch; batch ALL_TERMINAL → `audit` một lần
  (`runs/b20-full/audit.json`, `runs/b20-sm50/audit.json`). Thoát sớm nếu kernel nào terminal khác COMPLETE.
  Log `runs/b20-watch/watch.log`, state `runs/b20-watch/state.json`.
  Theo dõi: `tail -n 100 -F -- /home/odixe/project/mobihet/runs/b20-watch/watch.log`.
- Watcher mất (đóng phiên/reboot): kiểm PID; không bật watcher trùng. Chạy tay
  `python scripts/run_b20_full.py collect && python scripts/run_b20_full.py audit` và tương tự `run_b20_sm50.py`.
- Sau audit: verdict từng arm vào [design](../research/screening-compute-b20.md), gộp phụ lục vào design, cập nhật
  bảng compute Notion Proposal 2 nếu arm PASS. Backup: rsync `runs/b20-full/`, `runs/b20-sm50/` (bỏ `.pt`) vào `_receipts/` từng run,
  `dvc add` per-run SD42/SD43/SM42/SM43 + `dvc push` ([backup](backup.md)); SM42P đã DVC 13/11. Nhiều arm PASS → owner chọn arm rẻ nhất; combo SM+SD chỉ sau verdict từng arm.

## Đã xong gần đây

- **Baseline 3 seed** (`../fl_dagsnet_veremi_v2/runs/seeds/`): 15 baseline × seed 41/43 audit PASS/PASS_BOUNDED,
  NILM s41/s43 p2 đủ 50 round (audit 21:51 12/11). SCAFFOLD loại khỏi Notion (phân kỳ r2–5, NaN r15).
  Chưa xác minh hàng NILM-FL 3 seed trên Notion đã cập nhật.
- **CBB s41** `thuyanh316/mh-k-cbb-s41-r50/1`: audit 10/10, macro .9077; 3 seed .9078 ± .0004 — [report](../../runs/cbb-s41/report.md).
- B13–B19, A1, N1, N2: đóng, FAIL — [sổ tay](../research/method-building-brief.md) §4–5. Không job/collector nào của các batch này còn sống;
  state controller cũ `running` (B14, B18) là stale sau reboot, không restart.

## Việc kế

1. Chờ waiter B21 full-scale và waiter SM43 (completion đánh thức phiên). Không model polling.
2. B21: xem mục trên. SM43 xong: verdict SD + SM vào design B20, gộp phụ lục, DVC per-run.
3. Mục tiêu compute (owner 13/11): nhỏ nhất khả thi mà không ảnh hưởng performance → chọn arm FLOPs thấp nhất PASS r50.
4. Backlog: mô hình trộn kiến trúc full-scale (owner 13/11: để sau); privacy bootstrap (hoãn 09/11).

## Quyền / việc tay

Owner: đổi mọi module/compute, hỗ trợ khác kiến trúc; experiment chỉ DAGSNet W1; thang sim → pilot r20 → r50 hai seed đã ủy quyền
tự lên khi PASS. Credential process-scope; không đổi global credential, không xóa artifact, không hủy job khỏe.
Git push chỉ khi owner yêu cầu (13/11: đã yêu cầu push GitHub). Chưa cần thao tác tay.


## Cập nhật kiểm trực tiếp 13/11 19:15 VN (Codex)

- B20 SM42/SM43 COMPLETE, pulled, audit PASS; macro .9057/.9058, utility PASS cả2seed. Evidence `runs/b20-sm50/audit.json`; design đã ghi kết quả. SD vẫn PASS hai seed.
- B21 pilot: T3P/T4FP audit + utility PASS; T4P artifact PASS nhưng utility FAIL (Δ TB cuối5 −.0016 < −.0015), dừng T4. Evidence `runs/b21-full/audit.json`.
- T142/T143 COMPLETE partial r22/r18, chưa là r50 FAIL. Đang nối T142, T3P, T4FP qua batch `runs/b21-resume/batch.json`; preflight hai rank/RNG/round trước decode. T143 quota1.35h chưa đủ ngân sách setup + checkpoint an toàn, hoãn reset14/11 07:00 VN.
- Watcher Claude cũ không còn process; không dựa completion của chúng. Phiên này không Git/DVC push.

- Controller phục hồi: `mh-b21-resume-20261113`, supervisor PID10729, trạng thái `running`; owner thread `01a1d4cf-a312-7411-a6dc-df58a2c8d9f5`, delivery queue; state `/home/odixe/.codex/long-jobs/jobs/mh-b21-resume-20261113/state.json`; log `runs/b21-resume/watch.log`. Poll1200s. Theo dõi: `tail -n 200 -F -- /home/odixe/project/mobihet/runs/b21-resume/watch.log`.
- Ba phiên nối đã PUSHED v1; source/config giữ nguyên. Watcher chỉ chờ terminal; completion phải collect + audit partial trước nối tiếp. Launcher phục hồi lưu `runs/b21-resume-script.py`; KHÔNG rerun: intent/launch-complete đã tồn tại. Seed43 mới cho T3/T4F chưa launch; cần cấp quota sau reset.

- T1 partial checks PASS r22/r18: `runs/b21-t1/partial-audit/audit.json`; utility r50 PENDING.


## Cập nhật pull & audit 13/11 23:45 VN

- Cả 3 kernel nối `b21-resume` COMPLETE trên Kaggle, đã pull đủ về `artifacts/b21-resume/` qua `scripts/run_b21_resume.py`.
- Audit 10/10 kiểm tra tính toàn vẹn artifact: PASS 100% cả 3 run (`rounds_exact`, `n_params`=395408, `finite_loss`, `skip_le_1pct`, `clients_applied`, `lr_eq_cbb`, `hp_eq_launched`, `overlay_sha`, `cm_support_eq_cbb`, `ten_metrics_from_cm`).
- Số round hoàn thành thực tế (dừng theo budget session 2):
  * **T3P** (`emtrangtri/mh-k-b21-t3p-r20-c2`): dừng round 27 (cap 2,41 h). Macro .9026 vs CBB42 .9034 (Δend −.00087, Δ TB 5 round cuối −.00046). `t_train` tích lũy 5,05 h vs CBB 4,94 h.
  * **T4FP** (`keocaosu/mh-k-b21-t4fp-r20-c2`): dừng round 24 (cap 1,9 h). Macro .9019 vs CBB42 .9019 (Δend +.000006, Δ TB 5 round cuối −.00043). `t_train` tích lũy 5,04 h vs CBB 4,39 h.
  * **T142** (`novahihi/mh-k-b21-t142-r50-c2`): dừng round 25 (cap 1,64 h). Macro .9025 vs CBB42 .9024 (Δend +.000078, Δ TB 5 round cuối +.00020). `t_train` tích lũy 4,82 h vs CBB 4,57 h.
- Verdict: Partial checks PASS, quỹ đạo non-inferiority vững; verdict chung r50 giữ `PENDING_R50` (chờ quota tuần mới reset 14/11 ~07:00 VN để nối tiếp tới round 50).
- Evidence: `runs/b21-resume/audit.json`, `runs/b21-resume/status.json`.

## Đang chạy 14/11 03:20 VN — B21 nối chéo account (Claude Code)

Owner chọn chuyển chéo account thay vì chờ reset 07:00 (owner cũ chỉ còn 1,0–1,35 h). Pipeline `scripts/run_b21_transfer.py`
(plan khóa `runs/b21-transfer/plan.json`): mỗi arm tuần tự pull output parent (state 2 rank + models, ~9,6 GB) → kiểm round/RNG/hp
→ tar `resume.bin` → dataset private owner mới `mh-b21-<arm>-ckpt-r<N>` → tải lại kiểm SHA (stream) → xóa scratch → notebook
`<run>-x<N>` (ô `dataset_import`, bỏ ô kernel_import cũ, budget theo quota sống) → push v1.

| arm | parent | round | owner mới | kernel |
|---|---|---:|---|---|
| T143 | `odixe0502/mh-k-b21-t143-r50/1` | 18 | outlookcheck | `mh-k-b21-t143-r50-x18` |
| T4FP | `keocaosu/mh-k-b21-t4fp-r20-c2/1` | 24 | minhtran0601 | `mh-k-b21-t4fp-r20-x24` |
| T142 | `novahihi/mh-k-b21-t142-r50-c2/1` | 25 | dinhan25 | `mh-k-b21-t142-r50-x25` |
| T3P | `emtrangtri/mh-k-b21-t3p-r20-c2/1` | 27 | annhtnt | `mh-k-b21-t3p-r20-x27` |

06:40 VN: cả 4 đã push (caps T143 6,93 h, T4FP 6,68 h, T142 2,76 h, T3P 2,69 h), dataset kiểm SHA; pipeline thoát `B21XFER_RUN_DONE`. Collect: `python scripts/run_b21_transfer.py collect`.

- Bash background task `b344kvvxy` (chết theo phiên). Log `tail -F runs/b21-transfer/run.log`; từng arm `runs/b21-transfer/<arm>/`
  (`ready.json`, `batch.json`, `launch-complete.json`). Idempotent: chạy lại `python scripts/run_b21_transfer.py run` bỏ qua arm đã push;
  có `create-intent.json` mà thiếu `ready.json` → kiểm dataset trên Kaggle trước, không tạo lại.
- Collect: `python scripts/run_b21_transfer.py collect` → `artifacts/b21-transfer/`. Audit partial/r50 chưa viết cho batch này.
- Phiên kế sau đó: nối tiếp cùng owner mới qua kernel_sources (quota reset 07:00). Seed43 mới T3/T4F vẫn chưa launch.
- 14/11 03:40: pull state đổi sang Range 16 đoạn/file (`fetchstate`, chạy dưới credential owner parent); tốc độ vẫn ~5,7 MB/s ⇒ nghẽn
  đường truyền local, không phải số luồng. Task `bctu2nigb` lỗi ChunkedEncodingError 03:30 → thêm retry 10 lần giữ range piece; task hiện hành `bpva3yfam` (task `b344kvvxy` đã dừng, xóa `.part` scratch của nó).
- **Seed43 r50 phiên 1 đã push** (`python scripts/run_b21_full.py <cmd> s43` → `runs/b21-s43/`, `launch-complete.json` có — không rerun launch):
  T343 `trietbackup/mh-k-b21-t343-r50` cap 2,34 h; T4F43 `annhannh331/mh-k-b21-t4f43-r50` cap 2,0 h; cả hai RUNNING 03:45.
  Dừng theo budget (~vài round) → nối cùng owner sau reset 07:00. Collect: `python scripts/run_b21_full.py collect s43`. Chưa có watcher.
