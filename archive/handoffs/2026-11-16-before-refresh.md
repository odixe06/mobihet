> Snapshot16/11/2026; trạng thái sống tại docs/operations/continue.md của repo.

# Bàn giao sống — kiểm 15/11/2026 00:45 UTC+7

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

## Watcher chuỗi phiên 14/11 10:59 VN (Claude Code, owner yêu cầu)

`scripts/watch_chains.sh` (setsid/nohup, **sống qua đóng phiên**; PID xem `pgrep -f watch_chains.sh`, khởi động lại 11:09), tick 1200 s, tối đa 120 tick (~40 h):
mobihet `run_b21_transfer.py collect` → `run_b21_chain.py collect` → `run_b21_chain.py launch` (nối cùng owner mọi head
COMPLETE < r50); fl_veremi_bert `run_batch.py collect`. Chờ nếu collect khác đang chạy. Log `runs/chain-watch/watch.log`,
chi tiết từng tick `runs/chain-watch/{mobihet,bert}-<tick>.log`. Dừng: `pkill -f watch_chains.sh`. Không bật watcher thứ hai.
Chưa tự audit B21 r50: khi arm đủ r50 cần viết/chạy audit chuỗi (rounds trải qua nhiều artifact dir).
Phiên chain 10:40: T142 `dinhan25/mh-k-b21-t142-r50-s34`, T3P `annhtnt/mh-k-b21-t3p-r20-s36`, T343 `trietbackup/mh-k-b21-t343-r50-s7`,
T4F43 `annhannh331/mh-k-b21-t4f43-r50-s4` (cap 11 h); T143/T4FP x-session còn chạy.
- 11:05 fl_veremi_bert: fedavg/fedprox/fd_ids **p4 ERROR** trước decode (p4 mount cả dataset chuyển account lẫn output p3 → 2 `state.pt`).
  Sửa `run_batch.py` (phiên kế bỏ `resume_dataset`), push p5 nối từ p3 (r39/r41/r35). p4 ERROR giữ làm bằng chứng.

## Trạng thái 15/11 00:45 VN (hiện hành, thay mọi mục “Đang chạy”/watcher/“Việc kế” ở trên)

- **Không kernel, watcher hay job nền nào còn sống** (`pgrep -f watch_chains.sh` rỗng; log dừng 14/11 15:28). 6 head B21 COMPLETE 14/11 21:17,
  đã pull vào `artifacts/b21-chain/` (`pulled.ok`).
- **Audit chuỗi r50 xong** (Claude Code 15/11 00:30): `python scripts/run_b21_chain.py audit` → `runs/b21-r50/audit.json`, 6/6 artifact PASS,
  prefix phiên trước PASS, **T1/T3/T4F PASS r50 hai seed**; guard/cohort `runs/b21-r50/report.txt`. Kết quả ở
  [design B21](../research/screening-dose-b21.md), bài học ở [LESSONS](../../LESSONS.md). B20/B21 không còn bậc thang nào treo.
- `run_b21_chain.py` đã sửa glob (14/11) và thêm lệnh `audit` (15/11); chưa commit.

### Việc kế

1. **Owner chọn arm** vào phương pháp chính: T3 (4,94×F, FLOPs thấp nhất PASS — đúng luật khóa) hoặc T4F (5,10×F, p10 +.012/+.013).
   Sau khi chọn: cập nhật [proposal](../research/proposal.md), bảng compute Notion Proposal 2.
2. Seed 41 r50 của arm được chọn (CBB/baseline báo 3 seed; B21 mới có 42/43) — ~9,5–10,5 h GPU, cần owner cho phép launch.
3. Backup khi owner yêu cầu: `artifacts/b21-chain/*` (6 run), `artifacts/b21-transfer/mh-k-b21-t143-r50-x18`, `…-t4fp-r20-x24` chưa DVC;
   rsync `runs/b21-chain/`, `runs/b21-r50/`, `runs/b21-transfer/`, `runs/b21-s43/` vào `_receipts/` trước `dvc add` ([backup](backup.md)).
4. Backlog không đổi: mô hình trộn kiến trúc full-scale (owner 13/11: để sau); privacy bootstrap (hoãn 09/11).

## Quyết định owner 15/11 — hiện hành

- Chọn **T3 làm phương pháp chính**, CBB giữ làm đối chứng; proposal/method/CONTEXT đã cập nhật.
- Ủy quyền chạy T3 DAGSNet seed41 r50, đồng thời BERT 3 seed; đã hỏi chốt BERT T3 hay 14 baseline trước thay repo BERT.
- Không dùng quota B21 cũ để launch: `run_w20.inventory()` trả cache nếu file đã tồn tại. Batch mới phải lấy quota mới trong thư mục riêng.

- Owner đã chốt: BERT = bổ sung T3 seed41/42/43 **và** pull/nối baseline 3 seed đã có. Cùng code/cấu hình T3, chỉ thay model/engine. BERT handoff ở `../fl_veremi_bert/docs/operations/continue.md`.
- DAGSNet T341 đã PUSHED v1 `minhtran0601/mh-k-b21-t341-r50/1`, cap11.5h, source/config khóa `runs/t3-s41/batch.json`; launcher `python -m scripts.run_t3_s41 quota|launch|collect`. KHÔNG launch lại (marker có).
- Watch job `mh-t341-watch-20261115`, log `runs/t3-s41/watch.log`, state `/home/odixe/.codex/long-jobs/jobs/mh-t341-watch-20261115/state.json`, delivery queue, interval1200s. Completion → collect/audit so CBB41 r50; partial phải nối cùng state/RNG hai rank. Auditor B20 chỉ có CBB42/43 nên **chưa dùng audit cũ cho seed41**; cần mở audit seed41 đúng control. Chưa kết luận utility.

- Watch supervisor PID192486, owner thread `01a1db5a-19af-7920-a785-66374f2ab856`; lần kiểm đầu kernel **QUEUED**, chưa có bằng chứng round1. Theo dõi `tail -n 200 -F -- /home/odixe/project/mobihet/runs/t3-s41/watch.log`.

- BERT smoke r1–2 đã audit PASS; controller `flb-t3-launch-watch-20261115` (PID194998, queue về thread trên) đang kiểm resume2→3 rồi tự launch seed41/42/43 và watch1200s. Log `/home/odixe/project/fl_veremi_bert/runs/t3/launch-watch.log`; chi tiết BERT handoff. Chưa gọi BERT launched trước push marker thật.

- BERT T3: resume smoke3 đã audit PASS. Launcher lỗi cell-ID ngẫu nhiên đã sửa, rebuild byte-identical PASS; controller thay thế `flb-t3-launch-watch-20261115-r2`, PID196485, queue/thread như trên, log `/home/odixe/project/fl_veremi_bert/runs/t3/launch-watch-r2.log`. Chỉ sửa đóng gói notebook, không source train/HP.

- BERT launch-watch r2 lỗi inventory trước push (dinhan25 kernel lịch sử HTTP404). Seed41 đổi owner minhtran0601, vẫn phải kiểm quota/slot sống; seed42 annhtnt,43 trietbackup. Controller hiện hành `flb-t3-launch-watch-20261115-r3`, PID198101, queue/thread như trên, log `/home/odixe/project/fl_veremi_bert/runs/t3/launch-watch-r3.log`. Chưa kết luận BERT đã push trước receipt.

- T3 BERT3seed đã PUSHED v1 (minhtran0601/flb-t3-s41, annhtnt/flb-t3-s42, trietbackup/flb-t3-s43), watcher r3 còn sống. Baseline collector lỗi quota drift trước intent sau nối7run seed41; đã sửa WAIT và test4PASS. Watcher hiện hành `flb-three-watch-20261115`, PID200707, queue/thread như trên, log `/home/odixe/project/fl_veremi_bert/runs/three-watch-controller.log`, poll1200s; collect+nối+audit cả3seed.

## Kiểm trực tiếp 15/11 13:17 VN

- T341 DAGSNet RUNNING; BERT T3 seed41 COMPLETE, seed42/43 RUNNING (exact v1 query). COMPLETE chưa chứng minh đủ r50.
- Baseline BERT seed42: 14/14 audit PASS_BOUNDED complete=True. Seed41/43 mỗi seed còn APFed/NILM/LwFedNIDS/pFedES; APFed r42, NILM r28 đang WAIT quota (0.3–0.4h); watcher hiện hữu tiếp tục collect/nối, không tạo trùng.
- Pull BERT T3 seed41: controller `flb-t3-s41-pull-20261115`, supervisor PID359562, delivery queue, state `/home/odixe/.codex/long-jobs/jobs/flb-t3-s41-pull-20261115/state.json`, log `/home/odixe/project/fl_veremi_bert/runs/t3/pull-s41.log`; owner thread theo state controller. Output `runs/t3/pulls/flb-t3-s41` trong repo BERT. Completion phải audit artifact/round/RNG/config; thiếu r50 thì nối đúng checkpoint. Chưa audit T3 seed41.
- Các watcher cũ T341/T3 BERT/baseline vẫn sống; không sửa file chúng sở hữu. Không Git/DVC push, không xóa artifact.

## Owner 15/11 — cho phép nối chéo account

- Owner cho phép chuyển run thiếu quota sang account khác, giữ checkpoint/source/HP/RNG và tổng round.
- Quota sống: novahihi17.42h, thuyanh31616.06h, accanniez15.35h, minhtriethihi14.61h. Controller `flb-cross-resume-20261115` chuyển tuần tự APFed41→novahihi, NILM41→thuyanh316, APFed43→accanniez, NILM43→minhtriethihi qua `scripts/transfer_global.py`; kiểm inventory/slot/quota trước push. Chưa gọi launched trước pushed receipt.
- Log `/home/odixe/project/fl_veremi_bert/runs/cross-resume-20261115.log`; state `/home/odixe/.codex/long-jobs/jobs/flb-cross-resume-20261115/state.json`; delivery queue. Watcher baseline hiện hữu sẽ nhận head mới sau receipt. Completion cần audit parent/artifact, xử lý stage lỗi, không restart từ r1.

## T3 BERT seed41 pull/audit 15/11 13:40 VN

- Pull job succeeded exit0: 147/147 files, 0missing, 3902MiB. Artifact `runs/t3/pulls/flb-t3-s41` trong repo BERT.
- CM/support/10metric audit PASS_BOUNDED r1–36, complete=False (`runs/t3/audit-s41.json`). State audit PASS r36 cả2rank, 2672+2671xe, finite private banks, strict FeatureBERT rebuild, source overlay/base và HP khớp batch, RNG/gen hai rank (`runs/t3/state-audit-s41.json`). Không stored predictions nên không claim replay inference.
- Nối r36→50 cùng owner minhtran0601: job `flb-t3-s41-resume-20261115`, PID363969, delivery queue; state `/home/odixe/.codex/long-jobs/jobs/flb-t3-s41-resume-20261115/state.json`, log `/home/odixe/project/fl_veremi_bert/runs/t3/resume-s41.log`. Live gate quota≥6.18h/slot<2 trước push `flb-t3-s41-p2`; chưa claim launched trước pushed marker. Giữ source/HP/50round schedule, import parent trước decode. Worker watch1200s sau push; completion pull/audit tiếp.

## T3 BERT seed41 resume launcher phục hồi PATH

- Job `flb-t3-s41-resume-20261115` failed trước gửi push: subprocess không tìm thấy `kaggle` trong PATH. Inventory parent hợp lệ, state audit vẫn PASS r36; không training failure. Intent cũ giữ `flb-t3-s41-p2.intent-before-path-fix`; reply lỗi giữ nguyên tới retry. CLI nckh2.2.5 verified.
- Sửa worker prepend thư mục sys.executable vào PATH. Job thay thế `flb-t3-s41-resume-20261115-r2`, supervisor PID364383, delivery queue về thread hiện hành; state `/home/odixe/.codex/long-jobs/jobs/flb-t3-s41-resume-20261115-r2/state.json`; log `/home/odixe/project/fl_veremi_bert/runs/t3/resume-s41-r2.log`. Kiểm inventory/quota/slot lại trước push; không restart r1. Chưa claim launched trước pushed receipt.

## T3 BERT seed41 p2 ERROR — sửa import guard

- Resume r2 push thành công `minhtran0601/flb-t3-s41-p2/1`, sau đó ERROR14:02; pull listing `.log` rỗng, chưa có traceback remote. Notebook frozen có lỗi xác định: generator thiếu --method nên import guard đòi method=fedavg, checkpoint=mobihet_compute. Replay guard checkpoint thật r36 RED→GREEN PASS khi sửa method; không coi đây là traceback remote.
- Giữ p2 intent/pushed/reply và artifact parent. Sửa launcher truyền --method mobihet_compute; thêm assert ô import sinh đúng method. Slug mới p3, parent vẫn phiên gốc r36, không dùng ERRORp2 làm parent. Source train/HP không đổi.
- Controller `flb-t3-s41-resume-20261115-r3`, delivery queue; state `/home/odixe/.codex/long-jobs/jobs/flb-t3-s41-resume-20261115-r3/state.json`; log `/home/odixe/project/fl_veremi_bert/runs/t3/resume-s41-r3.log`. Kiểm quota/slot lại trước push, watch1200s. Chưa kết luận launched/r50 trước receipt/audit.

## T3 BERT seed41 p3 đã push — phục hồi watcher

- Job resume-r3 failed ở watcher, không ở push: receipt xác nhận `minhtran0601/flb-t3-s41-p3/1` đã push. Watcher dùng lại resume-watch của p2, batch SHA mismatch nên dừng trước query. Không launch lại training.
- Sửa launcher dùng thư mục watch theo slug. Watch-only job `flb-t3-s41-p3-watch-20261115`, supervisor PID368898, delivery queue; state `/home/odixe/.codex/long-jobs/jobs/flb-t3-s41-p3-watch-20261115/state.json`; log `/home/odixe/project/fl_veremi_bert/runs/t3/p3-watch.log`; poll1200s, out `runs/t3/flb-t3-s41-p3-watch`. Giữ state p2 làm evidence. Completion cần collect/audit round thực.

## Nối chéo hoàn tất staging/push 15/11 14:55 VN

- Controller cross-resume succeeded exit0; đủ4 ready dataset + parent round + resume_dataset + intent/pushed/version1 verified. APFed41 r42→novahihi/flb-apfed-s41-p3 RUNNING; NILM41 r28→thuyanh316/flb-nilm-s41-p2 RUNNING; APFed43 r42→accanniez/flb-apfed-s43-p3 RUNNING; NILM43 r28→minhtriethihi/flb-nilm-s43-p2 QUEUED (query14:54:53–54). Chưa audit r50.
- Baseline watcher PID200710 sống, đã nhận head mới seed41; seed43 mới push sẽ nhận qua sessions/intent ở lượt kế. Không watcher trùng.
- pFedES41 vẫn r27 WAITquota0.4h. Khởi transfer bổ sung `flb-pfedes41-cross-20261115` tới emtrangtri, live quota/slot gate trước push; log `/home/odixe/project/fl_veremi_bert/runs/pfedes41-cross-20261115.log`; state `/home/odixe/.codex/long-jobs/jobs/flb-pfedes41-cross-20261115/state.json`, delivery queue. Không claim launched trước receipt. LwFedNIDS41 đang nối p3 từr47; LwFedNIDS43/pFedES43 head trước còn running theo collector.

## pFedES41 chuyển account xong 15/11 15:15 VN

- Job `flb-pfedes41-cross-20261115` succeeded exit0; pull59/59file,0missing. Dataset `emtrangtri/flb-pfedes-s41` ready, state2rank sizes832642276/832329888 bytes; checkpoint r27 từ `odixe0502/flb-pfedes-s41-p2/1`.
- Receipt intent/pushed + metadata private mount đúng dataset verified; không mount kernel parent trùng. Kernel `emtrangtri/flb-pfedes-s41-p3/1` đã PUSHED cap8.95h, query15:15:34 QUEUED. Chưa có bằng chứng round28/r50.
- Watcher baseline PID200710 vẫn sống, sessions() nhận head mới từ intent/pushed; tiếp tục collect/nối/audit3seed1200s. Không tạo watcher trùng. Không cần thao tác tay.


## Kiểm 2026-11-15T21:21:01.161052+07:00 — T3 completion

Owner yêu cầu xử lý cả T3 DAGSNet. Exact `minhtran0601/mh-k-b21-t341-r50/1` COMPLETE; local rounds tới r50. Watcher cũ không còn process. Job collect hiện hành `t3-collect-all-20261115-evening` trong repo BERT (PID29693, queue) sẽ pull đầy đủ DAGSNet41 sau audit BERT; log `/home/odixe/project/fl_veremi_bert/runs/t3/collect-all-evening.log`. Completion audit so CBB41, đủ r50 thì không nối. T342/43 đã audit r50 theo hub. Shared helper scripts còn nguyên; BERT nay dùng bản local, không phụ thuộc watcher repo này. Không Git/DVC push hoặc xóa artifact.


## T3 DAGSNet41 hoàn tất — 15/11/2026 tối

- `minhtran0601/mh-k-b21-t341-r50/1` COMPLETE, pull188/188file,0missing. CM/support/LR/source/HP/10metric audit PASS r1–50 so đúng CBB41; utility PASS (> .9048). Macro-F1 0.907866023, CBB41 .9077.
- State hai rank PASS: 2672+2671xe, r50, RNG, bank hữu hạn, strict DAGSNet rebuild, provenance/hash nguồn khớp. Không cần nối T341. Seed42/43 đã r50 trước phiên này.
- Evidence `runs/t3-s41/audit.json`, `state-audit.json`, `report.md`; artifact `artifacts/t3-s41/mh-k-b21-t341-r50/`. Không replay inference; không Git/DVC push/xóa artifact.
