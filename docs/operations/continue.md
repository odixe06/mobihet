# Bàn giao sống — 17/11/2026

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [bài học vận hành](lessons.md).
[Snapshot trước khôi phục](../../archive/operations-20261117/continue-before-codex-recovery.md).

## Quyết định đang áp dụng

- Khôi phục Claude session `810671d5-dff9-4284-b80d-28eb447cf9b9`, interrupt 11:21 +07; lịch sử tại
  `~/.claude/projects/-home-odixe-project-mobihet/810671d5-dff9-4284-b80d-28eb447cf9b9.jsonl`.
  Tool cuối resolve QUERY_ERROR bị từ chối, chưa thực thi; lúc tiếp quản không batch/intent/receipt push FB.
- Owner xác nhận **8 sim FB = Fixed-row-budget, model DAGSNet**, không phải FeatureBERT.
  Arm/HP/gate khóa tại [screening-budget](../research/screening-budget.md); seed42, panel200xe, r20cos20, CAP2,75h/kernel.
  Compute ≤1,01× trung bình toàn mạng theo useful-work estimator; cho phép bớt dòng thật xe ≥4096 để thay synthetic.
  Sim PASS → pilot DAGSNet + FeatureBERT theo luật đã khóa; FAIL dừng arm, không tune/kéo dài.
- **Hoãn push lại Lite2 BERT r50 trên mọi account. Không giữ quota cho Lite2; ưu tiên FB.**
- Owner yêu cầu commit/push thay đổi; quyền cũ gồm cả MobiHet/BERT trên `main`. Không DVCpush hoặc cleanup trong yêu cầu này.

## FB — đang triển khai

Spec frozen `runs/sim-kaggle-fb/batch.json`, base `1cc91d96` + overlay ba file `methods/mobihet_{compute,lite,budget}.py`.
Không sửa notebook/HP/source đã khóa; mỗi ref dưới là version1:

| Arm | Exact kernel |
|---|---|
| F0 | `novahihi/mh-sim-fb-b-u1-s42-r20/1` |
| R48 | `novahihi/mh-sim-fb-r48-s42-r20/1` |
| N1 | `annhannh331/mh-sim-fb-n1-48-8-s42-r20/1` |
| N2 | `annhannh331/mh-sim-fb-n2-40-16-s42-r20/1` |
| N3 | `annhtnt/mh-sim-fb-n3-45-8-s42-r20/1` |
| N4 | `annhtnt/mh-sim-fb-n4-37-16-s42-r20/1` |
| Z0 | `accanniez/mh-sim-fb-z0-nosynth-s42-r20/1` |
| ZK | `accanniez/mh-sim-fb-zk-nosynth-qreal-s42-r20/1` |

Launch đang do job `mh-fb-launch-20261117` sở hữu, supervisor17367, deliveryqueue, ownerthread `01a1e828-c4a1-77c3-aa09-f97a0ec4a22b`.
Log `runs/sim-kaggle-fb/launch-recovery.log`; state `~/.codex/long-jobs/jobs/mh-fb-launch-20261117/state.json`.
Lượt gate đầu timeout owned-list novahihi, chưa intent/push; worker chỉ retry đọc khi chưa có bất kỳ pushintent nào.
Chưa xác nhận đủ8 ở thời điểm ghi mục này; đọc `launch-complete.json` và từng `.push.json` trước mọi retry.
Quota/slot đã đọc mới: novahihi12,618h; annhannh33111,485h; annhtnt10,897h; accanniez10,443h;
không active/unknown, reserved0 ở inventory; mỗi account2×2,75h +15% =6,325h. Launcher kiểm lại quota/input/slug trước mutation.
Receipt `quota-recovery.json` giữ toàn inventory; `quota.json` chỉ4owner được chọn, `quota-claude.json` giữ lần đọc cũ.
`takids1909` loại vì còn QUERY_ERROR; anhtruong1909 quota ConnectTimeout, dinhan25 list timeout; không suy slot trống từ lỗi.

**Lỗi vận hành Kaggle17/11 do mạng chập chờn/timeout**, không training/kernel FAIL hoặc hết quota.
Nhiều kernel cũ QUERY_ERROR trả COMPLETE sau hỏi lại không đổi. QUERY_ERROR luôn UNKNOWN tới khi đọc được.
Preflight Claude được tái dùng, không train/test lại: tests budget6/6, Lite2 8/8; smokeN2/ZK compiled2round PASS/skip0;
`recovery-preflight.json` ghi hash source/log hiện tại. Không coi smoke2round là utilitysim.

Watcher FB: `runs/sim-kaggle-fb/watch.sh`, launcher chỉ start sau đủ8receipt, job dự kiến `mh-sim-fb-watch-20261117`,
log `runs/sim-kaggle-fb/watch.log`, state `~/.codex/long-jobs/jobs/mh-sim-fb-watch-20261117/state.json`, deliveryqueue.
Chưa xác nhận running tại thời điểm ghi mục này. Khóa `collector.lock` + exact-version
`watch_kaggle_batch.py`, interval1200s → collect → audit_sim_batch fb20 → gate → DONE chỉ khi đủ8artifact/kiểmPASS.
Không start watcher/collector thứ hai. Owner Codex thread `01a1e828-c4a1-77c3-aa09-f97a0ec4a22b`.

## Lite2 — đã hoàn thành, không làm lại

- Sim12/12auditPASS: `runs/sim-kaggle-lite2/{audit,verdict-r20}.json`; chọnB+U1. Pilot DAGSNetr20PASS,
  `dinhan25/mh-lite2-bu1-s42-r20/1`: macro.896761456, Δr20−.002757432, ΔTB−.002790657, bootstrap340663,237B.
- **DAGSNet r50 xong/auditPASS gate Lite2 cũ:** `dinhan25/mh-lite2-bu1-s42-r50/1`, parentpilot trên;
  r1–50/prefixpilot/source/HP/LR/CM-support/10metric/finite/applied-stepPASS. Macro.905876977 vsT3.907678902;
  tổng354116,304B/xe, estimator1,364× ≤1,37×. Chưa đạt đíchFB1,01×. [Kết quả/giới hạn](../research/screening-lite2.md),
  [receipt đã lưu Git](../research/lite2-dagsnet-r50-audit.json). Không collect/audit hoặc nối thêm lần nữa.
- Watcher Claude cũ mất sau host reboot13:02:17; PID8006 stale, không process. Codex phục hồi **watcher**, không training:
  job `mh-lite2-r50-recover-20261117`, supervisor11155, **succeeded/exit0**, deliveryqueue đã accepted.
  Log `runs/lite2-full-r50/recovery-watch.log`; state `~/.codex/long-jobs/jobs/mh-lite2-r50-recover-20261117/state.json`;
  ownerthread `01a1e828-c4a1-77c3-aa09-f97a0ec4a22b`. DONE/audit.json có thật, completion đã xử lý.
- Pilot BERT r20 **PASS**, `annhannh331/flb-lite2-bu1-s42-r20/1`, COMPLETE09:18 +07; pull108file/0thiếu;
  macro.898173076, Δr20−.004746125, ΔTB−.005172594, bootstrap340663,237B. Bounded audit; chưa state2rank local.
  `../fl_veremi_bert/runs/lite2-pilot/{DONE,audit.json}` có thật, watcher đã thoát. Không collect lại; r50BERT hoãn.
- T3 DAGSNet/BERT đủr50/3seed; pFedES BERT đủr50/3seed, audit/Notion xong. [Comparison](../research/comparison.md).

## Tiếp theo

1. Xác nhận đủ8receipt FB; nếu push reply mất/intent còn thì kiểm exactversion/source trước retry, không đẩy version2 mù.
2. Completion watcher FB: đọc state/log, terminal.json, pulled.ok, audit.json và verdict-r20.json; phải đủ8arm/r20.
   COMPLETE/exit0 không thay artifact audit; ERROR vẫn pull log/output. Lỗi mạng chỉ retry query/pull, không restarttraining.
3. Gate sim: N1–N4 PASS nếu ΔTB r16–20≥−.007 và Δr20≥−.008 vs S0, cost≤1,01×; chọn macroTB cao nhất.
   F0/R48/Z0/ZK chỉ report; Z0/ZK không tự lênpilot. PASS candidate → pilot2model theo [luật](../research/screening-budget.md),
   kiểm quota live, port ba file method byte-identical sang BERT, không dùng Lite2 BERT r50 thay pilotFB.
4. Cập nhật screening-budget/triedindex/lessons từ kết quả thật. Không sửa source/ledger đang do watcher sở hữu.

Artifact mới giữ local; chưa DVCbackup, không xóa cache/output. [Backup](backup.md) giữ remote/restore và phạm vi Git.
Không cần thao tác tay hiện tại.
