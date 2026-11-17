# Bàn giao sống — 17/11/2026

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [bài học vận hành](lessons.md).
[Snapshot khôi phục Claude](../../archive/operations-20261117/continue-before-codex-recovery.md),
[snapshot trước đủ8push](../../archive/operations-20261117/continue-before-fb-push-complete.md).

## Đã push đủ 8 sim FB DAGSNet — đang chờ kết quả

Owner xác nhận FB = **Fixed-row-budget**, không phải FeatureBERT. Seed42, panel200xe, r20cos20, CAP2,75h/kernel.
[Luật khóa](../research/screening-budget.md): compute≤1,01× trung bình toàn mạng; cho phép thay một phần dòng thật xe lớn bằng synthetic.
[Receipt milestone đã lưu Git](fb-launch-20261117.json): đủ8reply version1, hash notebook/metadata, status remote đầu tiên.
`runs/sim-kaggle-fb/launch-complete.json` và đủ8`.push.json` có thật; batch frozen `batch.json`,
base `1cc91d96` + overlay ba file `methods/mobihet_{compute,lite,budget}.py`. Không đổi HP/notebook/source đã chạy.

Status remote tại 2026-11-17T14:33:59.367681+07:00; đây là vận hành, chưa kết luận train/utility/hardware PASS:

| Arm | Exact kernel | Status |
|---|---|---|
| F0 | `novahihi/mh-sim-fb-b-u1-s42-r20/1` | RUNNING |
| R48 | `novahihi/mh-sim-fb-r48-s42-r20/1` | RUNNING |
| N1 | `annhannh331/mh-sim-fb-n1-48-8-s42-r20/1` | RUNNING |
| N2 | `annhannh331/mh-sim-fb-n2-40-16-s42-r20/1` | RUNNING |
| N3 | `annhtnt/mh-sim-fb-n3-45-8-s42-r20/1` | RUNNING |
| N4 | `annhtnt/mh-sim-fb-n4-37-16-s42-r20/1` | QUEUED |
| Z0 | `accanniez/mh-sim-fb-z0-nosynth-s42-r20/1` | RUNNING |
| ZK | `accanniez/mh-sim-fb-zk-nosynth-qreal-s42-r20/1` | RUNNING |

## Watcher duy nhất đang chạy

- Job **`mh-sim-fb-watch-20261117`**, supervisor **38803**, delivery **queue**;
  ownerthread `01a1e828-c4a1-77c3-aa09-f97a0ec4a22b`.
- Log `runs/sim-kaggle-fb/watch.log`; state `~/.codex/long-jobs/jobs/mh-sim-fb-watch-20261117/state.json`.
  Theo dõi: `tail -n 200 -F runs/sim-kaggle-fb/watch.log`.
- Worker `runs/sim-kaggle-fb/watch.sh`: `collector.lock` + `watch_kaggle_batch.py`, exact owner/slug/version,
  **1200s/lần**, không model polling. Đủ8terminal → collect → audit_sim_batch fb20 → gate → DONE chỉ khi artifact đủ/kiểmPASS.
  Ledger `watch-state/{latest,history,terminal}.json*`; output `artifacts/sim-kaggle-fb/`.
- `MOBIHET_KAGGLE_API_VIA_WWW=1` được kế thừa; `kaggle_as_bounded.py` áp route/timeout **trước auth**.
  Không start watcher/collector/launcher thứ hai, không sửa source hoặc ledger worker đang sở hữu.

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
  [Kết quả](../research/screening-lite2.md), [10metric/giới hạn](../research/lite2-dagsnet-r50-audit.json).
  Recovery watcher `mh-lite2-r50-recover-20261117` succeeded; DONE/audit có thật, completion đã xử lý.
- **Owner hoãn push lại Lite2 BERT r50 trên mọi account; ưu tiên FB, không giữ quota riêng cho Lite2.**
  Pilot `annhannh331/flb-lite2-bu1-s42-r20/1` đã PASS, không collect lại.
- T3 DAGSNet/BERT đủr50/3seed, pFedES BERT đầy đủ; [comparison](../research/comparison.md).

## Khi watcher completion

1. Đọc state/log + terminal.json + pulled.ok; kiểm đủ8arm/r20 trước verdict. ERROR vẫn thu log/output.
   COMPLETE/exit0 không thay artifact audit. Query lỗi = UNKNOWN, không restart training.
2. Gate N1–N4: ΔTB r16–20≥−.007 và Δr20≥−.008 vs S0, cost≤1,01×; chọn macroTB cao nhất.
   F0/R48/Z0/ZK report-only, Z0/ZK không tự lênpilot. FAIL dừng arm, không tune/kéo dài.
3. PASS candidate → pilot full-scale DAGSNet + FeatureBERT theo luật đã duyệt; kiểm quota live/compatible source,
   port ba file method byte-identical sang BERT; không dùng Lite2 BERT r50 thay pilotFB. Mixed-model chưa được duyệt.
4. Cập nhật screening-budget/triedindex/lessons từ artifact thật. Ghi bàn giao trước dừng phiên.

Owner đã yêu cầu commit/push `main` cảhai repo. Receipt GitHub/HEAD/Gitbundle: `runs/maintenance-20261117/publish.json` mỗi repo.
Artifact mới giữ local, chưa DVCbackup; không xóa cache/output. [Backup](backup.md). Không cần thao tác tay hiện tại.
