Snapshot mốc pilot20; bàn giao sống: docs/operations/continue.md. Không dùng snapshot làm trạng thái hiện hành.

# Bàn giao sống — kiểm 07/11/2026 11:27 UTC+7

Đọc [CONTEXT](../../CONTEXT.md), [LESSONS](../../LESSONS.md), [operations lessons](lessons.md).
Đây là **bàn giao hiện hành duy nhất**. State/log/receipt thật có ưu tiên hơn snapshot tài liệu.

## Quyền và ưu tiên mới nhất

1. Nối các kernel chưa đủ round; được chuyển account, ưu tiên quota còn>3h, pull đa luồng.
2. Owner02:39 cho thử ứng viên khả thi **ngay trên quota dư**. Bỏ lệnh hoãn toàn bộ đến07:00 lúc02:28;
   batch dài chờ quota hồi, vẫn kiểm quota/slot thật trước push. Bốn receiver Y50 giữ ưu tiên.
3. Được vòng thử generator → phân tích FAIL → thử cơ chế tiếp. Giữ protocol/model/loss/Q/floor khi isolate generator;
   không chọn HP/round bằng official test, không nới verdict cũ. **Owner chốt pilot generator tổng20round**,
   giữ cosine50, dừng audit/phân tích; không tự nối50. Lịch utility50 đã hủy trước push.
4. Credential theo process, global login giữ nguyên; W&B online/private inline key đã được phép, history giữ key vĩnh viễn.
   Không Git/DVC push, xóa artifact hoặc sửa source/ledger worker đang sở hữu.

## Kernel cần theo dõi

| arm / seed | round đã audit → đích | exact session / bước hiện tại |
|---|---|---|
| Y3 noQ42 | 27→50 | parent `novahihi/mh-y3-noq-s42-r50-c2/1`; fullstate đã verify, receiver `dinhan25` đang verify private dataset |
| Y4 noQ43 | 26→50 | parent `nguyetquedo/mh-y4-noq-s43-r50-c2/1`; receiver `minhtriethihi`, dataset ready/SHA verified |
| Y5 nofloor42 | 20→50 | parent `accanniez/mh-y5-nofloor-s42-r50-c2/1`; receiver `anhtruong1909`, dataset ready/SHA verified |
| Y6 nofloor43 | 47→50 | parent `outlookcheck/mh-y6-nofloor-s43-r50-c2/1`; receiver `heykewtiie`, dataset ready/SHA verified |
| Z0 standard42 | 20/20 audit PASS | `trietbackup/mh-z0-stdcop-s42-t4r2-p20-c3/1`; pilot gate FAIL, đã dừng |
| Z1 standard43 | 20/20 audit PASS | `odixeuit/mh-z1-stdcop-s43-t4r2-p20-c3/1`; pilot gate FAIL, đã dừng |
| Z2 forest42 | 20/20 audit PASS | `minhtrit06/mh-z2-forestq-s42-t4r2-p20-c2/1`; pilot gate FAIL, đã dừng |
| Z3 forest43 | 20/20 audit PASS | `emtrangtri/mh-z3-forestq-s43-t4r2-p20-c2/1`; pilot gate FAIL, đã dừng |

Forest đã đóng2r/2seed auditPASS03:48:23; `runs/forest-t4/{closure,artifact-recheck}.json` và[runtime results](../research/results-generator-runtime.md).
R2 kémCore một sốmetric/cohort, chưa làutilityFAIL/PASS; so sánh pilot với Core cùng r20, không chọn HP từ r2.
Hai seed khác nhau là42/43; Y50 có4run ablation, không phải4seed độc lập.
Y parent đều COMPLETE nhưng thiếu đích. Pull đã xong03:11:50, đủ8state≈34,12GB và4compilecachetar;
verify round/config/seed/RNG/private shapes/source overlay bằng weights_only+mmap PASS tại
`runs/y-transfer/checkpoint-artifact-audit.json`. Không phải fulltensorfinite/inference hoặc archiveSHA end-to-end proof.
Transferworker kiểm identity PID/start đúng lúc11:27; Y4/Y5/Y6 private datasets ready/verified, Y3 đã create và đang verify.
Worker đợi đủ bốn ready trước push; không bật watcher Y50 khác.

## Worker sống — không bật trùng

Owner thread: `01a1b116-77ad-7d22-b83b-86020adaa518`, delivery **queue**.
State: `/home/odixe/.codex/long-jobs/jobs/<job ID>/state.json`. Transferworker PID/start identity đã đối chiếu11:27;
utility50 cũ canceled04:04:27 trước push; pilot20 succeeded11:22:41, re-audit artifact đủ4run/r20 PASS.
Pullworker/forestruntimeworker/pilot20 đã đóng; không bật lại.

| job ID | PID | log | file/tác vụ sở hữu |
|---|---:|---|---|
| `y50-crossaccount-resume-20261107` | 14247 | `runs/y-transfer/pipeline.log`, `<arm>/stage.log` | plan/private datasets/notebook-x3+/canonicalY50ledger/collect/audit |

```bash
tail -n 100 -F runs/y-transfer/pipeline.log
```

Workers poll Kaggle1200s; khi chỉ chờ thì kết thúc turn, completion đánh thức owner thread.
Phiên mới đọc state/process identity trước mọi retry. Queue vẫn gửi về thread trên; mở thread khác không tự đổi ownership
hoặc tạo quyền bật watcher thay thế. `run_y50.py`/`keep_y50.py` không chạy cùng transferworker đang giữ `wave2/worker.lock`.

## Pipeline tự tiếp tục

1. Pull exact4parent/version1 đã xong; `pull-complete.json` SHA khớp `parents.json`, source/cache/state audit PASS.
2. Transfer worker đã nhận marker, verify round/config/RNG2rank; Y4/Y5/Y6 private datasets đã ready/SHA verify, đang đợi Y3.
   Sau đó mới push `-x3`, giữ out/W&B ID gốc, restore npy/Inductor cache; private kernel không mount chéo account.
3. Y50 audit partial trước autochain cùng owner khi cần; r50 mới publish canonical `runs/y-study/wave2/closure.json`
   và `analyse_y50.py`. Giữ [luật Y50](../research/screening-y50.md), không coi r47 làr50.
4. Pilot20 đã đóng, đủ4run/audit/source/CM/teacher/compile/cache/resume2rank. Standard và forest đều FAIL gate;
   [kết quả/diagnosis/đủ10metric](../research/results-generator-pilot20.md), `runs/generator-pilot20/{analysis,diagnosis,artifact-recheck}.json`.
   Giữ Core; không nối50 hoặc tune bằng official test. Cơ chế mới cần bằng chứng train/val và falsifiable gate trước launch.

## Đã đóng — không chạy lại khi nhận notification muộn

- `generator-pilot20-0700-20261107` succeeded11:22:41, đủ4run đúng20; đã re-audit thực và đóng báo cáo. Không restart.
- `y50-checkpoint-pull-20261107` succeeded03:11:50, exit0/marker; artifact đã recheck. Log `runs/y-transfer/pull.log`.
- Y0 oracle42 đủr50; Y1/Y2 prior FAILr20. [Kết quả](../research/results-y50.md).
- Standard Z0/Z1 đủr2 hai seed, resume2rank/cache restore/audit PASS; [runtime evidence](../research/results-generator-runtime.md).
  Job `standard-t4-resume-20261107` succeeded02:28:58. Chưa utility50.
- Forest local preflight/job `forest-quantile-preflight-20261107` và GPU sampler PASS, artifacts đã recheck; không refit.
- Forestruntime `forest-quantile-t4-now-20261107` succeeded03:48:23, audit2r/source/CM/teacher/staticstats/finitePASS, skip0;
  chưautility50. Khôngrestartlauncher2r từnotificationmuộn.
- `generator-utility50-0700-20261107` canceled04:04:27: owner yêu cầu pilot20; chưa GPU push.
  Giữ `runs/generator-utility50/` và source cũ làm evidence, không bật lại từ notification muộn.
  Notification hủy đã xử lý; kiểm SHA notebook/metadata và không có launch receipt: `runs/generator-utility50/cancellation-artifact-check.json`.
- Canceled local dispatcher/watchers có chủ đích: `y50-wave2-resume-20261107` (handover),
  `forest-quantile-t4-20261107` (hoãn), `forest-quantile-t4-0700-20261107` (owner cho chạy ngay). Không remote training bị hủy.

## Locator và việc tay

Quota/owner allocation chỉ là snapshot: `runs/quota-now-20261107/{quota,kernel-status}.json`,
`runs/y-transfer/{parents,plan,quota}.json`, `runs/forest-t4/{batch,launch-complete,status}.json`.
Pilot20: `runs/generator-pilot20/{plan,closure,analysis,diagnosis,artifact-recheck}.json`; launcher đã đóng, source frozen giữ evidence.
Phân tích lại bằng `scripts/analyse_generator_pilot20.py`, không chạy launcher để tái tạo báo cáo.
Y snapshot audit: `runs/y-study/wave2/collection/Y{3,4,5,6}-r{27,26,20,47}/`; recheck
`runs/y-transfer/cancelled-watcher-artifact-check.json`. Nguồn/config frozen trong provenance và batch manifests.

Không có bước tay bắt buộc hiện tại; giữ host bật cho worker/delivery. Reboot cần phục hồi collector theo identity/lock,
không restart kernel vì watcher mất. Y/Z mới và docs/code phiên này chưa backup mới; không cleanup/dvc add vào workspace partial.
Workspace đang dirty, chưa commit/push. Migration docs: [index](../../archive/docs-20261107/README.md),
`runs/docs-handoff-20261107/` giữ manifest và invariants. Kiểm active links bằng `python scripts/check_docs.py`.
