# Continue — bàn giao sống, 01/11/2026

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [vận hành](lessons.md).

## Hiện tại

Owner đã yêu cầu Git/DVC push, archive phần lỗi thời và dọn local sau backup xác minh.
DVC đang push nền; chưa xóa checkpoint/cache. Trạng thái/kết quả backup tại [backup](backup.md).
Không kernel train hoặc collector còn sống. Chưa launch batch sau M.

M0–M15 exactversion1 đủ20round: fullaudit PASS. [Kết quả](../research/results-m20.md),
[10metric](../research/metrics-m20.md), [đề xuất chẩn đoán](../research/diagnosis.md).
54tests PASS; phân tích/audit sau migration khớp JSON/CSV trước migration; 5overlay training giữ nguyên SHA.

## Job backup đang chạy

- Job `mobihet-dvc-backup-20261101`, supervisorPID10111; deliveryqueue(auto), viewer none.
- Ownerthread `01a18d65-e3b8-73f1-80b6-7b66531915da`.
- State `/home/odixe/.codex/long-jobs/jobs/mobihet-dvc-backup-20261101/state.json`.
- Log `/home/odixe/project/mobihet/runs/maintenance-20261101/dvc-backup.log`.
- Theo dõi: `tail -n 200 -F -- /home/odixe/project/mobihet/runs/maintenance-20261101/dvc-backup.log`.
- Worker `runs/maintenance-20261101/push_verify.py`: push đúng68target trong inventory → status-c →
  kiểm size mọiobject/MD5directory +20dataobject → ghi `docs/operations/backup-verification.json` → `BACKUP_DVC_VERIFIED`.
- Worker sở hữu inventory/manifests/cache/payload: giữ bất biến khi chạy; chưa cleanup/Gitpush/bundle.
- Completion: đọc state/log/receipt, kiểm độc lập → Gitpushmain/ls-remote → bundleallrefs/verify →
  cleanup65payload +eligiblecache trong inventory, giữ evidence/log; cập nhật backup/bàn giao, push metadata cuối.
- Không watcher/collector trùng, không modelpollwaiting. Nếu fail giữ toàn bộ local, sửa/retry đúng target.

## Job đã đóng, không chạy lại

Recovery `kaggle-m-v1-recover-collect-audit-20261101`: succeeded/exit0,
`M20_ARTIFACT_AUDIT_PASS` + `M20_PULL_AUDIT_DONE`; 1360file,16×20round đã kiểm artifact.
State `/home/odixe/.codex/long-jobs/jobs/kaggle-m-v1-recover-collect-audit-20261101/state.json`;
log `/home/odixe/project/mobihet/runs/m-study/recover-collect-audit.log`.
Ownerthread `01a18d65-e3b8-73f1-80b6-7b66531915da`, supervisor cũ6298, deliveryqueue(auto), viewer none.
Completion có thể đến muộn: đã xử lý, không recollect/retry delivery.
Watcher trước reboot `kaggle-m-v1-watch-collect-20m-20261031` có state running stale, PID đã mất;
không sửa state/kill PID tái chế/restarttraining. Các job L cũng đã đóng.

## Dữ liệu cần mở lần sau

- `evidence/m20`: config/CM/probe/external/provenance, không modelbank.
- `evidence/m20-summary`: batch/audit/analysis/collection/teacher-tables.npz, source/smoke/equivalence proof.
- `evidence/research-data`: train_counts/test_support5343×16.
- `artifacts/m20/*.dvc`: snapshot đầy đủ; restore chọn lọc khi cần. Archive không đọc mặc định.
- Frozen source/config/data contract ở [spec](../research/screening-m20.md) và provenance, không thay bằng HEAD khi replay.

## Tiếp tục sau bảo trì

1. Kiểm backup state/log/artifact nếu nhận completion; hoàn tất cleanup đúng danh sách đã xác minh.
2. Dựa M20, khóa factorial tách realgate khỏi syntheticfallback; sau đó mixturetarget khỏi realKDweight.
3. Kiểm fullQ so diagonal/offdiagonal-uniform trước tăng packet/đổi generator. Theo own-test/tail/xe nhỏ/absent-class.
4. Giữ20round chẩn đoán/cosine50, seed42/43, DAGSNetW1, protocol chung; không tự50round hoặc chọn HP theo test.
   Thiết kế mới phải ghi evidence → mechanism → falsifier trước khi launch.

Không việc tay bắt buộc hiện tại. Không raw local data/privateweight trong FL; chưa DP/mixedarchitecture/contactdrop proof.
