# Continue — bàn giao sống, 01/11/2026

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [vận hành](lessons.md).

## Hiện tại

Owner đã yêu cầu Git/DVC push, archive phần lỗi thời và dọn local sau backup xác minh.
Backup đã xác minh và cleanup hoàn tất; metadata/bundle cuối đang cập nhật. Trạng thái/kết quả backup tại [backup](backup.md).
Owner vừa yêu cầu bắt đầu kiểm chứng: đang preflight batch N20 (4ô factorial ×2seed), chưa push kernel.

M0–M15 exactversion1 đủ20round: fullaudit PASS. [Kết quả](../research/results-m20.md),
[10metric](../research/metrics-m20.md), [đề xuất chẩn đoán](../research/diagnosis.md).
54tests PASS; phân tích/audit sau migration khớp JSON/CSV trước migration; 5overlay training giữ nguyên SHA.

## Backup đã hoàn tất

`mobihet-dvc-backup-20261101`: succeeded02:24:44/exit0, marker `BACKUP_DVC_VERIFIED`; đã kiểm độc lập artifact.
Supervisor cũ10111, ownerthread `01a18d65-e3b8-73f1-80b6-7b66531915da`, deliveryqueue(auto), viewer none.
State `/home/odixe/.codex/long-jobs/jobs/mobihet-dvc-backup-20261101/state.json`;
log `/home/odixe/project/mobihet/runs/maintenance-20261101/dvc-backup.log`. Không retry/restart job.
DVC/Git/bundle đã kiểm trước cleanup;65payload/1913cacheobject đã xóa, giữ evidence và logs. [Receipt/cách restore](backup.md).
Còn cập nhật bundle cho commit metadata cuối, không upload lại68target.

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

## Phân tích trong lúc DVC push

Owner yêu cầu phân tích tiếp và reference gắn đúng cơ chế đã thử.
[Phân tích sâu](../research/analysis-m20-mechanisms.md), [đối chứng đề xuất](../research/diagnostic-next.md);
`python scripts/diagnose_m20.py` tái tạo JSON+figure từ evidence nhẹ. Phân tích không sửa payload/evidence đã backup.
Bài học mới đã ghi LESSONS; FD/KD reference gắn code/Mrun và deviation tại method.md. Không thêm paper chưa thử vào methodreference.
Phân tích đã commit c7fa5cb và push; `runs/diagnosis-m20` là output tái tạo được, giữ local khi cleanup.
Chưa train/launch mới hoặc đổi protocol.

## N20 đã được owner cho chạy, đang khóa/push

[Thiết kế khóa](../research/screening-n20.md):4ô realKDoff/gate × synthsoft/fallback,2seed42/43,20round/cosine50/full5343xe.
8account riêng trong `runs/n-study/quota.json`:23,45–25,88hfree/account,reserved0,identityintrospected; cap6h/run+margin. Globaldinhan25 không đổi.
54tests PASS30,23s. `n20-local-preflight-fix-20261101` succeeded:173train/46testrows,ids613/2029,W1,localgroup128;
compilefinite (grad_rel.0524),4ô round1bitexact,codec544B/chiều/CM10metric/fullstate+strictmodelrebuild PASS.
Fixture đầu `n20-local-preflight-20261101` failed do2xe/4slot gate; đã đóng, không retrydelivery/rerun.
Hai job có ownerthreadgốc, deliveryqueue(auto),viewer none; supervisor cũ12382/13202;
state tại `/home/odixe/.codex/long-jobs/jobs/<job>/state.json`, logs `runs/n-study/local-preflight.log` và`local-preflight-fix.log`.
Đã xử lý completion từ state/artifact; notification đến muộn không khởi chạy lại.
Chưa exactresume/T4proof; compilegate từng rank và W&Bstartup phải xác nhận từ kernel.

Scripts active: prepare_n20/launch_n20, watcher/collector generic; batch/notebookSHA khóa trướcpush, PUSHEDreceipt/intent chống launchtrùng.
Bước đang làm: commit/push source+cleanupmetadata → generate8private notebooks → launch → status/W&B → watcher1200s.
W&Binlinekey theo quyền đã có, chỉnotebookprivate/ignored, khôngGit/DVC; key ở lịch sử private Kaggle vĩnh viễn.

## Tiếp tục sau bảo trì

1. Hoàn tất metadata/bundle cuối rồi chạy N20 theo cổng preflight/quota, không khởi động lại M.
2. Dựa M20, khóa factorial tách realgate khỏi syntheticfallback; sau đó mixturetarget khỏi realKDweight.
3. Kiểm fullQ so diagonal/offdiagonal-uniform trước tăng packet/đổi generator. Theo own-test/tail/xe nhỏ/absent-class.
4. Giữ20round chẩn đoán/cosine50, seed42/43, DAGSNetW1, protocol chung; không tự50round hoặc chọn HP theo test.
   Thiết kế mới phải ghi evidence → mechanism → falsifier trước khi launch.

Không việc tay bắt buộc hiện tại. Không raw local data/privateweight trong FL; chưa DP/mixedarchitecture/contactdrop proof.
