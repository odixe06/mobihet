# Continue — bàn giao sống, 01/11/2026

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [vận hành](lessons.md).

## Hiện tại

Owner đã yêu cầu Git/DVC push, archive phần lỗi thời và dọn local sau backup xác minh.
Backup đã xác minh và cleanup hoàn tất; cập nhật bundle theo commit metadata cuối. Trạng thái/kết quả backup tại [backup](backup.md).
Owner đã yêu cầu bắt đầu kiểm chứng: batch N20 (4ô factorial ×2seed) đã khóa, worker đang launch rồi watch/collect/audit.

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

## N20 đang launch/watch — một worker duy nhất

[Thiết kế khóa](../research/screening-n20.md), [manifest public khôngcredential](../research/n20-launch.json).
4ô realKDoff/gate × synthsoft/fallback,2seed42/43,20round/cosine50/full5343xe. 5trainingoverlay byte-identical M.
BaseGit `00c8fba5a34ccf320f752a931ef424f357943eb3`; batchSHA `f6f35bb881f251544fedb6c7d120eb7455533ab52d93bb73a0dba2a65f47f158`.
8account riêng,23,45–25,88hfree/account,reserved0 lúc khóa,identityintrospected; cap6h/run+margin. Globaldinhan25 không đổi.

- Job **kaggle-n-v1-watch-collect-20m-20261101**, supervisorPID14570, ownerthread `01a18d65-e3b8-73f1-80b6-7b66531915da`, deliveryqueue(auto),viewer none.
- State `/home/odixe/.codex/long-jobs/jobs/kaggle-n-v1-watch-collect-20m-20261101/state.json`.
- Log `/home/odixe/project/mobihet/runs/n-study/watch-collect.log`.
- Theo dõi: `tail -n 200 -F -- /home/odixe/project/mobihet/runs/n-study/watch-collect.log`.
- Worker `run_n20_batch.py`: push once theo SHA/intent → startupW&B/logs +exactversionstatus1200s → pull4×8connections → audit8×20.
  Startup chỉ đọc, cần W&B metric thực/world2/backend và log2T4/compile2rank; chưaready thì worker kiểm tiếp, không modelpoll.
- Manifest/runtime: `runs/n-study/batch.json`, *.push-intent.json/*.push.json, launch-complete.json;
  collection/watch/{latest,terminal,attention}.json, collection/startup/startup.json. Artifact tới `artifacts/n20/SLUG/SLUG`.
- Sau lỗi push/timeout: đọc intent/remote trước retry, không launchtrùng. Khi kernelERROR: worker pull terminal đểgiữbằngchứng rồi báo lỗi;
  các kernel khỏe tiếp tục, không tựcancel/rerun. COMPLETE chưa thay CM/Q/source/audit.
- File worker sở hữu/read-only tới terminal: run_n20_batch/launch_n20/watch_collect_screening/watch_kaggle_batch/check_screening_startup/
  audit_m20/audit_screening/run_bounded_python/kaggle_cli_bounded,5trainingfile, batch và8notebook private. Không sửa source đang chạy.

54tests PASS30,23s. Realpreflight2xe ids613/2029,173train/46test,W1,batch64/synth64/localgroup128:
compilefinite (grad_rel.0524),4ô round1bitexact,codec544B/chiều/CM10metric/fullstate+strictmodelrebuild PASS.
`n20-local-preflight-fix-20261101` succeeded; fixture đầu `n20-local-preflight-20261101` failed do2xe/4slotgate, đã xử lý.
Jobpreflight có ownerthreadgốc/deliveryqueue/viewer none, supervisor cũ12382/13202;
state `/home/odixe/.codex/long-jobs/jobs/<job>/state.json`; logs `runs/n-study/local-preflight{,-fix}.log`.
Notification đến muộn không chạy lại. Localsm86 không là T4/exactresume/utilityproof.
W&Bentity `21522798-uit`, project `mobihet-v4`, runID=slug. Inlinekey theo quyền cósẵn, chỉnotebookprivate/ignored;
khôngGit/DVC, key ở lịch sử private Kaggle vĩnh viễn.

## Tiếp tục sau bảo trì

1. Để worker N20 chạy; khi completion đọc state/log/intent/artifact, audit đủ8×20 trước kết luận, không khởi động lại M.
2. Phân tích Nfactorial và interaction theo per-seed/class/cohort; sau đó mới chốt cải tiến tiếp.
3. Kiểm fullQ so diagonal/offdiagonal-uniform/permutation trước tăng packet/đổi generator. Theo own-test/tail/xe nhỏ/absent-class.
4. Giữ20round chẩn đoán/cosine50, seed42/43, DAGSNetW1, protocol chung; không tự50round hoặc chọn HP theo test.
   Thiết kế mới phải ghi evidence → mechanism → falsifier trước khi launch.

Không việc tay bắt buộc hiện tại. Không raw local data/privateweight trong FL; chưa DP/mixedarchitecture/contactdrop proof.
