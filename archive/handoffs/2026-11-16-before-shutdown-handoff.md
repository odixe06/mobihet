> Snapshot milestone16/11; bàn giao sống tại docs/operations/continue.md.

# Bàn giao sống — 16/11/2026

Đọc [CONTEXT](../../CONTEXT.md), [LESSONS](../../LESSONS.md), [bài học vận hành](lessons.md).
Lịch sử job/verdict: [snapshot](../../archive/handoffs/2026-11-16-before-refresh.md); đây là bàn giao hiện hành duy nhất.

## Phương pháp và trạng thái

Owner chọn **T3**, gọi **MobiHet** trên Proposal 2. DAGSNet cả3seed41/42/43 đủr50:
seed41 `runs/t3-s41/audit.json` + `state-audit.json`; seed42/43 `runs/b21-r50/audit.json` (T3P/T343).
B20, B21/T1/T4F đã đóng/audit; không waiter cũ nào sống khi kiểm đầu phiên16/11. Không nối T341 hoặc chạy lại batch cũ.
BERT41 r50;42/43 cần pull/audit head mới từ repo BERT. Không gọi partial là headline50.

## Công việc owner yêu cầu16/11

Ưu tiên collect/audit kernel hoàn thành đa luồng; cập nhật toàn bộ tài liệu còn hiệu lực hai repo, gọn hub;
Notion Proposal2 thay mô tả và bảng MobiHet cũ bằng T3, giữ14baseline. Performance giảm dần macroF1;
communication tăng dần tổng50round gồmbootstrap (cột riêng khôngbootstrap); compute tăng dần. Có8bảng,2model cho mỗi loại performance/communication/clientcompute/servercompute.
Cho phép commit/push Git+DVC cảhai repo. Không xóa artifact, không sửa protocol/HP/source đã khóa.

## Job collect/audit hiện hành — repo BERT sở hữu

`repo-collect-audit-20261116`, supervisor PID269238, owner thread `01a1e31b-3489-76b3-a36f-f8ce699825eb`, delivery queue.
State `/home/odixe/.codex/long-jobs/jobs/repo-collect-audit-20261116/state.json`.
Log `/home/odixe/project/fl_veremi_bert/runs/maintenance-20261116/collect.log`.
Đã kiểm running14:05VN; không tạo collector trùng. Chi tiết việc kế ở `../fl_veremi_bert/docs/operations/continue.md` (đường tính từ root repo này).

## Việc kế sau audit

- Đồng bộ method/proposal/results/index với T3 và3seed thực tế; giữ lịch sử control CBB rõ nhãn.
- Notion `40d730cee68d80f0a49ad050377772d2`: T3/MobiHet duy nhất, số đo/ước lượng có nguồn, không claim server<5ms khi chưa đo.
- Backup artifact mới B20/B21/T341 và receipts; tránh dvc add lại workspace partial (xem [backup](backup.md)).
- Kiểm link/test phù hợp, push DVC+Git+bundle trong quyền owner16/11. Không xóa local hoặc dirty VFS cache.

Chưa có việc tay bắt buộc.

Job độc lập: `mobihet-inventory-20261116`,PID272971,queue,cùng owner thread; log `runs/maintenance-20261116/inventory.log`,state `/home/odixe/.codex/long-jobs/jobs/mobihet-inventory-20261116/state.json`. Hai job server-cost/DAGSNet42 evidence nằm bàn giao repo BERT. Method đã viết lại đúng T3/source; Notion/bảng/backup chờ audit ưu tiên owner.

Server-costr2 đã succeeded và kiểm artifact Np69993. Job verify hai repo `repo-verify-20261116` (repo BERT sở hữu) chạy test/link/T3compute. Collector baseline/T3 vẫn đang audit, chưa publish bảng/backup.

## Completion mobihet-inventory xử lý16/11

Inventory24/24exact kernel COMPLETE. Đã kiểm từng slug có artifact/config local và đối chiếu audit B20/B21/T341; mọi artifact check PASS, riêng T4P utility FAIL vẫn giữ verdict. Không pull trùng24phiên đã lưu. T3DAGSNet3seed r50 macro .907956319 ± .000331909(sampleSD), bảng12metric tại `docs/research/results-t3.md` trong repo mobihet. Collector BERT vẫn sở hữu phần baseline/T3BERT; không khởi trùng.

## Backup MobiHet đang chạy16/11

`mobihet-dvc-backup-20261116`, supervisorPID277319, deliveryqueue về ownerthread `01a1e31b-3489-76b3-a36f-f8ce699825eb`; đã kiểm running. State `/home/odixe/.codex/long-jobs/jobs/mobihet-dvc-backup-20261116/state.json`; log `/home/odixe/project/mobihet/runs/maintenance-20261116/backup.log`. Worker `/tmp/backup_mobihet_20261116.py`: chỉ dvcadd run mới chưa có manifest trong b21-chain/b21-transfer/t3-s41 + sanitized receipts/inventory; push-j16 rồi status-c. Không đè manifest partial cũ, không xóa. Target/log chi tiết trong mobihet/runs/maintenance-20261116/. Sau completion kiểm manifest/status thực, cập nhật backup record; Git/bundle và BERT DVC vẫn chờ.
Theo dõi: `tail -n 200 -F -- /home/odixe/project/mobihet/runs/maintenance-20261116/backup.log`.

## Owner16/11 bổ sung pFedES41 phiên cuối

pFedES41p3 COMPLETE nhưng audit chỉr45, không đủ50. Owner yêu cầu push bổ sungr45→50 và **tạm bỏ pFedES khỏi mọi bảng**. Script comparison đã loại method này (không dùng partial/2seed thay3seed).
Job `pfedes41-final-20261116`,supervisorPID281246,deliveryqueue cùng ownerthread,đã kiểm running; state `/home/odixe/.codex/long-jobs/jobs/pfedes41-final-20261116/state.json`;log `/home/odixe/project/fl_veremi_bert/runs/maintenance-20261116/pfedes41-final.log`. Worker `/tmp/resume_pfedes41_20261116.py` audit2rankr45/finite/gen → chọn owner quota≥4.2h+slot → private checkpointdataset → pushp4cap3.8h (5round×~1682s +startup/save) → watch1200s → fullpull/audit. Emtrangtri quota0, phải chuyểnaccount; chưa claim launched trước receipt.
Historical baseline state không lưu riêng torchCPU/CUDARNG; giữ saved methodgen và resume semantics gốc, không gọi exactfullRNGresume. Source/HP/50roundschedule/privatebanks giữ nguyên; không chạy r1. Không sửa source tốc độ trong phiên cuối. Saucompletion kiểm artifact+round50, rồi mới thêm lại pFedES vào bảng.
Theo dõi: `tail -n 200 -F -- /home/odixe/project/fl_veremi_bert/runs/maintenance-20261116/pfedes41-final.log`.

## Owner yêu cầu chuẩn bị tắt máy

Hoàn tất các task đang dở rồi **dừng phiên**, cập nhật toàn bộ bàn giao để bật máy tiếp tục. pFedES có thể tiếp tục trên Kaggle khi local tắt; không cancel remote. Không tạo nghiên cứu/training mới ngoài phiên cuối đã ủy quyền. Trước dừng: hoàn tất pull/audit/T3/Notion, Git+DVC/bundle; ghi trạng thái cloudVFS riêng và exact run còn sống. Copy worker `/tmp/*.py` dùng trong phiên vào `runs/maintenance-20261116/workers/` (repoBERT) để không mất sau reboot.

MobiHet DVC10manifest/438objects VERIFIED mứcmount (push-j16 exit0/status-c in sync); cloud upload chưa xác minh riêng. RepoBERT baselinebackup đang chạy: `bert-baseline-backup-20261116`,supervisorPID282599,queue;state `/home/odixe/.codex/long-jobs/jobs/bert-baseline-backup-20261116/state.json`;log `/home/odixe/project/fl_veremi_bert/runs/maintenance-20261116/bert-baseline-backup.log`. Khởi tạo DVCOneDrive sibling `fl_veremi_bert/dvc-storage`,hardlink snapshot phiên baseline đãdone, không ghi vào pull collector. T3/receipts backup saucollect.

BERT baselinebackup r1failed trước DVCadd vì phát hiện `v4npy/train_X.npy.part`5.49GB còn từ pull full trước. Đây là cache dataset tạm, không phải model/CM đã audit; không xóa hoặc backup file dở. Worker r2 loại toàn v4npy (dataset cache tái tạo được), tái dùng hardlink đã staging đúnginode; chỉ phần cache này được phép .part, runartifact .part khác vẫn chặnbackup. Job `bert-baseline-backup-20261116-r2`,PID283142,queue;log `runs/maintenance-20261116/bert-baseline-backup-r2.log`,state theo tênjob. DVCinit/remote đã xong; chưa xác nhậnpush trước completion.

Verifyr2: T3compute12PASS, linkactive hai repoPASS; notebook/AFPHAfixture đãsửa ởrepoBERT,9PASS/1SKIP integrationparquetoffline. Credential-patternscan/diffcheck hai repoPASS.

Git source/docsđãpush16/11: BERT `72a5b024ae78f6f3b5b3d999f5c81612911c7518`, MobiHet `5dfae538428b287b649b40fee36b309e8ce39dc8`, main; còncommitbàn giao/manifestcuối/bundle saubackup. Notion8bảng đãreplace/readbackPASS,childpagegiữnguyên; source kếtquảhiện hành `docs/research/comparison.md`/`.json`.
Notifier `pfedes41-launch-receipt-20261116`,PID287248,queue;logBERT `runs/maintenance-20261116/pfedes-launch-receipt.log`; chỉchờreceiptlocal,cấmqueryKaggle hoặclaunchthêm. MainpFedESworker vẫnowns staging/push/watch; notifierbáo sớm khiđãpush đểchốtbàn giao/tắtmáy, khôngchờtraining3.8h. State theo tênjob ởlong-jobs/jobs;workerđãcopydurable trongruns/maintenance/workers.

BERTT3/evidencebackup-r2 completion VERIFIED:811objects pushed/status-c in sync;T3logical10.990GB/896files,evidence20.06MB/720files. BackupBERT3manifest đãxongmứcmount. OneDrivecloud còn7762VFSdirtyfiles lúc kiểm; không xóaVFS/cache hoặc gọicloudsyncxong. Localartifact cònnguyên; rcloneupload tiếp khi máy/mount chạy lại. pFedES đangstagingdatasetoutlookcheck (quota19.63hđãgate), chưaclaimpush trướcreceipt.
