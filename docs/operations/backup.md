# Git / DVC backup record

Updated: 01/11/2026, Asia/Ho_Chi_Minh — DVC/Git/bundle xác minh; cleanup hoàn tất.
Procedure: [repo-backup](../../.agents/skills/repo-backup/SKILL.md).

## 1. Destinations

- Git `https://github.com/odixe06/mobihet.git`, branch `main`. Backup trước cleanup: `256b741318c81bbea9da2f247650dfb5b3380f00`, ls-remote khớp. Metadata/source N đã push `00c8fba5a34ccf320f752a931ef424f357943eb3`; runtime metadata nằm ở commit tiếp theo. `git ls-remote origin refs/heads/main` là latest authoritative.
- DVC `onedrive`: `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/mobihet-v2/dvc-storage`.
- Bundle cố định `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/mobihet-v2/mobihet.bundle`: fullhistory/allrefs, verify PASS trước cleanup; được thay bằng bundle đã verify sau commit metadata cuối.
- Mount `onedrive:`/`fuse.rclone` đã kiểm; filesystem backup PASS. Cloud upload **unchecked**; giữ nguyên rclone/VFSuploadcache.

## 2. Tracked artifacts

[Inventory đầy đủ](backup-inventory.json):68manifest,2540object,30.696.795.246B duy nhất. Giữ3evidence, xóa65payload local; manifest còn nguyên.

| Manifest | Directory MD5 | Files | Logical bytes | Local workspace |
|---|---|---:|---:|---|
|`archive/artifacts/batches/h-screening.dvc`|`abc4273a2038cee31a1cd8b5140f9085.dir`|302|712581556|deleted 01/11/2026|
|`archive/artifacts/batches/j-screening.dvc`|`efb800c5c24454219ddad7ae152b7650.dir`|256|368621295|deleted 01/11/2026|
|`archive/artifacts/batches/k-screening.dvc`|`ad5c6275ced9773851d7149029740a58.dir`|259|914885361|deleted 01/11/2026|
|`archive/artifacts/batches/l-screening.dvc`|`21a69c0194659ce4640665f87b819449.dir`|133|38107709|deleted 01/11/2026|
|`archive/artifacts/batches/m-study.dvc`|`67e7adfdc8f0104f750bb05f574c5076.dir`|161|34527154|deleted 01/11/2026|
|`archive/artifacts/local-cache.dvc`|`bee24986d46d4b7b668b1e7f0232dfc2.dir`|31|186947122|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-g0-off.dvc`|`4831345d4a51193c59259505c722101c.dir`|23|384791609|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-g1-oracle.dvc`|`ecdf5b2fe8777255204a8a90b1843763.dir`|23|384595644|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-g2-short.dvc`|`69d4669dbb42e5c33881108ce3d2b519.dir`|23|384770987|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-g3-mix.dvc`|`368ce2a20b959b5e20d82a0b8f05b4f4.dir`|23|384759322|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-g4-copula.dvc`|`f1169be8d46dbb11530f7f1a6aed1139.dir`|23|384669068|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-h0-copula-s43.dvc`|`0be170c283483bb84787f54eac35c375.dir`|48|384779185|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-h0-gauss-s43.dvc`|`9d1f46b79c5a7526df40e197ba102d77.dir`|48|384824696|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-h1-cluster.dvc`|`1c7311ae08e60a397f68dceff58c4d0c.dir`|48|384786723|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-h2-copulamix.dvc`|`6ce4d0babb9f55ecbc0ae47b349e3a20.dir`|48|384857562|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-h3-dose.dvc`|`4fc47bf2b4d9d0b334412445b63823b3.dir`|48|384758126|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-h4-onehot.dvc`|`0541ef9b0f36732e3e43dfd852bc8fe4.dir`|48|385002001|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-j0-anchor-class.dvc`|`4f545b12127300980218b9ff8529207a.dir`|48|384831109|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-j1-anchor-kd.dvc`|`f06a7dc4b31a4c2d46da31be6e59ba1a.dir`|48|387482216|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-j2-gauss-kd.dvc`|`d42aaa273d165c40ce0bb8974a01e6eb.dir`|48|387490700|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-j3-anchor-dose.dvc`|`61540c4a453a1687a14c11b615bae7ed.dir`|48|387480031|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-j4-no-real-kd.dvc`|`c91a23a8c7e3d1210be4f45ed8fbbcce.dir`|48|385046577|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-k0-combo-s42.dvc`|`e17640ec8c271326157ef4510da2753a.dir`|48|385002342|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-k1-combo-s43.dvc`|`659cf30c69c5e9ac24249538c4585836.dir`|48|384985144|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-k2-dose-s43.dvc`|`b3a8bb7c9c13736d790c3668c2557b22.dir`|48|384753314|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-k3-noreal-s43.dvc`|`e8c245feaed63896ea19444065049f33.dir`|48|385033735|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-k4-aug-s42.dvc`|`2c669878ef25842cde877a17c0dfc49a.dir`|48|385138715|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-k5-aug-s43.dvc`|`810b6e1d6e0c4b29d9d7540b45d77c76.dir`|48|385124248|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-k6-bucket-s42.dvc`|`20afbeddc82ca0c82c9dba825c586916.dir`|48|385089873|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-k7-bucket-s43.dvc`|`bc68a56d7d7d4638c4c4645cd72cb575.dir`|48|385070246|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-k8-gate-s42.dvc`|`d6b9a968ed3c799d414acbf785d2932a.dir`|48|384802645|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-k9-gate-s43.dvc`|`4ce16c64f56d2c4d558601a0ac23c03e.dir`|48|384787103|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-l0-softfix-s42.dvc`|`9618f720e8787018819e78b959e9a2e4.dir`|55|387570020|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-l1-softfix-s43.dvc`|`632cc4ba7394636db7defacfa9653f15.dir`|55|387549727|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-l10-gatewire-s42.dvc`|`e9ed7bfd62723dac78db36adda998c91.dir`|55|387269390|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-l11-gatewire-s43.dvc`|`2d2f1aa13d8bd327cd6953d9458f3f7d.dir`|55|387249470|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-l2-hardadapt-s42.dvc`|`a2d24d2729b522f104e73809fd7274a2.dir`|55|388006061|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-l3-hardadapt-s43.dvc`|`3e018a0ec3443e1864c47d21d55cd4ea.dir`|55|387987264|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-l4-hardgatefix-s42.dvc`|`8fc329ee792e1ec8696d94dc2a250d45.dir`|55|387524462|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-l5-hardgatefix-s43.dvc`|`7405260fc7871519a794fc153cd7492b.dir`|55|387514748|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-l6-softgatefix-s42.dvc`|`97dc85abd578ec7b69e0b38682973729.dir`|55|387324478|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-l7-softgatefix-s43.dvc`|`6de2ac58c649afa8d2826f1667e40a10.dir`|55|387304617|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-l8-combowire-s42.dvc`|`e9868247f6c7581fab23590e047a083f.dir`|55|387659838|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-l9-combowire-s43.dvc`|`81306a8b8594fcd7ce514eb9f1c57316.dir`|55|387638281|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-s0-dml.dvc`|`26718d8e8b7760b66c2e7cfef0c341e1.dir`|21|349430202|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-s1-srv.dvc`|`9f3ac66d34e21c404ddfce1314ba0665.dir`|23|389524553|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-s2-srvclient.dvc`|`ce830de48112d47dbcc822a10a3f9e2a.dir`|23|389482292|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-s3-cheap.dvc`|`6cb9f622e49cf8af85261c46e9db41fb.dir`|23|391350334|deleted 01/11/2026|
|`archive/artifacts/pulls/mh-s4-none.dvc`|`14cbee5317d3f8c94c9926350ca7546e.dir`|23|384713055|deleted 01/11/2026|
|`artifacts/m20/mh-m0-aug-s42.dvc`|`e8a9340a03d427b86847b77bac390732.dir`|85|773472267|deleted 01/11/2026|
|`artifacts/m20/mh-m1-aug-s43.dvc`|`a6eeefd683d8852ffeb16f92a9fdc7ac.dir`|85|773436956|deleted 01/11/2026|
|`artifacts/m20/mh-m10-sqrt-s42.dvc`|`8b9d7832a803c3c15e46967fd84f0f8b.dir`|85|772935012|deleted 01/11/2026|
|`artifacts/m20/mh-m11-sqrt-s43.dvc`|`bd1411d80863dc685d88a0323cf5e498.dir`|85|772904448|deleted 01/11/2026|
|`artifacts/m20/mh-m12-uniform-s42.dvc`|`fe0b67180012bc6b246cbf9bc560b629.dir`|85|773339414|deleted 01/11/2026|
|`artifacts/m20/mh-m13-uniform-s43.dvc`|`ddefe707d8fa692ec534c31953cbacb6.dir`|85|773315320|deleted 01/11/2026|
|`artifacts/m20/mh-m14-local-s42.dvc`|`48c62202ce43a01bf62399f4b55ddcc5.dir`|85|773575029|deleted 01/11/2026|
|`artifacts/m20/mh-m15-local-s43.dvc`|`7b4205a2a8e7ab45324c0ba1057071b1.dir`|85|773540861|deleted 01/11/2026|
|`artifacts/m20/mh-m2-soft-s42.dvc`|`857e3330be95f95d9acf3210b25f8ea7.dir`|85|773459648|deleted 01/11/2026|
|`artifacts/m20/mh-m3-soft-s43.dvc`|`91858dcad3c35a69c06271ec822e283e.dir`|85|773423596|deleted 01/11/2026|
|`artifacts/m20/mh-m4-gate-s42.dvc`|`2831bf3888ac936655fe635463b0ccbf.dir`|85|772982285|deleted 01/11/2026|
|`artifacts/m20/mh-m5-gate-s43.dvc`|`cd1b9c178bc5bb6eb58a8433f5646246.dir`|85|772948079|deleted 01/11/2026|
|`artifacts/m20/mh-m6-fallback-s42.dvc`|`883ba21cc9c0f4f0be167992ddbd6dfd.dir`|85|773135491|deleted 01/11/2026|
|`artifacts/m20/mh-m7-fallback-s43.dvc`|`9929d34492b09d91098cf8e1c39150eb.dir`|85|773117915|deleted 01/11/2026|
|`artifacts/m20/mh-m8-post-s42.dvc`|`3a78371ca628aea2d024d106e4bbd769.dir`|85|773041165|deleted 01/11/2026|
|`artifacts/m20/mh-m9-post-s43.dvc`|`6a5abfe09efbd0faba9cdb367ae0a128.dir`|85|773005609|deleted 01/11/2026|
|`evidence/m20-summary.dvc`|`7c8a584244ad815af2d617891f0764f1.dir`|13|4308003|present|
|`evidence/m20.dvc`|`260e366299d5d12f545d675310e7faed.dir`|1040|190814895|present|
|`evidence/research-data.dvc`|`7c5204f7f20792663c9cfc76541c8101.dir`|2|1368064|present|

Không re-add workspace partial/đã xóa. Archive không đọc mặc định; chỉ restore khi cần bằng chứng cụ thể.

## 3. Checks that passed (reusable while inputs are unchanged)

| Check | Scope | Result |
|---|---|---|
| `dvc push` / `dvc status -c` |68manifest trước cleanup| exit0, `2540 files pushed`, in sync |
| Remote object audit |2540object size;68directory MD5;20data MD5 +3data độc lập| PASS, không fullchecksum mọidata |
| Git push / bundleverify |main/allrefs| `256b741` khớp ls-remote và bundle trước cleanup |
| Migration/research |5frozen trainingfile;JSON/CSV audit/analysis;links;54tests| PASS; phân tích sâu có paired/formula checks riêng |
| Evidence sau cleanup |627cacheobject được giữ| mọi MD5 PASS |
Chi tiết [backup-verification](backup-verification.json), [cleanup-verification](cleanup-verification.json); credential hiện dùng không xuất hiện trong file public/artifacttext được kiểm.

## 4. Local cleanup

- Owner yêu cầu01/11/2026. Đã xóa đúng65target trong `workspace_payloads_deleted` của receipt +1913cacheobject = inventory.objects trừ keep_cache; 30,502,462,664B cache logical.
- Disk free trước=106,320,158,720B; sau=137,378,594,816B; tăng thực tế **31,058,436,096B (31.06GB)**. Có I/O khác/VFScache, không đồng nhất với byte logical.
- Xóa notebook generated cũ (có thể chứa credential), output audit/analysis trùng và bản runs/m-study trùng; danh sách chính xác ở receipt. Giữ log completion gốc.
- Giữ evidence190MB, source/docs/manifest/Githistory, `runs/diagnosis-m20` và maintenancejournal. Dataset gốc ngoài repo không đụng tới.
- Không xóa cloudobject/sharedcache/credential/rclonecache. Không việc tay bắt buộc.

## 5. Next session

- Push mới bằng target cụ thể: `dvc add <complete-dir>` rồi `dvc push <manifest>.dvc`; không add lại target cũ đã cleanup.
- Phân tích M không cần modelrestore: `python scripts/analyse_m20.py`, `python scripts/diagnose_m20.py`, `python scripts/report_m20.py`.
- Restore một run: `dvc pull artifacts/m20/mh-m0-aug-s42.dvc` (~0,78GB); toànM~12,4GB. Kiểm disk trước, restore chọn lọc.
- `deleted` trong cloudstatus sau cleanup có thể chỉ thiếu localcache; không suy remote mất. Không upload lại history đã xác minh.
- Input/manifest/remote đổi làm mất hiệu lực kiểm tương ứng. Metadata/checkpoint mới của batch tiếp chưa thuộc68manifest backup này.
- Nghiên cứu/job hiện tại ở [continue](continue.md); không đọc archive để suy trạng thái live.
