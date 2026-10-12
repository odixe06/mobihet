# Git / DVC backup record

Updated: 2026-10-12 UTC+7 — backup bổ sung9 target hoàn tất trênG:,verify+cleanup xong; không chờcloudsync.
Procedure: [repo-backup](../../.agents/skills/repo-backup/SKILL.md).

## 1. Destinations

- Git: `git@github.com:odixe06/mobihet.git`, `main`; source/manifest snapshot `67d990982f822b27dbf52167391528925eedd679`, hồ sơ throttle `1678163ab4431a5899cee7c0b5da5d0aa5430e12` khớpremote trướccleanup. Hồ sơ nàycommit/push tiếp; HEAD kiểm riêng bằng `git ls-remote origin refs/heads/main`.
- DVC `gdrive`: `/mnt/g/My Drive/NCKH/veremi_build/mobihet/dvc-storage`, chungarchive; khôngGCremote.
- Bundle: `/mnt/g/My Drive/NCKH/veremi_build/mobihet/mobihet.bundle`; tạo từHEAD cuối, `--all`; verify/hash trước thaytrênG:.
- Cloud upload chưa kiểm/khôngchờ; đủobject+manifest trênG: làgatecleanup. Giữ cacheDrive `D:\ggdrive_cached`.

## 2. Tracked artifacts

[Chỉ mục35 manifest+lệnhpull](artifact-index.md); [inventory/proof/alias/cleanup mới](backup-delta-2026-10-12.json).
Targetpartial đượcgộp oldentry+localnew theo relpath, localoverride cùngpath; giữ4.487 entryremote-only.
Không tạo bảntrùng từ158 `runs/` symlink; mappingrelative cótronginventory.

| Manifest cập nhật/mới | MD5 | Files | Logical bytes | Local |
|---|---|---:|---:|---|
| `papers/pilot/fixture.dvc` | `cb782c4671b1aa25a8a2c56bf465587c.dir` | 67 | 159554510 | đã xóa |
| `papers/v2/kaggle.dvc` | `3c49c7ec142513d62f8e31da23534ee1.dir` | 2422 | 31674561783 | đã xóa |
| `papers/v3/kaggle.dvc` | `6c988e51b04beb21f4fecffd3d68f193.dir` | 4556 | 134856256251 | đã xóa |
| `papers/v4/kaggle.dvc` | `f216e372ed5243268f08347ab0e0bd27.dir` | 1784 | 19725857055 | đã xóa |
| `papers/v5/kaggle.dvc` | `d55a71674bddb8f37e0f674d21e32a74.dir` | 3802 | 44386753600 | đã xóa |
| `papers/v6/input.dvc` | `04561c8e53872cc5f20d0c1a22514e68.dir` | 396 | 639024584 | đã xóa |
| `papers/v6/kaggle.dvc` | `ec521a9fcccb9633cf60cf28bf88911a.dir` | 1666 | 14056604948 | đã xóa |
| `papers/v6/resume_reference.dvc` | `0c303ed9c577e514a3166ba867acce3a.dir` | 60 | 438528517 | đã xóa |
| `papers/v6/smoke.dvc` | `f43239f0cee543729662d5e67faf475b.dir` | 240 | 1749631771 | đã xóa |

26manifest khác giữ snapshot trước; [proof31target cũ](backup-2026-10-12-inventory.json). Hash mới xemchỉmục, không dùnghash bảngsnapshotcũ choV3–V5 hiện tại.
Backup không thaytrainingaudit; source/publicnotebook/report/proposals trênGit, artifact trênDVC.

## 3. Checks that passed

- Quét4.648 filetext vàstaged227file: không thấycredential theo mẫuW&B/Kaggle/APIassignment; khôngstaged file>10MB. Credential private_push vàruntimecache khôngbackup.
-9 target merge/add:13.251 object unique,246.388.340.141B=229,47GiB trongscope. Prepush7.383 object/136.722.405.963B mới; mọientrycũ thiếulocal được giữ.
- Theo chủdựán: syncsong song, cap20→60MiB/s,2worker; bufferretry256KiB. Mộtobject31,5MiB gặpENOMEM, retryV6exit0 (`2 files pushed`).
- Finalpush cả9target exit0: `Everything is up to date.`; status-c exit0: `new/missing=0`,4.072path `deleted` chỉ làremote-only/cachelocal thiếu.
- Kiểmđộc lập toàn13.251object: tồn tại+sizeđúng, mọi9directorymanifestMD5PASS; thiếu/lỗi0. Filemới quawrapper kiểmMD5payload trướcatomiccommit. Khôngrehash toànpayload246GB trênremote.
- G: khônggiữ0444 khiếnDVCcheck hash lạipayloadcũ; wrappertạm chỉvới remote/scope đãchốt táidùngproof theoexistence+size vàMD5directory. Không sửa installedDVC/config. Logs ở `/tmp/mobihet-backup-delta-20261012/`.
- Git vàbundle kiểmriêng; index/link kiểm trướccommit cuối.

## 4. Local cleanup

- Chủdựán yêu cầu xóa saubackup trongđợt bổ sung2026-10-12. Xóa9workspace+5aliasdir và9.179cacheobject thuộcscope đãverify; exactpaths tronginventory.
-10.506fingerprint size/mtime/inode vàmanifesthash khôngđổi; khôngcollector/writer local trướcxóa. GitHEAD khớpremote trướccleanup.
- Thu hồi152673230848B=142.19GiB trongWSL; free799852273664→952525504512B. Giữ code/docs/manifests,private_push vàcache ngoài scope.

- Đợtcleanup trước thu hồi155,99GiB; đó làđợt riêng, không cộnghaiđợt thành dunglượng hiện tại.

- WindowsVHDX chưacompact. Nếu cần trả dung lượng vật lý vềC:, thao tácmanual sau khi kết thúc mọiWSL:
  1. TrongWSL: `sudo fstrim -v /` (cần password; agent không có sudo khôngpassword, chưatrim).
  2. PowerShell Administrator: `wsl --shutdown`.
  3. Mở `diskpart`, chạy:

     ```text
     select vdisk file="C:\Users\odixe\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc\LocalState\ext4.vhdx"
     attach vdisk readonly
     compact vdisk
     detach vdisk
     exit
     ```

## 5. Next session

- Backup/cleanup đợt nàyxong; không target thiếu, khôngmanual bắtbuộc choGit/DVC. Cloud tựsync.
- `conda activate nckh` → `dvc pull <manifest>` từ [chỉ mục](artifact-index.md). V3kaggle snapshot hiện≈125,60GiB logical; kiểmđĩa trước pull.
- Metadata audit V2–V5 nhẹ: `python scripts/restore_audit_metadata.py`; sauđóworkspacepartial, không `dvc add` thaymanifestđầyđủ.
- Restorealias theo đoạnPython trongchỉmục saupullkaggle. Đườngdẫn aliasrelative, không phụthuộc vịtríclone.
- Targetđổi ⇒verify mới trướccleanup. Khôngadd workspaceđãxóa/partial; gộpentrycũ hoặcpullbổsung trướcadd.
- Restoreproof cũ: [metadata](restored-metadata-2026-10-12.json), [V5H4resume](../../papers/v5/anchor-target-diagnostics.json). Source/artifact đãchạy giữhistory.
- Archive: repo `odixe06/mobihet-archive-b9-b21`, tag `archive-3`, bundle `mobihet-archive-b9-b21.bundle`; giữremotehistory.
