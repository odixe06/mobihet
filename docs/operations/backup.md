# Git / DVC backup record

Updated: 2026-10-12 UTC+7 — backup toàn bộ thay đổi được chủ dự án yêu cầu; Git đã push;28 target DVC đã verify/cleanup,3 target còn thiếu do cache D: đầy.
Procedure: [repo-backup](../../.agents/skills/repo-backup/SKILL.md).

## 1. Destinations

- Git: `git@github.com:odixe06/mobihet.git`, nhánh `main`; backup snapshot `33ce0f6a4495059794dd1a4274d3574cca6cb146` đã khớp remote; hồ sơ cleanup được push tiếp.
- DVC `gdrive`: `/mnt/g/My Drive/NCKH/veremi_build/mobihet/dvc-storage`, chung archive; không GC remote.
- Bundle: `/mnt/g/My Drive/NCKH/veremi_build/mobihet/mobihet.bundle`; snapshot `33ce0f6` đã verify trên G:; cập nhật lại sau commit hồ sơ.
- Cloud upload chưa kiểm; chỉ yêu cầu object đã nằm trên G: trước cleanup.
- Drive cache: `D:\ggdrive_cached`; trước push D: còn97.469.865.984B, G: còn92.596.371.456B. Không xóa cache Drive.

## 2. Tracked artifacts

| Manifest | MD5 | Files | Logical bytes | Local workspace |
|---|---|---:|---:|---|
| `knowledge/ckpt_round_005.pt.dvc` | `c1466d52f38bdb05ec5facc97d500af2` | 1 | 4924379 | đã xóa2026-10-12 |
| `papers/new-fl/input.dvc` | `d35c8fbd6931d9ab021e4b328f111b04.dir` | 162 | 319302266 | đã xóa2026-10-12 |
| `papers/new-fl/kaggle.dvc` | `2a40c2e202c8c5416ab275ab665644e2.dir` | 622 | 7999577235 | giữ local; remote chưa đủ |
| `papers/new-fl/resume_reference.dvc` | `0291d66b53d9538c8e7298eafeb11227.dir` | 13 | 100116057 | đã xóa2026-10-12 |
| `papers/new-fl/runs.dvc` | `d751713988987e9331980363e24189ce.dir` | 0 | 0 | đã xóa2026-10-12 |
| `papers/new-fl/smoke.dvc` | `0a67a58b6ba9c09beac284e920193d09.dir` | 79 | 599131852 | đã xóa2026-10-12 |
| `papers/p2/kaggle.dvc` | `05d670af724058a6a42f6c291c1dd054.dir` | 52 | 1289660 | đã xóa2026-10-12 |
| `papers/p2/runs.dvc` | `bbba5ff279d47c2932690e74b63c9c6f.dir` | 44 | 1251162 | đã xóa2026-10-12 |
| `papers/p2/source_audit.dvc` | `36f834ab0207195ab6f24922755fe781.dir` | 32 | 306212 | đã xóa2026-10-12 |
| `papers/pilot/fixture.dvc` | `cb782c4671b1aa25a8a2c56bf465587c.dir` | 67 | 159554510 | đã xóa2026-10-12 |
| `papers/pilot/runs.dvc` | `ca8715563f5d94932638a588f9999bda.dir` | 32 | 1833638 | đã xóa2026-10-12 |
| `papers/v2/input.dvc` | `e1ca6a39074fe9722422189daaadc96c.dir` | 170 | 319384452 | đã xóa2026-10-12 |
| `papers/v2/kaggle.dvc` | `3c49c7ec142513d62f8e31da23534ee1.dir` | 2422 | 31674561783 | đã xóa2026-10-12 |
| `papers/v2/resume_reference.dvc` | `f2d71818d8042456a84190ae8f20fee4.dir` | 108 | 1234921763 | giữ local; remote chưa đủ |
| `papers/v2/runs.dvc` | `d751713988987e9331980363e24189ce.dir` | 0 | 0 | đã xóa2026-10-12 |
| `papers/v2/smoke.dvc` | `f0da440b6a89ca4822cd1530b539f7d2.dir` | 612 | 6862940249 | đã xóa2026-10-12 |
| `papers/v3/input.dvc` | `806adf3969b5fc9c790c6d413a05e8bc.dir` | 534 | 958219444 | đã xóa2026-10-12 |
| `papers/v3/kaggle.dvc` | `73d45219fbc936bd72f4398f0ef14b8e.dir` | 1569 | 46301158058 | đã xóa2026-10-12 |
| `papers/v3/resume_reference.dvc` | `aa3ae580e241cf9a81a63b3c4966b2f5.dir` | 216 | 4760862893 | đã xóa2026-10-12 |
| `papers/v3/runs.dvc` | `d751713988987e9331980363e24189ce.dir` | 0 | 0 | đã xóa2026-10-12 |
| `papers/v3/smoke.dvc` | `57f87a065d3b2adb2742e8aa8b2e386c.dir` | 888 | 21931873202 | đã xóa2026-10-12 |
| `papers/v4/input.dvc` | `1651b7aa22451ee804890c0fa0a1b629.dir` | 376 | 638897962 | đã xóa2026-10-12 |
| `papers/v4/kaggle.dvc` | `6001ef5a3a5b6ffade336ecb071b617f.dir` | 832 | 7007229659 | đã xóa2026-10-12 |
| `papers/v4/resume_reference.dvc` | `cac2264d8acce624b3ab5de91cfe194b.dir` | 89 | 906936054 | đã xóa2026-10-12 |
| `papers/v4/runs.dvc` | `d751713988987e9331980363e24189ce.dir` | 0 | 0 | đã xóa2026-10-12 |
| `papers/v4/smoke.dvc` | `3d13b231afb166fa9e2797c2f5f5e135.dir` | 243 | 2406524180 | đã xóa2026-10-12 |
| `papers/v5/input.dvc` | `9903334ead144123e02263bd76b47af5.dir` | 291 | 479217431 | đã xóa2026-10-12 |
| `papers/v5/kaggle.dvc` | `f39dc32022b6b8789d3e0caefe03dbc4.dir` | 1494 | 24602498665 | giữ local; remote chưa đủ |
| `papers/v5/resume_reference.dvc` | `4b00c9e1fcdd5f0f3549546b58c86dec.dir` | 135 | 1345588556 | đã xóa2026-10-12 |
| `papers/v5/runs.dvc` | `d751713988987e9331980363e24189ce.dir` | 0 | 0 | đã xóa2026-10-12 |
| `papers/v5/smoke.dvc` | `e48def3346891828a1b2755be71e4526.dir` | 711 | 6837402748 | đã xóa2026-10-12 |

[Inventory](backup-2026-10-12-inventory.json):31 manifest,8.847 unique object,157.353.597.697B; cần upload157.187.257.520B.
Snapshot `kaggle` có thể là collection chưa hoàn chỉnh; backup không xác nhận training/audit. Giữ12 `.part` V3 để resume download.

## 3. Checks that passed

- `dvc status` trước add: up to date;3 target cũ đầy đủ, không thay manifest bằng partial workspace.
- `dvc add`30 target: exit0; cache+remote đủ mọi object, không missing cảhai nơi.
- Quét4.610 file text ngoài private_push: không thấy credential thật theo mẫu W&B/Kaggle/API assignment.
- Git staged: không thấy credential hoặc file>10MB. Generated SVG/CSV có whitespace; giữ nguyên artifact.
- Push tổng bị lỗi `Errno5 Input/output error` khi D: cache gần đầy; đã dừng, không gọi backup toàn bộ hoàn tất.
- 28 target đủ: `dvc push <28 manifests>` exit0, `Everything is up to date.`; `dvc status -c` exit0, in sync.
- Kiểm độc lập:7.312/8.847 object tồn tại+đúng kích thước,120,04GiB; hash manifest của target cleanup PASS.
- 3 target còn1.535 object chưa xác minh,26,51GiB: `papers/new-fl/kaggle.dvc`, `papers/v2/resume_reference.dvc`, `papers/v5/kaggle.dvc`.
- Kiểm nội dung: hash source do `dvc add`; mọi object kiểm tồn tại/size, directory manifest MD5. Không rehash157GB content trên remote.
- Backend local DVC ép copy `batch_size=1`; resume qua wrapper tạm `/tmp/mobihet-backup-20261012/push_parallel.py`,4 worker atomic của DVC; không sửa installed source/config.

## 4. Local cleanup

- Chủ dự án yêu cầu2026-10-12: đẩy toàn bộ Git/DVC rồi xóa artifact local tiết kiệm đĩa.
- Đã xóa28 workspace trong [inventory](backup-2026-10-12-inventory.json) và6.737 cache object chỉ thuộc target đủ; giữ cache cần3 target chưa đủ. Giữ code/docs/manifest và credential private_push; không động dataset ngoài repo/cache Drive/remote history.
- Không còn collector/writer local lúc kiểm. Collector phiên sau phải `dvc pull` target cần thiết trước resume; không `dvc add` workspace partial.
- Thu hồi133.614.632.960B=124,43GiB. WSL free785.104.105.472→918.718.738.432B. Giải phóng ext4 không tự giảm VHDX Windows; compaction là bước manual riêng.

## 5. Next session

- Chủ dự án sẽ giải phóng thêm≈30GiB trên D: ngoài cache Drive và báo. Khi đủ: `dvc push papers/new-fl/kaggle.dvc papers/v2/resume_reference.dvc papers/v5/kaggle.dvc`, verify từng manifest+object rồi cleanup3 workspace/cache còn lại. Không chờ cloud sync.
- Không `dvc add` target đã xóa. Nguồn30 snapshot đã cố định; giữ manifest hiện tại.

- Khôi phục: `dvc pull papers/v3/kaggle.dvc` hoặc target cụ thể trong bảng; kiểm dung lượng trước pull.
- Chỉ `dvc add` target mới/đầy đủ; sau collect partial phải giữ entry cũ hoặc pull bổ sung.
- Push thành công và remote object/manifest đầy đủ mới cleanup. Cloud upload không phải gate, trạng thái cloud báo riêng.
- G: từ chối chmod0444; judged by exit/status, không theo DEBUG protection warning.
- Archive vẫn ở repo `odixe06/mobihet-archive-b9-b21`, tag `archive-3`, bundle `mobihet-archive-b9-b21.bundle`; không xóa lịch sử.
