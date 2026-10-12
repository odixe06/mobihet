# Git / DVC backup record

Updated: 2026-10-12 UTC+7 — toàn bộ31 target đủ trênG:, cleanup xong; cloud tựsync, không chờ.
Procedure: [repo-backup](../../.agents/skills/repo-backup/SKILL.md).

## 1. Destinations

- Git: `git@github.com:odixe06/mobihet.git`, nhánh `main`; snapshot nghiên cứu `33ce0f6a4495059794dd1a4274d3574cca6cb146`, hồ sơ cleanup28 target `4c32b8ab627e237e8bbff5a34b8977262c873f64` đã khớp remote. Hồ sơ hoàn tất được commit/push tiếp; HEAD kiểm bằng `git ls-remote origin refs/heads/main`.
- DVC `gdrive`: `/mnt/g/My Drive/NCKH/veremi_build/mobihet/dvc-storage`, chung archive; không GC remote.
- Bundle: `/mnt/g/My Drive/NCKH/veremi_build/mobihet/mobihet.bundle`; tạo từ HEAD sau commit hồ sơ này, `--all`, verify trước thay trênG:.
- Cloud: object đã nằm trênG: là backup đủ; cloud upload chưa kiểm, tựtiếp tục. Không xóa cache Drive `D:\ggdrive_cached`.

## 2. Tracked artifacts

| Manifest | MD5 | Files | Logical bytes | Local workspace |
|---|---|---:|---:|---|
| `knowledge/ckpt_round_005.pt.dvc` | `c1466d52f38bdb05ec5facc97d500af2` | 1 | 4924379 | đã xóa2026-10-12 |
| `papers/new-fl/input.dvc` | `d35c8fbd6931d9ab021e4b328f111b04.dir` | 162 | 319302266 | đã xóa2026-10-12 |
| `papers/new-fl/kaggle.dvc` | `2a40c2e202c8c5416ab275ab665644e2.dir` | 622 | 7999577235 | đã xóa2026-10-12 |
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
| `papers/v2/resume_reference.dvc` | `f2d71818d8042456a84190ae8f20fee4.dir` | 108 | 1234921763 | đã xóa2026-10-12 |
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
| `papers/v5/kaggle.dvc` | `f39dc32022b6b8789d3e0caefe03dbc4.dir` | 1494 | 24602498665 | đã xóa2026-10-12 |
| `papers/v5/resume_reference.dvc` | `4b00c9e1fcdd5f0f3549546b58c86dec.dir` | 135 | 1345588556 | đã xóa2026-10-12 |
| `papers/v5/runs.dvc` | `d751713988987e9331980363e24189ce.dir` | 0 | 0 | đã xóa2026-10-12 |
| `papers/v5/smoke.dvc` | `e48def3346891828a1b2755be71e4526.dir` | 711 | 6837402748 | đã xóa2026-10-12 |

[Inventory/proof/cleanup](backup-2026-10-12-inventory.json):31 manifest,8.847 unique object,157.353.597.697B=146,55GiB, đủ trênremote.
Snapshot `kaggle` có thể là collection chưa hoàn chỉnh; backup không xác nhận training/audit.12 `.part` V3 đã được giữ trongDVC để resume download.

## 3. Checks that passed

- `dvc status` trước add: up to date;3 target cũ đầy đủ. `dvc add`30 target exit0, giữ manifest snapshot đã chốt.
- Quét4.610 file text ngoài private_push: không thấy credential theo mẫu W&B/Kaggle/API assignment. Git staged không thấy credential/file>10MB; generated SVG/CSV giữ nguyên.
- Đợt1: push tổng bị `Errno5` do cacheD: đầy.28 target đủ đãpush exit0/status in-sync rồicleanup; không xóa3 target thiếu.
- Sau chủ dự án giải phóngD:, push3 target còn lại exit0: `1535 files pushed`; `dvc status -c`3 target exit0: in sync.
- Kiểm độc lập toàn bộ8.847 object: tồn tại+đúng kích thước; mọi directory manifest MD5 PASS; thiếu/lỗi0. Không rehash157GB content remote; hash nguồn được tạo bởi `dvc add`.
- 28 target đãverify đợt1;3 target đãverify đợt2. Git/bundle kiểm riêng, không suy từDVC.
- Backend local DVC ép copy `batch_size=1` dù `-j`; wrapper tạm4worker atomic củaDVC dùng đểresume; không sửa installed source/config. Không cần wrapper khi restore/push thông thường.

## 4. Local cleanup

- Chủ dự án yêu cầu2026-10-12: đẩy Git/DVC rồi xóa artifact local; chỉ cần đủ trênG:, không chờsync.
- Đã xóa31 workspace và8.847 cache object thuộc manifest đãverify; danh sách exact tronginventory. Giữ code/docs/manifest, cache không thuộc phạm vi, credential private_push. Không động dataset ngoài repo/cache Drive/remote history.
- Không collector/writer local trước xóa. Các snapshot local đã xóa, không `dvc add` lại khi còn thiếu.
- Thu hồi167.494.365.184B=155.99GiB qua2 đợt; WSL free785.104.105.472→952.597.159.936B.

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

- Backup/cleanup đợt này hoàn tất; không còn target thiếu hoặc manual bắt buộc choGit/DVC. Cloud tựsync.
- Khôi phục selected: `dvc pull papers/v3/kaggle.dvc`; cần≈46,3GB logical. Fixture: `dvc pull papers/pilot/fixture.dvc`; kiểm dung lượng trướcpull.
- Chỉ `dvc add` target mới/đầy đủ. Với target đãxóa, pull trước; collection partial phải giữ entry cũ hoặc pull bổ sung.
- Target thay đổi ⇒check cũ hết hiệu lực; verify snapshot mới trướccleanup.
- G: từ chối chmod0444; judged by exit/status, không theo DEBUG protection warning. `status -c` saucleanup cóthể `deleted`=thiếu cachelocal, không phải mấtremote.
- Archive giữ repo `odixe06/mobihet-archive-b9-b21`, tag `archive-3`, bundle `mobihet-archive-b9-b21.bundle`.
