# Git / DVC backup record

Updated: 2026-10-14 UTC+7 — delta verify G: hoàn tất; đã xóa 2026-10-14.
Procedure: [repo-backup](../../.agents/skills/repo-backup/SKILL.md).

## 1. Destinations

- Git: `git@github.com:odixe06/mobihet.git`, `main`; HEAD cuối kiểm bằng `git ls-remote origin refs/heads/main` và `git rev-parse HEAD`.
- DVC `gdrive`: `/mnt/g/My Drive/NCKH/veremi_build/mobihet/dvc-storage`.
- Bundle: `/mnt/g/My Drive/NCKH/veremi_build/mobihet/mobihet.bundle`; `--all`, verify trước atomic replace.
- Cloud sync không kiểm/không chờ. Giữ Drive cache `D:\ggdrive_cached`; G: đủ object là gate cleanup.

## 2. Tracked artifacts

[Chỉ mục toàn manifest/lệnh pull](artifact-index.md). [Inventory/proof/alias/cleanup](backup-delta-2026-10-14.json).
Entry remote-only giữ nguyên; local override cùng relpath. Không add partial trực tiếp.

| Manifest | MD5 | Files | Logical bytes | Local |
|---|---|---:|---:|---|
| `papers/boundary-cells/controller-fixture.dvc` | `195b422d05d1a14eacf7e563725caef3.dir` | 76 | 16932 | đã xóa 2026-10-14 |
| `papers/boundary-cells/input.dvc` | `5d398f36a4ac5baebdc135fe9a364549.dir` | 230 | 319715058 | đã xóa 2026-10-14 |
| `papers/boundary-cells/kaggle.dvc` | `57c7e72fca8bd50a0ffcfc71391d09c6.dir` | 338 | 1240071864 | đã xóa 2026-10-14 |
| `papers/boundary-cells/smoke.dvc` | `bd1f76e8e88b0ab664224c4609eb5417.dir` | 210 | 2190821198 | đã xóa 2026-10-14 |
| `papers/full-validation/input.dvc` | `b50f3b5852c683e91058cb0dcef9e27c.dir` | 200 | 1282683 | đã xóa 2026-10-14 |
| `papers/full-validation/kaggle.dvc` | `789efb2da148689d271aa55e3f6d5170.dir` | 968 | 9278181515 | đã xóa 2026-10-14 |
| `papers/full-validation/mount-output.dvc` | `f98af0c07296f21e2b58d1f5469941da.dir` | 18 | 53171 | đã xóa 2026-10-14 |
| `papers/full-validation/resume_reference.dvc` | `a7c0717142b58fb5d24a157bc113b564.dir` | 63 | 658095993 | đã xóa 2026-10-14 |
| `papers/full-validation/smoke.dvc` | `da8ab9332b7cba1e5535639071bc93cc.dir` | 221 | 1314131032 | đã xóa 2026-10-14 |
| `papers/hybrid-full/diagnostics.dvc` | `b4c3578df19e67fff29bba3066d02465.dir` | 20 | 935609 | đã xóa 2026-10-14 |
| `papers/hybrid-full/input.dvc` | `637a09a84948d0db883c54aa8b18c861.dir` | 212 | 1297638 | đã xóa 2026-10-14 |
| `papers/hybrid-full/kaggle.dvc` | `89ba8972c2ee1062ab77a27d58fb08bc.dir` | 384 | 2834658221 | đã xóa 2026-10-14 |
| `papers/hybrid-full/resume_reference.dvc` | `b2ddeb62e648d9190650ce62ddcf1a1b.dir` | 81 | 1086355227 | đã xóa 2026-10-14 |
| `papers/hybrid-full/smoke.dvc` | `4dd92ca56796decf59ae3e200f13ee46.dir` | 111 | 1524677711 | đã xóa 2026-10-14 |
| `papers/pilot/fixture.dvc` | `cb782c4671b1aa25a8a2c56bf465587c.dir` | 67 | 159554510 | đã xóa 2026-10-14 |
| `papers/v10/input.dvc` | `743718b1dd83290b17491188e085e90d.dir` | 222 | 319641954 | đã xóa 2026-10-14 |
| `papers/v10/kaggle.dvc` | `8cde4b9a4e724ddc0fa35047c6f9cb48.dir` | 568 | 4523882900 | đã xóa 2026-10-14 |
| `papers/v10/resume_reference.dvc` | `456967a838675e3d0cf7763cec988aef.dir` | 102 | 1283401033 | đã xóa 2026-10-14 |
| `papers/v10/smoke.dvc` | `752fdb98ccc1bca8431d31d5f3024819.dir` | 246 | 2992749358 | đã xóa 2026-10-14 |
| `papers/v2/kaggle.dvc` | `3c49c7ec142513d62f8e31da23534ee1.dir` | 2422 | 31674561783 | đã xóa 2026-10-14 |
| `papers/v6/kaggle.dvc` | `ec521a9fcccb9633cf60cf28bf88911a.dir` | 1666 | 14056604948 | đã xóa 2026-10-14 |
| `papers/v7/input.dvc` | `fe759ba87afb08430620383790d01da1.dir` | 204 | 319545868 | đã xóa 2026-10-14 |
| `papers/v7/kaggle.dvc` | `6f1ac7fde1746d98c4ffd0bc2ede9460.dir` | 1618 | 14100779740 | đã xóa 2026-10-14 |
| `papers/v7/resume_reference.dvc` | `5a5d85b0c7fb4ade93e7e3add7915f2c.dir` | 60 | 422721575 | đã xóa 2026-10-14 |
| `papers/v7/smoke.dvc` | `94c9e76fb1d7f258b4a4bb8167a1e0e0.dir` | 150 | 1053189840 | đã xóa 2026-10-14 |
| `papers/v8/input.dvc` | `1f1c3423d9dc96f16606930a98ea6365.dir` | 210 | 319566420 | đã xóa 2026-10-14 |
| `papers/v8/kaggle.dvc` | `5e8f82e41e5204fff1d0efca49cbaaab.dir` | 1242 | 20412224345 | đã xóa 2026-10-14 |
| `papers/v8/resume_reference.dvc` | `ee5d508c6c69cd47549940bf6be64459.dir` | 51 | 663906613 | đã xóa 2026-10-14 |
| `papers/v8/smoke.dvc` | `b0da1af1f8498adb7a526192686c2fc4.dir` | 111 | 1526827464 | đã xóa 2026-10-14 |
| `papers/v9/input.dvc` | `347e0af95ba09acf7825b755fbc95e36.dir` | 222 | 319643036 | đã xóa 2026-10-14 |
| `papers/v9/kaggle.dvc` | `b9220ec4e71f82a1cb3fcedf034402fa.dir` | 2798 | 37425239839 | đã xóa 2026-10-14 |
| `papers/v9/resume_reference.dvc` | `9c657c34734d22cc2caebefa9ec1dfe6.dir` | 153 | 1944387156 | đã xóa 2026-10-14 |
| `papers/v9/smoke.dvc` | `f3ea67005856b761dc86fda011f40b57.dir` | 684 | 8402355983 | đã xóa 2026-10-14 |

Manifest khác giữ snapshot trước; proof cũ: `backup-delta-2026-10-12.json`, `backup-2026-10-12-inventory.json`.

## 3. Checks that passed

- CAS transfer trực tiếp theo format DVC `files/md5`, 2 worker, tổng cap60MiB/s, buffer256KiB; payload mới MD5 khi copy trước atomic replace.
- Uploaded 8107 object / 105632333883 byte; toàn 11887 object chọn lọc tồn tại+size đúng; mọi directory manifest MD5 PASS. Payload cũ không rehash toàn remote.
- Local file hash, size/mtime/inode lưu inventory; manifest mới giữ entry lịch sử remote-only.
- Git candidate không file >10MiB; credential-assignment scan không hit. Private_push/runtimecache không đưa lên Git/DVC.
- DVC push exit0, 0 object còn phải push; status: new/missing=0, deleted=11.787 chỉ thiếu cache local. Wrapper tạm tái dùng existence/size proof trong scope; không sửa installed DVC/config.
- Git snapshot artifact `9d1bd95ec4fe1b2ee629b2f2f03c1028dec22727` đã push; bundle --all verify+SHA256 PASS trên G: trước cleanup. Hồ sơ cleanup commit/push tiếp; bundle thay bằng HEAD cuối.
- Không coi cloud sync là điều kiện hoàn tất.

## 4. Local cleanup

- Chủ dự án yêu cầu 2026-10-14; đã xóa 2026-10-14. Exact target/file/alias trong inventory; cache chỉ object thuộc scope đã verify.
- Thu hồi 108.68 GiB WSL; số free trước/sau trong inventory sau cleanup.
- Giữ code/docs/notebook/manifests, private_push, cache ngoài scope, Drive upload cache và remote history.
- Windows VHDX chưa compact. Nếu cần trả dung lượng vật lý C: về Windows: `sudo fstrim -v /`, PowerShell Administrator `wsl --shutdown`, rồi diskpart compact VHDX. Không bắt buộc cho backup/cleanup WSL.

## 5. Next session

- `conda activate nckh`; `dvc pull <manifest>` từ chỉ mục; kiểm đĩa trước pull.
- Target partial: gộp manifest cũ+local delta theo relpath, giữ entry remote-only; không blanket `dvc add`.
- Target đổi ⇒ verify mới trước cleanup. Không GC remote.
- DVC `deleted` có thể chỉ thiếu local cache; `new/missing` mới là thiếu remote.
- Archive `odixe06/mobihet-archive-b9-b21`, tag `archive-3`, bundle cũ giữ nguyên.
