# Cấu trúc hiện tại

Tên mô tả cơ chế/nhiệm vụ; nhãn V2–V10 chỉ để truy vết đợt thí nghiệm.
[Tên cũ/mới](experiment-names.md) · [Ownership code](package-ownership.md) · [Map hiện tại](workspace-layout.json).

```text
core/                              code Python chung, tên theo nhiệm vụ
methods/<method>/builds/<build>/    phần còn đang làm
  source/                          implementation/snapshot frozen
  notebooks/                       source/config/version đã push
  scripts/                         helper của build
  probes/                          probe/calibration/fixture proof
  artifacts/                       input/output/checkpoint, Git ignored hoặc DVC
  logs/                            logs riêng build
  private_push/                    notebook key private, ignored0600
  *.dvc                            metadata artifact, giữ object hashes
archive/methods/<method>/builds/    build đã kết thúc, cùng cấu trúc bên trên
archive/tools/                     helper/migration lịch sử
archive/logs/operations/           logs công cụ đã kết thúc
docs/
  README.md                        mục lục đọc nhanh
  proposals.md                     một danh mục active/đã kết thúc
  methods/                         thiết kế và kết quả hướng đang làm
  lessons/                         bài học dùng tiếp từ hướng đã kết thúc
  archive/                         thiết kế/report/handoff lịch sử
  operations/                      hiện trạng, lỗi, map/quota/checks
  research/                        protocol và căn cứ còn áp dụng
  file-index.md                    index khu làm việc, dẫn sang index archive
scripts/                           công cụ dùng chung đang dùng
logs/operations/                   log công cụ chung đang dùng
knowledge/                         dữ liệu/model/runtime nền
.compat/                           các alias/import lịch sử, không có artifact copy
mh/ -> .compat/mh/                 facade 40 source names frozen
papers/ -> .compat/papers/         runtime source paths cũ
```

Khu làm việc giữ Gaussian, DML, Hybrid, BoundaryCells, ResponseRisk, EMA và đối chứng full chung.
Pilot/probe hoàn tất chuyển archive; bài học vẫn ở LESSONS/root hoặc docs/lessons. Không coi archive là phủ định
cả họ phương pháp hoặc bằng chứng R1 đã giải quyết. Một build so nhiều cơ chế có một owner artifact, không nhân đôi.

Mỗi build chứa probes/calibration/pilot/continuation/logs của nó. Log nằm trong artifact DVC đã chốt giữ vị trí
bên trong payload; không tháo log khỏi archive làm đổi .dir hash. README điều hướng đặt ngoài frozen payload/private keys.

Source Python/notebook frozen giữ byte và source filename map khi dựng runtime; tên canonical có thể khác tên import.
Alias không giữ `__file__.resolve()` depth: core có cùng độ sâu root, helper dùng root discovery, verifier được dựng
đúng runtime cũ hoặc ROOT điều chỉnh ở lớp gọi. Không sửa scientific code để che lỗi đường dẫn.

## Git/DVC và khôi phục

`.dvc` chuyển cùng build. Đợt archive giữ md5/size/nfiles và relpath payload; outs.path đổi sang artifacts/<target> tương đối từ manifest.
Các alias nằm trong `.compat/` và được loại khỏi DVC/Git scan. Artifact/private ignored ở cả active lẫn archive.
Migration trước giữ HEAD/staged; phiên tiếp chủ dự án yêu cầu commit/push Git. Kiểm DVC và lệnh pull tại [chỉ mục canonical](artifact-index.md); không DVC add workspace partial.
Remote-only history vẫn có .dvc và index; local partial restore không được add thành manifest mới.

Fresh checkout: `python3 scripts/restore_compat_aliases.py`. Helper dựng symlink/private folder rỗng từ
[map](compatibility-aliases.json), không ghi đè source hoặc pull data. CLI cũ trỏ `scripts/check_vN...py` đã rời
khu làm việc; dùng canonical script trong build hoặc archive. Import helper cũ vẫn có compatibility lookup.

## Cập nhật tiếp

1. Build mới đặt tên theo cơ chế/phạm vi; ID đợt cũ chỉ là provenance.
2. Thiết kế/kết quả ở docs/methods; bài học chung ở LESSONS; lỗi vận hành ở CONTEXT/sổ lỗi.
3. Build đóng: chuyển archive và giữ bài học; kiểm dependency trước, không sửa source đã push.
4. Chạy index, package/layout checks; so SHA/source parent/DVC identities/HEAD/index/key boundary.
5. Lưu kernel/version, actual committed round, quota và việc còn thiếu trong handoff.
