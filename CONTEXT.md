# CONTEXT — MobiHet / federated IDS V2X

Bắt đầu: CONTEXT → [LESSONS](LESSONS.md) → [bàn giao hiện tại](docs/operations/continuation-2026-10-15-evening.md).
[Mục lục](docs/README.md) · [Danh mục phương pháp](docs/proposals.md) · [Index file](docs/file-index.md).

## Bài toán và ràng buộc

Xe học IDS cộng tác qua RSU với giao tiếp hữu hạn. Giữ66features, partition/scaler/protocol gốc;
không raw/perturbed record hoặc per-record activation/gradient rời xe. Mọi learned model hiện DAGSNet;
heterogeneous R2 và contact/deadline R1 chưa được chứng minh.
Train/probe production nặng trên Kaggle, local chỉ fixture/algebra/collect/audit/report.
Official test chỉ báo cáo, không chọn HP/best round/continuation; group/old pilot nội bộ có protocol riêng.
[Protocol](docs/research/protocol.md) · [Dữ liệu/kiến trúc](knowledge/README.md).

## Quyết định còn hiệu lực

- **Owner16/10 ~03:00Z: mục tiêu mới = cải thiện dự đoán của các client, không phải proxy.**
  Owner ~04:00Z chốt: **A primary** (private gp toàn cục), báo kèm local/client tệ nhất; **giữ R2** (private độc lập);
  vòng 2 giữ + cổng phụ private gp đã khóa.
  [Tổng hợp phương pháp + bằng chứng](docs/research/client-model-directions-2026-10-16.md).
- **Owner16/10 ~16:45Z:** full chỉ nối `h_dml` r41→r50 (đối chứng r50 cho m4b full sau); **dừng** `h_dml_own` r39,
  `h_hybrid` r35, `h_hybrid_own` r33 (phẳng). Seed 43 hai session (vòng 6). **R1: mô phỏng contact trước khi chọn
  số tổng hợp/epoch** ([phân tích bước 1](docs/research/contact-budget-2026-10-16.md)); sub16 chờ.
- Owner15/10 tối: **proxy head là headline DML/Hybrid**, private gp báo kèm (ProxyFuse cho thấy private không thêm tín hiệu).
  ⚠ Bị mục tiêu16/10 thay về hướng ưu tiên; giữ làm số đo kèm.
  Full giữ protocol §1 (50×1 epoch, cosine T50, seed42, 20clients, 100%train80%), chờ r50.
  Hybrid continuation chỉ `h_hybrid`/`h_hybrid_own`; controls dùng full-validation (trùng bit-exact). hybf3 đã dừng.
- Account `nguyenminhtran060113` (30h, reset 17/10 00:00 UTC) dành cho pilot mới; ba account annhtnt/takids1909/accanniez
  **không query quota**, envelope89,7/90h đã phân. Thử mọi kỹ thuật chống học thuộc + QProxy (owner duyệt).
- Owner15/10 ~15:00Z: **ưu tiên thử kỹ thuật mới**, còn chỗ trống mới chạy tiếp full. Sàng lọc phương pháp bằng
  **một seed chung 42**; seed khác để sau. Duyệt FullDrift (kiểm thẳng trên full, holdout nội bộ, epoch1–6).
  Account mới token-only: annhannh331, anhtruong1909 (FullDrift), thuyanh316 (dự trữ). [Account](docs/operations/kaggle-accounts.md).
- Freeze source/notebook đã PUSHED và toàn40module `mh/*.py`; source build mới riêng.
  Exact-resume từ actual committed checkpoint, không W&B summary hoặc r1. Đổi vận hành dùng tag mới.
  [Lỗi và recovery](docs/operations/notebook-errors.md).
- Giữ scientific settings/gp/gates đã chốt; nghi vấn khoa học hoặc budget vượt scope thì hỏi owner trước.
  Official test chỉ báo cáo. Không nối adaptation FAIL (ClassTail, CountSketch repair, BoundaryCells, CCR, ProxyFuse).
- Không Git/DVC add/commit/push/cleanup nếu owner chưa yêu cầu; selective restore được phép. Giữ owner edits credential reference.
  Private inline W&B key chỉ ignored private_push0600. [Layout](docs/operations/project-layout.md) · [Account](docs/operations/kaggle-accounts.md).

## Trạng thái 15/10/2026 tối (đọc ~12:10Z, không live)

- Chẩn đoán chững sớm: full mỗi round ≈4.200 bước/client ⇒ Gaussian/local đỉnh r1 rồi giảm (học thuộc receiver);
  proxy DML/Hybrid bão hòa ~0,78–0,79 từ r3–r5, private 0,732/0,733; khối 4 lớp nhầm lẫn chặn macro F1.
  [Chẩn đoán](docs/research/plateau-diagnosis-2026-10-15.md).
- ProxyFuse probe **FAIL** (fused −0,032/−0,034 so proxy). CCR seed42 utility **FAIL** (pooled−CE +0,0017);
  seed43 recovery v2 đang chạy chỉ để mô tả.
- AntiMem+QProxy r20 hai seed **AUDITED** (~19:00Z): Gaussian private PASS mọi kỹ thuật (m4b +0,025, swappos +0,016–0,018,
  private_ema +0,035–0,039); DML/Hybrid proxy FAIL mọi kỹ thuật, proxy EMA hại; **QProxy q4 PASS** (±0,001, byte 7,65× nhỏ).
  [Kết quả](docs/methods/anti-memorization/results.md). Đề xuất (chưa launch): q4 cho full DML/Hybrid.
- FullDrift seed42 ep6 **AUDITED** (~21:30Z): drift là nút thắt — sub4 (4 lần tổng hợp/epoch) DML proxy +0,026,
  FedAvg +0,034 (benign FedAvg −0,061); FedProx giảm drift 5× nhưng F1 0 (benign +0,098); FedAvgM β0,9 phân kỳ;
  FedAvg ≈ DML (+0,009); Gaussian swappos +0,052, private_ema Gaussian +0,024/+0,036. [Kết quả](docs/methods/full-drift/results.md).
- Q2 toàn bộ COMPLETE/AUDITED: full local r23, Gaussian r20, DML r23, DML-own r21; Hybrid r18, Hybrid-own r17.
  Test (mô tả): DML proxy 0,7869, Hybrid proxy 0,7727, private gp 0,7332/0,7348. Tới r50 còn ≈116 session-hours, cần owner cấp quota.
  [Bảng kernel và việc còn thiếu](docs/operations/continuation-2026-10-15-evening.md).

## Phiên sau

Đọc mục **"16/10 ~17:10Z"** cuối [bàn giao tối](docs/operations/continuation-2026-10-15-evening.md).
0. FullDrift vòng 2/4/5 **AUDITED** ~17:00Z ([kết quả](docs/methods/full-drift/results.md)): **m4b@e15 private +0,036/proxy
   +0,086 PASS** (benign proxy không sụp; `fd_dml` phẳng), m4b_q4 noninferior, **m4b_sub4_q4 +0,031 PASS**, sub16_q4 PASS,
   sub4 horizon PASS; hkd (FedKD) FAIL, prox FAIL; Gaussian m4b +0,067 PASS.
1. **Vòng 6–7 seed 43 RUNNING** (anhtruong1909 k/l PUSHED 16:52Z ~21:10Z; annhannh331 m swappos + Gaussian m4b PUSHED 17:32Z ~21:45Z): collect → `report_fd.py` → quy tắc headline
   [design §Vòng 6](docs/methods/full-drift/design.md) (m4b@e15 đã PASS; cần m4b vs dml s43 PASS private + proxy).
2. **q5 `h_dml` r41→r50** (thuyanh316, 9,8h): ckpt 256 file PASS, CPU gate PASS; GPU push **sau 00:00Z 17/10** (quota reset).
3. q4 full/Hybrid **AUDITED** (h_dml r41, own r39, h_hybrid r35, own r33); report full/Hybrid tạo lại.
4. R1 bước 2: mô phỏng cohort 20 client theo kịch bản RSU owner chọn.

## Muốn biết → mở

| cần biết | tài liệu |
|---|---|
| Kernel, checkpoint, budget, việc chưa làm, lệnh tiếp tục | [Bàn giao tối](docs/operations/continuation-2026-10-15-evening.md), [chiều](docs/operations/continuation-2026-10-15-afternoon.md) |
| Toàn bộ phương pháp, PASS/FAIL/chưa kiểm, research mới | [Danh mục](docs/proposals.md), [hồ sơ](docs/methods/README.md) |
| Mọi con số/full metrics đã audit | [H1/H2](docs/methods/shared-baselines/builds/federated-baselines-full/results/latest-rounds.md), [Hybrid](docs/methods/gaussian-proxy-hybrid/builds/hybrid-full/results/latest-rounds.md) |
| Lỗi runtime/recovery và nguyên tắc không lặp | [Sổ lỗi](docs/operations/notebook-errors.md), [LESSONS](LESSONS.md) |
| Dữ liệu, kiến trúc, protocol | [Knowledge](knowledge/README.md), [protocol](docs/research/protocol.md) |
| Script, layout, alias, restore/backup | [Scripts](scripts/README.md), [layout](docs/operations/project-layout.md), [backup](docs/operations/backup.md) |

CONTEXT giữ trạng thái/quyết định và đường dẫn; số đo/commands/contracts ở docs của build hoặc operations.
Tri thức dữ liệu/kiến trúc ở knowledge; quy trình dùng lại ở skills. Bài học mới cập nhật LESSONS ngay.
Sau thêm/chuyển file: cập nhật index, kiểm link/hash/ignore; không tự backup hoặc dọn artifact.
