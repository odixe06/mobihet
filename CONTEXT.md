# CONTEXT — MobiHet / federated IDS V2X

Bắt đầu: CONTEXT → [LESSONS](LESSONS.md) → [bàn giao hiện tại](docs/operations/continuation-2026-10-17.md).
[Mục lục](docs/README.md) · [Danh mục phương pháp](docs/proposals.md) · [Phương pháp theo tên](docs/research/methods/README.md) · [Index file](docs/file-index.md).

## Bài toán và ràng buộc

Xe học IDS cộng tác qua RSU với giao tiếp hữu hạn. Giữ66features, partition/scaler/protocol gốc;
không raw/perturbed record hoặc per-record activation/gradient rời xe. Mọi learned model hiện DAGSNet;
heterogeneous R2 và contact/deadline R1 chưa được chứng minh.
Train/probe production nặng trên Kaggle, local chỉ fixture/algebra/collect/audit/report.
Official test chỉ báo cáo, không chọn HP/best round/continuation; group/old pilot nội bộ có protocol riêng.
[Protocol](docs/research/protocol.md) · [Dữ liệu/kiến trúc](knowledge/README.md).

## Quyết định còn hiệu lực

- **Trạng thái 18/10 ~09:00Z — đọc [bàn giao §Trạng thái gộp 18/10 ~09:00Z](docs/operations/continuation-2026-10-17.md) trước mọi việc**
  (bảng kernel vòng 16–18, lệnh collect, việc kế tiếp, câu hỏi chờ owner).
- **Headline ứng viên `fd_dml_m4b_gpool_frz_gkd`** (teacher = snapshot gộp cố định cả epoch + mẫu Gaussian gộp học KL từ snapshot): vòng 15 s42
  ep15 private **head 0,8030** (+0,069 vs gpool gp; test 0,8046; client tệ nhất +0,128). Nguyên nhân chững cũ (probe 14–19): tầng quyết định private
  trôi về local trong epoch dài vì teacher DML cũng trôi; phần còn lại là thiếu input phủ client khác. [Results](docs/methods/full-drift/results.md).
- **Owner 17/10 ~20Z (còn hiệu lực):** giữ nguyên dataset Kaggle đã xử lý (đối chiếu [fl_dagsnet_veremi](https://github.com/odixe06/fl_dagsnet_veremi));
  mọi client DAGSNet; **không đề xuất đổi mô hình**; mọi đề xuất giữ R1 (byte ít, ngắn hạn) và R2 (client khác kiến trúc — bảo đảm thiết kế).
- **Owner 18/10:** (1) **luật chính = luật chính xác hơn** ⇒ họ frz đọc **head**, DML cũ giữ gp ([design §Luật](docs/methods/full-drift/design.md));
  (2) gkd ổn ⇒ **thử mọi cải tiến có bằng chứng (quota thoải mái), chắc chắn tốt nhất thì chạy 50 round** — vòng 17 (`la`, `kdpx`, `la_kdpx`;
  `gkd:50` nối tiếp ep15 chờ pa s43 PASS) + vòng 18 (`cop` Gaussian copula, `pce`, `t2`, `g512`, `gk2`, `q4_frz_gkd`);
  (3) kiểm đúng FL ⇒ [kiểm toán luồng thông tin](docs/research/federated-information-flow.md): huấn luyện federated, SecAgg chỉ giả lập;
  **~09:15Z: biện pháp bảo mật (a) ngưỡng ô/DP, (d) scale chung q4/q2, SecAgg thật chỉ ghi tài liệu (§7 file đó), chưa cài — mục tiêu chính là
  performance**; (4) tối ưu code: skill `federated-training-speed`; **~09:15Z duyệt lớp tolerance `fd.py --fast` (fused AdamW + nondet, −25 % local)
  chỉ cho kernel build từ vòng 19; kernel cũ (tới vòng 18, gồm rd) giữ đối chứng cũ; arm `--fast` chỉ so đối chứng `--fast`**
  ([design §Code nhanh](docs/methods/full-drift/design.md)); (5) skill: `.claude` là bản chính, `.agents` bản Codex; CLAUDE.md §7 định tuyến skill.
- **Owner 17/10 ~19:50Z: giữ headline DML-m4b + gpool** dù gpool:15 horizon FAIL (lợi so m4b +0,016 ep6 → +0,002 ep15;
  lợi tốc độ đầu = ít round/byte hơn cho cùng F1). gw2 FAIL. Cổng ep6 chỉ đo tốc độ học ⇒ arm PASS lặp horizon + seed 43
  (vòng 14: rej). [Kết quả](docs/methods/full-drift/results.md).
- **Owner 17/10 ~16Z:** chạy thử A (dồn Gaussian vào lớp hiếm), B (B1 lọc mẫu bằng proxy; B2 = P2 mixture), C (gp theo
  prior thực) và các đề xuất owner (file ưu tiên 17/10 → đã tách thành từng file trong
  [research/methods](docs/research/methods/README.md), file gốc đã xóa). Rà thực nghiệm cũ trước khi train; arm cần train chạy
  **từ đầu ep1→6** (không fork checkpoint); P4 chỉ teacher-only (tiền lệ âm Build13 S13); q2 từ đầu. C/P3/P4/P1-coverage/P2-mode
  = kernel probe không train. [Design §Vòng 13](docs/methods/full-drift/design.md).
- **Owner 17/10 ~13–14Z:** ưu tiên (1) **communication budget hạn chế** — thông tin trao đổi phải thật quan trọng, truyền
  hợp lý; (2) **hỗ trợ đa mô hình** (R2); mục tiêu = metrics của model private. Computational cost không phải mục tiêu, nhưng
  **tối ưu mã không đổi phương pháp được làm** (Kaggle 2×T4 16GB, giảm thời gian tối đa); không đổi gì làm mất hiệu quả
  huấn luyện. **Bỏ mọi hướng giảm độ phức tạp model** (không proxy tỉa thưa/slim).
- **Owner 17/10 ~10:10Z: mục tiêu hiện tại = cải thiện phương pháp huấn luyện để đạt metrics tốt hơn.** Mọi kịch bản
  dùng một kiểu trao đổi như [FL_NIDS_10client](https://github.com/odixe06/FL_NIDS_10client): mỗi round mọi client train
  hết 1 epoch local trên toàn bộ dữ liệu rồi mới gộp đồng bộ với server (một lần/round).
- **Owner 16/10 ~20:30–21:30Z:** kiểm mọi nhánh của [kế hoạch 17/10](docs/archive/research/algorithm-improvement-plan-2026-10-17.md)
  (giai đoạn A: chất lượng trước, compression sau), dùng mọi slot quota; cổng §7.1 + client tệ nhất; control
  `fd_dml_m4b` s42 đã audit dùng lại. Account nhỏ chỉ push khi giờ-tới-reset × session < GPU left. **Headline DML = m4b**
  (seed 43 PASS). Layout: không còn symlink gốc `mh`/`papers` (Python local cần `.pth` → `.compat`, xem
  [project-layout](docs/operations/project-layout.md)); research mỗi phương pháp một file.
- **Owner16/10 ~03:00Z: mục tiêu mới = cải thiện dự đoán của các client, không phải proxy.**
  Owner ~04:00Z chốt: **A primary** (private gp toàn cục), báo kèm local/client tệ nhất; **giữ R2** (private độc lập);
  vòng 2 giữ + cổng phụ private gp đã khóa.
  [Tổng hợp phương pháp + bằng chứng](docs/archive/research/client-model-directions-2026-10-16.md).
- **Owner16/10 ~16:45Z:** full chỉ nối `h_dml` r41→r50 (đối chứng r50 cho m4b full sau; xong 17/10); **dừng** `h_dml_own` r39,
  `h_hybrid` r35, `h_hybrid_own` r33 (phẳng). Seed 43 hai session (vòng 6).
- Owner15/10 tối: **proxy head là headline DML/Hybrid**, private gp báo kèm (ProxyFuse cho thấy private không thêm tín hiệu).
  ⚠ Bị mục tiêu16/10 thay về hướng ưu tiên; giữ làm số đo kèm.
  Full giữ protocol §1 (50×1 epoch, cosine T50, seed42, 20clients, 100%train80%), chờ r50.
  Hybrid continuation chỉ `h_hybrid`/`h_hybrid_own`; controls dùng full-validation (trùng bit-exact). hybf3 đã dừng.
- Owner 17/10 ~05:30Z: **query quota mọi account (kể cả annhtnt/takids1909/accanniez) và dùng hết**. Thử mọi kỹ thuật
  chống học thuộc + QProxy (owner duyệt 15/10).
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
  [Chẩn đoán](docs/archive/research/plateau-diagnosis-2026-10-15.md).
- ProxyFuse probe **FAIL** (fused −0,032/−0,034 so proxy). CCR seed42 utility **FAIL** (pooled−CE +0,0017);
  seed43 recovery v2 đang chạy chỉ để mô tả.
- AntiMem+QProxy r20 hai seed **AUDITED** (~19:00Z): Gaussian private PASS mọi kỹ thuật (m4b +0,025, swappos +0,016–0,018,
  private_ema +0,035–0,039); DML/Hybrid proxy FAIL mọi kỹ thuật, proxy EMA hại; **QProxy q4 PASS** (±0,001, byte 7,65× nhỏ).
  [Kết quả](docs/methods/anti-memorization/results.md). Đề xuất (chưa launch): q4 cho full DML/Hybrid.
- FullDrift seed42 ep6 **AUDITED** (~21:30Z): FedProx giảm drift 5× nhưng F1 0 (benign +0,098); FedAvgM β0,9 phân kỳ;
  FedAvg ≈ DML (+0,009); Gaussian swappos +0,052, private_ema Gaussian +0,024/+0,036. [Kết quả](docs/methods/full-drift/results.md).
- Q2 toàn bộ COMPLETE/AUDITED: full local r23, Gaussian r20, DML r23, DML-own r21; Hybrid r18, Hybrid-own r17.
  Test (mô tả): DML proxy 0,7869, Hybrid proxy 0,7727, private gp 0,7332/0,7348. Tới r50 còn ≈116 session-hours, cần owner cấp quota.
  [Bảng kernel và việc còn thiếu](docs/archive/operations/continuation-2026-10-15-evening.md).

## Phiên sau

Chi tiết + lệnh: [bàn giao §Trạng thái gộp 18/10 ~09:00Z](docs/operations/continuation-2026-10-17.md). Theo thứ tự:
1. Bật lại Monitor `poll.sh` (scratchpad) nếu đã tắt; poll nền chết khi phiên đóng ⇒ kiểm status Kaggle tay.
2. Collect + audit từng session xong (một tiến trình mỗi lần): pd (**frz s43 PASS +0,037**), pb (gpool s43 control), **pa (gkd s43 PASS
   +0,059 vs gpool, +0,069 vs m4b)** đã audit ~09:40Z; pc (q2_frz_gkd) **FAIL −0,226** (snapshot q2 hỏng teacher frz);
   vòng 17: **ra la PASS +0,016** (0,8189), rb kdpx FAIL +0,006 (proxy +0,016), **rc la_kdpx PASS +0,0195 (0,8225)**; vòng 18 sa cop FAIL +0,003, sc t2 FAIL, se gk2 FAIL,
   **sf q4_frz_gkd PASS noninferior (byte proxy ÷7,6)**; sb pce FAIL (proxy +0,059), sd g512 FAIL +0,001;
   vòng 16–18 xong, còn rd;
   `report_fd.py`; ghi results/LESSONS/CONTEXT.
3. **rd đã push 09:39Z** (`trietbackup/mobihet-fd-s42rd-20261015-r17-v1`, gkd ep16→50, code r17, ~11,75h) ⇒ collect khi xong.
4. **Vòng 19 push 14:45Z** (`--fast`: ta/td gkd đối chứng s42/s43, tb/te la_kdpx s42/s43, tc q4+la_kdpx) ⇒ collect, cổng `@fast`;
   arm thắng nối tiếp ep16→50 + gkd `--fast` ep50; g512 (sd) còn chạy. [Bàn giao §Vòng 19](docs/operations/continuation-2026-10-17.md).
5. Vòng 19 (tag mới) build với `'fast': True` trong plan `gen_fd.py`: kèm đối chứng gkd s42 ep15 `--fast` (+ đối chứng khác cổng cần);
   cổng ghi khóa `@fast`. ms/bước T4 của `--fast` đo từ run đối chứng đó (`ms_per_step.py`).
6. Quota tuần (reset 24/10 00:00Z): còn ~21–30h/account (query 06:00Z 18/10); không commit Git/DVC khi chưa được yêu cầu.

## Muốn biết → mở

| cần biết | tài liệu |
|---|---|
| Kernel, checkpoint, budget, việc chưa làm, lệnh tiếp tục | [Bàn giao 17/10](docs/operations/continuation-2026-10-17.md); cũ: [15–16/10](docs/archive/operations/continuation-2026-10-15-evening.md) |
| Toàn bộ phương pháp, PASS/FAIL/chưa kiểm, research mới | [Theo tên](docs/research/methods/README.md), [danh mục](docs/proposals.md), [hồ sơ](docs/methods/README.md) |
| Mọi con số/full metrics đã audit | [H1/H2](docs/methods/shared-baselines/builds/federated-baselines-full/results/latest-rounds.md), [Hybrid](docs/methods/gaussian-proxy-hybrid/builds/hybrid-full/results/latest-rounds.md) |
| Lỗi runtime/recovery và nguyên tắc không lặp | [Sổ lỗi](docs/operations/notebook-errors.md), [LESSONS](LESSONS.md) |
| Dữ liệu, kiến trúc, protocol | [Knowledge](knowledge/README.md), [protocol](docs/research/protocol.md) |
| Chi phí tính toán (Big-O client/server, đo) so `fl_dagsnet_veremi` | [Computational cost](docs/research/computational-cost.md) |
| Có đúng FL không: cái gì rời client, SecAgg giả lập, scaler, holdout gộp | [Luồng thông tin](docs/research/federated-information-flow.md) |
| Script, layout, alias, restore/backup | [Scripts](scripts/README.md), [layout](docs/operations/project-layout.md), [backup](docs/operations/backup.md) |

CONTEXT giữ trạng thái/quyết định và đường dẫn; số đo/commands/contracts ở docs của build hoặc operations.
Tri thức dữ liệu/kiến trúc ở knowledge; quy trình dùng lại ở skills. Bài học mới cập nhật LESSONS ngay.
Sau thêm/chuyển file: cập nhật index, kiểm link/hash/ignore; không tự backup hoặc dọn artifact.
