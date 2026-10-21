# Phiên tối 15/10 — chững sớm, ProxyFuse, AntiMem/QProxy

Đọc sau [CONTEXT](../../../CONTEXT.md) → [LESSONS](../../../LESSONS.md). Bàn giao chiều giữ lịch sử:
[continuation-2026-10-15-afternoon](continuation-2026-10-15-afternoon.md). Số đo đầy đủ ở doc từng phương pháp.

## Quyết định owner phiên này

- Hybrid-full chạy lại arm trùng full-validation ⇒ owner đã **dừng hybf3** (takids1909). Batch Hybrid sau chỉ nối
  `h_hybrid` + `h_hybrid_own`; controls lấy từ full-validation. Không collect hybf3 (100% trùng), giữ kernel.
- **Proxy head là headline DML/Hybrid**, private gp báo kèm. Full giữ protocol §1, chờ r50.
- CCR không nối reference; chốt theo cổng own/shuffle/CE sau audit.
- Thử mọi kỹ thuật chống học thuộc (swap, M4b, EMA+BN local, kết hợp) + QProxy trên pilot r20 hai seed,
  account `nguyenminhtran060113` (30h, reset 17/10 00:00 UTC). Không cần query lại quota account này.

- ~15:00Z: owner thêm 3 account token-only (annhannh331/anhtruong1909/thuyanh316, 30h mỗi account); **ưu tiên kỹ thuật
  mới**, còn chỗ mới chạy full; sàng lọc bằng **một seed chung 42**, seed khác để sau. Duyệt FullDrift trên full.

## Trạng thái kernel (đọc ~12:10Z, cập nhật 15:35Z)

| kernel | trạng thái | việc tiếp |
|---|---|---|
| annhtnt/mobihet-hybf1-s42-20261015-q2-v1 | COMPLETE, AUDITED r9/r9 (1 round mới/4h) | — |
| annhtnt/mobihet-hybf2-…-q2-v1, takids1909/mobihet-hybf4-…-q2-v1 | RUNNING, heartbeat mới | collect khi terminal |
| takids1909/mobihet-hybf3-…-q2-v1 | owner dừng (trùng) | không collect |
| accanniez/mobihet-full1-…-q2-v1, fulld-…-q2-v1 | RUNNING, heartbeat mới | collect khi terminal |
| minhtriethihi/mobihet-cr-u-s42-20261015-v2 | terminal, AUDITED (7/8 r20, dml r16) | seed42 FAIL |
| minhtriethihi/mobihet-cr-u-s43-20261015-v1 | ERROR emptybytes r3, AUDITED; cost r3 PASS | đã recovery |
| minhtriethihi/mobihet-cr-u-s43-20261015-v2 | PUSHED10:49Z (2,25h), CPU mount r3 PASS, r4 heartbeat | collect khi terminal |
| nguyenminhtran060113/mobihet-pf-probe-20261015-v1 | COMPLETE, ProxyFuse FAIL | xong |
| nguyenminhtran060113/mobihet-am-s42/s43-20261015-v1 | COMPLETE 18:52Z, **AUDITED** (s42 10/11 r20, s43 11/11) | gates xong → [kết quả](../../methods/anti-memorization/results.md) |
| nguyenminhtran060113/mobihet-am-s42/s43-20261015-tail-v1 | NOT PUSHED, SUPERSEDED (tác vụ tự push đã dừng) | không push |
| odixeuit/mobihet-am-s42-20261015-tail-v2, minhtrit06/mobihet-am-s43-20261015-tail-v2 | đã sinh (cùng source/arm tail-v1, 3,75h; quota 4,89/4,85h nên không nối phiên); dataset gốc `nguyenminhtran060113/mobihet-antimem-src-20261015` owner share (account này 403 khi tạo dataset), names/sizes PASS; **COMPLETE + AUDITED** (4/4 arm r20 mỗi seed, predictions→10metrics) | gates cần `am_dml`/`am_gaussian`/`am_hybrid` của session chính (~19:00Z) → `report_antimem.py` |
| annhannh331/mobihet-fd-s42a-20261015-v2 | PUSHED 15:55Z (v1 15:35Z lỗi gate, xem sổ lỗi), 7h cap: fd_dml ‖ fd_dml_sub4 | collector FullDrift (chưa viết) |
| annhannh331/mobihet-fd-s42b-20261015-v2 | PUSHED 15:55Z (v1 15:35Z lỗi gate, xem sổ lỗi), 8,5h cap: fd_dml_prox ‖ fd_gaussian→fd_gaussian_swappos | như trên |
| anhtruong1909/mobihet-fd-s42c-20261015-v2 | PUSHED 15:55Z (v1 15:35Z lỗi gate, xem sổ lỗi), 7,5h cap: fd_fedavg→fd_fedavg_sub4 ‖ fd_dml_avgm | như trên |
| thuyanh316/mobihet-fulld-s42-20261015-q3-v1 | từ q2 audited (DML r23, DML-own r21), 11,5h; ckpt 253 file names/sizes PASS, CPU import gate PASS; **PUSHED 16:47Z** | collect khi terminal (`collect_full.py --only "*20261015-q3-v1" --once`) |
| thuyanh316/mobihet-hybh-s42-20261015-q3-v1 | Hybrid r18 (hybf2-q2), Hybrid-own r17 (hybf4-q2), 11,5h; ckpt 213 file PASS, CPU gate PASS; **PUSHED 16:47Z** | collect hybrid khi terminal |

## Q2 đã collect/audit (~12:30Z)

Bốn kernel q2 COMPLETE, collector PASS (source/strict weights/predictions→10metrics), report regenerate có bảng proxy headline.
Full: local r23, Gaussian r20, DML r23, DML-own r21. Hybrid: Hybrid r18, Gaussian r18, Hybrid-own r17.
Official test (mô tả): DML proxy r23 0,7869 (r20 0,7879), private gp 0,7332; Hybrid proxy r18 0,7727, private gp 0,7348;
Gaussian 0,5893; local 0,5072. Proxy headline: DML > Hybrid. [Full](../../methods/shared-baselines/builds/federated-baselines-full/results/latest-rounds.md),
[Hybrid](../../methods/gaussian-proxy-hybrid/builds/hybrid-full/results/latest-rounds.md).

Ước phần còn lại tới r50 từ nhịp q2 (4h/session): local ~0,5h/round, Gaussian/G-own ~0,6h, DML/DML-own ~1,25h,
Hybrid/Hybrid-own ~1,3h. Session 2 arm song song ⇒ ≈ 18h (local+Gaussian) + 36h (DML+DML-own) + 43h (Hybrid+Hybrid-own)
+ 19h (G-own) ≈ **116 session-hours** (diag/lr4 tạm hoãn). Envelope q1+q2 89,7/90h đã dùng; cần owner cấp quota/thứ tự.

## Việc còn thiếu — phiên sau

- [x] **FullDrift** audited + report (21:30Z): [kết quả](../../methods/full-drift/results.md); vòng 2 chờ owner.
- [x] **AntiMem** audited + gates (19:00Z): Gaussian PASS, proxy FAIL, QProxy q4 PASS; `am_hybrid_swappos` s43 hai account bit-exact.
- [ ] **CCR seed43 v2** collect (`collect_rank_utility.py`), report (`report_rank_utility.py`); mô tả, CCR đã FAIL theo seed42.
- [ ] **q2 full/Hybrid** collect theo lệnh trong bàn giao chiều; regenerate reports. Báo proxy head làm headline
  DML/Hybrid (cần thêm cột/bảng proxy vào `report_full_validation.py`/latest-rounds).
- [ ] Batch continuation q3 ([plan](continuation-plan-2026-10-15-q3.json)): thuyanh316 2×11,5h ≈ 11 round/arm; còn thiếu tới r50 sau q3 (DML ~r34, Hybrid ~r29).
- [ ] Nếu AntiMem có kỹ thuật PASS cả hai seed: đề xuất kiểm full cho owner (chi phí ~1,8h/round/arm). Không tự mở.
- [ ] Artifact mới (ProxyFuse, AntiMem, CCR v2, hybf1 q2) chưa DVC backup; không Git/DVC push nếu owner chưa yêu cầu.

## Lệnh

```bash
conda run -n nckh python methods/cross-client-ranking/builds/rank-utility/scripts/collect_rank_utility.py
conda run -n nckh python methods/cross-client-ranking/builds/rank-utility/scripts/report_rank_utility.py
H=.agents/skills/kaggle-training-notebook/scripts
conda run -n nckh python $H/kaggle_as.py nguyenminhtran060113 -- python $H/pull_kernel_output.py \
  nguyenminhtran060113/mobihet-am-s42-20261015-v1 -p methods/anti-memorization/builds/antimem-pilot/artifacts/kaggle/am-s42-20261015-v1 \
  -j 6 --pattern '^(source/|runs/[^/]+/(complete/|config\.json|result\.json|metrics\.csv|log\.txt|predictions/round_020)|runs/antimem_summary)'
```

## Proof

ProxyFuse: [kết quả](../../methods/proxy-fusion/results.md), restore proof `methods/proxy-fusion/builds/fuse-probe/probes/`.
AntiMem preflight: [README](../../methods/anti-memorization/README.md). CCR: [build](../../methods/cross-client-ranking/builds/rank-utility/README.md).
Lỗi build phiên này: [sổ lỗi](../../operations/notebook-errors.md#phiên-tối-1510--lỗi-build-và-cách-tránh).

## Dừng phiên ~22:30Z 15/10 (owner khởi động lại máy) — làm tiếp từ đây

**Đang chạy trên Kaggle (không phụ thuộc máy local):** `thuyanh316/mobihet-fulld-s42-20261015-q3-v1` (h_dml từ r24,
h_dml_own từ r22) và `thuyanh316/mobihet-hybh-s42-20261015-q3-v1` (h_hybrid r19, h_hybrid_own r18), PUSHED 16:47Z,
11,5h ⇒ terminal ~04:30Z 16/10. Startup đã kiểm (resume đúng round, progress W&B). Collect khi terminal:

```bash
conda run -n nckh python methods/shared-baselines/builds/federated-baselines-full/scripts/collect_full.py --only '*20261015-q3-v1' --once
conda run -n nckh python -c 'import runpy; runpy.run_path("methods/gaussian-proxy-hybrid/builds/hybrid-full/scripts/collect_hybrid_full.py")["collect"]()'
conda run -n nckh python methods/shared-baselines/builds/federated-baselines-full/scripts/report_full_validation.py
conda run -n nckh python methods/shared-baselines/builds/federated-baselines-full/scripts/report_full_validation.py --build hybrid-full
```
(Kiểm collector hybrid có nhận rid `hybh-…-q3-v1` không trước khi chạy; q2 dùng tên hybf1–4.)

**FullDrift vòng 2 — dở, CHƯA push.** Owner duyệt cả bốn hướng proxy (sub4+q4, sub16+q4, sub4+prox, horizon ep15);
Gaussian chờ. Contract khóa: [design §Vòng 2](../../methods/full-drift/design.md). Đã làm:
- `fd.py`: thêm 6 arm (`fd_{dml,fedavg}_sub4_q4`, `_sub16_q4`, `_sub4_prox`), port QProxy q4 từ antimem,
  parse token (`sub4`/`sub16`/`q4`/`prox`), `--stop_after 15` ghi thư mục `<arm>_s42_e15`.
- `test_fd_local.py` tách `--part checks|smoke` (owner: máy bận, chỉ một tiến trình CUDA mỗi lúc, đỉnh ~2,5GB).
  **Chưa chạy lại sau thay đổi vòng 2.** Chạy lần lượt, kiểm `free -m` trước:

```bash
W=<scratch>/fd-smoke; M=methods/full-drift/builds/fd-probe/artifacts/kaggle/fd-s42a-20261015-v2/source/data_manifest.json
PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 conda run -n nckh python scripts/run_local_guarded.py \
  python methods/full-drift/builds/fd-probe/scripts/test_fd_local.py --work $W --part checks --manifest $M
PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 conda run -n nckh python scripts/run_local_guarded.py \
  python methods/full-drift/builds/fd-probe/scripts/test_fd_local.py --work $W --part smoke
```
Kỳ vọng: bit-exact fd_dml/fd_gaussian vs full_run, sub4 một phần == đối chứng, sub4_q4 gần dense (0<rel<0,5),
mọi arm smoke rc0 (sub16 dùng batch32).

Còn phải làm trước push:
1. `gen_fd.py`: PLAN vòng 2 (suffix `r2-v1`), cho phép mỗi phần tử queue mang epoch (vd `fd_dml:15`) và RUN truyền
   `--stop_after`; session ≤ 12h. Plan đề xuất (ước ~26,5 session-h):
   anhtruong1909 S1 [fd_dml:15] ‖ [fd_dml_sub4:15] ~10h; anhtruong1909 S2 [fd_fedavg:15 → fd_fedavg_sub4_q4] ‖
   [fd_fedavg_sub4:15 → fd_fedavg_sub16_q4] ~8,5h; annhannh331 S3 [fd_dml_sub4_q4 → fd_dml_sub4_prox] ‖
   [fd_dml_sub16_q4 → fd_fedavg_sub4_prox] ~8h. Quota ước (không query lại 3 account cũ): annhannh331 ~18h,
   anhtruong1909 ~24h, thuyanh316 ~7h sau q3, odixe0502 4h; reset 17/10 00:00 UTC.
2. Stage mới (`fd.py` đổi ⇒ `artifacts/input/fd-src-20261015` hiện là bản vòng 1; chuyển sang `…-v2` hoặc tag mới),
   upload version dataset cho từng owner, verify names/sizes bằng account chạy, push.
3. `collect_fd.py`/`report_fd.py`: nhận session vòng 2 và thư mục `_e15`; cổng vòng 2 so đối chứng vòng 1.
4. Kiểm startup (W&B progress) sau push, rồi để chạy.

**Kết quả đã chốt phiên này:** AntiMem ([results](../../methods/anti-memorization/results.md)) và FullDrift vòng 1
([results](../../methods/full-drift/results.md)); LESSONS thêm 3 bài học; CONTEXT/proposals cập nhật.
Artifact mới (AntiMem, FullDrift, ProxyFuse, CCR v2, q2) chưa DVC backup; không Git/DVC push nếu owner chưa yêu cầu.

### Tiếp tục 16/10 ~02:00Z

- Test vòng 2 PASS (checks + smoke 14/14, RAM ≤2,4GB/lượt). `gen_fd.py --round r2`, collector/report nhận vòng 2.
- **PUSHED 02:19Z:** `anhtruong1909/mobihet-fd-s42d-20261015-r2-v1`, `…-s42e-…-r2-v1`, `annhannh331/mobihet-fd-s42f-20261015-r2-v1`.
  Quota live trước push: annhannh331 20,7h, anhtruong1909 26,0h. Xong dự kiến ~12:00Z. Collect: `collect_fd.py`, `report_fd.py`.
- q3 (thuyanh316) W&B 02:00Z: h_dml r32 (proxy test 0,786), h_dml_own r30, h_hybrid r26 (0,764), h_hybrid_own r25; khỏe, ~57–62 phút/round.

### Dừng phiên 16/10 ~04:30Z (owner tắt máy; mọi job đang chạy nằm trên Kaggle)

**Owner đổi mục tiêu (~03:00Z): cải thiện dự đoán của client, không phải proxy.** Chốt ~04:00Z: primary = private gp
trên holdout/test toàn cục (A), báo kèm local (prior nhãn từng client) và client tệ nhất; giữ **R1** (contact/byte
hạn chế) và **R2** (model riêng mỗi client; hiện DAGSNet). Tổng hợp nguồn + bằng chứng:
[client-model-directions](../research/client-model-directions-2026-10-16.md); tool [client_view.py](../../../scripts/client_view.py).

| kernel | nội dung | trạng thái | việc kế |
|---|---|---|---|
| thuyanh316/mobihet-fulld-s42-20261015-q3-v1 | h_dml từ r24, h_dml_own từ r22 | **COMPLETE** (W&B 03:30Z h_dml r33) | `collect_full.py --only '*20261015-q3-v1' --once` |
| thuyanh316/mobihet-hybh-s42-20261015-q3-v1 | h_hybrid từ r19, h_hybrid_own từ r18 | **COMPLETE** (h_hybrid r27, own r27) | `collect_hybrid_full.py` (duyệt mọi folder PUSHED, nhận hybh) |
| anhtruong1909/…-fd-s42d/e-20261015-r2-v1, annhannh331/…-s42f-…-r2-v1 | FullDrift vòng 2 | RUNNING (W&B 03:29Z ổn) | xong ~12:00Z → `collect_fd.py`, `report_fd.py` |
| nguyenminhtran060113/mobihet-fd-s42g-20261015-r3-v1 | vòng 3: `fd_dml_m4b` ‖ `fd_dml_swappos` | **PUSHED 04:17Z**, 4,5h; W&B 04:26Z cả 4 arm g/h có progress | xong ~08:45Z → collect |
| nguyenminhtran060113/mobihet-fd-s42h-20261015-r3-v1 | vòng 3: `fd_dml_ntd` ‖ `fd_dml_q4` | **PUSHED 04:17Z**, 4,5h | như trên |

Vòng 3: cổng private gp khóa ~04:05Z ([design §Vòng 3](../../methods/full-drift/design.md)); test local checks +
smoke 18/18 PASS (đỉnh 2,5GB); dataset `nguyenminhtran060113/mobihet-fulldrift-src-20261015-r3` 48 file names/sizes PASS.
Vòng 2 có thêm cổng phụ private gp (khóa ~03:50Z, trước kết quả ep6). `report_fd.py` in thêm bảng góc nhìn client.

Lệnh collect (chạy lần lượt, một tiến trình nặng mỗi lúc):
```bash
conda run -n nckh python methods/shared-baselines/builds/federated-baselines-full/scripts/collect_full.py --only '*20261015-q3-v1' --once
conda run -n nckh python methods/gaussian-proxy-hybrid/builds/hybrid-full/scripts/collect_hybrid_full.py
conda run -n nckh python methods/full-drift/builds/fd-probe/scripts/collect_fd.py   # chỉ audit session terminal
conda run -n nckh python methods/full-drift/builds/fd-probe/scripts/report_fd.py
```
Sau collect q3: regenerate report full/Hybrid (`report_full_validation.py`, report hybrid) như q2.

**Quota live ~04:00Z (mọi account reset 17/10 00:00Z, phần dư mất):** nguyenminhtran060113 16,6h (−9h vòng 3 ⇒ ~7,6h),
annhannh331 19,0h (f còn chạy tới ~12:50Z), anhtruong1909 12,3h (10,3h giữ cho d/e), thuyanh316 8,4h (rảnh),
odixe0502 4,0h, odixeuit 2,4h, khanhmay0304 2,5h, minhtriethihi 2,3h, minhtrit06 2,1h, trietbackup 2,1h,
catbaochau 1,1h, dinhan25 0,9h, minhtran0601 0,7h. Không query annhtnt/takids1909/accanniez.
Đọc quota từng account bằng `kaggle_account.account_quota(user)` (lệnh `quota` của helper quét cả 3 account cấm).

Việc kế (theo thứ tự, owner duyệt từng launch):
1. Collect q3 → report. Rồi cân nhắc full q4 (thuyanh316 8,4h + phần dư) trước 00:00Z nếu owner muốn dùng nốt quota.
2. Khi vòng 2/3 terminal: collect, report, đánh giá cổng private (A) + proxy; ghi results/LESSONS/CONTEXT.
3. Ứng viên tiếp theo cho client (R1/R2): hidden-state KD có projector local (FedKD), FedRoD hai head; nhánh B cần
   fixture holdout theo client. Kết hợp kỹ thuật thắng vòng 3 với q4.

### Tiếp tục 16/10 ~05:15Z

- q3 **AUDITED** (pull `-j 6`, 8,8GB/346s): h_dml r34, h_dml_own r32, h_hybrid r28, h_hybrid_own r27, continuation verified.
  Report full/Hybrid tạo lại. Collector Hybrid lặp vô hạn (hybf1-cont01/hybf3-q2 không AUDITED) ⇒ dừng tay; collector có `--jobs`,
  `--once` (Hybrid), `--only` (FD) theo skill pull-outputs (~32 kết nối tổng, một tiến trình mỗi session).
- **q4** ([plan](continuation-plan-2026-10-16-q4.json), owner 15/10: quota dư chạy tiếp full; quota reset 17/10 00:00Z):

| kernel | parents | trạng thái |
|---|---|---|
| thuyanh316/mobihet-fulld-s42-20261016-q4-v1 (7,8h) | h_dml r34, h_dml_own r32 (q3) | ckpt 352 file PASS, CPU gate PASS, **PUSHED 07:22Z**; W&B 07:50Z r35/r33 chạy |
| annhannh331/mobihet-hybh-s42-20261016-q4-v1 (7,8h) | h_hybrid r28, h_hybrid_own r27 (q3) | ckpt 303 file PASS, CPU gate PASS, **PUSHED 07:32Z**; W&B 08:09Z r29/r28 chạy |

- **FullDrift vòng 4** `odixe0502/mobihet-fd-s42i-20261015-r4-v1` PUSHED 06:32Z (3,95h): `fd_dml_hkd` ‖ `fd_gaussian_m4b`,
  cổng [§Vòng 4](../../methods/full-drift/design.md). W&B 07:02Z có progress.
- W&B 07:02Z (chưa audit, không dùng cho cổng): m4b ep5 private gp 0,693 (fd_dml ep5 0,658), swappos ep4 0,679 (0,638),
  ntd ep4 0,591, q4 ep4 0,639; m4b proxy ep5 0,807 (0,765). Vòng 2: sub4_q4/sub16_q4 xong ep6.
- Quota live ~06:30Z: odixe0502 4,01h (vòng 4 dùng hết), thuyanh316 8,41h (q4), annhannh331 16,53h (f + Hybrid q4),
  nguyenminhtran060113 2,4h + 9,76h giữ cho g/h (sau g/h còn ~9h).

Collect (song song, một tiến trình mỗi session, chia ~32 kết nối):
```bash
conda run -n nckh python methods/full-drift/builds/fd-probe/scripts/collect_fd.py --only 'fd-s42g-*' --jobs 16 &
conda run -n nckh python methods/full-drift/builds/fd-probe/scripts/collect_fd.py --only 'fd-s42h-*' --jobs 16 &
wait; conda run -n nckh python methods/full-drift/builds/fd-probe/scripts/report_fd.py
```

### 16/10 ~08:30Z

- **FullDrift vòng 3 AUDITED** (collect lần 1 lỗi: assert so fd.py với source sống; lần 2 song song bị guard dừng ~4GB/audit;
  đã sửa: so với bản stage của vòng, audit giữ lock): m4b private gp +0,036/proxy +0,052, swappos +0,033/+0,049 PASS;
  ntd −0,042 FAIL; q4 noninferior. [Kết quả](../../methods/full-drift/results.md).
- Owner ~08:20Z: vòng 5 = horizon + ghép; headline full chờ horizon + seed 43.
  **PUSHED 08:26Z** `nguyenminhtran060113/mobihet-fd-s42j-20261015-r5-v1` (8,95h; quota 9,27h): `fd_dml_m4b:15` ‖
  `fd_dml_m4b_q4` → `fd_dml_m4b_sub4_q4`. Xong ~17:25Z. [Cổng §Vòng 5](../../methods/full-drift/design.md).
- Đang chạy: vòng 2 d/e/f (~12:00Z), vòng 4 i (~10:20Z), q4 full (~15:10Z), q4 Hybrid (~15:20Z), vòng 5 j (~17:25Z).
  Quota còn rảnh sau các job: anhtruong1909 ~8h sau d/e (~12:00Z) — chưa có kế hoạch, cần owner nếu muốn dùng trước 00:00Z.

### 16/10 ~17:10Z

- Collect song song (3 tiến trình, audit tuần tự qua lock, đỉnh 4,07GB): FD d/e/f (vòng 2), i (vòng 4), j (vòng 5),
  q4 full (`h_dml` r41, `h_dml_own` r39), q4 Hybrid (`h_hybrid` r35, `h_hybrid_own` r33) đều **AUDITED**; report FD/full/Hybrid tạo lại.
  Kết quả: [FullDrift](../../methods/full-drift/results.md) (vòng 2, 4, 5), LESSONS thêm 3 bài học.
- **Owner ~16:45Z:** (1) full chỉ nối `h_dml` tới r50, dừng own/Hybrid; (2) seed 43 hai session; (3) R1 cần mô phỏng contact
  trước khi chọn tần suất tổng hợp → [bước 1](https://github.com/odixe06/mobihet-contact/blob/main/docs/contact-budget-r1.md) (q4 vừa một contact mọi giả định;
  contact/epoch phụ thuộc mật độ RSU × slowdown máy xe: sub4 cần RSU ≤1 km và slowdown ≥5×).
- **Vòng 6 PUSHED 16:52Z** (cổng khóa ~16:50Z, [design §Vòng 6](../../methods/full-drift/design.md)); W&B k 17:04Z, l 16:58Z có run:

| kernel | GPU0 | GPU1 | xong |
|---|---|---|---|
| anhtruong1909/mobihet-fd-s43k-20261015-r6-v1 (4,2h) | fd_dml s43 | fd_dml_m4b s43 | ~21:10Z |
| anhtruong1909/mobihet-fd-s43l-20261015-r6-v1 (4,2h) | fd_dml_sub4_q4 s43 | fd_dml_m4b_sub4_q4 s43 | ~21:10Z |
| annhannh331/mobihet-fd-s43m-20261015-r7-v1 (4,2h, vòng 7, PUSHED 17:32Z, owner ~17:25Z) | fd_dml_swappos s43 | fd_gaussian_m4b s43 | ~21:45Z |

```bash
conda run -n nckh python methods/full-drift/builds/fd-probe/scripts/collect_fd.py --only 'fd-s43k-*' --jobs 16 &
conda run -n nckh python methods/full-drift/builds/fd-probe/scripts/collect_fd.py --only 'fd-s43l-*' --jobs 16 &
conda run -n nckh python methods/full-drift/builds/fd-probe/scripts/collect_fd.py --only 'fd-s43m-*' --jobs 8 &
wait; conda run -n nckh python methods/full-drift/builds/fd-probe/scripts/report_fd.py   # gate keys '<arm>@s43'
```

- **q5** ([plan](continuation-plan-2026-10-17-q5.json)): `thuyanh316/mobihet-fulld-s42-20261017-q5-v1` (`h_dml` từ r41, 9,8h);
  ckpt `thuyanh316/mobihet-fulld-s42-20261017-q5-v1-ckpt` 256 file names/sizes PASS; **CPU exact import/state/SHA gate PASS** (~17:37Z). W&B vòng 7 m có run 17:37Z.
  GPU push **chỉ sau 00:00Z 17/10** (thuyanh316 còn 1,68h trước reset):
```bash
H=.agents/skills/kaggle-training-notebook/scripts
conda run -n nckh python $H/kaggle_as.py thuyanh316 -- kaggle kernels push -p papers/full-validation/private_push/fulld-s42-20261017-q5-v1
# rồi ghi PUSHED vào private_push/<rid> và notebook/<rid> như q4; collect: collect_full.py --only '*20261017-q5-v1' --once
```
- Quota live ~16:45Z: anhtruong1909 9,36h (vòng 6 dùng ≤8,4h), annhannh331 5,29h, thuyanh316 1,68h, nguyenminhtran060113 1,12h,
  odixe0502 0,39h, account khác ~2,1–2,5h. Reset 17/10 00:00Z. annhannh331 dùng 4,2h cho vòng 7.
- `scripts/update_file_index.py` cần `rg`: máy không có trong PATH; chạy với `PATH=<vscode-server>/node_modules/@vscode/ripgrep-universal/bin/linux-x64:$PATH`.
