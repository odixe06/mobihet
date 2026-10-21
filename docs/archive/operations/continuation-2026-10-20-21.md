# Nhật ký vận hành run cuối seed42, 20/10 16Z → 21/10 08Z

Chuyển nguyên văn từ [continue.md](../../operations/continue.md) lúc 21/10 ~09Z (trước compact). Trạng thái hiện hành ở continue.md.

## Trạng thái hiện tại

- r24 zc/zi đã collect/audit:54file/kernel, đúng source,0round. Crash Gaussian đọc pilot manifest; không checkpoint để resume.
- Owner đã Cancel za/ze/zg/zk; API xác nhận đủ4 CANCEL_ACKNOWLEDGED.
- r25 sửa active fixture context + notebook propagate childrc. Local compiled parity/resume/detached eval PASS;
  non-smoke final-only hai arm×20/50/100c×2rank +tail PASS.
- Owner yêu cầu **push đủ6kernel Kaggle seed42**, kiểm kỹ startup. Chỉ Kaggle; không Git/DVC/backup/cleanup.
- Đã push đủ6r25 lúc16:39–16:44Z; source49file/account verified. Remote source/input pull-back6/6 PASS.
  [Launch ledger/quota/PUSHED](../../operations/evidence/final-2026-10-20/r25-launch.json); [health](../../operations/evidence/final-2026-10-20/r25-health.json), [startup](../../operations/evidence/final-2026-10-20/r25-startup.json),
  [remote source proof](../../operations/evidence/final-2026-10-20/r25-remote-source.json).
  Không gọi train ổn định từ RUNNING.
- Manual action hiện không còn sau khi owner Cancel4kernel; phát sinh lỗi quyền/quota cần ghi exact bước.

## Kiểm live gần nhất

20/10 ~16:46Z (23:46 VN): zi/zk RUNNING, log đã xác thực W&B online, chưa traceback; đang chuẩn bị dữ liệu.
za/zc/ze/zg QUEUED chờGPU, không repush. Chưa có training heartbeat/round committed của r25,
**chưa đủ bằng chứng SGD ổn định**; phiên sau phải kiểm bước này trước khi coi startup hoàn tất.
Chuẩn bị dữ liệu đã đo30–45phút, absence W&B training run trong setup chưa là crash.

**20/10 18:23Z:** zg (50c gkd) COMPLETE, collect + audit PASS: 2/2 round committed (session cap), `evaluation_pending`. Timing: chuẩn bị dữ liệu 1.076 s, setup `fd.py` 124 s, round 1 1.032 s (gồm warm-up compile), round 2 810 s; cả session 52 phút. Cần chấm r1–r2 và nối tiếp train. 18:23Z: za ?, zc/ze/zi/zk RUNNING.

**20/10 18:44Z:** ze (50c q3) ERROR, collect PASS: chuẩn bị dữ liệu 4.420 s ⇒ budget 2h còn 0,77h < 1h, không chạy arm (0 round). Không lỗi code; cần repush với session dài hơn ([sổ lỗi](../../operations/notebook-errors.md)).

**20/10 19:20Z — r25 kết thúc cả 6, collect + audit PASS cả 6:** za (20c q3) 4/4 round, zc (20c gkd) 4/4, zg (50c gkd) 2/2, zk (100c gkd) 2/2 — tất cả `evaluation_pending`. **ze, zi ERROR, 0 round**: hết budget session trong lúc chuẩn bị dữ liệu (4.420 s / 8.792 s) — [sổ lỗi](../../operations/notebook-errors.md). Timing thực: chuẩn bị 1.019–8.792 s; setup `fd.py` 99–126 s; round 20c 775–943 s, 50c 810–1.032 s, 100c 1.300–1.515 s (round 1 gồm warm-up). Phương án tiếp: [final-pipeline §Lộ trình nhanh nhất](../../operations/final-pipeline.md).

**20/10 ~20Z — owner duyệt lộ trình nhanh nhất** ([final-pipeline §Lộ trình](../../operations/final-pipeline.md)): session ~11,5h không cap round,
cache fixture bằng kernel CPU, chấm nhiều round/notebook. Đã push:
- 4 notebook chấm đo thời gian: `odixe0502/mobihet-fd-e01-04-s42za-…-r25-v1-v1`, `…-e01-04-s42zc-…`, `khanhmay0304/mobihet-fd-e01-02-s42zg-…`,
  `…-e01-02-s42zk-…` (chuẩn bị dữ liệu trong session; collect: `collect_eval.py <rid>`).
- 6 kernel cache CPU `<owner>/mobihet-fd-fixture-k{K}-20261015-r25`: minhtriethihi/annhtnt/odixe0502 K20, annhannh331/thuyanh316 K50,
  khanhmay0304 K100; xong thì `gen_fixture.py --owner <o> --clients <K> --check` ghi marker để generator gắn `kernel_sources`.
- Source r25 đang upload cho nguyenminhtran060113, accanniez, takids1909, anhtruong1909 (K100), catbaochau, trietbackup (K100),
  dinhan25 (K50), minhtran0601 (K20) ⇒ rồi push cache cho các account đó.

**20/10 ~21Z:** cache xong và S1 đã push: zc02 (annhtnt, nối r4), ze-v2 (thuyanh316, từ r1), zk02 (nguyenminhtran060113, nối r2),
zg02 (annhannh331, nối r2) — đều 11,5h, không cap, gắn cache; zc02 lên W&B ~1–2 phút sau khi chạy (cache hoạt động). Chờ cache minhtriethihi (za02),
accanniez (zi-v2). Chấm za r1–4 audited: 772–805 s/round (20c). Mỗi session nối tiếp có run W&B riêng (config có khối `parent`).
Push S1 còn lại: `launch_s1.sh <owner> <K> cont <parent rid> <session>` / `... fresh <session>` (scratchpad; lệnh tương đương:
`gen_fixture.py --check`, `gen_continue.py`, `upload_stage.py`, `kaggle kernels push -p private_push/<rid>`).

**20/10 22:51Z — S1 đủ 6 chuỗi, W&B ổn định ≥3 round** (loss hữu hạn, heartbeat mới): zc02 r15 (780 s/round), ze-v2 r11 (812 s),
zg02 r12 (792 s), za02, zk02 r8 (1.207 s), zi-v2 r3 (1.439 s). Dự báo S1 (11,5h): zc02, zg02 tới r50; ze-v2 ~r48, za02 ~r48 (S2 ngắn);
zk02 ~r35 (S2 ~15 round, takids1909), zi-v2 ~r27 (S2 ~23 round, anhtruong1909). S1 kết thúc ~07:30–09Z 21/10 ⇒ collect, gen_continue S2,
chấm các round S1 trên account chấm (gắn cache). Chấm zc r1–4 audited (792–820 s/round; chuẩn bị trong session 7.392 s — luôn gắn cache).

Phân bổ: S1 train za→minhtriethihi, zc→annhtnt, zg→annhannh331, ze(v2, từ r1)→thuyanh316, zk→nguyenminhtran060113, zi(v2)→accanniez;
S2 100c: zk→takids1909, zi→anhtruong1909; S2 ngắn 20c/50c trên chính account S1; chấm: odixe0502, khanhmay0304, catbaochau, trietbackup,
dinhan25, minhtran0601 + phần dư account train. S1 push ngay khi cache của account đó COMPLETE.

**21/10 ~02:35Z — bộ chấm nhanh (owner: "chấm nhanh nhất có thể").** S1 cả 6 chuỗi vẫn chạy, W&B 02:20Z: zc02 r30, zg02 r26, za02 r24,
ze-v2 r24, zk02 r17, zi-v2 r11; tự dừng ~06:40–09:00Z. Owner hỏi có nên dừng tay để chấm sớm: khuyên **không** (không rút ngắn đường găng zi,
mất round dở dang, output nhiều GB khi cancel chưa được chứng minh lưu đủ; kernel cancel cũ chỉ chứng minh file nhỏ được giữ).
- Cache odixe0502 K20 COMPLETE (18.854 s CPU), marker ghi. Còn khanhmay0304/takids1909/catbaochau K100 RUNNING.
- `fd_eval_fast.py` (nguồn chấm tag **r26** = r25 + file này; train vẫn r25): forward GEMM fold-BN, worker bền nhiều round + hàng đợi client,
  gộp/nén chồng lên round sau. Bench T4 `odixe0502/mobihet-fd-evalbench-20261021-v1`: 18–19 s/lượt so với 30–34 s (1,7×), lệch 0,02 % dòng.
  Chi tiết [final-pipeline §Bộ chấm nhanh](../../operations/final-pipeline.md). Test: `test_fast_eval.py`, `test_final_pipeline.py` 20c/100c PASS.
- Đang chấm lại bằng bộ nhanh (bằng chứng tương đương + kết quả chính thức đồng nhất): `odixe0502/…-e01-04-s42za-…-f1`, `minhtran0601/…-e01-04-s42zc-…-f1`,
  `dinhan25/…-e01-02-s42zg-…-f1`, `trietbackup/…-e01-02-s42zk-…-f1`. Xong ⇒ `collect_eval.py <rid>` rồi `compare_eval.py <rid cũ -v1> <rid -f1>`.
- Push chấm: [launch_eval.sh](../../../methods/full-drift/builds/fd-probe/scripts/launch_eval.sh) `<train rid> <a-b> <owner>` (= `gen_eval.py` + `upload_stage.py` nguồn r26 và checkpoint + push + `PUSHED`).

**21/10 ~03:20Z — xác nhận bộ chấm nhanh (r26), backup.** za/zc r1–4, zg r1–2 `-f1` collect + audit + `compare_eval.py` PASS: khớp 99,96–99,97 %,
round sau 301–357 s (20c) / 843 s (50c) / 1.449 s (100c, zk 03:45Z), ~2,2–2,8× bản cũ ([bảng + lịch chấm](../../operations/final-pipeline.md)). Các round này dùng kết quả `-f1` (cả chiến dịch một evaluator). Cache K100 catbaochau 03:38Z, takids1909 03:44Z COMPLETE ⇒ **đủ 14 cache** (K20 minhtriethihi/annhtnt/odixe0502/minhtran0601; K50 annhannh331/thuyanh316/dinhan25; K100 khanhmay0304/nguyenminhtran060113/accanniez/takids1909/anhtruong1909/catbaochau/trietbackup).
Chấm đại trà dùng **tag r27** (chỉ nới gate 0,1 → 0,25; `gen_eval.EVAL_TAG`); `launch_eval.sh` tự upload nguồn r27 cho account mới.
Git `b269ab0`/`1c505e6` + DVC fd-probe (kaggle/input/fixture-cache) + bundle xong lúc 02:51Z ([backup](../../operations/backup.md)).
Cache K100 khanhmay0304 COMPLETE (26.426 s CPU), marker ghi; còn takids1909, catbaochau K100.

**21/10 ~05:30–06:40Z — sinh mẫu có chủ đích (owner: "ngoài Gaussian, đặt ưu tiên").** Rà build cũ: ưu tiên chỉ từng thử trên Gaussian nhãn
cứng (fill/rej/P1) hoặc đổi lượng/độ thật (g512/cop/gk2); chưa ai ưu tiên theo chỗ private sai. Probe 20 local (weights xa ep15): Gaussian gkd bất đồng
teacher sai chỗ (cặp tốc độ, cos 0,09), lỗi thật ở khối benign; lai hai dòng own theo cặp bất đồng cos 0,55 ([results §Probe 20](../../methods/full-drift/results.md)).
Cài arm `xo` + đối chứng `xou` (`fd.py`, tag **r28**, [design §Vòng 24](../../methods/full-drift/design.md)); test `--part r24`/`par` rồi push 2 session
trên **odixeuit** (30h, ngoài roster final). odixeuit/minhtrit06 **tạo dataset được lại** (probe 27 B 06:07Z, đã xóa) ⇒ thêm 60h cho chấm/screening.
Poll S1 khởi động lại (scratchpad bị xóa): `<scratchpad>/poll_s1.sh`.

**21/10 06:18Z — zc02 COMPLETE ⇒ chuỗi zc (20c gkd) train xong r50.** Collect + AUDITED: 149 file, 2.974 MiB, r5–r50 (46 round, 604 phút ≈ 788 s/round),
weights strict/hữu hạn, `evaluation_pending`. Chấm r27 đã push ~06:45Z: `minhtran0601/mobihet-fd-e28-50-s42zc02-20261015-r25-v1-f1` (r28–50, gồm r50),
`odixe0502/mobihet-fd-e05-27-s42zc02-20261015-r25-v1-f1` (r5–27); checkpoint 1,56 GB/notebook VERIFIED. Xong ⇒ `collect_eval.py <rid>`.

**21/10 ~06:55Z — screening vòng 24 đã push** (odixeuit, nguồn `odixeuit/mobihet-fulldrift-src-20261015-r28` VERIFIED 49 file): `odixeuit/mobihet-fd-s42xo-20261015-r28-v1` (xo), `odixeuit/mobihet-fd-s42xu-20261015-r28-v1` (xou); s42 ep1→15 `--fast`, 7h/session. Test local trước push: `--part r24` PASS (6 arm cũ bit-exact với r25 hai epoch; xo/xou epoch 1 = control), `--part par` PASS (compiled 1 = 2 worker; bộ lọc test bỏ trường thời gian/bộ nhớ khối `cost`). Xong ⇒ `collect_fd.py --only fd-s42x[ou]-20261015-r28-v1`, `report_fd.py`, cổng [design §Vòng 24](../../methods/full-drift/design.md).

**21/10 07:13–07:37Z — S1 kết thúc 4 chuỗi nữa, collect + AUDITED:** ze-v2 r1–48 (655 phút), zg02 r3–50, za02 r5–50 ⇒ **za, zc, zg train xong r50**;
zk02 COMPLETE (collect 100c đang chạy); zi-v2 còn chạy. S2 ze: `thuyanh316/mobihet-fd-s42ze03-20261015-r25-v1` (r49–50, trần 2h; `gen_continue.py --hours`).
Sửa công cụ: `gen_fd.FROZEN` (nối tiếp tag cũ kiểm stage theo manifest của chính nó, source sống đã sang r28); test r24 không ghi `__pycache__` vào stage.
Chấm r27 đang push: ze r29–48 + r9–28 (dinhan25), r1–8 (annhannh331); za r28–50 (annhtnt), r5–27 (minhtran0601).
Owner hỏi chấm 843 s/round (50c = 101 model × 10,76M dòng): chọn **song song thêm** (không giảm EMA, chưa tối ưu kernel tiếp) ⇒ cache K50 trên
minhtrit06 + odixeuit (CPU) để mở thêm luồng cho zg r3–50; notebook chấm đã push giữ nguyên, không cancel. **08:05Z:** owner đang dùng odixeuit cho dự án khác (giải thích quota 30 → 16,8h) ⇒ chỉ minhtrit06 làm luồng chấm mới; cache K50 odixeuit (CPU, đã push trước khi biết) không dùng. Dataset `odixeuit/mobihet-fulldrift-src-20261015-r25`: CLI báo create lỗi nhưng status ready.

## Exact kernels mới

Mỗi ID: `<account>/mobihet-fd-s42<mã>-20261015-r25-v1`, version1.

| mã | account | clients | arm | cap | giờ/session |
|---|---|---:|---|---:|---:|
| za | minhtriethihi | 20 | q3_zl la_kdpx | 4 | 2 |
| zc | annhtnt | 20 | gkd control | 4 | 2 |
| ze | thuyanh316 | 50 | q3_zl la_kdpx | 2 | 2 |
| zg | annhannh331 | 50 | gkd control | 2 | 2 |
| zi | odixe0502 | 100 | q3_zl la_kdpx | 2 | 3 |
| zk | khanhmay0304 | 100 | gkd control | 2 | 3 |

Sáu seed43 chưa generate/push. Không repush r24 hoặc notebook có PUSHED. Nguồn stage:
`methods/full-drift/builds/fd-probe/artifacts/input/fd-src-20261015-r25/`.
[Local proof](../../operations/evidence/final-2026-10-20/final-recovery-local-2026-10-20.json), source-upload-verification.json từng notebook là names/sizes;
runtime SHA gate chứng minh source mount thực. Private notebookignored0600 mang inline W&B theo quyền đã có.

