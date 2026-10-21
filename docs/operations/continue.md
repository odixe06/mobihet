# Continue — final seed42, 20/10/2026

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → trang này.
[Pipeline/contract/tests/timing](final-pipeline.md) · [Results đã audit](../methods/full-drift/results.md).

## Trạng thái hiện tại

- r24 zc/zi đã collect/audit:54file/kernel, đúng source,0round. Crash Gaussian đọc pilot manifest; không checkpoint để resume.
- Owner đã Cancel za/ze/zg/zk; API xác nhận đủ4 CANCEL_ACKNOWLEDGED.
- r25 sửa active fixture context + notebook propagate childrc. Local compiled parity/resume/detached eval PASS;
  non-smoke final-only hai arm×20/50/100c×2rank +tail PASS.
- Owner yêu cầu **push đủ6kernel Kaggle seed42**, kiểm kỹ startup. Chỉ Kaggle; không Git/DVC/backup/cleanup.
- Đã push đủ6r25 lúc16:39–16:44Z; source49file/account verified. Remote source/input pull-back6/6 PASS.
  [Launch ledger/quota/PUSHED](evidence/final-2026-10-20/r25-launch.json); [health](evidence/final-2026-10-20/r25-health.json), [startup](evidence/final-2026-10-20/r25-startup.json),
  [remote source proof](evidence/final-2026-10-20/r25-remote-source.json).
  Không gọi train ổn định từ RUNNING.
- Manual action hiện không còn sau khi owner Cancel4kernel; phát sinh lỗi quyền/quota cần ghi exact bước.

## Kiểm live gần nhất

20/10 ~16:46Z (23:46 VN): zi/zk RUNNING, log đã xác thực W&B online, chưa traceback; đang chuẩn bị dữ liệu.
za/zc/ze/zg QUEUED chờGPU, không repush. Chưa có training heartbeat/round committed của r25,
**chưa đủ bằng chứng SGD ổn định**; phiên sau phải kiểm bước này trước khi coi startup hoàn tất.
Chuẩn bị dữ liệu đã đo30–45phút, absence W&B training run trong setup chưa là crash.

**20/10 18:23Z:** zg (50c gkd) COMPLETE, collect + audit PASS: 2/2 round committed (session cap), `evaluation_pending`. Timing: chuẩn bị dữ liệu 1.076 s, setup `fd.py` 124 s, round 1 1.032 s (gồm warm-up compile), round 2 810 s; cả session 52 phút. Cần chấm r1–r2 và nối tiếp train. 18:23Z: za ?, zc/ze/zi/zk RUNNING.

**20/10 18:44Z:** ze (50c q3) ERROR, collect PASS: chuẩn bị dữ liệu 4.420 s ⇒ budget 2h còn 0,77h < 1h, không chạy arm (0 round). Không lỗi code; cần repush với session dài hơn ([sổ lỗi](notebook-errors.md)).

**20/10 19:20Z — r25 kết thúc cả 6, collect + audit PASS cả 6:** za (20c q3) 4/4 round, zc (20c gkd) 4/4, zg (50c gkd) 2/2, zk (100c gkd) 2/2 — tất cả `evaluation_pending`. **ze, zi ERROR, 0 round**: hết budget session trong lúc chuẩn bị dữ liệu (4.420 s / 8.792 s) — [sổ lỗi](notebook-errors.md). Timing thực: chuẩn bị 1.019–8.792 s; setup `fd.py` 99–126 s; round 20c 775–943 s, 50c 810–1.032 s, 100c 1.300–1.515 s (round 1 gồm warm-up). Phương án tiếp: [final-pipeline §Lộ trình nhanh nhất](final-pipeline.md).

**20/10 ~20Z — owner duyệt lộ trình nhanh nhất** ([final-pipeline §Lộ trình](final-pipeline.md)): session ~11,5h không cap round,
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
  Chi tiết [final-pipeline §Bộ chấm nhanh](final-pipeline.md). Test: `test_fast_eval.py`, `test_final_pipeline.py` 20c/100c PASS.
- Đang chấm lại bằng bộ nhanh (bằng chứng tương đương + kết quả chính thức đồng nhất): `odixe0502/…-e01-04-s42za-…-f1`, `minhtran0601/…-e01-04-s42zc-…-f1`,
  `dinhan25/…-e01-02-s42zg-…-f1`, `trietbackup/…-e01-02-s42zk-…-f1`. Xong ⇒ `collect_eval.py <rid>` rồi `compare_eval.py <rid cũ -v1> <rid -f1>`.
- Push chấm: [launch_eval.sh](../../methods/full-drift/builds/fd-probe/scripts/launch_eval.sh) `<train rid> <a-b> <owner>` (= `gen_eval.py` + `upload_stage.py` nguồn r26 và checkpoint + push + `PUSHED`).

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
[Local proof](evidence/final-2026-10-20/final-recovery-local-2026-10-20.json), source-upload-verification.json từng notebook là names/sizes;
runtime SHA gate chứng minh source mount thực. Private notebookignored0600 mang inline W&B theo quyền đã có.

## Làm tiếp theo thứ tự

1. Kiểm launch ledger mới và PUSHED từng notebook, account-scoped status; không push trùng nếu reply mất.
   Source dataset phải PRIVATE, ready và verify đủ49file; quota live≥1.15×session cap, tối đa2session/account.
2. Đọc log SSE35s và exact W&B. Chứng minh auth/input/device/source/compile/Gaussian, heartbeat SGD tiến triển trên cả worker,
   không traceback/nonfinite/AMP lỗi. QUEUED chờGPU bình thường; setup dài chưa được gọi treo.
3. Terminal train ⇒ collect mọi weights/resume/round và executed notebook. Có0audits không phải successful training.
   Audit fingerprint/source/data/strict finite/hashchain; chọn round committed cuối.
4. Từng round một evaluator, account khác train; upload/verify source r25 và checkpoint bridge PRIVATE, push notebook private.
   Evaluator có advancing heartbeat ⇒ sinh/push continuation cùng source/schedule từ parent cuối đã audit; không chờ chấm xong.
5. Eval terminal ⇒ collect predictions→confusion→đủ10metrics, chạy report_final. Seed42 cả hai arm×ba kịch bản
   chỉ hoàn tất khi train r50 và eval đủr1–50. Officialtest không dùng chọn HP/round.

```bash
B=methods/full-drift/builds/fd-probe/scripts
conda run -n nckh python scripts/run_local_guarded.py python $B/collect_fd.py --only fd-s42zc-20261015-r25-v1 --jobs 16
conda run -n nckh python $B/gen_eval.py --session fd-s42zc-20261015-r25-v1 --round 1 --owner dinhan25
# Upload/verify private checkpoint stage + owner source r25, rồi push private_push/<eval-rid>.
conda run -n nckh python scripts/run_local_guarded.py python $B/collect_eval.py <eval-rid> --jobs 16
conda run -n nckh python $B/gen_continue.py --parent fd-s42zc-20261015-r25-v1 --owner dinhan25 --session zc02
# Upload/verify parent bridge + source, push sau evaluator có heartbeat tăng.
conda run -n nckh python $B/report_final.py
```

Quota reset24/10 00:00Z; batch đầu14session-hours, ước lượng200–300h toàn chiến dịch cũ chưa tính đúng prep30–45phút/notebook, cần đo/tính lại.
Ưu tiên phiên sau: tối ưu repeated preparation cho evaluator bằng fixture cache có provenance và equivalence, tag mới;
không sửa source r25 đã push. Chưa coi275h quota snapshot cũ đủ toàn chiến dịch.
Global login giữ `thuyanh316`; dùng `kaggle_as.py OWNER -- ...`, không đưa credentials vào output/docs.
Kaggle native MCP chưa cấu hình; authenticated CLI hoạt động. SSE có thểđọc live dù CLIlogs thường rỗng.

## Bàn giao lịch sử và giới hạn

[17–20/10 trước r25](../archive/operations/continuation-2026-10-17.md),
[r24 recovery](../archive/operations/final-recovery-2026-10-20.md).
`check_project_layout.py` còn mismatch DVC migration14/10; phiên này không đổi DVC. Không sửa DVC để làm check xanh.
Source/test local không thay remote2T4 proof; run status không thay scientific artifact audit.
