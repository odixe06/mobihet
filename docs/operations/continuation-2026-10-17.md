# Bàn giao 17/10/2026 (UTC; viết ~22:00Z 16/10)

Trước đó: [bàn giao 15–16/10](../archive/operations/continuation-2026-10-15-evening.md) (q4, vòng 1–7, lệnh cũ).

## Owner chốt (16/10 ~20:30–21:30Z)

- Kiểm thử mọi nhánh trong [kế hoạch 17/10](../archive/research/algorithm-improvement-plan-2026-10-17.md), dùng mọi slot quota
  còn lại; thứ tự không quan trọng. Frozen teacher theo plan (tắt KD private trước snapshot gộp + control warm-up).
  Cổng §7.1 (private gp Δ ≥ +0,010, benign ≥ −0,005, client tệ nhất ≥ −0,010), dùng lại control `fd_dml_m4b` s42 đã audit.
- Quota: 3 account lớn push ngay; account nhỏ chỉ push khi (giờ tới reset 00:00Z × số session) < GPU left. Phần sau
  reset ăn quota tuần mới. Theo dõi bằng poll nền (W&B + status Kaggle), pull/audit ngay khi xong.
- Tổ chức lại: bỏ symlink gốc `mh`/`papers` (giữ `.compat` ẩn + `.pth`), research mỗi phương pháp một file
  ([research/methods](../research/methods/README.md)), giữ cấp `builds/`, operations chỉ giữ file còn dùng
  (62 file → [archive](../archive/operations/)). Chi tiết: [project-layout](project-layout.md).

## Kết quả audited trong phiên

- Vòng 6 s43 (k) và vòng 7 s43 (m) AUDITED: m4b vs dml private +0,031/proxy +0,039, swappos +0,034/+0,031 —
  **PASS hết** ⇒ headline DML chuyển m4b. Gaussian-m4b s43 0,6568 (mô tả),
  EMA hại Gaussian-m4b cả hai seed. [Kết quả](../methods/full-drift/results.md).
- Vòng 8–10 (17/10 ~04:40Z, 15 session AUDITED, trừ `w`): **`gpool` PASS 3 cổng × 2 seed** (private gp 0,7174 / 0,7175;
  +0,016 / +0,024 so m4b; hơn `gown` +0,030, hơn `replay` +0,032; client tệ nhất không giảm; +4,2 MiB/6 epoch).
  Own controls PASS (DML +0,043/+0,040, Gaussian +0,051/+0,052). m4b_q4 s43 noninferior. Gaussian s43 m4b/swappos PASS.
  FAIL: kdcap (private −0,028/−0,025, proxy +0,024/+0,033), frz (gp 0,587/0,577; head 0,714/0,698), FedDyn (−0,032/−0,034),
  SCAFFOLD (LR* 0,1: −0,010 so sgd10, −0,007 so m4b), JS cons (+0,003). Rerun m4b bit-exact vòng 3.

## Kernel vòng 8–10 (FullDrift, source tag `20261015-r8`/`-r9`/`-r10`, cổng [design §Vòng 8–10](../methods/full-drift/design.md))

| kernel | GPU0 | GPU1 | push | xong (ước) |
|---|---|---|---|---|
| accanniez/mobihet-fd-s42n-20261015-r8-v1 (5,0h) | fd_dml_m4b_kdcap | fd_dml_m4b_kdcap_frz | 21:02Z | AUDITED ~04:40Z |
| accanniez/mobihet-fd-s42o-20261015-r8-v1 (5,0h) | fd_dml_m4b_kdcap_wu | fd_dml_m4b_dyn | 21:02Z | AUDITED ~04:40Z |
| annhtnt/mobihet-fd-s42p-20261015-r8-v1 (4,6h) | fd_dml_m4b_own | fd_gaussian_m4b_own | 21:02Z | AUDITED ~04:40Z |
| annhtnt/mobihet-fd-s42q-20261015-r8-v1 (8,5h) | fd_dml_swappos_cons | fd_dml_swappos_cons0 | 21:02Z | AUDITED ~04:40Z |
| takids1909/mobihet-fd-s43r-20261015-r8-v1 (5,4h) | fd_gaussian s43 | fd_gaussian_swappos s43 | 21:02Z | **AUDITED 00:05Z, PASS** |
| takids1909/mobihet-fd-s42s-20261015-r9-v1 (5,0h) | fd_dml_m4b_sgd3 | fd_dml_m4b_scaf3 | 21:42Z | AUDITED ~04:40Z |
| khanhmay0304/mobihet-fd-s42u-20261015-r9-v1 (5,0h) | fd_dml_m4b_sgd1 | fd_dml_m4b_scaf1 | 21:46Z | AUDITED ~04:40Z |
| minhtriethihi/mobihet-fd-s42v-20261015-r9-v1 (5,0h) | fd_dml_m4b_sgd10 | fd_dml_m4b_scaf10 | 21:57Z | AUDITED ~04:40Z |
| trietbackup/mobihet-fd-s42t-20261015-r9-v1 (5,4h) | fd_dml_m4b_gpool | fd_dml_m4b_gown | 22:07Z | AUDITED ~04:40Z |
| annhannh331/mobihet-fd-s43w-20261015-r9-v1 (10h) | fd_dml_m4b:15 s43 | fd_dml:15 s43 | 22:37Z | **AUDITED ~08:20Z, PASS** |
| ~~trietbackup/…-s42x-…-r9-v1~~ | thay bằng `y` (không push) | | | |
| anhtruong1909/mobihet-fd-s42y-20261015-r10-v1 (5,4h) | fd_dml_m4b_replay | fd_dml_m4b (rerun, log KD) | 22:20Z | AUDITED ~04:40Z |
| thuyanh316/mobihet-fulld-s42-20261017-q5-v1 (9,8h) | `h_dml` r41→r50 | — | 22:32Z (W&B resume r42 22:57Z) | **AUDITED ~08:20Z, r50** |
| nguyenminhtran060113/mobihet-fd-s43z-20261015-r10-v1 (5,0h) | fd_dml_m4b_own s43 | fd_gaussian_m4b_own s43 | 23:05Z | AUDITED ~04:40Z |
| catbaochau/mobihet-fd-s43ka-20261015-r10-v1 (5,0h) | fd_dml_m4b_kdcap s43 | fd_dml_m4b_kdcap_wu s43 | 23:10Z | AUDITED ~04:40Z |
| dinhan25/mobihet-fd-s43kb-20261015-r10-v1 (5,0h) | fd_dml_m4b_kdcap_frz s43 | fd_dml_m4b_q4 s43 | 23:21Z | AUDITED ~04:40Z |
| minhtran0601/mobihet-fd-s43ga-20261015-r10-v1 (5,4h) | fd_dml_m4b_gpool s43 | fd_dml_m4b_gown s43 | 23:32Z | AUDITED ~04:40Z |
| odixe0502/mobihet-fd-s43gb-20261015-r10-v1 (5,4h) | fd_dml_m4b_replay s43 | fd_dml_m4b_dyn s43 | 23:50Z | AUDITED ~04:40Z |

Dataset source: `<account>/mobihet-fulldrift-src-20261015-r8|r9`, 48 file names/sizes PASS; data datasets đọc được.
Test local trước push: arm cũ bit-exact so fd.py vòng 5/8 (2 epoch), kdcap(∞)=m4b, frz ep1 = wu ep1, đại số FedDyn và
SCAFFOLD client/server, JS, replay; smoke main() mọi arm mới; compiled backend (teacher, batch đôi, private+aux) PASS.

Poll nền của phiên (chết khi phiên Claude đóng): `watch.py` (W&B + collect tự động, audit qua lock) và `pusher.py`
(push hẹn giờ có kiểm quota) trong scratchpad. Nếu phiên đã đóng, làm tay:

```bash
B=methods/full-drift/builds/fd-probe/scripts
conda run -n nckh python scripts/run_local_guarded.py python $B/collect_fd.py --only 'fd-s4*-20261015-r[89]-v1' --jobs 16
conda run -n nckh python $B/report_fd.py
# q5 (nếu chưa PUSHED): kiểm quota thuyanh316 ≥ 10,1h rồi
H=.agents/skills/kaggle-training-notebook/scripts
conda run -n nckh python $H/kaggle_as.py thuyanh316 -- kaggle kernels push -p methods/shared-baselines/builds/federated-baselines-full/private_push/fulld-s42-20261017-q5-v1
conda run -n nckh python methods/shared-baselines/builds/federated-baselines-full/scripts/collect_full.py --only '*20261017-q5-v1' --once
```

Push hẹn giờ chưa xảy ra thì kiểm `notebooks/<rid>/PUSHED` trước; không push lại kernel đã có PUSHED.

## Theo dõi (owner 16/10 ~22:10Z: đảm bảo đã train, dừng nếu bất ổn)

~22:20Z: 18/18 run W&B đang train (vòng 8–9), Kaggle RUNNING. Không phân kỳ/NaN. Quan sát:
`kdcap_frz` r1 = `wu` r1 bit-exact; r2 private **head 0,584** (> wu 0,549) nhưng **gp 0,244** — teacher gộp mang prior
toàn cục, gp sửa quá tay (cổng gp đã khóa, báo head kèm; không phải train hỏng). `dyn` drift 0,11 (so 0,60), r2 private
0,601, proxy 0,546 (m4b r2: 0,648/0,718). kdcap: `w` median 2,6–3,4, cap_hit 0,95–1,00. Monitor nền `health.py`
(10 phút; heartbeat > 25 phút, loss không hữu hạn, skip ≥ 12, metric rơi > 0,15 từ r2) thoát khi có cảnh báo.
**Không dừng được kernel bằng CLI** (SDK có cancel-session nhưng cần session id không lộ qua API): nếu cần dừng, owner dừng
trên web Kaggle (Edit → Stop session / Cancel run).

~00:00Z 17/10: pusher xong mọi push hẹn giờ (q5, w, z, ka, kb, ga, gb); mỗi kernel kiểm ~8 phút sau push: Kaggle
RUNNING + W&B có bước train (q5 resume `h_dml_s42-c0f238e2` r42). 32 run FullDrift + q5, 0 cảnh báo. `r` AUDITED PASS.

## Việc tiếp (cập nhật 17/10 ~10:40Z)

Owner 17/10 ~05:10Z: headline client = **DML-m4b + `gpool`**; trước khi chạy đủ 50 round thử cải tiến kỹ thuật tốt nhất
(ghép/bổ sung); phương pháp FAIL chỉ chạy lại khi có chỉnh sửa. ~05:30Z: query quota mọi account (kể cả
annhtnt/takids1909/accanniez) — tuần mới 30h mỗi account, reset 24/10 00:00Z. **~10:10Z: mục tiêu = cải thiện phương pháp
huấn luyện để metrics tốt hơn; mọi kịch bản một lần gộp đồng bộ sau mỗi epoch local (như FL_NIDS_10client).**

`w` và q5 AUDITED ~08:20Z: horizon m4b s43 PASS (private gp +0,025, proxy +0,063); `h_dml` đủ r50 (test proxy 0,7994,
private gp 0,7363, phẳng từ r41). [FullDrift](../methods/full-drift/results.md) · [full H1/H2](../methods/shared-baselines/builds/federated-baselines-full/results/latest-rounds.md).

**Vòng 11** (tag `20261015-r11`, nguồn upload 48 file VERIFIED mỗi owner; [design §Vòng 11](../methods/full-drift/design.md)):

| kernel | GPU0 | GPU1 | push | RUNNING | xong (ước) |
|---|---|---|---|---|---|
| accanniez/mobihet-fd-s42la-20261015-r11-v1 (5,4h) | fd_dml_m4b_gpool_kd2 | fd_dml_m4b_gpool_kdpx | 08:23Z | ~09:50Z | ~14:10Z |
| takids1909/mobihet-fd-s42lc-20261015-r11-v1 (11h) | fd_dml_m4b_gpool:15 | fd_dml_m4b_gpool_gw2 | 08:23Z | ~09:57Z | ~20:10Z |

QUEUED ~1,5h sau push (không lỗi, không phải hết GPU); W&B có bước train ~10:05Z. Test local (`test_fd_local.py --part r11`):
arm cũ bit-exact so stage r10; kd2 (×1) = kdpx (cap ∞) = gw2 (×1) = gpool, khác ở giá trị thật; smoke `main()` 2 epoch PASS.
Upload nguồn: script tạm `upload_src.py` (scratchpad) — `kaggle datasets create -p <stage copy> -t`, chờ `ready`, so mọi trang
`datasets files` (tên/kích thước) với `checkpoint-dataset.json`, ghi `dataset-files-verified.json`.

Collect khi xong:
```bash
B=methods/full-drift/builds/fd-probe/scripts
conda run -n nckh python scripts/run_local_guarded.py python $B/collect_fd.py --only 'fd-s42la-20261015-r11-v1' --jobs 16
conda run -n nckh python scripts/run_local_guarded.py python $B/collect_fd.py --only 'fd-s42lc-20261015-r11-v1' --jobs 16
conda run -n nckh python scripts/run_local_guarded.py python $B/report_fd.py
```

Tiếp theo: đề xuất vòng 12 cải thiện metrics (bằng chứng → cơ chế → dấu hiệu sai), cùng kiểu một lần gộp/epoch.
Git/DVC: chưa commit thay đổi từ 16/10 (owner quyết).

## Cập nhật 17/10 ~14:30Z

Owner ~13–14Z: ưu tiên communication budget hạn chế + hỗ trợ đa mô hình + metrics private; tối ưu mã không đổi phương pháp
(2×T4); bỏ hướng giảm độ phức tạp model. **la AUDITED** (~14:00Z): `kd2` +0,001, `kdpx` −0,008 (proxy +0,031) FAIL.

Harness mới (source sau r11, mọi thay đổi bit-exact, test `checks`/`r8`/`r9`/`r11`/`r12`/`par` PASS): vòng lặp không đồng bộ
host (−20 %/bước trên RTX 3050), `QPacket` int4 thật, `--workers W` (rank 0 + W−1 tiến trình con, GPU xoay vòng theo rank,
IPC pickle thường). Notebook dạng pool: `gen_fd.py` plan `{'workers': W, 'arms': [...]}` (arm nối tiếp, mỗi arm dùng cả 2 GPU).

| kernel | arm | push | ghi chú |
|---|---|---|---|
| accanniez/mobihet-fd-bench-t4-20261017-v1 | benchmark v0/v1 × 1–3 tiến trình/GPU (không train thật) | 13:17Z | QUEUED lâu; kết quả `bench.json` chọn W |
| odixe0502/mobihet-fd-s42qa-20261015-r12-v1 (4h) | `fd_dml_m4b_gpool_q4`, `--workers 2` | 14:25Z | dataset r12 48 file VERIFIED; cổng design §Vòng 12 |
| takids1909/mobihet-fd-s42lc-20261015-r11-v1 | gpool:15 ‖ gw2 | 08:23Z | xong ~20:10Z |

```bash
B=methods/full-drift/builds/fd-probe/scripts
conda run -n nckh python scripts/run_local_guarded.py python $B/collect_fd.py --only 'fd-s42qa-20261015-r12-v1' --jobs 16
conda run -n nckh python scripts/run_local_guarded.py python $B/collect_fd.py --only 'fd-s42lc-20261015-r11-v1' --jobs 16
conda run -n nckh python scripts/run_local_guarded.py python $B/report_fd.py
H=.agents/skills/kaggle-training-notebook/scripts   # benchmark: kéo bench.json + log
conda run -n nckh python $H/kaggle_as.py accanniez -- kaggle kernels output accanniez/mobihet-fd-bench-t4-20261017-v1 -p <scratch>
```
Đề xuất chờ owner chọn (đã gửi 14:10Z): A dồn mẫu Gaussian vào lớp hiếm local; B nâng chất lượng mẫu (lọc theo proxy /
hỗn hợp moment); C luật gp theo prior huấn luyện thực (đổi luật khóa — cần duyệt). Mix đa kiến trúc (R2) chưa được chọn.

## Cập nhật 17/10 ~16:50Z — vòng 13

Owner ~16Z: chạy A, B, C + đề xuất owner (file ưu tiên đã tách: [research/methods](../research/methods/README.md),
[communication budget](../research/communication-budget.md); file gốc xóa). Owner chọn: arm train **từ đầu ep1→6**, P4 teacher-only,
q2 từ đầu. Cổng khóa trước push: [design §Vòng 13](../methods/full-drift/design.md). Kernel cũ: lc RUNNING ổn (W&B gw2 ep6 0,7125
chưa audit; gpool:15 ep10 0,7284), qa + benchmark vẫn QUEUED.

Source tag `20261015-r13` (test `--part r13` PASS: 10 arm cũ bit-exact so r12; rej/p1* epoch 1 == gpool; fill water-fill; QPacket
2-bit 109.512 B/chiều). Arm: `gpool_fill` (ma, annhannh331), `gpool_rej` (mb, anhtruong1909), `gpool_p1` (mc, catbaochau),
`gpool_p1self` (md, dinhan25), `gpool_p1cm` (me, khanhmay0304), `gpool_q2` (mf, minhtran0601); `--workers 2`, 5,0h.
Probe (không train): `gen_probe13.py` → `nguyenminhtran060113/mobihet-fd-probe13-20261015-r13-v1` (C gpe, P3 BN, P1 coverage,
P4 RFF ridge, P2 mode) → `probe13.json`.

```bash
B=methods/full-drift/builds/fd-probe/scripts
for s in ma mb mc md me mf; do conda run -n nckh python scripts/run_local_guarded.py python $B/collect_fd.py --only "fd-s42$s-20261015-r13-v1" --jobs 8; done
conda run -n nckh python scripts/run_local_guarded.py python $B/report_fd.py
```

**17:03Z probe COMPLETE (source khớp manifest, gp tái hiện audit):** C gpe FAIL (−0,094/−0,095 hai seed); P3 BN real-only
−0,032/−0,043 → backlog; P1 gate phủ 4,3 % mẫu; P2 hỗn hợp không tăng nhận mẫu → không mở P2a; P4 ridge M256 0,378 ≪ 0,717 → FAIL
teacher gate. [Kết quả](../methods/full-drift/results.md). Artifact `artifacts/kaggle/fd-probe13-20261015-r13-v1/`.
Vòng 13 train: mb–mf RUNNING từ 16:54Z (`--workers 2`: ~46 s/2048 bước/worker), ma QUEUED.
18:48Z: ma v1 bị owner cancel nhầm khi còn QUEUED (chưa chạy, không có run W&B) → push lại **version 2** cùng kernel id/dataset
(collector kéo version mới nhất). 17:13Z benchmark T4 xong → chọn `--workers 2` ([computational cost §5](../research/computational-cost.md)).
17:57Z qa RUNNING.
18:59Z lc COMPLETE → AUDITED (~19:05Z): gpool:15 horizon FAIL (+0,0019 vs m4b ep15), gw2 FAIL (−0,0049, tệ nhất −0,0104).
report_fd.py: sửa guard `gpe` cho arm không có role private (FedAvg).

19:02–19:28Z md/me/mf/mb COMPLETE → AUDITED: **rej PASS** (+0,0108, tệ nhất +0,012), q2 noninferior (private +0,0001; proxy −0,124),
p1self/p1cm ≈ gpool. **Vòng 14** ([design §Vòng 14](../methods/full-drift/design.md); source = r13): `trietbackup/…-s42na-…-r14-v1`
(rej:15), `minhtriethihi/…-s43nb-…-r14-v1` (rej s43), `accanniez/…-s42nc-…-r14-v1` (q2:15), push ~19:45Z.

## Cập nhật 17/10 ~20:30Z — chẩn đoán private chững (probe 14–15)

Owner ~19:55–20:20Z: phân tích thật kỹ vì sao private không cải thiện, bằng phép thử thực tế; giữ nguyên dataset Kaggle đã xử lý
(chỉ đọc, đối chiếu `fl_dagsnet_veremi`), mọi client DAGSNet, không đề xuất đổi mô hình; đề xuất sau phải giữ R1 (byte ít, ngắn hạn)
và R2 (client khác kiến trúc, không chạy thử). Khi chắc giới hạn headline thì đề xuất phương pháp mới từ kết quả phân tích.

19:55Z mc (`gpool_p1`) AUDITED: FAIL (+0,0001). 20:21Z qa (`gpool_q4`) AUDITED: noninferior PASS (−0,0001, proxy +0,004).
Quota tuần mới (reset 24/10): ~25–30h/account.

| kernel | nội dung | push |
|---|---|---|
| `nguyenminhtran060113/mobihet-fd-probe14-20261015-p14-v1` | không train: ma trận chéo client, khoảng theo số mẫu local, refit head trên backbone đóng băng, bias oracle, dòng trùng khác nhãn + kNN own/gộp ([design §Probe 14](../methods/full-drift/design.md)) | 20:14Z |
| `nguyenminhtran060113/mobihet-fd-probe15-20261015-p15-v1` | 1 epoch có kiểm soát × 8 mục tiêu × 20 client từ m4b ep15 (quên do mục tiêu local vs thiếu phủ input; [design §Probe 15](../methods/full-drift/design.md)), ~3h | 20:21Z |

Dataset nguồn: `nguyenminhtran060113/mobihet-fd-probe14-src-20261015-p14` (53 file), `…-probe15-src-20261015-p15` (52 file), size khớp.
Kéo kết quả: `kaggle_as.py nguyenminhtran060113 -- kaggle kernels output <id> -p methods/full-drift/builds/fd-probe/artifacts/kaggle/<rid>`.
Phân tích local đã có: [results §Chẩn đoán](../methods/full-drift/results.md).

**~21:55Z:** probe14 AUDITED (20:40Z) → nút thắt = tầng quyết định (refit Linear gộp 0,84 vs gp 0,73). probe16 AUDITED (21:35Z) → backbone
không FL cũng đủ; nhãn proxy trên input thật 0,831 ≈ nhãn thật; Gaussian + nhãn proxy 0,767. Push thêm
`nguyenminhtran060113/mobihet-fd-probe17-20261015-p17-v1` (21:57Z; mẫu Gaussian tinh chỉnh bằng proxy CE/BN cho tầng quyết định; CCVR sửa;
[design §Probe 17](../methods/full-drift/design.md)). Dataset `…-probe16-src-…-p16` (57 file), `…-probe17-src-…-p17` (55 file), size khớp.
probe15 RUNNING (~3h từ 20:21Z). Vòng 13/14 (ma, na, nb, nc) RUNNING ổn trên W&B (21:23Z).
- 22:40Z ma (`gpool_fill`, v2) AUDITED: FAIL (−0,0012). collect_fd: `PUSH_VERSION` lấy executed notebook v2 của ma.
- 23:10Z nb (`gpool_rej` s43) AUDITED: +0,0082 (benign +0,005, tệ nhất +0,018) ⇒ FAIL cổng gain đã khóa (+0,010), cùng chiều s42.

## Cập nhật 18/10 ~00:00Z — kết luận chẩn đoán + vòng 15

probe15 (23:21Z) + probe17 (23:42Z) AUDITED — [results §Chẩn đoán](../methods/full-drift/results.md). Nguyên nhân cốt lõi: backbone đủ;
tầng quyết định private trôi về local trong epoch dài vì teacher DML (bản proxy) cũng trôi; teacher gộp cố định giữ được (+0,043 trong
1 epoch); phần còn lại là thiếu input phủ client khác (oracle dòng thật → 0,842 ≈ proxy). nb (rej s43) FAIL cổng (+0,0082).

Vòng 15 ([design §Vòng 15](../methods/full-drift/design.md), source tag `20261015-r15`, test r15 + par PASS), push 23:45Z, 7,0h:
`thuyanh316/mobihet-fd-s42oa-20261015-r15-v1` (`fd_dml_m4b_frz:15`), `annhtnt/…-s42ob-…` (`fd_dml_m4b_gpool_frz:15`),
`khanhmay0304/…-s42oc-…` (`fd_dml_m4b_gpool_frz_gkd:15`). Dataset `<acct>/mobihet-fulldrift-src-20261015-r15` (48 file, size khớp).
Cổng arm frz đọc luật **head** — cần owner xác nhận trước khi nhận làm headline.

```bash
B=methods/full-drift/builds/fd-probe/scripts
for s in oa ob oc; do conda run -n nckh python scripts/run_local_guarded.py python $B/collect_fd.py --only "fd-s42$s-20261015-r15-v1" --jobs 16; done
conda run -n nckh python scripts/run_local_guarded.py python $B/report_fd.py   # gates '...:private_head_vs_..._gp@ep15'
```
- 01:10Z na (`gpool_rej:15`) AUDITED: ep15 0,7455 (+0,0113 vs gpool@e15, +0,0132 vs m4b@e15), test 0,7599 ⇒ horizon PASS; lợi không biến mất như gpool.
- 01:15Z nc (`gpool_q2:15`) AUDITED: private ep15 0,7334 (−0,0008 vs gpool@e15) noninferior PASS; proxy 0,725 (−0,118).
- 03:45Z oa (`m4b_frz:15`) AUDITED: head ep15 0,7690 (+0,0367 vs m4b@e15 gp; benign +0,166; tệ nhất +0,045), test 0,7728 ⇒ PASS; proxy −0,023.
- 04:20Z ob (`gpool_frz:15`) AUDITED: head ep15 0,7717 (+0,0376 vs gpool@e15 gp; benign +0,179; tệ nhất +0,092), test 0,7795 ⇒ PASS; proxy −0,028.
- 04:45Z oc (`gpool_frz_gkd:15`) AUDITED: head ep15 **0,8030** (+0,0688 vs gpool@e15 gp; +0,0312 vs gpool_frz), test 0,8046, tệ nhất +0,128 ⇒ PASS cả hai cổng; proxy −0,011.

## Cập nhật 18/10 ~05:05Z — vòng 15 PASS, vòng 16 + probe 18

Vòng 15 AUDITED (oa 03:45Z, ob 04:20Z, oc 04:45Z): PASS cả ba ở ep15 s42 (luật head). Headline ứng viên `fd_dml_m4b_gpool_frz_gkd`
(0,8030; test 0,8046). Luật head cho arm frz + nhận headline **chờ owner**.

Vòng 16 ([design §Vòng 16](../methods/full-drift/design.md), source `20261015-r16` = r15 + 1 dòng ARMS), push 05:01Z, 7,0h:
`anhtruong1909/mobihet-fd-s43pa-20261015-r16-v1` (gkd:15 s43), `dinhan25/…-s43pb-…` (gpool:15 s43, control), `catbaochau/…-s42pc-…`
(q2+frz+gkd:15, R1), `minhtran0601/…-s43pd-…` (frz:15 s43). Dataset `<acct>/mobihet-fulldrift-src-20261015-r16` (48 file, size khớp).
Probe 18 (`nguyenminhtran060113/mobihet-fd-probe18-20261015-p18-v1`, 05:04Z): đo probe 14 trên weights vòng 15.

```bash
B=methods/full-drift/builds/fd-probe/scripts
for s in 43pa 43pb 42pc 43pd; do conda run -n nckh python scripts/run_local_guarded.py python $B/collect_fd.py --only "fd-s$s-20261015-r16-v1" --jobs 16; done
conda run -n nckh python scripts/run_local_guarded.py python $B/report_fd.py
```
Probe 19 (`nguyenminhtran060113/mobihet-fd-probe19-20261015-p19-v1`, 05:28Z): bias 16 lớp ước federated ([design §Probe 19](../methods/full-drift/design.md)). Probe 18 AUDITED 05:25Z.
- 05:35Z probe 19 AUDITED: bias federated (CE/bCE, chuyển từ proxy) không giúp private (≤ +0,02) ⇒ FAIL hợp đồng; lợi +0,045 cần F1 oracle riêng từng private.

## Cập nhật 18/10 ~06:30Z — owner: luật chính xác hơn làm luật chính; cải tiến nhỏ gkd rồi 50 round

Owner ~06Z: so head vs gp, luật nào chính xác hơn dùng làm luật chính (mục tiêu = model client); gkd ổn thì thử cải tiến nhỏ, chắc chắn
là cách tốt nhất thì chạy 50 round. **Luật:** họ frz đọc head (hơn gp mọi góc: global, local F1/acc, client tệ nhất), arm DML cũ giữ gp
([design §Luật](../methods/full-drift/design.md)). Rò prior local còn lại trong head gkd: độ dốc +0,13 (`scripts/prior_leak.py`).

Vòng 17 ([design §Vòng 17](../methods/full-drift/design.md), source tag `20261015-r17` = r16 + token `la` + 3 arm + `--parent`; test
`--part r17`, `--part parent`, `--part par` (la_kdpx compiled) PASS). Dataset nguồn `<acct>/mobihet-fulldrift-src-20261015-r17` (48 file VERIFIED);
parent `trietbackup/mobihet-fd-parent-gkd-s42-e15` (config + marker + weights ep15 của oc, phẳng; notebook dựng lại thư mục, fd.py kiểm SHA).

| kernel | arm | push | ghi chú |
|---|---|---|---|
| odixe0502/mobihet-fd-s42ra-20261015-r17-v1 (7h) | `fd_dml_m4b_gpool_frz_gkd_la:15` | 06:31Z | cổng vs gkd@e15 head 0,8030 |
| annhannh331/mobihet-fd-s42rb-20261015-r17-v1 (7h) | `fd_dml_m4b_gpool_frz_gkd_kdpx:15` | 06:31Z | cổng + proxy phải tăng ≥ 0,010 |
| minhtriethihi/mobihet-fd-s42rc-20261015-r17-v1 (7h) | `fd_dml_m4b_gpool_frz_gkd_la_kdpx:15` | 06:31Z | tổ hợp |
| trietbackup/mobihet-fd-s42rd-20261015-r17-v1 (11,75h) | `fd_dml_m4b_gpool_frz_gkd:50` từ ep15 oc | **chưa push** — chờ pa (gkd s43) PASS | ~35 epoch × 18 min |

```bash
B=methods/full-drift/builds/fd-probe/scripts; H=.agents/skills/kaggle-training-notebook/scripts/kaggle_as.py
# rd sau khi pa PASS:
conda run -n nckh python $H trietbackup -- kaggle kernels push -p methods/full-drift/builds/fd-probe/private_push/fd-s42rd-20261015-r17-v1
for s in ra rb rc rd; do conda run -n nckh python scripts/run_local_guarded.py python $B/collect_fd.py --only "fd-s42$s-20261015-r17-v1" --jobs 16; done
conda run -n nckh python scripts/run_local_guarded.py python $B/report_fd.py   # gates '..._gkd_<t>@e15:private_head_vs_..._gkd@e15_head@ep15'
```
Arm vòng 17 PASS → nối tiếp ep16→50 từ ep15 của chính nó: thêm plan `parent` (session của nó) trong `gen_fd.py`, upload parent bằng
`$B/upload_parent.py` (dataset phẳng config/marker/weights) rồi push; collector kiểm chuỗi parent với session đã AUDITED.

## Cập nhật 18/10 ~08:00Z — kiểm toán FL + vòng 18

- Owner ~06:50Z hỏi phương pháp có đúng FL: [kiểm toán luồng thông tin](../research/federated-information-flow.md) — huấn luyện federated; SecureSum
  chỉ giả lập (ô client×lớp 4–10 dòng ⇒ cần SecAgg thật), q4/q2 không cộng được dưới SecAgg, scaler tập trung, holdout gộp. Sửa (b) `la` prior từ
  SecureSum và (c) compile probe dùng dòng của chính tiến trình đã vào source r18; (a) ngưỡng ô/DP và (d) scale chung chờ owner.
- Owner ~07:00Z: quota thoải mái, thử mọi phương án có bằng chứng ⇒ vòng 18 ([design §Vòng 18](../methods/full-drift/design.md)): `cop` (Gaussian copula,
  kiểm local: teacher đồng ý 0,673 vs 0,473), `pce`, `t2`, `g512`, `gk2`, `q4_frz_gkd`. Test `--part r18` + `parent` PASS.
- rc (`minhtriethihi/…-s42rc-…-r17-v1`) RUNNING từ 06:32Z, run W&B chỉ xuất hiện sau ~1,5h (khởi động chậm, **không kẹt**); ~08:25Z đã xong ep1.
- W&B (chưa audit) ~07:50Z: la ep4 0,7805 vs gkd 0,7463; kdpx ep4 0,749; pa (gkd s43) ep9 0,788; q2_frz_gkd ep9 0,673 (tụt mạnh — frz phụ thuộc snapshot).

Vòng 18 push 08:03–08:04Z (source tag `20261015-r18`, dataset `<acct>/mobihet-fulldrift-src-20261015-r18` 48 file VERIFIED; test `--part r18`,
`parent`, `par` cop compiled workers 2 PASS):

| kernel | arm (s42, ep1→15) |
|---|---|
| accanniez/mobihet-fd-s42sa-20261015-r18-v1 (7h) | `fd_dml_m4b_gpool_frz_gkd_cop` |
| takids1909/mobihet-fd-s42sb-20261015-r18-v1 (7h) | `fd_dml_m4b_gpool_frz_gkd_pce` |
| khanhmay0304/mobihet-fd-s42sc-20261015-r18-v1 (7h) | `fd_dml_m4b_gpool_frz_gkd_t2` |
| nguyenminhtran060113/mobihet-fd-s42sd-20261015-r18-v1 (8,5h) | `fd_dml_m4b_gpool_frz_gkd_g512` |
| annhtnt/mobihet-fd-s42se-20261015-r18-v1 (7h) | `fd_dml_m4b_gpool_frz_gkd_gk2` |
| thuyanh316/mobihet-fd-s42sf-20261015-r18-v1 (7h) | `fd_dml_m4b_gpool_q4_frz_gkd` |

```bash
B=methods/full-drift/builds/fd-probe/scripts
for s in sa sb sc sd se sf; do conda run -n nckh python scripts/run_local_guarded.py python $B/collect_fd.py --only "fd-s42$s-20261015-r18-v1" --jobs 16; done
conda run -n nckh python scripts/run_local_guarded.py python $B/report_fd.py
```

## Trạng thái gộp 18/10 ~09:00Z (để compact) — đọc mục này trước

**Owner 18/10 (theo thứ tự):** ~06Z luật chính = luật chính xác hơn ⇒ họ frz đọc **head**, DML cũ giữ gp (design §Luật); gkd ổn ⇒ thử cải tiến nhỏ
rồi chạy 50 round. ~06:50Z kiểm có đúng FL ⇒ [kiểm toán](../research/federated-information-flow.md); sửa (b) `la` prior SecureSum + (c) compile probe
dữ liệu tiến trình đã vào r18; **(a) ngưỡng ô/DP, (d) scale chung q4/q2 chờ owner**. ~07Z quota thoải mái ⇒ thử mọi cải tiến có bằng chứng (vòng 18).
~08:30Z tối ưu code: skill mới `federated-training-speed` (portable, `.claude` chính + `.agents` Codex), chuẩn hóa mô tả mọi skill, bảng định tuyến
skill trong CLAUDE.md/AGENTS.md §7; được phép benchmark để tìm cách viết code nhanh nhất. **Câu hỏi tầng 2 (không bit-exact, ~1,2–1,8×) chưa
trả lời đủ** — xem "Tối ưu code" dưới.

### Kernel (status 09:00Z; poll `scratchpad/poll.sh` qua Monitor, hết hạn 30 phút thì bật lại)

| vòng | session | arm | trạng thái / việc |
|---|---|---|---|
| 16 | pa anhtruong1909 s43 | gkd:15 | **AUDITED ~09:39Z: PASS** head 0,7953, +0,0586 vs gpool s43, +0,0690 vs m4b s43 |
| 16 | pb dinhan25 s43 | gpool:15 (control) | **AUDITED ~09:40Z**: gp ep15 0,7367 |
| 16 | pc catbaochau s42 | q2_frz_gkd:15 | **AUDITED ~09:58Z: FAIL** head 0,5773 (−0,226), proxy 0,479 |
| 16 | pd minhtran0601 s43 | frz:15 | **AUDITED ~09:40Z: PASS** head 0,7631, +0,0369 vs m4b s43 gp (benign +0,172, tệ nhất +0,044) |
| 17 | ra odixe0502 | gkd_la:15 | **AUDITED ~11:25Z: PASS** +0,0160 (benign +0,070, tệ nhất +0,037); rò prior +0,133 → +0,030 |
| 17 | rb annhannh331 | gkd_kdpx:15 | **AUDITED ~11:10Z: FAIL** +0,0058 (proxy +0,016) |
| 17 | rc minhtriethihi | gkd_la_kdpx:15 | **AUDITED ~12:55Z: PASS** +0,0195 (0,8225; proxy +0,0105) |
| 18 | sa accanniez | gkd_cop:15 | **AUDITED ~13:00Z: FAIL** +0,0031 (syn_accept 0,48 → 0,67) |
| 18 | sc khanhmay0304 / se annhtnt | gkd_t2 / gkd_gk2 :15 | **AUDITED ~14:20Z: FAIL** −0,0058 / −0,0002 |
| 18 | sf thuyanh316 | q4_frz_gkd:15 | **AUDITED ~14:20Z: PASS** noninferior +0,0013, byte proxy ÷7,6 |
| 18 | sb takids1909 | gkd_pce:15 | **AUDITED ~14:25Z: FAIL** +0,0047, benign −0,053; proxy 0,8917 (+0,059) |
| 18 | sd nguyenminhtran060113 | gkd_g512:15 | **AUDITED ~15:40Z: FAIL** +0,0014 |
| 17 | ra odixe0502 / rb annhannh331 / rc minhtriethihi | gkd_la / gkd_kdpx / gkd_la_kdpx :15 | RUNNING (rc khởi động chậm, không kẹt); W&B ~09:05Z la ep7 +0,010, kdpx ep8 +0,007 (proxy +0,017) |
| 17 | rd trietbackup | gkd:50 nối tiếp ep15 oc | **push 09:39Z** (code r17, 11,75h) |
| 18 | sa accanniez … sf thuyanh316 | cop / pce / t2 / g512 / gk2 / q4_frz_gkd :15 | push 08:03Z, RUNNING 08:05–08:28Z; pce/g512/q4 chưa thấy run W&B lúc 09:05Z (kiểm lại); cop ep2 +0,009 |

Collect từng session (một tiến trình mỗi lần, RAM): `for s in 43pd 43pa 43pb 42pc; do ... collect_fd.py --only "fd-s$s-20261015-r16-v1"; done`,
r17 `fd-s42{ra,rb,rc,rd}-20261015-r17-v1`, r18 `fd-s42{sa..sf}-20261015-r18-v1`; rồi `report_fd.py` (cổng R16 + vòng 17/18 trong `R16`).
Sau collect: cập nhật results.md (§Vòng 16/17/18), LESSONS, CONTEXT. Arm vòng 17/18 PASS → vòng 19 tổ hợp (vd la + kdpx + cop) rồi nối tiếp ep16→50
từ ep15 của chính nó (`--parent`, `upload_parent.py`) + seed 43.

### Tối ưu code (skill `federated-training-speed`)

- Harness hiện (deterministic, modules compile reduce-overhead, sync-free, `--workers 2`): gkd 26,3 ms/bước/tiến trình trên T4 (`scripts/ms_per_step.py`).
- Dò đồng bộ (`torch.cuda.set_sync_debug_mode`): trong bước chỉ còn **1 sync/bước** (`GradScaler.step` → `found_inf.item()`); còn lại là cuối epoch.
- Benchmark local bước gkd thật (RTX 3050, 1 client fixture, `scratchpad/bench/bench_step.py`): base 30,4 ms; `fused` AdamW 24,9 (−18 %, bỏ sync
  scaler); `nondet` (cudnn.benchmark, bỏ deterministic) 27,2 (−11 %); `fused_nondet` **22,9 (−25 %)**. Đều thuộc **lớp tolerance** (không bit-exact) ⇒
  cần owner duyệt + chạy lại đối chứng gkd trong harness mới. Bước tiếp: bench T4 (kernel không train) cho base/fused/nondet/fused_nondet ±
  gộp 2 forward teacher (kiểm bit-exact local trước) ± capture cả bước.
- Trả lời owner (đã soạn): lợi lâu dài có (mọi run sau), chi phí một lần ≈ cài + test + 1 run đối chứng; làm sau được nhưng mỗi lần đổi harness phải
  chạy lại đối chứng; không đổi model/client/dữ liệu/loss/byte, chỉ đổi thứ tự làm tròn.

### Skill (18/10)

`.claude/skills` là bản chính (Claude Code), `.agents/skills` bản Codex (+ `agents/openai.yaml`), khác nhau chỉ ở đường dẫn/host. Mới:
`federated-training-speed` (perf-federated/perf-2xT4/local-gpu-probe chuyển từ kaggle skill; two-worker-harness, equivalence-tests, ms_per_step.py).
kaggle skill: template trỏ GitHub `odixe06/fl_dagsnet_veremi` (lwfednids/, perfedskd/); bổ sung vào `.claude` các mục chỉ `.agents` có (kaggle_as
concurrent, Range resume, test_pull_kernel_output.py). Mô tả mọi skill có cụm kích hoạt tiếng Việt.

### Quyết định owner 18/10 ~09:15Z

- **Duyệt lớp tolerance** cho kernel build từ vòng 19: `fd.py --fast` (fused AdamW + cuDNN autotune, bỏ deterministic). Kernel đã chạy/đã
  build (vòng 16–18, rd) giữ code cũ và đối chứng cũ; arm `--fast` chỉ so với đối chứng `--fast` — [design §Code nhanh](../methods/full-drift/design.md).
  Code: cờ `--fast` (config `fast`, W&B `-fast`), `gen_fd.py` plan dict `'fast': True`, `report_fd.py` khóa `@fast`, test
  `test_fd_local.py --part fast` **PASS** (không cờ = bit-exact r18; có cờ: 1 bước logit lệch ≤ 6e-3, 0 argmax đổi trên dòng phân định rõ,
  cùng applied/skipped; compiled 2 worker chạy hết). `rg` đã cài vào env nckh, `docs/file-index.md` đã tạo lại.
- **Bảo mật FL**: chỉ ghi lại ([luồng thông tin §7](../research/federated-information-flow.md)), bổ sung khi cần; mục tiêu chính là performance.

### Vòng 19 push 14:45Z (`--fast`, source tag `20261015-r19`, dataset `<acct>/mobihet-fulldrift-src-20261015-r19` 48 file VERIFIED)

Test: `--part fast` PASS, smoke + `par` (compiled, 1 == 2 worker) arm mới `q4_frz_gkd_la_kdpx` PASS. [Design §Vòng 19](../methods/full-drift/design.md).

| kernel (7h) | seed | arm ep1→15 `--fast` | cổng (report key `@fast`) |
|---|---|---|---|
| minhtran0601/mobihet-fd-s42ta-20261015-r19-v1 | 42 | `fd_dml_m4b_gpool_frz_gkd` | equiv vs gkd@e15 cũ (≤ 0,010) |
| dinhan25/mobihet-fd-s42tb-20261015-r19-v1 | 42 | `fd_dml_m4b_gpool_frz_gkd_la_kdpx` | gain+worst vs ta — **AUDITED ~18:35Z** 0,8269, 18,75 ms/bước (−25,6 %) |
| annhannh331/mobihet-fd-s42tc-20261015-r19-v1 | 42 | `fd_dml_m4b_gpool_q4_frz_gkd_la_kdpx` | noninferior vs tb |
| anhtruong1909/mobihet-fd-s43td-20261015-r19-v1 | 43 | `fd_dml_m4b_gpool_frz_gkd` | equiv vs gkd@e15@s43 cũ — **AUDITED ~18:40Z: PASS** Δ +0,0031, 19,34 ms/bước |
| odixe0502/mobihet-fd-s43te-20261015-r19-v1 | 43 | `fd_dml_m4b_gpool_frz_gkd_la_kdpx` | gain+worst vs td |

```bash
for s in 42ta 42tb 42tc 43td 43te; do conda run -n nckh python scripts/run_local_guarded.py python $B/collect_fd.py --only "fd-s$s-20261015-r19-v1" --jobs 16; done
conda run -n nckh python scripts/run_local_guarded.py python $B/report_fd.py
```
Sau vòng: arm thắng nối tiếp ep16→50 (`--parent`, `upload_parent.py`) kèm gkd `--fast` ep50; q4 chưa nối tiếp được (`load_parent` chặn codec).

### Nối tiếp ep16→50 `--fast` (chuẩn bị ~19:30Z 18/10, CHƯA push — chờ cổng ta/te)

`gen_fd.py --round r19 --sessions <list>` (cùng tag r19, nguồn sẵn trên account). Test `--part parent` PASS (gồm: fast→fast nối tiếp chạy hết,
non-fast nối tiếp parent fast bị từ chối).

| session | kernel (11,75h) | arm `:50` từ ep15 | parent dataset | trạng thái |
|---|---|---|---|---|
| ua | dinhan25/mobihet-fd-s42ua-20261015-r19-v1 | la_kdpx s42 (tb) | `dinhan25/mobihet-fd-parent-lakdpx-fast-s42-e15` VERIFIED | notebook sinh, chờ push |
| ub | anhtruong1909/mobihet-fd-s43ub-20261015-r19-v1 | gkd s43 (td) | `anhtruong1909/mobihet-fd-parent-gkd-fast-s43-e15` VERIFIED | notebook sinh, chờ push |
| uc | minhtran0601/…-s42uc-… | gkd s42 (ta) | upload sau khi ta audit | chưa sinh |
| ud | odixe0502/…-s43ud-… | la_kdpx s43 (te) | upload sau khi te audit | chưa sinh |

Push khi tb (vs ta) và te (vs td) PASS gain+worst: `kaggle_as.py <owner> -- kaggle kernels push -p methods/full-drift/builds/fd-probe/private_push/fd-<s>-20261015-r19-v1`.
Parent uc/ud: `upload_parent.py <owner>/mobihet-fd-parent-<gkd|lakdpx>-fast-s<seed>-e15 <run dir> 15 <scratch>`.
