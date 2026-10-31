# FullDrift — drift và kỹ thuật FL trên full data

Owner duyệt 15/10 tối: kiểm thẳng trên full (holdout nội bộ), một seed chung 42, epoch 1–6 lịch T50.
Owner 17/10: mọi arm một lần gộp đồng bộ sau mỗi epoch local (như FL_NIDS_10client).
**Kết quả audited:** [results.md](../../methods/full-drift/results.md) — học thuộc receiver là nút thắt (m4b), `gpool` là headline client; Gaussian swappos +0,052.
[Thiết kế/cổng khóa](../../methods/full-drift/design.md) · [Chẩn đoán gốc](../research/plateau-diagnosis-2026-10-15.md)

| kernel | GPU0 | GPU1 |
|---|---|---|
| annhannh331/mobihet-fd-s42a-20261015-v2 | fd_dml | — |
| annhannh331/mobihet-fd-s42b-20261015-v2 | fd_dml_prox | fd_gaussian → fd_gaussian_swappos |
| anhtruong1909/mobihet-fd-s42c-20261015-v2 | fd_fedavg | fd_dml_avgm |

Preflight local (pilot fixture 1024 dòng/client): `fd_dml`/`fd_gaussian` một epoch **bit-exact** epoch `full_run`
đã freeze (4032/3840 tensor); smoke 2 epoch mọi arm rc0; FedAvg compiled
preflight PASS. Ước ~4–6h/session (DML ≈34 phút train/epoch, eval holdout nhỏ, test một lần cuối).
W&B project `mobihet-fulldrift`. Source/notebook/artifact: [build](../../../methods/full-drift/builds/fd-probe).

Vòng 2 PUSHED 16/10 02:19Z (stage/dataset `…-20261015-r2`, contract [design §Vòng 2](../../methods/full-drift/design.md)):

| kernel | GPU0 | GPU1 |
|---|---|---|
| anhtruong1909/mobihet-fd-s42d-20261015-r2-v1 (9,5h) | fd_dml ep15 | — |
| anhtruong1909/mobihet-fd-s42e-20261015-r2-v1 (9h) | fd_fedavg ep15 | — |

Preflight vòng 2: checks (bit-exact giữ nguyên) và smoke rc0, đỉnh RAM 2,4GB mỗi lượt; controller queue `arm:15` thử bằng
worker giả PASS; dataset 48 names/sizes PASS.

Vòng 3 (owner 16/10: mục tiêu client, private gp primary, R1/R2) PUSHED 16/10 04:17Z, stage/dataset
`nguyenminhtran060113/mobihet-fulldrift-src-20261015-r3`, contract [design §Vòng 3](../../methods/full-drift/design.md):

| kernel | GPU0 | GPU1 |
|---|---|---|
| nguyenminhtran060113/mobihet-fd-s42g-20261015-r3-v1 (4,5h) | fd_dml_m4b | fd_dml_swappos |
| nguyenminhtran060113/mobihet-fd-s42h-20261015-r3-v1 (4,5h) | fd_dml_ntd | fd_dml_q4 |

Preflight vòng 3: checks (bit-exact giữ, NTD KL bỏ qua logit lớp đúng của thầy = KL 15 lớp, m4b/swappos đặt 0
13/4 feature, finite) + smoke 18/18 rc0, đỉnh 2,5GB; dataset 48 names/sizes PASS bằng account chạy.
`report_fd.py`: cổng private gp (vòng 3 primary) + bảng góc nhìn client (local/client tệ nhất).

Vòng 4 PUSHED 16/10 06:32Z: `odixe0502/mobihet-fd-s42i-20261015-r4-v1` (3,95h, quota 4,01h) GPU0 `fd_dml_hkd`,
GPU1 `fd_gaussian_m4b`; dataset `odixe0502/mobihet-fulldrift-src-20261015-r4` 48 file PASS; checks + smoke PASS
(hkd giữ proxy y hệt đối chứng, private khác). [Cổng §Vòng 4](../../methods/full-drift/design.md). Collect: `collect_fd.py --only 'fd-s42i-*'`.

Vòng 3 **AUDITED** (m4b/swappos PASS, ntd FAIL, q4 noninferior): [kết quả](../../methods/full-drift/results.md).
Vòng 5 PUSHED 16/10 08:26Z: `nguyenminhtran060113/mobihet-fd-s42j-20261015-r5-v1` (8,95h) GPU0 `fd_dml_m4b:15`,
GPU1 `fd_dml_m4b_q4`; dataset r5 48 file PASS; checks + smoke PASS. [Cổng §Vòng 5](../../methods/full-drift/design.md).
Collector: pull song song (`--only`, `--jobs`), audit tuần tự qua lock (~4GB/audit).

Vòng 2, 4, 5 **AUDITED** 16/10 ~17:00Z ([kết quả](../../methods/full-drift/results.md)): prox FAIL; hkd FAIL, Gaussian m4b PASS; m4b@e15 PASS
(proxy +0,086), m4b_q4 noninferior.
Vòng 6 (seed 43, stage/dataset `anhtruong1909/mobihet-fulldrift-src-20261015-r6` 48 file PASS, bytes = r5) PUSHED 16/10 16:52Z,
[cổng §Vòng 6](../../methods/full-drift/design.md); smoke local vốn chạy `--seed 43`. Report/collector nhận seed (`<arm>@s43`).

| kernel | GPU0 | GPU1 |
|---|---|---|
| anhtruong1909/mobihet-fd-s43k-20261015-r6-v1 (4,2h) | fd_dml | fd_dml_m4b |
| annhannh331/mobihet-fd-s43m-20261015-r7-v1 (4,2h, vòng 7, PUSHED 17:32Z) | fd_dml_swappos | fd_gaussian_m4b |

Vòng 7 dataset `annhannh331/mobihet-fulldrift-src-20261015-r7` 48 file PASS ([cổng §Vòng 7](../../methods/full-drift/design.md)).
Collect: `collect_fd.py --only 'fd-s43k-*' --jobs 16`, `--only 'fd-s43m-*'` (song song), rồi `report_fd.py`.

Vòng 6–7 AUDITED (17/10 ~21:05–21:43Z). **Vòng 8–9** (plan 17/10; source tag `20261015-r8`, `-r9`; datasets
`<account>/mobihet-fulldrift-src-20261015-r8|r9` 48 file PASS): 12 kernel, bảng và giờ push ở
[bàn giao 17/10](continuation-2026-10-17.md); cổng [§Vòng 8–9](../../methods/full-drift/design.md). Test local `test_fd_local.py --part r8|r9`
(bit-exact arm cũ so stage vòng 5/8, đại số FedDyn/SCAFFOLD, JS, replay) + smoke + compiled PASS.
Vòng 8–10 (17 kernel, thêm vòng 10 s43) AUDITED 17/10 ~04:40Z, `w` (horizon ep15 s43) ~08:20Z: **`gpool` PASS hai seed**,
m4b horizon s43 PASS; kdcap/frz/FedDyn/SCAFFOLD/JS FAIL ([kết quả](../../methods/full-drift/results.md)). Collect sau khi máy ngủ: 4 tiến trình `--only` rời nhau.

Vòng 11 (tag `20261015-r11`, datasets `accanniez|takids1909/mobihet-fulldrift-src-20261015-r11` 48 file PASS; [cổng §Vòng 11](../../methods/full-drift/design.md)):

| kernel | GPU0 | GPU1 |
|---|---|---|
| accanniez/mobihet-fd-s42la-20261015-r11-v1 (5,4h) | fd_dml_m4b_gpool_kd2 | fd_dml_m4b_gpool_kdpx |
| takids1909/mobihet-fd-s42lc-20261015-r11-v1 (11h) | fd_dml_m4b_gpool:15 | fd_dml_m4b_gpool_gw2 |

PUSHED 08:23Z, QUEUED ~1,5h, RUNNING ~09:55Z (W&B có bước train). Test local `--part r11`: arm cũ bit-exact so stage r10;
kd2 (×1) = kdpx (cap ∞) = gw2 (×1) = gpool.

Vòng 11 `la` AUDITED 17/10 ~14:00Z (kd2/kdpx FAIL, [kết quả](../../methods/full-drift/results.md)). **Vòng 12** (tag `20261015-r12`, dataset
`odixe0502/mobihet-fulldrift-src-20261015-r12` 48 file VERIFIED; [cổng §Vòng 12](../../methods/full-drift/design.md)): `odixe0502/mobihet-fd-s42qa-20261015-r12-v1`
`fd_dml_m4b_gpool_q4`, một arm trên 2 GPU (`--workers 2`), PUSHED 14:25Z. Source r12 = r11 + vòng lặp không đồng bộ host,
`QPacket` int4 thật, `--workers` — bit-exact (test `--part r12`, `par`: 7 arm song song == tuần tự).
