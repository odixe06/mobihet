# Continue — bàn giao sống, 30/10/2026 ~11:20 (giờ VN)

**File duy nhất cho "đang ở đâu, làm gì tiếp".** Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) →
[bài học vận hành](lessons.md) → trang này. Sửa tại chỗ.

## Đang chạy

Không có kernel MobiHet nào chạy. (Repo baseline còn `fl2-nilm-r2`, `fl2-tinyproto-r3`: xem bàn giao của repo đó.)

## Đã xong (30/10) — chi tiết + cổng: [screening-r10](../research/screening-r10.md)

Sàng lọc 10 round, full 5.343 xe, W = 1, seed 42, `group_rows` 4.096, budget 4 h, W&B `mobihet-v4`. Pull: `runs/pulls/<slug>/`
(git-ignored, bỏ `state*.pt`); audit 10/10 PASS (đủ round, CMs, `models/rXXX.pt`, không NaN, HP đúng protocol).

| kernel | account | SHA | arm / synth | private F1 r10 | p10 |
|---|---|---|---|---|---|
| `mh-s0-dml` | odixe0502 | 6c3536d | dml (v1) | 0,8584 (r9, dừng budget) | 0,542 |
| `mh-s1-srv` | odixe0502 | 6c3536d | srv | 0,8514 — FAIL | 0,587 |
| `mh-s2-srvclient` | minhtriethihi | 6c3536d | srv, n_k | 0,8451 | 0,582 |
| `mh-s3-cheap` | minhtran0601 | 6c3536d | cheap | 0,7778 — FAIL | 0,464 |
| `mh-s4-none` | trietbackup | 6c3536d | **none** / gauss | **0,8679** | 0,567 |
| `mh-g0-off` | catbaochau | f86e137 | none / off | 0,8497 | 0,489 |
| `mh-g1-oracle` | catbaochau | f86e137 | none / oracle (trần) | 0,8802 | 0,590 |
| `mh-g2-short` | nguyenminhtran060113 | f86e137 | none / short | 0,8669 — FAIL | 0,542 |
| `mh-g3-mix` | annhannh331 | f86e137 | none / mix | 0,8667 — FAIL | 0,580 |
| `mh-g4-copula` | annhtnt | f86e137 | none / **copula** | **0,8732** — FAIL cổng +0,010 | 0,574 |

Kết luận: bỏ proxy (S4) tốt nhất và rẻ nhất (529 s/round, 544 B/round, 4,48 × fwd/dòng); synthetic vẫn cần; copula tốt nhất trong
các bộ sinh, trần oracle còn +0,012. Owner 30/10: không proxy vẫn cho model khác nhau mỗi xe — chỉ xét về phương pháp, không chạy kiểm chứng.

## Bước tiếp — chờ owner duyệt đề xuất [improvement-plan §5](../research/improvement-plan.md)

1. H0 (seed 43 cho S4 và G4) + H1–H4 trên nền G4, 10 round, song song (6 run × ~1,6 h). Code cần thêm:
   - H0: chỉ `--set seed=43` (kiểm tên run W&B không trùng).
   - H1: copula hỗn hợp theo (lớp × cụm scenario·run) — cụm từ `clients.parquet` như `methods/afpha.py` của repo baseline.
   - H2: `synth=copula_mix` (mix trên dòng copula); H3: hp số dòng synthetic/bước (mặc định batch/2); H4: đích synthetic one-hot.
   - Test toy + smoke local (`--subset 100 --until 2 --compile --set group_rows=1024 eval_rows=2048`; GPU local 4 GB OOM với mặc định).
2. F: arm thắng (hoặc G4) + S4, 50 round (~8 h/run, budget 11,5 h); so [results baseline](https://github.com/odixe06/fl_dagsnet_veremi_v2/blob/main/docs/research/results.md).

Push: `python scripts/gen_notebook.py --owner <acct> --slug <slug> --method mobihet --rounds 50 --until 10 --max-hours 4 --set arm=none group_rows=4096 synth=<s>`
→ `embed_wandb_key.py <nb> --metadata <meta>` → `chmod 600` → `ast.parse` mọi cell → `kaggle_as.py <acct> -- kaggle kernels push -p notebooks/<slug> -t 14400`.
Quota refresh hằng ngày 00:00 UTC; đọc live `kaggle_account.py quota` trước khi push.

## Việc tay (owner)

- Duyệt đề xuất §5 (H0–H4, F).
