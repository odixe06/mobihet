# MobiHet L4 có logit adjustment — archive 24/11/2026

Owner 24/11/2026 nhận **A3 (L4 bỏ logit adjustment, `la_mode=off`) làm MobiHet chính**, vẫn tên `mobihet`.
Lý do: [screening LA](../../docs/research/screening-la.md) và ablation R50 (bỏ LA dương 24/24 cặp seed).
Mọi hàng và số liệu từng chạy với L4 **có LA** được chuyển khỏi repo `~/project/experiment` vào đây. Đây là lịch sử, không phải
kết quả hiện hành. Kết quả MobiHet hiện hành ở `~/project/experiment/experiments/<kịch bản>/results.csv` (method `mobihet`).

## Nội dung

| đường dẫn | nguồn trong experiment (trước 24/11) | ý nghĩa |
|---|---|---|
| `experiment/scenario-rows/<kịch bản>/results.csv` | hàng `mobihet_l4` của `experiments/<kịch bản>/results.csv` | 10 metric R50, 3 seed × 9 kịch bản (27 hàng) |
| `experiment/scenario-rows/<kịch bản>/compute.csv`, `compute-detail.csv` | hàng `mobihet_l4` của hai bảng compute | FLOP L4 (panel 200, quy về 50 round) |
| `experiment/communication-budget/*.csv` | hàng `mobihet_l4` của `artifacts/communication-budget/` | byte L4 đo từ log (canonical + pilot) |
| `experiment/ablation-l4/` | `experiments/ablation-l4/` | ablation A1–A4 **từ L4** (93/108 ô R50), byte, FLOP |
| `experiment/docs/mobihet-l4.md`, `mobihet-l4-ablation.md` | `docs/research/` | thiết kế, kết quả và kiểm chứng L4 + ablation L4 |

Repo mobihet vẫn giữ lịch sử phát triển L4 của chính nó: [FB-L](../../docs/research/screening-fbl.md) (sim, pilot, r50 DAGSNet/BERT).

## Artifact gốc (không chép vào Git)

- Bản pull Kaggle ablation L4: `~/project/experiment/runs/ablation-l4/pulls/<owner>/<slug>/` → ORICO
  `/run/media/odixe/ORICO/experiment-data/experiment/ablation-l4/pulls/`. Ledger `runs/ablation-l4/push-ledger.jsonl`, trạng thái
  `runs/ablation-l4/watch-collect.json`. 15 ô chưa thu đã bị Kaggle hủy hoặc thuộc owner mất quyền; không thu tiếp.
- Run L4 chính: nguồn ghi trong cột `source` của từng hàng archive (exact `owner/slug/version`).
- Trace FLOP L4: `~/project/experiment/runs/compute-legacy/20261119/`, `runs/compute-20261120/`, `runs/compute-detailed-v3/`
  (tên `*-mobihet_l4`, `dags-mh-l4`, `bert-mh-l4`, `*-abl-a*`).

## Tái lập

Code L4 nằm trong lịch sử Git của experiment trước commit chuyển sang MobiHet (`L4_HP` không có `la_mode`), hoặc dùng
`la_mode=full` với code hiện tại (`methods/mobihet_budget.py`). Ablation A3 cũ = MobiHet hiện tại (test bit-exact
`tests/test_mobihet_ablation.py::test_mobihet_is_l4_without_la`); 24 run A3 R50 đã audit nay là hàng `mobihet`
(`scripts/promote_mobihet_rows.py`, receipt `runs/mobihet-promote/receipt.json`).
