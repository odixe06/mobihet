# Ablation MobiHet L4

Bốn ablation độc lập từ L4 đầy đủ (owner 21/11/2026). Thiết kế, quyết định và kiểm chứng: [mobihet-l4-ablation](../../docs/research/mobihet-l4-ablation.md).
Bảng baseline và L4 ở `experiments/<kịch bản>/` không đổi. So sánh với hàng `mobihet_l4` cùng seed trong bảng đó.

| arm | bỏ |
|---|---|
| a1 | synthetic data + tier budget (kéo theo copula/KM, class-Q, chọn dòng theo CE) |
| a2 | chọn dòng thật theo CE (`real_select=random`) |
| a3 | logit adjustment |
| a4 | nén Q (raw fp32 dense, 1.024 B mỗi chiều) |

Mỗi thư mục con là một kịch bản (`dagsnet`, `bert`, `tabm`, `sae`, `mlp`, `cnn-bigru`, `mixed-case1..3`):

- `results.csv`: arm, seed, view, round 50, 10 metric, client F1-macro mean/p10, F1 từng lớp (JSON), status, exact ref.
  Ghi bằng `scripts/audit_ablation.py --record` sau khi audit đủ 50 CM.
- `cost.csv`: byte mỗi xe (bootstrap lên/xuống, lên/xuống cộng 50 round, tổng) và tổng mạng (× 5.343), theo từng seed.
- `compute.csv`: Tops client/server/bootstrap/khác/tổng (không gồm eval) cho L4 và từng arm, panel 200, 50 round ngoại suy,
  `x_l4` = tổng/L4. Sinh bằng `python scripts/compute/report_ablation.py`.

Lịch sử từng round, CM, source snapshot và manifest của mỗi run nằm trong bản pull ở
`runs/ablation-l4/pulls/<owner>/<slug>/` (ORICO). Checkpoint ở output Kaggle.
