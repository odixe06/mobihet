# MobiHet L4 — ablation study (owner 21/11/2026)

Bốn ablation độc lập, mỗi cái xuất phát thẳng từ L4 đầy đủ ([L4](mobihet-l4.md)). Không cộng dồn thay đổi. Protocol, dataset v4,
5.343 xe, 50 round, seed 41/42/43 và 44 HP L4 còn lại giữ nguyên. L4 gốc và kết quả của nó không đổi. Kết quả nằm ở
[experiments/ablation-l4/](../../experiments/ablation-l4/), mỗi kịch bản một thư mục con (`dagsnet`, `bert`, `tabm`, `sae`,
`mlp`, `cnn-bigru`, `mixed-case1..3`, owner 21/11). Không ghi vào bảng baseline.

## Component → code

Chuỗi kế thừa: `mobihet` → `study` → `next` (Live) → `wstudy` → `sim` → `refine` → `compute` → `lite` → `budget` → `l4`.

| | component | vị trí |
|---|---|---|
| C1 | Federated Gaussian copula + K-means mixture (bootstrap 1 lần) | `mobihet.py:_copula_bootstrap`, `mobihet_sim.py:_kmeans`, codec km2 `mobihet_lite.py:_kmeans`, nén downlink `mobihet_wstudy.py:_copula_bootstrap` |
| C2 | Class-level KD (ma trận Q) | thu `_csum` ở `mobihet_budget.py:_loss_tier`, gộp theo count ở `mobihet.py:_round_private`, codec 8-bit `mobihet_sim.py:_round_private` |
| C3 | Logit adjustment | `la_shift` ở `mobihet.py:_bootstrap`, cộng vào logit dòng thật ở `_loss_tier` (budget/l4) |
| C4 | Synthetic data (CE r1, KL(Q[y] ‖ p) từ r2) | `mobihet.py:_synth`, liều theo tầng/round `mobihet_compute.py:_rows_for`, loss `_loss_tier` |
| C5 | Chọn dòng thật theo CE lưu (`real_select=top`) | `mobihet_budget.py:_real_rows`, `_score` |
| C6 | Ngân sách theo tầng + bước synthetic-only (≥ 16) | `real_tier_rows`, `synth_tier_*` (budget/compute), `synth_step_floor` ở `core/local.py` |

BERT L4 chạy harness `fl_veremi_bert` 49b2d88 + overlay 3 file (`scripts/run_t3.py`, `core/local_t3.py`, method
`mobihet_budget`). Snapshot đóng băng (manifest giống nhau ở cả 6 phiên) ở
`runs/recovery-20261120/references/bert-l4-artifacts/flb-fbl-l4-top-s41-r50/provenance/`. Logic method giống hệt repo này,
chỉ khác wrapper `cost_scope`.

## Bốn arm — quyết định owner 21/11

Code: [methods/mobihet_ablation.py](../../methods/mobihet_ablation.py) (mixin trước L4; hp `ablation`, bảng `HP`),
mixed [methods/mobihet_ablation_mixed.py](../../methods/mobihet_ablation_mixed.py) + `scripts/run_mixed_ablation.py`.
`ablation=none` = L4 nguyên vẹn (đối chứng).

| arm | bỏ | thay đổi | byte |
|---|---|---|---|
| A1 | synthetic (C4 + C6) | CE(z + la_shift) trên **mọi** dòng thật, batch 64, `synth_step_floor=0`. Không còn dòng synthetic nên C1, C2, C5 không được dùng: không dựng copula/KM, không thu/gộp Q, không lưu CE dòng (owner chọn bỏ hết) | bootstrap chỉ class count 64 B lên + 64 B xuống; 0 B/round |
| A2 | C5 | `real_select=random` (chỉ HP). Giữ số dòng thật/synthetic, ngân sách tầng, mọi loss. Không thêm forward | như L4 |
| A3 | C3 | `la_shift = 0` sau bootstrap. CE train và CE lưu cho C5 (chính là CE đó) không còn shift (owner chọn). Count, bootstrap, Q, synthetic, lấy mẫu giữ nguyên | như L4 |
| A4 | nén Q | Q raw fp32 dense 16 × 16 hai chiều. Không uint8, không ép fp16, không bitmask, không zlib. Giữ chuẩn hóa hàng khi giải mã như hai decoder L4, gộp theo count, lịch KD và codec bootstrap (owner chọn raw thay cho fp16) | 1.024 B lên + 1.024 B xuống mỗi xe mỗi round |

Lưu ý A4: Q gộp fp32 có tổng hàng < 1 khi một xe có lớp c nhưng round đó không train dòng nào của c (ngân sách tầng). Cả decoder
8-bit lẫn fp16 của L4 đều chuẩn hóa lại hàng, nên A4 giữ bước này. Đây là decode, không phải nén.

## Kiểm chứng local

`tests/test_mobihet_ablation.py` (CPU/GPU local, panel 20 xe, 2 round), `tests/test_mobihet_ablation_bert.py` (snapshot BERT):

- đối chứng `none` = `mobihet_l4` bit-exact (state gồm Q và CE dòng, CM) trên DAGSNet, TabM, SAE, MLP, CNN-BiGRU; BERT
  `none` = `mobihet_budget`; mixed case2 `none` = `mobihet_mixed_l4`.
- A1: bootstrap 64 + 64 B, không bảng generator, Q None, 0 B/round, số bước tối ưu = Σ⌈n/64⌉, loss và gradient = CE(z + LA).
- A2: không có CE dòng, `real_frac`/số bước/byte như L4. A3 = L4 với shift đặt 0 bằng tay (bit-exact), bootstrap như L4.
  A4: round 1 = L4 (Q chưa dùng), Q fp32 lệch L4 ≤ 2/255, không qua fp16, 1.024 B.
- resume r1 → r2 = chạy liền cho cả bốn arm. Mixed: mọi arm chạy 2 round, health từng backbone.

## Kiểm chứng Kaggle (2×T4, toàn bộ 5.343 xe)

Mixed case2 seed 42, pilot 2 round. Macro-F1 r2 của đối chứng `none` là .676158 (`accanniez`) và .676146 (`chiencosieuhang`, cùng
notebook). Pilot L4 gốc (`gaming3/mixl4-case2-s42/1`) là .676016. Ở r1 lệch 668/9.347.679 dòng test, loss lệch ở chữ số thứ 7.
Source thực thi chỉ khác L4 gốc ở wrapper `cost_scope` (`nullcontext`). L4 trên 2×T4 vì vậy **không tái lập bit-exact giữa các
phiên**: cộng atomic CUDA (`index_put_(accumulate=True)`, scatter trong backward) đổi thứ tự cộng float. Tương đương code chứng minh
bằng test local bit-exact. Trên Kaggle, đối chứng nằm trong dải nhiễu chạy lại (~1e-4 macro-F1 ở r2, so với SD giữa seed .00067 ở
R50). Mọi chênh lệch ablation − L4 phải đọc cùng dải nhiễu này.

## Chạy

Builder: `scripts/build_l4_fl.py --ablation <arm>` (DAGSNet, TabM, SAE, MLP, CNN-BiGRU), `scripts/build_mixed_l4.py --ablation
<arm>`, `scripts/build_bert_l4_ablation.py`. Notebook nhúng source + manifest; `--stop-after 2` = pilot, phiên sau `--parent`.
Run directory `<backbone>-abl-<arm>-s<seed>`, `mixl4-abl-<arm>-<case>-s<seed>`, `bert-abl-<arm>-s<seed>`.

Thứ tự: pilot 2 round DAGSNet và mixed case2 (seed 42, đủ 5 arm, đối chiếu `none` với L4 gốc r1–r2) → nối 50 round → mở rộng.
Chi phí ước từ L4 (giờ/run 50 round): DAGSNet 7,3, BERT 8,7, CNN-BiGRU 4,8, TabM/SAE/MLP 0,75, case1 10,5, case2 1,2,
case3 8,3. Cả ma trận 4 × 9 × 3 ≈ 550 GPU-h. Kernel và trạng thái: [bàn giao](../operations/continue.md).

## Compute (panel 200, 50 round, không eval)

`experiments/ablation-l4/<kịch bản>/compute.csv`. A1 rẻ hơn L4 ở backbone nhỏ (case2 0,49×, SAE 0,68×, MLP 0,78×, TabM 0,95×)
nhưng đắt hơn ở DAGSNet và BERT (1,09×): L4 chỉ train 62,6 % dòng thật theo ngân sách tầng, còn chi phí synthetic cố định
(~0,16–0,19 Tops/round) chỉ trội ở model nhỏ. A2/A3/A4 bằng L4 (chênh < 1e-6 Tops).

## Báo cáo

Mỗi `experiments/ablation-l4/<kịch bản>/` có `results.csv` (arm, seed, view, round, 10 metric, client mean/p10, F1 từng lớp
dạng JSON, status, exact ref) và `cost.csv` (byte bootstrap lên/xuống, lên/xuống 50 round, tổng, Tops client/server/tổng).
Lịch sử từng round, CM, checkpoint và source hash của mỗi run nằm trong output Kaggle đã pull (ORICO), CSV chỉ trỏ tới.

Mỗi arm × backbone/case × seed: 10 metric, F1 từng lớp, client mean/p10 ở R50. Byte bootstrap/lên/xuống/tổng 50 round.
FLOP client/server/tổng không gồm eval (harness `scripts/compute/`, cùng quy ước [compute](compute.md)). Mean ± SD mẫu,
chênh lệch so với L4 cùng seed.
