# Slim proxy — proxy DAGSNet thu hẹp kênh (FedKD), private giữ DAGSNet đầy đủ

**Trạng thái (17/10 ~14:00Z):** **Owner: không chạy** — tạm bỏ mọi hướng giảm độ phức tạp model; trọng tâm là cải thiện
quá trình huấn luyện (communication budget hạn chế + hỗ trợ đa mô hình). Giữ ghi chú làm hồ sơ.

## Cơ chế
Proxy chung = DAGSNet cùng cấu trúc (patch 6×11, bốn nhánh, head) nhưng mọi số kênh × w (w = 0,3: stem 29, growth 10,
inception 10, alex 38, fire 10/14, head 77). Private vẫn là DAGSNet 395.024 tham số, không bao giờ gửi. DML trao đổi
tri thức chỉ qua 16 logit trên dòng thật, nên proxy khác kiến trúc private không cần căn shape (đúng R2).

## Trao đổi (byte thật, mỗi client mỗi chiều mỗi round)
| proxy | tham số | fp32 | q4 ([QPacket](qproxy-quantization.md)) |
|---|---:|---:|---:|
| DAGSNet (hiện tại) | 395.024 | 1.593.440 B | 208.268 B (7,65×) |
| slim w 0,5 | 101.520 | 412.896 B | 56.364 B (28×) |
| **slim w 0,3** | **38.414** | **157.896 B** | **22.783 B (70×)** |

So sánh: LwFedNIDS (`fl_dagsnet_veremi`) 35.891 tham số fp32 ≈ 0,14 MB/chiều.

## Bằng chứng
- `fl_dagsnet_veremi` (cùng VeReMi 80/20, 20c, 50 round, test): LwFedNIDS (DAGSNet cắt 70 % kênh, 9,09 % tham số, FedAvg)
  F1 macro **0,7917**, không thấp hơn DAGSNet đầy đủ (AFPHA 0,775; FD-IDS đỉnh 0,782). Khác giao thức (không có baseline
  chưa cắt cùng run, D5) nên chỉ là bằng chứng hướng: sức chứa DAGSNet không phải nút thắt trên 66 feature.
- Pilot P1 (r20): proxy MLP thua rõ (proxy `dml_mlp256` 0,675 vs DAGSNet 0,774; private gp −0,035) ⇒ giữ inductive bias
  DAGSNet, chỉ thu hẹp kênh, không đổi sang MLP.
- Khoảng private–proxy (gpool s42 ep6, group): 56 % nằm ở lớp < 1 % dữ liệu local; private học teacher qua logit
  ⇒ điều cần giữ là chất lượng logit của proxy, không phải số tham số của nó.

## Kiểm (đề xuất)
`fd_dml_m4b_gpool_px30` (và `_px30_q4`) so `fd_dml_m4b_gpool` s42 ep6, cổng noninferior: private gp Δ ≥ −0,010, benign
≥ −0,010, client tệ nhất ≥ −0,010; proxy head báo kèm. Sai nếu Δ < −0,010 (proxy hẹp làm teacher kém đi).

## Nguồn
FedKD (Wu et al., Nat. Commun. 2022, [arXiv:2108.13323](https://arxiv.org/abs/2108.13323)); LwFedNIDS (Bouayad et al. 2024)
qua báo cáo `fl_dagsnet_veremi/report/lwfednids`; [pilot P1](../../archive/research/probes/pilot-p1.md).
