# DML — private ↔ proxy chung học tương hỗ

**Trạng thái (17/10 ~04:40Z):** Nền đang dùng; headline DML + m4b; lợi FL so proxy không gộp +0,040–0,043 (hai seed).

## Cơ chế
Mỗi xe giữ private `f_i` (kiến trúc tự do, R2) và proxy `g_i` DAGSNet chung. Cùng batch local: CE cho cả hai +
KL hai chiều với hệ số `w = 1/stopgrad(CE_priv + CE_proxy)`. RSU chỉ gộp proxy (FedAvg dense hoặc q4); private không rời xe.
Tiền lệ: Deep Mutual Learning, Federated Mutual Learning (FML).

## Trao đổi
Proxy dense ≈ 1,59 MB/chiều (q4 208 kB). Private không gửi. Primary hiện tại: private gp trên group holdout (owner 16/10).

## Bằng chứng (audited)
- Full test (mô tả): `h_dml` r41 proxy 0,801, private gp 0,748; r23 proxy 0,787 > private gp 0,733.
- FullDrift ep6 group: `fd_dml` private gp 0,6655, proxy 0,763 phẳng từ ep4 (benign proxy 0,74→0,36 tới ep15).
- Khoảng private–proxy do private học shortcut receiver, không do kênh KD hẹp ([FedNTD/FedKD](fedntd-fedkd-hidden.md)).
- Vòng 8–10: m4b vs proxy không gộp (`_own`) +0,043 / +0,040 (client tệ nhất +0,138 / +0,132); trọng số KD
  `1/(CE+CE)` median 2,4→4,1, giảm về ≤1 (`kdcap`) hại private −0,025…−0,028 nhưng proxy +0,024/+0,033 ([KD cap](kd-cap-frozen-teacher.md)).

## Việc tiếp
Nối `h_dml` r41→r50 (q5). Cải tiến cách private hấp thụ teacher: [kdcap/frozen](kd-cap-frozen-teacher.md), drift:
[FedDyn](feddyn-proxy.md), [SCAFFOLD](scaffold-proxy.md). Đối chứng FL: `fd_dml_m4b_own` (vòng 8).

## Nguồn
Zhang et al., *Deep Mutual Learning*, CVPR 2018 [arXiv:1706.00384](https://arxiv.org/abs/1706.00384);
Shen et al., *Federated Mutual Learning*, 2020 [arXiv:2006.16765](https://arxiv.org/abs/2006.16765).
Nội bộ: [hồ sơ](../../methods/proxy-mutual-distillation/README.md), [FullDrift](../../methods/full-drift/results.md), [hướng client](../../archive/research/client-model-directions-2026-10-16.md).
