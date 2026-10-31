# FedProx / FedAvgM trên proxy

**Trạng thái (17/10):** FAIL (không nối).

## Cơ chế
FedProx: `μ/2‖w − w_t‖²` (μ=0,01) trên proxy. FedAvgM: vận tốc server β=0,9 trên delta trung bình.

## Bằng chứng (audited, s42 ep6)
FedProx giảm drift 5× nhưng F1 không đổi (benign +0,098).
FedAvgM β0,9 phân kỳ.

## Bài học
Kéo về model chung không tăng utility; correction có hướng ([FedDyn](feddyn-proxy.md), [SCAFFOLD](scaffold-proxy.md))
cũng FAIL ở full. Không bác toàn họ.

## Nguồn
Li et al., FedProx [arXiv:1812.06127](https://arxiv.org/abs/1812.06127); Hsu et al., FedAvgM [arXiv:1909.06335](https://arxiv.org/abs/1909.06335).
Nội bộ: [FullDrift](../../methods/full-drift/results.md).
