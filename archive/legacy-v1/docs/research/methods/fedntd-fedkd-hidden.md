# Đổi kênh KD — FedNTD, hidden-state KD (FedKD)

**Trạng thái (17/10):** FAIL (không quét tiếp).

## Cơ chế
FedNTD: private chỉ distill lớp không đúng. FedKD hidden: private khớp tầng ẩn 256-d của proxy qua projector local
(không upload), cùng hệ số thích nghi.

## Bằng chứng (audited, s42 ep6 private gp)
FedNTD −0,042; hidden KD +0,006 (< +0,010); cùng kênh KD, m4b +0,036.

## Bài học
Khoảng private–proxy không do lệch prior hay kênh KD hẹp, mà do private học shortcut receiver.
Trước khi thêm kênh chưng cất, kiểm private có học shortcut local không.

## Nguồn
Lee et al., FedNTD, NeurIPS 2022 [arXiv:2106.03097](https://arxiv.org/abs/2106.03097); Wu et al., FedKD, Nat. Commun. 2022
[arXiv:2108.13323](https://arxiv.org/abs/2108.13323). Nội bộ: [FullDrift vòng 3–4](../../methods/full-drift/results.md), [hướng client](../../archive/research/client-model-directions-2026-10-16.md).
