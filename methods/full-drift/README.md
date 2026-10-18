# FullDrift

`builds/fd-probe/source/proj/fd.py`: recipe full (`full_run`/`full_method` đã freeze) trên fixture full có holdout nội bộ
+ arm prox/avgm/FedAvg/swappos/m4b/gpool…, role `private_ema`; một lần gộp sau mỗi epoch local. `fd_prepare.py`: dựng fixture holdout trên Kaggle.
`scripts/gen_fd.py`: stage dataset + notebook public/private (không upload/push). `scripts/test_fd_local.py`: bit-exact
`fd_dml`/`fd_gaussian` so epoch `full_run`, smoke mọi arm. `artifacts/`: input stage và output Kaggle (ignored).
Thiết kế/cổng/kết quả: [docs](../../docs/methods/full-drift/README.md).
