# Gaussian từ moment lớp gộp

**Trạng thái (17/10 ~04:40Z):** Có lợi; m4b PASS s42 + s43; pooled > own +0,05 hai seed; ghép vào DML (`gpool`) PASS.

## Cơ chế
Client gửi theo lớp count, tổng x, tổng xxᵀ; RSU trả mean/covariance gộp. Private train CE real + CE trên mẫu
`N(μ_c, Σ_c)` sinh tại xe (hard label). Dữ liệu tĩnh: bootstrap moment một lần là đủ.

## Trao đổi
FP32 tam giác 16 lớp: d=66 145.856 B/chiều; m4b chỉ cần 53 tọa độ ⇒ 95.104 B (marginal chính xác, chưa đo codec).
Hợp R2 nhất (input-space, không cần cùng kiến trúc).

## Bằng chứng (audited)
- Full: đỉnh test ở r1 rồi giảm (học thuộc receiver). FullDrift s42 ep6: 0,5882; swappos +0,052; **m4b +0,067 (0,6553)**.
- Private EMA: +0,024 (không mask), +0,036 (swappos) nhưng **−0,010 với m4b (s42), −0,021 (s43)** ⇒ nhánh EMA dừng.
- Seed 43 (vòng 8 `r`): m4b vs `fd_gaussian` +0,053 (0,6568), swappos +0,049; EMA giúp không mask (+0,026) và swappos
  (+0,031), hại m4b (−0,021).
- Moment gộp vs riêng (`_own`, vòng 8/10): +0,051 (s42) / +0,052 (s43), client tệ nhất +0,044 / +0,091.
- Moment gộp làm dữ liệu phụ cho private DML-m4b (`gpool`): +0,016 / +0,024, xem [hybrid-private-only](hybrid-private-only-synthetic.md).

## Việc tiếp
Ghép vào DML (PASS): [hybrid-private-only](hybrid-private-only-synthetic.md).

## Nguồn
Nội bộ: [hồ sơ](../../methods/gaussian-class-moments/README.md), [FullDrift](../../methods/full-drift/results.md), [AntiMem](../../methods/anti-memorization/results.md), [plan §4.3](../../archive/research/algorithm-improvement-plan-2026-10-17.md).
