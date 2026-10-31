# Bài học — Kết hợp Gaussian moment và proxy DAGSNet

Hai thành phần có lợi riêng không chứng minh kết hợp có lợi. So hybrid với max(H1,H2), samecompute/BN/prior, khóa cổng trước chạy.

- Hai seed đã audit7arm×20round: hybrid hơn thành phần mạnh nhất≈+0,0407/+0,0436,
  hơn own≈+0,1470/+0,1499; utility paired PASS.
- Interaction≈−0,0677/−0,0722: hơnmax(H1,H2) khôngchứng minh synergy cộngdương.
  Giữma trận2×2 vàowncontrols; không gọi hai cơchế cộnglợi độc lập.
- Đợt dài đã mở:100%train/50round/seed42, đủ7two-DAGScomputecontrols.
  Không dùnglocal/Gaussian mộtDAGS của fullH1/H2 như compute-matched controls chohybrid.
  [Report](../../archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/results/report.md), [cổng](../../../archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/probes/gates.json), [fullcontract](builds/hybrid-full/protocol.md).

[Thiết kế/nguồn](README.md).

## Full-data đầu run — audit 14/10

7 arm giữ được18round, last r2/r3; chưa50round. Hybrid hơn DML cùngr2 +0,054897,
hơn hybrid-own cùngr3 +0,054330; interactionr2 âm. Tín hiệu pilot còn ở full-data đầu,
không chứng minh độ bền tới r50 hoặc R1/R2. Không so round/compute controls khác nhau.

Hai kernel ERROR do budget/parentassert3, worker rc0/checkpoint PASS, không rerun r1.
Nongaussian controls có dummy/eval-BN work và walltime cao hơn; cùng model/step khônglàsamewalltime.
Đo budget theoactualarm, không mượn timing H1/H2. [Report/giới hạn](builds/hybrid-full/results/feasibility-20261014.md).
