# Kết hợp Gaussian moment và proxy DAGSNet

[Danh mục](../README.md) · [Bài học](lessons.md)

Nhãn lịch sử: H1+H2/V8. Pilot hai seed đã audit/utility paired PASS.
Full100%train/50round/seed42: đã audit 7arm tới r2–r3, chưa r50;
[tính khả thi hiện có](../../papers/hybrid-full/feasibility-20261014.md).

## Cơ chế FL

Private/proxy DAGSNet học tương hỗ; Gaussian gộp cung cấp thêm input synthetic tại xe.

## Tài liệu liên quan

- [CONTEXT.md](../../CONTEXT.md)
- [context-history-before-method-folders-2026-10-12.md](../../docs/operations/context-history-before-method-folders-2026-10-12.md)
- [pilot-p1.md](../../docs/research/probes/pilot-p1.md)
- [Manifest V8](../../docs/research/pilot-v8-hybrid.md), [ledger](../../papers/v8/rebuild.md)
- [Report đãaudit](../../papers/v8/report.md), [cổng](../../papers/v8/gates.json)
- [Đợt dài đãchốt, nguồn riêng](../../docs/research/full-hybrid.md)
- [Full ledger/preflight/READY](../../papers/hybrid-full/rebuild.md)
- [Full 10 metrics cuối](../../papers/hybrid-full/latest-rounds.md)

Mọi learned model giữ DAGSNet; không raw record rời xe. ChưaR1/R2proof.
