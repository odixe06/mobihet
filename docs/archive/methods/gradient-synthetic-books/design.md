# Synthetic book khớp gradient DAGSNet — căn cứ và cơ chế

## Căn cứ xây dựng

V5 IPC1/4 đạo hàm khả thi nhưng codebook không đủ utility. Bằng chứng và giới hạn tại [bài học phương pháp](../../../lessons/gradient-synthetic-books.md) và [tổng hợp bằng chứng](../../overview/evidence-summary.md). Không suy codec/probe PASS thành utility PASS.

## Cách thực hiện

Noise book IPC1/4; gradient matching so latent-mean, frozen/trained encoder;80 bước tổng round-robin lớp, refresh5round, giới hạn2×baseline.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế gốc, code và bằng chứng

- [pilot-v5-books.md](../../research/pilot-v5-books.md)
- [v3-knowledge-sources.md](../../research/v3-knowledge-sources.md)
- [v5_books.py](../../../../mh/v5_books.py)
- [report.md](../synthetic-knowledge/builds/anchor-synthetic-books/results/report.md)
- [rebuild.md](../synthetic-knowledge/builds/anchor-synthetic-books/results/rebuild.md)

Không raw/perturbed record rời xe. Mọi learned model hiện DAGSNet; R2 khác kiến trúc chỉ ở thiết kế. Pilot StageU không chứng minh contact hữu hạn, secure aggregation hoặc DP.

[Kết quả và các build](README.md).
