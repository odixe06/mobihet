# Synthetic book khớp latent mean — căn cứ và cơ chế

## Căn cứ xây dựng

DM tĩnh và mean-latent refreshed không vượt Gaussian liên tục. Bằng chứng và giới hạn tại [bài học phương pháp](../../../lessons/latent-mean-synthetic-books.md) và [tổng hợp bằng chứng](../../overview/evidence-summary.md). Không suy codec/probe PASS thành utility PASS.

## Cách thực hiện

Tối ưu noise book theo mean latent DAGSNet; so encoder đóng băng/encoder private đang học và book refresh. Chỉ book synthetic đi qua RSU.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế gốc, code và bằng chứng

- [pilot-new-fl.md](../../research/pilot-new-fl.md)
- [pilot-v5-books.md](../../research/pilot-v5-books.md)
- [pilot-new-fl.md](../../research/probes/pilot-new-fl.md)
- [new_fl.py](../../../../core/artifact_driver.py)
- [v5_books.py](../../../../mh/v5_books.py)
- [report.md](../synthetic-distribution-matching/builds/static-synthetic-books/results/report.md)
- [report.md](../synthetic-knowledge/builds/anchor-synthetic-books/results/report.md)
- [rebuild.md](../synthetic-knowledge/builds/anchor-synthetic-books/results/rebuild.md)

Không raw/perturbed record rời xe. Mọi learned model hiện DAGSNet; R2 khác kiến trúc chỉ ở thiết kế. Pilot StageU không chứng minh contact hữu hạn, secure aggregation hoặc DP.

[Kết quả và các build](README.md).
