# Fourier sketch phân phối lớp — căn cứ và cơ chế

## Căn cứ xây dựng

S-a KME không hơn Gaussian đủ cổng; không mở S-b. Bằng chứng và giới hạn tại [bài học phương pháp](../../../lessons/fourier-distribution-sketch.md) và [tổng hợp bằng chứng](../../overview/evidence-summary.md). Không suy codec/probe PASS thành utility PASS.

## Cách thực hiện

SecureSum đặc trưng Fourier theo lớp; tối ưu synthetic bank từ noise để khớp sketch, xe train trên bank.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế gốc, code và bằng chứng

- [de-xuat-bo-sung-2026-10-11.md](../../research/de-xuat-bo-sung-2026-10-11.md)
- [pilot-new-fl.md](../../research/pilot-new-fl.md)
- [pilot-new-fl.md](../../research/probes/pilot-new-fl.md)
- [new_fl.py](../../../../core/artifact_driver.py)
- [report.md](../synthetic-distribution-matching/builds/static-synthetic-books/results/report.md)
- [rebuild.md](../synthetic-distribution-matching/builds/static-synthetic-books/results/rebuild.md)

Không raw/perturbed record rời xe. Mọi learned model hiện DAGSNet; R2 khác kiến trúc chỉ ở thiết kế. Pilot StageU không chứng minh contact hữu hạn, secure aggregation hoặc DP.

[Kết quả và các build](README.md).
