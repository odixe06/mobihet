# Truyền snapshot qua contact hữu hạn — căn cứ và cơ chế

## Căn cứ xây dựng

P2 audit deadline/byte/prior; chưa chứng minh đầy đủ R1. Bằng chứng và giới hạn tại [bài học phương pháp](../../../lessons/contact-fragmentation.md) và [tổng hợp bằng chứng](../../overview/evidence-summary.md). Không suy codec/probe PASS thành utility PASS.

## Cách thực hiện

Phân mảnh object, trả phí frame/ACK/retry; chỉ kích hoạt snapshot coherent đúng version trước deadline.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế gốc, code và bằng chứng

- [de-xuat-huong-moi.md](../../research/de-xuat-huong-moi.md)
- [pilot-plan.md](../../research/pilot-plan.md)
- [pilot-p2.md](../../research/probes/pilot-p2.md)
- [channel.py](../../../../core/rsu_contact.py)
- [report.md](builds/fragmented-snapshots/results/report.md)

Không raw/perturbed record rời xe. Mọi learned model hiện DAGSNet; R2 khác kiến trúc chỉ ở thiết kế. Pilot StageU không chứng minh contact hữu hạn, secure aggregation hoặc DP.

[Kết quả và các build](README.md).
