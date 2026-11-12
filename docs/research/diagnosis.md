# Chẩn đoán — bằng chứng còn hiệu lực (kiểm 09/11/2026)

Phương pháp: [proposal](proposal.md). Nút thắt chi tiết: [bottleneck 08/11](../../archive/docs-20261113/research/bottleneck-review-20261108.md).
Phép thử đang mở: chỉ ở [catalog đề xuất](improvement-proposals.md) (ngoài generator) — generator đã đóng băng ([catalog](generators.md)).

## Những điều đã có bằng chứng

| hiện tượng | điều được phép kết luận | nguồn |
|---|---|---|
| Gap KM50 r50 → APFed nằm ở benign recall + precision lớp tấn công trên benign tương lai | nút thắt là tổng quát hóa của model riêng (train P .99 vs test .84), không phải generator | [catalog đề xuất](improvement-proposals.md), [bottleneck](../../archive/docs-20261113/research/bottleneck-review-20261108.md) |
| Synthetic là kênh tri thức toàn mạng, kể cả ở xe lớn | bỏ synthetic hại đơn điệu; tăng trọng số synthetic cũng hại | [sim b1/b2](screening-sim-foldval.md) |
| Q và floor đều `needed` ở r50 | giữ cả hai; Q giữ ranh giới benign, floor giữ xe nhỏ/lớp vắng | [Y50](results-y50.md) |
| Oracle +.018 so Core r50 | dư địa generator có thật; KM lấy ≈ nửa; biến thể generator tiếp đều ≤ 0 ở sim | [Y50](results-y50.md), [km-lite](screening-km-lite.md) |
| Lợi r20 có thể chỉ là tăng tốc | phải xác nhận ở r50 (bank mất lợi) | [F50](results-f50.md) |
| Prior shift lúc train tăng pooled nhưng hại timeDelay/absent | không nhận từ pooled gain | [Y50](results-y50.md) |
| Tỷ lệ/nhiệt synthetic chỉ dời ngưỡng benign↔timeDelay | knob share/temp không thêm tri thức | [km-lite](screening-km-lite.md) |
| Prior local lúc suy luận, EMA, Q theo cụm, Q mềm đều FAIL (b11) | FP benign không do prior/ngưỡng; nội dung/độ sắc Q hiện tại đúng mức | [catalog đề xuất](improvement-proposals.md) |
| Divergence/sqrtcount/smoothing b13 và interpolation/logpool b14 đều FAIL hai seed | không có ứng viên promote; confidence cao và byte mới không bảo đảm generalization | [phân tích 11/11](../../archive/docs-20261113/research/b14-analysis-20261111.md) |

## Ranh giới suy luận

Centralized cùng 77 feature phân biệt timeDelay tốt hơn personalized; không gọi feature set bất lực tuyệt đối.
Không cộng gain ablation; live teacher đổi cùng student nên muốn kết luận nhân quả về nội dung Q cần replay.
Không mở lại mixture/projection/BNsplit/proxy/generator đã FAIL bằng tên mới.
Owner10/11 đã cho đổi module/tăng compute, bắt buộc hỗ trợ khác kiến trúc, experiment chỉDAGSNet;
capcompute cũ không còn cho nghiên cứu mới. Không tự giả định public real data hoặc đổi protocol sang test-time adaptation.
