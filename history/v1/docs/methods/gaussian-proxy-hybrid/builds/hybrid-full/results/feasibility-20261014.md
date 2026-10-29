# Hybrid full — tính khả thi sau audit 14/10/2026

**Có tín hiệu hiệu năng tích cực trên 100% train; chưa xác nhận dài hạn r50.**
Đủ 4 kernel đã pull/audit, 7 arm và 18 round hoàn chỉnh của lịch cố định 50 round,
20 client, seed42. Hai kernel ERROR vẫn giữ checkpoint hợp lệ; không chạy lại từ r1.

[10 metrics các round cuối](latest-rounds.md) · [Report mọi round](report.md) ·
[CSV](metrics-all.csv) · [F1 theo lớp](per-class-f1.csv) · [Contract đã chạy](../protocol.md)

## Kết quả hiện có

Mean20client, private/test/gp; raw test20% đầy đủ, scaler đúng một lần.
Official test chỉ mô tả; không chọn HP, best round hoặc cổng mới theo các số này.

| arm | round cuối / target | F1 macro | benign F1 |
|---|---:|---:|---:|
| local hai DAGSNet | 2/50 | 0,544887 | 0,328742 |
| DML | 2/50 | 0,639706 | 0,421561 |
| DML-own | 2/50 | 0,625714 | 0,413811 |
| Gaussian full hai DAGSNet | 3/50 | 0,630724 | 0,343524 |
| Gaussian-own | 3/50 | 0,573754 | 0,345790 |
| Hybrid | 3/50 | 0,704922 | 0,452020 |
| Hybrid-own | 3/50 | 0,650592 | 0,437595 |

So sánh cùng round trong đúng bộ controls này:

| round | so sánh | chênh F1 macro |
|---:|---|---:|
| 2 | Hybrid − DML, thành phần mạnh nhất tại r2 | +0,054897 |
| 2 | Hybrid − Gaussian | +0,059539 |
| 3 | Hybrid − Hybrid-own | +0,054330 |
| 3 | Gaussian − Gaussian-own | +0,056970 |
| 2 | DML − DML-own | +0,013992 |

Interaction 2×2 tại r2 = −0,035280. Hybrid thắng từng thành phần nhưng chưa chứng minh
tương tác cộng hưởng dương. Không so Hybrid r3 với DML r2 để gọi là hơn DML.
Hybrid gp tăng r1→r3; Gaussian gp giảm trong đoạn này. Đoạn đầu không dự báo r50,
không cô lập overfit/prior/BN làm nguyên nhân và không cho phép đổi quy tắc quyết định.

## Kiểm chứng đã thực hiện

- Pull exit0: 376 file tổng cộng, 0 file thiếu. Log: [collection](../../../../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/logs/collection-20261014.log).
- Executed notebook version1 khớp source đã PUSHED; bản có key chỉ trong ignored `private_push/`,
  bản redacted ở notebook gốc. Source manifest/driver fingerprint/data manifest khớp.
- 43.045.415 train và 10.761.343 test row; test-label/order hash khớp manifest cho từng round.
- Checkpoint chứa đủ 20 private +20 proxy, strict DAGSNet rebuild và BN/finite PASS;
  state/payload CRC/version/cohort/chain, applied/skipped, lịch cosine50, pred→CM→10metrics PASS.
- 42 kiểm inference trên 16.384 raw-test-prefix row, client0, r1 và round cuối từng arm;
  private head/gp và proxy head khớp prediction trên điểm có margin>0,02.
  [Proof](../../../../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/probes/inference-proof.json), [log](../../../../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/logs/inference-20261014.log). Không full-inference hoặc leakage proof.

## Vì sao hai kernel ERROR, cần chạy lại không?

`hybf1` và `hybf3`: worker exit0, dừng bởi `budget stop before round 3`.
Controller cuối notebook yêu cầu mọi arm đủ3round nên phát assertion và kernel ERROR.
Local/DML/DML-own đủ2round; Gaussian-own vẫn đủ3round trong kernel ERROR.
Không thấy traceback nonfinite/auth/OOM của worker trong log terminal.

Output đủ để audit và lấy checkpoint. Không rerun notebook calibration y nguyên hoặc restart r1.
Muốn đủ r3/tiếp r50: nối đúng parent đã audit, last-round riêng mỗi arm; controller nhận
completed boundary theo budget, không bắt mọi arm có cùng số round trong một session.
Source/model/HP/LR50 giữ nguyên; import gate trước decode dữ liệu, marker publish cuối.

## Chi phí và quyết định tiếp theo

| arm | worst round đã đo, giây |
|---|---:|
| local | 6652,1 |
| DML | 6648,6 |
| DML-own | 6508,6 |
| Gaussian | 3706,1 |
| Gaussian-own | 3669,1 |
| Hybrid | 3656,1 |
| Hybrid-own | 3814,1 |

Hai DAGSNet/cùng số SGD step không bảo đảm cùng walltime. Trong adaptation này,
controls không Gaussian đắt hơn nhánh Gaussian/Hybrid; dummy/eval-BN work thuộc chi phí thật.
Chưa có profiling để gán toàn bộ chênh cho một nguyên nhân.

Giữ pairing hiện tại, worst-round đã đo và dư15%: khoảng **314,70 giờ phiên** còn lại cho
Hybrid tới r50, **chưa gồm** các lần preparation/import/compile lặp lại.
[Ước tính và giới hạn](../../../../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/probes/budget-remaining-20261014.json), [quota snapshot](../../../../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/probes/quota-20261014.txt).
Đây là dự toán điều kiện, không reservation hoặc bảo đảm runtime; đổi cách xếp arm hoặc tốc độ
round tương lai sẽ đổi dự toán. Không cộng hai GPU trong cùng notebook như hai phiên account.

Ưu tiên hiệu năng vẫn là Hybrid và DML; Gaussian còn lợi so own nhưng không có cơ sở hứa
thêm round tự tăng metric. Tiếp theo quota/refresh17Oct đã chốt; không sửa full run theo test.
BoundaryCells mới cost PASS, chưa pilot r20; EMA chưa triển khai. Không kéo dài adaptation
V2–V7/V9/V10 đã FAIL để tìm điểm thắng.

R1 contact hữu hạn/ACK/deadline chưa đo ở đợt full StageU này. R2 khác kiến trúc vẫn chỉ thiết kế;
tất cả learned models DAGSNet. Synthetic/statistic exchange chưa chứng minh DP/secure aggregation crypto.
