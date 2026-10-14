# Dự báo khả năng cải thiện hiệu năng — 2026-10-13

**Dự báo trước kiểm nghiệm, không phải kết quả.** Căn cứ artifact đã audit của dự án;
không suy khả năng tăng IDS metrics từ việc codec, đạo hàm hoặc notebook chạy được.
Mức cao/vừa/thấp là thứ tự ưu tiên định tính, không phải xác suất đã hiệu chuẩn.
Không đủ dữ liệu để dự báo một giá trị F1 r50 hoặc khoảng tin cậy số cho phương pháp mới.

Đọc cùng [bằng chứng hiện có](../../proposals/evidence-summary.md),
[trạng thái phương pháp](../../proposals/status-summary.md),
[nguồn primary cho hướng mới](new-fl-primary-sources-2026-10-13.md).

## 1. Hai câu hỏi phải tách

1. Phương pháp có khả năng **hơn local/own/no-FL** không? Đây là lợi cộng tác.
2. Phương pháp có khả năng **hơn Gaussian hoặc DML fresh mạnh nhất**, trong cùng ngân sách không?
   Đây là cải tiến có ích so với phương án hiện có.

Tăng metric từ round đầu đến round cuối không tự chứng minh câu hỏi nào.
So sánh paired cùng seed, fixture, lịch LR, quyết định chính và compute; không so số của P1
với V8/full như cùng protocol. Không chọn best round sau khi thấy holdout/test.
Đích chính là private/group/gp F1 macro; đủ 10 metrics, benign và từng lớp là guard/diagnostic.
Accuracy, micro và weighted có thể tăng trong khi macro giảm; không dự báo tất cả cùng tăng.

## 2. Hướng hiện có: đáng kiểm dài đến đâu?

| hướng | cơ sở đã đo | dự báo hơn local | dự báo lợi của kéo dài | nguy cơ / bằng chứng bác bỏ |
|---|---|---|---|---|
| Gaussian class moments | P1 hai seed: global covariance hơn own; full > lr4 > diag. Full calibration đã audit r1–r3, chưa r50 | **Cao nhất trong các hướng thống kê hiện có** | **Vừa**: có thể học kỹ tri thức quần thể hơn; moment tĩnh không bổ sung thông tin mới theo round | Thuộc synthetic/local; gap global−own mất ở cuối; không cứu positionMirroring bằng thêm epoch nếu thiếu quan hệ phi tuyến |
| DAGSNet mutual distillation | P1 private/teacher có lợi; full r1–r3 có gap DML−own, vẫn là early descriptive | **Cao** trên pilot cùng kiến trúc | **Vừa** cho teacher/head; **không chắc cho private gp** | Prior private dịch về global làm gp cũ sửa quá tay; gap FL−own/benign giảm. Carrier nhỏ đã FAIL không được hưởng dự báo của dense |
| Gaussian + proxy hybrid | V8 seed43 utility PASS, hơn thành phần mạnh hơn; seed42 chưa audit | **Vừa**, còn thiếu paired confirmation | **Vừa nhưng bất định**: đầu run có thể hưởng hai nguồn tri thức, cuối run có thể bão hòa | Interaction 2×2 âm; không chứng minh synergy. R15→r20 seed43 gp hơi giảm: không ngoại suy tăng đơn điệu |
| EMA backbone + local BN + head fit riêng | Lịch sử có lợi nhỏ; EMA cả head hỏng. Một số bằng chứng cũ dùng record cache, vi phạm R3 hiện tại | **Vừa** cho ổn định công thức local; chưa là lợi FL | **Vừa** cho độ bền cuối run, **thấp/vừa** cho tăng mức đỉnh | Không có raw peer cache trong bản mới; head fit quá dài overfit. Nếu local/own tăng bằng FL thì chỉ là recipe local |
| Các adaptation paired FAIL V2–V7 | Own/shuffle/reference đã bác cổng chất lượng hoặc numerical/cost; riêng V10 F1 candidate r20 chưa đo | **Thấp cho cấu hình đã thử** | **Không có cơ sở mở dài nguyên trạng** | Phải sửa cơ chế từ nguyên nhân có thể kiểm chứng; không kéo dài để tìm round thắng hoặc đổi headline |
| V9 residual anchor teacher | Seed42 cost PASS, quality FAIL vs class-mean/hard/local; seed43 còn audit | **Thấp cho adaptation hiện tại** | **Thấp**: fit MSE nhỏ chưa mang ranh giới hữu ích | Không hơn class-mean/hard labels; không mở private KD hoặc kiểm dài từ cost/fit PASS |

Nguồn số: [P1](probes/pilot-p1.md), [V8](../../papers/v8/report.md),
[full calibration](../../papers/full-validation/report.md),
[V9](../../papers/v9/report.md), [V10](../../papers/v10/report.md).
Các run full hiện tại giữ source/HP/quy tắc và lịch 50 round; đây không phải đề nghị sửa run đang chạy.

## 3. Hướng khác cơ chế hiện tại: dự báo có điều kiện

### BoundaryCells — đếm nhãn theo vùng input, học trên record của chính xe

**Dự báo trước probe: vừa so local, thấp/chưa rõ so Gaussian-full hoặc DML.**
Lý do ủng hộ: lỗi chính là ranh giới giữa receiver; mean lớp không đủ. Bảng đếm theo conjunction
giữ phụ thuộc nhãn–vùng input, không phụ thuộc latent đang trôi hoặc Gaussian coverage.
Targets đặt tại x local nên gradient tác động toàn DAGSNet, không chỉ head.

Lý do phản đối: 2/4 cặp và 4 bin rất thô; quan hệ ≥3 feature, hướng đường và temporal pattern có thể mất.
Suppress ô rare có thể loại chính các lớp khó. Balanced target có thể làm gp cũ lệch.
Đây là adaptation, không kế thừa kết quả RFL dùng learned rule network.
[Nguồn RFL](https://proceedings.mlr.press/v306/qin26g.html).

**Direct so teacher trung gian:** direct có cơ sở tốt hơn về fidelity của target và cost;
teacher DAGSNet có thể làm mượt/ràng buộc hàm nhưng cũng thêm sai số. V5/V9 chứng minh fit teacher
không đủ; không mặc định teacher tốt hơn. Cả hai được chủ dự án chốt kiểm.

Trước pilot: đo train-only table coverage theo lớp, log-score, entropy, own và cell-shuffle;
kiểm chuyển sang receiver-group holdout. Targets không hơn own/shuffle thì hạ dự báo xuống thấp và dừng.
Probe target PASS chỉ là điều kiện cần, không dự báo DAGSNet F1 PASS.

**Sau bounded probe đầu tiên, giữ dự báo trước ở trên:** P4 table diagnostic hơn own/shuffle,
nhưng entropy/log16≈0,963, F1≈0,228; P2 thiếu coverage vài lớp.
Ưu tiên tín hiệu P4 hơn P2, vẫn kiểm hai cỡ; chưa nâng dự báo cao hoặc hứa hơn Gaussian/DML.
[Chi tiết/giới hạn](../../proposals/boundary-cell-counts/design.md).

### ResponseRisk — chia sẻ trọng số consistency dưới stress receiver local

**Dự báo thấp/vừa so local; thấp/chưa rõ so Gaussian hoặc DML.**
Bằng chứng receiver shortcut mạnh ủng hộ giảm phụ thuộc noise/profile. Giữ input66; swap đồng bộ
9 feature cùng lớp/cùng client/khác receiver, record gốc và biến đổi đều ở xe.
Nhưng lợi augmentation local không đồng nghĩa lợi FL: pooled response weights phải hơn uniform/own/shuffle.

Fidelity extractor có thể kiểm derived closure, không tự chứng minh can thiệp vật lý/causal hoặc label invariance.
Nếu measured motion liên quan sensor-noise regime, block swap vẫn có thể tạo tổ hợp không thật.
JS nhỏ có thể do logits collapse. Tất cả xe cùng dùng shortcut thì consensus có thể củng cố sai chung.
Không mang scalar-only AFL/REx sang private khác model rồi gọi là tối ưu risk toàn cục.
[AFL](https://proceedings.mlr.press/v97/mohri19a.html).

Trước pilot: toy extractor, support/label-policy gate, số donor hợp lệ và probe responses trên dữ liệu
chưa dùng train teacher. Fail fidelity thì không train. Cost hai forward phải đo T4.
Ưu tiên first-order consistency trước penalty đạo hàm bậc hai.

Bounded fidelity mới:4toy extractor cases và40.951/40.960sample swaps giữ derived closure/policy.
Đây là điều kiện cần; chưa nâng dự báo utility. [Probe](../../papers/receiver-response/fidelity-probe.json).

### InputSensitivity — consensus gradient input-margin

**Dự báo thấp**, giữ backlog nghiên cứu. R2 có tọa độ input chung, nhưng mean/second gradient
không giữ vị trí ranh giới; consensus có thể củng cố shortcut chung. Binary/profile gradients đi ngoài support.
Mixed derivative có rủi ro cost/RAM. Không kế thừa theorem parameter-gradient của Fishr.
[Fishr](https://proceedings.mlr.press/v162/rame22a.html).
Chưa đủ cơ sở chi quota trước BoundaryCells/ResponseRisk.

### ErrorCosts — thống kê lỗi theo cặp lớp

**Thấp nếu đứng riêng; vừa có điều kiện khi bổ sung H1/H2.** Bảng chi phí có thể chỉnh ưu tiên lớp khó,
nhưng không truyền vị trí ranh giới peer. Dự báo macro recall có thể tăng đổi lấy precision/benign;
cần guard, không hứa tăng đồng thời 10 metrics. Không đếm là lời giải FL chính khi chỉ gửi scalar/cost.

## 4. Quy tắc triển khai từ dự báo

1. Tiếp continuation H1/H2 theo quota đã chốt; không thêm run dài cho adaptation FAIL.
2. Hybrid chỉ mở full sau audit đủ hai seed và paired utility PASS.
3. Fresh pilot EMA 2×2 + own; đo interaction để tách recipe local và lợi FL.
4. BoundaryCells: bounded train-only probe → lock quyết định gp/HP → smoke/resume/payload → Kaggle pilot.
5. ResponseRisk: fidelity trước; thống kê phải trên subset chưa train shadow teacher; fresh matched controls.
6. Hướng mới chỉ mở dài nếu utility PASS cả hai seed, benign guard và cost/bytes phù hợp.

Mỗi manifest cần ghi **ngày dự báo, baseline dự báo, bằng chứng ủng hộ/phản đối, điều kiện,
cổng bác bỏ, mức tin cậy và kết quả sau kiểm**. Giữ dự báo cũ khi cập nhật kết quả để kiểm chất lượng suy luận.
R1/R2 vẫn chưa được xác minh đầy đủ bằng pilot DAGSNet không giới hạn contact.

## 5. Cập nhật sau audit ngày 13/10, giữ dự báo ban đầu ở trên

- **Hybrid:** V8 paired utility PASS, hơn thành phần mạnh nhất +0,04070/+0,04355.
  Nâng mức tin cậy lợi pilot, đủ điều kiện kiểm full; interaction vẫn âm. Bốn notebook
  first3-of50 đã push, chưa kết quả full. Không ngoại suy tăng đơn điệu tới r50.
- **DML full:** tới r7, private/test/gp F1 0,705758; cùng round hơn own +0,045123.
  Có lợi cộng tác còn duy trì ở đoạn đã đo. Official test là mô tả, chưa r50/two-seed/R1/R2 proof.
- **Gaussian full:** tới r10, gp F1 0,610127, hơn own cùng round +0,048671;
  giảm từ đoạn đầu trong khi DML tăng. Hạ kỳ vọng “thêm round tự tăng metrics” cho moment tĩnh;
  chưa cô lập overfit/prior/BN làm nguyên nhân. Không thay run/HP theo test.
- **V9:** cả hai seed cost PASS, quality FAIL cho cả M64/M512. Hạ adaptation này xuống không ưu tiên;
  fit/refresh contribution không thay utility, không mở dài hoặc private KD.
- **BoundaryCells:** cả P/mode qua T4 cost 1,49–1,60× single-DAGS.
  R2 direct P4−single +0,00635/+0,00417; direct P2 đổi dấu; teacher thấp hơn single cả hai seed.
  Giữ dự báo thấp/chưa rõ so Gaussian mạnh; mở paired r20 với controls đã khóa.
  Không chọn riêng P4/seed hoặc sửa teacher theo kết quả calibration.

[10 metrics H1/H2 cuối](../../papers/full-validation/latest-rounds.md),
[calibration BoundaryCells](../../papers/boundary-cells/calibration-report.md).

## 6. Cập nhật audit full Hybrid ngày14/10

Giữ dự báo trước ở trên. Full100%train hiện có r2–r3/50, seed42-only:
Hybrid hơn DML cùngr2 +0,054897, hơn hybrid-own cùngr3 +0,054330; interactionr2 vẫn âm.
Tăng tin cậy tín hiệu đầu full-data, chưa tăng tin cậy độ bền tới r50 hoặc R1/R2.
Không so round khác nhau, không đổi HP/gp theo official test.

Hybrid/DML vẫn ưu tiên kiểm dài; Gaussian có lợi so own nhưng gp giảm trong đoạn đã đo,
không có cơ sở hứa tăng metrics bằng kéo dài đơn thuần. Cost Hybrid phải đọc từ actualT4:
controls khôngGaussian có worst-round cao hơn, hai kernel dừng budget nhưng giữ checkpoint hợp lệ.
Đây là hạn chế budget/calibration, không quality FAIL r50 hoặc numerical failure.
[Số đo và giới hạn](../../papers/hybrid-full/feasibility-20261014.md).

BoundaryCells vẫn costPASS/chưa utility; EMA cả hai nền và cổng r20 đã chốt/chưa source.
ResponseRisk quyết định mới nhất chỉlocalaugmentation trước; chia sẻ response-weight chưa mở pilot.
Không gọi componentlocal là phương pháp FL đã có lợi. [Phiên và việc tiếp](../operations/collection-2026-10-14.md).
