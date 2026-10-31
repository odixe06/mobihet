# V9 M512 — nguyên nhân thiếu lợi từ residual không gian

Không nối adaptation hiện tại. Hai seed r20 đã quality FAIL: group/private/gp FL hơn class-mean
chỉ +0,001596/+0,000877; khoảng thiếu so dense reference khoảng0,35F1.
[Kết quả utility đã audit](report.md) giữ toàn10metrics; chẩn đoán dưới đây không thay utility.

Restore chọn lọc config và resume r20 M512-FL hai seed, giữ manifest DVC gốc.
Đã kiểm fingerprint/round và toàn40source SHA khớp config. Không SGD, không official test,
không tìm bandwidth mới. Mỗi hold T/group lấy128record/lớp, seeded1729, tổng2.048record;
target/anchor dùng toàn512 từ checkpoint.
[Restore proof](../../../../../operations/restored-v9-forensics-2026-10-15.json),
[script](../../../../../../../archive/methods/anchor-function-residual/builds/anchor-residual/scripts/forensics_20261015.py),
[artifact](../../../../../../../archive/methods/anchor-function-residual/builds/anchor-residual/probes/forensics-20261015.json).

| chỉ dấu saved M512 r20 | seed42 | seed43 |
|---|---:|---:|
| Correction energy còn sau loại mean chung giữa anchors | 4,08% | 4,81% |
| P(nhãn bootstrap) trung bình tại anchor sau update | 94,99% | 95,16% |
| Anchor đổi argmax so bootstrap | 0/512 | 0/512 |
| Group normalized kernel entropy, 1=uniform | 0,9550 | 0,9559 |
| Group effective anchors /512 | 64,38% | 64,34% |
| Group nearest-anchor label accuracy | 26,76% | 27,78% |
| Group mean kernel mass đúng lớp | 9,12% | 9,10% |
| Group any-anchor coverage ở cutoff gốc | 99,17% | 99,07% |
| Group same-class-anchor coverage | 98,97% | 98,73% |
| Group kernel diagnostic F1 bootstrap→saved target | 0,24405→0,25425 | 0,26048→0,26321 |

Kernel diagnostic chỉ nội suy softmax target đã lưu, không là DAGSNet teacher hoặc private score.
Nó đổi prediction trên23,68%/15,63% group sample dù anchor argmax không đổi;
vì vậy label flip không phải điều kiện cần cho utility. Group không có row kernel underflow;
T có1/2.048row mỗi seed, ghi riêng thay vì để effective-anchor diagnostic vô hạn.

Kết quả hỗ trợ cơ chế failure: update chủ yếu là vector chung theo lớp, bootstrap95% vẫn chi phối,
kernel bandwidth đã khóa trộn khoảng330anchor/record nên residual mang ít thông tin ranh giới riêng từng anchor.
Coverage cao/zero-mass0 trong saved diagnostics bác cách giải thích đơn giản rằng không phủ được group.
Teacher fit MSE khoảng0,0017 vẫn đi cùng utility FAIL, nên giảm fit loss đơn thuần chưa có căn cứ cứu IDS.

Đây là chẩn đoán adaptation M512 đã khóa, không cô lập duy nhất BN/teacher-fit/extrapolation;
không chứng minh mọi residual/KME phương pháp đều bất khả thi. Không tune bandwidth/weight từ group,
không kéo dài hoặc đổi mục tiêu để tìm win. Muốn mở lại cần cơ chế mới tạo tín hiệu không gian rõ
và cổng trước trial; hiện chuyển quota sang DML/Hybrid và ClassTail, vốn không phụ thuộc teacher synthetic này.
