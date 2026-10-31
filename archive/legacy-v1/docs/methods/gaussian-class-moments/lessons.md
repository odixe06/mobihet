# Bài học — Gaussian từ moment lớp gộp

Hiệp phương sai input mang tri thức; tâm lớp riêng không đủ. Moment input không trôi theo backbone. Một lần bootstrap vẫn phải nhận đủ trước activation, không chia byte trung bình cho round để gọi khả thi.

Bằng chứng và giới hạn: [README](README.md). Kết luận chỉ áp dụng adaptation/cấu hình đã thử; không phủ định toàn họ phương pháp.

Full H1/H2 r10 vẫn hơn Gaussian-own cùng round≈+0,04867 gpF1, nhưng giảm so đoạn đầu.
Hybrid controls hai model cũng thấy Gaussian giảm r1→r3 dù hơn own. Moment tĩnh có lợi
không bảo đảm thêm epoch tăng metrics; chưa cô lập overfit/prior/BN. Không đổi HP/head theo test.
[10 metrics H1/H2](../shared-baselines/builds/federated-baselines-full/results/latest-rounds.md),
[Hybrid controls](../gaussian-proxy-hybrid/builds/hybrid-full/results/feasibility-20261014.md).
