# LESSONS — bài học chung về huấn luyện MobiHet

Đọc sau [CONTEXT](CONTEXT.md). Lỗi/quy tắc vận hành nằm ở CONTEXT và [sổ lỗi kernel](docs/operations/notebook-errors.md).
Bài học riêng còn dùng ở [docs/lessons](docs/lessons/README.md); bằng chứng chi tiết ở [archive](docs/archive/README.md); mọi con số phải trỏ báo cáo đã audit.
[Bản đầy đủ trước tổ chức](docs/archive/legacy/lessons-before-layout-20261014.md) giữ lịch sử, không xóa im lặng bài học bị bác bỏ.

## Tri thức nào cải thiện IDS

- Nút thắt là ranh giới đa receiver: private dễ học thuộc receiver của mình. Head-only/prototype trung bình lớp
  không chở đủ ranh giới; cần tác động backbone và so đối chứng quần thể/own. [Bằng chứng kế thừa](docs/archive/legacy/kien-thuc-ke-thua.md).
- Gaussian input covariance mang tri thức quần thể: own Gaussian chỉ giải thích phần nhỏ lợi; diag/lr/full giữ cấu trúc
  khác nhau. Input không trôi theo encoder. Mean lớp hoặc biến đổi mean nhiều lần không thay covariance.
  [Gaussian moments](docs/methods/gaussian-class-moments/README.md).
- Latent statistics cũ đi khi backbone còn học: offline probe tích cực không bảo đảm loss online có lợi.
  Snapshot/head đóng băng là phạm vi khác, phải kiểm riêng. [Bài học kế thừa](docs/archive/legacy/kien-thuc-ke-thua.md).
- Teacher phải đủ chất lượng. Ensemble local/teacher yếu có thể làm nhãn Gaussian miễn phí xấu đi;
  có teacher gate và class-mean control trước KD. Proxy DAGSNet có sức chứa tốt hơn proxy yếu trong P1.
- Ít byte không tự giữ sức chứa: seeded subspace có tối đaK bậc tự do chung. Compression contribution PASS
  không thay retention/utility PASS. [Subspace](docs/archive/methods/proxy-seeded-subspace/README.md), [carriers](docs/archive/methods/proxy-compression/README.md).
- Synthesis loss giảm, codec/đạo hàm PASS không chứng minh IDS tốt hơn. KME/DM/gradientbooks đã trượt utility;
  coverage cao có thể đi cùng soft target gần uniform. [Synthetic knowledge](docs/archive/methods/synthetic-knowledge/README.md).

## Những cơ chế làm train kém hoặc gây kết luận sai

- Sketch-space error feedback không mặc định co: collision có thể khuếch đại residual và parameter rồi BN overflow.
  Đo decoded/residual norm, tính co và full-epoch stability; không suy toàn họ FetchSGD/EF bị bác bỏ từ adaptation này.
- Gaussian liên tục và real có feature atom/missing có thể khác support; ratio fit cao không chứng minh ratio hữu ích.
  Clip/class-normalize không sửa support hoặc tạo ESS tốt. Clip floor có thể triệt input score correction;
  finite burn-in không chứng minh MALA mixing. [Ratio](docs/archive/methods/gaussian-density-ratio/README.md), [MALA](docs/archive/methods/gaussian-mala-transport/README.md).
- Support/fidelity PASS không tự tăng F1. Joint mask có lợi tương tự control full66 chưa chứng minh geometry mechanism.
  Khoảng sai số đại số rộng không chứng minh faithful reconstruction, nhất là missing/coincident và denominator gần0.
  [Geometry](docs/archive/methods/geometry-primitive-moments/README.md).
- DAGSNet bounded-logit classifier trên R66 không tự là density chuẩn hóa: cần base/domain rõ ràng.
  Gaussian-base sửa tính hợp lệ nhưng không tự sửa sampler quality/runtime. [Energy](docs/archive/methods/dagsnet-joint-energy/README.md).
- gp theo prior local có thể sửa quá tay khi KD làm private gần global; benign F1 có thể giảm dù head tăng.
  Định nghĩa prior hiệu dụng từ loss trước run, primary rule cố định. BoundaryCells khóa target-mass gp;
  các full cũ giữ gp đã chốt, không đổi headline để cứu cổng.
- Tổng CEreal+CEsynthetic cao chưa chứng minh real CE cao hoặc đã chống học thuộc. Tách loss và dùng holdout.
  Ngược lại, loss FL tăng khi LR gần0 có thể vì weights tải xuống không được fit thêm, không tự là trôi.
- EMA không gồm toàn head; BN phải khớp backbone EMA và head triển khai fit riêng. Recipe local có lợi chưa là lợi FL:
  so 2×2 +own, interaction báo riêng. [EMA thiết kế](docs/methods/ema-deployment-recipe/design.md).
- Hybrid tốt hơn thành phần mạnh nhất nhưng interaction âm; cộng hai cơ chế không có nghĩa synergy.
  Full dữ liệu sớm có tín hiệu tốt, chưa r50. Gaussian đã giảm metric qua một số round dù còn lợi so own;
  thêm epoch không tự tăng performance. [Hybrid full](docs/methods/gaussian-proxy-hybrid/builds/hybrid-full/results/feasibility-20261014.md).
- Cùng số model/SGD step không bảo đảm cùng walltime: dummy/eval-BN có overhead thật.
  Không dùng dummy-inflated denominator để gọi phương pháp rẻ; chưa profiling thì không chốt nguyên nhân duy nhất.

## Đánh giá để rút đúng bài học

- Train/partition đã z-score, test raw áp scaler đúng một lần. Dữ liệu sắp theo lớp: sample ngẫu nhiên seeded,
  không prefix tùy tiện. Không chia lại partition hoặc dùng official test chọn HP/round.
- Holdout ngẫu nhiên theo dòng lạc quan hơn temporal/receiver-group. Receiver noise/profile và Sybil first-session
  là shortcut; benign từ runSybil có source bias. [Dataset](knowledge/dataset.md).
- positionMirroring chưa chạm trần thông tin; Gaussian/DML private chưa học được không có nghĩa dữ liệu vô ích.
  timeDelay thiếu dấu hiệu trong content theo generator: kiểm trần trước quy lỗi thuật toán.
- Probe phải dùng dữ liệu chưa train và đủ lớp/cohort. Local derivative/coverage/inference prefix là điều kiện cần,
  không là utility, full coverage, leakage hoặc physical consistency proof.
- Khóa gp/cổng/seed/controls trước chạy, so cùng round và một yếu tố mỗi arm. Mean-client metrics khác pooled confusion;
  lưu predictions, per-client CM, đủ10metric/per-class/benign để kiểm lại.
- Pilot ngắn không dự báo độ bền; full thường giảm lợi hoặc đổi dấu. Không kéo dài adaptation FAIL để tìm điểm thắng.
  Full một seed là xác nhận mô tả theo protocol, không paired confidence interval.
- Lợi mutual learning khi chưa nhận snapshot RSU không được gọi là lợi FL. Tính đủ bootstrap/prior/header/ACK/retransmission;
  deadline-cut trả phí nhưng không commit, chỉ activate snapshot coherent. [Contact](docs/archive/methods/contact-fragmentation/README.md).
- R1 giới hạn giao tiếp và R2 khác kiến trúc phải kiểm riêng; unlimited-channel/DAGSNet-only PASS không giải quyết cả hai.
  Synthetic exception không tự là secure aggregation hoặc DP.

Khi bằng chứng mới bác bỏ bài học: sửa rõ phạm vi và dẫn artifact đã audit, giữ bản lịch sử. Cải tiến mới phải dự báo
performance từ bằng chứng, chỉ ra dấu hiệu có thể sai rồi kiểm ngắn trước khi tăng quy mô.
