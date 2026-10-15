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
- V9 M512 r20: 95–96% correction energy chung giữa anchors; bootstrap vẫn khoảng95%, kernel group entropy0,955
  dù coverage>99%. Coverage/teacher-fit loss thấp không bảo đảm chở ranh giới; không kéo dài adaptation này.
  Chẩn đoán saved-target không cô lập toàn bộ BN/fit/extrapolation, không bác cả họ residual.
  [Forensics hai seed](docs/archive/methods/anchor-function-residual/builds/anchor-residual/results/forensics-20261015.md).

## Những cơ chế làm train kém hoặc gây kết luận sai

- Sketch-space error feedback không mặc định co: collision có thể khuếch đại residual và parameter rồi BN overflow.
  Đo decoded/residual norm, tính co và full-epoch stability; không suy toàn họ FetchSGD/EF bị bác bỏ từ adaptation này.
  Repro15/10 tối giản CPU FP32 cho residual2×, không cần BN/AMP. Line-search kiểm **wire-decoded** patch
  chặn overshoot trong80ca đại số. Repaired hai seed terminal: smallr20 vẫn qualityFAIL;
  mediumr17/r14 còn thiếu horizon, last5 group/gp phẳng/giảm và xa own. Decoded patch chỉ đổi0,638%/7,818%
  tọa độ small/medium, không đủ để quy lỗi duy nhất cho collision/ties. Co norm không chứng minh teacher
  fidelity hoặc giữ ranh giới. [Sửa, forensic và cổng cũ](docs/methods/proxy-countsketch-repair/builds/contractive-ef-pilot/results/report.md).
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
- Tail-risk bằng histogram có lỗi lượng tử hóa: quantile ở mép bin có thể làm tail rỗng và chỉ đổi scale loss.
  Kiểm tail fraction/membership thực tế, overflow và cohort trước utility; weighted target mass cần audit nhưng
  x-dependent weighting không tự có bảo đảm label-shift. ClassTail cả hai seed: probe r2PASS nhưng utilityr20FAIL;
  pooled không hơn local/own/shuffle/class-mass, kém DML0,1967/0,1917. Probe chỉ loại tail rỗng/calibration hỏng,
  không chứng minh hard-example knowledge hữu ích. [Gates đã audit](docs/methods/class-tail-risk/builds/class-tail-pilot/results/gates.json).
- BoundaryCells P2 phủ95,65% group probe nhưng target entropy0,915/table F1≈0,129; coverage không thay
  discrimination. Target tĩnh không nhận tri thức mới khi thêm round. P4direct/P2-P4teacher r20 hai seed cũng FAIL
  so Gaussian; learned teacher head≈0,144–0,260 phù hợp table yếu. Không nối cả bốn adaptation;
  không suy cả họ bị bác. [Actual payload forensics](docs/methods/boundary-cell-counts/builds/boundary-cell-pilot/direct-target-forensics-20261015.md).
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
- CrossClientRank CE-r2: residual/delta macro cực lớn có thể do class delta gần0 và fit/test extrapolation;
  không mặc định SVD lỗi. Diagnostic cùng mẫu tái hiện energies, condition≈5,2; class5/client17 có coefficient lớn
  và vài test record chi phối. Probe gate PASS không định lượng tri thức hoặc utility; báo energy tuyệt đối,
  singular values và class influence trước khi diễn giải. [Probe/diagnostic đã audit](docs/methods/cross-client-ranking/builds/rank-probe/results/report.md).
- Pilot ngắn không dự báo độ bền; full thường giảm lợi hoặc đổi dấu. Không kéo dài adaptation FAIL để tìm điểm thắng.
  Full một seed là xác nhận mô tả theo protocol, không paired confidence interval.
- Exact-resume fixture phải cover state của cả local/control và FL arm. CrossClientRank pooled replay PASS
  đã bỏ sót CE empty-byte state, khiến production verifier dừng dù r3 hợp lệ;
  [repro và recovery](docs/operations/notebook-errors.md) giữ nguyên weights/source, không cần SGD lại.
- Lợi mutual learning khi chưa nhận snapshot RSU không được gọi là lợi FL. Tính đủ bootstrap/prior/header/ACK/retransmission;
  deadline-cut trả phí nhưng không commit, chỉ activate snapshot coherent. [Contact](docs/archive/methods/contact-fragmentation/README.md).
- R1 giới hạn giao tiếp và R2 khác kiến trúc phải kiểm riêng; unlimited-channel/DAGSNet-only PASS không giải quyết cả hai.
  Synthetic exception không tự là secure aggregation hoặc DP.

Khi bằng chứng mới bác bỏ bài học: sửa rõ phạm vi và dẫn artifact đã audit, giữ bản lịch sử. Cải tiến mới phải dự báo
performance từ bằng chứng, chỉ ra dấu hiệu có thể sai rồi kiểm ngắn trước khi tăng quy mô.
