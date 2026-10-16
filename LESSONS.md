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

- **Tri thức quần thể nằm trong proxy chung, không trong private** (khi đánh giá trên test toàn cục). DML proxy head hơn private gp
  +0,115/+0,122 group pilot và +0,056 full r20; trộn private vào proxy (ProxyFuse) **kém** proxy −0,032/−0,034,
  private đúng mà proxy sai chỉ ≈2% dòng. Owner15/10: proxy là headline DML/Hybrid. Cải tiến nên nhắm proxy
  (model triển khai) hoặc khả năng hấp thụ, không chỉ private. [ProxyFuse](docs/methods/proxy-fusion/results.md).
- **Kênh thống kê vô hướng ≈ local.** ClassTail, class-mass, CCR pooled/own/shuffle/fixed/classweight đều trong
  ±0,002 của CE local ở pilot group r20; Gaussian (+0,08) và proxy dense (+0,19–0,31) mới mang ranh giới.
  Không mở thêm kênh scalar/score-moment nếu không chở thông tin hàm/input mới.
- **Chống shortcut receiver giúp model chỉ thấy dữ liệu local; ở pilot không giúp proxy chung** (⚠ full vòng 3 bác bỏ phần proxy, xem bài dưới). AntiMem pilot r20 hai seed:
  Gaussian private gp m4b +0,025, swappos +0,016–0,018, swap +0,013, private_ema +0,035–0,039 (kết hợp +0,056–0,058);
  DML/Hybrid proxy head mọi kỹ thuật −0,012…+0,006, proxy EMA −0,028…−0,038. Proxy đã gộp 20 client nên shortcut
  không còn là nút thắt; muốn cải thiện headline phải nhắm tối ưu/tổng hợp proxy. [Kết quả](docs/methods/anti-memorization/results.md).
- **Proxy dense thừa độ chính xác.** QProxy q4 (int4 theo hàng, EF phía server) = dense ±0,001 proxy và private
  hai seed, byte nhỏ 7,65×. Nén theo tọa độ có chặn lỗi giữ được utility, khác sketch/subspace đã FAIL.
- **Phạm vi byte q4 — kiểm source 17/10:** tỷ lệ7,65× trên là bộ đếm codec dự kiến, chưa phải byte wire end-to-end.
  FullDrift `quantize` đếm4bit nhưng `latest_packet` ghi `q` int8 chưa pack nibble; buffer/counter được đếm nhưng
  packet nhánh này chưa serialize đủ, upload cộng tensor decoded trực tiếp. Source khớp artifact vòng5 đã audit.
  Utility lượng tử hóa vẫn có bằng chứng; R1 cần codec pack/unpack thật, tính `len(packet)` cùng overhead và
  decode/commit qua contact. [Chi tiết và source](docs/research/algorithm-improvement-plan-2026-10-17.md#31-q4-utility-đã-kiểm-wire-bytes-chưa-được-chứng-minh-đầy-đủ).
- **Drift là nút thắt của proxy ở full; tổng hợp thường hơn giúp, regularize thì không.** FullDrift seed42 ep6 group:
  4 lần tổng hợp/epoch (cùng compute/LR) DML proxy +0,026, FedAvg +0,034 (ep1: 0,751 so 0,187). FedProx μ0,01 giảm
  drift 5× mà F1 không đổi (benign +0,098); FedAvgM β0,9 phân kỳ. FedAvg thuần ≈ DML proxy (+0,009).
  Gaussian swappos ở full +0,052, ngừng suy giảm sau r1. [Kết quả](docs/methods/full-drift/results.md).

- **Ở full, bỏ khối feature receiver giúp cả client lẫn proxy DML.** FullDrift vòng 3 seed42 ep6 group: m4b private gp
  +0,036, proxy +0,052, benign +0,11; swappos +0,033/+0,049. 0 byte/contact thêm (R1), chỉ phía client (R2). Pilot r20
  từng FAIL proxy ⇒ học thuộc receiver chỉ lộ khi mỗi epoch ~4.200 bước. FedNTD hại private (−0,042): khoảng private–proxy
  không do lệch prior. q4 ở 1 contact/epoch không thua. [Kết quả](docs/methods/full-drift/results.md).
- **Học thuộc receiver là lý do proxy DML chững và benign sụp theo epoch ở full** (vòng 5, ep15): `fd_dml` proxy phẳng
  0,763 từ ep4, benign proxy 0,74→0,36; m4b proxy còn tăng 0,815→0,851, benign giữ ~0,77, lợi lớn dần (+0,052 ep6 →
  +0,086 ep15; private +0,036). Plateau không phải hết capacity hay LR; kéo dài run không bỏ shortcut thì vô ích.
  Gaussian m4b +0,067. Kỹ thuật cộng được: m4b + q4 (±0,000) + sub4 (private +0,031). Seed 43 đang kiểm.
- **Đổi kênh KD không đóng khoảng private–proxy; đổi input phía client thì có.** FedNTD −0,042, hidden-state KD (FedKD,
  projector local) +0,006, cùng kênh KD m4b +0,036. Trước khi thêm kênh/loss chưng cất, kiểm private có học shortcut local không.
- **Khi đã tổng hợp thường (sub4), FedProx chỉ làm chậm học local** (proxy −0,008, private −0,016). q4 ở sub4 làm benign
  −0,014…−0,016 (lỗi lượng tử cộng theo số lần tổng hợp), ở 1 contact/epoch thì không. sub16_q4 thêm +0,041 proxy/+0,017
  private nhưng 16 contact/epoch: R1 cần mô phỏng contact thực tế trước khi chọn tần suất (owner 16/10).

## Những cơ chế làm train kém hoặc gây kết luận sai

- **Full ≠ pilot về động học.** Full mỗi round = 1 epoch ≈4.200 bước/client; Gaussian/local đạt đỉnh test ở r1 rồi giảm
  (học thuộc receiver), proxy bão hòa từ r3–r5 khi LR còn ~1e-3 (FullDrift vòng 5: cũng do học thuộc receiver, m4b bỏ được). Pilot2,5% (~100 bước) không tái hiện; kết luận về
  drift/hội tụ từ pilot (vd FedAvg > DML proxy +0,027) có thể đảo ở full (repo anh em NILM proxy 0,823 > AFPHA 0,775).
  Macro F1 còn bị khối {benign, dataReplay, positionMirroring, timeDelay} chặn; lỗi dịch chuyển trong khối.
  [Chẩn đoán](docs/research/plateau-diagnosis-2026-10-15.md).

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
- **"Proxy hơn private" chỉ đúng khi mọi client đánh giá trên cùng test toàn cục.** Đánh trọng số lại confusion theo
  prior nhãn của từng client (full DML r23): private head local macro F1 0,677 ≈ proxy 0,675, local accuracy 0,818 so 0,771;
  gp tốt cho global nhưng kém cho local. Trước khi so phương pháp "cho client", chốt định nghĩa metric (global/local)
  và báo client tệ nhất. Mới mô phỏng lệch nhãn, chưa lệch receiver. [Bảng](docs/research/client-model-directions-2026-10-16.md).
  gp có thể làm hỏng client tệ nhất ở góc local: sub16_q4 private gp local acc min 0,288 (head 0,673). Góc local dùng head.
- Khóa gp/cổng/seed/controls trước chạy, so cùng round và một yếu tố mỗi arm. Mean-client metrics khác pooled confusion;
  lưu predictions, per-client CM, đủ10metric/per-class/benign để kiểm lại.
- CrossClientRank CE-r2: residual/delta macro cực lớn có thể do class delta gần0 và fit/test extrapolation;
  không mặc định SVD lỗi. Diagnostic cùng mẫu tái hiện energies, condition≈5,2; class5/client17 có coefficient lớn
  và vài test record chi phối. Probe gate PASS không định lượng tri thức hoặc utility; báo energy tuyệt đối,
  singular values và class influence trước khi diễn giải. [Probe/diagnostic đã audit](docs/methods/cross-client-ranking/builds/rank-probe/results/report.md).
- Reference cùng recipe/seed/data ở build khác là cùng phép tính: full DML/DML-own trùng bit-exact giữa hai build,
  Gaussian/local lệch ≤0,008. Tái dùng audited reference thay vì chạy lại; pilot DML không bit-exact giữa build (≈0,002)
  nên gate cần chênh ≥0,01 mới nên dựa reference khác build. [Sổ lỗi](docs/operations/notebook-errors.md).
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
