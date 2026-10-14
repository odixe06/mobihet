"""Separate current operations from general training lessons after the approved layout change."""
import os
from pathlib import Path
import re

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])

CONTEXT='''# CONTEXT — MobiHet / federated IDS V2X

Đọc CONTEXT → [LESSONS](LESSONS.md) → [mục lục tài liệu](docs/README.md).
[Tất cả đề xuất/trạng thái](docs/proposals.md) · [Index từng file](docs/file-index.md).

## Bài toán và quyết định đang áp dụng

Xe học IDS cộng tác qua RSU: giao tiếp hữu hạn, private model có thể khác kiến trúc ở thiết kế.
Không raw/perturbed record rời xe; synthetic từ noise được phép. Giữ 66 feature, partition/protocol gốc.
Mọi learned model hiện DAGSNet; R2 đa kiến trúc chưa kiểm. Local chỉ smoke/probe/collect/audit/report;
train nặng trên Kaggle, W&B online/key trước train. Official test chỉ báo, không chọn HP/best round.

- **14/10, quyết định mới:** chuyển vật lý theo phương pháp/build; probe/calibration/pilot cùng build.
  Tài liệu đề xuất và kết quả vào docs; logs vật lý trong từng build/logs. Alias cũ giữ để tái lập.
  [Cấu trúc](docs/operations/project-layout.md), [manifest/hash](docs/operations/layout-migration-2026-10-14.json).
- Không tự Git/DVC add/commit/push/backup/cleanup. Được restore chọn lọc để phân tích hoặc dựng checkpoint input.
  Việc kiểm Git/DVC là read-only; workspace move chưa được commit.
- **14/10, đã cho phép:** dùng quota còn lại ngay để nối H1/H2/Hybrid và BoundaryCells; nối tiếp sau refresh khi cần.
  Giữ H1/H2/Hybrid 20client, 100%train80%, 50round, seed42 và đủ7controls, lịchLR50 từr1.
- BoundaryCells direct/teacher P2/P4 đã costPASS hai seed; utility20round giữ toàn bộ controls/gp target-mass/cổng đã chốt.
  ResponseRisk chỉ local augmentation trước; EMA cả Gaussian/DML giữ thiết kế/cổng, chưa triển khai.
- Đã ổn định thì ngừng health polling; vẫn collect/audit terminal. Bất ổn training quan sát được thì báo và yêu cầu chủ dự án dừng.

## Trạng thái — 14/10/2026

26 nhóm đề xuất +1recipe EMA; 3 nhóm có pilot tích cực: Gaussian moments, DML DAGSNet, Hybrid.
Không nhóm nào đủ r50/R1/R2 proof. Adaptation FAIL V2–V7/V9/V10 không rerun y nguyên.

Full H1/H2 đã audit r7–r11; Hybrid full r2/r3, 7arm/18round. Hai Hybrid calibration ERROR do budget/controller,
checkpoint vẫn hợp lệ. Batch continuation và BoundaryCells utility đã chuẩn bị/push; trạng thái exact version,
checkpoint bridge, quota và phần còn thiếu tại [bàn giao phiên này](docs/operations/reorganization-2026-10-14.md).
PUSHED/RUNNING không thay terminal artifact audit. Mọi số đo tại [hồ sơ build](docs/proposals.md).

## Lỗi/lưu ý bắt buộc trước tạo notebook

Đọc [sổ lỗi kernel](docs/operations/notebook-errors.md), dùng skill
[Codex](.agents/skills/federated-experiment-workflow/SKILL.md) hoặc
[Claude Code](.claude/skills/federated-experiment-workflow/SKILL.md).

- Kernel ERROR có thể sau nhiều arm/round hợp lệ: pull/audit trước rerun, nối actual committed round, không r1.
- Hybrid controller cũ đòi3round dù worker budget-stop rc0: controller mới chấp nhận boundary; phân biệt target50 với sessioncomplete.
- Output listing đọc được chưa chứng minh attach được parent ERROR. Kaggle có thể refused input mà push vẫn exit0;
  dùng dataset private checkpoint có SHA/import gate; require-resume trước data decode. Mất response thì kiểm status trước retry.
- Alias không giữ `__file__.resolve()` depth. Source frozen dựng đúng runtime tree; helper dùng root discovery.
  `git check-ignore` phải dùng đường dẫn canonical `.resolve()` trước nhúng key, nếu không lỗi beyond symbolic link.
- Fingerprint hash toàn bộ mh/*.py: không sửa/thêm module trong exact-resume hoặc đổi source run đã PUSHED.
- Metadata cả notebook/JSON, pinned image, assert2T4/sm75/torch trước train; không reinstall torch hoặc bỏ guard.
- W&B preflight không tự tăng restored step; key chỉ ignored private_push0600. Private platform history vẫn chứa key;
  pull executed vào private, audit redacted; không key trong dataset/log/Git/chat.
- Upload checkpoint dataset `-r zip -t`; kiểm file/hash/size, không tabular conversion hoặc mất thư mục.
- Budget cả batch: worst round từngarm, pairmax haiGPU, preparation/import/compile/eval/I/O và dư≥15%.
  Refresh service hiện báo17/10; không suy timezone chưa được API xác nhận. Phiên quota không là GPU-worker hours.
- Quota/listing network reset: retry read-only có giới hạn; UNKNOWN nếu chưa đọc được. Không tự lặp push.
- Verifier CPU map_location; CUDA deterministic/stable ties/CUBLAS trước CUDA; lỗi checker không yêu cầu train lại.
- DVC workspace thiếu sau cleanup không được add thay manifest đầy đủ. `deleted` cloud-status có thể là thiếu localcache;
  kiểm remote .dir/object riêng. DriveFS Errno5 có thể do cacheD: đầy, không tự là artifact nguồn hỏng.

## Muốn biết → mở

| nhu cầu | tài liệu |
|---|---|
| Toàn bộ đề xuất, thành công/thất bại/chưa kiểm | [docs/proposals](docs/proposals.md) |
| Căn cứ/cơ chế/build/kết quả/artifact | [Hồ sơ phương pháp](docs/methods/README.md) |
| Đọc lại 10 metrics các round cuối | [H1/H2](docs/methods/shared-baselines/builds/full-h1-h2-50rounds/results/latest-rounds.md), [Hybrid](docs/methods/gaussian-proxy-hybrid/builds/full-50rounds/results/latest-rounds.md) |
| Kernel/quota/việc còn thiếu | [Bàn giao hiện tại](docs/operations/reorganization-2026-10-14.md), [sổ lỗi](docs/operations/notebook-errors.md) |
| Bài học khiến train chưa tốt | [LESSONS](LESSONS.md), lessons từng phương pháp |
| Dữ liệu/kiến trúc/scaler | [knowledge](knowledge/README.md), [protocol](docs/research/protocol.md) |
| Scripts dùng chung, cấu trúc, khôi phục | [scripts](scripts/README.md), [layout](docs/operations/project-layout.md), [backup/restore](docs/operations/backup.md) |

Cập nhật trạng thái/lỗi ở CONTEXT, bài học chất lượng train ở LESSONS; số đo chi tiết ở docs của build.
Sau thêm/chuyển file chạy index và kiểm link/hash/ignore. Không backup hoặc dọn artifact nếu chưa được yêu cầu.
'''

LESSONS='''# LESSONS — bài học chung về huấn luyện MobiHet

Đọc sau [CONTEXT](CONTEXT.md). Lỗi/quy tắc vận hành nằm ở CONTEXT và [sổ lỗi kernel](docs/operations/notebook-errors.md).
Chi tiết từng adaptation ở [hồ sơ phương pháp](docs/proposals.md); mọi con số phải trỏ báo cáo đã audit.
[Bản đầy đủ trước tổ chức](docs/legacy/lessons-before-layout-20261014.md) giữ lịch sử, không xóa im lặng bài học bị bác bỏ.

## Tri thức nào cải thiện IDS

- Nút thắt là ranh giới đa receiver: private dễ học thuộc receiver của mình. Head-only/prototype trung bình lớp
  không chở đủ ranh giới; cần tác động backbone và so đối chứng quần thể/own. [Bằng chứng kế thừa](docs/legacy/kien-thuc-ke-thua.md).
- Gaussian input covariance mang tri thức quần thể: own Gaussian chỉ giải thích phần nhỏ lợi; diag/lr/full giữ cấu trúc
  khác nhau. Input không trôi theo encoder. Mean lớp hoặc biến đổi mean nhiều lần không thay covariance.
  [Gaussian moments](docs/methods/gaussian-class-moments/README.md).
- Latent statistics cũ đi khi backbone còn học: offline probe tích cực không bảo đảm loss online có lợi.
  Snapshot/head đóng băng là phạm vi khác, phải kiểm riêng. [Bài học kế thừa](docs/legacy/kien-thuc-ke-thua.md).
- Teacher phải đủ chất lượng. Ensemble local/teacher yếu có thể làm nhãn Gaussian miễn phí xấu đi;
  có teacher gate và class-mean control trước KD. Proxy DAGSNet có sức chứa tốt hơn proxy yếu trong P1.
- Ít byte không tự giữ sức chứa: seeded subspace có tối đaK bậc tự do chung. Compression contribution PASS
  không thay retention/utility PASS. [Subspace](docs/methods/proxy-seeded-subspace/README.md), [carriers](docs/methods/proxy-compression/README.md).
- Synthesis loss giảm, codec/đạo hàm PASS không chứng minh IDS tốt hơn. KME/DM/gradientbooks đã trượt utility;
  coverage cao có thể đi cùng soft target gần uniform. [Synthetic knowledge](docs/methods/synthetic-knowledge/README.md).

## Những cơ chế làm train kém hoặc gây kết luận sai

- Sketch-space error feedback không mặc định co: collision có thể khuếch đại residual và parameter rồi BN overflow.
  Đo decoded/residual norm, tính co và full-epoch stability; không suy toàn họ FetchSGD/EF bị bác bỏ từ adaptation này.
- Gaussian liên tục và real có feature atom/missing có thể khác support; ratio fit cao không chứng minh ratio hữu ích.
  Clip/class-normalize không sửa support hoặc tạo ESS tốt. Clip floor có thể triệt input score correction;
  finite burn-in không chứng minh MALA mixing. [Ratio](docs/methods/gaussian-density-ratio/README.md), [MALA](docs/methods/gaussian-mala-transport/README.md).
- Support/fidelity PASS không tự tăng F1. Joint mask có lợi tương tự control full66 chưa chứng minh geometry mechanism.
  Khoảng sai số đại số rộng không chứng minh faithful reconstruction, nhất là missing/coincident và denominator gần0.
  [Geometry](docs/methods/geometry-primitive-moments/README.md).
- DAGSNet bounded-logit classifier trên R66 không tự là density chuẩn hóa: cần base/domain rõ ràng.
  Gaussian-base sửa tính hợp lệ nhưng không tự sửa sampler quality/runtime. [Energy](docs/methods/dagsnet-joint-energy/README.md).
- gp theo prior local có thể sửa quá tay khi KD làm private gần global; benign F1 có thể giảm dù head tăng.
  Định nghĩa prior hiệu dụng từ loss trước run, primary rule cố định. BoundaryCells khóa target-mass gp;
  các full cũ giữ gp đã chốt, không đổi headline để cứu cổng.
- Tổng CEreal+CEsynthetic cao chưa chứng minh real CE cao hoặc đã chống học thuộc. Tách loss và dùng holdout.
  Ngược lại, loss FL tăng khi LR gần0 có thể vì weights tải xuống không được fit thêm, không tự là trôi.
- EMA không gồm toàn head; BN phải khớp backbone EMA và head triển khai fit riêng. Recipe local có lợi chưa là lợi FL:
  so 2×2 +own, interaction báo riêng. [EMA thiết kế](docs/methods/ema-deployment-recipe/design.md).
- Hybrid tốt hơn thành phần mạnh nhất nhưng interaction âm; cộng hai cơ chế không có nghĩa synergy.
  Full dữ liệu sớm có tín hiệu tốt, chưa r50. Gaussian đã giảm metric qua một số round dù còn lợi so own;
  thêm epoch không tự tăng performance. [Hybrid full](docs/methods/gaussian-proxy-hybrid/builds/full-50rounds/results/feasibility-20261014.md).
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
  deadline-cut trả phí nhưng không commit, chỉ activate snapshot coherent. [Contact](docs/methods/contact-fragmentation/README.md).
- R1 giới hạn giao tiếp và R2 khác kiến trúc phải kiểm riêng; unlimited-channel/DAGSNet-only PASS không giải quyết cả hai.
  Synthetic exception không tự là secure aggregation hoặc DP.

Khi bằng chứng mới bác bỏ bài học: sửa rõ phạm vi và dẫn artifact đã audit, giữ bản lịch sử. Cải tiến mới phải dự báo
performance từ bằng chứng, chỉ ra dấu hiệu có thể sai rồi kiểm ngắn trước khi tăng quy mô.
'''


def main():
    archive=ROOT/'docs/legacy/lessons-before-layout-20261014.md'
    if not archive.exists():
        original=(ROOT/'LESSONS.md').read_text()
        def fix(match):
            target=match[1]
            if target.startswith(('http','#','mailto','app:')):return match[0]
            path,sep,anchor=target.partition('#')
            return ']('+os.path.relpath(ROOT/path,archive.parent)+(sep+anchor if sep else '')+')'
        archive.write_text('# Snapshot lịch sử — trước tổ chức14/10\n\n'
            'Quyết định mới nhất ở [CONTEXT](../../CONTEXT.md), bài học chung hiện tại ở [LESSONS](../../LESSONS.md).\n\n'+
            re.sub(r'\]\(([^)\n]+)\)',fix,original))
    (ROOT/'CONTEXT.md').write_text(CONTEXT)
    (ROOT/'LESSONS.md').write_text(LESSONS)
    print('CONTEXT: current operations; LESSONS: training insights; prior full text retained')


if __name__=='__main__':main()
