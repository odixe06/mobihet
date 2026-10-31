# Bàn giao nghiên cứu synthetic — dừng để compact, 22/10/2026 ~03:00Z

**Snapshot lịch sử trước compact.** Owner đã tiếp tục, P29–31 và report final đã xong;
đọc [bàn giao hiện hành](continuation-2026-10-22.md) cho trạng thái/bước tiếp.

**Owner yêu cầu tổng hợp tài liệu, dừng phiên, compact trước mọi task tiếp theo.** Không mở nghiên cứu, launch,
poll hay tối ưu mới trong bước đóng phiên. Chỉ tiếp tục khi owner yêu cầu. Không có local probe/collector của luồng này còn chạy;
subagent nghiên cứu đã kết thúc. Trạng thái remote của luồng chấm final độc lập chưa kiểm lại trong phiên nghiên cứu này.

## Đọc lại sau compact

1. [CONTEXT](../../../CONTEXT.md) → [LESSONS](../../../LESSONS.md) → trang này.
2. [Hồ sơ kết quả/cổng](../../research/methods/synthetic-learning-recovery-2026-10-22.md): nguồn chuẩn cho mọi số đo Probe23–28.
3. [Audit build cũ](../../research/methods/synthetic-build-history-audit-2026-10-22.md): duplicate/veto, control đúng, lỗi không lặp.
4. [Nguồn primary + loss/client/r50](../../research/methods/ccvr-budget-primary-sources-2026-10-22.md): phân tích sâu và giới hạn suy luận.
5. Luồng final50 có bàn giao riêng ở [continue](../../operations/continue.md); snapshot đó không thay kiểm live khi owner muốn nối lại.

## Yêu cầu đã chốt

- Mục tiêu: synthetic giúp **private learning**, giữ model client độc lập, không chỉ teacher-fit/acceptance đẹp.
- Giữ dataset/66 feature/partition/scaler; không raw/perturbed record hoặc per-record activation/gradient rời client.
- R1: tổng byte/client ≤1,5× `xa` **cả 15 và 50 round**; ≤231.000 B/contact **kể cả bootstrap**.
  Owner cho tìm scheme tối ưu/nhiều phương án; chưa chốt contact schedule mới. Không coi thêm contact miễn phí.
- Owner đã cho thử local/Kaggle sau đủ bằng chứng; yêu cầu notebook nhanh, hỏi khi chưa rõ yêu cầu, báo việc tay.
  Yêu cầu dừng hiện tại có ưu tiên; không dùng quyền launch trước đó để mở task trong lúc compact.
- Không chọn HP/checkpoint bằng official test. R50 hiện dùng mô tả, old/group là development đã dùng nhiều lần.
- Phương pháp chốt vẫn `gkd + la + kdpx`, proxy `q3_zl`. `xo`, `cvl` giữ FAIL; không nối adaptation thất bại y nguyên.

## Kết luận đã đủ chứng cứ

| Phép kiểm | Kết luận / phạm vi |
|---|---|
| P23 CCVR Project48 | Codec calibration nhỏ hơn full~26,2×; teacher gain PASS hai seed Kaggle. Chưa phải online/R1 PASS. |
| P24 teacher×support, đủ20 client | Teacher transfer FAIL cả hai seed; inversion riêng FAIL gain; hard label làm benign kém. Không launch online. |
| P26 class inclusion weights | Benign/worst cải thiện, gain chưa đạt; local s42, dùng population fixture. Prior-only đã viết **chưa chạy**. |
| P27 class-matched pooled real-input oracle | Cùng teacher tốt: real support vượt synthetic rõ; local s42, chỉ chẩn đoán **R3-ineligible**, không deploy. |
| P28 shared-head gradient + curvature | Aggregate khớp autograd, safe step giảm train CE nhưng teacher gate FAIL; local s42, không launch. |
| Final50 cũ | Proxy tiếp tục tăng, private tăng chậm; LR/AMP/one-round lag không đủ giải thích. Chưa có một căn nguyên duy nhất được chứng minh. |

Không coi5 client đầu là cohort, không coi component agreement là ground-truth accuracy.
Không dùng offline teacher gain thay private benign/worst gates hoặc ep15/horizon/R2 proof.
Weighting là yếu tố góp phần; support oracle là can thiệp trên frozen heads, không chứng minh online final50 đã được sửa.

## Kaggle hoàn tất và audit

- Kernel private version1: [`thuyanh316/mobihet-fd-ccvr-probes-20261022-v1`](https://www.kaggle.com/code/thuyanh316/mobihet-fd-ccvr-probes-20261022-v1).
- Hai T4, hai seed song song, mỗi seed P23→P24 đủ20 client; input source/fixture/weights cache, image pin runtime.
  Feature cache theo private/source, chỉ fit head; không compile backbone cho probe ngắn. Tổng execution~360s.
- [AUDITED.json](../../../methods/full-drift/builds/fd-probe/artifacts/kaggle/fd-ccvr-probes-20261022-v1/AUDITED.json):
 59 file kéo đủ, exact source/executed/input hashes,400 lượt prediction→CM→10 metric; teacher PASS, transfer/inversion FAIL cả42/43.
- [Notebook/source manifest/PUSHED/executed redacted](../../../methods/full-drift/builds/fd-probe/notebooks/fd-ccvr-probes-20261022-v1/).
  [Input stage bất biến](../../../methods/full-drift/builds/fd-probe/artifacts/input/fd-ccvr-probes-20261022-v1/).
- Dataset private `thuyanh316/mobihet-ccvr-probes-20261022-v1`; W&B locator
  [`21522798-uit/mobihet-fulldrift/jl1umyh6`](https://wandb.ai/21522798-uit/mobihet-fulldrift/runs/jl1umyh6).
  Terminal artifacts là bằng chứng, không phải status dashboard. Không cần poll lại probe đã audit.
- Private inline notebook ở `methods/full-drift/builds/fd-probe/private_push/fd-ccvr-probes-20261022-v1/`, ignored0600.
  Không in key/copy sang public source. Global login vẫn thuyanh316, thao tác account-scoped. Quota phải đọc live trước push mới.

## Source và artifact để tái lập

Code mới: [fd-probe/source/proj](../../../methods/full-drift/builds/fd-probe/source/proj/)
`fd_probe23.py`, `fd_probe24.py`, `fd_probe26.py`, `fd_probe27.py`, `fd_probe28.py`;
[scripts](../../../methods/full-drift/builds/fd-probe/scripts/) `gen_ccvr_probe.py`, `check_ccvr_probes.py`, `collect_ccvr_probes.py`.
Không đổi production/frozen `fd.py`; source đã push phải giữ nguyên, sửa bằng tag mới.

Fixture local/upload cùng bytes: `archive/methods/shared-baselines/builds/pilot-baselines/artifacts/fixture`.
Train sample2,5%, old/group65.536 mỗi tập, không official test; manifest/full train counts được phân biệt trong dossier.
Weights dùng round015 của `fd-s42xa-20261015-r22-v1` và `fd-s43ya-20261015-r23-v2`,
trong `methods/full-drift/builds/fd-probe/artifacts/kaggle/<rid>/runs/<arm>/weights/round_015.pt`.

JSON/NPZ local ở [full-drift/results](../../methods/full-drift/results/): `probe23-xa-s42-e15`,
`probe24-xa-s42-e15-all` + `.joint.json`, `probe-calibration-priors`,
`probe26-classweight-teacher-s42`, `probe26-classweight-transfer-s42`,
`probe27-real-support-oracle-s42`, `probe28-head-gradient-s42`.
P24 local5 client chỉ preliminary; P24-all/P26/P27 mỗi bộ đã audit200 model×split, P28 có exact-gradient/CE checks.

## Chưa giải quyết — chỉ xét sau owner tiếp tục

1. Đọc veto/history trước khi đề xuất. **Không mở synthesis chỉ khớp unconditional head gradient:** label term chỉ mang
   class counts/means, perturb riêng head không thêm label covariance. Class-conditional projected carrier chưa utility/packet proof.
2. Nếu mở phương án mới: ghi thông tin mới được truyền, gradient/vị trí học thay đổi, matched controls, falsifier và stop gate;
   phân biệt source/support/label precision. Không quét lại KD dose/LR/round hoặc inversion cũ để né FAIL.
3. Prior-only chưa chạy; sửa sample weighting không được gọi là đã chữa plateau. Production phải dùng true full counts,
   không counts fixture; xử lý lớp hiếm không được sample cần thiết kế trước.
4. R1 bootstrap hiện vượt cap; final50 q3 có peak vượt cap. Tái dựng proxy từ seed/hash chỉ là đề xuất **chưa implement/test**.
   Tách transaction cần cùng lịch contact baseline/candidate, phí ACK/header/crypto/cohort wait và snapshot age; R1 chưa PASS.
5. Online từ đầu chỉ sau mechanism/transport gates đủ; cổng giữ nguyên trong dossier. Mixed architecture R2 chưa thực nghiệm.

Probe17 controls đã sửa trong history; DI-hi mềm không đạt gain≥0,02 cả hai seed so Gaussian-hard.
A5/E14 thiếu terminal artifact local, chỉ ghi procedural snapshot từ memory07/10, có thể cũ; không suy kết quả chất lượng.

## Đóng workspace

- Có nhiều thay đổi/untracked từ trước phiên và luồng chấm final; không reset/xóa/ghi đè. HEAD lúc đóng:
  `f6ed8ed9ed5228237e1a8f303edcbe07c3b0a59a`; không tạo commit, stage, Git/DVC push, backup hay cleanup.
- Cập nhật hub, danh mục, results, LESSONS và index. Kiểm12 tài liệu: relative links PASS;8 Python source AST PASS;
  whitespace tài liệu tổng hợp PASS. Whole-tree `git diff --check` còn CRLF/trailing whitespace tại hai CSV sinh từ trước
  (`metrics-all.csv`, `per-class-f1.csv`); giữ nguyên, không dọn ngoài phạm vi. Frozen `fd.py` không đổi, staged diff rỗng;
  hai notebook private ignored0600. Không chạy thêm thí nghiệm sau yêu cầu dừng.
- Skill khi nối lại: `federated-experiment-workflow`; Kaggle→`kaggle-training-notebook`, speed→`federated-training-speed`,
  monitor→`wandb-training-monitor`, chỉnh hub→`context-hub`. Không dùng backup skill nếu owner chưa yêu cầu.
- Việc tay hiện tại: gõ `/compact` trong Codex composer. Assistant không có tool trực tiếp gọi compact.
  Sau compact, yêu cầu đọc lại CONTEXT và bàn giao này trước khi làm tiếp.
