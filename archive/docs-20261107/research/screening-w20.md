# W20 — decoupled KD, step floor synthetic, bootstrap nén

Owner 02/11 ủy quyền chạy cả batch sau preflight local PASS. Bằng chứng nền: [R20/V20](../../docs/research/results-r20-v20.md).
Giữ DAGSNet W1/full 5343/20 round/cosine50/seed42–43/batch64/AdamW reset/AMP fp16/clip1; 1 epoch trên dữ liệu local
đúng như repo benchmark. Mỗi ô đổi **một** thành phần so với control đã audit; không chọn HP/round bằng official test.

## Ô đã khóa (`scripts/w20_common.py`)

| ô | công thức | control so sánh | câu hỏi → dấu hiệu bác bỏ |
|---|---|---|---|
| replay_real_nckd | replay teacher Q6/Q7 + realKD binary-gate, KL thay bằng (1−t_y)·NCKD | R4/R5 (noReal), R6/R7 (KD đầy đủ) | Phần target-class có phải kênh hại? Bác bỏ nếu timeDelay vẫn giảm >.02 so R4/R5 hoặc other15 mất phần lớn lợi của R6/R7 |
| live_real_nckd | mixed_input live + realKD binary-gate NCKD | Q6/Q7, Q10/Q11 | NCKD có cắt vòng feedback? Bác bỏ nếu Q[13,13] r20 thấp như Q10 (.58) hoặc timeDelay giảm >.02 so Q6/Q7 |
| synth_nckd | stream; synthetic: −log p_y + (1−t_y)·NCKD thay KL(Q[y]‖p) | Q0/R0/R2, Q1/R1/R3 | Hàng teacher nhầm có làm hại qua synthetic? Bác bỏ nếu timeDelay không tăng vượt spread (.0035) hoặc pooled giảm >.0005 |
| step_floor | stream + bước synthetic-only tới tối thiểu 16 bước/round | như trên | Xe ít bước có thiếu tối ưu? Bác bỏ nếu low256/p10 không tăng vượt spread (.003/.006) |
| boot_compact | stream; downlink CDF uint16 + fp16 mean/Cholesky, bỏ moment Gaussian | như trên | Nén ~70% bootstrap có giữ utility? Chấp nhận nếu mọi chỉ số trong spread phiên |

NCKD = phần non-target của Decoupled KD ([Zhao et al., CVPR 2022](https://arxiv.org/abs/2203.08679), Eq. 2–3):
KD = TCKD + (1−t_y)·NCKD; ta giữ đúng trọng số (1−t_y) của phân rã, không thêm α/β. Đây là adaptation cho teacher theo lớp,
không reproduce DKD gốc (teacher theo input, nhiệt độ). Gradient của NCKD lên logit lớp thật bằng 0 (test đơn vị).

Step floor 16 = số bước của xe 1024 dòng ở batch 64, cố định trước; chạm 2226 xe (42%), thêm 20.705 bước client
(+3,5% so 586.405 bước thật). Bước thêm chỉ có 64 dòng synthetic, không cập nhật BN running buffer; mọi dòng local vẫn
được dùng đúng một lần mỗi round (owner 02/11: 1 epoch là 1 epoch trên dữ liệu local; bước synthetic là cơ chế phương pháp).

## Byte

| ô | bootstrap up | bootstrap down | round up/down | tổng B/xe/20r |
|---|---:|---:|---:|---:|
| stream (control) | 1.655.872 | 1.655.872 | 544/544 | 3.333.504 |
| synth_nckd, step_floor | như control | | 544/544 | 3.333.504 |
| boot_compact | 271.031 | 729.408 | 544/544 | 1.022.199 |
| replay/live real_nckd | như Q mixed_input | | 33.344/33.344 | 4.645.504 |

Uplink compact = 64 (count) + 73.847 (histogram thưa, trung bình đo offline, không mất mát) + 197.120 (tổng hiệp phương sai fp32).
Downlink compact = 64 (prior) + 630.784 (CDF uint16) + 98.560 (fp16 mean/Cholesky). Chưa tính transport/ACK.

## Gate và vận hành

- Preflight local sm86, 2 seed × 3 round fixture thật: knob mặc định bit-exact với Live stream/real và Replay real;
  mỗi ô finite, skip0, step floor đúng số bước, state roundtrip, strict rebuild 395.408, diagnostics không đổi state/RNG.
  `runs/w-study/preflight.json`.
- Cache npy dựng một lần bởi kernel builder public (owner 02/11), manifest SHA256 từng file kiểm lại trong mỗi run;
  run W mount cache thay vì build 448 s.
- Remote: 2 T4, compile gate PASS mọi rank (từ chối eager fallback) + guard finite scoring như Q/R; skip fraction ghi mỗi round,
  audit đối chiếu với mức R/V (≤ ~2e-5).
- Audit đủ 20 round như R; đọc kết quả theo spread phiên ở [R20/V20 §1](../../docs/research/results-r20-v20.md#1-độ-ồn-giữa-các-phiên-cùng-công-thức-cùng-seed).
- Không ghép gain các ô để dự đoán combo; combo chỉ chạy sau khi từng ô qua gate.
