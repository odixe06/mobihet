# X20 — combo thành phần đã qua W20 và non-target KD giới hạn trong lớp xe có

**Đã đóng 03/11:** core PASS, bank PASS, nckd_local xác nhận, full không qua → [kết quả](results-x20.md).

Thiết kế sau [W20](results-w20.md). Owner 02/11 (sau W20) ủy quyền chạy các kiểm nghiệm cần thiết cho phương pháp cuối.
Giữ nguyên protocol W: DAGSNet W1/full 5343/20 round/cosine50/seed 42–43/batch64/AdamW reset/AMP fp16/clip1, 1 epoch dữ liệu local.
Không chọn HP/round bằng official test; floor 16 và codec giữ nguyên như W20 (không tinh chỉnh).

## Ô đã khóa (`scripts/x20_common.py`)

| ô | công thức | control | câu hỏi → quy tắc đọc |
|---|---|---|---|
| core | stream + step floor 16 + bootstrap nén | W6/W7 (floor), W8/W9 (nén), R0–R3 | Hai thành phần qua gate có cộng được? Nhận nếu low256/p10 trong spread của W6/W7 và pooled/timeDelay trong spread stream |
| bank | mixed_input + floor + nén | X core, Q6/Q7 | Input-bank còn lợi khi đã có floor? Giữ tầng nếu p10 > core +.006 hoặc low256 > core +.003 ở cả hai seed; nếu không, bỏ (tốn 1,31 MB) |
| live_real_nckd_local | W2/W3 với tập non-target chỉ gồm lớp xe có | W2/W3, Q6/Q7 | Hại lớp vắng do NCKD ép logit lớp vắng? Xác nhận nếu absent ≥ Q6/Q7 −.01 và pooled ≥ Q6/Q7 +.0015; bác bỏ nếu absent vẫn < Q6/Q7 −.02 |
| full | bank + nckd_local | X bank | **Chỉ đọc khi nckd_local qua gate.** Đưa realKD vào phương pháp nếu pooled > bank +.001, absent ≥ bank −.01, low256/p10 không dưới bank − spread |

nckd_local: non-target set = lớp xe có trong train local ∖ {y}; trọng số = khối lượng teacher trên tập đó (khi xe có đủ 16 lớp
trùng NCKD W20). Logit lớp vắng không nhận gradient từ KD trên dòng thật; vẫn học từ synthetic như control. 0 byte thêm,
xe đã biết tập lớp của mình. 62,7% xe thiếu ≥1 lớp.

## Byte/xe/20 round

| ô | bootstrap up/down | round up/down | tổng |
|---|---:|---:|---:|
| core | 271.031 / 729.408 | 544 / 544 | 1.022.199 |
| bank, full | 271.031 / 729.408 | 33.344 / 33.344 | 2.334.199 |
| live_real_nckd_local | 1.655.872 / 1.655.872 | 33.344 / 33.344 | 4.645.504 |

## Vận hành

- Preflight local sm86 PASS (`runs/x-study/preflight.json`): 25 test (bỏ chọn test replay cần checkpoint Q6/Q7 đã dọn local,
  X không có replay); knob mặc định bit-exact Live stream/real và Replay; 4 ô × 2 seed finite, skip0, floor đúng số bước,
  state roundtrip, strict rebuild, diagnostics không đổi state/RNG. `nckd()` mặc định bit-exact bản W20 (test).
- Pipeline `python scripts/run_w20.py --study x20`: dùng lại cache npy `odixe0502/mobihet-v4npy-cache/1`, 2 kernel/owner,
  receipt `runs/x-study/`, artifact `artifacts/x20/`, audit `audit_w20.py --study x20`.
- Combo đọc trực tiếp, không cộng gain từng ô W.
