# Generator pilot20 — khóa 07/11 trước nối r3

Owner sửa phạm vi: **chỉ pilot tổng cộng20round**, không tự chạy50. Kế hoạch utility50 cũ đã hủy
trước mọi GPU push. Standard/forest đã qua [runtime T4 r2 hai seed](results-generator-runtime.md).

## So sánh và điểm dừng

| cơ chế | seed | parent r2 | control cùng r20 |
|---|---|---|---|
| standard latent mean0/rownorm1 |42| `trietbackup/mh-z0-stdcop-s42-t4r2-c2/1` |F2 Core42|
| như trên |43| `odixeuit/mh-z1-stdcop-s43-t4r2-c2/1` |F3 Core43|
| forest12edge + conditional58R + transportR |42| `minhtrit06/mh-z2-forestq-s42-t4r2/1` |F2 Core42|
| như trên |43| `emtrangtri/mh-z3-forestq-s43-t4r2/1` |F3 Core43|

Nối state/RNG hai rank r2, chạy r3→r20, tức18round thêm. Giữ `--rounds 50 --until 20`:
cosine50 của parent, model/data/client/batch/loss/Q/floor/codec giữ nguyên. Không đổi thành cosine20.
Core đối chiếu bằng CM r20 có sẵn, không so pilot20 với Core50. Source/overlay/parent auditSHA kiểm trước launch.

**Ngưỡng sàng lọc pilot:** cả hai seed macroF1−Core≥.003; từng seed không giảm quá
p10 .006, low256 .003, timeDelay .0035, benign .0005, absent accuracy .01. Giữ ngưỡng đã đề xuất,
áp tại endpoint20 theo yêu cầu mới; không điều chỉnh theo kết quả. Đủ r20/audit mới tính `pass_pilot`.
PASS pilot chưa là utility50 hoặc thay Core. Không chọn best round/HP bằng official test; hai seed/test dùng lặp
không là fresh confirmation/significance. Báo đủ10metric, lớp/cohort, byte và timing.

**Dừng sau r20 → pull/audit → phân tích. Không tự nối50, kể cả pilot PASS.** Các Y50 cũ vẫn theo endpoint riêng.

## Chạy và khôi phục

- Runner `scripts/run_generator_pilot20.py`; receipt `runs/generator-pilot20/plan.json`.
  Chờ07:00UTC+7 ngày07/11, kiểm fresh quota/slot thật; không dùng bốn receiver Y50.
- Giữ owner/out/W&B ID parent; import đúng state/RNG hai rank trước npy decode, không fallback r1.
  Giữ pinned2T4, compile, npy manifestSHA và Inductor cache qua session.
- Budget phần còn lại tới20 theo worst round +90s save/round +1,5h setup/session, margin×1,15; cap≤11h.
  Partial→pull/audit round thật→nối tối đa20; ERROR cũng pull, không restart vì watcher lỗi.
- Local worker poll1200s; chỉ sau đủ bốn run/audit r20 mới ghi closure/analysis và completion.
  Artifacts ở `artifacts/{standard-t4,forest-t4}/<run>/<run>`, như parent.

Kế hoạch cũ lưu tại `runs/generator-utility50/`; job cũ canceled04:04:27 trước push, không bật lại từ notification muộn.
