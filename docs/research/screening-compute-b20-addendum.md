# B20 — phụ lục luật (owner 12/11, ghi trước khi có kết quả)

File [design](screening-compute-b20.md) nằm trong source-lock của pipeline đang chạy nên không sửa; phụ lục này bổ sung,
áp cho B20 và các bậc sau của nó. Gộp vào design sau khi batch đóng.

**Owner:** guard fail thì phải đọc quá trình học còn tốt không (train dài hơn có cải thiện thêm không); chấp nhận miễn
performance vẫn trong **top 3**. Round headline vẫn 50, không kéo dài run.

1. **Đọc quỹ đạo** mọi arm/seed: `scripts/analyse_b20_trajectory.py b20 20` (sau audit + `analyse_b13.py b20`) →
   `runs/sim-kaggle-b20/trajectory.{json,md}`: Δ(r) so C cùng seed cho macro và guard p10/low256/benign/timeDelay/absent,
   slope Δ nửa sau, Δ TB r11–15 → r16–20, nhãn `closing` / `widening` / `flat/mixed`, slope macro của chính arm r16–20.
   Pilot r20 cosine50 và r50 đọc tương tự (nửa sau / 10 round cuối).
2. **Guard fail** vẫn report-only (scope macro-first), nhưng phải báo kèm trend; guard `widening` ở cả hai seed được nêu
   rõ trong kết quả, không veto.
3. **Sim FAIL non-inferiority nhưng quỹ đạo tốt** (Δmacro `closing` cả hai seed và Δmacro r20 ≥ −.005 mỗi seed) →
   vẫn lên bậc kế (pilot r20 hoặc r50 với SD), ghi là "lên theo quỹ đạo". `widening` hoặc Δ r20 < −.005 → dừng nhánh.
   Lưu ý: sim cosine20 có LR → 1e-5 ở r20 nên cuối đường phẳng theo thiết kế; trend đọc trên Δ so control cùng lịch.
4. **Luật r50 (thay cho non-inferiority so CBB):** arm PASS nếu macro F1 r50 **mỗi seed > .9048** (FedAvg TB 3 seed,
   hạng 4 hiện tại; top 3 = APFed .9130, CANPerFL .9097, CBB .9078). Báo thêm Δ so CBB, 10 metric, guard + trend, FLOPs.
5. Nhiều arm PASS r50 → owner chọn; combo (vd SH+SD) chạy riêng sau khi từng arm có verdict, cùng thang.
