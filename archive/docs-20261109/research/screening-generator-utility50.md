# Generator utility50 — khóa07/11 trước nối r3

**ĐÃ HỦY trước push.** Owner yêu cầu chỉ pilot20. Tài liệu thực thi hiện hành:
[generator pilot20](screening-generator-pilot20.md). Nội dung dưới giữ làm bằng chứng kế hoạch cũ, không phải quyền chạy50.

Owner đã cho tiếp vòng thử generator/phân tích FAIL, nối session và chạy batch dài sau quota reset07:00UTC+7.
Standard và forest đều qua [runtime T4 r2 hai seed](results-generator-runtime.md). Đây là phép thử utility mới;
không đổi verdict local FAIL hoặc gọi runtime PASS là utility PASS.

## Phép so sánh đã khóa

| cơ chế | seed | parent r2 / account | control r50 |
|---|---|---|---|
| standard latent mean0/rownorm1 |42| `trietbackup/mh-z0-stdcop-s42-t4r2-c2/1` |F2 Core42|
| như trên |43| `odixeuit/mh-z1-stdcop-s43-t4r2-c2/1` |F3 Core43|
| forest12edge + conditional58R + transportR |42| `minhtrit06/mh-z2-forestq-s42-t4r2/1` |F2 Core42|
| như trên |43| `emtrangtri/mh-z3-forestq-s43-t4r2/1` |F3 Core43|

Nối chính state r2 hai rank, không train lại r1; out/W&B ID gốc, cosine50, headline r50. Model/data/client/batch,
1epoch thật/round,64synthetic/step,Q/LA/floor/codec của từng parent giữ nguyên; không chọn phương pháp/HP từ r2.
Shared training files/overlay và parent auditSHA kiểm trước generation. Core controls có cùng shared harness,
không chạy lại control vì không đổi infrastructure/optimizer/participation. Generator thay RNG theo cơ chế có chủ đích,
không gọi tolerance-equivalent code optimization.

**Utility PASS mỗi phương pháp:** cả hai seed macroF1−Core≥.003; từng seed không giảm quá
p10 .006,low256 .003,timeDelay .0035,benign .0005,absent accuracy .01. Dùng cùng ngưỡng đề xuất trước các probe generator,
khóa trước khi đọc endpoint50. Không thay luật nếu forest r2 kém hoặc standard r2 gần Core.
Chỉ áp khi đủ50round/audit; không có gate utilityr2 hay chọn best round. Báo đủ byte/timing/tradeoff;
PASS phép thử này chưa tự thay phương pháp Core đã chọn. FAIL→phân tích lớp/cohort/statistics trước cơ chế mới,
không tune bằng official test. Hai seed/test dùng lặp không là fresh confirmatory/significance.

## Kỹ thuật và quota

- `scripts/run_generator_utility50.py`, receipts `runs/generator-utility50/plan.json`; hard not-before07:00UTC+7 ngày07/11.
  Fresh quota/slots sau deadline; nếu quota chưa hồi thì worker chờ1200s, không suy reset từ clock.
- Giữ owner parent để mount private state, không tải/chuyển thêm8,5GB. Không dùng4receiverY50; ưu tiên Y50 được giữ.
- Budget toàn phần còn lại theo worst round từr2 trở đi +90s save allowance/round +1,5h session setup, margin×1,15;
  cap≤11h/session. Partial→pull/audit endpoint thật→nối compatible parent tới50, không tự giảm đích.
- Import state/RNG2rank/cùnground/seed/config trước npy decode, giữ npySHA/Inductorcache, compiled/2T4 image pin.
  W&B private inline key theo quyền cũ, lưu vĩnh viễn trong private versionhistory; không Git/DVC notebook.
- Poll1200s bằng localworker, pull COMPLETE/ERROR, audit source/config/CM→10metric/teacher/finite/bootstrap/staticstats.
  ERROR/audit failure giữ evidence và completion về thread; không restart training từ watcher lỗi.

Artifacts tiếp tục ở `artifacts/{standard-t4,forest-t4}/<run>/<run>`; báo cáo kiểm từng endpoint trong collection của method.
Chỉ sau4run r50/audit mới publish closure/analysis. Khả năng biểu diễn, TSTR và runtime không bảo đảm mức tăng utility này.
