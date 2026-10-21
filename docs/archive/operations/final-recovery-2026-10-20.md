# Recovery final seed42 — 20/10/2026

## Lỗi xác nhận

Source r24: `mh.v7_nystrom.Method.__init__` hash cứng `papers/pilot/fixture/manifest.json`.
Production `--final` chỉ tạo `papers/final/fixture`; zc (20c control) và zi (100c q3) crash `FileNotFoundError`
trước round1, cả rank0/rank1. Collector kéo đủ 54 file/kernel, executed notebook/source khớp; **0 round committed, không có checkpoint để resume**.
`AUDITED.json` có `audits=[]` chỉ xác nhận collection/source, không chứng minh training thành công.
Notebook vòng ngoài không raise khi subprocess rc1 ⇒ Kaggle COMPLETE, W&B parent finished; không được coi là kết quả train.

Owner xác nhận đã Stop/Cancel bốn kernel za/ze/zg/zk. [Status snapshot](../../operations/evidence/final-2026-10-20/final-recovery-2026-10-20.json).
Giữ nguyên source/dataset/notebook đã push r24. Không repush chúng.

## Sửa và cổng trước push

- `fd.py`: adapter constructor Gaussian giữ toán học và packet layout legacy; context hash dùng đúng fixture đang hoạt động.
  Truyền final/smoke từ Run, áp dụng cả train worker và evaluator. 40 module legacy không đổi.
- `gen_fd.py`: notebook raise khi child rc khác0 hoặc không có summary. Lưu log/summary trước khi raise.
- `gen_continue.py`: lấy source tag từ parent, bỏ hardcode r24; cùng source scientific trước/sau resume.
- Plan r25 chỉ sáu seed42; giữ arm, 20/50/100c, cap4/2/2, cosineT50, batch512/512/256.
- `test_final_startup.py`: final-only constructor tái hiện lỗi legacy rồi PASS adapter; pilot context/vector khớp;
  chạy actual notebook cell với child process rc7/rc0, kiểm lỗi lan truyền.
- Bổ sung non-smoke Run/Method + private/proxy epoch cho hai arm × ba client counts × hai rank.
  Chỉ thay production full-row gate bằng gate fixture1023/client, test256; không sửa gate production.
- Chạy lại compiled full pipeline: weights parity với r23, q3 resume, detached eval1/2 workers, predictions/10metric và SHA rejection.

Cổng local PASS: compiled20c parity/resume/detached1/2worker và non-smoke final-only đủ6scenario×2rank, có tail batch.
Bằng chứng: [local recovery](../../operations/evidence/final-2026-10-20/final-recovery-local-2026-10-20.json). Stage r25 chỉ `fd.py` đổi, 40module legacy còn nguyên.
Đang upload/verify source riêng annhtnt; chuẩn bị push **một kernel đối chứng20c trước**.
Chờ remote haiT4 qua setup/Gaussian và heartbeat SGD tiến triển rồi mới mở năm kernel còn lại.
Local một GPU không chứng minh physical2T4; không gọi run full hoàn thành từ fixture.
