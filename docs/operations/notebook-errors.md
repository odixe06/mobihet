# Sổ lỗi kernel — bắt buộc đọc trước tạo notebook

Trạng thái ERROR không quyết định rerun. Pull/verify round đã commit, source đã chạy và nguyên nhân trước.
Đủ output thì giữ kết quả; checkpoint hợp lệ thì nối boundary, không khởi động r1.

| kernel / build | lỗi đã biết và bằng chứng | ngăn lặp khi tạo notebook mới | quyết định |
|---|---|---|---|
| V2-A seed42 minhtran0601; seed43 dinhan25 | EF sketch khuếch đại residual, loss/parameter tăng, BN calibration nonfinite; [chẩn đoán](../../archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/probes/sketch-diagnostics.json) | Guard finite update/state/BN, đo residual contraction/norm ở đúng epoch/batch; preflight codec không đủ. Không dùng lại adaptation numerical FAIL | Không rerun y nguyên; các arm hoàn tất vẫn có giá trị |
| V2-B seed42 minhtran0601; seed43 dinhan25 | K512/own512 proxy loss không hữu hạn r1, trước commit; K1024 hoàn tất nhưng teacher gate FAIL | Preflight đúng batch/epoch production; smoke4step/batch32 không thay batch512/full epoch. Gating độc lập từngK, không mở KD/ZO khi teacher FAIL | Không rerun512; không dùng1024 che lỗi512 |
| V6-A seed42 minhtran0601; seed43 dinhan25 | G59 bearing=dx/dist bùng nổ trên synthetic gần dist0, 12arm lỗi; control/G61 vẫn hoàn tất | Toy missing/coincident + sampled synthetic support/norm/nonfinite gate trước SGD; không gọi interval rộng là faithful reconstruction | G59 numerical FAIL, G61 quality/byte FAIL; không rerun y nguyên |
| Hybrid full calibration hybf1 minhtrit06; hybf3 trietbackup | Workers rc0, budget stop trước r3, controller đòi mọiarm đúng3 nên kernel ERROR; source/checkpoint r2/r3 hợp lệ | Controller phân biệt session boundary, số round mới và target50. Budget từngarm/worst/pairmax +setup/compile/I/O +15%; không assert mọiarm phải có cùng round | Nối actual r2/r3; không train lại r1 |
| Hybrid cont01 minhtrit06 14/10 | Push trả `not valid kernel sources` cho parent ERROR dù owner-scoped output listing đọc được; notebook vẫn được push | Push parser phải phát hiện refused input; require-resume trước data decode. ERROR output cần dataset private checkpoint +hash gate khi không attach được; listing không thay mount proof | Giữ notebook đã PUSHED, tạo tag mới dùng bridge; trietbackup dùng bridge trước push |

Exact IDs của tám ERROR lịch sử tại [status snapshot](../archive/operations/kernel-status-2026-10-12.json) và
[status hiện có](kernel-status-current.json); status có thời điểm, không thay audit. Các bảng/report từng build tại
[đề xuất](../proposals.md), [Hybrid full audit](../methods/gaussian-proxy-hybrid/builds/hybrid-full/results/feasibility-20261014.md).

## Những lỗi preflight/runtime đã gặp

- Dataset/checkpoint upload: `-r zip -t`, manifest SHA/size và marker-last; không CSV conversion, không bỏ thư mục.
- Private kernel source có thể bị service từ chối mà push vẫn exit0. Kiểm response, actual mount/import guard;
  mất response thì status trước retry, không push trùng.
- W&B resume: preflight log không được tự tăng step; BoundaryCells dùng adapter `commit=False` cho preflight-only.
  Kiểm history monotonic từ round checkpoint; W&B resume không tự nạp weights.
- Zero GPU hoặc sai runtime: metadata cả notebook và JSON, pinned image, assert2T4/sm75/torch trước train;
  không bỏ assert hoặc reinstall torch để cứu run.
- Verifier load GPU tensor: `map_location='cpu'` trước so với packetCPU. Lỗi checker thì sửa checker, kiểm output cũ,
  không train lại model đang đúng.
- CUDA deterministic: CUBLAS config trước CUDA, deterministic algorithms, stable sort/tie; không warn-only để gọi exact-resume PASS.
- Fingerprint tất cả `mh/*.py`: đóng băng source trong full-vs-resume, kể cả module chưa dùng.
- Notebook private-inline có key trong platform history; không quét platform notebook để kết luận driver training lỗi.
  Pull nền tảng vào ignored private_push0600, audit bản redacted; không đưa key vào dataset/log/docs.
- Network reset khi quota/listing: retry **read-only** có giới hạn, giữ UNKNOWN nếu chưa đọc được. Không suy auth hỏng,
  không lặp push có tác động bên ngoài.

Guard runtime phát hiện lỗi; không biến adaptation FAIL thành PASS. Nếu thấy training đang active có nonfinite/OOM/worker kill,
báo chủ dự án và yêu cầu dừng; không tự đổi HP hoặc bỏ client. Stable thì ngừng health polling, vẫn collect terminal.

[CONTEXT](../../CONTEXT.md) · [Notebook preflight batch](resume-batch-preflight-2026-10-14.json) ·
[BoundaryCells helper/controller preflight](boundary-utility-preflight-2026-10-14.json).
