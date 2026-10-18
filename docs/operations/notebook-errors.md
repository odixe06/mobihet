# Sổ lỗi kernel — bắt buộc đọc trước tạo notebook

Trạng thái ERROR không quyết định rerun. Pull/verify round đã commit, source đã chạy và nguyên nhân trước.
Đủ output thì giữ kết quả; checkpoint hợp lệ thì nối boundary, không khởi động r1.

| kernel / build | lỗi đã biết và bằng chứng | ngăn lặp khi tạo notebook mới | quyết định |
|---|---|---|---|
| H1 full1 q1 accanniez15/10 | Host maintenance/Docker daemon sau16299,7s; API0artifact/executed404; W&B summary25/22 nhưng không weights | Status ERROR không là numericalFAIL; xác minh output và restore checkpoint trước. W&B không là checkpoint. Dùng audited parent16/14 nếu output mất; logging project mới tách nhánh recovery | Giữ kernel lỗi; q2 nối16/14, không r1; [proof](../archive/operations/full1-unavailable-2026-10-15-afternoon.json) |
| V2-A seed42 minhtran0601; seed43 dinhan25 | EF sketch khuếch đại residual, loss/parameter tăng, BN calibration nonfinite; [chẩn đoán](../../archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/probes/sketch-diagnostics.json) | Guard finite update/state/BN, đo residual contraction/norm ở đúng epoch/batch; preflight codec không đủ. Không dùng lại adaptation numerical FAIL | Không rerun y nguyên; các arm hoàn tất vẫn có giá trị |
| V2-B seed42 minhtran0601; seed43 dinhan25 | K512/own512 proxy loss không hữu hạn r1, trước commit; K1024 hoàn tất nhưng teacher gate FAIL | Preflight đúng batch/epoch production; smoke4step/batch32 không thay batch512/full epoch. Gating độc lập từngK, không mở KD/ZO khi teacher FAIL | Không rerun512; không dùng1024 che lỗi512 |
| V6-A seed42 minhtran0601; seed43 dinhan25 | G59 bearing=dx/dist bùng nổ trên synthetic gần dist0, 12arm lỗi; control/G61 vẫn hoàn tất | Toy missing/coincident + sampled synthetic support/norm/nonfinite gate trước SGD; không gọi interval rộng là faithful reconstruction | G59 numerical FAIL, G61 quality/byte FAIL; không rerun y nguyên |
| Hybrid full calibration hybf1 minhtrit06; hybf3 trietbackup | Workers rc0, budget stop trước r3, controller đòi mọiarm đúng3 nên kernel ERROR; source/checkpoint r2/r3 hợp lệ | Controller phân biệt session boundary, số round mới và target50. Budget từngarm/worst/pairmax +setup/compile/I/O +15%; không assert mọiarm phải có cùng round | Nối actual r2/r3; không train lại r1 |
| Hybrid cont01 minhtrit06 14/10 | Push trả `not valid kernel sources` cho parent ERROR dù owner-scoped output listing đọc được; notebook vẫn được push | Push parser phải phát hiện refused input; require-resume trước data decode. ERROR output cần dataset private checkpoint +hash gate khi không attach được; listing không thay mount proof | Giữ notebook đã PUSHED, tạo tag mới dùng bridge; trietbackup dùng bridge trước push |

Exact IDs của tám ERROR lịch sử tại [status snapshot](../archive/operations/kernel-status-2026-10-12.json) và
[status hiện có](../archive/operations/kernel-status-current.json); status có thời điểm, không thay audit. Các bảng/report từng build tại
[đề xuất](../proposals.md), [Hybrid full audit](../methods/gaussian-proxy-hybrid/builds/hybrid-full/results/feasibility-20261014.md).

## Những lỗi preflight/runtime đã gặp

- Dataset/checkpoint upload: `-r zip -t`, manifest SHA/size và marker-last; không CSV conversion, không bỏ thư mục.
- Create dataset trả success trước xử lý xong; ListDatasetFiles có thể403 trong thời gian đó. Đọc status/đợi ready,
  retry read có giới hạn, không upload/push trùng. Vẫn kiểm owner/access;403 riêng lẻ không chứng minh nguyên nhân.
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
- ClassTail local15/10: `scan_seconds` trong algorithm/diag làm full-vs-resume khác dù weights đúng.
  Timing chỉ ở round diagnostics; loại khỏi algorithm state. Local4arm exact-resume đã PASS.
- Dataset checkpoint15/10 hiện giữ `package/` và `runs/` trong listing; bản cũ từng bỏ một level.
  So actual all-page names/sizes, recursive discovery và CPU production import; không hardcode việc bỏ prefix.
- Notebook private-inline có key trong platform history; không quét platform notebook để kết luận driver training lỗi.
  Pull nền tảng vào ignored private_push0600, audit bản redacted; không đưa key vào dataset/log/docs.
- Network reset khi quota/listing: retry **read-only** có giới hạn, giữ UNKNOWN nếu chưa đọc được. Không suy auth hỏng,
  không lặp push có tác động bên ngoài.

Guard runtime phát hiện lỗi; không biến adaptation FAIL thành PASS. Nếu thấy training đang active có nonfinite/OOM/worker kill,
báo chủ dự án và yêu cầu dừng; không tự đổi HP hoặc bỏ client. Stable thì ngừng health polling, vẫn collect terminal.

[CONTEXT](../../CONTEXT.md) · [Notebook preflight batch](../archive/operations/resume-batch-preflight-2026-10-14.json) ·
[BoundaryCells helper/controller preflight](../archive/operations/boundary-utility-preflight-2026-10-14.json).

15/10 CrossClientRank utility local emitted-controller fixture: `if not active and pending: break`
đã dừng queue ngay khi pair vừa hoàn tất dù còn budget. Sửa trước GPU push: chỉ break khi còn thời gian
không đủ worst-round+margin; normal/costFAIL/workerFAIL/budget fixtures kiểm đủ8arms.
Dataset scientific source chưa đổi; controller chưa PUSHED khi phát hiện.

15/10 W&B config JSON chuyển integral float1,0 thành int1; tái hash config từ API có thể báo fingerprint
mismatch giả. Inspector khôi phục declared argparse float types trước hash, rồi so frozen source/fixture/fork.
Không sửa fingerprint run hoặc rerun training vì lỗi checker.

15/10 CrossClientRank utility seed42: CE checkpoint r3 có `wire=b''`; PyTorch weights-only loader
từ chối `GLOBAL bytes`. Minimal empty-byte fixture tái hiện cả local2.13, không phải lỗi riêng2.10.
Fixture trước chỉ chạy pooled arm, không chứa empty CE wire; preflight phải cover cả CE/control state.
Scoped `torch.serialization.safe_globals([bytes])` giữ `weights_only=True`, không sửa frozen source/model.
Actual r3 weights/state/payload/predictions audit PASS; cost1,2556/1,2180×CE.
Recovery dùng tag mới và actual r3, CPU mount/audit trước GPU, không SGD lại r3.
Helper dataset verifier cũng đã sửa root-level manifest filename bị `split('/')[1]` IndexError;
đây là checker bug, không dataset corruption hoặc lý do upload/train lại.

## Phiên tối 15/10 — lỗi build và cách tránh

| sự cố | bằng chứng | quy tắc cho build sau |
|---|---|---|
| CCR seed43 v1 cùng lỗi emptybytes như seed42, nằm ERROR ~1,5h không ai biết | lỗi sau167s; owner Console "vẫn chạy" là hàng đợi/UI cũ | Lỗi tất định từ code/serializer của một seed ⇒ kiểm ngay status mọi seed cùng build; chuẩn bị recovery song song. Platform RUNNING/Console không thay status API + log |
| Hybrid-full chạy lại arm đã có ở full-validation | h_dml/h_dml_own bit-exact 8/8 round; local/Gaussian/G-own lệch ≤0,008; hybf3 100% trùng, owner dừng | Trước khi lập build/batch: so config (bỏ fingerprint source) với arm đã audit ở build khác; trùng thì tái dùng reference, chỉ chạy arm mới (Hybrid: h_hybrid, h_hybrid_own) |
| Budget CCR 2,25h không đủ cho reference | rank arm ~1 phút/round, cr_dml ~4,75 phút/round ⇒ dml dừng r16 | Ước budget theo **loại arm chậm nhất** (DML/Hybrid = 2 model + eval), không trung bình; xếp arm chậm sớm hoặc session riêng |
| Session full 4h chỉ được 1 round | hybf1 q2: worst-round 7.020s, session kết thúc 1,95h | Round full 1,6–2h/arm: session phải ≥ setup + k×worst-round; 4h chỉ ra 1 round, lãng phí boundary |
| Collector CCR báo SHA lệch giả ở run resume | `metrics.csv` (và result.json) được driver ghi lại khi chạy tiếp | Checker miễn file tổng hợp được ghi lại; vẫn bắt buộc round/config/payload khớp. Đã sửa `collect_rank_utility.py` |
| Verifier dataset fail vì `__pycache__` | code test local chạy trong thư mục input sau khi lập danh sách | Lập manifest/danh sách file **trước**, chạy test với `PYTHONDONTWRITEBYTECODE=1` hoặc ngoài thư mục stage |
| `kaggle datasets create` lỗi `Expecting value` | 110/110 file upload thành công, status `ready` | Lỗi parse response sau upload: kiểm `datasets status` + all-page names/sizes, không upload lại |
| `pkill -f <pattern>` giết chính shell đang chạy | exit144 hai lần | Dừng tiến trình nền bằng PID cụ thể, không pattern trùng dòng lệnh hiện tại |
| Push session thứ 3 bị từ chối | `Kernel push error: Maximum batch GPU session count of 2 reached`; push_nb không tạo PUSHED, status 404 | Mỗi account tối đa **2 batch GPU session**; lập kế hoạch arm theo 2 session/account, phần dư xếp sau khi session kết thúc |
| 14 arm/seed không vừa 7h | AntiMem round 191–259s (eval 4 role mỗi round) so ước128s | Đo round đầu thật trước khi chốt số arm/session; eval nhiều role tốn thêm ~50%; xếp arm quan trọng (đối chứng/QProxy) sớm |
| `kaggle datasets create` 403 ở account khác | odixeuit/minhtrit06: upload file xong rồi `CreateDataset` 403 (OAuth và KGAT token đều vậy); list/đọc vẫn được, dung lượng 2–12GB | Account bị chặn tạo dataset mới: đừng chép dataset sang account đó. Dùng **dataset gốc share cho account chạy** (Kaggle web → dataset Settings → Sharing), kiểm `kaggle datasets files` bằng account chạy trước khi push. `gen_antimem.py --dataset_owner` tách owner dataset/kernel |
| FullDrift v1: mọi arm chết ở `Run.__init__` sau setup | `TypeError: int + list`: gate production cộng `manifest['counts']['train_k']` (histogram 16 lớp) như số | Smoke local bỏ qua nhánh `if not smoke` ⇒ tách gate production thành hàm và **test trên manifest thật** (`test_fd_local.py --manifest`, pull `source/data_manifest.json` của session đầu). Session v1 chỉ tốn ~5–10 phút/session |
| Collect chậm, collector Hybrid không thoát; audit FD song song bị guard dừng (~4GB/tiến trình) | Collector hardcode `pull_kernel_output -j 6` (q3 full 8,8GB/346s, 25MiB/s); `collect_hybrid_full.py` lặp vô hạn vì hbf1-cont01/hybf3-q2 không bao giờ AUDITED (16/10) | Theo skill [pull-outputs](../../.claude/skills/kaggle-training-notebook/references/pull-outputs.md): tốc độ theo tổng kết nối, ~32 bão hòa. Collector có `--jobs` (mặc định 16); nhiều session thì **một tiến trình mỗi session** (`collect_fd.py --only`), chia ~32 kết nối; Hybrid dùng `--once`. **Pull song song, audit tuần tự**: `collect_fd.py` giữ lock `papers/.artifact-verification.lock` khi audit |
| Collector FD vòng 3 assert sai sau khi thêm arm vòng 4 | `collect_fd.py` so `fd.py` pulled với `source/proj/fd.py` **hiện tại**; fd.py đã đổi (hkd, gaussian_m4b) sau stage r3 (16/10) | So với bản **stage của vòng đó** (`artifacts/input/fd-src-<tag>/fd.py`), không với source đang sửa; nhiều vòng song song thì source sống luôn đi trước |
