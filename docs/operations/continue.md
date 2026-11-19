# Bàn giao sống — 20/11/2026 ≈ 03:15 +07 (máy tắt sau phiên này)

**Lượt sau chỉ cần đọc file này để tiếp tục.** Chi tiết khi cần: [FB-L](../research/screening-fbl.md) (luật/kết quả),
[compute harness](../research/compute-harness.md), [LESSONS](../../LESSONS.md), [bài học vận hành](lessons.md), [backup](backup.md).
Baseline (14 PP × 6 kịch bản + mixed) ở repo khác: `~/project/experiment`, bàn giao `docs/operations/continue.md` của repo đó.

## 1. Dự án và đích

- MobiHet: FL IDS V2X trên VeReMi NextGen v4 (5.343 xe, 77 input, 16 lớp); mỗi xe giữ model riêng; chỉ trao thống kê/xác suất
  (bootstrap copula/KM + Q), không trao tham số. Hai model: DAGSNet (repo này), FeatureBERT (`../fl_veremi_bert`).
- Đích owner: byte ≤ 0,40 MB/xe/50r; compute ≤ 1,01× FedAvg (useful lẫn executed, cùng engine); r50 DAGSNet > .8998, BERT ≥ .8951;
  Macro-F1 cách baseline tốt nhất ≤ .015. Protocol [protocol](../research/protocol.md); headline r50, không chọn round/HP theo test.
- Phương pháp = **FB-L, cấu hình L4** (owner 19/11): FB N4 (dòng thật 64/37/37 theo tầng cỡ xe, synthetic 64,64/16,16/16,16)
  + chọn dòng thật theo CE cũ (`real_select=top`) + engine `compact=true` (executed 0,997× FedAvg). Code `methods/mobihet_budget.py`,
  byte-identical ở `../fl_veremi_bert/methods/`. Sim r20 6/6 PASS, pilot r20 12/12 audit PASS (bảng [FB-L](../research/screening-fbl.md)).
- Compute tổng (20/11 00:45, [harness](../research/compute-harness.md)): L4 FLOP 0,920× (DAGSNet) / 0,917× (BERT) FedAvg.

## 2. Đang chạy: full r50 L4, 3 seed × 2 model (Kaggle, không phụ thuộc máy)

| chain | phiên 1 (xong) | phiên 2 (h2) | trạng thái 20/11 03:00 |
|---|---|---|---|
| DAGSNet s42 | `heykewtiie/mh-fbl-l4-top-s42-r50` r1–46 | `lelovo/mh-fbl-l4-top-s42-r50-h2` | RUNNING, cần ≈ 1,1 h |
| DAGSNet s43 | `outlookcheck/mh-fbl-l4-top-s43-r50` | — | phiên 1 RUNNING (CAP 7,7 h, ước xong r50 trong phiên ≈ 04:30) |
| DAGSNet s41 | `annhannh331/mh-fbl-l4-top-s41-r50` r1–25 | `hoangdang20t/mh-fbl-l4-top-s41-r50-h2` | RUNNING, cần ≈ 4 h |
| BERT s42 | `novahihi/flb-fbl-l4-top-s42-r50` r1–45 | `gaming3/flb-fbl-l4-top-s42-r50-h2` | **PUSHED 03:34** (lần 3; 2 lần trước dataset không được tạo do `inductor-cache/` 5.581 file — `../fl_veremi_bert/scripts/run_fbl_pilot.py::stage` nay bỏ thư mục đó, chưa commit). Cần ≈ 1,4 h |
| BERT s43 | `outlookcheck/flb-fbl-l4-top-s43-r50` r1–26 | `chiencosieuhang/flb-fbl-l4-top-s43-r50-h2` | HOP_PUSHED 03:06, cần ≈ 4,7 h |
| BERT s41 | `catbaochau/flb-fbl-l4-top-s41-r50` r1–21 | `goilaaiudi/flb-fbl-l4-top-s41-r50-h2` | RUNNING, cần ≈ 5,6 h |

**DAGSNet s42 CHAIN_DONE 03:27 (2 phiên).** Kết quả từng hop: `runs/fbl-r50/orchestrate.log` (dòng `HOP_PUSHED` / `BLOCKED` / `CHAIN_DONE`).

**Orchestrator** `runs/fbl-r50/orchestrate.py` (poll 1200 s): COMPLETE → collect → r < 50 ⇒ hop sang account rảnh quota lớn nhất
(bỏ `dinhan25` + `runs/fbl-r50/exclude.json` = account mà job khác đang giữ slot), CAP = gpu_left − 1 phút;
ERROR/mutation lỗi ⇒ `BLOCKED`, không retry. Đủ 6 chain ⇒ audit DAGSNet + BERT, ghi `ORCH_DONE`. Máy tắt ⇒ orchestrator dừng;
kernel Kaggle vẫn chạy, chỉ hop/collect/audit chờ tới khi chạy lại.

**Bật lại máy — bước đầu tiên:**

```bash
cd ~/project/mobihet && setsid /home/odixe/Tools/miniconda3/envs/nckh/bin/python runs/fbl-r50/orchestrate.py >> runs/fbl-r50/orchestrate.log 2>&1 < /dev/null &
echo $! > runs/fbl-r50/orchestrate.pid
```

Nó đọc lại batch/hop đã khóa (`S*h*/batch.json`, `artifacts/fbl-hop/*/ready.json`), không push lại slug cũ. Trước đó sửa
`exclude.json` theo job đang giữ slot (20/11 03:00 thêm `bestyorn`: `exp-count-cnnbigru-a` của repo experiment; gỡ khi kernel đó xong).
Hop dở dang (có `create-intent.json`, chưa `ready.json`) ⇒ script chỉ verify lại dataset, không tạo mới. Nếu dataset không tồn tại
(`kaggle datasets list --mine` của owner rỗng): đổi tên `create-intent.json`/`create-reply.txt` thành `*.failed-<owner>.*` rồi
`python scripts/run_fbl_r50.py build S<seed>h<k> <owner khác> <CAP>` + `launch S<seed>h<k>` (cwd `../fl_veremi_bert`; DAGSNet: `hop` + `launch-hop` ở repo này).
Slot lúc tắt máy: `chiencosieuhang` 2/2 (central BERT s42 + BERT s43h2), `gaming3` 2/2 (central CNN-BiGRU s41 + BERT s42h2), `bestyorn` 2/2,
`lelovo`/`goilaaiudi`/`hoangdang20t` 2/2; `abbafan` còn 1 slot.

## 3. Việc tiếp theo

1. Khởi động lại orchestrator (trên). Quota reset **21/11 00:00 +07**.
2. `ORCH_DONE`: đọc `runs/fbl-r50/audit.json` + `../fl_veremi_bert/runs/fbl-r50/S<seed>*/audit.json`; báo 10 metric, mean ± SD 3 seed,
   Δ vs T3 cùng seed, byte, compute; cập nhật [FB-L](../research/screening-fbl.md), [comparison](../research/comparison.md) (chỉ khi PASS),
   [tried index](../research/tried-methods-index.md). `BLOCKED` ⇒ đọc log kernel, xử lý tay, không push lại slug đã có.
3. Dọn ổ (owner 20/11: DVC trước → xoá bớt local → phần còn lại sang ORICO `/run/media/odixe/ORICO`, exFAT: không symlink/hardlink/quyền bên trong):
   - Đã làm: `artifacts/fbl-hop/mh-fbl-l4-top-s4{1,2}-r50/pull` `dvc add` + `dvc push -j 16` (OneDrive, `status -c` in sync 20/11 02:58;
     manifest `pull.dvc` đã commit). `upload/` cạnh nó trùng byte `pull/` (`diff -rq` chỉ khác
     `dataset-metadata.json`), dataset đã lên Kaggle ⇒ xoá được 19 GB (lệnh xoá bị bộ lọc quyền chặn ⇒ việc tay, mục 5).
   - Còn làm: dời `.dvc/cache` (≈ 35 GB, hardlink với workspace) sang ORICO (`dvc cache dir --local <ORICO>/mobihet-dvc-cache` + copy),
     rồi xoá workspace `pull/` (DVC-backed; chain không cần vì stage đọc `ready.json`). Giữ local: code, `.git`, `cache/` (data npy),
     `artifacts/{fbl-r50,fbl-full,b21-chain,t3-s41}` (collect + REF audit), `runs/fbl-r50`. `../fl_veremi_bert/artifacts/fbl-hop` (~19 GB) chưa DVC.

## 4. Quy tắc cần nhớ

- Git: owner 20/11 cho commit ⇒ `main` local = `38754dd` (FB-L code/docs/DVC manifest), **chưa push GitHub** (origin/main `a6e77a1`).
  Notebook chain clone GitHub tại SHA đã khóa trong batch (`base_sha`) + overlay ⇒ hop DAGSNet không phụ thuộc HEAD; chỉ push GitHub
  khi owner yêu cầu. Không sửa `scripts/w20_common.py::SOURCES`, `core/`, `methods/mobihet_budget.py` khi còn chain (overlay sha256).
  `../fl_veremi_bert` vẫn còn thay đổi FB-L chưa commit; **không commit repo đó khi còn chain BERT**: `run_fbl_r50.py build` assert
  `HEAD == base_sha == origin/main`.
- Backup OneDrive 20/11: Git bundle `…/veremi_build/mobihet-v2/mobihet.bundle` (verify OK, `main` 38754dd); `dvc push -j 16` toàn repo
  xong 03:12, exit 0, 398 file. rclone upload cloud ~24 MB/s, có thể chưa xong lúc tắt máy ⇒ tự tiếp khi bật lại (không xoá `~/.cache/rclone`).
- Kaggle: credential theo process (`kaggle_as.py`), quota `kaggle_quota.py`; `UNVERIFIED`/`QUERY_ERROR` = chưa rõ. Mutation không retry khi
  chưa kiểm intent/marker/version. COMPLETE ≠ kết quả: audit artifact trước. Luật gate khóa trước push, không sửa hồi tố.
- Job nền: `setsid … & echo $! > pid`; dừng theo PGID (`kill -TERM -<pid>`). Giữ ≥ 30 GB trống trước hop (mỗi hop copy state 4–10 GB;
  20/11 02:17 ổ đầy làm hop BERT s43 BLOCKED).
- rclone OneDrive mount (`~/.cache/rclone`, full mode) còn giữ file DVC chưa lên cloud (≈ 17 GB lúc 03:00): không xoá thư mục này.

## 5. Việc tay

- (tuỳ) Giải phóng 19 GB bản trùng: `rm -rf ~/project/mobihet/artifacts/fbl-hop/mh-fbl-l4-top-s4{1,2}-r50/upload`
- (quyết định) Có push `main` repo này lên GitHub không (an toàn với chain đang chạy); `../fl_veremi_bert` chờ `ORCH_DONE`.
