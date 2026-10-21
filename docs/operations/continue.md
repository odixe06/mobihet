# Continue — run cuối seed42 + vòng 25, 21/10/2026 ~13Z (phiên dừng, owner tắt máy)

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → trang này.
[Pipeline/contract/timing](final-pipeline.md) · [Results đã audit](../methods/full-drift/results.md) ·
[nhật ký 20/10 16Z → 21/10 08Z](../archive/operations/continuation-2026-10-20-21.md) (S1, cache, bộ chấm nhanh, chi tiết từng sự kiện).

## Train run cuối (seed42, source r25, mọi round đã audit = `AUDITED.json` + weights strict/hữu hạn)

| chuỗi | arm | K | đã train + audit | S2 đang chạy (11,5h trừ ze03) |
|---|---|---:|---|---|
| za | q3_zl la_kdpx | 20 | **r1–50 xong** (S1 `fd-s42za02-…-r25-v1`) | — |
| zc | gkd control | 20 | **r1–50 xong** (`fd-s42zc02-…`) | — |
| ze | q3_zl la_kdpx | 50 | **r1–50 xong** (r49–50 `annhannh331/…-s42ze04-…`, audited ~12:25Z; ze03 thuyanh316 ERROR không khởi động) | — |
| zg | gkd control | 50 | **r1–50 xong** (`fd-s42zg02-…`) | — |
| zi | q3_zl la_kdpx | 100 | r1–26 (`fd-s42zi-…-r25-v2`) | `anhtruong1909/mobihet-fd-s42zi03-20261015-r25-v1` r27–50, push 08:48Z |
| zk | gkd control | 100 | r1–33 (`fd-s42zk02-…`) | `takids1909/mobihet-fd-s42zk04-20261015-r25-v1` r34–50, push 11:35Z (zk03 ERROR không khởi động) |

Round S1 ổn định: 20c ~788 s, 50c ~815 s, 100c ~1.270–1.440 s. Seed43 chưa generate/push.

## Chấm (bộ nhanh; ID `<owner>/mobihet-fd-e<aa>-<bb>-s42<chuỗi>-20261015-r25-v<k>-f<n>`)

**Tag r29 từ 21/10 ~10:40Z** (gate tương đối ≤ 2 % max|logit|; r27 tuyệt đối 0,25 vỡ ở ze r34 —
[final-pipeline §Bộ chấm nhanh](final-pipeline.md), [sổ lỗi](notebook-errors.md)). Notebook r27 còn chạy để nguyên; vỡ gate ⇒
`collect_eval.py <rid> --partial` rồi chấm phần còn lại bằng r29. Đã chấm + audit trước 21/10: za/zc r1–4, zg/zk r1–2.

| chuỗi | AUDITED | đang chạy (tag) | chưa xếp |
|---|---|---|---|
| za 20c | r1–44 (r28–44 `--partial`) | r45–50 odixe0502 `-f2` (push 16:04Z; `-f1` minhtran0601 không khởi động, ERROR log rỗng) | — |
| zc 20c | **r1–50 xong** (r28–50 `-f2` audited 21/10 ~16:30Z) | — | — |
| ze 50c | r1–8, r29–48 | r9–28 minhtrit06 `-f2` **COMPLETE chưa pull**; r49–50 dinhan25 (push 16:06Z) | — |
| zg 50c | r1–34 | r35–50 minhtrit06 (push ~16:30Z) | — |
| zk 100c | r1–2, r23–33 | r3–12 accanniez `-f2` COMPLETE, pull dở (`.part`, chưa audit); r13–22 trietbackup COMPLETE chưa pull | r34–50 sau zk04 |
| zi 100c | r9–17 | r1–8 trietbackup COMPLETE chưa pull; r18–26 đã pull đủ, **chưa audit** (collect bị dừng) | r27–50 sau zi03 |

Account chấm ≠ account train (zg train annhannh331, ze train thuyanh316). Owner 21/10: song song thêm, giữ EMA mọi round,
không tối ưu kernel tiếp; notebook đã push không cancel.

## Screening vòng 24 — sinh mẫu có chủ đích ([design §Vòng 24](../methods/full-drift/design.md))

**xo FAIL** gain (audited ~12Z; Δ +0,0076 vs xa, [results §Vòng 24](../methods/full-drift/results.md)) ⇒ dừng hướng.
xou audited ~12:20Z (mô tả: +0,0030 vs xa; xo − xou +0,0046). Phân tích nguyên nhân + đề xuất vòng 25 `cv` chờ owner
([hồ sơ](../research/methods/teacher-virtual-feature-calibration.md)).
`report_fd.py` 21/10 bỏ qua run `final` (S1/S2 cùng arm gây 'duplicate arm'; run cuối do `report_final.py`).

## Vòng 25 `cv` (owner 21/10 ~16Z: ngân sách ≤ 1,5× byte phương pháp cũ ⇒ **chỉ cvl**)

Byte sửa theo `result.json` xa (mỗi client): xa 5,15 MB/15 epoch, 12,95 MB/50 round; cvf ×6,8 ⇒ loại; cvl ×1,45 (15 ep) / ×1,62 (50 round)
([design §Vòng 25](../methods/full-drift/design.md), [hồ sơ](../research/methods/teacher-virtual-feature-calibration.md)); số +48 %/+3,3 % cũ sai.
Test local PASS (`--part r25`, `--part par`). **Cổng khóa ~16:15Z; push 21/10 ~16:20Z `khanhmay0304/mobihet-fd-s42cl-20261015-r30-v1`**
(tag r30, 6,3h, `--fast --workers 2`, xo mất 4,46h ⇒ ước ~5h). Collect: `collect_fd.py --only fd-s42cl-20261015-r30-v1`, rồi `report_fd.py`
và đọc cổng. PASS ⇒ bản 16-bit (×1,34 ở 50 round) + kiểm offline F1 bằng fp32, rồi s43 (owner quyết).

## Account — lưu ý hôm nay ([kaggle](kaggle.md))

- **odixeuit dùng chung với dự án khác của owner** (~11h GPU tới 08Z): không xếp chấm/train final; owner nên dùng ≤ ~8h GPU trước ~12Z
  để screening không hết quota; slot GPU (2) bận tới khi xo/xou xong; không thu hồi token trước khi pull. Cache K50 odixeuit (CPU) không dùng.
- minhtrit06 (30h, cache K50 COMPLETE + `--check` 10:55Z) = luồng chấm K50 chính. Quota reset 24/10 00:00Z.
- Dự án kia (owner 21/10) chạy trên minhtrit06 ×2, anhtruong1909, takids1909 ⇒ chia quota; xếp đợt mới ưu tiên account khác ([kaggle](kaggle.md)).
- Owner còn dùng Kaggle cho dự án khác: quota hụt bất thường là do dự án đó. `kaggle_account.py quota` đã trừ phần giữ chỗ của
  session đang chạy (trietbackup 0h lúc 10:20Z khi chạy 2 notebook là bình thường).
- Quota ~07:45Z: odixeuit 17,2 · anhtruong1909 15,9 · takids1909 15,8 · catbaochau 15,3 · trietbackup 13,2 · dinhan25 11,0 · nguyenminhtran060113 7,6 ·
  minhtriethihi 7,5 · khanhmay0304 6,8 · minhtran0601 6,7 · accanniez 6,3 · odixe0502 6,3 · annhtnt 4,8 · thuyanh316 4,2 · annhannh331 4,1 (trước các push sau 07:45Z).

## Làm tiếp theo thứ tự

1. Notebook chấm COMPLETE ⇒ `collect_eval.py <rid> --jobs 16`. ERROR ⇒ pull log; vỡ gate r27 ⇒ `--partial` + chấm phần thiếu
   bằng r29; ERROR log `[]` không output ⇒ push lại nguyên trạng `launch_eval.sh … <suffix f2>`.
2. Slot dinhan25 trống (ze r34–48 hoặc zg r23–34 xong) ⇒ `launch_eval.sh fd-s42ze04-20261015-r25-v1 49-50 dinhan25`.
3. Slot minhtrit06 trống ⇒ zg r35–50.
4. Owner chọn có thử `cv` (vòng 25) không ⇒ nếu có: cài arm trên source tag mới, test local, khóa cổng, push trên account ngoài run cuối.
5. zk04/zi03 COMPLETE (zi03 ~20Z, zk04 ~19–20Z) ⇒ collect ⇒ chấm zk r34–50, zi r27–50 trên account K100 còn quota (anhtruong1909/takids1909 sau khi train xong).
6. Đủ eval r1–50 cả 6 chuỗi ⇒ `report_final.py` (đã sửa đọc proof nhiều round). Không Git/DVC push khi owner chưa yêu cầu.

```bash
B=methods/full-drift/builds/fd-probe/scripts
conda run -n nckh python scripts/run_local_guarded.py python $B/collect_fd.py --only <train rid> --jobs 16
IDS=<poll list> bash $B/launch_eval.sh <train rid> <a-b> <owner> [f2]            # gen_eval + upload nguồn r29/checkpoint + push + PUSHED
IDS=<poll list> bash $B/launch_continue.sh <owner> <K> <parent rid> <session> [giờ] # gen_continue (--hours) + upload + push
conda run -n nckh python scripts/run_local_guarded.py python $B/collect_eval.py <eval rid> --jobs 16
conda run -n nckh python $B/report_final.py
```

## Bẫy đã gặp hôm nay (chi tiết ở nhật ký và [sổ lỗi](notebook-errors.md))

- `kaggle datasets create` báo lỗi nhưng dataset vẫn ready (annhtnt, odixeuit), hoặc lỗi thật tạm thời (trietbackup 3,3 GB, thử lại được).
  `launch_eval.sh` chỉ in dòng cuối nên mất thông điệp: đọc `datasets status`, chạy lại `upload_stage.py` (bỏ create khi ready), push tay.
  Chạy lại launcher sẽ dừng ở "đã tồn tại" vì generator tạo notebook/stage trước upload.
- Nối tiếp tag r25 khi source sống đã sang r28: `gen_fd.FROZEN` (gen_continue bật) kiểm stage theo manifest của chính nó.
- Test import `fd.py` từ thư mục stage tạo `__pycache__` ⇒ upload từ chối ("flat stage"); test r24 đặt `sys.dont_write_bytecode`.
- Kernel mình push mà RUNNING > 20–30 phút không log live (`kaggle_as.py <owner> -- kaggle kernels logs -f <id>` rỗng) và không lên W&B
  = chưa có máy/slot (owner dùng một số GPU cho dự án khác); 21/10 năm kernel như vậy ERROR log `[]` sau 2,5–4h ⇒ xếp lại account khác sớm.
- Poll song song nhiều `kaggle_as.py` (15 tiến trình) đẩy RAM local tụt ⇒ probe bị bộ canh RAM dừng; `status.sh` giới hạn 4 song song.
- Script poll/W&B ở scratchpad mất khi máy khởi động lại (xảy ra 21/10 ~10Z; dựng lại từ marker `PUSHED` chưa có `AUDITED.json`): poll = vòng `kaggle_as.py <owner> -- kaggle kernels status <id>` mỗi 5 phút đến
  COMPLETE/ERROR/CANCEL; W&B = `wandb.Api().runs('<entity>/mobihet-fulldrift')` lọc tên arm.
  Tên run W&B chấm = `evalfast-<run>-r<NNN>` (tên arm, không tên chuỗi): za/ze/zi trùng tên, phân biệt bằng `eval/clients_done` (K) và giờ tạo.

## Bàn giao lịch sử và giới hạn

[17–20/10 trước r25](../archive/operations/continuation-2026-10-17.md),
[r24 recovery](../archive/operations/final-recovery-2026-10-20.md), [20/10 16Z → 21/10 08Z](../archive/operations/continuation-2026-10-20-21.md).
`check_project_layout.py` còn mismatch DVC migration14/10; phiên này không đổi DVC. Không sửa DVC để làm check xanh.
Source/test local không thay remote2T4 proof; run status không thay scientific artifact audit.
