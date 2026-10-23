Snapshot trước khi collect đủ/chốt report22/10. Trạng thái hiện hành: [continue](../../operations/continue.md).

# Continue — run cuối seed42, 22/10/2026 ~01:40Z (chờ chấm zk r34–50; cvl FAIL)

Đọc [CONTEXT](../../../CONTEXT.md) → [LESSONS](../../../LESSONS.md) → trang này.
[Pipeline/contract/timing](../../operations/final-pipeline.md) · [Results đã audit](../../methods/full-drift/results.md) ·
[nhật ký 20/10 16Z → 21/10 08Z](continuation-2026-10-20-21.md) (S1, cache, bộ chấm nhanh, chi tiết từng sự kiện).

**22/10 ~03:00Z — owner dừng phiên để compact trước mọi task tiếp theo.** Nghiên cứu synthetic có
[bàn giao riêng](continuation-synthetic-2026-10-22.md): local/Kaggle đã xong/audit; teacher Project48 PASS,
private transfer/inversion FAIL hai seed, gradient-only FAIL; không launch online mới, giữ phương pháp chốt.
Bảng final dưới giữ snapshot01:40Z, chưa kiểm live lại bởi phiên nghiên cứu; chỉ nối sau owner yêu cầu.

## Train run cuối (seed42, source r25, mọi round đã audit = `AUDITED.json` + weights strict/hữu hạn)

| chuỗi | arm | K | đã train + audit | S2 đang chạy (11,5h trừ ze03) |
|---|---|---:|---|---|
| za | q3_zl la_kdpx | 20 | **r1–50 xong** (S1 `fd-s42za02-…-r25-v1`) | — |
| zc | gkd control | 20 | **r1–50 xong** (`fd-s42zc02-…`) | — |
| ze | q3_zl la_kdpx | 50 | **r1–50 xong** (r49–50 `annhannh331/…-s42ze04-…`, audited ~12:25Z; ze03 thuyanh316 ERROR không khởi động) | — |
| zg | gkd control | 50 | **r1–50 xong** (`fd-s42zg02-…`) | — |
| zi | q3_zl la_kdpx | 100 | **r1–50 xong** (r27–50 `anhtruong1909/…-s42zi03-…`, audited 21/10 ~18:05Z) | — |
| zk | gkd control | 100 | **r1–50 xong** (r34–50 `takids1909/…-s42zk04-…`, audited 22/10 ~00:10Z, 450 phút; zk03 ERROR không khởi động) | — |

Round S1 ổn định: 20c ~788 s, 50c ~815 s, 100c ~1.270–1.440 s. Seed43 chưa generate/push.

## Chấm (bộ nhanh; ID `<owner>/mobihet-fd-e<aa>-<bb>-s42<chuỗi>-20261015-r25-v<k>-f<n>`)

**Tag r29 từ 21/10 ~10:40Z** (gate tương đối ≤ 2 % max|logit|; r27 tuyệt đối 0,25 vỡ ở ze r34 —
[final-pipeline §Bộ chấm nhanh](../../operations/final-pipeline.md), [sổ lỗi](../../operations/notebook-errors.md)). Notebook r27 còn chạy để nguyên; vỡ gate ⇒
`collect_eval.py <rid> --partial` rồi chấm phần còn lại bằng r29. Đã chấm + audit trước 21/10: za/zc r1–4, zg/zk r1–2.

| chuỗi | AUDITED | đang chạy (tag) | chưa xếp |
|---|---|---|---|
| za 20c | **r1–50 xong** (r45–50 `-f2` odixe0502 audited 21/10 ~17:57Z) | — | — |
| zc 20c | **r1–50 xong** (r28–50 `-f2` audited 21/10 ~16:30Z) | — | — |
| ze 50c | **r1–50 xong** (r9–28 `-f2` audited ~18:20Z) | — | — |
| zg 50c | **r1–50 xong** (r35–50 `minhtrit06/…-e35-50-s42zg02-…-f1` audited 22/10 01:24Z) | — | — |
| zk 100c | **r1–33** | r34–38 anhtruong1909 (~02:45Z), r39–43 minhtrit06 (~03:35Z), r44–47 takids1909 (~03:15Z), r48–50 minhtrit06 (~02:50Z) — `…-e<aa>-<bb>-s42zk04-20261015-r25-v1-f1`, cả 4 PUSHED + có máy (log live 01:25Z) | — |
| zi 100c | **r1–50 xong** (r27–50 bốn notebook `…-s42zi03-…-f1` audited 22/10 01:14–01:31Z) | — | — |

**`report_final.py` tạm 21/10 ~18:25Z** (`docs/methods/full-drift/results/final/`, 283 round train, chỉ mô tả, chưa đủ 6 chuỗi): private head r50 —
K20 q3_zl la_kdpx 0,8321 vs gkd 0,8098 (proxy 0,8603 vs 0,8545); K50 la_kdpx 0,8178 (gkd mới tới r34). Chạy lại khi zg/zk/zi đủ r1–50.

Account chấm ≠ account train (zg train annhannh331, ze train thuyanh316). Owner 21/10: song song thêm, giữ EMA mọi round,
không tối ưu kernel tiếp; notebook đã push không cancel.

## Screening vòng 24 — sinh mẫu có chủ đích ([design §Vòng 24](../../methods/full-drift/design.md))

**xo FAIL** gain (audited ~12Z; Δ +0,0076 vs xa, [results §Vòng 24](../../methods/full-drift/results.md)) ⇒ dừng hướng.
xou audited ~12:20Z (mô tả: +0,0030 vs xa; xo − xou +0,0046). Phân tích nguyên nhân + đề xuất vòng 25 `cv` chờ owner
([hồ sơ](../../research/methods/teacher-virtual-feature-calibration.md)).
`report_fd.py` 21/10 bỏ qua run `final` (S1/S2 cùng arm gây 'duplicate arm'; run cuối do `report_final.py`).

## Vòng 25 `cv` (owner 21/10 ~16Z: ngân sách ≤ 1,5× byte phương pháp cũ ⇒ **chỉ cvl**)

Byte sửa theo `result.json` xa (mỗi client): xa 5,15 MB/15 epoch, 12,95 MB/50 round; cvf ×6,8 ⇒ loại; cvl ×1,45 (15 ep) / ×1,62 (50 round)
([design §Vòng 25](../../methods/full-drift/design.md), [hồ sơ](../../research/methods/teacher-virtual-feature-calibration.md)); số +48 %/+3,3 % cũ sai.
Test local PASS (`--part r25`, `--part par`). **Cổng khóa ~16:15Z; push 21/10 ~16:20Z `khanhmay0304/mobihet-fd-s42cl-20261015-r30-v1`**
(tag r30, 6,3h, `--fast --workers 2`, xo mất 4,46h ⇒ ước ~5h). **Audited 22/10 ~01:10Z: FAIL** (Δ −0,0136, benign −0,0185, byte ×1,451;
[results §Vòng 25](../../methods/full-drift/results.md)) ⇒ dừng; 16-bit/s43 không chạy. (Lệnh đã dùng: `collect_fd.py --only fd-s42cl-20261015-r30-v1`, rồi `report_fd.py`
và đọc cổng.)

## Account — lưu ý hôm nay ([kaggle](../../operations/kaggle.md))

- **odixeuit + takids1909** (dùng chung dự án khác của owner): owner 21/10 ~18:35Z cho dùng khi có slot, kể cả chấm trên account đã train chuỗi đó.
- minhtrit06 (30h, cache K50 COMPLETE + `--check` 10:55Z) = luồng chấm K50 chính. Quota reset 24/10 00:00Z.
- Dự án kia (owner 21/10) chạy trên minhtrit06 ×2, anhtruong1909, takids1909 ⇒ chia quota; xếp đợt mới ưu tiên account khác ([kaggle](../../operations/kaggle.md)).
- Owner còn dùng Kaggle cho dự án khác: quota hụt bất thường là do dự án đó. `kaggle_account.py quota` đã trừ phần giữ chỗ của
  session đang chạy (trietbackup 0h lúc 10:20Z khi chạy 2 notebook là bình thường).
- Quota live 22/10 00:15Z (trước push zk): minhtrit06 14,8 · odixeuit 13,2 · takids1909 7,1 · anhtruong1909 6,9 · minhtran0601 5,5 · minhtriethihi 5,2 ·
  thuyanh316 4,2 · odixe0502 4,2 · annhtnt 2,9 · dinhan25 1,9 · annhannh331 1,4 · trietbackup 1,2 · catbaochau/nguyenminhtran060113/accanniez ~1,05 · khanhmay0304 0. Reset 24/10 00:00Z.
  Cache K100: accanniez, anhtruong1909, catbaochau, khanhmay0304, minhtrit06, nguyenminhtran060113, takids1909, trietbackup (`artifacts/fixture-cache/`).

## Làm tiếp theo thứ tự

0. **RAM local 9,9 GB, owner chạy dự án khác song song:** tối đa 2 collect cùng lúc; **không SIGSTOP pull** (đứt stream ⇒ `ChunkedEncodingError`,
   22/10 01:05Z) — pull nối tiếp được từ `.part`. Script poll/chain ở scratchpad mất khi tắt máy: dựng lại từ marker `PUSHED`.
1. 4 notebook chấm zk COMPLETE ⇒ `collect_eval.py <rid> --jobs 16` từng cái (≤ 2 song song).
2. Đủ 6 chuỗi ⇒ `report_final.py && report_final_md.py` ⇒ [report đầy đủ](../../methods/full-drift/results/final/report.md) (sinh tự động từ CSV đã audit,
   10 metric × 50 round × private/private_ema/proxy, so sánh `fl_dagsnet_veremi` r50, cost). Cập nhật CONTEXT/results; Git/DVC chỉ khi owner yêu cầu
   (weights zk r34–50 + output 22/10 chỉ ở local).
3. Cache K100 odixeuit v2 (push 00:56Z sau khi owner Stop v1 treo >6h): COMPLETE ⇒ `gen_fixture.py --owner odixeuit --clients 100 --check`.

```bash
B=methods/full-drift/builds/fd-probe/scripts
conda run -n nckh python scripts/run_local_guarded.py python $B/collect_fd.py --only <train rid> --jobs 16
IDS=<poll list> bash $B/launch_eval.sh <train rid> <a-b> <owner> [f2]            # gen_eval + upload nguồn r29/checkpoint + push + PUSHED
IDS=<poll list> bash $B/launch_continue.sh <owner> <K> <parent rid> <session> [giờ] # gen_continue (--hours) + upload + push
conda run -n nckh python scripts/run_local_guarded.py python $B/collect_eval.py <eval rid> --jobs 16
conda run -n nckh python $B/report_final.py
```

## Bẫy đã gặp hôm nay (chi tiết ở nhật ký và [sổ lỗi](../../operations/notebook-errors.md))

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

[17–20/10 trước r25](continuation-2026-10-17.md),
[r24 recovery](final-recovery-2026-10-20.md), [20/10 16Z → 21/10 08Z](continuation-2026-10-20-21.md).
`check_project_layout.py` còn mismatch DVC migration14/10; phiên này không đổi DVC. Không sửa DVC để làm check xanh.
Source/test local không thay remote2T4 proof; run status không thay scientific artifact audit.
