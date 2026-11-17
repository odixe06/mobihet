Snapshot trước khôi phục Claude 17/11; trạng thái sống: [continue](../../docs/operations/continue.md).

# Bàn giao sống — 17/11/2026

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [bài học vận hành](lessons.md).
Lịch sử: [snapshot](../../archive/operations-20261116/before/docs/operations/continue.md).

## Hiện tại

- **Xong 17/11 06:58:** pilot [Lite2](../research/screening-lite2.md) **B + U1** DAGSNet seed 42 r1–20 (`dinhan25/mh-lite2-bu1-s42-r20/1`,
  base `1cc91d9` + overlay `methods/mobihet_{compute,lite}.py`): audit artifact PASS, verdict **PASS** — Δmacro TB r16–20 −.0028, r20 −.0028
  (.8968 vs .8995), bootstrap 340 663 B (`runs/lite2-full/{audit.json,audit.log,watch.log}`, artifact `artifacts/lite2-full/`). Watcher pilot đã thoát.
- **Đang chạy (push 17/11 07:03 +07, RUNNING lúc kiểm 07:08):** nối r21–50 DAGSNet `dinhan25/mh-lite2-bu1-s42-r50/1`, CAP 7,5 h,
  parent `dinhan25/mh-lite2-bu1-s42-r20/1` (cell resume kiểm state 2 rank r20), run dir giữ tên `mh-lite2-bu1-s42-r20`, cùng hp/source;
  spec `runs/lite2-full-r50/batch.json` (`scripts/run_lite2_full.py … r50`). Watcher detached PID trong `runs/lite2-full-r50/watch.pid`
  (`watch.sh`, poll 1200 s ≤ 30 lượt → pull `artifacts/lite2-full-r50/` → `run_lite2_full.py audit r50` → `DONE`); log `runs/lite2-full-r50/watch.log`;
  owner = phiên Claude Code 17/11 (phiên mất thì đọc `DONE`/`audit.log`/`audit.json`). Không start watcher/launcher thứ hai.
- **Xong (COMPLETE 17/11 09:18 +07; pull/audit PASS):** pilot **B + U1** FeatureBERT seed 42, r1–20 cos50, repo `../fl_veremi_bert`:
  `annhannh331/flb-lite2-bu1-s42-r20/1`, CAP 8,5 h, base BERT `b93198a` + overlay `methods/mobihet_{compute,lite}.py`
  (byte-identical bản MobiHet đã audit sim); spec `../fl_veremi_bert/runs/lite2-pilot/batch.json`, launcher
  `../fl_veremi_bert/scripts/run_lite2_pilot.py`. Watcher detached PID trong `../fl_veremi_bert/runs/lite2-pilot/watch.pid`
  (`watch.sh`, poll 1200 s ≤ 32 lượt → pull → `run_lite2_pilot.py audit` → `DONE`); log `../fl_veremi_bert/runs/lite2-pilot/watch.log`.
  Watcher đã thoát, DONE/audit.json có thật: macro r20 .898173076, Δr20−.004746125, ΔTB−.005172594, bootstrap340663B; verdict PASS.
  Bounded auditor đã chạy lại độc lập trong phiên tổng hợp brief; chưa có state2rank local. Không collect lại pilot.
  Control = T3 BERT seed 42 r1–20 (`artifacts/t3-pulls/flb-t3-s42-p3`: macro r20 .9029, TB r16–20 .9009, `t_train` 5,95 h).
- Port BERT xong 17/11 (chưa commit, repo BERT): copy 2 file method; `check_t3_port.py` → `T3_BERT_REAL_LOSS_GRAD_PASS`;
  smoke local 100 xe 1 round B + U1 (`group_rows=2048`, GPU 4 GiB) compile gate PASS, skip 0, không chứng minh 2×T4.
- Sim Lite2 xong 17/11 02:27, 12/12 audit PASS (`runs/sim-kaggle-lite2/{audit,verdict-r20}.json`, `report.txt`); watcher sim đã thoát.
  Luật khóa chọn **B + U1** (B + D8 FAIL TB −.0051; A32 + D8 PASS nhưng byte bootstrap cao hơn B).
- T3 DAGSNet/BERT đủr50/3seed; [comparison](../research/comparison.md).
- pFedES41BERT đãpull/audit đủr50; thêm lại mọi bảng/Notion, readbackPASS. [Bàn giao BERT](../../../fl_veremi_bert/docs/operations/continue.md)
  giữexactkernel/job/receipt; completion đã xử lý, không collector/launcher mới.
- [Task brief Fable5.2](../../task_brief.md) đã cập nhật đủbaseline; T3BERTcommunication hạng6 khi cópFedES.
  Owner chốt seed42, đơnmodelDAGSNet/BERT, sim→pilot→full, performance/communication/computetop5; riêng compute bằng FedAvg là đạt, không cần thấp hơn, Macro-F1 cáchbestbaseline≤.015, compute đích1×FedAvg (chốt mới17/11).
  Thắng nhóm trao thông tin khác là mục tiêu, không cổng thắng từng baseline. Brief đã bổ sung seed42/full roster, cost fairness, thiếu evidence (§13–15).
  Gate Lite2≤1,37× giữ nguyên theo thí nghiệm cũ; PASS không đồng nghĩa đạt đích mới.
- Dọncontext xong:102scriptstudyđóng vàreceiptvận hành cũ ra [archive](../../archive/operations-20261116/README.md).
  [Manifest](../../archive/operations-20261116/migration.json): source/test/DVC303filegiữhash,56scriptcòn dùng/helperfrozen.
  Link/AST/importPASS,208testactivecollectđược. Không đổi khoa học/verdict/runpayload.

## Bước tiếp

1. Task hiện tại = Lite2 theo [brief](../../task_brief.md) §1/§10; launch chỉ trong thang đã duyệt. Archive chỉ đọc khi cần evidence cụ thể.
   Owner chốt đích 16/11 (seed42): tổng byte ≤ 0,40 MB/xe/50r; compute ≤ 1,37× FedAvg; Macro-F1 r50 DAGSNet > .8998, BERT ≥ .8951
   (−.015 và top5); mở lại knob chi phí cũ như adaptation mới, control ghép cặp, gate mới khóa trước push (verdict cũ giữ nguyên).
   Probe byte local không train: `runs/lite-byte-probe/` (CDF/hist 64 bin, K 6–16, Cholesky 5/6-bit, ngưỡng ô moment bậc hai).
   Owner duyệt toàn bộ arm + gate (16/11); sim đã push 17/11 (mục Hiện tại).
   - Code xong, chưa commit: `methods/mobihet_lite.py` (bootstrap km2; km2 tắt = T3 bit-exact), tier 8 dòng trong
     `methods/mobihet_compute.py` (xoay lớp 0–7/8–15 theo step), `tests/test_mobihet_lite.py` 8/8 PASS.
     Suite: 204 PASS, 12 FAIL có sẵn do thiếu artifact DVC local (`test_r20*`, `test_mobihet_study`, `test_data_npy`), không do thay đổi này.
   - Preflight byte bootstrap, panel sim 200 xe, không train (`runs/lite-byte-probe/boot_preflight{,2}.json`; ngân sách ≈382 KB):
     T3 1 196 036 · **B** (K6, 6-bit, τ16, frac .25, 64 bin, mean fp16) 362 363 · A (K8, 5-bit, τ16, mean int8) 392 019 **vượt** ·
     **A24** (τ24) 380 664 · **A32** (τ32) 372 355 · B8 (K8, 6-bit, τ32, fp16) 422 343 vượt ·
     cô lập: I1 chỉ 64 bin 914 323 · I2 chỉ K6 590 185 · I3 chỉ τ16+frac .25 895 129. Assert moment lớp ≥ 2·D dòng PASS ở frac .25.
     Chạy lại: `V4_ROOT=<thư mục có label_mapping.json> python runs/lite-byte-probe/boot_preflight.py <out.json> "<tên>:<k=v;-k>"`.
   - Xong 17/11: smoke local B+D8 1 round PASS (`runs/lite-byte-probe/smoke-b-d8/`: compile gate PASS, skip 0, byte bootstrap 362 363,
     macro r1 .8044; lệnh smoke phải có `--compile`); `scripts/run_sim_lite2.py` (quota/launch/collect/gate); gate khóa ở
     [screening-lite2](../research/screening-lite2.md); 12/12 push OK (`runs/sim-kaggle-lite2/launch-complete.json`, `launch-gates.json`).
   - **Tiếp theo:**
     5. Khi `runs/lite2-full-r50/DONE`: đọc `watch.log`, `audit.log`, `audit.json` (verdict full: audit PASS gồm r1–20 trùng pilot +
        macro r50 > .8998 + bootstrap + mọi round ≤ 400 000 B + compute 1,364× ≤ 1,37×); ERROR → đọc log kernel trước khi quyết;
        dừng ở budget trước r50 (COMPLETE thiếu đích) → nối tiếp cùng owner từ state mới nhất, không là utility FAIL.
        Ghi kết quả vào screening-lite2 + tried index + comparison nếu PASS; FAIL → dừng arm, báo owner.
        BERT: DONE/audit.json đã có, pilot PASS (gate −.010, cùng ngưỡng byte); chưa thấy receipt launch r50 trong inventory đọc17/11 09:23.
        PASS → nối r21–50 (`gen_notebook.py --method mobihet_lite --run flb-lite2-bu1-s42-r20 --resume-kernel annhannh331/flb-lite2-bu1-s42-r20`;
        r21–50 ≈ 9,5–10 h train: đọc quota live, `annhannh331` còn ≈ 11 h sau pilot; thiếu → `--resume-dataset` sang account khác).
        Pull pilot BERT không gồm `state*.pt` (1,6 GB): state 2 rank chỉ được cell resume của notebook kiểm.
     6. Pilot (5 343 xe, r1–20 cos50, DAGSNet + BERT) vs T3 seed 42 cùng round: DAGSNet ≥ −.006, BERT ≥ −.010 (cả TB r16–20 và r20),
        byte bootstrap thật ≤ 382 000 B.
        Full r50: DAGSNet > .8998, BERT ≥ .8951, tổng ≤ 0,40 MB, compute ≤ 1,37×. Trộn DAGSNet+BERT sau khi đơn model PASS.
2. Owner đã yêu cầu push mọi thay đổi hiện tại **cảhai repo**, branch `main`; gồm brief/context/archive và kết quả pFedES.
   Receipt xác minh HEAD/remote/Gitbundle cuối: `../fl_veremi_bert/runs/maintenance-20261116/final-publish.json` tính từrootrepo.
3. [Backup](backup.md) giữ remote/restore; mount-sync kháccloud-sync. Không xóa artifact/cache hoặc addđèmanifestpartial.

Không cần thao tác tay hiện tại.
