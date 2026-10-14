# Nối phiên sau bật lại máy — 2026-10-13 UTC+7

Chủ dự án yêu cầu tiếp tục sau tắt/bật máy. Quyền/HP/cổng giữ quyết định đã chốt;
không Git/DVC push/backup/cleanup, không train nặng local, không rerun kernel có đủ artifact.

## Mốc mới nhất — 15:24 UTC+7

- V8 và V9 cả hai seed đã pull/audit. Hybrid utility paired PASS; residual M64/M512 paired FAIL.
- Full H1/H2 cont01 đủ output, audit round riêng mỗi arm: DML/own r7, lr4 r9,
  Gaussian full/diag/own r10, local r11; target vẫn r50. [10 metrics cuối](../../methods/shared-baselines/builds/federated-baselines-full/results/latest-rounds.md).
- BoundaryCells hai calibration seed đã push14:30/terminal/audit; tất cả cost PASS,
  chưa utility r20. [Report](../../methods/boundary-cell-counts/builds/boundary-cell-pilot/results/calibration-report.md).
- Hybrid full 4 notebook push14:57: first3/50, 100% train80%, seed42, đủ7 controls.
  W&B 7/7 arm có finite SGD/applied/skip trong ngưỡng, ngừng health polling.
  [Ledger](../../methods/gaussian-proxy-hybrid/builds/hybrid-full/results/rebuild.md). Collector `collect_hybrid_full.py` vẫn sống sau nối phiên.
- ResponseRisk và EMA cấu hình bổ sung đã được chủ dự án chốt; chưa pilot.
- Các bảng/process IDs/trạng thái chuẩn bị bên dưới là lịch sử đầu phiên, không dùng làm live state.

Tiếp: chuẩn bị BoundaryCells paired r20 với controls, giữ source/calibration checkpoint;
ResponseRisk/EMA source riêng, cổng utility trước pilot; H1/H2 cont02 dùng parent cont01 và quota/17Oct.

## Đã xác minh khi nối lại

- Không collector/report watcher local còn sống. Khởi động lại collect V8seed42, V9seed43,
  fullH1/H2 continuation và report watcher; giữ mọi `.part`, không ghi đè source đã chạy.
- Snapshot latest-version50kernel:43COMPLETE/6ERRORcũ/1RUNNING;
  full2cont01 cònRUNNING, full1/3/4cont01 terminal. Đây là status, chưa artifact proof.
  [Snapshot](../../operations/kernel-status-current.json).
- BoundaryCells source vẫn exactSHA với [smokeinitial2](../../../methods/boundary-cell-counts/builds/boundary-cell-pilot/probes/local-proof-initial2.json).
  5arm×3round/20client×256row,4packetrefusal, strictrebuild và full-vs-resume
  weights/state/predictions PASS local. Không T4cost/F1utility proof.
- Private input owner minhtran0601 READY, đủ114/114file và size listing;
  [access](../../methods/boundary-cell-counts/builds/boundary-cell-pilot/results/input-access-all-initial2.csv).
- Notebook cũ hai seed `bc-cal-s4243-20261013-initial2-v1` **chưa PUSHED**.
  Giữ bản chuẩn bị; thay bằng hai notebook, mỗi seed một session như workflow Kaggle;
  thêm `nb.metadata.kaggle` cạnh kernel-metadataJSON để khóa GPU/Internet.
  Scientific `proj/*.py`, fixture, LR20schedule/HP/cổng không đổi; không chạy lại smoke đã PASS.
- Hai notebook mới/controller8case PASS: normal, baseline thiếu commit,
  candidatefailed và parentbudget. CostFAIL vẫn là calibration hợp lệ; không tự nối utility.
  [Proof](../../../methods/boundary-cell-counts/builds/boundary-cell-pilot/probes/notebook-proof-initial2.json).

## Local jobs đã nối

| việc | log | session local |
|---|---|---:|
| V8 h,4downloadconnections | `papers/v8/collection-reboot-20261013.log` | 94689 |
| V9 a,2connections | `papers/v9/collection-reboot-20261013.log` | 95872 |
| Full collector,6connections | `papers/full-validation/collection-reboot-20261013.log` | 37815 |
| Report/index watcher | `docs/operations/report-watch-reboot-20261013.log` | 16803 |
| Input dinhan25 upload | `papers/boundary-cells/input-upload-dinhan25-initial2.log` | 69722 |

Session IDs chỉ dùng khi process còn sống; sau reboot phải kiểm process/marker trước khởi động lại.
Full1 pull đãđủ164file/1.973MiBnew, source audit đang kiểm từng arm. Chưa kết luận r50.

## Calibration chuẩn bị, chưa PUSHED tại mốc viết này

| seed | owner/kernel | input |
|---:|---|---|
| 42 | minhtran0601/mobihet-bc-cal-s42-20261013-initial2-v1 | owner42 READY/114file |
| 43 | dinhan25/mobihet-bc-cal-s43-20261013-initial2-v1 | owner43 đang upload clone private |

Mỗi notebook: baseline1DAGS +directP2/P4FL +teacherP2/P4FL,2completedround của20roundschedule,
all10metrics/group-old; W&B authonline trước train. Baseline trước,4candidate chia hai GPU.
Parentstartcut1,5h, worker≤0,5h ⇒ maximum khoảng2h, reserve2,3h/account gồm15%margin.
Fresh quota: owner42≈2,83h, owner43≈3,33h; đủ riêng từng account,
không lấy quota chênh trongRUNNING làm chi phí cuối. Global active vẫndinhan25.

Cost≤2×single-DAGS; FAIL giữoutput/dừng adaptation, không sửaHP để cứu.
R20utility còn chưa đo: mỗiP/mode/seed phải hơn own/shuffle≥0,010,
hơn strongest(local/classmean/Gaussianfull)≥0,015; benign giảm≤0,005.
[Contract nguồn](../../methods/boundary-cell-counts/builds/boundary-cell-pilot/results/contract.md),
[dự báo lợi hiệu năng](../../research/performance-feasibility-2026-10-13.md).

## Tiếp theo

1. Owner43 input READY/filelisting, budgetfreshreservation và private-key/sourceguard → push cảhai calibration.
2. InitialW&B/heartbeat health; ổn định thì ngừnghealthpoll, terminalcollect/audit vẫn tiếp tục.
3. V8/V9/full terminalaudit → report/bài học/10metric actualrounds; Hybridfull chỉ khi pairedV8PASS.
4. Calibration qua cost cảhai seed mới mở pairedutility với đủcontrols. Chưa modelutility từ boundedprobe.
5. ResponseRisk fidelity đãPASS closure/policy, chưa physicalcausal/utility; EMA recipe fresh2×2+own đã chốt,
   source/training chưa triển khai. Không tính chúng là pilotPASS.

W&B key chỉignoredprivate_push0600; privateversion/executednotebook giữ inlinekey theo ngoại lệ đã chốt.
Không cần thao tác tay mới tại mốc này; nếu auth/mount/GPU lỗi thật thì ghi yêu cầu cụ thể cho chủ dự án.
