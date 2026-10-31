# Phiên chiều15/10 — terminal audit, q2 và CrossClientRank

Chốt bàn giao15/10/2026 sau yêu cầu owner **dừng poll**; không remote call/launch thêm trong lúc chốt tài liệu.
Snapshot dưới đây là bằng chứng đã đọc trước yêu cầu, không trạng thái live. Đọc [CONTEXT](../../../CONTEXT.md) rồi [LESSONS](../../../LESSONS.md).
[Bàn giao đầu ngày](continuation-2026-10-15.md) giữ lịch sử trước batch này.
PUSHED/RUNNING và W&B không thay terminal artifact audit.

## Batch cũ đã collect/audit

12session đầu ngày:11COMPLETE/1ERROR maintenance. Mọi output có thể lấy đã pull xong,
exact executed/source/config/fingerprint, strict weights/BN/state, payload và predictions→10metrics đã audit.
Không còn bulk collector đang chạy. Full reports: [H1/H2](../../methods/shared-baselines/builds/federated-baselines-full/results/latest-rounds.md),
[Hybrid](../../methods/gaussian-proxy-hybrid/builds/hybrid-full/results/latest-rounds.md).

H1 `accanniez/mobihet-full1-s42-20261015-q1-v1` mất toàn output sau16299,7s:
owner Console xác nhận host maintenance/Docker daemon; API0artifact, executedv1 HTTP404.
W&B local25/Gaussian22 không có weights, không dùng làm checkpoint. Recovery từ audited14/10r16/r14.
Giữ kernel lỗi. [Proof](full1-unavailable-2026-10-15-afternoon.json).
DML audited20/own18; Hybrid local8/DML8/Gaussian15/Hybrid15/Gaussian-own15/DML-own8/Hybrid-own14.

| adaptation | bằng chứng cuối | quyết định |
|---|---|---|
| ClassTail | hai seed đủ8arm×r20; pooled−local−0,002683/−0,002533, −DML−0,196729/−0,191719; own/shuffle/classmass/benign cũng FAIL; cost PASS | không nối hoặc retune bins/λ để tìm điểm thắng |
| CountSketch repaired small | hai seed r20 qualityFAIL; decoded contraction PASS nhưng teacher/retention FAIL | không nối adaptation này |
| CountSketch repaired medium | r17/r14, kém aligned own0,2486/0,2577, last5 phẳng/giảm; chưa strict-r20 | không nối; không gọi đủ horizon hoặc bác cả họ EF |
| BoundaryCells P2/P4 direct/teacher | cont02 đủ12arm×r20 mỗi seed; cả4candidate đều trượt necessary strongest-Gaussian gate | không nối candidates hoặc controls còn thiếu của P2direct |

[ClassTail gates/forensics](../../methods/class-tail-risk/builds/class-tail-pilot/results/report.md),
[CountSketch](../../methods/proxy-countsketch-repair/builds/contractive-ef-pilot/results/report.md),
[BoundaryCells](../../methods/boundary-cell-counts/builds/boundary-cell-pilot/results/utility-report.md).
ClassTail late bin-floor/tail membership và BoundaryCells static weak teacher hỗ trợ chẩn đoán,
chưa cô lập nguyên nhân duy nhất. CountSketch co norm không chứng minh giữ ranh giới.
V9 M512 correction chủ yếu chung giữa anchors; không nối nguyên trạng.

## Six q2 full đã PUSHED

[Plan](continuation-plan-2026-10-15-q2.json):6×4h;62,1h cũ+27,6h mới có15%=89,7/90h.
Không query quota annhtnt/takids1909/accanniez theo owner. Source/lịchLR50 từr1 giữ nguyên;
20clients,100%train80%,seed42,target50 và đủ controls. Official test chỉ báo cáo,
không dùng chọn HP/round/continuation. Pilot nội bộ trước đó hỗ trợ DML/Hybrid.

| owner / kernel suffix | checkpoint import | GPU push UTC | initial fresh applied heartbeat |
|---|---|---|---|
| annhtnt / hybf1-s42-20261015-q2-v1 | local8/DML8 |08:04| có, r9 |
| annhtnt / hybf2-s42-20261015-q2-v1 | Gaussian15/Hybrid15 |08:22| có, r16 |
| accanniez / full1-s42-20261015-q2-v1 | local16/Gaussian14 |08:04| có, r17/r15 |
| accanniez / fulld-s42-20261015-q2-v1 | DML20/own18 |08:27| có, r21/r19 |
| takids1909 / hybf3-s42-20261015-q2-v1 | Gaussian-own15/DML-own8 |08:35| chưa, chỉ thấy history parent15/8 |
| takids1909 / hybf4-s42-20261015-q2-v1 | Hybrid-own14 |08:41| có, r15 |

Mọi dataset all-page names/sizes, mounted SHA, CPU production full-state/priors/history import và
4controller cases PASS trước GPU. Recovery H1 dùng W&B project
`mobihet-full-validation-recovery-20261015` tách history mất25/22; project không đổi scientific fingerprint.
Các proof `startup-q2-*.json` tại folder này. Không health-poll lại9identities đã ổn định.

Hybrid-own pair chưa xác định worker startup; owner thấy đang huấn luyện nhưng Console không load.
Giữ session, không cancel/rerun. Snapshot09:10UTC vẫn history parent, không nonfinite mới quan sát được;
RUNNING chưa chứng minh worker đã thực hiện round mới. Không cần thao tác tay thêm hiện tại.

## Hướng mới: CrossClientRank

[Nghiên cứu nguồn sơ cấp](../research/new-directions-2026-10-15-afternoon.md): exchange score moments,
pairwise squared-ranking loss trên real local records, không thêm teacher/sampler/synthetic book.
Không kế thừa guarantee shared-model AUC. SAM/ASAM chỉ recipe dự phòng, chưa được chọn triển khai.

Probe và diagnostic hai seed đã terminal/audit. Reuse audited CE-r2, train-only, không SGD mới:
fullT scan; max256/client/class seed1729 chia fit/test; own≥32 mỗi phía;
global≥128/≥5contributors; mỗi lớp≥5validclients. Projection16class-CE+15bias+1centered-logit,
fit-RMS/rcond1e−10, equal clients; support/signal16/16, coverage100%, locked probePASS.
Residual41878,0607/168,0373 không là lượng tri thức. Diagnostic giữ nguyên weights/sample/basis/gates:
condition≤5,20 bác SVD ill-conditioning; class5/client17 fit-scale nhỏ, coefficient817,95/58,76,
vài heldout record chi phối. Cap-ratio descriptive0,3540/0,3614; không thay locked gate.
[Báo cáo đầy đủ](../../methods/cross-client-ranking/builds/rank-probe/results/report.md).

Owner đã chốt [utility protocol](../../methods/cross-client-ranking/builds/rank-probe/utility-proposal.md):
r20 seeds42/43,8arms, λ0,25/T1/margin1, six CE-r2 forks và fresh Gaussian/DML;
primary group/private/gp true-class prior cố định mọi arm, head diagnostic.
FullT refresh mỗi round; common support pooled/own; shuffle means xoay1class; fixedr2.
Classweight NNLS16class-CE fit-half, βclip[0,4], loss(1+β[y])CE, không renormalize/exact match.
Gate mỗi seed: own/shuffle+0,010, CE+0,015, benign≥max(CE,strongest)−0,005;
chỉ mở dài nếu hơn strongestGaussian/DML≥0,015 và cost≤2×CE.

[Build rank-utility](../../methods/cross-client-ranking/builds/rank-utility/README.md) version1 hai seed PUSHED09:00UTC,
CPU mounted117files/116inputSHA/20strict warm models PASS trước GPU; local actual gradient/NNLS/SGD/replay,
atomic driver và8controller fixtures PASS. Source riêng, frozen40mh và probe source không đổi.
Seed42 ERROR sau hai validr3: CE state chứa empty `b''`, weights-only loader từ chối GLOBAL bytes.
Không phải numericalFAIL hoặc lệch phiên bản; fixture trước chỉ cover pooled, bỏ sót CE state.
Artifact r3 đã audit bằng scoped builtin-bytes allowlist, weights_only vẫnTrue;
algorithm/full-round cost1,255634/1,217979×CE đều PASS. Không dùng r3 utility để chọn HP.

Recovery version2 giữ nguyên source/fingerprint và checkpoint r3; launcher chỉ allow builtin bytes.
Không SGD lại r3, không resetr1. Seed42 CPU exact r3/state/wire/80metric checks PASS;
GPUv2 PUSHED09:20UTC, budget2,25h. Initial resumedr4 hai arm đã có heartbeat mới/finite;
dừng health polling seed42, chỉ còn seed43 chưa startup.
Seed43 version1 RUNNING, chưa heartbeat; chưa có artifact để audit/chuẩn bị recovery.
Budget recovery dự kiến2,25h/seed từ5h đã được phép, không cộng thêm batch90h.
Quota chỉ đọc minhtriethihi:09:13UTC còn6,20075h; không query ba owner mới.
Seed43 owner trả lời vẫn đang chạy; chưa có Console chi tiết hoặc W&B mới.
Giữ session, không push trùng; hiện không yêu cầu thao tác tay thêm. Platform RUNNING không thay fresh heartbeat.

## Việc còn thiếu — phiên sau

Không còn yêu cầu trả lời/chốt khoa học pending. Owner đã trả lời cả hai Console: vẫn thấy chạy;
không có log chi tiết. Không hỏi lại chỉ để lấy cùng trạng thái RUNNING; chỉ yêu cầu thao tác tay
khi xuất hiện lỗi/bằng chứng mới cần UI. Không có collector loop được để chạy nền.

- [ ] **Collect six q2 khi terminal**, không health-poll run đã stable. Full collector hỗ trợ `--once --only`;
  Hybrid gọi trực tiếp `collect()` một lần vì CLI hiện lặp mỗi55s và chưa có `--once`.
  Chưa có terminal artifacts/AUDITED của q2; chưa biết actual committed rounds mới hoặc đủr50.
- [ ] **CrossClientRank seed42v2:** chờ terminal collect, kiểm exact executed/source/compat SHA,
  inheritedr3 không đổi, full state/wire/predictions và actual rounds. R4 heartbeat chỉ startup proof.
- [ ] **CrossClientRank seed43v1:** giữ session đang có; khi terminal pull/audit trước quyết định.
  Nếu cùng emptybytes error và chỉ CE/pooledr3 hợp lệ, kiểm r3 cost≤2×CE cả algorithm/wall rồi mới dùng
  `gen_rank_recovery.py 43`. Không giả định lỗi/round của seed42 áp sang seed43. Nếu costFAIL: dừng.
- [ ] **Recovery43 nếu cần:** cập nhật budget từ runtime đã dùng và quota owner cũ, giữ tổng utility5h đã chốt;
 2,25h chỉ dự kiến, chưa launch. Dataset private `-r zip -t`, all-page names/sizes,
  `check_bytes_compat.py`, `check_recovery_controller.py`, actual CPU mountedr3 import/audit PASS trước GPU.
  Source40mh/proj và compat launcher đã PUSHED giữ bytes/fingerprint; không SGD lại r3 hoặc resetr1.
  `gen_rank_recovery.py` hiện yêu cầu đúng hai audited arms logicalr3; trạng thái khác cần xem artifact trước.
- [ ] **Nếu ERROR thiếu summary:** full/Hybrid collector hiện cần `runs/summary_s42.json`; phase
  `failure before driver summary` không chứng minh mất committed checkpoints. Kiểm markers/weights/resume
  từng arm, audit trực tiếp trước recovery; không để marker AUDITED phase-failure bỏ qua round hợp lệ.
- [ ] **Regenerate reports sau audit:** H1/H2, Hybrid đủ10metrics/per-class; CrossClientRank dùng latest audited
  attempt mỗi seed, inheritedr1/r2 không tạo metrics mới. Utility chỉ chốt khi đủ8arms×r20 mỗi seed;
  partial là partial, không FAIL do thiếu horizon. Post-warm costr3–20 riêng với r3 preflight.
- [ ] **Continuation tiếp:** lấy actual committed boundary, đủ controls và target/LR đã chốt.
  Envelope ba owner mới đã phân89,7/90h có15%, không coi90h còn nguyên hoặc tự mở q3 ngoài plan.
  Không query quota ba owner này. Nếu scope/budget không đủ, chuẩn bị plan cụ thể rồi hỏi owner.
  Không nối ClassTail/CountSketch adaptation hiện tại/BoundaryCells candidates đã FAIL.
- [ ] **Chỉ sau kết quả utility mới:** đánh giá CCR theo locked own/shuffle/CE/benign/strong-reference/cost gates;
  FAIL không đổi gp/gate/λ để cứu. ResponseRisk/EMA/SAM còn là đề xuất, không tự launch.
- [ ] **Kết phiên sau:** cập nhật CONTEXT/LESSONS/build docs/handoff, index/link/hash/ignore.
  Artifact đến 15/10 tối đã DVC backup ([backup](backup-history-2026-10-14-to-20.md)); artifact collect sau đó chưa. Account mới `nguyenminhtran060113`
  ([roster](kaggle-accounts-history-2026-10-15-to-17.md)) chưa xác nhận quota. Không Git/DVC push/cleanup nếu chưa có yêu cầu mới.

### Lệnh sẵn có — không chạy trong phiên bàn giao này

Từ repository root, env `nckh`. Mỗi collector dưới đây chỉ đọc một lượt, chỉ pull terminal output;
không tự bật loop. Không chạy hai bulk collectors cùng lúc nếu tổng connections vượt budget.

```bash
conda run -n nckh python methods/shared-baselines/builds/federated-baselines-full/scripts/collect_full.py --only '*20261015-q2-v1' --once
conda run -n nckh python -c 'import runpy; runpy.run_path("methods/gaussian-proxy-hybrid/builds/hybrid-full/scripts/collect_hybrid_full.py")["collect"]()'
conda run -n nckh python methods/cross-client-ranking/builds/rank-utility/scripts/collect_rank_utility.py
```

Sau terminal audit, mới regenerate:

```bash
conda run -n nckh python methods/shared-baselines/builds/federated-baselines-full/scripts/report_full_validation.py
conda run -n nckh python methods/shared-baselines/builds/federated-baselines-full/scripts/report_full_validation.py --build hybrid-full
conda run -n nckh python methods/cross-client-ranking/builds/rank-utility/scripts/report_rank_utility.py
```

Nếu seed43 cần recovery và đã có đủ artifact/cost/budget proof, các helper ở
`methods/cross-client-ranking/builds/rank-utility/scripts/`:
`gen_rank_recovery.py 43 --hours <budget đã tính>`, `check_bytes_compat.py`,
`check_recovery_controller.py`, `audit_recovery_mount.py 43`.
`audit_recovery_mount.py` chỉ đọc status một lần; CPU chưa terminal thì dừng, không loop.
Lệnh generate/check không upload hoặc GPU push. CPU/GPU launch chỉ sau các gate nêu trên;
public/private PUSHED phải đồng bộ, không sửa attempt đã PUSHED.

### Proof, giới hạn và tài liệu chuẩn

- Global [kernel snapshot](kernel-status-current.json) đọc09:01:51UTC, **cũ hơn lỗi/recovery utility**;
  không dùng con số RUNNING từ đó như trạng thái hiện tại. Initial exact-fingerprint startup q2 tại
  `startup-q2-*.json`; CCR tại `methods/cross-client-ranking/builds/rank-utility/probes/startup.json`.
- Q2 terminal collector/report chưa chạy trên output mới. CCR seed42v1 chỉ hai validr3 đã audit;
  v2/source-preserving recovery r4 mới W&B, chưa terminal audit. Seed43 chưa artifact.
- Emptybytes minimal refusal/scoped loader/subprocess PASS; actual tiny CE và pooled driver
  full-vs-resume r3/r4 exact PASS, orphan/tamper refusal PASS. Local chỉ synthetic20×512rows,
  W&B mock chỉ fixture; không production SGD local hoặc independent production weight inference.
- [Layout proof](layout-verification-2026-10-15-afternoon.json): DVC identities/remote manifests và frozen source
  giữ nguyên; local link/ignore checks PASS. Git HEAD vẫn`4a6893eeb8eaaeed799a2b59680a7b57cef13d4a`,
  stagedempty; không Git/DVC push/backup. Các artifact mới không tự có cloud backup.
- Entry point phiên sau: [CONTEXT](../../../CONTEXT.md), [LESSONS](../../../LESSONS.md), tài liệu này;
  contracts/measurements theo link build ở trên. Không lấy snapshot đầu ngày thay quyết định chiều15/10.

Giữ owner edits credential reference. Private inline key chỉ ignored private_push0600;
private Kaggle version history vẫn chứa key. Không ghi key vào public notebook/dataset/log/docs.
