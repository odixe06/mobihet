# CONTEXT — MobiHet v2 trên VeReMi NextGen v4 (5.343 xe)

Đọc CONTEXT → [LESSONS](LESSONS.md) + [bài học vận hành](docs/operations/lessons.md) → [bàn giao](docs/operations/continue.md).
Lịch sử MobiHet v1 (Build1–P37, 20/50/100 client, 66 feature): [history/v1](archive/legacy-v1/CONTEXT.md) — **giữ mọi lesson, đọc theo chủ đề**.

## Bài toán

IDS V2X16lớp;5343xe nhận làclient. Mỗi xe giữ private model, có thể khác kiến trúc; trao đổi thông tin gộp theo schema
đầu ra lớp, **không share local data** (không dòngthật/featuretừngmẫu/nhãntừngdòng). MobiHetv1 dùng proxyQ3; v4 đang kiểm
thiết kế bỏproxy, chọn thông tin đánggửi lên/xuống. Owner31/10 cho phép xây lại từ kiếnthức đãcó, ưu tiênF1 và báo trade-off
byte/thời gian, chưa trầnbyte; giữDAGSNet chung trước. [Thiết kế](archive/docs/research/federated-design.md).
Protocolbaseline [odixe06/fl_dagsnet_veremi_v2](https://github.com/odixe06/fl_dagsnet_veremi_v2): [50round,mọixe,1epoch,W1,AdamWcosine,own-test](docs/research/protocol.md).

## Quyết định owner

- 30/10: tách repo `mobihet` mới (repo cũ → [`odixe06/mobihet-archive-veremi-v1`](https://github.com/odixe06/mobihet-archive-veremi-v1),
  archived, public; artifact DVC v1 trên OneDrive — [backup](docs/operations/backup.md)).
- 30/10 — hướng cải tiến: (1) train proxy song song như model thứ hai tốn tài nguyên/thời gian; (2) synthetic (Gaussian gpool) chưa hiệu quả;
  (3) cải thiện distill proxy ↔ private; (4) **cố gắng bỏ việc train proxy**. Kế hoạch: [improvement-plan](archive/docs/research/improvement-plan.md).
- Kế thừa v2 (29–30/10): W = 1; gộp theo n_k; một seed 42; W&B online (project `mobihet-v4`); repo public, không đẩy data/PDF/credentials;
  commit không có `Co-Authored-By`.

## Trạng thái — 01/11/2026 ~01:22 (giờ VN)

- Owner chốt **M: chẩn đoán 20 round**, own-test từng xe là chính; tập trung ba khâu upload → server aggregate → client học,
  xe ít mẫu/nonIID và giảm chênh chất lượng, không giảm xe mạnh để thu hẹp khoảng cách. Chưa benchmark/chốt phương pháp tổng.
  [Spec M16run](docs/research/screening-m20.md): augmentation/noReal/gate controls, fallback/postepochQ/server√n,
  uniform vs local class-support mixture, hai seed42/43; cosine50/full5343/W1 giữ nguyên. M0–M15v1 exactCOMPLETE, đủ20roundmetric/CM,source/config/T4/compile2rankđãkiểm.
  Owner tắtmáy làm watcherPID46308mất; statecũrunning làstale, khôngrestarttraining. Quickanalysis đãcó,
  đangkhôiphụcfullsnapshotpull/audit trước kếtluận vàbatchmới; runtimeởbàngiao.
  Local47tests/3controlCPUbitexactfrozenL/real4xeW1compiledAMP+strictbankrebuild PASS. [Bàn giao](docs/operations/continue.md).
- `AGENTS.md` đã thêm quy tắc skill global **codex-long-jobs**: worker poll1200s, khôngmodelpolling khi chờ,
  completion phải kiểm state/log/artifact; runtime job ID/PID ở bàn giao, không đặt trong AGENTS.

- **L0–L11v1 pull/auditPASS**,120testCM/48trainCM/1632proberecord/120snapshot,4.331GiB.
  [L kếtquả và cơchế](archive/docs/research/results-l10.md): softfixedgateL6/L7=.887030/.887433; softfixednoReal=.886315/.886861,
  đềuPASSgainhai seed. Fixed1 tốt hơnadaptive; softteacher cógiátrị khikiểmđúngscale, lợirealgate nhỏcầnlặp.
- Watcher20phút vàcollect/auditlongjobs đềusucceeded, completionđãconsumed; **khôngcònjobL/watch/collect chạy**.
  [Bàn giao](docs/operations/continue.md). Chưa50round/heldoutđộclập/mixedarchitecture; batch sauL làM20 đangchạy.

- Owner chốt **bắt đầu thực hiện**. [BatchL12run](archive/docs/research/screening-l10.md) khóa target×scale×realgate,
  thêm4codeccontrol. L0–L11v1COMPLETE;12/12xácnhận2×T4/source/W&Bauth lúc14:21;
  đãaudit10round/trainCM/probe/snapshots. [Runtime](docs/operations/continue.md).
  Local36testsPASS, 3caselegacybitexact, smoke4xe/2roundW1compiledAMP+train/testCM+strictrebuildPASS.

- **K0–K9 COMPLETE, đã pull/audit đủ10round**, source/HP/LR/5343ID/support/10metric khớp,100snapshot hữu hạn.
  [K kết quả và khả thi](archive/docs/research/results-k10.md). Bộmainpull~3,59GiB, khôngpull fullprivate state.
- GateK8/K9 F1=.883935/.884584 **PASS cảhai seed**; seed43 sátngưỡng0,000076. AugmentationK4/K5=.883348/.884334,
  chỉPASS42 nhưng p10/xeítmẫu tốt hơn và0learnedB/round saubootstrap. Combo/conditionalFAIL.
- Cảhaihướngcòn bootstrapcopula3,31MB/xe. Chưa50round/kiếntrúctrộn/contactevaluation; chưa chốtmethodtổng.
  [Phân tích nguyên nhân và đối chứng tiếp](archive/docs/research/root-cause-experiments.md): gate chỉ tắt realKDtimeDelay r2–10;
  đãtách target×loss scale×realKD, instrumentgradient và kiểmcodec trongL. Chưa50round.
- CLIglobaldinhan25 giữ nguyên; MCPJSONrepo đãxóa. KhôngGit/DVCpush, khôngNotion, khôngcleanupdata.

## Cần biết → mở

| câu hỏi | tài liệu |
|---|---|
| Đang làm gì, bước tiếp, việc tay? | [continue](docs/operations/continue.md) |
| Protocol chung (giống repo baseline) | [protocol](docs/research/protocol.md) |
| Cải tiến MobiHet: phân tích, thí nghiệm, cổng, đề xuất tiếp | [improvement-plan](archive/docs/research/improvement-plan.md) |
| Kết quả sàng lọc S0–S4, G0–G4 | [screening-r10](archive/docs/research/screening-r10.md) |
| H0–H4: giả thuyết, cổng, source, bootstrap | [screening-h10](archive/docs/research/screening-h10.md) |
| Teacher theo mẫu, dataset chẩn đoán, batch J | [screening-j10](archive/docs/research/screening-j10.md), [nguồn KD](archive/docs/research/synthetic-kd-next.md) |
| Xây federated từ bài toán, kiểm chứng hai seed K | [federated-design](archive/docs/research/federated-design.md), [screening-k10](archive/docs/research/screening-k10.md), [results-k10](archive/docs/research/results-k10.md) |
| Nguyên nhân học yếu, đối chứng nào tiếp? | [root-cause-experiments](archive/docs/research/root-cause-experiments.md) |
| BatchL đã khóa target/loss/codec/probe ra sao? | [screening-l10](archive/docs/research/screening-l10.md) |
| L đã kiểm chứng nguyên nhân nào, còn vấn đề gì? | [results-l10](archive/docs/research/results-l10.md) |
| M20 chẩn đoán upload/server/client và xe ít dữ liệu ra sao? | [screening-m20](docs/research/screening-m20.md) |
| Chi phí tính toán / băng thông | [cost](archive/docs/research/cost.md) |
| Dataset, DAGSNet, máy | [knowledge](knowledge/README.md) |
| Kaggle, lỗi kernel, backup | [kaggle](docs/operations/kaggle.md), [sổ lỗi](docs/operations/notebook-errors.md), [backup](docs/operations/backup.md) |
| v1: thiết kế arm chốt, kết quả 20/50/100 client, synthetic P23–P37 (FAIL), mọi lesson | [history/v1 CONTEXT](archive/legacy-v1/CONTEXT.md), [design](archive/legacy-v1/docs/methods/full-drift/design.md), [LESSONS v1](archive/legacy-v1/LESSONS.md) |
