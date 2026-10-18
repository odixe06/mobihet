# Snapshot lịch sử — trước tổ chức14/10

Quyết định mới nhất ở [CONTEXT](../../../CONTEXT.md), bài học chung hiện tại ở [LESSONS](../../../LESSONS.md).

# LESSONS — điều bắt buộc nhớ khi mở phiên mới

Đọc ngay sau [CONTEXT.md](../../../CONTEXT.md). Phần lớn bài học kế thừa từ archive 3 (`archive3:LESSONS.md`, 2026-10-10), đã lọc theo hướng mới.
Chi tiết và số gốc: [kiến thức kế thừa](kien-thuc-ke-thua.md). Bài học bị bác bỏ thì sửa tại đây, không xóa im lặng.

## 1. Chủ dự án và cách làm việc

- **2026-10-13, chủ dự án:** dự báo khả năng **cải thiện hiệu năng** từ bằng chứng đã có trước khi mở hướng/pilot;
  không dùng codec/đạo hàm/khả thi chạy code thay dự báo utility. Tách hơn local, hơn đối chứng mạnh và lợi kiểm dài.
  [Dự báo trước kiểm](../research/performance-feasibility-2026-10-13.md).
- **2026-10-13, chốt cải tiến/hướng mới:** BoundaryCells kiểm direct soft-target và teacher DAGSNet trung gian;
  receiver stress được swap đồng bộ 9 noise/profile feature cùng lớp/cùng client/khác receiver, chỉ local,
  phải qua fidelity gate. Fresh EMA backbone + BN local + head fit riêng kiểm 2×2 và own controls;
  giữ nguyên run dài đang hoạt động. Mọi learned model vẫn DAGSNet; R2 chỉ ở thiết kế.
- **2026-10-13, chốt BoundaryCells tiếp:** P2/P4, B4, z-scoreedges−1/0/1, alpha1/lambda0,5;
  probe hơn own/shuffle trước pilot. gp chính theo khối lượng target CE thực tế; gp legacy/head báo phụ.
  [Thiết kế/cổng đã chốt](../../methods/boundary-cell-counts/design.md).
- **2026-10-13, BoundaryCells utility/cost đã chốt:** mỗi P/mode, seed42/43 tại r20 phải hơn own/shuffle≥0,010,
  hơn strongest(local/classmean/Gaussianfull)≥0,015; benign giảm≤0,005. Cost≤2×local một DAGSNet,
  đo calibration trước rồi mới pilot; vượt giữoutput/dừng, không dummy-inflate denominator.

- **2026-10-13, cấu hình ResponseRisk/EMA đã chốt:** ResponseRisk giữ receiver-group
  g%20=1 từ train pilot làm probe, loại khỏi SGD ở mọi arm, giữ holdout cũ; swap9,
  λJS=0,1, refresh5 round, variance/mean², mean weight1/cap2, controls đồng nhất.
  EMA backbone half-life2 round, bỏ toàn bộ head, BN4096 local/head16 bước own real
  và synthetic hợp lệ của H2; gp cũ là chính, head/head-fit prior phụ. Không sửa full đang chạy.

- **2026-10-13, quyết định mới nhất ResponseRisk:** chủ dự án chọn **chỉ kiểm local augmentation trước**;
  chưa mở shared/own response-weight pilot. Không gọi kết quả augmentation local là lợi FL.
  EMA chọn cả nền Gaussian và DML; r20 mỗi seed: FL+EMA−FL plain≥0,010,
  FL+EMA−own+EMA≥0,015, benign giảm≤0,005, cost≤2× plain cùng nền; interaction báo riêng.
  Cổng chỉ mới chốt, chưa có source/training/utility proof.

- **2026-10-13, chốt đợt dài:** H1/H2 giữ7arm/100%train/50round/seed42;
  chạy theo quota hiện có, nối checkpoint sau refresh17/10. Không restart r1 hoặc đổi lịch LR50.
  Hybrid nếu utilityPASS cảhai seed:100%train/50round/seed42, đủ7compute-matched controls.
  [Phiên collect/continuation](../operations/continuation-2026-10-13.md).

- **2026-10-12, quyết định mới nhất:** các tiến trình đãhoạtđộngổnđịnh khôngcầntheodõitiếp.
  Chỉhealthcheck khi mớistart/chưarõ/lỗi; collect/auditoutputterminal vẫntiếptục.

- **2026-10-12, chốt tiếp:**K2 residual M64/512 teacher-first, commonprobe weights+BN cóphí,
  η1/L2trust0,25/targetclip±8/20publicstep/canonicalBN4096; K4commonGaussian-base+MH,
  τ0,05 local/energy0,1/20burn-in/1transition-batch. Khôngrawrecord rời xe.
  Chủdựán yêu cầu pushcác hướng khác đãsẵnsàng, khôngchỉ chờkernelcũ. K2/K4 khóa cost≤2×freshdenseCE T4.


- **2026-10-12, mở rộng:**H1/H2 full20client/100%train80%/50round/seed42-only,7arm local/full/diag/lr4/own/DML/DMLown;
  Nyströmrank4/8 explicitW/FP32up-FP16down/cổngdesign§7; Hybrid2×2/own pilot42/43.
  NotebookW&B online; giám sát banđầuround1–5; quyếtđịnhmới: đãổnđịnh thìngừnghealthpoll, chỉcollect/auditterminal. Cóbấtổn/lỗi quan sátđược khitrain đangchạy thì báo vàyêucầuchủdựándừng.
  [Kế hoạch/runđãpush](../operations/expansion-2026-10-12.md).

- **2026-10-12:** tổ chức theo tên phương pháp trong `proposals/`; source/artifact đã chạy giữ đường dẫn.
  Được khôi phục DVC chọn lọc để collect/audit; không tự add/push/cleanup.
  K5 chốt G51/G53 × marginal/joint mask; giữ primitive nhánh missing không xác định; hai cổng riêng.
  [Quyết định đầy đủ](../operations/decisions-2026-10-12.md).

- Trả lời tiếng Việt, kiểu caveman (CLAUDE.md). Chưa chắc thì **hỏi trước**. Việc phải làm bằng tay thì liệt kê từng bước.
- Đề xuất phương pháp phải có: **bằng chứng vấn đề → cơ chế → dấu hiệu kiểm chứng có thể sai**. Không thấy dấu hiệu thì dừng hướng đó.
- Phương án mới phải kiểm nghiệm ngắn để xác định khả thi. Bất hợp lý thì chỉnh từ nguyên nhân hoặc bỏ; **không kéo dài run để tìm điểm thắng**.
- Làm tới mốc nào cập nhật docs tới đó. Trước khi compact phải lưu kernel ID, source/config, kết quả đã verify, quota và bước còn thiếu.
- **2026-10-11: chủ dự án yêu cầu cập nhật đủ tài liệu rồi dừng phiên.** [Bàn giao](../operations/handoff-2026-10-11.md).
  Sau đó chủ dự án yêu cầu tiếp tục **chỉ viết đề xuất Markdown**, chưa code/chạy. [Bộ V2](../research/de-xuat-v2/README.md).
  Sau đó đã yêu cầu **triển khai kiểm khả thi V2**, chốt A cả2cỡ, B FO512/1024 trước, C theo cấu hình đề xuất.
  [Manifest/cổng](../research/pilot-v2.md), [ledger](../methods/shared-comparisons/builds/countsketch-subspace-ratio/results/rebuild.md). Git/DVC vẫn chỉ khi yêu cầu riêng.
- **2026-10-11, chốt trong phiên đề xuất V2: mọi mô hình học được đều DAGSNet**, không chỉ private/proxy;
  gồm teacher, encoder, model phụ. Không đưa tree/GMM học được hoặc MLP phụ vào phương án chính.
  Moment H2 gộp đã được phép vẫn là thống kê nền. R2 khác kiến trúc tiếp tục chỉ ở thiết kế.
- **2026-10-11 (chủ dự án): tổng hợp tài liệu phải có index toàn dự án.** [README](../../README.md) định tuyến theo nhu cầu;
  [docs/file-index.md](../../file-index.md) liệt kê từng file và vị trí artifact. Sau thêm/xóa/chuyển file, chạy
  `python3 scripts/update_file_index.py`; kiểm link trước bàn giao để phiên sau tìm đủ code/docs/notebook/skill.
- Không backup, không xóa artifact khi chưa được đồng ý trong phiên. Git/DVC theo quyết định mới nhất bên dưới.
- Kaggle: agent push trực tiếp sau preflight. Không ngại quota cho kịch bản ngắn; run dài hoặc nhiều client thì hỏi trước.
- **2026-10-11 (chủ dự án): mọi notebook huấn luyện tạo mới bắt buộc tích hợp Weights & Biases (`wandb`) API key
  và theo dõi online trực tiếp.** Khởi tạo/kết nối W&B trước train, bật Internet, log tiến trình mỗi round
  (loss, LR, bước applied/skipped, thời gian; đủ10metric khi đánh giá). Thiếu key/kết nối ⇒ fail trước train,
  không âm thầm offline/disabled. Key chỉ nhúng vào bản push private tạm dưới `papers/<method>/private_push/`
  (đã gitignore) theo workflow đã cho phép, không vào Git/log/chat; kiểm `git check-ignore` trước nhúng key.
  Notebook đã PUSHED giữ source; không sửa/rerun chỉ để thêm W&B. Quy định áp dụng từ notebook mới tiếp theo.
- **2026-10-11: tiến trình nặng (train/pilot nhiều arm, protocol đầy đủ) chạy trên Kaggle, không chạy ở máy local.**
  Local chỉ để smoke test, dựng fixture nhỏ, report, collect.
- **2026-10-11 (quyết định mới nhất): không tự commit/push Git, DVC add/push/backup; chỉ làm khi chủ dự án yêu cầu rõ.**
  Quyết định này thay quyền commit/push theo mốc đã cho trước đó. Tập trung nghiên cứu, kiểm nghiệm và tài liệu local.
- 2026-10-11: mọi client dùng DAGSNet, **kể cả proxy** (H1, H1+H2): cấu hình đồng nhất cho toàn bộ client. Chưa chạy đa mô hình.
- 2026-10-11 (phiên tiếp): được đề xuất ngoài H1–H6. RSU được đọc **codebook synthetic từ nhiễu**; cấm raw-init/record thật.
  Đây là ngoại lệ Q4 cho synthetic, không phải tự động có secure aggregation hay DP. DAGSNet/local-smoke được xác nhận lại.
- **2026-10-10:**
  - R3 cấm mọi dạng gửi mẫu `(x, y)`, kể cả lên RSU. **Không client nào thấy local data của client khác** (tiêu chuẩn FL, chủ dự án nhắc lại),
    kể cả trong arm đối chứng/upper reference của thí nghiệm. Số record cache cũ chỉ còn là lịch sử, không chạy lại.
  - Input 66 feature gốc, cùng protocol `fl_dagsnet_veremi`.
  - Được thử mọi dạng tri thức không phải dữ liệu thô, nhưng phải tính đến kết quả cũ.

## 2. Dữ liệu

- `train/` và partition FL **đã z-score**, `test/` **raw**: áp `scaler.json` đúng một lần. Kiểm: mean(`f_snd_spd`) test raw = 7,4493.
- Không chia lại dataset. Partition α=0,5 (20/50/100 client) là lệch tỉ lệ, không làm thiếu lớp.
- File client **sắp theo lớp**: luôn lấy mẫu ngẫu nhiên có seed. Khóa phiên: `(scenario, source_run, receiver_id, sender_alias)`.
- Shortcut:
  - Sybil có `f_first_in_session` và cụm feature điền 0.
  - Benign lấy từ run Sybil.
  - Vân tay receiver (9 feature noise/profile, η² 0,8–0,95) giúp đoán nhãn 80–90% trong một client.
- Trần do bộ sinh dữ liệu: timeDelay không để lại dấu hiệu trong nội dung tin (centralized F1 0,23). Đọc generator trước khi đổ lỗi cho phương pháp.

## 3. Đánh giá

- Holdout chia ngẫu nhiên theo dòng lạc quan +0,09…+0,12. Dùng temporal/receiver-group. Test chỉ để báo theo protocol khóa, không dùng chọn HP.
- Khóa **một** quy tắc quyết định chính (`gp`) và cổng **trước** khi chạy. Cặp seed, một yếu tố/arm, đủ 10 metric, F1 theo lớp, benign F1.
- Thành phần mượn mà chưa đo riêng thì phải đo trước hoặc thiết kế 2×2 (M15: interleave che mất lợi của cơ chế mới).
- Pilot → full scale: lợi thường còn khoảng một nửa, có khi đổi dấu. Slope ngắn không dự báo trôi. Lợi kiểu thêm tri thức tích lũy theo cycle.
- Probe offline chỉ là điều kiện cần (N3 offline + / online −). Thống kê tri thức phải tính trên dữ liệu chưa train.
- Kernel COMPLETE ≠ artifact hợp lệ: pull rồi verify bằng **đúng source đã chạy**.

## 4. Phương pháp — điều đã biết

- **2026-10-14, Hybrid full đã pull/audit đủ 4 kernel:** 7 arm, 18 round hợp lệ;
  local/DML/DML-own r2, Gaussian/own/Hybrid/own r3 của lịch50. Hybrid r3 gpF1≈0,704922;
  hơn hybrid-own cùng r3 +0,054330; hơn DML cùng r2 +0,054897. Tín hiệu full-data đầu
  có lợi, chưa r50 hoặc paired confirmation full. Interaction r2 −0,035280 vẫn âm.
  Gaussian gp giảm r1→r3 dù còn lợi so own: thêm epoch không tự tăng metrics.
  Không so round khác nhau, đổi HP/gp/best round theo official test.
  [Báo cáo/giới hạn](../../methods/gaussian-proxy-hybrid/builds/hybrid-full/results/feasibility-20261014.md),
  [đủ 10 metrics](../../methods/gaussian-proxy-hybrid/builds/hybrid-full/results/latest-rounds.md).

- **2026-10-14, cost controls Hybrid full:** cùng hai DAGSNet/số bước không bảo đảm cùng
  walltime. Nhánh không Gaussian worst round≈6.509–6.652s, Gaussian/Hybrid≈3.656–3.814s
  ở các arm đã đo; dummy/eval-BN work có phí thật, chưa profiling nguyên nhân duy nhất.
  Không dùng dummy-inflated baseline để tuyên bố cost hiệu quả chung. Budget phải theo từng arm
  và lấy max ở pairing hai GPU, cộng preparation/import/compile lặp lại.

- **2026-10-14, inference từ checkpoint:** Hybrid full42check và H1/H2 full64check PASS
  trên16.384 raw-test-prefix row/client0, private head/gp và proxy head tại r1/round cuối hiện có.
  Scaler một lần, decisive margin>0,02; giữ giới hạn prefix/client0, không full-inference/leakage proof.
  H1/H2 proof mới đã bao round cuối tới r11; hạn chế “chỉ r1–r3” ở ghi chú13/10 đã superseded.
  [Hybrid proof](../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/probes/inference-proof.json),
  [H1/H2 proof](../../../methods/shared-baselines/builds/federated-baselines-full/probes/inference-proof.json).

- **2026-10-13, Hybrid V8 đã audit cả hai seed:** hơn thành phần mạnh nhất
  +0,04070/+0,04355 F1 macro, hơn hybrid-own +0,14704/+0,14994; utility paired PASS.
  Interaction 2×2 −0,06769/−0,07224: thắng từng thành phần không chứng minh tương tác cộng hưởng.
  Mở kiểm full đã được chốt: 20 client/100% train80%/50 round/seed42, đủ 7 controls hai DAGSNet.
  Chưa có kết quả full r50 hoặc bằng chứng R1/R2. [Cổng](../../../archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/probes/gates.json).

- **2026-10-13, H1/H2 continuation đã audit:** private/test/gp hiện tới DML/own r7,
  Gaussian full/diag/own r10, lr4 r9, local r11; chưa r50. Cùng r7, DML−own +0,04512;
  cùng r10, Gaussian−own +0,04867. DML F1 tăng từ r3 đến r7, Gaussian full/diag/lr4
  giảm ở đoạn đã đo: kéo dài không tự cải thiện mọi phương pháp. Official test chỉ mô tả;
  không chọn HP/best round hoặc đổi gp theo các số này. Chưa cô lập nguyên nhân giảm.
  Audit source/weights/pred→CM/10metrics đủ round hiện có; sample-inference r1–r3 cũ
  không tự thành inference proof cho round mới. [10 metrics cuối](../../methods/shared-baselines/builds/federated-baselines-full/results/latest-rounds.md).

- **2026-10-13, BoundaryCells T4 calibration hai seed đã audit:** cả direct/teacher P2/P4
  qua cost≤2× single-DAGS; ratio 1,49–1,60. Direct P4−single tại r2 +0,00635/+0,00417;
  direct P2 đổi dấu, teacher thấp hơn single cả hai seed. Cost PASS cho phép pilot r20,
  không phải utility PASS; giữ mọi HP/cổng/control đã chốt, không sửa teacher để cứu r2.
  Benign F1 tăng trong khi macro ít đổi: cải thiện lớp lớn không bảo đảm cải thiện toàn bộ lớp.
  [Calibration/10 metrics](../../methods/boundary-cell-counts/builds/boundary-cell-pilot/results/calibration-report.md).

- **2026-10-13, receiver stress bounded fidelity:**4toy extractor case (finite/missing/nonfinite) closurePASS;
  40.951/40.960sample row có donor cùng lớp/client/khác hashedreceiver, giữ55feature ngoài block.
  Copy đồng bộ9noise/profile giữ identities của extractor; chưa bảo đảm sensor-generation/causal/utility.
  Thiếu donor phải skip, không mượn record client khác. gp theo CEtargetmass vẫn có label-shift assumption,
  không tự đảm bảo calibrated posterior. [Probe](../../../methods/receiver-response-consistency/builds/receiver-augmentation/probes/fidelity-probe.json).

- **2026-10-13, BoundaryCells bounded probe:**40.960sample train/4.096group row, không SGD/test.
  P4 hơn own/shuffle nhưng entropy/log16≈0,963/tableF1≈0,228; chưa utilityDAGSNet.
  P2 meancoverage92,87% che randomPositionOffset11,56%/zeroSpeed35,23%.
  Table transfer dương không bảo đảm cải thiện model mạnh hơn bảng.
  Shuffle giữ multiset target trên bảng không tự giữ mass/entropy khi lookup trên record.
  [Thiết kế/probe](../../methods/boundary-cell-counts/design.md).

- **2026-10-13, full H1/H2 calibration:**7arm đủr1–r3 với50roundschedule,
  43.045.415train/10.761.343testrow. Source/BN/weights/payload/pred→10metric auditPASS;
  toàn bộtest-label/orderhash và32sample-inferencechecks trênrawtest/scaleronce PASS.
  Prefix/client0 inference khônglàfull-inference/leakageproof. R3 privategpGaussian−local≈+0,0854,
  DML−local≈+0,1268; DML−own≈+0,0261. Đây earlydescriptive, khôngutilityr50,
  khôngđổiHP/decisionhead/bestround theoofficialtest. Chiphí đoT4 rồi budgettoànchuỗi;
  group2arm/GPU2 dùngmaxwalltime đểđọcquota, khôngcộng2GPU-time thànhsession-time.
  [Report](../../methods/shared-baselines/builds/federated-baselines-full/results/report.md), [budget](../../../methods/shared-baselines/builds/federated-baselines-full/probes/budget-measured.json).

- **2026-10-13, V7 Nyström đủ20arm×20round/hai seed đãaudit:** rank4/8 compressionPASS
  (byte/full≈0,198/0,321) nhưngqualitypairedFAIL. FL−exactlr4≈−0,0195/−0,0240;
  FL−exactlr8≈−0,0183/−0,0117, vượt mức giảm cho phép0,01.
  Rank8−shuffle≈+0,0109/+0,0125 PASS riêng; rank4−shuffle chỉ+0,0017/+0,0009.
  Hơnown/cócontribution/giảmbyte không tự giữ covarianceutility. Khôngtune rank/Ω/ridge
  hoặc kéo dài adaptation trượt cổng đểtìmđiểm thắng; khôngcontact/full adaptation này.
  [Report](../methods/covariance-nystrom/builds/nystrom-rank4-rank8/results/report.md), [cổng](../../../archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/probes/gates.json).

- **V9 residual teacher cả hai seed đã audit 26 arm-seed×20 round:** M64/M512 cost PASS nhưng quality paired FAIL.
  FL−own≈+0,0595/+0,0382; FL−classmean≈+0,0016 cảhai; FL−hard≈+0,0033/−0,0014.
  Hơnown không chứng minh mangranhgiới; residualđộng chưađủhơn thống kêtheolớp hoặc nhãnGaussian miễnphí.
  Refreshcontribution M64PASS/M512FAIL khôngđồngnghĩa teacherutilityPASS.
  Cảhai chọn h4 từ1.000.369pilottrainrow; r20fitMSE≈0,000171/0,001728 vẫnqualityFAIL.
  Coverage/fitPASS khôngchứngminh tri thức ra ngoài Gaussiananchorbook; chưacôlập nguyênnhân riêng.
  M512 điểmcaohơnM64 vẫnthualocal; khôngđổiM/bandwidth/bestround đểcứucổng.
  Seed43: M64/M512−local −0,26382/−0,13688; −classmean −0,00356/+0,00088;
  −shuffle +0,00042/+0,00325. M64 refresh contribution PASS, M512 FAIL như seed42.
  Paired confirmation đã có; không private KD/contact/full hoặc rerun y nguyên.
  [Report](../methods/anchor-function-residual/builds/anchor-residual/results/report.md), [cổng](../../../archive/methods/anchor-function-residual/builds/anchor-residual/probes/gates.json).

- **V10 Gaussian-base+MH cảhai seed đãaudit:**energyFL chỉ2round, cost/freshdenseCE≈2,969/3,107>2.
  CE-matched≈2,958/3,042, CEaug≈2,897/2,942; q-negative khôngMH≈2,323/2,248 cũngFAIL.
  Khônggánmọichiphí choMH: cảpositive/negative extra-forwards vàFP32 density training
  cùngcótrongadaptation; chưacôlậptừngphần. Numeric/source/weights/BN/prediction gatesPASS
  cho committedround;12candidatearm giữ2commit,2CErawcontrol đủ20round.
  Khôngcór20candidateF1: **costFAIL, qualitychưađo**, không gọiqualityFAIL hoặcphủđịnhJEM/FLfamily.
  **r1 sampler FL FAIL cả hai seed:** movement≈5,27%/6,31% nhưng có2/1chain đứngyên;
  seed43 chênh mean-energy hai khởi tạo≈0,177 pooled-SD>0,1. ESS energy≈4,20/4,04 mỗi chain.
  Probe local nhỏ PASS không bảo đảm sampler production; acceptance/movement riêng không chứng minh mixing.
  Dừngadaptation, khôngprivate/contact/rerun y nguyên. Sửa ghi chú r1PASS trước đây theo artifact đã audit.
  [Report](../methods/dagsnet-joint-energy/builds/gaussian-base-energy/results/report.md), [cổng](../../../archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/probes/gates.json).


- **V6 cả4kernel/30arm đãaudit:**12G59numericaloverflow,18control/G61 đủ20round;
  G61M/Jquality/compressionpairedFAIL. JointG61−full66joint≈+0,0075/+0,0079,
  −shuffle≈+0,0013/+0,0044; byte/control≈0,846. Owncontrastpositive chưađủgeometryproof.
  Joint−marginal≈+0,0197/+0,0223 nhưngfull66≈+0,0186/+0,0176; interaction nhỏ.
  Conditionalbootstrap≈14×marginal. Dừngadaptation, khôngcontact/fullscale/rerun y nguyên.
  [Kết quả](../research/probes/pilot-v6.md), [gates](../../../archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/probes/gates.json).

- **V5H actual target diagnostic:**4resume khôiphục6,69MiB,MD5/markerSHA/source/packetPASS.
  h4 chọn từ1.000.369trainrow của pilot; coverage≈98,92%/99,29%, không43Mfulltrain.
  Entropy/log16≈0,982; small CE−targetentropy≈6e−7nats nhưngF1≈0,06: objectivefit
  khôngchứng minh generalization. Targetargmaxalignment≈31%/36% (>6,25%chance),
  nên khônggọi hoàn toànuniform/vôtri-thức hoặcgánnguyênnhân duy nhất.
  Target trùngbit2seed: hai seed đo stochastictraining, không independentknowledge-estimation.
  [Chẩn đoán](../methods/kernel-anchor-distillation/diagnostics.md), [proof](../../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/probes/anchor-target-diagnostics.json).

- **V5 kernel-anchor teacher cảhai seed:**small/mediumpairedFAIL đãaudit trướccleanup,
  metadata khôiphụcexactMD5. FLhơnown chưađủ; thuaGaussianhardlabels vàlocal.
  Coverage khônglàteacherquality; khôngmởprivateKD/contact hoặc tuneh cứu cổng.
  [Bài học phương pháp](../../lessons/kernel-anchor-distillation.md), [report](../methods/synthetic-knowledge/builds/anchor-synthetic-books/results/report.md).

- **Audit mới đủcảhai seed:**MALApairedFAIL dùfinite/full20round; thuaown/shuffle/diag.
  GradientbookIPC1/4cost2×PASS nhưngqualitypairedFAIL; grad-trainedthuaGaussiancontinuous,
  khôngđủhơnmeanlatent/frozenencoder. Khônggánnguyênnhân duy nhất.
  [Tổng hợp](../overview/evidence-summary.md), [V4](../methods/gaussian-sampling/builds/support-mala/results/report.md), [V5](../methods/synthetic-knowledge/builds/anchor-synthetic-books/results/report.md).

- **K5 missingtoy actualextractor:**47case/94row synthetic, không realdata/model;
  NaN/nonfinite→0 làm nhiều công thức không thể suy từ primitive đã impute. G51/G53FAIL;
  giữ8derived⇒cùngG59, không chạy arm trùng. 7công thức eligible qua toy; chưaFP16/F1proof.
  Full59cov khôngđạtcarrier65%; chủdựánchọnqualityG59×marginal/joint32.
  [Proof](../methods/geometry-primitive-moments/fidelity-missing.json), [quyếtđịnh](../methods/geometry-primitive-moments/decisions.md).

- **Prototype trung bình lớp** (FedProto-like): so với không trao đổi +0,045, nhưng mọi cách sửa (cluster, InfoNCE, soft-label, tách bank, hard proto, projection head) chỉ ±0,006.
  Trung bình lớp không chở ranh giới. Đừng đầu tư thêm vào biến đổi trung bình lớp.
- **Thống kê latent bị cũ đi khi backbone còn học:** CCVR-like N3 offline +0,013, online −0,014/−0,010.
  Dùng chúng cho head triển khai trên snapshot đóng băng, không làm loss liên tục.
- **Thống kê lớp bậc hai ở không gian input mang được tri thức đa client mà không lộ dòng nào** (H2-T0-lite 2026-10-10, MLP, 4 client × 2 seed):
  Gaussian full-cov gộp +0,10, diag +0,05, điểm trung bình −0,02 so với replay cân bằng local. Thông tin nằm ở hiệp phương sai, không ở tâm.
  Input không trôi theo representation, nên tránh được lỗi của N3 latent. Chưa xác nhận trên DAGSNet/ngân sách/DP.
  [Kết quả](../research/probes/h2-input-stats.md).
- **Pilot P1 (2026-10-11, DAGSNet 20c, 2,5% dữ liệu, 20 round; [kết quả](../research/probes/pilot-p1.md)):**
  - H2 xác nhận trên DAGSNet: Gaussian full gộp +0,11; Gaussian của **chính xe** chỉ +0,024. ⇒ lợi là tri thức quần thể, không phải làm mượt.
    Lợi tăng theo cấu trúc bậc hai giữ được: diag +0,08 (13 kB một lần), lr4 +0,09, full +0,11.
  - H1 (DML) tăng theo sức chứa proxy: MLP64 +0,05, MLP256 +0,12, DAGSNet +0,16. Proxy toàn cục còn cao hơn private (0,57 / 0,68 / 0,77).
  - **Tri thức chỉ chạm head thì vô dụng khi backbone học thuộc:** LG +0,02, FedGen +0,02, FedGH −0,02.
    Dấu hiệu nhận biết: loss local r20 vẫn ≈ 0,01 như `local`.
  - **Teacher yếu thua nhãn miễn phí:** DS-FL trên điểm Gaussian (+0,075) thua chính nhãn lớp của Gaussian (+0,107).
    Ensemble của 20 model local ~0,46 làm nhãn nhiễu đi.
  - **`gp` sửa quá tay khi model private bị chưng cất về model toàn cục** (DML): `head` tăng đều nhưng `gp` rơi từ r15, benign F1 −0,17.
    `gp` giả định prior của private là π_i. Phương pháp kéo về toàn cục phải ước prior hiệu dụng, hoặc báo cả `head`.
  - positionMirroring **không** chạm trần thông tin: FedAvg +0,26. Gaussian (H2) và DML private không học được lớp này (+0,01).
    Chỉ timeDelay có trần (centralized 0,23).
- **KD/teacher:** teacher peer (Build7) và ridge teacher RSU (S13) chỉ ngang làm mượt theo lớp.
  Mọi teacher cần cổng chất lượng + đối chứng class-mean.
- **S-a hai seed đã verify (2026-10-11):** KME hơn Gaussian-target bank chỉ +0,0019/+0,0024,
  thua Gaussian diag liên tục −0,0412/−0,0442. DM so codebook chưa tối ưu −0,0002/+0,0036.
  Cả hai trượt cổng dù synthesis fit giảm mạnh. Khớp mục tiêu synthesis không chứng minh tri thức có ích.
  Chưa suy nguyên nhân riêng bank/encoder từ đối chứng hiện có; chưa phủ định họ FedDM/DP-MERF.
  Không tune/kéo dài run cứu cổng, không mở S-b cho hai adaptation tĩnh. [Kết quả](../research/probes/pilot-new-fl.md).
- Loss train H2/KME/DM được lưu là **CE real + CE synthetic**, không phải CE real riêng.
  Tổng gần0 ⇒ cả hai thành phần nhỏ; tổng cao chưa chứng minh real CE cao hay đã chặn học thuộc local.
  Chẩn đoán chống học thuộc cần tách loss và dùng holdout; không suy từ tổng loss đơn lẻ.
- **Rà soát thiết kế V2, chưa phải kết quả chạy:** ít byte không tự giữ sức chứa. DAGSNet với
  `θ=θ₀+Za`, K hệ số chỉ có≤K bậc tự do chung dù mọi tầng đổi được; kết quả fine-tune LLM không chứng minh
  học IDS từ đầu. Seed không tái dựng BN local hoặc xóa chi phí bootstrap/catch-up.
- **Rà soát density ratio V2:** real có feature rời rạc/điền0, Gaussian liên tục có thể khác support;
  không mặc định `p_real/q` tồn tại. Data/noise accuracy cao không chứng minh ratio hữu ích, clip không sửa support.
  [V2-C](../research/de-xuat-v2/c-density-ratio.md) đề nghị mục tiêu làm trơn local rõ ràng, ESS và đối chứng shuffle;
  augmentation/cổng vẫn là đề nghị chưa triển khai, mọi record local đã perturb vẫn không được upload.
- **Khác kiến trúc:** chỉnh head không cứu được backbone yếu (M2). Tri thức phải tác động toàn model.
- **Nút thắt:** ranh giới quyết định đa receiver. Xe sai 4% trên receiver của mình, 45% toàn cục.
  32 mẫu thật/lớp từ client khác ≈ 70% lợi tối đa (M1, số lịch sử; dạng này vi phạm R3, không chạy lại).
- **Chia sẻ model đi được trên dữ liệu này:** FedAvg/FedProx 0,77, NILM-FL proxy 0,82 / private 0,70 (test 20c, `fl_dagsnet_veremi`).
- **Công thức local có lợi:** `gp` (+0,025); EMA + head triển khai fit riêng mỗi cycle; tính lại BN local cho model EMA (lợi chuyển tiếp, giảm dần).
  EMA trung bình cả head thì hỏng.
- Trôi do học thuộc cục bộ có ở mọi cấu hình full data (CE 0,01–0,03); LR cosine chặn đà trôi.
- Loss train FL **tăng** ở cuối lịch cosine (FedAvg 0,28 → 0,45, DML) không phải trôi.
  LR ~1e-5 thì model vừa tải xuống hầu như không khớp thêm vào dữ liệu local, nên loss trung bình ≈ loss của model tải xuống. Đọc điểm đánh giá, không đọc loss.

## 5. Kênh truyền

- **P2 audit 2026-10-11:** lr4 giảm download nhưng upload vẫn145.792B; B0 không kích hoạt tri thức trong20round.
  DML B0 không nhận proxy mới vẫn hơn local do mutual learning tại xe; không gọi đó là lợi FL.
  Prior miễn phí và byte giả định fp16 chưa lượng tử hóa làm cổng metric chưa đủ chứng minh R1.
  Pilot phải lưu prediction/confusion từng client và weights để kiểm lại mean metric; confusion tổng không đủ.
  [Bằng chứng và giới hạn](../research/probes/pilot-p2.md).

- Ngân sách stress archive (`contact_scale` 0,01, `rate_scale` 0,2, cap 32 KiB) ≈ 4,5–4,7 kB/contact. Đó là knob, không phải số đo vật lý; bỏ scale ≈ 2,2 MB/contact.
- Kế toán byte: header, ACK, retransmission. Deadline cắt frame thì trả phí nhưng không commit.
  Không có gì miễn phí. Chỉ kích hoạt snapshot coherent.
- Đo idle/ACK/lệch nguồn của kênh hiện có trước khi thêm contact (ACK 14%, 37% contact bỏ trống, lệch cid corr 0,93).

## 6. Vận hành

- **2026-10-14, Hybrid full hai kernel ERROR do budget:** worker exit0/dừng trước r3;
  controller assert đủ3round khiến kernel ERROR. Local/DML/DML-own vẫn có r2 hợp lệ,
  Gaussian-own cùng kernel ERROR vẫn đủ r3. Không auth/nonfinite/OOM worker failure theo log.
  Đã audit toàn bộ checkpoint/metrics; không rerun từ r1 hoặc sửa source đã PUSHED.
  Continuation dùng last-round riêng mỗi arm và chấp nhận completed boundary theo budget.
  COMPLETE chỉ là session kết thúc, target r50 vẫn chưa đạt.

- **2026-10-14, crash giữa collect:** cả4 marker Hybrid full đã ghi trước crash;
  kiểm marker/process rồi tiếp report/inference, không tải hoặc train lại phần đã đủ.
  376file/0thiếu sau pull exit0; lưu log/hash/source trước chuyển sang bước kế.
  [Phiên/việc còn thiếu](../operations/collection-2026-10-14.md).

- **2026-10-13, nối sau reboot:** kiểm process thực trước khởi động lại collector, giữ `.part`/source/PUSHEDmarkers.
  BoundaryCells notebook chuẩn bị chưapush thiếu `nb.metadata.kaggle`; bổ sungGPU/Internet trongnotebookdocument
  vàtáchmỗiseed mộtsession trướcpush. Scientificsource/smokeSHA giữ nguyên, khôngreruntraining.
  Maskfalse không tự che giá trị raretarget trongpacket; sanitize trướcserialize. GP phải dùng priordecoded
  từpaidwire, khôngđọcservercountstate trực tiếp. [Bàn giao](../operations/resume-after-reboot-2026-10-13.md).

- **2026-10-13 live-log:** CLI hiệncó `kaggle kernels logs <owner/slug> --follow`
  vàSSE endpoint đãđọcthật trênfull2cont01. Ghi chú APIchỉRUNNING/no-log cũ khôngcònápdụng
  tuyệtđối. Dùngowner-scopedchild/timecap, redactkey trướclogpublic; không cầnownercopylog
  khiendpoint hoạtđộng. Full2 đãauth/2×T4/import r3/datafull trongstream dùW&Bworkers
  lúcđó vẫnoldr3. Noheartbeat chưađủkếtluận hung; pipconflictwarning khôngtựlàtrainingtraceback.

- **2026-10-13 datasetaccess:** `datasets status trietbackup/veremi-fl-20client`
  trả404 từaccountminhtrit06, nhưng `datasets files` cùngaccount/slug PASS.
  Khôngsuy inaccessible từprocessing-statusendpoint. Dùngowner-scopedfilelisting,
  rồi giữactualmount/datahash gate trongnotebook; metadataread khônglàfull-data-readproof.

- **2026-10-13 CPUmount preflight:** chỉthayGPUassert chưađủ; dònglog
  `get_device_name(0/1)` vẫnkhởi tạoCUDA trênCPUworker vàlỗi trướcimportgate.
  Giữ4notebookmount01PUSHED/source/log, sửa generator bỏmọiGPUquery vàkiểmstatic,
  dùngmount02 riêng. Khônggọi đây làcheckpoint/model/trainingfailure hoặcrerunGPUtraining.
  Mount02 sửaCUDAquery nhưng lộ thiếu dependency `check_v3_driver.equal` của frozenverifier;
  localimportPASS cóthể dựa vàoROOT/scripts sẵn có. Đóng gói đúng supportmodule ngoài
  scientificfingerprint, kiểmimport bằng isolated subprocess rồi dùngmount03. Khôngsửa fullsource đãtrain.

- **2026-10-13, collector thực sự dùng helper nào phải kiểm:**V8/V9 gọi `.claude/` còn
  full gọi `.agents/`; bảnClaude cũ chưaRange-resume và tựxóa`.part` khi kếtthúc.
  Khôngsuy tínhnăng từbảnAgents đãđọc. Đãsync helperhai nơi: kiểmContent-Range,
  appendđúngoffset, giữpartial, exitnonzero khi thiếu. Networkchange/EOF/readtimeout
  khônglàtrainingfailure. Khôngprint`repr(exception)` vì cóthể chứa signedprivate-outputURL;
  logtypeexception, giữtênartifact. Các logtải phiên này đãredact URL, khôngsửaartifact/source đãtrain.

- **Teacher cost/resume instrumentation:**preflight W&B heartbeat dùngcommit=False trước loground
  cóstep=r, tránhadvance step tựđộng trướcr đầu sauresume. Publicteacher_applied vàfitMSE
  báo riêng: K2clientresidualscan khôngSGD nênapplied0 khôngtự làtrainingerror.
  Cost2× soalgorithm_seconds của2completedroundfreshCE, khôngdummy-inflatedbaseline.
  Khôngtrộn walltime vàalgorithmstate đểđòibitwise. Sourceallmh frozen trướcfinalsmoke/push.


- **V7 notebook/math preflight:**full covariance upload packedupper-triangle, khôngfull66×66.
  Checker đầu dùngsai shape nênassertFAIL; sửachecker và kiểm lại math trướctrain,
  khônggọi lỗi checker làthấtbại phươngpháp. 10arm/4exactresume/payload/inferencePASS local
  chỉ chứngminhcode bounded; chưaT4/fullpilot/F1 hoặccontact.


- **Backupdelta2026-10-12:** restore/collectpartial không được thaymanifest đầyđủ bằng local snapshot.
  Gộpoldentry+localnew theo relpath, giữ entryremote-only/hashhistory. `runs/` cóthể làsymlinkalias
  vào`kaggle/`; backup data tạiđích, giữ mappingrelative đểrestorealias, không thêm bảntrùng.
  Chủ dự án chốt ghiG:/cacheD: với tốcđộgiớihạn đểsyncsong song; đủG: làcleanup, khôngchờcloudsync.
  G: không giữmode0444 khiến`LocalHashFileDB.oids_exist/check` đọc/hashlại payloadcũ, làmresume chậm.
  Wrapper tạm chỉ vớiremote/scope đãchốt: táidùngproof bằngexistence+size, MD5directory; filemớiMD5trướcatomiccommit.
  Khônggọi đây làrehash toànpayloadremote; không sửa thưviện/configtoàncục.
  [Chỉ mục pull](../../operations/artifact-index.md).

- **V6 G59 logterminal:**seed42cả6arm dừng bởi `V6 synthetic fp16 input overflow`,
  phầnlớn trướccommit r1, jointshuffle có1commit. ParentERROR doassertallarmscomplete,
  khônglàm mất3controlfull66 đã xong20round. KhôngW&B/auth/OOMfailure.
  Probe4096draw/3smoke khôngbảochứng tails khi productiondrawhàngtriệupoint.
  Chỉlog chưa chỉrafeature gâyoverflow; inverse(dist>0) cóđiểm illconditioned làcơchế
  khảdĩ, chưađổlỗicho duy nhấtbearing. Cảhai seed đãaudit. Guard kiểm cảhai syntheticdraw, kểcảdraw chưa dùng; không suy riêng input đãfeed overflow.
  [Manifest](../research/pilot-v6-geometry.md), [live](../operations/expansion-2026-10-12.md).

- **V3 cả8kernel/48run đãaudit:**R/T/O/P đều pairedFAIL, khôngnumericalfailure.
  MediumR giữ≈45,0%/44,4%, O≈28,0%/20,8%, PowerSGD≈8,8%/15,2% lợi dense32−own;
  cổng yêu cầu≥80%. EF contribution và PowerSGD adaptive-basis contribution đềuPASS,
  nhưng khôngđồngnghĩa compressionutilityPASS. Rollingtrain hơncommunication-only
  nhưng thuaowncodec. Dense32teacherPASS; khôngđổihead/bestround đểcứuprivategp.
  [Gates](../../../archive/methods/proxy-compression/builds/proxy-carriers/probes/gates.json), [report](../methods/proxy-compression/builds/proxy-carriers/results/report.md).

- **V6 checker-only lỗi device:**15armtrain/4exactresume đã chạyqua, customverifier
  loadresume không `map_location=CPU` rồi so vớipacketCPU gâydevice-mismatch.
  Sửa checker, verify lại outputcũ; khôngreruntraining/source khi modelkhônglỗi.

- **V6 math probe lấy prefix raw fixture sai:** `train_y[:256]` trên20client
  chỉ phủbenign/accelerationMultiplication, 14lớp không córecord. Globalcohortguard từchối
  làđúng; sửa probe thành512uniformseededrow/client, không sửa sourceV2–V5.
  **Không suy smoke cũ thiếu14lớp:** `P.Run` chọn3000random trước `v2_run` trunc256;
  prefix trong probe mới khác prefix sau random của driver. Phải đọc đúng chuỗi preprocessing.

- **Khôi phục metadata 2026-10-12:** `dvc pull` một batch nhiều đường dẫn con
  không dựng metadata như dự kiến và báo `D` trên hai thư mục đang collect.
  Không checkout DVC vào output đang có writer. Dùng manifest directory đã kiểm MD5,
  đọc object remote và kiểm MD5 từng file để restore chọn lọc; không thay manifest/cache cũ.
  Phải kiểm đủ file sau restore; exit0 riêng không chứng minh target con đã có.

- **Backup2026-10-12:** DriveFS cache trênD: đầy cóthể báo `Errno5 Input/output error` khi ghiG:,
  không phải bằng chứng artifact nguồn hỏng. G: đủ object+manifest là gate cleanup, không chờ cloud sync.
  Push tổng lỗi vẫn được cleanup từng target đãpush exit0/status in-sync/verify độc lập; giữ workspace/cache của target thiếu.
  DVC local backend hiện tại ép copy `batch_size=1` dù `-j`; wrapper4worker chỉ dùng tạm cho đợt này.31 target/8.847 object đãverify đủ rồicleanup, không còn target thiếu.
  [Hồ sơ](../../operations/backup.md).

- **K5 algebra bounded train probe:**40.960rows/259MiB,17identity khôngvi phạm interval
  sauFP16ULP/scalerpropagation;6rows denominatorinterval chứa0, missingheading/coincidence
  branch chưa phủ. Bearingcos center-error tới0,782/closing20,07 dùintervalPASS.
  Không suy faithful reconstruction từbound rộng; G51/G53 cầnchốt trướcpilot.
  [Bằngchứng](../../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/probes/algebra-probe.json), [thiếtkế§10](../research/v5-algebra-sources.md).

- **V4support cảhai seed đãaudit:** structural−diag−0,00212/−0,00824;
  −own+0,10153/+0,10926,−shuffle+0,07744/+0,06789,−marginal+0,00347/+0,00549.
  PairedFAIL; first-sessioncontribution chưaquacảhai seed, khôngcontact/rerun.
  Fidelity0invalid khôngchứng minh IDS tốt hơn; không đổiheadline/head hoặcHP đểcứucổng.

- **V5 exactresume:** weights/books/packets trùngbit; mismatch duy nhất ởwalltime
  `algorithm.diag.condensation_seconds`. Thời gian làdiagnostic trongroundmarker/report,
  khôngđược trộn vàostate thuậttoán cầnbitwise. Sửa trướcpush, giữsmoke cũ; source cuối kiểm tagmới.

- **Clipped-ratio score probe trên C đãverify:**64freshGaussianpoint/lớp,FP32,2seed;
  r_y≤−10 khoảng57,8%/63,6%, median `||score(z)+z||`=0 ở13/16 và15/16lớp.
  Trong vùngclipfloor, ratio không cóinputgradient hướngdẫn MALA; Gaussian drift/noise vẫn hoạt động.
  Không suy MALA vôích hoặc20burn-in đãmixing từprobe này; không thayeta/cap đểcứupilottừdiagnostic.
  [Bằngchứng](../../../archive/methods/gaussian-sampling/builds/support-mala/probes/ratio-score-probe.json); snapshotC lịch sử, không phải teacherV4 đãtrain.

- **K2b coverage local sample:**40960trainrow, M32/512; h1/2 khôngqua90%/80%class,
  h4 coverage≈98,8%/99,2%, nhưng normalizedsofttargetentropy≈0,983/0,982 vàmaxprob≈0,089/0,091.
  Coverage cao cóthể đi kèmtargetgầnuniform; không coi coveragePASS làteacher/F1PASS.
  Chưa chọnbandwidth hoặc fulltraingate. [Probe](../../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/probes/anchor-probe.json).

- Fingerprint driver hash **toàn bộ `mh/*.py`**, kểcả module mới chưa dùng bởiarm.
  Không thêm/sửa module khi đang kiểm full-vs-resume: V4MALA smoke đã bịguard từchối resume do
  thêm schema/support giữacheck, không numerical failure. Giữoutput cũ; smoke lại tagmới trênsource đóngbăng.

- **V2-C cảhai seed đãverify đủ8arm×20round:** FL−diag −0,01666/−0,02082;
  FL−own +0,00128/−0,00172; FL−shuffle −0,00932/−0,00845. ESS fraction từnglớp≈0,007–0,024.
  Clip/class-normalize không tự tạo sampling hiệu quả; ratio fit không chứng minh tri thức hữu ích.
  C FAIL, không contact/rerun y nguyên. Chưa cô lập support/ESS như nguyên nhân duy nhất.
- **K3 local mixed-derivative probe:** DAGSNet FP32/eval,32trainrow,IPC1/4×4step PASS;
  peak1732MiB. Chỉ khả thi đạo hàm, chưa F1 hoặc cost2× trênT4. Runtime2.10 remote vẫn phải preflight.

- **V2-B seed42 terminal:** K512 và own512 đều lỗi `proxy loss` không hữu hạn tại client0/r1,
  trước round commit đầu tiên. Không phải auth/OOM; không rerun y nguyên. Local4step/batch32 finite
  không chứng minh full epoch/batch512 ổn định. Cổng từng K độc lập, lỗi512 không được che kết quả1024.
- **V2-B cảhai seed đãaudit:** seed43K512/own512 lỗi tại client17/r1, không córoundcommit.
  K1024teacherhead≈0,30781/0,33803, retention≈0,299/0,373 của dense−own, thua localprivategp.
  Teacher gate FAIL; không mởprivateKD,ZO/contact; không phủ định toànhọseed-based FL.
- **K1support preflight:**40960trainrow sample,7337first-session,0vi phạm13zero-imputed atom;
  firstflag và2nhómprofile binary hợp lệ,profileone-hot0vi phạm. Atom phảiđổi bằngscaler rồi lượng tử
  nhưfixture; độc lậpBernoulli3profile cóthểsinhmulti-hot. Đây làsample/schema proof, khôngF1hoặcfullscan.
- **DAGSNet/JEM preflight:**head cóLayerNorm trước2Linear/ReLU, logits bịchặn tuyệtđối khiweights hữu hạn.
  TrênR66 khôngbase measure/domaincompact, `sum_y exp(logit_y(x))` cólowerbound dương nênintegralvôhạn.
  Không lấy classifier DAGS làm densityLebesgue chưa chuẩn hóa. Gaussianbase/mixturebase hoặcdomainhữu hạn
  phảiđịnhnghĩa trước sampling; finiteSGLD20step không chứng minhnormalizability/mixing.
  [Probe](../../../archive/methods/proxy-compression/builds/proxy-carriers/probes/energy-probe.json) kiểmheadbound bằngcheckpointdense đãverify; kết luận chính từđại số.

- **V2-A remote2026-10-11:** kernel ERROR có thể xảy ra sau khi nhiều arm đã hoàn tất.
  Thu từng arm/checkpoint trước quyết định rerun. Seed42: EF small/medium tăng loss rồi BN calibration
  tạo state không hữu hạn; không chạy lại y nguyên để cứu cổng. Chờ audit hai seed để chốt bằng chứng.
- **V2-A checkpoint seed42 đã audit:** sketch-space EF không mặc định là phép nén co.
  Sau decode top-k rồi sketch lại, residual/target ở r2–4 của small≈16,75–17,92;
  medium r2–6≈2,93–3,97. Residual tăng small46→238.090 và medium28→16.296.
  Parameter max small≈4.703, medium≈182 trước BN AMP overflow. Bù lỗi tại sketch có thể khuếch đại collision;
  codec đúng/resume đúng không chứng minh cơ chế ổn định. Không kết luận toàn bộ FetchSGD/EF bị bác bỏ:
  adaptation này không có momentum/heavy-hitter guarantee như nguồn. [Chẩn đoán](../../../archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/probes/sketch-diagnostics.json).
- Notebook private-inline: Kaggle tự sinh `__notebook__.ipynb` chứa source/key và HTML liên quan.
  Quét key trên artifact do driver tạo (`runs/`, `source/`), không quét notebook nền tảng để kết luận training lỗi.
  Bulk pull chỉ lấy `runs/`, `source/`; executed source có key chỉ tải vào ignored private_push, audit bản redacted.
  Notebook A đã PUSHED giữ nguyên; B/C chưa push đã sửa bộ quét. Quyền private-inline có chi phí lưu key trong history.

- **V2 local smoke:** GPU CountSketch `scatter_add` atomic có thể khác thứ tự cộng giữa hai lần chạy,
  residual r1 lệch bit dù decoded patch/weights trùng. Exact-resume cần deterministic algorithms và
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` trước CUDA; kiểm lại state, không chỉ prediction/weights.
  Đây là bài học numeric backend, không phải kết quả F1 hoặc bảo đảm bitwise giữa account/thiết bị.
  CUDA median-with-indices bị deterministic guard chặn dù chỉ lấy values; dùng stable-sort middle value,
  stable top-k/index tie. Không tắt guard hoặc dùng warn-only để gọi kiểm exact-resume đã qua.
- Local: RAM 7,6 GiB, RTX 3050 4 GB. Job nặng **không** chạy local (chủ dự án 2026-10-11). Job vừa (smoke, fixture) chạy qua `run_local_guarded.py`.
- Đọc parquet bằng `pyarrow.dataset` phải tắt readahead (`batch_readahead=1, fragment_readahead=1`), nếu không RSS lên 4,6 GB.
  `centralized/test/` có lẫn `*.stats.json`, nên glob `*.parquet`.
  `conda run` nuốt stdout. Chờ theo PID, không `pgrep -f` (tự khớp chính lệnh chờ).
- RTX 3050: DAGSNet train ~20k dòng/s một job (bị giới hạn bởi launch). **Hai job song song** cho tổng ~30k dòng/s (×1,5), peak RAM ~4,5 GB.
  Sửa module mà hàng đợi đang dùng: chép ra bản sao, smoke test, rồi `mv` thay nguyên tử. Job sau đọc file mới, job đang chạy không bị ảnh hưởng.
- `xargs ... bash -c '...' arg0 arg1`: `$0` là arg đầu, nên `shift` chỉ bỏ `$1`. Đọc `$?` **trước** `$(date)` trong cùng lệnh echo.
- Kaggle:
  - 2×T4, image pin. Tiêu đề ≤ 50 ký tự. Không sửa notebook đã PUSHED.
  - Budget phiên = log thật + dư ≥ 15%. Nạp checkpoint về CPU.
  - Pull bằng `pull_kernel_output.py` (Range resume). Collector cần cả `python` lẫn `kaggle` trong PATH env `nckh`.
  - Kaggle không trùng bit giữa các account. Preflight phải chạy vòng arm của driver với đúng số cycle production (lỗi E14: arm 2 chạy 0 cycle).
  - S-a seed42: từ push→COMPLETE khoảng123phút, worker log50,2phút. Không đồng nhất thời gian chờ dịch vụ với train/GPU-time.
    Quota free khi RUNNING13,6791h tăng thành14,3519h sau COMPLETE; báo chi phí từ snapshot đã kết thúc + log,
    không coi chênh quota khi RUNNING là chi phí cuối. [Ledger](../methods/synthetic-distribution-matching/builds/static-synthetic-books/results/rebuild.md).
- DVC:
  - Chỉ `dvc add` workspace đầy đủ.
  - `.git` phình vì `tmp_obj_*` sau push bị ngắt: kiểm `git count-objects -vH`.
  - Chỉ xóa local khi chủ dự án yêu cầu; không chờ cloud sync, đủ khi object đã lên G:.
