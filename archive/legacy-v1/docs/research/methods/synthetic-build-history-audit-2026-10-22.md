# Sinh synthetic có chủ đích — rà bằng chứng build cũ phục vụ CCVR hiện tại

Ngày 22/10/2026. Rà tài liệu/source/artifact **local**, không chạy GPU, không pull/launch Kaggle.
Mục đích: tránh lặp adaptation FAIL, giữ lại control và cơ chế còn có ích. Không phải danh mục mọi build.

**Hai thế hệ không được trộn:** Build7–21 trong `archive3` cũ dùng **raw record cache/replay**, bị owner dừng10/10 vì R3.
Các pilot FL mới P1/S-a/V5/V7/V9 dùng thống kê/model/noise-init synthetic. Chúng khác fixture/horizon/model role;
không đặt số tuyệt đối cạnh final50 để xếp hạng.
Nguồn route lịch sử: [kiến thức kế thừa](../../archive/legacy/kien-thuc-ke-thua.md), [lessons index](../../lessons/README.md).

## 1. Kết luận ảnh hưởng trực tiếp quyết định hiện tại

1. **CE+BN input inversion đã thử**, probe17 hai seed; không xem thêm lần nữa là cơ chế mới.
   Điểm mới có thể kiểm là **teacher được sửa bằng covariance riêng lớp trong basis48** + mục tiêu latent covariance, thay BN toàn cục.
   P24 phiên hiện tại đã kiểm một adaptation MSE48; không kéo dài y nguyên chỉ vì fit/acceptance lên.
2. **Latent virtual low-rank đã có từ Build5 N3:** offline tăng nhưng online giảm. Cần snapshot/BN chung cố định,
   không dùng bài học này để bác mọi calibration sau training.
3. **Input Nyström4/8 đã FAIL utility**, dù compression PASS. Latent task-basis48 là đối tượng khác,
   phải nói rõ khác không gian/nguồn teacher/rank, giữ full-covariance control.
4. **Prior/sample weighting là confound cũ, không phải nguồn FL mới.** S-a đã cảnh báo producer-uniform khác sample-weighted pooling.
   Với calibration16.384/client hiện tại phải kiểm lại đúng trọng số trước kết luận covariance/inversion không đủ.
5. **Global head gradient trên cùng frozen teacher chưa thấy trùng adaptation đã audit trong các nguồn local đã rà.**
   Khác LG, FedGH, D3, gradientbooks; chưa có utility PASS. Phân tích riêng ở §7.

## 2. Prototype/head-only: bài học có điều kiện, không phải cấm sửa head

| Bằng chứng | Điều kiện/số đo | Dùng lại được / không suy quá |
|---|---|---|
| Prototype cũ Build2–4 | Build2 test20c r50 0,5436; biến thể InfoNCE/clustering/soft-label/projection đa số±0,006. Nguồn retained registry, raw `archive3:docs/experiments/overview.md` chưa có local | Mean lớp thiếu ranh giới. Không gọi mọi latent moment hoặc head calibration là prototype-mean |
| P1 `lg` | Group gp r20 +0,019/+0,021 so local; local train loss≈0,009. Chỉ FedAvg private last head, private extractor vẫn học local | Đây là head trên latent các client đang khác nhau; không tương đương gradient tại cùng teacher/extractor |
| P1 `fedgh` | Group gp 0,4453/0,4507 vs local0,4646/0,4633; server fit head trên20×16 tâm latent | Thất bại của **mean prototypes + latent/head mismatch**, không phải thử full real-risk global head SGD |
| P1 FedGen adaptation | Group gp0,4823/0,4807≈LG0,4838/0,4839; loss local≈0,009 | Generator chỉ chạm predictor trong adaptation đó. Không suy mọi generator input-space đều yếu |
| Probe14/16 FullDrift mới | m4b ep15 s42 private gp0,7323 → Linear refit pooled0,8427; own refit0,6864; MLP pooled0,8754 | Trên backbone này, nhãn/input pooled ở head có giá trị lớn; bác cách áp dụng vô điều kiện khẩu quyết "head-only vô dụng" |

Nguồn số/cơ chế P1: [report P1 §2.3–2.5](../../archive/research/probes/pilot-p1.md),
[thiết kế H3](../../archive/research/de-xuat-huong-moi.md),
[lesson head-transfer](../../lessons/federated-head-transfer.md).
Probe14/16: [results FullDrift](../../methods/full-drift/results.md).
Oracle pooled real là phép chẩn đoán, không thành kênh FL hợp lệ.

**Giữ nguyên phạm vi:** backbone yếu/học shortcut có thể giới hạn head-only; backbone đủ nhưng head lệch lại hưởng lợi calibration.
CCVR cho **common teacher** cũng khác việc copy head teacher sang private khác kiến trúc.

## 3. Inversion và synthetic: đã làm những objective nào

### Probe17: input CE + Mahalanobis + BN, không phải class-specific latent covariance

Source đã chạy: [fd_probe17.py trong artifact](../../../methods/full-drift/builds/fd-probe/artifacts/kaggle/fd-probe17-20261015-p17-v1/source/fd_probe17.py);
[kết quả JSON](../../../methods/full-drift/builds/fd-probe/artifacts/kaggle/fd-probe17-20261015-p17-v1/runs/probe17.json);
[report](../../methods/full-drift/results.md).

- Teacher/backbone DAGSNet đóng băng, eval/fp32; input khởi tạo **pooled Gaussian noise**, m4b luôn0.
- Adam trên input200 bước, lr0,05, 16batch×4.096 dòng; nhãn sinh theoπ_g để BN batch hợp global running stats.
- `ce`: CE(teacher(x),c)+0,1×Mahalanobis lớp/53chiều; `di_lo` thêm0,1 BN MSE; `di_hi` thêm1,0 BN MSE.
- Fit riêng Linear cuối private trên own+synthetic, nhãn proxy hard/soft; không train lại full FL.

| Private frozen backbone | Gaussian hard | Gaussian soft | CE refinement soft | DI-hi soft | Real peer input oracle |
|---|---:|---:|---:|---:|---:|
| m4b ep15 s42 | 0,762491 | 0,744480 | 0,768227 | 0,783167 | 0,830616 |
| m4b ep15 s43 | 0,764735 | 0,751703 | 0,760316 | 0,777837 | 0,820612 |

**Đọc đúng cột:** `ce (mềm)`0,7682/0,7603 trong report không phải Gaussian-soft.
Raw JSON cho DI-hi−Gaussian-soft **+0,038687/+0,026134**, còn so strong Gaussian-hard chỉ**+0,020675/+0,013102**;
DI-hi−CE-soft **+0,014940/+0,017521**. Gate+0,02 cả hai seed so strong reference vẫnFAIL.
Vì vậy nói "DI chỉ tăng0,015 soGaussian" sẽ trộn contrast; đúng là gain lớn so weak soft baseline nhưng chưa đủ so mạnh nhất.
1-NN distance5,07→3,9–4,4 (real2,05), label agreement0,43→0,65–0,69: realism tăng, còn khoảng oracle.

**Điểm tái dùng:** frozen-backbone refit, soft/hard target factorial, prior-matched synthetic batches, oracle real-peer input,
realism độc lập và per-client guards. **Không làm lại:** chỉ tăng CE/BN inversion steps hoặc lấy agreement cao làm PASS.

### P23/P24 hiện tại khác ở đâu

Source [fd_probe24.py](../../../methods/full-drift/builds/fd-probe/source/proj/fd_probe24.py): noise Gaussian input66 →
khớp **từng virtual feature48** rút từ Gaussian latent lớp, normalized feature MSE +0,01 input trust penalty;
80input steps/lr0,03, bank4.096; teacher gốc/mới dùng cùng shared teacher features để tách source/support.
Private backbone **và BN** frozen, chỉ head1.000 bước CE own+real/synthetic KL.
[P23](../../methods/full-drift/results/probe23-xa-s42-e15.json),
[P24 all-client](../../methods/full-drift/results/probe24-xa-s42-e15-all.json) là local offline diagnostic, chưa online FL/terminal Kaggle utility.

Đây không phải BN-only DI cũ, cũng không phải trực tiếp minimize **batch covariance error** của synthesized features;
matching tới latent draws chỉ gián tiếp mang mean/covariance của target. Acceptance cao chưa đủ xác định label/manifold đúng.
Giữ nguyên teacher×support control; nếu muốn thay objective phải nêu thông tin/gradient mới và falsifier, không rename MSE48.

### Static KME/DM rồi dynamic gradientbooks

| Build | Đối chứng và cổng | Số đo liên quan | Bài học dùng lại |
|---|---|---|---|
| S-a KME |20c,1.000.369train,old/group65.536,20round,2seed; KME−max(diag,kme_gauss)≥0,015 | KME0,4989/0,4985; diag0,5401/0,5427; hơn kme_gauss chỉ+0,0019/+0,0024 ⇒FAIL | Finite bank/optimizer Gaussian-target phải cùng recipe; Fourier loss≈4e−7 không là utility |
| S-a DM |4point/class/producer, random frozen encoder, static bank; DM−dm_gauss≥0,010 |0,4882/0,4913; Δ−0,0002/+0,0036 ⇒FAIL | Mean nonlinear embedding một encoder không đủ trong adaptation đó; không bác mọi condensation |
| V5 IPC1 |Mean/gradient × frozen/current private encoder; refresh5round,80steps/book,28run IPC1/4 cả2seed;≤2×cost |grad-trained−continuous diag−0,0751/−0,0795;−noise+0,0026/−0,0001 ⇒FAIL | Đã thử refresh/current encoder; không đề xuất như ý tưởng chưa có |
| V5 IPC4 |Cùng factorial; grad-trained−diag≥0,015,−noise≥0,015,−own≥0,010 |−diag−0,0496/−0,0568;−mean-trained−0,0037/−0,0052;−grad-frozen−0,0036/−0,0012 ⇒FAIL | Tăng IPC1→4 không cứu objective/current-encoder contribution; costPASS không thay utility |

Nguồn: [S-a contract](../../archive/research/pilot-new-fl.md),
[S-a report/phạm vi](../../archive/research/probes/pilot-new-fl.md),
[V5 contract](../../archive/research/pilot-v5-books.md),
[V5 report](../../archive/research/probes/pilot-v5.md),
[source condensation](../../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/source/mh/gradient_latent_books.py).

V5 target gradient lấy từ **private encoder riêng từng producer**, ≤32real/class; greedy bank noise tối ưu cosine matrix/tensor-gradient,
không gửi gradient thật, không gộp gradient tại common θ. Recipient học CE thật+CEbook, không phải distillation từ teacher CCVR đã sửa.
V5 mean/gradient bank đều refreshed nên factorial **không** cô lập refresh vs static; S-a vsV5 đổi nhiều yếu tố.
Không quy tất cả thất bại cho bank tĩnh, BN, optimizer hoặc số bước mà không đối chứng.

## 4. Covariance sketch và function residual: đường nén từng thử

**V7 input covariance Nyström**,20round2seed, group/private/gp; full-cov0,626426/0,631264;
exact lowrank8 0,612890/0,614231; Nyström8 FL0,594610/0,602510.
Nyström8 so exact8 −0,018280/−0,011721, dù byte ratio0,32145 PASS; rank4 byte ratio0,19817 cũng qualityFAIL.
So own tốt không đủ, rank4 FL−shuffle chỉ0,00170/0,00093.
[Report+gates](../../archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/results/report.md).
**Đổi hợp lệ hiện tại:** học feature teacher đủ chất lượng, chọn task directions, rank48, calibration head frozen;
không gọi random rank4/8 input sketch y nguyên là hướng mới hoặc hứa compression không mất gain.

**V9 anchor residual M64/M512**,26arm×20round đã audit; M512 FL−classmean chỉ0,001596/0,000877,
qualityFAIL/costPASS. Forensics hai seed: sau loại mean chung, correction energy chỉ4,08%/4,81%; bootstrap label mass≈95%,
kernel entropy0,955; any-anchor coverage>99% nhưng nearest-anchor label accuracy≈27%.
Fit MSE≈0,0017 vẫn FAIL; không chứng minh một mình BN/fit/extrapolation là nguyên nhân.
[Report](../../archive/methods/anchor-function-residual/builds/anchor-residual/results/report.md),
[forensics](../../archive/methods/anchor-function-residual/builds/anchor-residual/results/forensics-20261015.md).
**Dùng lại:** classmean/own/shuffle controls, phân rã target common-vs-spatial, teacher gate trước private KD.
**Không suy:** mọi aggregate gradient/residual đều thành class-mean; V9 dùng anchor/kernel interpolation riêng.

## 5. ClassTail/CCR và sample prior: điều phối không tự mang nguồn tri thức mới

- **ClassTail:** pooled CE-tail histograms r2 calibrationPASS, r20 group/private/gp0,461645/0,458453;
  local0,464328/0,460985; classmass0,462358/0,458617. Không hơn own/shuffle/classmass; utilityFAIL hai seed.
  Cổng−local≥0,015,−own/shuffle≥0,010; không thay α/bin/λ sauFAIL.
  [Report](../../methods/class-tail-risk/builds/class-tail-pilot/results/report.md).
- **CCR=CrossClientRank:** pooled0,466075,−CE+0,001735,−own−0,000261,−shuffle+0,001196,
  −classweight+0,001142 ở seed42r20 ⇒FAIL; cần−CE≥0,015,−own/shuffle≥0,010.
  Seed43 recovery chỉ mô tả, không suy cảhai seed terminal nếu chưa có audit hiện tại.
  [Build/contract](../../methods/cross-client-ranking/builds/rank-utility/README.md).
- **S13 Build13:** ridge/RFF teacher input−off+0,018/+0,019 nhưng−classmean chỉ+0,0003/+0,0038;
  thấp hơn raw replay−0,09/−0,08 trên group gp archive c9–12. Source raw `archive3:docs/experiments/build13/results.md`
  không có local hiện tại; số lấy retained [registry §4](../../archive/legacy/kien-thuc-ke-thua.md).
  RFF trên raw input lại teacherFAIL ở FullDrift probe13: M2560,378, M10240,484≪gpool0,7174;
  [P4 dossier](ridge-rff-teacher.md). Không đề xuất lại RFF teacher để né CCVR byte.

**Prior đúng là hợp đồng estimator.** S-a report đã cảnh báo DM dùng mixture producer đều/loạiown, diag pool theo số dòng,
nên chênh F1 không cô lập chỉ finite bank. Với P23/P24, sample16.384/client làm distribution khác tổng record pool:
suddenStop mass0,033853 vs**toàn fixture pilot**0,020231; reweight theo n_k đưa về0,019958.
Chữ `full` trong JSON này là toàn fixture≈1M dòng, **không** toàn train80%≈43M; không trộn hai population.
[Calibration prior diagnostic](../../methods/full-drift/results/probe-calibration-priors.json).
Đây là trọng số nguồn cần kiểm trong estimator, không phải bằng chứng scalar prior correction mang FL knowledge mới.
Không đổ mọi min-client drop cho prior từ correlation này; cần intervention giữ snapshot/support/loss cố định.

## 6. Build12 A5 / Build14 E14: tái dùng kiểm soát và lịch, không tái dùng raw replay

**Giới hạn truy hồi:** local hiện không có `docs/experiments/build12/`, `build13/`, `build14/` của repo `archive3`,
cũng không thấy bundle lịch sử tại các vị trí dự án đã kiểm. Không restore/download hoặc suy kết quả terminal.
Retained registry chỉ xác nhận Build12 AMP guard và N4/N5 BN lessons;
[LESSONS trước layout](../../archive/legacy/lessons-before-layout-20261014.md) còn cảnh báo E14 arm2 từng chạy0cycle,
và M15 interleave từng che contribution của cơ chế mới.

Thông tin dưới là **memory-derived từ handoff07/10, snapshot có thể cũ; chưa xác minh lại primary artifact local**:

- **A5** chỉ sửa lifetime AMP failure counter từ session sang client×cycle; không phải phương pháp sinh mẫu/học mới.
  Local exactness20client×8cycle từngPASS; A5 remote lúc handoff chưa terminal-audited.
  Primary cần đọc nếu tiếp tục: `archive3:docs/experiments/build12/handoff.md`,
  `archive3:papers/build12/preflight/a5_cycle_guard_proof.json`.
- **E14** so `tail` full natural epoch rồi32replay updates với `interleave` chèn update j sau `ceil(j*N/32)` natural steps;
  giữ stream/count/optimizer/EMA/head/bytes. Gate lịch sử interleave−tail≥0,005 cả2seed, old≥−0,005,
  benign recall≥−0,01 vàcycle2>cycle1. Handoff lúc đó **chưa có terminal quality**.
  Primary cần đọc: `archive3:docs/experiments/build14/README.md`, `handoff.md`, `archive3:papers/build14/preflight/proof.json`.

**Tái dùng hiện tại:** ordering-only control với cùng input/update count; actual driver mọi arm phải thật sự chạy đủ horizon;
AMP/BN health không thay quality. **Không bê nguyên:** E14 dùng raw peer cache bị loại theo R3.
`gkd` hiện đã có256synthetic xen mỗi real batch, nên lời khuyên "hãy interleave synthetic" trùng baseline;
chỉ đổi cadence/tail khi có giả thuyết khác, đối chứng matched-count và cổng mới.

## 7. Same-snapshot global head gradient: khác cơ chế nào, cần falsifier gì

Rà local tài liệu/source nêu trên **không tìm thấy run đã audit** tính gradient nhãn thật tại **cùng frozen common teacher/extractor**,
SecureSum gradient full last-Linear rồi server cập nhật head teacher. Đây là kết quả tìm kiếm hữu hạn, không khẳng định lịch sử chưa từng có.

| Tiền lệ | Khác với gradient common teacher |
|---|---|
| LG/head FedAvg | Private encoder/head học nhiều bước local; averaging endpoints khác một gradient tại cùngθ/BN |
| FedGH/prototype | Fit trên mean lớp, bỏ phân bố h; gradient thật phụ thuộc h cùng posterior từng dòng |
| D3 deployment head | Fit từ raw peer cache trên EMA backbone; raw input channel không dùng được hiện tại |
| V5 gradientbooks | Private θ_i khác nhau; gradient targets không rời producer, chuyển qua bank nhỏ và cosine matching |
| V9/CCR | Anchor/kernel residual hoặc scalar score means; không gradient CE của common16×256head |

Nguồn phân biệt quan trọng: [V3 knowledge-sources §K3](../../archive/research/v3-knowledge-sources.md) nói rõ local condensation
**không** là global-gradient target khiθ_i khác nhau; muốn đổi phải sync cùngθ/BN và trả byte.
Hiện có snapshot proxy chung nên điều kiện cùng extractor có thể kiểm được, không cần share private architecture.

Suy luận toán cho ứng viên, **chưa là kết quả**:

```text
client k, frozen shared h=f_v(x), q=softmax(W_v h+b_v):
G_k = Σ_(x,y) (q−onehot(y)) hᵀ       # 16×256
g_k = Σ_(x,y) (q−onehot(y))          # 16
G = SecureSum(G_k); g = SecureSum(g_k)
(W,b)_new = (W,b)_v − η (G,g)/N     # pooled-record CE, một query
```

G/g là vector **aggregate theo client**, không gradient từng record; raw h/x/y ở client. Common fixed-point scale/clipping/
SecAgg và numerator/denominator phải khóa; nominal FP32 up16.448B +head down16.448B chưa packet/contactPASS.
Tính mean gradient mỗi client rồi average đều đổi risk thành client-uniform; dùng summed gradient +N nếu muốn record-union risk.
Class-weighted risk phải khóa weights và weighted denominator chung; không chọn weights theo test hay gọi classprior change là knowledge gain.

Một gradient chỉ xác định đạo hàm tạiθ_v. Sau update θ thay, **không** lấy lại G cũ như gradient chính xác ởθ_new;
nhiều bước adaptive cần thêm query/byte/contact hoặc phải ghi approximation. Full teacher CE head risk convex với h cố định
không bảo đảm macro-F1/worst client tăng sau một bước, không tự cải thiện private còn support lệch.

**Offline falsifier trước online:** cùng snapshot/sample weights, so no-op, exact pooled one-step head-gradient,
fullCCVR và compressedCCVR; old/group/per-class/benign, không test selection. Head teacher không hơn no-op đủ gain ⇒ dừng ứng viên.
Teacher lên nhưng private same-support transfer không lên/worst fail ⇒ nút tiếp nằm transfer/support, không tăng head steps để né guard.
Đối chứng own gradient hoặc shuffled label phải cùng norm/step protocol nếu dùng để claim tri thức peer;
không lấy oracle pooled head solve nhiều bước làm kết quả khả thi của one-query gradient.

## 8. Dùng lại gì ngay, đóng gì

- Giữ **real-input pooled oracle** làm trần chẩn đoán: probe15 `cov_real`0,8422 vsGaussian0,7921;
  probe16 nhãn proxy trên real-peer0,8306 vsreal-label oracle0,8427. Không gửi real-peer input trong phương pháp.
- Giữ **teacher×support factorial**, đúng source weighting, head/benign/worst guard, horizon15+seed43; đối chiếu same actual teacher snapshot.
- Trước khi thêm synthesis loss, xác định aggregate tri thức mới và đo private correction **trừ spoiled predictions**;
  accuracy/teacher-fit/acceptance/covariance fidelity không thay utility.
- Đóng adaptation không đổi: `xo`, covariance chung `cvl`, CE+BN-only inversion, static KME/DM,
  V5 IPC1/4 gradientbooks, random input Nyström4/8, V9M512 kernel residual, ClassTail/CCRscalar, RFFteacher yếu.
  Đóng adaptation không đồng nghĩa bác cả họ; mở lại phải có cơ chế khác và controls đối đúng thất bại đã thấy.

Không có bằng chứng build cũ nào cho phép bỏ R1/R2/R3 hoặc nới gate hiện tại.
Raw archive thiếu chỉ hạn chế kết luận A5/E14/S13; chưa cần thao tác tay để tiếp tục chẩn đoán hiện tại từ source/artifact đang có.

**Veto bổ sung trước khi dừng phiên22/10 — synthesis chỉ khớp unconditional head-gradient chưa đủ căn cứ launch.**
Với extractor h đóng băng, `h̃=[h;1]`, cùng sample/weights w và softmax head A:
`G(A)=E_w[p_A(h)h̃ᵀ]−E_w[onehot(y)h̃ᵀ]`. Thành phần phụ thuộc nhãn chỉ là class counts/mean của teacher features;
thêm probe **chỉ perturb head** thay marginal term, còn `G(A₁)−G(A₂)` triệt tiêu label term
(perturb nhỏ cho Hessian-vector của head, không tự thêm class-specific covariance). Gần optimum, tổng G có thể gần0 do cancellation;
khớp một hướng G thấp không xác định synthetic distribution hoặc nhãn hữu ích cho private khác kiến trúc.
Class-conditional gradients tránh một phần cancellation nhưng raw FP32 `16×16×257×4=263.168B` đã vượt B2 riêng upload;
bỏ một output row dư vẫn246.720B. Projection48 hạ còn50.176B nhưng đổi thông tin, chưa utility/packetPASS.
Khác V5 ở shared θ vẫn chưa giải quyết finite-bank/teacher-support/mean-collapse đã thấy ở V5/KME/V9.
Nguồn gradient-condensation gốc dùng **per-class gradient matching và nhiều model states**, không chứng minh one-head/one-query carrier:
[Zhao et al., ICLR2021](https://arxiv.org/abs/2006.05929),
[official code, vòng class/net/outer-inner](https://github.com/VICO-UoE/DatasetCondensation/blob/master/main.py).
Đây là suy luận đại số + history veto cho adaptation cụ thể; không bác mọi gradient condensation.
**Chưa đề nghị launch biến thể này**; phiên dừng theo owner, không điều tra/chạy tiếp.
