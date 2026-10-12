# V5-H — target kernel gần phẳng trên artifact đã chạy

Ngày2026-10-12 UTC+7. **Chẩn đoán mô tả sau pilot đã khóa và FAIL**; không chọn lại h,
head, round, sharpening hoặc mở run cứu cổng. Mọi số dưới đây từ4 `resume/round_020.pt`
đã AUDITED trước backup; restore chọn lọc, kiểm MD5/SHA rồi phân tích CPU.
[Script](../../scripts/probe_v5_anchor_targets.py), [bằng chứng đầy đủ](../../papers/v5/anchor-target-diagnostics.json),
[report10metric](../../papers/v5/report.md), [hợp đồng pilot](../../docs/research/pilot-v5-anchors.md).

## 1. Kết luận hẹp

- h=4 được ghi trên cả small/medium, seed42/43; chosen-bandwidth coverage qua90%overall/80%class.
- Target entropy/log16≈0,982; maximum probability trung bình chỉ≈0,088/0,091.
  Target có độ tương phản thấp; coverage tốt chưa tạo supervision sắc.
- Target vẫn mang thông tin: argmax khớp Gaussian generating label31,25%/36,52%, hơn chance6,25%.
  Không gọi target “uniform hoàn toàn” hoặc “không có tri thức”.
- Hai seed có **target decoded trùng bit**, do dùng cùng fixture và fixedanchor RNG.
  Hai seed kiểm train khác initialseed; không hai ước lượng tri thức độc lập.
- Small teacher đã fit soft-target objective trên anchors gần entropy floor nhưng IDS vẫn thấp.
  Fit target chưa generalize. Medium còn gap nhỏ; chưa cô lập hoàn toàn optimization/model capacity.
- Gaussian hard-label teacher cũng thua local/dense; nearuniform chỉ hỗ trợ một giả thuyết,
  không giải thích duy nhất toàn bộ failure finiteanchor/support/teacher.

## 2. Khôi phục và phạm vi audit

Manifest `papers/v5/kaggle.dvc`:
`f39dc32022b6b8789d3e0caefe03dbc4.dir`; directorymanifest MD5 PASS.
Chỉ restore4 file FLsmall/medium×seed42/43, tổng **7.013.072B =6,688MiB**;
small1.615.380B/file, medium1.891.156B/file. Không checkout toànkernel, không đổi manifest/cache,
không Git/DVCadd/push/cleanup. Atomiccopy từ remoteobject vào đúng path, từngfile MD5 PASS.
[DVC/restoration policy](../../docs/operations/backup.md#6-khôi-phục-phục-vụ-nghiên-cứu2026-10-12).

Probe kiểm thêm:

- Source4module `v5_anchor.py/v2_math.py/v2_methods.py/v2_run.py` trùng SHA trong AUDITED.
- ResumeSHA trùng `complete/round_020.json`; fingerprint trùngconfig; savedround=20;
  đúngarm đủ20round theoAUDITED.
- Decode targetpacketfp16, normalize giống source, **trùng tensor target lưu**.
- `torch.load(...,map_location='cpu',weights_only=True)`; torchthreads1; guard exit0,
  peak569MiB lần cuối. Không GPU/modelinference/train, không đọc record train/test.

Đây audit bổ sung4knowledge snapshots; không chạy lại wholeweights/predictions/sampledinference
của18kernel. Sourcecode đã chạy chỉ được đọc, không sửa. Output chỉ count/scalar/classaggregate,
không M×16 targettable, M×66 anchorpoint hoặc sample/rawdata.
[Proof file hash/packet/metadata](../../papers/v5/anchor-target-diagnostics.json).

## 3. Coverage toàn bộ T của pilot, không full43M train

Source duyệt mọi `R.X/R.Y` của fixturepilot, chunk4096; queriedrealrows=**1.000.369**.
Đây toàn trainingT của pilot20client/2,5%, không toàn originaltrain43M.
Coverage criterion: nearestkernel `k≥1e−3`, tương đương squaredscaled distance
`≤−2*h²*log(1e−3)`; scan h1,2,4,√66 theo thứ tự, lấy đầu đạt overall90%/minclass80%.
[Source thực thi:make_targets](../../papers/v5/kaggle/v5h-s42-20261012-v1/source/v5_anchor.py#L38).

| anchors | recorded h, cả2seed | overall coverage | minclass coverage | queriedpilotT |
|---|---:|---:|---:|---:|
|32|4|98,9208%|90,3515%|1.000.369|
|512|4|99,2940%|92,6115%|1.000.369|

Overall tính weighted theo decoded aggregate classprior; sai sốfloat32 rất nhỏ.
Mincoverageclass làdataReplay ở cảhai cỡ; perclasscoverage lưu trongJSON.
Source lựa chọn smallestpassingh, nhưng checkpoint chỉ lưu chosenh/fractions; **không có bảng coverage
h1/2 đã bị loại**. Probe này không tự recompute closestdistance trên T, không claim đã kiểm độc lập
tất cả candidatebandwidth. Chosenh/selectedcoverage/actualsource có bằng chứng, không infer từ sampleprobe cũ.

## 4. Target đãdecode và độ sắc

`T_jy ∝ (Σ_i Σ_(x∈localclassy) exp(−||x/scale−anchor_j/scale||²/(2h²)))/N_y`,
row-normalize16class rồi serializefp16, decode/renormalize đểtrain.
Classcountdivision có chủ ý: dùng classconditional density với prior đồng đều.
[Source thực thi:class normalization/packet](../../papers/v5/kaggle/v5h-s42-20261012-v1/source/v5_anchor.py#L74).

Entropy report `H(T_j)/log16`. Maximum entropy làlog16≈2,772589nats;
uniformprobability1/16=0,0625. Sốsmall/medium dưới đây giốnghệt giữa2seed vì target giốngbit.

| diagnostic target |32anchors|512anchors|
|---|---:|---:|
| mean entropy/log16 |0,982958|0,981591|
| minimum entropy/log16 |0,971066|0,909526|
| meanmaximumprob |0,087824|0,090732|
| maximumprob p90 |0,108074|0,118706|
| maximumprob max |0,142700|0,272713|
| meantop1−top2margin |0,008569|0,011962|
| meanprob ofGaussian generatinglabel |0,079750|0,083746|
| argmaxalignment with generatinglabel |31,25%|36,5234%|
| meangeneratinglabelrank |5,25|4,40625|
| meanTV(T_j,uniform) |0,108201|0,110511|
| meanKL(T_j||uniform),nats |0,047252|0,051042|

Entropy cao ở đa sốanchor; một sốmediumanchor sắc hơn, maxprob0,273.
Không đổi toàn target thànhhardargmax dựa vào số này để cứu cổng đã khóa.
Gaussian generatinglabel là class chọn để sinh point, **không groundtruthIDS của record thật**.
31–36%alignment không accuracy của oracle; nó chỉ contrast với Gaussianhardlabelcontrol.
[Aggregate artifact diagnostics](../../papers/v5/anchor-target-diagnostics.json).

## 5. Classmean và prior: không tự gán lỗi implementation

Gọi `qbar=mean_j T_j`. Target khácuniform có phần do bias classmean chung mọianchor,
phần do variation theo vị tríanchor:

| diagnostic |32anchors|512anchors|
|---|---:|---:|
| TV(qbar,uniform) |0,097156|0,093576|
| H(qbar)/log16 |0,987168|0,988328|
| meanKL(T_j||qbar),nats |0,011674|0,018680|
| meanKL(T_j||qbar)/log16 |0,004211|0,006737|
| varying-anchor share of squared deviation fromuniform |26,22%|38,09%|
| TV(qbar,decodedtrainingprior) |0,444388|0,442007|

Varying-anchor share=`mean||T_j−qbar||² / mean||T_j−uniform||²`.
Đa số squareddeviationfromuniform ởclassbias chung; phân bốtarget thayđổi theoanchor ít hơn
so vớiclassmeanbias. MeanKL tớiqbar là divergence của **bảng tổng hợp trênnoise anchors**,
không mutualinformation củaIDS trong dân số thật hoặc ngưỡngteacherquality đã được kiểm nghiệm.
[Aggregate evidence](../../papers/v5/anchor-target-diagnostics.json).

Decodedprior từ `ratio_pi=count_y/Σcount`, không smoothing; TV(prior,uniform)=0,396821.
Benignprior≈0,21928,dos≈0,14453,Sybil≈0,22051; smallqbar tương ứng≈0,07616/0,02401/0,05292.
Không lấy TV lớn vớiprior làm bằng chứng “chia counts sai”: mục tiêu classbalanced đã được thiết kế,
Gaussiananchors cũng sinh nhãn cân bằng. Probe không thay `head` bằngprior-adjusted rule sau results.
[Prior source](../../papers/v5/kaggle/v5h-s42-20261012-v1/source/v2_methods.py#L60),
[target source](../../papers/v5/kaggle/v5h-s42-20261012-v1/source/v5_anchor.py#L87).

## 6. Fit target và failure IDS là hai quan sát khác nhau

Teacherloss ởsource làsoftCE trực tiếp trên toànanchorbank cốđịnh,20optimizersteps/round;
BNcalibrate bằngGaussian chung, model.eval/dropoutoff. Entropy target là lowerbound củasoftCE
với predictions hợp lệ. Gap dưới đây dùng **meanloss trên appliedsteps r20**, không inference mới
của finalteacher và chịufloat32/AMP rounding.
[Source proxy_epoch](../../papers/v5/kaggle/v5h-s42-20261012-v1/source/v5_anchor.py#L112).

| anchors | seed | r20meanoptimizerCE | targetentropy,nats | CE−entropy | FLteacher group/headF1 | hardGaussianF1 |
|---|---:|---:|---:|---:|---:|---:|
|32|42|2,72533754|2,72533693|0,00000062|0,060332|0,127409|
|32|43|2,72533748|2,72533693|0,00000056|0,055411|0,120686|
|512|42|2,72288796|2,72154698|0,00134098|0,157065|0,299813|
|512|43|2,72324873|2,72154698|0,00170176|0,151083|0,293774|

Smallteacherfit fixedtargets gầnprecisionfloor mà vẫn FAIL trêngroup.
Không có bằng chứng từ loss để cứukernel này bằng train thêm cho fit target tốt hơn.
Mediumgap nhỏ nhưng khác0; chưa gọi optimization đạtnghiệm chínhxác, không tách riêng modelcapacity.
Tănganchors32→512 giúpF1 cảFL vàGaussianhardlabels nhưng khôngqua gate; finiteanchors và sampler
còn limitations ngay cảkhi không dùngsofttargets gầnphẳng.
[Audited report vàcổng](../../papers/v5/report.md), [diagnostic artifact](../../papers/v5/anchor-target-diagnostics.json).

## 7. Bài học và giả thuyết giữ được

1. **Coverage đo điểmtrain có anchor gần, không đo ranh giới/teacherquality.** FullpilotT đã qua,
   target vẫn entropy cao và teacherFAIL; bài học sampleprobe cũ nay cóartifact thật hỗ trợ.
2. **Khớp objective chưa chở tri thức ra holdout.** SmallCE≈H(target) cùngF1≈0,06;
   không đồng nhất synthesis/KD objectivefit với íchlợiIDS.
3. **Flat target không có nghĩa zeroinfo.** Alignment vượtchance, shuffle thấp hơnFL;
   vẫn cóanchor-specific variation. Không phủ định toànhọkernel-density/distillation.
4. **Seedtrain khác chưa chắc tri thức khác.** Fixedanchor RNG4902+M/cùngfixture khiến target
   trùngbit; đọc2seed nhưtrainreplication, không độbền trên2knowledge-estimationdatasets.
5. **Nguồn failure chưa cô lập.** h4broadsmoothing/66Ddistance, finiteMbank, Gaussianoff-support,
   trainingteacher/modelcapacity đều có thể góp. Hardlabelcontrol thualocal là dấu hiệu finiteanchor
   hoặc sampler/training recipe cũng cần kiểm nếu đề xuất hướng mới.

Nếu mởphương án kernel mới: hợp đồng mới trước run, tách mục tiêu coverage vàtargetcontrast,
giữfreshsame-anchorhardlabels/shuffle/own, đoclassmeanbaseline hoặcanchor-specificdivergence,
lockquality/cost/byte trênholdout. Không sửa h/sharpenexistingV5 hoặc selectnewrule từofficialtest.
Chưa đề xuất launch trong chẩn đoán này; privateKD/contact vẫn đóng theo cổng đãFAIL.

Không thao táctay cần thiết cho restore/probe.4resume nhỏ hiện có local, không xóa sauđọc;
DVCmanifest cũ vẫnsnapshotbackup, không tựaddpartialworkspace.
