# L — target, hệ số KD và realgate; khóa trước push31/10/2026

Owner31/10: **bắt đầu thực hiện** [kế hoạch chẩn đoán](root-cause-experiments.md).
Batch12run,5343xe,10round trên cosine50, DAGSNetW1/batch64/1epoch/mọixe/resetAdamW/AMPfp16/group_rows4096,
2×T4 mỗi run; toàn bộtrain+val, own-test đầy đủ mỗi round. Hai seed42/43. Không50round trongbatch này.

## Giả thuyết và cổng

K4/K5 đổi cảtarget lẫnscale so K0/K1. K8/K9 chỉgate realtimeDelay r2–10 nhưng syntheticQ[y] sai vẫn được dùng.
L tách công trạng softtarget, adaptivecoefficient và gatedrealKD. Không đổi copula/priors/optimizer/generator để tìmwin.
Ký hiệu C=CEreal cóLA, S=syntheticloss, R=KLreal gated, a=1/(C+ce_t).detach().

Giữ cổng gain≥.010,p10Δ≥−.010 vsG4 cùngseed (42=.873187514625593/.5737619047619047;
43=.8745079727196574/.5687566137566138). Báo đồngthời Δvsaugmentation và đối chứng mới cùngcodec.
Headline r10, khôngbest-round/HPsearch. Đây là screening thích nghi từ K, khôngfreshconfirmatorytest/significance.

## Các cell

| ID | seed | tài khoản | slug | loss từ r2 |
|---|---:|---|---|---|
| L0 |42|nguyenminhtran060113|mh-l0-softfix-s42|C+Ssoft |
| L1 |43|thuyanh316|mh-l1-softfix-s43|C+Ssoft |
| L2 |42|minhtriethihi|mh-l2-hardadapt-s42|C+a·Shard |
| L3 |43|annhtnt|mh-l3-hardadapt-s43|C+a·Shard |
| L4 |42|takids1909|mh-l4-hardgatefix-s42|C+R+Shard |
| L5 |43|annhannh331|mh-l5-hardgatefix-s43|C+R+Shard |
| L6 |42|trietbackup|mh-l6-softgatefix-s42|C+R+Ssoft |
| L7 |43|khanhmay0304|mh-l7-softgatefix-s43|C+R+Ssoft |
| L8 |42|catbaochau|mh-l8-combowire-s42|C+a·Ssoft |
| L9 |43|odixeuit|mh-l9-combowire-s43|C+a·Ssoft |
| L10 |42|accanniez|mh-l10-gatewire-s42|C+a·(R+Ssoft) |
| L11 |43|minhtran0601|mh-l11-gatewire-s43|C+a·(R+Ssoft) |

R1 mọi cell đều C+Shard vì chưa cóQ. Dose64 mọi bước, syntheticlabel cân bằng16lớp.
Fixed=1 cho cảR/S; adaptive giữ nguyên a; gate giữ denominator toàn bộreal, khôngnormalize riêng sampleđượcgiữ.
`ce_t` tiếp tục gồm samplebịgate; không đổi riêng thành phần này trongL.

K4/K5 augmentation làcontrolC0 dùnglại. LocalCPU2round bitexact nhánhno_kd và hai legacycombo/gate trước/sau thaycode;
chưa là bitexactproof cho mọi T4schedule. Nhánhaugmentation không tiêu thụQ nên codec mới không đổi toán học.
Nếu remote cho thấy source/đườngaugmentation khôngtươngthích, phải thêmcontrol, không quygain cho yếu tố khôngcôlập.

So sánh cùngseed:

- L0/1−K4/5: softtarget tại scale1; L8/9−L2/3: softtarget tại adaptive.
- L2/3−K4/5: adaptive trên hardtarget; L8/9−L0/1: adaptive trên softtarget. Báo tươngtác2×2.
- L4/5−K4/5 vàL6/7−L0/1: thêm gatedrealKD, cùngsynthetictarget/scale.
- L10/11−L6/7: adaptive trên gate.
- L8/9−K0/1 vàL10/11−K8/9: thaycodec, không gọi là ablationloss.

## Codec và hợp đồng federated

K classQ đãcộngsumfp32/float64, chưadecodefp16body. L dùng `q_codec=mean_fp16`:
client encode **classmeanprobability16×16fp16**; server decode, nhân n_kc cótừbootstrap, gộp theo lớp;
downlink encodefp16, client decode vànormalize hàng. L8–11 làcontrolcùngcodec mới vì thế batchtăng8→12run.
Schema vẫn16lớp chung, khôngprivateweights/rawrow/featuretừngmẫu/nhãntừngdòng.
Bootstrap giữ nguyên mô hìnhstatistic củaK; codecL chỉ kiểm kênhclassQ, không claim transport/secureaggregation/codecbootstrap hoànchỉnh.

PayloadclassQ:512B+32Bheader mỗi chiều=1088B/xe/round; bootstrapcopula3311744B/xe;
tổng10round3322624B/xe. CPUunit serializebytes rồi reconstruct độc lập kiểm decodedaggregation/downlink và544B/chiều.
Chưa transport/packetloss/contactdeadline thực. Thời gian đo runtimekernel, khônghardwarexe.

## Instrumentation và artifact

Shadowprobe sau round1/2/5/10, **34client** cố định: ID nhỏnhất trong mỗi traincountcohort×scenario/run khôngrỗng;
5cohort [0,256),[256,1024),[1024,4096),[4096,16384),[16384,100000).
Clientmanifest `runs/l-screening/diagnostic-clients.json`, không chọn theo testF1.

- Probe postepochprivateweights, teacher của roundvừatrain (Qr−1), fixedrealminibatch và syntheticseed theo client;
  eagerfp32/trainBN/dropout0, khôngreplaygradientAMP thực. Cloneweights/buffers, fork/restore RNG, khôi phụcsamplergenerator.
- Log C/R/S, ce_t, adaptive/usedweight, parametergradientnorm/cosine trướcclip; cosineundefined ghi null.
  Ghi shadowgradientvượtclip; **không gọi là actualtrainingclippingfraction**.
- Tỷ lệ cosineâm theo lớp là **logitgradient** trướcgate, tách với parametergradientcosine theo client/cohort.
- FulltrainCM và10metric tại1/2/5/10; đãtrain cảtrain+val nên chỉ đo fit/gapmôtả, khôngvalidationđộclập.
  Temporalheldout fitmới thuộc nhánhdiagnostic sau, chưa thực hiện trongL.
- TestCM/10metric/fullsupport mỗi round giữ nguyên. `t_diagnostics` tách khỏi `t_train`,`t_eval`; worstbudget gồm cảprobe.
- `diagnostics_rank0.jsonl`,`diagnostics_rank1.jsonl`; `train_cms_r001/002/005/010.npz`;
  rounds.jsonl, config.json,10testCM,10smallmodelsnapshots, provenance, fullstate mỗi rank tại endpoint/budgetstop.
  Snapshot nhỏ vẫn không phải fullprivatecheckpoint. Exactresume/CUDAglobalRNG chưa được chứng minh; cácrunL freshinit, khôngparentresume.

## Khóa source và vận hành

BaseSHA099e95e8466811881fb71dcd7d3c44268ed082d6; overlay3file:

- `methods/mobihet.py`:1ffaae2b824711df4798b1fc5f8b0c89ae9666f0b4fb770fd10911e33e57bf89.
- `methods/mobihet_diagnostics.py`:ebe4b1e6bc76ac75acb990e3f3ced4253ee568ea8804670327dc167d773d7504.
- `scripts/run_fl.py`:747fd50ba2a4ec7fbc3697e7a01f0b717e09e27c84884af0b7e626e1965bfe18.

Spec/notebookhash `runs/l-screening/batch.json`, SHA256b652af446ce8c1b3cd5e4ae796477dac0cfcc498336605e1b81a66a06e9358cd.
Không sửa frozen notebook saupush. Localtests36PASS; legacyCPUequivalence3casePASS;
realdataGPU4client/144306trainrows, W1/compiledAMP/2round PASS, skip0;
10metrictrain/testkhớpCM, fullprivatebankstrictrebuild vàteacherlag PASS (`runs/l-screening/smoke-audit.json`).
Remote2×T4 phải ghi kếtquả khi xong, không suy từlocal sm86.

Quota14:08VN:16account đều30hfree/0reserved, identityverified; chọn12account ởbảng,48hcapbatch=4h/run,
free≥1,15×cap, chưa phải48hthực dùng. Ướctính~2–3h/run gồmstartup/compile/train/eval/probe/stateI/O;
rechecktrướcpush. GiữglobalCLIdinhan25, accountscopedwrapper, private notebooks0600/W&Bonline theo quyền cũ.
KeyW&B tồn tại trong lịch sửversionprivate; khôngpublishnotebook. KhôngGit/DVCpush/cleanup.

Live trạngthái/startup/version/việctay ở [continue](../operations/continue.md).
PushL0–L11 version1 exit0 lúc14:16–14:18VN; lúc14:21 **12/12RUNNING**, source_manifest
5c6e9f9c4e0923b1a28a3b76a393e481018397168228f69fdc61743cd56f3b1b khớp12/12,2×T4 vàW&Bauth đãkiểm12/12.
Chưacompilegate hoặcmetricround; W&Blogin không chứngminh trainingrun đãkhởitạo. Khôngcài watcher/collector nền.
Khi collect: audit config/sourceoverlay3file/rounds/LR/support/10metric/Qfinite; kiểm trainCM khớptraincounts,
probeids/teacherlag/finite vàδcohort. Không gọi COMPLETE/W&B làutilityaudit.

16:50:53VN: watcherexactv1 xácnhận12/12COMPLETE, đãqueuecompletion;16:55bắtđầucollect/audit nền,
chưacókếtquảutility. Trạngtháijob vàevidenceởbàngiao, khôngthay frozen source/run cũ.

Collect/audit hoànthành16:59:20: [Lresults](results-l10.md),12/12đủ10round/source/metric/trainCM/probe/snapshotsPASS.
Khôngthaycổng/source đãkhóa; không dùngCOMPLETE làmutilityproof. Cácdòngtrạngtháitrên là lịch sửlúcstart.
