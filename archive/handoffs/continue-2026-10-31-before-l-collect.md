> Snapshot trước collectL31/10. Bàn giao sống: [continue](../operations/continue.md).

# Continue — trạng thái31/10/2026 ~16:11 (giờ VN)

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [lessons vận hành](../operations/lessons.md).
Snapshot trước collect K: [01:15](../archive/continue-2026-10-31-0115.md).
Snapshot trước chẩn đoán cơ chế: [10:28](../archive/continue-2026-10-31-pre-mechanism-diagnosis.md).
Snapshot trước launchL: [11:21](../archive/continue-2026-10-31-before-l-launch.md).
Repo này nghiên cứu federated/MobiHet; baseline/NILM/Notion ở `../fl_dagsnet_veremi_v2`.

## Quyền và ràng buộc

- Private model có thể khác kiến trúc; không gửi raw local row/feature từng mẫu/nhãn từng dòng/private weights.
- Chưa trần byte, ưu tiên F1, báo cả bootstrap+hai chiều và thời gian. Giữ DAGSNet chung trước, mixedarchitecture sau.
- Đã duyệt screening10round trên cosine50round; **chưa launch/chốt F50round**. Không Notion/Git/DVCpush/cleanupdata.
- Global CLI **dinhan25** giữ nguyên; repo dùng CLI, MCPJSON đã xóa theo owner. Account khác qua `kaggle_as.py`.

## Đang thực hiện L — owner đã duyệt launch31/10

Owner16:10 yêu cầu **pollnền mỗi20phút, khôngmodelpolling lúcchờ**. Đãdùng
[codex-long-jobs](/home/odixe/.codex/skills/codex-long-jobs/SKILL.md)0.4.2:

- Job `kaggle-l-v1-20m-20261031`, supervisorPID30885, statusrunning đãkiểm1lần.
- Ownerthread `01a18d65-e3b8-73f1-80b6-7b66531915da`, delivery`queue`(auto), viewer`none`.
- Log [watch-20m.log](../../runs/l-screening/watch-20m.log), state `/home/odixe/.codex/long-jobs/jobs/kaggle-l-v1-20m-20261031/state.json`.
- Worker [watch_kaggle_batch.py](../../scripts/watch_kaggle_batch.py), batchmanifesthash bịkhóa, statusCLIaccountscoped exact`owner/slug/1`,
  max4queryđồngthời, sleep1200giây giữađợt, khôngmodel/W&Bpolling/pull/relaunch.
- Lầnđầu16:10:58: **12RUNNING/0terminal**. `runs/l-screening/watch-20m/latest.json`, history.jsonl cậpnhật nền;
  kernelterminalđãxácnhận khôngquerylại. QUERY_ERROR khônggọilàterminal.
- Khi cả12v1 thuộc COMPLETE/ERROR/CANCEL_ACKNOWLEDGED: ghi `terminal.json`, markerALL_KERNELS_TERMINAL rồithoát;
  supervisorqueuecompletion vềthreadgốc. Đây là monitorhoànthành, **khôngchứngminh trainingđạt10round/utility**.
- Khôngstartwatchertrùng; khôngmodelpoll/statuslặplại trongthờigianchờ. Hostphải cònhoạtđộng; watcherkhôngsurvivereboot.
  Dừngwatcher chỉkhiowneryêucầu bằngcontrollercancel, khôngdừng remoteKagglekernel.

[Spec/cổngL12run](../research/screening-l10.md). **L0–L11 version1 pushthànhcông14:16–14:18**,12/12exit0.
**12/12RUNNING,2×T4/source/W&Bauth đúng lúc14:21**,0Traceback trongstartup; `runs/l-screening/startup.json`.
Cảbatch freshinit10round/cosine50/full5343xe/W1/2×T4. Sourceđãđóngbăng, khôngsửa/re-push phiênv1.
Launch/logchecks đãxong; remotejob tiếp tục. Watcherlocal sởhữu poll20phút theoquyềnmới; khôngcollector nền.
`runs/l-screening/batch.json` là account/slug/HP/source/notebookhash; mỗi `.push.json/.push.log` ghi exit0/version1.

Thayđổi: `kd_weight=fixed` tùycell; `q_codec=mean_fp16` encode/decodeclassmean cảhai chiều.
Combo/gatecontrol cùngcodec mới thêm4run so đềxuất8. AugmentationK4/K5 táidùng vì nhánhnoQ toán học khôngđổi.
Shadowprobe34client +fulltrainCM tại1/2/5/10, thời gian riêng `t_diagnostics`.
Probe fp32/trainBN/dropout0/postepoch, khôngactualAMPgradient; train+val đãfit nên trainF1 khônglàindependentval.

Gateslocal:36testsPASS; legacyCPU2round3casebitexact; real4xe144306trainrows/2roundW1compileAMP PASS,skip0;
10metrictrain/testkhớpCM, fullprivatestate strictrebuild, teacherlagPASS. `runs/l-screening/smoke-audit.json`.
Quota14:08/kiểmlạitrướcpush:16account30hfree/0reserved,12account×4hcap=48hcapbatch, globaldinhan25.
W&Bonline/privatekey theoquyềncũ, key tồn tại trong lịch sửKaggle. KhôngGit/DVCpush/cleanup/50round.
Source_manifest12/12=5c6e9f9c4e0923b1a28a3b76a393e481018397168228f69fdc61743cd56f3b1b.
Chưacompilegate/W&Btrainingrunconfig/metricround/utilityauditL: workerđangpreparecache/bootstrap;
W&Bread14:19chưathấyruntraining (khácvớilogin đãPASS). Không coi push/COMPLETE/W&B là kếtquả cuối.

## K đã hoàn tất — không còn run đang chạy trong batch này

10 kernel version1 COMPLETE. Mỗi run đủ r1–10, full5343xe, W1/batch64/group_rows4096, backendcompiled, 2×T4 đã kiểm startup.
Pull metric/source37file + snapshot10file/run, exit0/0missing sau retry ConnectTimeout; mainpull~3,59GiB.
100round CM→10metric/perclass/clientstats khớp JSONL, source overlay/manifest/HP/LR/ID/support đúng9.347.679test.
100snapshotloadweights_only=True hữu hạn, Qshape/normalization đúng, augmentationQ=None.
**Chưa fullprivateweightsbank/prediction→CM reconstruction, trainF1 hoặc exactresume proof**. Không gọi snapshot nhỏ là fullprivatecheckpoint.

[ResultsK đầy đủ](../research/results-k10.md), [spec/cổng](../research/screening-k10.md), [thiết kế](../research/federated-design.md).
Artifact: `runs/k-screening/audit.json`, `analysis.json`, `snapshot-audit.json`, `completion-status.json`.

| cơ chế | arm42/43 | F1 seed42/43 | p10 seed42/43 | cổng42/43 |
|---|---|---|---|---|
| combo64/noReal | K0/K1 | .882381/.883028 | .603950/.590930 | FAIL/FAIL |
| augmentation-only | K4/K5 | .883348/.884334 | .619494/.607795 | PASS/FAIL |
| conditional8 | K6/K7 | .881809/.883008 | .604328/.592062 | FAIL/FAIL |
| realKDgate | K8/K9 | .883935/.884584 | .614002/.599486 | PASS/PASS |

K2dose43=.877733/p10.595349; K3noReal43=.878173/.564622; cùng cell42H3/J4 đủfactorial.
Interactiondose×noReal dương+.001856/+.001629, nhưng combo chưaqua gain.
Gate hơnaugmentation chỉ+.000586/+.000250F1; marginqua cổngseed43=.000076.
Gate benignF1~.861 vsaugmentation~.832, timeDelayF1~.14 vsaugmentation~.315: đánh đổi giữa lớp thật.
Augmentation p10/ítmẫu tốt hơn, không learnedmessage saubootstrap nhưng vẫn là federatedinputstats/prior, không gọi local-only.
Conditional đắt hơncombo màkhôngtăngF1; khôngquétB/routertestđểtìmwin.

## Chi phí và kết luận khả thi

- Base/gate:1088B/xe/round +3311744Bbootstrap ⇒3322624B/10r.
- Augmentation:0learnedB/round +3311744Bbootstrap. Không có liênlạc learnedteacher saubootstrap.
- Conditional:8256B/round +3312768Bbootstrap ⇒3395328B/10r.
- Gate trainwallmeanr2–10~631/649s trên job2×T4;augmentation~622/667s. Hardwaredaođộng, khôngcoi làbenchmarktốcđộnhânquả.
- Chưa transport/contactdeadline/mấtgói/clientpartialparticipation/mixedarchitecture/50round; chưa phươngpháptốiưutoàncục.

Đềnghị giữ **gate vsaugmentation** cho xácnhận dài hạn cùngseed. Gate làứngviênpooled/benign; augmentation làứngviêntail/ítgiao tiếp.
Owner yêu cầu phân tích sâu nguyênnhân trước: [kế hoạch đối chứng](../research/root-cause-experiments.md).
ĐãchạyL10round theoquyềnmới; chưa50round. Nếu ownerchốt50round: khóaarm/control/seed, kiểmquota; kiểm/sửa exactresume trước dùngparentstate.
Không ngoại suy10roundhai seed thành kếtluậncuối; không chọn best-round/HP theo test.

## Chẩn đoán bổ sung — ưu tiên hiện tại

Đã chạy [diagnose_kd_mechanisms.py](../../scripts/diagnose_kd_mechanisms.py) trên fullcount,40Qsnapshot,4CMr10;
output `runs/k-screening/mechanism-diagnosis.json` có hashes. K8/K9 gate chỉloại lớp13 ởQr1–r9 dùngcho trainingr2–10,
5,151%train. Syntheticteacher vàadaptive`ce_t` vẫn dùng hàng13 sai. Low256client=15,871%xe nhưng0,359%trainrows.
ClassQbyteharness hiện chưadecodepayloadfp16 vào training; phải kiểmcodec trước claim utility với precisiontruyềnthật.

Đềxuất8runmới=4cell×2seed tách soft/hardtarget, fixed/adaptivescale và realgate, táidùng K khi tươngthích.
Vì thaycodec, đãthêmcontrolcùngcode mới, khóathành12runL. Code/probe/smoke đãxong; xem mụcLtrên.
Kkhôngcó gradient/trainF1/fullweightbank mới; Lsẽbổsungprobe/trainF1, chưa cóartifactL đểaudit.

## Kernel IDs và source đã đóng băng

| arm | account | slug |
|---|---|---|
| K0 | nguyenminhtran060113 | mh-k0-combo-s42 |
| K1 | thuyanh316 | mh-k1-combo-s43 |
| K2 | minhtriethihi | mh-k2-dose-s43 |
| K3 | annhtnt | mh-k3-noreal-s43 |
| K4 | takids1909 | mh-k4-aug-s42 |
| K5 | annhannh331 | mh-k5-aug-s43 |
| K6 | trietbackup | mh-k6-bucket-s42 |
| K7 | khanhmay0304 | mh-k7-bucket-s43 |
| K8 | catbaochau | mh-k8-gate-s42 |
| K9 | odixeuit | mh-k9-gate-s43 |

Mọi kernel version1. BaseGitSHA099e95e8466811881fb71dcd7d3c44268ed082d6;
source_manifest117cd4da82de23786275dbedb255b62b3d83f44fcad9dded24b2cc4b2ccefff1.
SourceSHA/notebookSHA `runs/k-screening/batch.json`; provenance mỗi `runs/pulls/SLUG/`.
Không sửa/re-push notebook/source đã chạy. Notebookprivate/ignored0600, W&Bkey theoquyềncũ.

## Pull/network và bước tiếp

Một số truy vấn `api.kaggle.com` ConnectTimeout (không auth401/403). Đã dùng savedtokenfilepath trong process,
khôngswitchgloballogin/khônghỏiloginlại. Metricfallback ở `runs/k-screening/token-pulls`, `bounded-pull` đãverifyexit0/0missing,
mergeimmutablev1byte-identical vào `runs/pulls`. SnapshotfallbackK5 giữ riêng; mainfoldercũngđủ10snapshot từretryexit0.
`runs/k-screening/pull_bounded.py` đặt timeout cảrequestsSession.request **vàsend** vìSDKgọiSession.sendtrực tiếp.
TácvụcollectK khôngcài watcher/collector nền. Quyềnmới31/10đãthêm watcherL20phút nhưmụcLtrên; khôngcollector nền.

1. Chờwatcher queuecompletion; khi đượcđánhthức đọcstate/log/terminal.json trước. Khôngmodelpolling hoặcstartjobtrùng lúcđợi.
   Lđãlaunch/startup12/12; khi tiếptục kiểm exactv1status+log/W&B trước, khôngpushtrùng.
   Khi xongpull/audit đủ10round/codec/probe/trainCM trước đánhgiá phươngpháp. Scriptread-only `runs/l-screening/check_live.py`,
   `wandb_check.py`; cácscriptkhôngcài watcher. Pulltheoskills rồi `scripts/audit_screening.py --batch runs/l-screening/batch.json`;
   ngoài auditchung cònphảikiểm overlay3file, trainCMsupport khớptrain_counts vàprobe34IDs/rank/teacherlag/finite theoLspec.
   Khôngrerun phiên đãpush khi chưakiểmversion/status/artifact; chưalaunch50round.
2. Nếu cần model/prediction audit: pull fullstate cả2rank đúngv1, kiểmdisk/RNG/source/config trướcrestore; khôngclaimexactresume sẵncó.
3. Khi so utility: dùngG4seed42=.873187514625593/p10=.5737619047619047; H0bseed43=.8745079727196574/.5687566137566138.
   Cổnggain≥.010,p10Δ≥−.010 từngseed; giữcổng dùsátngưỡng. Artifactđối chứng cũ ởH/J/G, khôngcopynguồn kếtquảbaselinerepo.

Không việc tay bắt buộc hiện tại. Phiên này sửa method/driver/probe, thêmtest và launchL theoowner;
khôngGit/DVCpush/cleanup. Giữmọichỉnhsửa cósẵn từphiêntrước.
Linktài liệusống phảiPASS; linkhỏngcũhistory/v1giữnguyên.
