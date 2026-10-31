> Snapshot trước phân tích cơ chế31/10. Bàn giao sống: [continue](../operations/continue.md).

# Continue — trạng thái 31/10/2026 ~10:28 (giờ VN)

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [lessons vận hành](../operations/lessons.md).
Snapshot trước collect K: [01:15](../archive/continue-2026-10-31-0115.md).
Repo này nghiên cứu federated/MobiHet; baseline/NILM/Notion ở `../fl_dagsnet_veremi_v2`.

## Quyền và ràng buộc

- Private model có thể khác kiến trúc; không gửi raw local row/feature từng mẫu/nhãn từng dòng/private weights.
- Chưa trần byte, ưu tiên F1, báo cả bootstrap+hai chiều và thời gian. Giữ DAGSNet chung trước, mixedarchitecture sau.
- Đã duyệt screening10round trên cosine50round; **chưa launch/chốt F50round**. Không Notion/Git/DVCpush/cleanupdata.
- Global CLI **dinhan25** giữ nguyên; repo dùng CLI, MCPJSON đã xóa theo owner. Account khác qua `kaggle_as.py`.

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
Chưa chạy runmới trong tácvụ này. Nếu ownerchốt50round: khóaarm/control/seed, kiểmquota; kiểm/sửa exactresume trước dùngparentstate.
Không ngoại suy10roundhai seed thành kếtluậncuối; không chọn best-round/HP theo test.

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
Không cài watcher/collector nền; không tiếp tục giám sát sau khi đóngphiên.

1. Khi owner yêu cầu tiếp: đọc resultsK và chốt phạmvi50round/đốichứng; khôngrerun cell10round đãaudit.
2. Nếu cần model/prediction audit: pull fullstate cả2rank đúngv1, kiểmdisk/RNG/source/config trướcrestore; khôngclaimexactresume sẵncó.
3. Khi so utility: dùngG4seed42=.873187514625593/p10=.5737619047619047; H0bseed43=.8745079727196574/.5687566137566138.
   Cổnggain≥.010,p10Δ≥−.010 từngseed; giữcổng dùsátngưỡng. Artifactđối chứng cũ ởH/J/G, khôngcopynguồn kếtquảbaselinerepo.

Không việc tay bắt buộc hiện tại. Phiên này chỉ thêm audit/analysis/report/handoff/lessons và sửa auditparser kiểmđủmetric;
không đổi method/traincode, khônglaunch/Git/DVCpush/cleanup. Giữmọichỉnhsửa cósẵn từphiêntrước.
Linktài liệusống phảiPASS; linkhỏngcũhistory/v1giữnguyên.
