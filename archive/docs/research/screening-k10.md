# K0–K9 — kiểm chứng cơ chế và thiết kế federated mới (31/10/2026)

Owner duyệt tiếp screening10round fullK, ưu tiênF1, không trầnbyte cứng; báo trade-off và tính cảbootstrap.
Có thể xây lại phương pháp; giữ DAGSNet chung trước, kiến trúc trộn sau. **Không share local data**:
không gửi dòng thật/feature từngmẫu/nhãn từngdòng; chỉ thống kê tổng hợp và meanprobability theo nhóm.
Không Git/DVCpush, không50round/F trong batch. Nền/cổng/lesson: [J](screening-j10.md), [H](screening-h10.md),
[thiết kế từ bài toán](federated-design.md), [lý do factorial](validation-k-design.md).

## Bằng chứng → cơ chế → bác bỏ

J4−G4 pooled+.003629 nhưng timeDelay+.214792, cả15lớp khác giảm; 2895client kém hơn, 2272tốt hơn,
nhóm<256mẫu/xe meanΔ−.008366. RealKD có lợi regularization cho nhiều lớp/xe; bỏ đồngloạt chưa phải kết luận thiết kế.
H3dose64 cải thiệnp10; cộngthuần H3+J4−G4≈.880524 vẫn dưới .883188. Combo chỉPASS nếu tươngtácdương đủ lớn.
J1sampleteacher hơnJ0bankQ[y] .002123 nhưng thuaG4; thêm131136B/round chưa cho utility đángchi.
Không tiếp tục anchorselector chỉ vì entropy/teacheragreement cao.

1. Factorial liều32/64 × realKD on/off ở hai seed, tái dùng cell đã audit. Bác bỏ tươngtác nếu combo không vượt tổng hiệuứng riêng.
2. Augmentation-only: mọi round CEreal+LA + CEsynthetic hệsố1, giống côngthứcr1; không learnedteacher, không adaptive raw, khôngQexchange.
   Vẫn dùng globalcopula/priorbootstrap; **không gọi local-only hoặc khôngFL hoàn toàn**. Đây là compoundchannelablation,
   không tách riêng soft-vs-hard hoặc hệsốloss. Nếu combo không hơncontrol, chưa chứng minh learnedKD đánggiữ.
3. Conditionalteacher từ dòng thật: bảng Q[y,b(x)] giữ thôngtininput, không cần post-epoch eval trên anchor OOD.
   Router cốđịnh3projection ±1 trên77input, seed20261031, b∈0…7. Cùngrouter mọi seed/rank; khôngfit trên test.
   Train-only bucketcounts mộtlần; trong forwardlocal có sẵn, gộp privateprob theo lớp×bucket. Gửi meanfp16,
   decode trướcgộpcount-weighted và decode/chuẩnhóa bảngdownlink. Bucketrỗng fallback vềmarginalclass.
   SyntheticKD dùng conditionalQ; adaptive raw vẫn marginalQ suy từ cùngbảng/count; realKDoff nhưcombo.
   Ngoài conditioning, arm này đổi encoding uplink thành meanfp16 thay pipelineclasssum cũ; không quy mọi Δ cho riêngrouter.
   Bác bỏ nếu conditional không hơncombo, bucketcollapse, hoặc utility/xeweak hỏng. Không quét B/routerseed theo test.
4. Label-agreement gate: trên dose64, giữ realKL chỉ ở dòng có argmax Q[y]=y. Nhãntrain local đãcó; khôngtestlabel/serverrowlabel.
   SyntheticKD/raw/cáchgộpQ giữ nguyên. Chia mean bằng toàn sốdòng thật, không renormalize phần giữ lại.
   Bác bỏ nếu không hơnH3 hoặc vẫn hại nhiều lớp/xe nhưnoReal; không gọi argmaxđúng là thầy luôn đángtin.

## Batch khóa trước push

Full5343xe, W1, batch64, seed42/43, group_rows4096, 2×T4; cosine50round dừngr10; W&Bmobihet-v4.
Mọi arm none/synthcopula/anchoroff; mỗi run cap4h. Không chọnHP/best-round theo test.

| arm | slug | seed | thayđổi soG4 |
|---|---|---:|---|
| K0 | mh-k0-combo-s42 |42| dose64 + real_kd=false |
| K1 | mh-k1-combo-s43 |43| nhưK0 |
| K2 | mh-k2-dose-s43 |43| dose64; cellH3seed43 |
| K3 | mh-k3-noreal-s43 |43| real_kd=false,dose32; cellJ4seed43 |
| K4 | mh-k4-aug-s42 |42| dose64,no_kd=true,real_kd=false |
| K5 | mh-k5-aug-s43 |43| nhưK4 |
| K6 | mh-k6-bucket-s42 |42| K0+kd_bins=8 |
| K7 | mh-k7-bucket-s43 |43| K1+kd_bins=8 |
| K8 | mh-k8-gate-s42 |42| dose64,real_kd_gate=true,realKDtrue |
| K9 | mh-k9-gate-s43 |43| nhưK8 |

Cellcũ táidùng: G4/H3/J4seed42, H0bcopulaseed43. Không rerun để tạo đối chứng mới.
Factorial interaction = combo−dose64−noReal+base, tính mỗi seed riêng.
K6−K0/K7−K1 kiểm conditional; K8−H3/K9−K2 kiểm gate; K0−K4/K1−K5 kiểm learnedteacher vs augmentation.
Hai seed chỉlặp dấu/độnhạy; không đủ khẳng định ýnghĩathốngkê, không ngoại suyfinal50.

## Cổng và kế toán

PrimaryprivatepooledF1r10. Seed42 giữ G4=.873187514625593,p10=.5737619047619047.
Seed43 dùng pairedH0b=.8745079727196574,p10=.5687566137566138. Cổngtừngseed: ΔF1≥.010,Δp10≥−.010.
Headlinegainkhóa42 không đổi. Báo từngseed/từnglớp/nhómclient, không lấyseed/round đẹp nhất.
Kèm10metric mỗi round, mean/p10/median/min, benign,timeDelay, loss/skip,t_train/t_eval,source/ID/support/snapshot.
Guardp10 cũgiữ nguyên; mean/xe và sốclasshại báo minh bạch, không tuyên bố mỗiclientđều đượclợi.

- K0/K1/K2/K3/K8/K9:1088B/xe/round, bootstrap3311744B ⇒ tổng10r3322624B.
- K4/K5: learnedteachertraffic0B; vẫn bootstrap3311744B. Chưa tốiưu packetinputstats/prior riêngcontrol.
- K6/K7:8256B/xe/round, thêm1024Bbootstrapcounts ⇒ bootstrap3312768B, tổng10r3395328B.
  Không gửiQmarginalriêng; dựng từ cùngconditionaltable/count. Header32B mỗichiều nhưharness.

Không gửi rawrow, sampleinput, hiddenrepresentation hayprivateweights. Server chỉ thấyclass/bucketcount vàmeanprob,
cùng pooledinputmoments/histogrambootstrap. Synthetic sinhlocal, không phải dòng thật truyền từclientkhác.
Không claim DP/secureaggregation vì chỉ có thốngkêcộng tính. Bytechưatransport/retransmission.
Budget toànbatch40hcap +margin trên10account; quotaaccount-scoped ngaytrướcpush, không đổi globallogin.

## Kiểm chứng trước launch

26CPUtestPASS; gauss/copula/dose64default2roundtoyvsJsourceBIT_EXACT.
Train-onlyrouterpreflight toàn410file/37361273dòng:125/128bucketoccupied (97,656%), entropyperclass0,251–1,956nats;
Sybilentropy0,251, suddenStop0,827 ⇒ khôngcoipartitioncânbằng/đúngboundary. CPUprojection chỉchẩnđoán, cóthểkhácGPUởđiểmsátranh.
Smoke100xe2roundcompiled+AMP, bucket+gatePASS/skip0; r2train55s/53s. Augmentation2roundGPUcũngPASS/skip0,r2train52s.
Bank/anchorJ đãđóngbăng, không sửasource hoặc re-push J. SourceK riêng cóhash/provenance, khôngGitpush.
Notebookprivate,ignored0600, W&Binlinekey theoquyềncũ; keytồn tại lịch sửversionprivate.
Runtime/evidence/account đúng ở [continue](../operations/continue.md), `runs/k-screening/batch.json`.

Push10/10version1,RUNNING~01:14; startup10/10đúng2×T4/source/W&Bauth, source_manifest117cd4da82de23786275dbedb255b62b3d83f44fcad9dded24b2cc4b2ccefff1. Chưa cór1/utilityK.


## Kết quả đã audit — 31/10 ~10:27

[Kết quả đầy đủ,10metric/cổng/chi phí/khảthi](results-k10.md). K8/K9gatePASScảhaiseed; K4augmentationPASS42nhưngK5FAIL43sátngưỡng.
Combo/conditionalFAIL; augmentationhơncombo ởcảhaiseed. Khôngđổicổng,khôngchọnbest-round. 10runđủr1–10/100snapshotfinite; chưa50round/mixedarchitecture.
