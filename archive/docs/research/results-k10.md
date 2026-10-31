# K0–K9 — kết quả và tính khả thi, 31/10/2026

**Gate theo nhãn train là ứng viên F1; augmentation-only là đối chứng giao tiếp tối giản cần giữ.**
K8/K9 qua cổng đã khóa ở hai seed, nhưng lợi so augmentation nhỏ và p10 thấp hơn. Chưa chốt phương pháp cuối hay50round.
Thiết kế/cổng khóa trước push: [screening-k10](screening-k10.md). Cấu trúc federated: [federated-design](federated-design.md).

## Bằng chứng và phạm vi

10 kernel version1 COMPLETE, r1–10 trên cosine50, full5343client/W1/batch64, đúng source overlay/provenance.
Snapshot pull10file/run cũng exit0,0thiếu; bộmainpulltổng3.59GiB (khôngtínhbảnretrytrùng);100/100snapshotloadweights_only=True hữu hạn; Q/bucketQshape/chuẩnhóa đúng, augmentationQ=None.
Metric/source đãpull exit0,37file/run,0thiếu (một số ConnectTimeout đã retry với credential account-scoped; globalCLI giữdinhan25).
`python scripts/audit_screening.py runs/pulls --batch runs/k-screening/batch.json --output runs/k-screening/audit.json`:
10round liên tục, HP/LR,IDs0…5342, supporttừngclient×lớp khớpG4,9.347.679test;10metric mỗi round và F1perclass khớpCM.
Config/source/featureorder/labelmapping đãđốichiếu; không có fingerprintcontent của toàn dataset trong config cũ.
Không có trainF1 (chỉtrainloss), nênchưađo generalizationgap.
Không có predictions/fullprivateweightsbank trong bộaudit này; chưa rebuildprediction→CM hoặc xácnhậnexactresume.

## Utility r10 và cổng

Seed42 soG4 F1=.873187514625593,p10=.5737619047619047; seed43 soH0b F1=.8745079727196574,p10=.5687566137566138.
PASS cần ΔF1≥.010 và Δp10≥−.010 mỗi seed. Không chọn best-round hoặc nới ngưỡng cho kết quả sát cổng.

| arm | F1 | ΔF1 cùng seed | p10 | Δp10 | mean/xe | cổng |
|---|---:|---:|---:|---:|---:|---|
| K0 combo64/noReal42 | 0.882381 | +0.009193 | 0.603950 | +0.030188 | 0.791141 | **FAIL** |
| K1 combo64/noReal43 | 0.883028 | +0.008520 | 0.590930 | +0.022174 | 0.786168 | **FAIL** |
| K2 dose64/realKD43 | 0.877733 | +0.003225 | 0.595349 | +0.026592 | 0.788801 | **FAIL** |
| K3 noReal32/43 | 0.878173 | +0.003665 | 0.564622 | -0.004135 | 0.771103 | **FAIL** |
| K4 augmentation42 | 0.883348 | +0.010161 | 0.619494 | +0.045732 | 0.798245 | **PASS** |
| K5 augmentation43 | 0.884334 | +0.009826 | 0.607795 | +0.039038 | 0.793477 | **FAIL** |
| K6 conditional42 | 0.881809 | +0.008622 | 0.604328 | +0.030566 | 0.791748 | **FAIL** |
| K7 conditional43 | 0.883008 | +0.008500 | 0.592062 | +0.023306 | 0.786680 | **FAIL** |
| K8 gate42 | 0.883935 | +0.010747 | 0.614002 | +0.040240 | 0.798086 | **PASS** |
| K9 gate43 | 0.884584 | +0.010076 | 0.599486 | +0.030729 | 0.792820 | **PASS** |

K4 chỉPASSseed42; K5seed43FAIL gain dù rất sát. K8/K9PASSgain ở cảhai seed; marginseed43 chỉ khoảng0,000076.
Hai seed cung cấp lặp dấu/độnhạy, chưa chứng minh ý nghĩa thống kê hoặc tổng quát hóa tới50round.

## Các đối chứng cơ chế đã khóa

| phép so sánh pooled F1 | Δ |
|---|---:|
| interaction_seed42 | +0.001856 |
| interaction_seed43 | +0.001629 |
| augmentation_minus_combo_seed42 | +0.000967 |
| augmentation_minus_combo_seed43 | +0.001307 |
| conditional_minus_combo_seed42 | -0.000572 |
| conditional_minus_combo_seed43 | -0.000019 |
| gate_minus_dose_seed42 | +0.007039 |
| gate_minus_dose_seed43 | +0.006851 |

- Tương tác dose×noReal dương cảhai seed, nhưng combo K0/K1 vẫnFAIL cổng gain. Không gọi synergy nhỏ là phươngphápđạt.
- Augmentation-only hơncombo ở cảhai seed cảpooledF1 vàp10. Trong10round, chưa có bằng chứng learnedclassQ/noReal đánggiữ hơncontrol này.
  Control đổi cảtarget/lossnormalization; không quy toàn bộ lợi cho riêng việc bỏpacket/softtarget.
- Conditional8bucket không hơncombo ở cảhai seed. Thêm thôngtin/router vàbyte chưa mua đượcutility; bácbỏ adaptationB8 trongscreen này,
  không suy ra mọi conditionalteacher đều vôích. Không quét B/routerseed theo test để tìmwin.
- Gate hơndose64/realKD ở cảhai seed. Giữ KD khiteacherargmax khớpnhãntrain hợp lý hơn bỏ đồngloạt vềpooledF1;
  vẫn cần phân tích client/lớp yếu, không coi nhãnargmaxđúng là bảo đảmteacher tốt trên mọi mẫu.

## Giao tiếp và thời gian

Số B/xe gồmupload+download; tổng10round cộngbootstrap. Đây làpayloadharness, chưa transport/ACK/retransmission/secureaggregation.
Giây train làwall-time đoạntrain của job2×T4, không làGPU-utilizationtime; hardware/setup giữaaccount códaođộng.

| arm | B/xe/round | bootstrap B/xe | tổng10r B/xe | train s/r r2–10 | eval s/r r2–10 | maxskip |
|---|---:|---:|---:|---:|---:|---:|
| K0 | 1,088 | 3,311,744 | 3,322,624 | 624.6 | 16.8 | 6.82e-06 |
| K1 | 1,088 | 3,311,744 | 3,322,624 | 613.7 | 16.4 | 3.41e-06 |
| K2 | 1,088 | 3,311,744 | 3,322,624 | 654.5 | 19.5 | 0 |
| K3 | 1,088 | 3,311,744 | 3,322,624 | 553.2 | 19.6 | 1.19e-05 |
| K4 | 0 | 3,311,744 | 3,311,744 | 621.6 | 17.3 | 0 |
| K5 | 0 | 3,311,744 | 3,311,744 | 667.3 | 20.5 | 0 |
| K6 | 8,256 | 3,312,768 | 3,395,328 | 619.4 | 16.4 | 1.02e-05 |
| K7 | 8,256 | 3,312,768 | 3,395,328 | 647.4 | 18.8 | 6.82e-06 |
| K8 | 1,088 | 3,311,744 | 3,322,624 | 630.9 | 17.6 | 0 |
| K9 | 1,088 | 3,311,744 | 3,322,624 | 648.9 | 18.6 | 0 |

Bootstrapcopula3.311.744B/xe chi phối tổngchi phí. Gate thêm10.880B/xe qua10round soaugmentation, khoảng0,33%bootstrap,
nhưng cần liên lạc lặp đểcậpnhậtQ; augmentation không cần learnedmessage saubootstrap. Không gọi0Bsteady là0Bfederated.
Conditional bootstrap3.312.768B+82.560B/10r, đắt hơncontrol vàkhôngtăngF1.
Chưa đo contactdeadline/mấtgói/clientpartialparticipation; không tuyên bố hiệu quảthựcđịa từbyte nhỏ.

## Client và lớp — đánh đổi gate/augmentation

| arm | benign F1 | timeDelay F1 | clienttốt hơn nền | clientkém hơn nền | meanΔ xe<256mẫu |
|---|---:|---:|---:|---:|---:|
| K4 | 0.832115 | 0.315631 | 3216 | 2025 | +0.060106 |
| K5 | 0.833090 | 0.314207 | 3146 | 2097 | +0.051252 |
| K8 | 0.860803 | 0.141247 | 3519 | 1659 | +0.036248 |
| K9 | 0.860854 | 0.134276 | 3474 | 1715 | +0.023740 |

Gate tăng benignF1 lên~0,861 soaugmentation~0,832–0,833; augmentation timeDelayF1~0,314–0,316 so gate~0,134–0,141.
Đây làđánh đổi thực giữa lớp, không chỉnhiễugiữa pooledmetric.

Các so sánhclient dùng cùngIDs/own-test và nền cùngseed; không giảđịnhclient độc lập để tạoCI/significance.
Chi tiếtΔ16lớp vànhómclient tại `runs/k-screening/analysis.json`; không tuyên bố mọiclient đềuđược lợi.

## Mười metric r10

| arm | accuracy | precision_macro | precision_micro | precision_weighted | recall_macro | recall_micro | recall_weighted | f1_macro | f1_micro | f1_weighted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K0 | 0.908778 | 0.896104 | 0.908778 | 0.899743 | 0.882037 | 0.908778 | 0.908778 | 0.882381 | 0.908778 | 0.898205 |
| K1 | 0.909078 | 0.898710 | 0.909078 | 0.900090 | 0.881957 | 0.909078 | 0.909078 | 0.883028 | 0.909078 | 0.898033 |
| K2 | 0.913064 | 0.927115 | 0.913064 | 0.911405 | 0.874791 | 0.913064 | 0.913064 | 0.877733 | 0.913064 | 0.891999 |
| K3 | 0.906903 | 0.895344 | 0.906903 | 0.898018 | 0.876227 | 0.906903 | 0.906903 | 0.878173 | 0.906903 | 0.895605 |
| K4 | 0.900832 | 0.883145 | 0.900832 | 0.893955 | 0.885775 | 0.900832 | 0.900832 | 0.883348 | 0.900832 | 0.896432 |
| K5 | 0.901337 | 0.885095 | 0.901337 | 0.894290 | 0.885867 | 0.901337 | 0.901337 | 0.884334 | 0.901337 | 0.896802 |
| K6 | 0.908446 | 0.895793 | 0.908446 | 0.899387 | 0.881510 | 0.908446 | 0.908446 | 0.881809 | 0.908446 | 0.897720 |
| K7 | 0.909115 | 0.898902 | 0.909115 | 0.900175 | 0.881723 | 0.909115 | 0.909115 | 0.883008 | 0.909115 | 0.898115 |
| K8 | 0.914947 | 0.924807 | 0.914947 | 0.912036 | 0.877558 | 0.914947 | 0.914947 | 0.883935 | 0.914947 | 0.898532 |
| K9 | 0.914982 | 0.926157 | 0.914982 | 0.912394 | 0.878518 | 0.914982 | 0.914982 | 0.884584 | 0.914982 | 0.898295 |

Mỗi round cóđủ10metric/perclass/clientmean/p10/median/min trong `runs/k-screening/audit.json`.

## Kết luận khả thi và việc còn thiếu

Phân tích offline bổ sung31/10: [nguyên nhân và đối chứng](root-cause-experiments.md). Qr1–r9 củaK8/K9 chỉreject
timeDelay; gate thực thi r2–10 chỉtắt KLreal lớp này, còn teacher sai vẫn vào synthetictarget/`ce_t`.
`runs/k-screening/mechanism-diagnosis.json` ghi SHA40snapshot và4CM, classweightconcentration, hai chiều nhầmbenign/timeDelay.
Auditcode cũng cho thấy classQ chưa encode→decodefp16 trong đường training;1088B/r là accountingpayload của harness,
chưa phải utilityproof của precision truyền thực. Không sửa sốK hoặc sourceđãchạy; kiểmcodec và controlcùngphiênbản trước claim triểnkhai.

**Có thể tiếp tục xác nhận gate vàaugmentation, chưa đủ chốt methodtổng.**
Vềgiao diện, chỉ gộp thốngkêinput/count vàmeanprobability16lớp; không gửi dòngthật/featuretừngmẫu/nhãntừngdòng/privateweights.
Private inference độc lập. Cơchế không buộc kiếntrúc giốngnhau, nhưng thínghiệm hiện **chỉDAGSNetW1**; mixedarchitecture chưa kiểm.
Thốngkêtổnghợp khôngtựđộngbảovệDP; source không dùngsecureaggregation, không claimbảođảmprivacy hìnhthức.

Bước đề nghị: khóa gate vsaugmentation rồi chạy đủ50round với đối chứng cùngseed, sau đó kiểmkiếntrúctrộn.
Không launch50round trong tácvụpull/audit này. Vì gate hơnaugmentation rất nhỏ và p10thấphơn, giữ cảhai khi xácnhận;
chưa cócơsở chọn bằng best-round hoặc bỏcontrol để kể câu chuyện KD.
Giảmbootstrap làtrục chi phí cần kiểm riêng nếu tiếptục; không gộp thaygenerator/codec vào xácnhậnutility cùnglúc.
