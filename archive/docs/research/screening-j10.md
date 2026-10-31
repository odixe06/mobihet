# J0–J4 — teacher theo mẫu và chẩn đoán KD (30/10/2026)

Owner phiên hiện tại: ưu tiên F1, báo đủ thời gian/byte; duyệt tiếp sàng lọc 10 round full 5.343 xe như trước.
Giữ [protocol](protocol.md), 50-round cosine dừng r10, W1, batch64, seed42, group_rows4096, 2×T4, W&B online.
Không launch 50 round. Không Git/DVC push. Thiết kế khóa trước push, không chọn HP theo test.
Nguồn/lý do kế thừa v1: [nghiên cứu KD](synthetic-kd-next.md); kết quả tiền nhiệm [H](screening-h10.md).

## Chẩn đoán dataset và teacher

Full scan 410 parquet train, 37.361.273 dòng, chỉ đọc client_id/label; đối chiếu 9.347.679 own-test từ CM G4 r10.
Lệnh tái lập: `python scripts/diagnose_private.py --data /home/odixe/project/mobihet-features/data/v4-scaled --run runs/pulls/mh-g4-copula/mh-g4-copula --output runs/j-screening/diagnosis.json`.

| train/xe | số xe | F1 mean/xe G4 r10 | p10 |
|---|---:|---:|---:|
| <256 | 848 | 0,5868 | 0,2833 |
| 256–1023 | 1.444 | 0,7558 | 0,5821 |
| 1024–4095 | 1.528 | 0,8229 | 0,7292 |
| 4096–16383 | 923 | 0,8547 | 0,8137 |
| ≥16384 | 600 | 0,8803 | 0,8490 |

Lớp absent ở train local: 26.999 dòng test (0,289%), accuracy 0,7189; lớp seen: 9.320.680 dòng, accuracy 0,9118.
Số mẫu train của đúng client×lớp từ1–16:49.318 dòng test, accuracy0,6923;17–64:191.881 dòng/0,7999;
65–256:667.728 dòng/0,8559;>256:8.411.753 dòng/0,9200. Thiếu lớp có hại, nhưng không giải thích phần lớn lỗi pooled. Không suy từ tương quan số mẫu tới tác động nhân quả.

Lỗi lớn nhất timeDelay(13)→benign(0): **460.575/482.549** mẫu timeDelay; Q[13,13]=0,2790, argmax Q[13]=benign.
Teacher Q thu trong train, không phải eval teacher sau epoch. Q sai lớp đích là bằng chứng kênh truyền có thể củng cố nhầm,
chưa chứng minh KD gây lỗi. Dataset đã biết timeDelay dời cả timestamp gửi/nhận, khó phân biệt từ feature hiện có.
Giữ 77 feature của protocol, không tự đổi dataset để tránh mất đối chiếu baseline.

## Cơ chế và đối chứng

Bank A cố định suốt run, M4096, cân bằng16 lớp, seed=seed*1009, cùng mọi rank/client. Giữ RNG sampler cũ khi dựng bank.
Mỗi bước lấy số dòng như nền (32; J3=64), đúng số mỗi lớp; chọn instance ngẫu nhiên trong lớp. Không mix dòng thật.
J1–J3: cuối epoch private eval trên A; upload round(255p) uint8, decode và chuẩn hóa trước gộp.
Gộp theo n(k,y_i), lượng tử uint8 rồi decode xuống; synthetic KD round r+1 dùng đúng q_i^r và x_i cùng index.
Real-row KD vẫn Q[y] như cũ; r1 chưa teacher thì synthetic CE. Hệ số adaptive giữ nguyên.
Không proxy, không thêm optimizer/model; private không rời xe, output16 lớp chung cho kiến trúc khác nhau.

| arm | slug | HP riêng | đối chứng / câu hỏi |
|---|---|---|---|
| J0 | mh-j0-anchor-class | synth=copula anchor=class | G4: bank hữu hạn có làm mất phủ? |
| J1 | mh-j1-anchor-kd | synth=copula anchor=ensemble | J0: teacher từng mẫu có thêm tri thức? |
| J2 | mh-j2-gauss-kd | synth=gauss anchor=ensemble | J1: copula còn có lợi với teacher mới? |
| J3 | mh-j3-anchor-dose | synth=copula anchor=ensemble synth_rows=64 | J1: kết hợp liều H3 với teacher mới? |
| J4 | mh-j4-no-real-kd | synth=copula real_kd=false | G4 streaming: KD nhãn trên real đang gây hại? |

Mọi arm arm=none, m_transfer=4096. Một yếu tố mỗi cạnh so sánh; J3 là tương tác đã nêu trước.
J4 chỉ đặt thành phần KL real về0; vẫn tính CE_teacher trong adaptive weight, giữ Q/bootstrap/synthetic KD và CE+LA.
Không đồng nhất J4 với bỏ toàn bộ KD. Không mở lại NTD, hidden-KD, GMM/MADE hoặc BN/head v1 FAIL.

## Cổng, chi phí, dấu hiệu bác bỏ

Primary private pooled F1-macro r10, G4=.873187514625593, p10=.5737619047619047.
Cổng cải tiến giữ nguyên: ΔF1≥.010 và Δp10≥−.010 vs G4; không chọn round đẹp, không đổi cổng khi FAIL.
So sánh J1−J0, J1−J2, J3−J1 mô tả cơ chế; seed42 đơn, chưa chứng minh ý nghĩa thống kê hay tối ưu toàn cục.
Báo đủ10 metric/per-class/mean/p10/min, benign, support/ID, loss/skip, source, t_train/t_eval và bootstrap+hai chiều.

- J1 không hơn J0: chưa hỗ trợ giả thuyết mất thông tin từng mẫu là nút thắt; có thể support ngoài phân phối.
- J0 thua G4: finite-bank thay đổi support có hại; không gán toàn bộ chênh J1−G4 cho teacher.
- J4 không hơn G4: chưa có bằng chứng real KD gây hại từ ablation này; không sửa tiếp dựa vào test.
- J3 không hơn J1: lợi dose H3 chưa chuyển sang teacher mới.

J0/J4 steady state1.088 B/xe/round; J1–J3 thêm hai gói (32+4096*16)B ⇒ **132.224 B hai chiều/round**.
J1/J3 copula bootstrap3.311.744 B ⇒ tổng10r **4.633.984 B/xe** nếu gửi đủ10 round.
J2 gauss bootstrap394.368 B ⇒ tổng10r1.716.608 B. J0/J4 tổng3.322.624 B.
Dù không cần teacher sau r10, vẫn tính downlink đó như harness, không trừ để làm đẹp chi phí.
Không tính transport/secure aggregation. Anchor gửi bằng seed, không truyền dữ liệu thật.
M4096 thêm **21.884.928** forward dòng/round (4096×5343); runtime phải đo trên hai T4, không suy từ FLOPs.
Cap4h mỗi run, 5 account độc lập, cần≥20h reserved budget toàn batch + margin; đo quota ngay trước push.

## Kiểm chứng

22test CPU PASS; rebuilt consensus từ post-epoch private+wire khớp. Gauss/copula/dose64 mặc định so sourceH2roundtoy BIT_EXACT.
Smoke100xe thật compiled+AMP2round, M4096: J1dose32 và J3dose64 PASS, skip0; r2train46s/51s, bytes132224hai chiều đúng.
Source/metadata/notebook compile/hash validated; notebookprivate/ignored0600. Giới hạn local1GPU chưa chứng minh2T4/fullK.
Trạng thái runtime ở [bàn giao](../operations/continue.md).

Push5/5v1, GPU/source/W&B auth5/5PASS~22:41; chưa có utilityJ. Runtime theo bàn giao.


## Kết quả audit 31/10 ~01:00

5kernel version1 COMPLETE; bulkpullbỏstate*.pt exit0,47file/run,0thiếu,~366–370MiB/run.
Selective metric/source pull cũng exit0,37file/run,0thiếu;50/50snapshot loadweights_only=True, tensorhữuhạn.
Audit `scripts/audit_screening.py runs/j-screening/audit-input --batch runs/j-screening/batch.json --output runs/j-screening/audit.json`.
Đủr1–10, HP/LR/source overlay/IDs0…5342/support từngclient×lớp khớp G4; đủ9.347.679test.
10metric mỗi round tái tính từCM khớpJSONL; chưaprediction→CM/fullprivatebank reconstruction.

| arm | F1r10 | ΔvsG4 | p10 | Δp10 | timeDelayF1 | train s/r(r2–10) | bootstrap+10r B/xe | cổng |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| J0 | 0.869116 | -0.004072 | 0.568346 | -0.005416 | 0.036948 | 537.2 | 3,322,624 | FAIL |
| J1 | 0.871238 | -0.001949 | 0.578149 | +0.004388 | 0.043489 | 562.9 | 4,633,984 | FAIL |
| J2 | 0.869407 | -0.003780 | 0.582198 | +0.008436 | 0.042637 | 587.0 | 1,716,608 | FAIL |
| J3 | 0.872993 | -0.000194 | 0.616190 | +0.042428 | 0.047261 | 663.8 | 4,633,984 | FAIL |
| J4 | 0.876817 | +0.003629 | 0.566417 | -0.007345 | 0.239879 | 539.8 | 3,322,624 | FAIL |

**Tất cả J FAIL cổng gain+0,010**. J1−J0=+0,002123: giữ target từngmẫu giúp nhỏ trên cùngbank, nhưng chưa bù mất support
J0−G4=−0,004072. J1−J2=+0,001831; J3−J1=+0,001755, p10tăng nhưng pooled chưa vượtG4.
Không mở tiếp anchor/selector chỉ vì teacheragreement cao: r10 J1=.8828,J3=.9033 nhưng utility vẫnFAIL.
J4−G4=+0,003629; p10−0,007345 còn trongguardrail. timeDelayF1tăng nhưng vẫn phải xét lớp/xe bịhại, không gọiKDreal hoàn toànvôích.
H3+J4 additive≈.880524, vẫn dưới .883188; combo cần tươngtácdương mớiPASS.

### Mười metric r10

| arm | accuracy | precision_macro | precision_micro | precision_weighted | recall_macro | recall_micro | recall_weighted | f1_macro | f1_micro | f1_weighted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| J0 | 0.908817 | 0.919626 | 0.908817 | 0.908001 | 0.863583 | 0.908817 | 0.908817 | 0.869116 | 0.908817 | 0.888490 |
| J1 | 0.910603 | 0.918231 | 0.910603 | 0.907338 | 0.865719 | 0.910603 | 0.910603 | 0.871238 | 0.910603 | 0.890510 |
| J2 | 0.909934 | 0.917146 | 0.909934 | 0.906681 | 0.864014 | 0.909934 | 0.909934 | 0.869407 | 0.909934 | 0.889755 |
| J3 | 0.911162 | 0.914724 | 0.911162 | 0.904896 | 0.868265 | 0.911162 | 0.911162 | 0.872993 | 0.911162 | 0.891303 |
| J4 | 0.906503 | 0.893110 | 0.906503 | 0.897611 | 0.875635 | 0.906503 | 0.906503 | 0.876817 | 0.906503 | 0.895199 |

Fullperround/perclass/client mean/p10/median/min: `runs/j-screening/audit.json`; nhómclient/Δclass: `runs/j-screening/comparisons.json`.
J4max skipped_frac=1,7053e-5; các armJkhác0. Chi phí là payloadharness, chưatransport/secureaggregation.

J4 tăng timeDelayF1+0,214792 nhưng15lớpcònlạiđềugiảm;2895clientkémhơn,2272tốthơn. Xe<256mẫumeanΔ−0,008366.
Không chọn bỏrealKDđồngloạt chỉtừpooled. Batchkiểmchứngtiếp: [K](screening-k10.md).
