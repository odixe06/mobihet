# Proposal 2 — MobiHet (T3)

MobiHet hiện tại là **T3**, owner chọn15/11/2026; các bảng chỉ dùng T3, không dùng kết quả Core/CBB/KM16 cũ thay T3.
VeReMi NextGen v4:5343xe nhận,77đầu vào,16lớp; train+val37361273dòng, test9347679dòng temporal holdout.50round×1epoch thật, batch64, AdamW cosine1e-3→1e-5, clip1, AMP; seed41/42/43.

Bootstrap một lần: histogram256bin/lớp/feature→CDF toàn mạng; normal-score mid-rank; moment lớp và k-means liên kết16cụm/lớp,4lượt Lloyd, covariance shrink w0=.5, loại cụm<20dòng. Lite6 nén moment/tâm int8, Cholesky6bit, CDF lossless. Xe tự sinh từ mixture copula; không gửi dòng thật/model/gradient/BN.

T3 phân tầng theo n_k(train+val): S<4096, M4096–16383, L≥16384. Synthetic/bước r1–25→r26–50: **S64→64; M64→32; L48→16**. Floor16bước; thiếu bước thật thì chỉ thêm synthetic, không thêm epoch thật. Loss synthetic lấy mean riêng, trọng số1; giảm số dòng không giảm hệ số loss.

Prior Laplace π_k,c=(n_k,c+1)/(n_k+16), π_g,c=(Σn_k,c+1)/(Σn_k+16). Loss thật=CE(z+logπ_k−logπ_g,y), synthetic dùng logits thô. Round1 synthetic hardCE; round2+ KL(Q_(r−1)[y] || softmax(f_k(x_syn))).

Online Q_r,c=ΣkΣi:y=c softmax(f_k(x_i))/Σkn_k,c, thu trên dòng thật ngay lúc train, dùng từr+1; không thêm lượt forward riêng. QQ8 gửi xác suất8bit, uplink chỉ lớp hiện diện. Mỗi xe giữ model riêng; schema độc lập kiến trúc. BERT/DAGSNet là hai thí nghiệm riêng, chưa full-scale trộn model.

## 1. Performance

Endpoint50; CANPerFL=FedAvg50 +10epoch fine-tune. Mean ± sample SD,3seed41/42/43. Ranking Macro-F1 giảm dần. Global/own-test/prototype khác luật suy luận; không gọi các view đồng nhất.

### BERT

| Hạng | Phương pháp | View | Accuracy | Macro recall | Macro F1 | Client P10 F1 | Client min F1 |
|---|---|---|---|---|---|---|---|
| **1** | **MobiHet** | **private** | **0.9277 ± 0.0002** | **0.9103 ± 0.0004** | **0.9148 ± 0.0002** | **0.7694 ± 0.0010** | **0.2416 ± 0.0107** |
| 2 | CANPerFL | client | 0.9247 ± 0.0011 | 0.9064 ± 0.0010 | 0.9117 ± 0.0015 | 0.7830 ± 0.0030 | 0.2005 ± 0.0417 |
| 3 | APFed | client | 0.9263 ± 0.0007 | 0.9021 ± 0.0011 | 0.9076 ± 0.0015 | 0.7620 ± 0.0030 | 0.1736 ± 0.0064 |
| 4 | FedAvg | global | 0.9211 ± 0.0013 | 0.8938 ± 0.0009 | 0.8990 ± 0.0018 | 0.7444 ± 0.0080 | 0.1770 ± 0.0014 |
| 5 | PerFed-SKD | client | 0.9123 ± 0.0026 | 0.8862 ± 0.0021 | 0.8922 ± 0.0031 | 0.6832 ± 0.0111 | 0.1838 ± 0.0086 |
| 6 | FedProx | global | 0.9166 ± 0.0012 | 0.8853 ± 0.0024 | 0.8886 ± 0.0024 | 0.7436 ± 0.0057 | 0.1831 ± 0.0059 |
| 7 | FedProto | client_proto | 0.8969 ± 0.0018 | 0.8877 ± 0.0020 | 0.8876 ± 0.0025 | 0.5845 ± 0.0109 | 0.0475 ± 0.0266 |
| 8 | AFPHA | global | 0.9156 ± 0.0008 | 0.8835 ± 0.0027 | 0.8868 ± 0.0026 | 0.7432 ± 0.0054 | 0.1834 ± 0.0063 |
| 9 | TinyProto | client_proto | 0.8946 ± 0.0023 | 0.8867 ± 0.0020 | 0.8857 ± 0.0026 | 0.5856 ± 0.0083 | 0.0850 ± 0.0385 |
| 10 | FedTGP | client_proto | 0.9078 ± 0.0022 | 0.8739 ± 0.0021 | 0.8826 ± 0.0027 | 0.5693 ± 0.0111 | 0.0000 ± 0.0000 |
| 11 | LwFedNIDS | global | 0.9125 ± 0.0019 | 0.8783 ± 0.0020 | 0.8823 ± 0.0017 | 0.7229 ± 0.0110 | 0.1741 ± 0.0083 |
| 12 | NILM-FL | client | 0.9142 ± 0.0012 | 0.8761 ± 0.0008 | 0.8821 ± 0.0010 | 0.5687 ± 0.0124 | 0.1142 ± 0.0157 |
| 13 | FD-IDS | global | 0.9130 ± 0.0023 | 0.8804 ± 0.0015 | 0.8814 ± 0.0003 | 0.7221 ± 0.0193 | 0.1963 ± 0.0137 |
| 14 | V2X-HtFL | client | 0.9073 ± 0.0006 | 0.8721 ± 0.0006 | 0.8770 ± 0.0007 | 0.5492 ± 0.0031 | 0.0588 ± 0.0303 |

### DAGSNet

| Hạng | Phương pháp | View | Accuracy | Macro recall | Macro F1 | Client P10 F1 | Client min F1 |
|---|---|---|---|---|---|---|---|
| 1 | APFed | client | 0.9278 ± 0.0000 | 0.9040 ± 0.0008 | 0.9130 ± 0.0010 | 0.7557 ± 0.0036 | 0.1965 ± 0.0706 |
| 2 | CANPerFL | client | 0.9236 ± 0.0005 | 0.8995 ± 0.0008 | 0.9097 ± 0.0009 | 0.7736 ± 0.0045 | 0.1969 ± 0.0570 |
| **3** | **MobiHet** | **private** | **0.9128 ± 0.0003** | **0.9088 ± 0.0001** | **0.9080 ± 0.0003** | **0.7781 ± 0.0003** | **0.2123 ± 0.0535** |
| 4 | FedAvg | global | 0.9230 ± 0.0002 | 0.8964 ± 0.0007 | 0.9048 ± 0.0009 | 0.7573 ± 0.0030 | 0.1733 ± 0.0227 |
| 5 | FedProx | global | 0.9217 ± 0.0006 | 0.8943 ± 0.0010 | 0.9005 ± 0.0017 | 0.7575 ± 0.0046 | 0.1867 ± 0.0090 |
| 6 | AFPHA | global | 0.9214 ± 0.0002 | 0.8931 ± 0.0007 | 0.8993 ± 0.0012 | 0.7563 ± 0.0036 | 0.1869 ± 0.0094 |
| 7 | FD-IDS | global | 0.9215 ± 0.0007 | 0.8911 ± 0.0009 | 0.8990 ± 0.0016 | 0.7527 ± 0.0024 | 0.2084 ± 0.0080 |
| 8 | LwFedNIDS | global | 0.9146 ± 0.0020 | 0.8823 ± 0.0016 | 0.8892 ± 0.0016 | 0.7169 ± 0.0157 | 0.1402 ± 0.1038 |
| 9 | NILM-FL | client | 0.9156 ± 0.0000 | 0.8719 ± 0.0003 | 0.8850 ± 0.0006 | 0.5772 ± 0.0036 | 0.0639 ± 0.0567 |
| 10 | FedTGP | client_proto | 0.9020 ± 0.0000 | 0.8612 ± 0.0002 | 0.8719 ± 0.0000 | 0.5716 ± 0.0005 | 0.0000 ± 0.0000 |
| 11 | PerFed-SKD | client | 0.8993 ± 0.0001 | 0.8603 ± 0.0004 | 0.8713 ± 0.0004 | 0.6549 ± 0.0030 | 0.1263 ± 0.0198 |
| 12 | FedProto | client_proto | 0.8972 ± 0.0005 | 0.8660 ± 0.0008 | 0.8693 ± 0.0009 | 0.5924 ± 0.0013 | 0.0131 ± 0.0226 |
| 13 | V2X-HtFL | client | 0.8973 ± 0.0003 | 0.8564 ± 0.0009 | 0.8666 ± 0.0004 | 0.5544 ± 0.0032 | 0.0198 ± 0.0186 |
| 14 | TinyProto | client_proto | 0.8873 ± 0.0009 | 0.8462 ± 0.0009 | 0.8541 ± 0.0006 | 0.5497 ± 0.0024 | 0.0131 ± 0.0226 |

## 2. Communication budget

Đơn vị B, MB=10⁶B; unicast từng xe, cả upload+download. Tổng50=bootstrap+Σround(up+down); tổng mạng=5343×tổng/xe. Chưa tính header mạng/retransmission. Baseline=payload_bytes(fp32), PerFed-SKD nhân selected_frac đo seed42; TinyProto thêm mask một lần. MobiHet=bytes_up/down50round, bootstrap trong config, trung bình3seed. Ranking tổng gồmbootstrap tăng dần; bootstrap luôn tách riêng.

### BERT

| Hạng | Phương pháp | Up B/xe/round | Down B/xe/round | Tỷ lệ chọn | Bootstrap MB/xe | 50round không bootstrap MB/xe | Tổng MB/xe | Mạng GB |
|---|---|---|---|---|---|---|---|---|
| 1 | TinyProto | 448.0 | 448.0 | 1.0000 | 0.000128 | 0.044800 | 0.044928 | 0.2401 |
| 2 | FedTGP | 4096.0 | 4096.0 | 1.0000 | 0.000000 | 0.409600 | 0.409600 | 2.1885 |
| 3 | FedProto | 4160.0 | 4096.0 | 1.0000 | 0.000000 | 0.412800 | 0.412800 | 2.2056 |
| 4 | V2X-HtFL | 4160.0 | 4096.0 | 1.0000 | 0.000000 | 0.412800 | 0.412800 | 2.2056 |
| **5** | **MobiHet** | **171.1** | **177.3** | **1.0000** | **1.151278** | **0.017421** | **1.168699** | **6.2444** |
| 6 | LwFedNIDS | 40776.0 | 40776.0 | 1.0000 | 0.000000 | 4.077600 | 4.077600 | 21.7866 |
| 7 | PerFed-SKD | 311616.0 | 311616.0 | 0.3811 | 0.000000 | 11.876868 | 11.876868 | 63.4581 |
| 8 | AFPHA | 311616.0 | 311616.0 | 1.0000 | 0.000000 | 31.161600 | 31.161600 | 166.4964 |
| 9 | APFed | 311616.0 | 311616.0 | 1.0000 | 0.000000 | 31.161600 | 31.161600 | 166.4964 |
| 10 | CANPerFL | 311616.0 | 311616.0 | 1.0000 | 0.000000 | 31.161600 | 31.161600 | 166.4964 |
| 11 | FD-IDS | 311616.0 | 311616.0 | 1.0000 | 0.000000 | 31.161600 | 31.161600 | 166.4964 |
| 12 | FedAvg | 311616.0 | 311616.0 | 1.0000 | 0.000000 | 31.161600 | 31.161600 | 166.4964 |
| 13 | FedProx | 311616.0 | 311616.0 | 1.0000 | 0.000000 | 31.161600 | 31.161600 | 166.4964 |
| 14 | NILM-FL | 311616.0 | 311616.0 | 1.0000 | 0.000000 | 31.161600 | 31.161600 | 166.4964 |

### DAGSNet

| Hạng | Phương pháp | Up B/xe/round | Down B/xe/round | Tỷ lệ chọn | Bootstrap MB/xe | 50round không bootstrap MB/xe | Tổng MB/xe | Mạng GB |
|---|---|---|---|---|---|---|---|---|
| 1 | TinyProto | 1664.0 | 1664.0 | 1.0000 | 0.000512 | 0.166400 | 0.166912 | 0.8918 |
| **2** | **MobiHet** | **130.6** | **137.8** | **1.0000** | **1.151278** | **0.013420** | **1.164697** | **6.2230** |
| 3 | FedTGP | 16384.0 | 16384.0 | 1.0000 | 0.000000 | 1.638400 | 1.638400 | 8.7540 |
| 4 | FedProto | 16448.0 | 16384.0 | 1.0000 | 0.000000 | 1.641600 | 1.641600 | 8.7711 |
| 5 | V2X-HtFL | 16448.0 | 16384.0 | 1.0000 | 0.000000 | 1.641600 | 1.641600 | 8.7711 |
| 6 | LwFedNIDS | 162000.0 | 162000.0 | 1.0000 | 0.000000 | 16.200000 | 16.200000 | 86.5566 |
| 7 | PerFed-SKD | 1594688.0 | 1594688.0 | 0.2861 | 0.000000 | 45.617687 | 45.617687 | 243.7353 |
| 8 | AFPHA | 1594688.0 | 1594688.0 | 1.0000 | 0.000000 | 159.468800 | 159.468800 | 852.0418 |
| 9 | APFed | 1594688.0 | 1594688.0 | 1.0000 | 0.000000 | 159.468800 | 159.468800 | 852.0418 |
| 10 | CANPerFL | 1594688.0 | 1594688.0 | 1.0000 | 0.000000 | 159.468800 | 159.468800 | 852.0418 |
| 11 | FD-IDS | 1594688.0 | 1594688.0 | 1.0000 | 0.000000 | 159.468800 | 159.468800 | 852.0418 |
| 12 | FedAvg | 1594688.0 | 1594688.0 | 1.0000 | 0.000000 | 159.468800 | 159.468800 | 852.0418 |
| 13 | FedProx | 1594688.0 | 1594688.0 | 1.0000 | 0.000000 | 159.468800 | 159.468800 | 852.0418 |
| 14 | NILM-FL | 1594688.0 | 1594688.0 | 1.0000 | 0.000000 | 159.468800 | 159.468800 | 852.0418 |

## 3. Computational cost

Client: cost.py đo một round2 toy CPU không đệm bằng FlopCounterMode (matmul/conv/bmm/SDPA), chia dòng train. Không tính elementwise/optimizer/codec/eval/bootstrap; không suy ra thời gian T4. BERT forward10.225664MFLOP, DAGSNet4.452096MFLOP. Baseline giữ cùng quy ước đo; PerFed-SKD trên dòng xe được chọn. CANPerFL=(50+10)/50×FedAvg, phân bổ fine-tune vào50round.

MobiHet ước lượng công việc hữu ích theo B21: c_train×(1+ρ), ρ=Σ max(ceil(n_k/64),16)×mean_dose_k/(64Σceil(n_k/64))=0.651785; mean_dose S/M/L=64/48/32. Dùng c_train riêng từng model, không nhân4.94 DAGSNet sang BERT. Chưa cộng sinh copula/codec/bootstrap. Dòng mask/đệm vẫn có thể được kernel thực thi; đây không phải phép đo total GPU FLOPs.

### Client — BERT

| Hạng | Phương pháp | MFLOP/dòng/round | ×forward | Loại bằng chứng |
|---|---|---|---|---|
| 1 | LwFedNIDS | 5.880 | 0.575 | toy measured |
| 2 | AFPHA | 40.022 | 3.914 | toy measured |
| 3 | FedAvg | 40.022 | 3.914 | toy measured |
| 4 | FedProto | 40.022 | 3.914 | toy measured |
| 5 | FedProx | 40.022 | 3.914 | toy measured |
| 6 | CANPerFL | 48.026 | 4.697 | toy measured; CANPerFL amortized |
| 7 | FD-IDS | 50.248 | 4.914 | toy measured |
| 8 | FedTGP | 50.248 | 4.914 | toy measured |
| 9 | TinyProto | 50.248 | 4.914 | toy measured |
| 10 | V2X-HtFL | 50.285 | 4.917 | toy measured |
| 11 | PerFed-SKD | 60.473 | 5.914 | toy measured |
| **12** | **MobiHet** | **66.108** | **6.465** | **ước lượng T3** |
| 13 | APFed | 80.044 | 7.828 | toy measured |
| 14 | NILM-FL | 80.044 | 7.828 | toy measured |

### Client — DAGSNet

| Hạng | Phương pháp | MFLOP/dòng/round | ×forward | Loại bằng chứng |
|---|---|---|---|---|
| 1 | LwFedNIDS | 1.288 | 0.289 | toy measured |
| 2 | AFPHA | 13.297 | 2.987 | toy measured |
| 3 | FedAvg | 13.297 | 2.987 | toy measured |
| 4 | FedProto | 13.297 | 2.987 | toy measured |
| 5 | FedProx | 13.297 | 2.987 | toy measured |
| 6 | CANPerFL | 15.957 | 3.584 | toy measured; CANPerFL amortized |
| 7 | FD-IDS | 17.749 | 3.987 | toy measured |
| 8 | FedTGP | 17.749 | 3.987 | toy measured |
| 9 | TinyProto | 17.749 | 3.987 | toy measured |
| 10 | V2X-HtFL | 18.290 | 4.108 | toy measured |
| **11** | **MobiHet** | **21.964** | **4.933** | **ước lượng T3** |
| 12 | PerFed-SKD | 22.201 | 4.987 | toy measured |
| 13 | APFed | 26.594 | 5.973 | toy measured |
| 14 | NILM-FL | 26.594 | 5.973 | toy measured |

Server: steady-state một round; loại client training/eval và bootstrap. K=5343,C=16,d=64(BERT)/256(DAGSNet), Np=count ô train+val có lớp. Aggregation ước lượng số phép nhân/cộng/chia từ code; server neural step đo matmul bằng FlopCounterMode,100epoch,20batch/epoch như implementation. Ranking theo con số trong phạm vi nêu; không là ranking latency thực. AFPHA là cận dưới do chưa tính clustering; không suy thứ hạng tuyệt đối khi cost bỏ sót có thể đổi thứ tự. Bootstrap copula/KM16 cần gộp histogram/moment, Lloyd4lượt và factorization cụm; chưa đo, không tính là0 hoặc<5ms.

### Server — BERT

| Hạng trong phạm vi | Phương pháp | Triệu FLOP/operation mỗi round | Cách tính/phạm vi |
|---|---|---|---|
| 1 | TinyProto | 0.490 | Np d_eff + C d_eff; aggregation only |
| **2** | **MobiHet** | **2.822** | **2 K C² + K C + 3 C²; weighted Q aggregation/normalization, excludes codec** |
| 3 | FedProto | 4.481 | Np d_eff + C d_eff; aggregation only |
| 4 | LwFedNIDS | 108.943 | 2 K_selected (P+B)+(P+B); weighted aggregation |
| 5 | PerFed-SKD | 317.368 | 2 K_selected (P+B)+(P+B); weighted aggregation |
| 6 | AFPHA | 832.560 | 2 K_selected (P+B)+(P+B); weighted aggregation; lower bound, excludes adaptive clustering/distances |
| 7 | APFed | 832.560 | 2 K_selected (P+B)+(P+B); weighted aggregation |
| 8 | CANPerFL | 832.560 | 2 K_selected (P+B)+(P+B); weighted aggregation |
| 9 | FD-IDS | 832.560 | 2 K_selected (P+B)+(P+B); weighted aggregation |
| 10 | FedAvg | 832.560 | 2 K_selected (P+B)+(P+B); weighted aggregation |
| 11 | FedProx | 832.560 | 2 K_selected (P+B)+(P+B); weighted aggregation |
| 12 | NILM-FL | 832.560 | 2 K_selected (P+B)+(P+B); weighted aggregation |
| 13 | FedTGP | 30242.259 | FlopCounter: setup/final +100×epoch; excludes elementwise/optimizer |
| 14 | V2X-HtFL | 33388.544 | FlopCounter: setup/final +100×epoch; excludes elementwise/optimizer |

### Server — DAGSNet

| Hạng trong phạm vi | Phương pháp | Triệu FLOP/operation mỗi round | Cách tính/phạm vi |
|---|---|---|---|
| 1 | TinyProto | 1.820 | Np d_eff + C d_eff; aggregation only |
| **2** | **MobiHet** | **2.822** | **2 K C² + K C + 3 C²; weighted Q aggregation/normalization, excludes codec** |
| 3 | FedProto | 17.922 | Np d_eff + C d_eff; aggregation only |
| 4 | LwFedNIDS | 432.824 | 2 K_selected (P+B)+(P+B); weighted aggregation |
| 5 | PerFed-SKD | 1219.075 | 2 K_selected (P+B)+(P+B); weighted aggregation |
| 6 | AFPHA | 4260.608 | 2 K_selected (P+B)+(P+B); weighted aggregation; lower bound, excludes adaptive clustering/distances |
| 7 | APFed | 4260.608 | 2 K_selected (P+B)+(P+B); weighted aggregation |
| 8 | CANPerFL | 4260.608 | 2 K_selected (P+B)+(P+B); weighted aggregation |
| 9 | FD-IDS | 4260.608 | 2 K_selected (P+B)+(P+B); weighted aggregation |
| 10 | FedAvg | 4260.608 | 2 K_selected (P+B)+(P+B); weighted aggregation |
| 11 | FedProx | 4260.608 | 2 K_selected (P+B)+(P+B); weighted aggregation |
| 12 | NILM-FL | 4260.608 | 2 K_selected (P+B)+(P+B); weighted aggregation |
| 13 | FedTGP | 139846.550 | FlopCounter: setup/final +100×epoch; excludes elementwise/optimizer |
| 14 | V2X-HtFL | 190186.717 | FlopCounter: setup/final +100×epoch; excludes elementwise/optimizer |

## Nguồn và giới hạn

CM/support→10metric audit PASS_BOUNDED; T3 bổ sung finite private banks, strict rebuild và RNG/source/HP hai rank khi có state audit. Chưa stored prediction→CM replay toàn mạng; chưa full-content dataset fingerprint. Không tuyên bố mixed-model full-scale, secure aggregation/DP hoặc significance từ3seed.

[BERT source/results](https://github.com/odixe06/fl_veremi_bert) · [MobiHet source/design](https://github.com/odixe06/mobihet) · [Baseline DAGSNet](https://github.com/odixe06/fl_dagsnet_veremi_v2)

Chi tiết số liệu12metric/seed và nguồn từng hàng: `runs/maintenance-20261116/comparison.json` ở repo BERT; bảng này tái sinh bằng `scripts/build_project_comparison.py`.
