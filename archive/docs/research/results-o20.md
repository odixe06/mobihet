# O20 — kết quả teacher replay

**8/8run đã audit đủ20round.** 816file/6,189,021,370B; SHA20snapshot/run, mọi file source, scientificclosure so N và20CM→10metric/support đã đối chiếu độc lập.
Fullaudit còn gồm finite/probe/externalhash/teacherlag/rawQ/target. Không privatebank/predictionreplay/fullcontentfingerprintdataset hoặc DPproof.
[Thiết kế khóa](screening-o20.md), [cơ chế/reference đã thử](../../../docs/research/method.md). R0=fullQ/noReal, R1=fullQ/realgate, R2=uniform/noReal, R3=permuted/noReal.
Own-test5343xe, endpoint20/cosine50, W1; Q nguồn train-only N0/N1, dùng lag1, không livefeedback từ Q_live.

## Own-test tại round20

| arm / cell / seed | pooledF1 | mean clientF1 | p10 | mean xe<256 | timeDelayF1 | ΔF1r15→20 |
|---|---:|---:|---:|---:|---:|---:|
| O0/R0/42 | 0.893383 | 0.834585 | 0.695345 | 0.767510 | 0.320134 | +0.003161 |
| O1/R0/43 | 0.894653 | 0.837897 | 0.701808 | 0.780438 | 0.322362 | +0.002940 |
| O2/R1/42 | 0.894795 | 0.837692 | 0.699952 | 0.773289 | 0.263823 | +0.003770 |
| O3/R1/43 | 0.895800 | 0.839676 | 0.701139 | 0.780116 | 0.262050 | +0.003842 |
| O4/R2/42 | 0.892237 | 0.834939 | 0.697408 | 0.772334 | 0.331264 | +0.002777 |
| O5/R2/43 | 0.893710 | 0.837303 | 0.700742 | 0.782280 | 0.332957 | +0.003033 |
| O6/R3/42 | 0.891584 | 0.833108 | 0.692871 | 0.772138 | 0.330117 | +0.002758 |
| O7/R3/43 | 0.893202 | 0.835523 | 0.699552 | 0.777437 | 0.330548 | +0.002985 |

## Đối chiếu cùng teacher nguồn

Chênh tuyệt đối endpoint20. Other15=mean pooledF1 của15lớp trừ timeDelay; không headline mới.

| contrast / seed | ΔpooledF1 | Δmean client | Δp10 | Δmean xe<256 | ΔtimeDelay | Δother15 | xe tăng/giảm |
|---|---:|---:|---:|---:|---:|---:|---:|
| O2-O0 real_gate/42 | +0.001412 | +0.003107 | +0.004606 | +0.005779 | -0.056310 | +0.005260 | 2816/2206 |
| O4-O0 uniform/42 | -0.001146 | +0.000353 | +0.002063 | +0.004825 | +0.011131 | -0.001965 | 2473/2579 |
| O6-O0 permuted/42 | -0.001799 | -0.001477 | -0.002474 | +0.004628 | +0.009983 | -0.002585 | 2343/2712 |
| O3-O1 real_gate/43 | +0.001147 | +0.001780 | -0.000669 | -0.000322 | -0.060312 | +0.005244 | 2757/2267 |
| O5-O1 uniform/43 | -0.000943 | -0.000594 | -0.001066 | +0.001843 | +0.010596 | -0.001713 | 2410/2639 |
| O7-O1 permuted/43 | -0.001451 | -0.002374 | -0.002257 | -0.003001 | +0.008187 | -0.002094 | 2336/2710 |

### Điều phép kiểm chứng đã làm rõ

1. **Tiếp nhận realKD vẫn có hại cục bộ khi giữ cùng Q:** R1−R0 timeDelay=−.056310/−.060312,
   other15=+.005260/+.005244; cả15lớp còn lại đều tăng pooledF1 ở cảhai seed. Pooled tăng+.001412/+.001147,
   nhưng meanxe<256 vàp10 chỉ tăng seed42. Không tắt/tăng KD đồng loạt hoặc chốt winner cho xe yếu.
2. **Teachertrajectory cũng quan trọng:** O2/O3 so livegateN2/N3 pooled tăng+.001890/+.001882,
   timeDelay tăng+.045720/+.045302, other15 giảm khoảng.00103/.00101. Cùng clientloss nhưng lịch teacher nguồn noReal khác livefeedback.
   Kết quả phù hợp feedback/teachertrajectory làm trầm trọng đánh đổi; không là phân rã trung gian nhân quả hoặc tỷ lệ lỗi do feedback.
   Q19 nguồn13 có Qyy≈.707/.705, còn Q_live do chính O2/O3 collect≈.595/.593 dù được lịch teacher nguồn mạnh hơn.
   Đây là train-onlyconfidence/động lực học, chưa phải ước lượng teacherquality trên tương lai.
3. **Không nên bỏ toàn bộ offdiagonal:** uniform giảm pooled.001146/.000943, permutation giảm.001799/.001451.
   Permutation giữ diagonal/entropy mà vẫn giảm: có bằng chứng utility của quan hệ tên lớp trong đúng phép replay này,
   chưa mọi permutation/architecture. Uniform lại tăng meanxe<256+.004825/+.001843 và timeDelay≈+.011/+.011;
   không thể suy fullQ hữu ích đồng đều cho mọi xe/lớp hoặc diagonal-only đủ thay thế.
4. **Chưa plateau toàn hệ thống:** R1 vẫn tăng r15→20+.003770/+.003842; không hứa mỗi round tăng hoặc ngoài20round.
   Gap train/test timeDelay còn lớn ở xe≥4096: R1train≈.749/.747 so test≈.177/.176.
   Thiếu mẫu/lớp không giải thích hết, cần validation theo thời gian chưa fit trước khi dùng confidence làm serverqualityweight.

Ưu tiên tiếp: giữ cấu trúc Q đã có utility, kiểm tiếp nhận KD có chọn lọc theo thông tin train-only và feedbackserver;
gate liên tục vẫn là giả thuyết, chưa chứng minh giải quyết crossclasshại hoặc chỉ giảm lossstrength.
Không tăng packet/đổi generator hoặc triển khai cacheteacher như liveFL từ kết quả này. [Registry/falsifier tiếp](diagnostic-next.md).

FullQ vs uniform thay cả entropy; fullQ vs fixedpermutation giữ diagonal/entropy nhưng phá tên lớp.
UniformO giữ diagonal/chia đều offdiagonal, khác uniformM12 trộn hard/softρ=.5; không cộng gain hai can thiệp.
Không realKD ở uniform/permuted nên argmax sau biến đổi không mở gate. Permutation chỉ một phép đã khóa, chưa kiểm nhiều permutation.
Sameclient giữa seed có tương quan; hai seed chưa significance. Không gọi pooledgain là mọi client được lợi.

## Tripwire replaycontrol so N0/N1

| control | Δpooled | ΔtimeDelay | Δp10 | Δmean xe<256 | giới hạn khóa |
|---|---:|---:|---:|---:|---|
| O0 | +0.000098 | +0.000515 | -0.000101 | -0.000522 | PASS |
| O1 | +0.000175 | +0.001366 | +0.000418 | +0.002905 | PASS |

Giới hạn đã khóa trước O: |Δpooled|≤.001, ba metric còn lại≤.01. Không CI/noisebound/utilitythreshold; không bitexact giữa T4sessions.

## Xe nhỏ và tổng quát hóa

Accuracy điều kiện trên1555absent-localtestrows của848xe<256; không macroF1 mới.

| arm | absentaccuracy xe<256 | matched trainrecall timeDelay | matched testrecall timeDelay | cross-client34 meanF1 (phụ) |
|---|---:|---:|---:|---:|
| O0 | 0.531833 | 0.841983 | 0.238242 | 0.795085 |
| O1 | 0.540193 | 0.847079 | 0.241400 | 0.797958 |
| O2 | 0.470096 | 0.762839 | 0.175357 | 0.792869 |
| O3 | 0.488103 | 0.761872 | 0.173730 | 0.794474 |
| O4 | 0.559486 | 0.857803 | 0.262653 | 0.800705 |
| O5 | 0.542765 | 0.857157 | 0.263477 | 0.803132 |
| O6 | 0.539550 | 0.849259 | 0.263068 | 0.799357 |
| O7 | 0.528617 | 0.848708 | 0.262854 | 0.802889 |

Matchedrecall cân theo supporttest481868rows cùng xe/lớp đã thấy; train+val đãfit, không independentvalidation.
Gap không tự tách overfit/temporalshift/featureoverlap. External34 chỉ evaluator, không vào train/Q/bootstrap và không dùng chọnHP.

## Byte và thời gian

Mọi run: packet-equivalent3333504B/xe/20round; offlinepreload3402795B gồm3311744bootstrap+10880uplink+80171Binputmodule.
Không cộng lại cached downlink hoặc cộng cả decoded61440B và cùng module như hai lượt tải. Replay là counterfactual, chưa liveFL/contactcostproof.

| arm | train s/r r2–20 | train tổng s | eval tổng s | diagnostic tổng s | notebooklog elapsed s |
|---|---:|---:|---:|---:|---:|
| O0 | 648.8 | 13073.9 | 636.1 | 608.2 | 15946.7 |
| O1 | 630.1 | 12718.4 | 638.1 | 570.5 | 14863.0 |
| O2 | 664.1 | 13369.1 | 643.8 | 608.5 | 16501.5 |
| O3 | 644.5 | 12989.3 | 613.1 | 578.4 | 15303.7 |
| O4 | 647.1 | 13043.6 | 625.7 | 576.5 | 15069.0 |
| O5 | 631.3 | 12735.1 | 614.5 | 557.0 | 14757.6 |
| O6 | 643.4 | 12976.2 | 667.5 | 609.3 | 16142.1 |
| O7 | 651.4 | 13129.9 | 628.5 | 590.7 | 15431.4 |

Giây walljob2×T4, không latency mỗi xe. Notebookelapsed gồm setup/compile/IO/conversion, chưa queue/collectlocal.
Đủ10metric mọi round và16lớp cuối: [phụ lục](metrics-o20.md). Officialtest đã đọc nhiều batch; không freshconfirmatorytest/best-round/HPtuning.
Chưa r>20/mixedarchitecture/DP proof; không chọn phương pháp tổng từ replay. Artifact mới chưa DVCbackup, giữ local.

![O20 replay own-test curves](../../../docs/research/figures/o20-curves.png)

Nguồn phân tích: `runs/o-study/analysis/analysis.json`; tái tạo `python scripts/analyse_o20.py`, rồi `python scripts/report_o20.py`.
