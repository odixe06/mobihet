# Bài học — covariance Nyström

V7 hai seed đãaudit20arm×20round: **compressionPASS, qualitypairedFAIL**.
Không mở contact/full hoặc rerun y nguyên. Không thêmfamily mới vào danh mục24hướng.
[Thiết kế/cổng](design.md), [proof](../../papers/nystrom/math-probe.json).

- Byte/full rank4≈0,198; rank8≈0,321. FL−exactlr4≈−0,0195/−0,0240;
  FL−exactlr8≈−0,0183/−0,0117, vượt mức giảm cho phép0,01.
- Rank8−shuffle≈+0,0109/+0,0125 đạt contribution riêng; rank4 chỉ+0,0017/+0,0009.
  Hơnown/cócontribution không đủ giữ utility của covariance chính xác.
- Local covarianceerror và quantizationerror là diagnostic, chưa cô lập nguyên nhân F1.
  Không đổi rank/Ω/ridge/head/bestround để cứu cổng.
- [Kết quả/audit](../../papers/v7/report.md), [cổng khóa](../../papers/v7/gates.json).

- 40.960uniformtrainrow, CPU78MiB: additiveFP32upload/FP64aggregate +FP16factor r4/8,
  PSD/finite/context/CRC/tamper/truncationgates PASS. KhôngMH/sourceV6mutation, khôngtrain/GPU.
- Actualapplication+paidprior r4=39.080B/xe,r8=66.088B; matchedfull219.016B.
  Wupper cótruyền, headers/context trảphí; khácpayloadestimate cũ khôngW/frame.
- Relativecovarianceerror mean≈0,493/0,432, tốt hơndiag≈0,658 nhưng kémexacteigenrank≈0,329/0,258.
  Sketch width=r hẹp không thừa hưởngpaper oversamplingguarantee; chưaF1proof.
- FactorFP16 quantizationerror worst≈0,045%, nhỏhơnapproximationerror nhiều.
  Khôngmặcđịnh tăngprecision chữa thiếucapacity của sketch trênmẫu này.
- Residualfloor3,88%, negative≈4,15e−9; PSD củafactor khôngchứngminh covariance/decisionboundarykhớpfull.
- Center saucohortaggregate, không cộnglocal-centeredcov bỏbetween-clientmeans.
- Nyström inputcovariance khônglatentdrift, nhưng Gaussian support/phi tuyến limitation vẫncòn.
- V6fulluplinkFP64, S1FP32: cầnfreshmatchedFP32full/exactlr controls đểisolatesketch;
  metadatareference V6 khôngtự làmatchedruntimecontrol.

Khôngtune Ω/seed/ridge theoholdout hoặc kéo dàitrain cứu cổng.
Utility trướccontact; B0r4 ít nhất6up+3downcontacts, byte nhỏchưa deadline/activationproof.
