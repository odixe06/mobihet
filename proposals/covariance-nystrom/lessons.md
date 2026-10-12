# Bài học — covariance Nyström

S1 chưautilitypilot; không gọi proposal đãthắng IDS hoặc thêmfamily mới vào danh mục24hướng.
[Thiết kế/cổng](design.md), [proof](../../papers/nystrom/math-probe.json).

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
