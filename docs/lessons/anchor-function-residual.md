# Bài học — Residual hàm từ nhãn local trên anchor

Giao diện hàm tránh đồng bộ private architecture nhưng không tự đồng bộ probe weights; cùng anchor chưa đảm bảo sharedrepresentation.

[Thiết kế/nguồn](../archive/methods/anchor-function-residual/README.md).

## V9 cả hai seed đã audit

13arm đều đủ20round; exactsource/BN/cohort/whole-statecodec/targets/checkpoint/pred→10metric/inference PASS.
M64/M512 costPASS (0,267/0,271×freshdenseCE hai round đầu), nhưng lockedteacherqualityFAIL.
FL−own+0,0595/+0,0382; FL−classmean+0,00161/+0,00160;
FL−hard+0,00330/−0,00140; FL−shuffle+0,00768/−0,00073.
Thualocalprivategp−0,25045/−0,13249. Hơnown chưađủranhgiới đa receiver.
M64refresh−frozen+0,02142 PASS; M512+0,00098 FAIL. ComponentcontributionPASS khônglàutilityPASS.
Tănganchors64→512 cải thiệnđiểm nhưng vẫntrượtcổng; chưacôlậpbottleneck bandwidth/BN/fit/synthetic support.
Actualtrain-onlycoverage chọnh4 cảhaiM trên1.000.369pilottrainrow, không43Mfulltrain;
zeroanchor-mass0 vàr20teacherfitMSE0,000171/0,001728. Fitbook tốt khôngchứng minh generalization.
Seed43: FL−local −0,26382/−0,13688, −classmean −0,00356/+0,00088;
−shuffle +0,00042/+0,00325. M64 refresh contribution PASS/M512 FAIL cả hai seed.
Cost PASS, quality paired FAIL: fit tốt và contribution riêng không thay hiệu quả teacher.
Không tune bandwidth/M/HP/head hoặc kéo dài run để tìm điểm thắng;
không mở private KD/contact/full hoặc rerun y nguyên.
[Report](../archive/methods/anchor-function-residual/builds/anchor-residual/results/report.md), [locked gates](../../archive/methods/anchor-function-residual/builds/anchor-residual/probes/gates.json),
[10metric CSV](../archive/methods/anchor-function-residual/builds/anchor-residual/results/metrics-all.csv), [F1 từng lớp](../archive/methods/anchor-function-residual/builds/anchor-residual/results/per-class-f1.csv).
