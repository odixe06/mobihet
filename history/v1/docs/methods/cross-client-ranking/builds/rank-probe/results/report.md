# CrossClientRank — audited probe and projection diagnostic

Train-only CE-r2 snapshot, không SGD/holdout/utility. Gate đã khóa giữ nguyên; cả hai seed số học PASS.
Residual macro rất lớn không được diễn giải thành lượng tri thức hoặc improvement F1.

| seed | support/signal | coverage | locked residual macro | max condition | max coefficient |
|---|---|---:|---:|---:|---:|
| 42 | 16/16 | 1.000 | 41878.060674 | 5.178 | 817.948 |
| 43 | 16/16 | 1.000 | 168.037282 | 5.199 | 58.765 |

Dominant class5: `dosAttack`. Normalized design condition≤5,20; SVD ill-conditioning giả thuyết bị bác.
Client17 fit class5 CE-column RMS1,10e−7/2,62e−6; original coefficients817,95/58,76.
Fit class5 delta-energy1,05e−18/5,37e−16, nhưng heldout có một số record khác; top5 đóng góp
99,999995%/99,998230% residual energy lớp này. Coefficient lớn khuếch đại extrapolation.
Ở các client khác, shared bias/temperature fit ưu tiên toàn probe cũng có thể để residual lớp gần-zero lớn theo tỷ lệ.

Diagnostic sensitivity (không đổi gate): cap từng class ratio ở1 → 0.354024/0.361357; bỏ class chi phối → 0.310959/0.318781.
Các giá trị này vẫn>0,20, nên PASS không chỉ phụ thuộc magnitude outlier. Tuy nhiên probe không chứng minh
update hữu ích hoặc bác mọi dạng calibration artifact. Không tự đổi sample/basis/rcond/gate hoặc mở utility.

Wire thực:20×928B up,20×746B down, bootstrap18880B mỗi seed; trước channel overhead utility chưa triển khai.
Probe elapsed70,32/65,97s gồm setup; không dùng timing này làm training cost≤2×CE proof.

[Gates và nguyên nhân](analysis.json) · [Protocol/nguồn/provenance](../README.md).
Audit exact executed/source/input/CE-r2 weights; independent wire/scalar reconstruction; diagnostic tái hiện original energies
trên T4. Không independent local full-gradient inference; không có classification metrics mới vì model không train/evaluate holdout.
