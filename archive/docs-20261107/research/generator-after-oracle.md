# Generator sau oracle Y50 — phân tích và đề xuất, 06/11/2026

Trạng thái: phân tích, chưa triển khai generator/launch mới. Giữ không trao dòng thật/model/gradient, chỉ thống kê gộp. Nguồn: [Y50](../../../docs/research/results-y50.md), [pipeline](pipeline-review.md), [LESSONS](../../../LESSONS.md), code `methods/mobihet.py::_copula_bootstrap/_synth`.

## Bằng chứng và giới hạn

Core hiện là Gaussian copula theo lớp: histogram biên256bin + mean/covariance sau normal-score transform, không phải Gaussian trực tiếp trên77feature. Histogram biên cho phép phân bố một chiều không Gaussian, nhưng dependence sau transform chỉ có một Gaussian mỗi lớp. Không có proof tối ưu.

Oracle r50 macro +.018422, p10 +.030453, low256 +.026738, absent +.046302. Cùng protocol nhưng nguồn oracle là pool dòng thật theo rank; thay generator cũng đổi sampling và teacher trajectory. Không tách được phần gain do từng thống kê/constraint riêng.

| lớp | Core F1 | Oracle F1 | Δ |
|---|---:|---:|---:|
| 1 | 0.879524 | 0.946952 | +0.067427 |
| 7 | 0.852538 | 0.914070 | +0.061532 |
| 11 | 0.830358 | 0.873895 | +0.043537 |
| 13 | 0.342007 | 0.376335 | +0.034327 |
| 0 | 0.840742 | 0.835773 | -0.004970 |

Ba lớp1/7/11 đóng58,5% tổng delta F1 các lớp. Precision lớp1 .835→.926, lớp7 .813→.892; timeDelay precision .415→.374 dù recall .291→.378. Benign F1 giảm .004970. Không suy fidelity tăng thì mọi metric/client đều tăng.

Tái tạo: `python scripts/analyse_generator_oracle.py`; artifact `runs/generator-study/oracle-gap.json`. Chỉ đọc CM r50 đã audit, không chọn HP.

## Chỗ thống kê có thể làm tốt hơn

- Mean/covariance không định danh joint distribution. Ví dụ hai feature U và V=U² với U đối xứng: correlation có thể0 dù phụ thuộc hoàn toàn; hai marginal không phục hồi được quan hệ đó.
- Histogram hai chiều theo lớp giữ nonlinear dependence và vùng support mà covariance không mô tả. Bảng của từng xe cộng được thành bảng toàn mạng; không cần row/prototype/model upload.
- Mass rời rạc, flags, thiếu dữ liệu và quan hệ cửa sổ tạo mixture/support constraints. Core nội suy trong bin; flags có thể fractional. Projection từng bị thử mà không tăng utility ổn định: không lặp đơn lẻ hoặc gọi đó là lời giải.
- Conditional statistics có thể tách multimode nhưng tăng byte và sparse cells; scenario8/mix từng không chứng minh utility. Không mở lại bằng tên mới.
- Sinh nhiều hơn hoặc tăng synthetic weight không sửa sai joint; có thể khuếch đại bias và hại benign. Giữ64rows/step, Q/LA/floor khi isolate generator.

## Ưu tiên: copula + hiệu chỉnh marginal hai chiều

**Problem → mechanism → falsifiable sign:** oracle còn gap, nhất là1/7/11; singleGaussian dependence thiếu khả năng mô tả nonlinear pair structure → thêm count bảng2D ở một số cặp, fit residual trên nền copula → synthetic khớp các bảng2D trên train và trên val theo thời gian; downstream r50 cải thiện mà không mất tail/benign. Hiện là hypothesis, chưa đo mismatch2D hoặc utility của nó.

1. Trên fold train/val nội bộ, đo discrepancy copula-vs-real cho cặp feature công bố trước, kèm train-vs-val discrepancy làm tham chiếu. Không chọn cặp/HP bằng official test. Scaler theo từng cột: không áp quan hệ vật lý trực tiếp lên giá trị đãscale; cần inverse transform nếu kiểm quan hệ vật lý.
2. Khóa danh sách nhỏ, ví dụ12cặp; ví dụ cần kiểm gồm msg_acl/msg_abs_acl, msg_spd/rel_spd, ses_dt/ses_dt_send, ses_dt/ses_dt_skew, mean/max cửa sổ tương ứng. Đây là candidate list theo cơ chế, chưa chứng minh liên quan đúng từng lớp. Ưu tiên cặp sai joint rõ và thông tin phân lớp trong train/val, không all-pairs miễn phí.
3. Mỗi xe đếm H[c,e,a,b] theo16bin định nghĩa chung; flags dùng2trạng thái khi liên quan. Gửi bảng count dày, chỉ tổng toàn mạng; server không cần bảng riêng xe. Tiếp tục báo chưa triển khai secure aggregation/DP.
4. Giữ copula q0, fit p(x|c) ∝ q0(x|c) exp(Σ_e λ[c,e,bin_i,bin_j]) bằng matching marginal/IPF trên mẫu synthetic server. Thêm ràng buộc singleton sẵn có để không làm hỏng marginal1D; update phải dùng marginal của phân bố hiện tại, không nhân một lần các ratio chồng lấn rồi tuyên bố khớp tất cả bảng. λ là potential của distribution, không neural client weights; payload mới phải ghi rõ là bảng thống kê phân bố suy ra từ tổng, không chứa tham số model client. Có thể truyền targethist để xe tự tính, nhưng sẽ lặp chi phí fit.
5. Xe sinh candidate mới từ q0, weighted-resampling theo potential rồi dùng đúng64row/step. Đây là sampling approximation, không exact draw từ p. Không giữ finite input-bank cũ. Khóa/gate candidate count và ESS trên train/val: ESS thấp, duplicate cao hoặc q0 không cover vùng support thì bác bỏ hiệu chỉnh kiểu này, không tăng weight vô hạn. Đo T4 throughput trước chấp nhận.

Bảng12cặp×16lớp×16×16=49.152entries; up/down32bit =393.216B/xe một lần, chưa header/selection/fitmetadata/SecAgg. Cộng Core-L50 hiện691.466B thành1.084.682B/xe/50round (+56,9%). Downfp16 có thể giảm nhưng cần codec/zero-cell gate riêng; chưa cộng savings vào con số chính. UInt64 cho secure aggregation tăng budget. Selection all77choose2 sẽ tốn nhiều hơn; không ẩn cost đó.

## Cổng trước khi gọi là đủ tốt

- Feasibility local: pooled counts bằng tổng count xe; encode/decode exact hoặc tolerance nêu rõ; generator không đọc dữ liệu xe khác; state/RNG/resume2rank; finite/probability/support; không sửa scientific code run đang chạy.
- Train→val: pair discrepancy ngoài sample noise giảm; discrepancy univariate không xấu, heldout classifier trained-on-synthetic/test-on-real tốt hơn control cùng dữ liệu/seed; đây là pooled diagnostic, không thay personalized FL. Đo train-fold subset mới vì val từng được nghiên cứu trước, không gọi freshconfirmatory.
- Nếu qua, mới khóa batch FL control-vs-generator,2seed,cosine50,1epochreal,64synthetic/step, cùng Q/floor theo trạng thái đã khóa; không trộn thêm đổi loss/target/prior. Chạy đối chứng cùng harness nếu sampler đổi execution/RNG.
- Ngưỡng utility đề xuất, chưa launch/lock: macro tăng≥.003 ở cảhai seed (thu≥16,3% gaporacle seed42; không phải dự báo), p10 không giảm quá.006, low256 quá.003, timeDelay quá.0035, benign quá.0005, absent quá.01; báo cả10metric, class precision/recall, byte/timing. Sau đọc Y50 ablation mới chốt baseline; không chọn best round.
- Nếu joint fidelity tăng mà FL không tăng hoặc hại tail/benign: bác bỏ utility của correction ở budget này. Nếu pair residual nhỏ mà oracle gap vẫn lớn, chuyển chẩn đoán higher-order/conditional structure, không mặc định tăng sốpair/mixture.

## Kết luận phạm vi

Có thể nâng khả năng biểu diễn của thống kê mà vẫn dùng tổng cộng được; chưa biết loại thống kê nào đủ utility trên v4. Ưu tiên thử mismatch cụ thể với budget rõ trước flow/diffusion hoặc full mixture. Flow/diffusion federated thường thêm trao parameters/gradient, cần đổi quyền hiện tại. Không lấy oracle làm bảo đảm gain hoặc proof tối ưu.

Nguồn primary và giới hạn selection/inference: [generator-primary-sources](../../../docs/research/generator-primary-sources.md). Private-PGM hỗ trợ marginal fitting, không chứng minh phép hiệu chỉnh copula này sẽ tăng utility.

Owner đã duyệt thử: [cổng feasibility khóa](screening-pair-generator.md), chạy06/11. Chưa có utilityFL.

Probe đãFAIL: [kết quả](results-pair-generator.md). Hướngmới giải quyếtfinitecandidatecollapse: [sinh trực tiếp có điều kiện](generator-direct-conditional.md), chưalaunch.
