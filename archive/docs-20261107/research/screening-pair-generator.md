# Pair generator — feasibility trước FL, owner duyệt 06/11/2026

Owner: “hãy thử nghiệm đi”, sau [phân tích](generator-after-oracle.md). Cho phép thử generator và batch đối chứng/2seed nếu qua feasibility; không đổi protocol hoặc mở hướng neural generator trao model.

## Cổng khóa trước khi chạy

Stage1 local train→val, không đọc officialtest. Một file giữa danh sách đãsort mỗi scenario/world; sample ngẫu nhiênseed6006 tối đa4096train/2048val/file. Giữ77feature scaled. Cần≥256train/128val mỗi lớp hoặc dừng báo coverage thiếu, không tự đổi sample để tìm gain.
CPU copula approximation theo Core compact: histogram256bin, normal-score mean/cov population+1e-4I, mean/Choleskyfp16, CDFuint16. Không torch/CUDA bitexact; chưa productiongenerator hoặc fullnetworkproof.

Candidate12cặp cố định theo feature schema, không chọn bằng officialtest:

- `msg_acl` / `msg_abs_acl`
- `msg_spd` / `rel_spd`
- `msg_spd` / `rel_spd_diff`
- `rel_bearing_sin` / `rel_bearing_cos`
- `ses_dt` / `ses_dt_send`
- `ses_dt` / `ses_dt_skew`
- `ses_implied_spd` / `ses_spd_resid`
- `ses_spd_resid` / `win_spd_resid_mean5`
- `win_spd_resid_mean5` / `win_spd_resid_max5`
- `win_pos_pred_mean5` / `win_pos_pred_max5`
- `rel_dist` / `nbr_min_dist`
- `msg_spd` / `p4_std_msg_spd`

Fit potential trên32768candidate/class, evaluate16384candidate/class **độc lập** cùng bộ parametercopula; realval theo thời gian độc lập với train. Histogram2D16bin đều[-4,4], thêm77ràng buộc singleton cùngbin. IPF xấp xỉ20sweep, damping.5, ratio[.1,10], pseudocount1e-6, potential[-6,6]. Các giá trị là cấu hìnhfeasibility khóa trước số, không phải đã chứng minh tối ưu. Regularization và candidate hữu hạn không đảm bảo exact marginal fit.

Qua thống kê khi đồng thời:

- PairTV meanmacro qua16class trênval giảm≥20% so copula.
- Không lớp nào mean singletonTV tăng>.02.
- Mọi lớp ESS/candidate≥.05 trên candidate đánh giáđộc lập.

Thiếu coverage hoặc trượt cổng: không launchFL, ghi blocker/mismatch, không tự nới ngưỡng. Unsupportedmass từngpair vẫn báo, không projecttarget vào support để che lỗi.

Chỉ nếuqua3cổng, fit3HistGradientBoosting (real, copula, corrected) cùng max_iter40/max_leaf_nodes31/seed42, dự đoánval16class. Hai nguồnsynthetic cùng4096row/class (calibratedweightedresample cólặp; copulabaseline lấykhônglặp), đây là probe TSTR finitepool, không streamingFL hoặc samecomputeproof. TSTRmacro corrected phải≥copula+.003. Realclassifier là mốc tham khảo, không utilitygate thayFL. Không lựa chọn từ officialtest.

## Stage2 khi feasibilityPASS

Tiếp tục quyềnowner hiện tại: implement streaminggenerator cùng cơ chế, kiểm state/RNG/resume/2rank/generator finite và payload encode/decode; đo throughput/ESS T4 trướcFLdài. Không đưa finitepoolTSTR thành fixedinputbank training. Khóa control-vs-generator2seed, cosine50, Q/floor/LA/64synthetic giữCore đãkhóa; cache npy và Inductor/compile giữnhưY50. Không thay runY50 sống.
Ngưỡngr50 dự kiến từ đề xuất: macro≥+.003 cảhai seed; guard p10−.006,low256−.003,timeDelay−.0035,benign−.0005,absent−.01; khóa bản launch riêng trướcđọc endpoint. Không tuyênbốFLsẽqua từTSTRPASS.

## Bằng chứng và job

Code `scripts/probe_pair_generator.py`, tests `tests/test_pair_generator.py` 2PASS (recoverdependence/singletons và reportimpossiblesupport). Plan immutable `runs/pair-generator/plan.json`, result `runs/pair-generator/result.json`, log `runs/pair-generator/probe.log`.
Job`pair-generator-feasibility-20261106`, controller queue vềthread gốc; không modelpollwaiting. Completion là worker xong, đọc result/gates mới quyết định bước2.
