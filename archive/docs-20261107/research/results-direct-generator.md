# Direct forest + Gaussian conditional — FAIL, 06/11

[Plan](screening-direct-generator.md); `runs/direct-generator/result.json`. State job succeeded/exit0, final marker; đủ16class/245760train/120515val, sampleSHA `2063b45b2dfabeb8c7991ac772b38e97f7047c3cff2db2c7a5378397e8cce826`. CPU approximation, khôngofficialtest hoặc productionFL.

- PairTV 0.199355→0.066747, giảm66,5%, qua20%gate.
- SingletonFAIL: class12 tăng+.0764, class15+.0374 (>+.02); TSTR khôngchạy theo plan.
- OutsidepairTV class12 +.1102,class15+.0650; regression không chỉmarginal.
- Runtime8.06s; peakRSS812MiB.

Chẩnđoán deterministicsamepanel/seed: top10singletonTVincrease cảclass12/15 đềuởR (ngoài19selected). Class12 `p4_std_ses_motion_hed_err`+.5324,`win_spd_resid_cum`+.5085,`win_hed_err_mean5`+.4736; class15 `p4_std_ses_acl_resid`+.2087,`map_opp_dist`+.2043. Cùngmodel.sample seed8100+c, evaluatebinnedTV16 trênval. Đây làdiagnosticposthoc, chưatách hếtindependentroot/coarseconditioning.

Cơchếphùhợp evidence: thayphânphốixS làmphânphốimarginalxR từGaussianconditional lệch; dùedgesS khớp. Khôngkếtluận mọiconditionalgeneratorFAIL. Giữtrial nguyênbytes vàgate.

## Hypothesis tiếp

[Marginal transport](screening-marginal-direct-generator.md): giữedgesS vàGaussianconditional, sửa riêng R bằng `F_R,train^-1(F_R,generated(xR))`. F_generated ước từ32768mẫusynthetic độc lập/class, khôngval/rawrows khácxe, không thốngkê uplinkmới. Monotone map giữ rankdependence (trừties/quantization), không bảođảm giữphysicalrelationships hoặcutility. Expected: singletonguardPASS, edgesS giữimprovement; bácbỏnếu TSTRkhôngtănghoặcoutsidepairs vẫnxấu.
