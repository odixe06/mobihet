# Pair-generator feasibility — FAIL, 06/11/2026

[Plan khóa](screening-pair-generator.md). Job `pair-generator-feasibility-20261106` exit0, marker có; artifact `runs/pair-generator/result.json` kiểm thực.
CPU bounded train→val: 245,760train, 120,515val, đủ16class; 60file, nguồnchỉtrain/. Không officialtest, không productionCUDA/FLproof.

- PairTVmacro val: 0.199427→0.751434, xấu đi; gate giảm20%FAIL.
- ESS trên16384candidateđộc lập/class: 1.000–7.588mẫu, gate≥819,2FAIL.
- SingletonTV guardFAIL. TSTR bỏqua đúngplan; không launchFL.
- Thời gian18.67s, guardedpeakRSS705MiB. Unit2PASS cho dependence/singleton và unsupportedmass.

Giải thích có giới hạn: potential fit trênfinitecandidate32768/class với77singleton+12pair có thể tạo densityratio rất nhọn, không generalize sangcandidate độc lập. Chưa lưu ESSfit để tách overfit/collapsefit; chưa isolate damping/clipping/support/dimensionality. Không chứng minh targetIPF lý thuyết thấtbại, Gaussian không fullsupport hoặc mọireweight vôích. Không nới gate hoặc quét HP để tìmwin.

Bước mở: [sinh trực tiếp theo conditionalhist rồi Gaussian conditional cho phầncòn lại](generator-direct-conditional.md). Chỉ đềxuất sau evidence mới, chưa implement hoặc launch.
