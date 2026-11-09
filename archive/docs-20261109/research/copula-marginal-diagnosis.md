# Copula — tách fitting error và temporal shift

Nguồn `runs/copula-marginal-diagnosis/result.json`:16class×77feature,65536synthetic/class, paneltrain→val đã dùng lặp.
Không official test hoặc FL trong probe này. [Trạng thái ứng viên](../../../docs/research/generators.md), [runtime đã audit](results-generator-runtime.md).

| macro TV16bin | giá trị |
|---|---:|
| train→val marginal | .0465053 |
| base synthetic→train | .0161293 |
| standard synthetic→train | .0031754 |
| base synthetic→val | .0506545 |
| standard synthetic→val | .0466045 |

Standard latent giảm train fitting error khoảng80,3%, nhưng valTV chỉ giảm khoảng8%. ValTV gần train→val reference
không chứng minh lower bound: một generator bias vẫn có thể vô tình gần val hơn train.
Giữ [local verdict FAIL](../../docs-20261107/research/results-standard-copula.md) của gate20%valTV;
không sửa gate sau số. TSTR cao chưa là personalized utility.

Can thiệp production: `methods/mobihet_copula_standard.py` chỉ postdecode mean0/normalize hàngCholesky;
Q/LA/floor và payload Core giữ nguyên. Đây là method change, không bit-exact optimization so với Core.
Local preflight hai seed và runtime2T4r2 đã PASS theo [bằng chứng riêng](results-generator-runtime.md);
không dùng runtime proof để đổi verdict probe hoặc tuyên bố utility50.
