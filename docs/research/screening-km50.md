# KM16 w0 .5 int8 full-scale r50 — luật khóa 08/11/2026 + kết quả

Cấu hình chọn **chỉ trên simulator fold val** ([screening-sim-foldval](screening-sim-foldval.md) b5–b8), không đọc official test:
KM16, co covariance cụm w0 0,5D (+.0070/+.0069 sim hai seed), codec upload int8 theo xe (+.0067/+.0065 sim hai seed,
0,88 MB lên + 1,74 MB xuống mỗi xe, một lần). Mọi hp khác = Core F2/F3 (probes off).

Run: `scripts/run_km50.py`, slug `mh-k-km16q8-s{42,43}-r50`, cosine50, `--until 50`, save mỗi 5 round, cap 11,5 h/phiên;
hết budget → nối cùng owner từ state đã lưu (không restart r1).

## Luật r50 (official own-test, vs Core F2/F3 r50 cặp seed)

PASS khi **cả hai seed** Δ f1_macro ≥ +.003 và từng seed Δp10 ≥ −.006, Δlow256 ≥ −.003, ΔtimeDelay ≥ −.0035, Δbenign ≥ −.0005,
Δabsent ≥ −.01 tại endpoint r50 (không chọn round). Cùng ngưỡng utility50/pilot20 đã khóa. Báo thêm (không gate): Δabsent r41–r50,
byte, thời gian bootstrap. Hai seed và test đã dùng nhiều lần → không phải xác nhận độc lập/significance.

## Kết quả r50 (`catbaochau/mh-k-km16q8-s{42,43}-r50/1`, `runs/km50/verdict-r50.json`, `scripts/analyse_km50.py`)

| seed | accuracy | P macro | P micro | P weighted | R macro | R micro | R weighted | **F1 macro** | F1 micro | F1 weighted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | .9112 | .9066 | .9112 | .9075 | .9098 | .9112 | .9112 | **.9077** | .9112 | .9090 |
| 43 | .9117 | .9079 | .9117 | .9081 | .9101 | .9117 | .9117 | **.9085** | .9117 | .9096 |

| seed | Δmacro | Δp10 | Δlow256 | ΔtimeDelay | Δbenign | Δabsent | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| 42 | +.0085 | +.020 | +.019 | +.007 | **−.0015** | +.038 | FAIL (benign) |
| 43 | +.0086 | +.023 | +.015 | +.012 | **−.0014** | +.033 | FAIL (benign) |

**Verdict theo luật: FAIL** (chỉ guard benign); Core vẫn là phương pháp chính. KM16 w0 .5 vẫn là nền nghiên cứu (mọi guard khác tăng
ở cả hai seed; xếp sau APFed .9119, ≈ CAN-PerFL .9088). Byte bootstrap gốc ≈3,67 MB/xe/50r → codec lite6 ở [screening-km-lite](screening-km-lite.md).
Bản KM16 fp32 w0 2D (pilot cũ) r50 cũng FAIL benign −.0018/−.0016: [archive](../../archive/docs-20261109/README.md).
