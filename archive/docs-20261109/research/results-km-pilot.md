# KM-copula K16 — pilot full-scale r20 (08/11/2026)

Run `takids1909/mh-k-km16-s{42,43}-p20/1` (`scripts/run_km_pilot20.py`), 5343 xe, official own-test, r20 của lịch cosine50,
KM16 w0 2D (cấu hình b3), mọi hp khác = Core F50. Control Core F2/F3 r20 cặp seed. Cả hai đủ 20/20 round, ≈700 s/round.
Gate (khóa trước, = generator pilot20): `scripts/analyse_km_pilot20.py` → `runs/km-pilot20/verdict-r20.json`.

## Mười metric r20

| run | accuracy | P macro | P micro | P weighted | R macro | R micro | R weighted | F1 macro | F1 micro | F1 weighted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Core F2 s42 | .9117 | .8984 | .9117 | .9033 | .8944 | .9117 | .9117 | .8933 | .9117 | .9047 |
| **KM16 s42** | .9154 | .9056 | .9154 | .9072 | .9013 | .9154 | .9154 | **.9008** | .9154 | .9089 |
| Core F3 s43 | .9119 | .8998 | .9119 | .9035 | .8947 | .9119 | .9119 | .8943 | .9119 | .9051 |
| **KM16 s43** | .9155 | .9062 | .9155 | .9073 | .9015 | .9155 | .9155 | **.9012** | .9155 | .9090 |

## Gate

| seed | Δmacro | Δp10 | Δlow256 | ΔtimeDelay | Δbenign | Δabsent | Δclient mean | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | **+.0075** | +.016 | +.013 | +.005 | +.003 | **−.028** | +.011 | FAIL (absent) |
| 43 | **+.0069** | +.017 | +.011 | +.002 | +.004 | +.015 | +.010 | PASS |

**Verdict theo luật: FAIL** (seed42 absent −.0283 < −.01). Không đổi luật sau khi xem số.

Đọc:
- Δmacro full-scale (+.0075/+.0069) lớn hơn sim (+.0057) và ≈40% trần oracle full-scale (+.018 r50). Mọi guard khác tăng ở cả hai seed.
- Absent (accuracy dòng lớp vắng ở 848 xe <256 dòng) chỉ 1.555 dòng, 365 dòng là lớp 14; seed42 thiếu 72 dòng đúng ở lớp 14, lớp khác ±15.
  Δabsent seed42 theo round: r15 +.003, r16 +.004, r17 +.011, r18 +.019, r19 +.055, r20 −.028; Core tự dao động .617 → .662 (r19→r20).
  Guard này bị nhiễu round chi phối, nhưng luật là luật: pilot20 FAIL.
- Byte: thêm 3,47 MB lên / 1,74 MB xuống mỗi xe một lần (Core cả 20 round ≈1,05 MB/xe; FedAvg 159,5 MB).

Tiếp: nối cùng run tới r50 theo lịch đã khóa ([screening-km-pilot50](screening-km-pilot50.md)), codec gọn trong sim b7/b8.
