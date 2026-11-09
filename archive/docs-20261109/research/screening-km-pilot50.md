# KM16 full-scale pilot → r50 — luật khóa 08/11/2026 (trước khi nối)

Nguồn: pilot20 [screening-sim-foldval](../../../docs/research/screening-sim-foldval.md) §b3, kết quả r20 trong [results-km-pilot](results-km-pilot.md).
Pilot K42/K43 (`takids1909/mh-k-km16-s{42,43}-p20/1`) đã khóa lịch cosine50 (`--rounds 50 --until 20`), lưu state r20 hai rank.

## Nối

`scripts/resume_km_pilot50.py`: kernel `takids1909/mh-k-km16-s{42,43}-p20-c2/1`, notebook = đúng bytes pilot + ô import state r20
đã kiểm (`resume_checkpoint_cells.kernel_import`: round 20, method, seed, rounds 50, RNG hai rank), `--until 50`, cap 8 h. Không đổi
method/hp/overlay (KM16 w0 mặc định 2, codec full fp32 — **không** phải cấu hình sim tốt nhất w0 .5; đổi giữa run là không hợp lệ).

## Luật r50 (official own-test, vs Core F2/F3 r50, từng seed cặp)

Giống utility50 / pilot20 đã khóa: PASS khi **cả hai seed** Δ f1_macro ≥ +.003 và từng seed Δp10 ≥ −.006, Δlow256 ≥ −.003,
ΔtimeDelay ≥ −.0035, Δbenign ≥ −.0005, Δabsent ≥ −.01. Chỉ endpoint r50, không chọn best round.
Pilot20 FAIL (absent seed42 −.028) giữ nguyên, không được thay bằng kết quả r50.

Báo thêm (không phải gate): absent theo round r41–r50 của cả hai arm; guard absent chỉ 1.555 dòng (365 dòng ở lớp 14),
Core tự dao động .617→.662 giữa r19 và r20 seed42. Byte bootstrap thêm 3,47 MB lên / 1,74 MB xuống mỗi xe (một lần).
