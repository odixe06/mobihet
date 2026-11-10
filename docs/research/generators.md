# Generator — catalog quyết định

**Generator đóng băng = KM16 w0 .5** (owner 09/11); **10/11 owner chốt CBB = KM16 + codec lite6 + QQ8 làm phương pháp chính** ([proposal](proposal.md)). Không mở biến thể generator mới (kể cả đổi tên);
cải tiến tiếp theo ở [catalog đề xuất](improvement-proposals.md). Job đang chạy chỉ ở [continue](../operations/continue.md).
Hồ sơ chi tiết các ứng viên đã đóng: [archive 07/11](../../archive/docs-20261107/README.md), [archive 09/11](../../archive/docs-20261109/README.md).

| ứng viên | kết quả cuối | trạng thái |
|---|---|---|
| Core copula (Gaussian copula theo lớp) | F1 r50 .8991 | đối chứng (CBB xây trên Core) |
| Oracle (pool dòng thật) | r50 s42 +.018 macro so Core, ngoài hợp đồng FL | chỉ trần chẩn đoán, [Y50](results-y50.md) |
| Pair IPF/reweight · Forest+Gaussian conditional · Copula+marginal transport | local/pair gate FAIL | đóng |
| Forest+quantile · Standard latent | runtime PASS, **pilot20 utility FAIL hai seed** | đóng |
| KM16 w0 2D fp32 | full-scale r20 +.0075/+.0069 (FAIL absent s42); r50 +.0087/+.0089 FAIL benign −.0018/−.0016 | thay bằng bản dưới |
| **KM16 w0 .5 int8** | full-scale r50 Δmacro **+.0085/+.0086** so Core, FAIL guard benign −.0015/−.0014 ([km50](screening-km50.md)) | **trong phương pháp chính CBB** (owner chấp nhận benign) |
| KM codec `lite6` | pilot r20 s42 FAIL (09/11, trong nhiễu) → kiểm lại LT6S43 r20 PASS; combo **CBB (lite6+QQ8) r50 PASS cả hai seed** (bootstrap 1,15 MB) | **trong CBB** — [km-lite](screening-km-lite.md), [codec](screening-codec.md) |
| K32, pop, part 0,1, EM1/2, whitening, benign share, temp, gap-fill, Q synthetic, KM8, K24, within/diag/top-R cov | sim FAIL | đóng ([sim](screening-sim-foldval.md), [km-lite](screening-km-lite.md)) |

## Cách đọc kết quả generator (vẫn áp dụng)

- Histogram biên + covariance không định danh joint distribution; expressive hơn không tự tăng F1.
- PairTV, TSTR, probe LL, teacher agreement, runtime và utility FL là các cổng khác nhau; chỉ utility FL quyết định.
- Panel fold val đã dùng nhiều lần, không là fresh confirmation. Official test chỉ để báo endpoint, không chọn HP/round.
- Code generator cũ giữ để tái lập: `methods/mobihet_{copula_standard,forest_quantile}.py`, probe `scripts/probe_*_generator.py`;
  KM ở `methods/mobihet_sim.py` (`cop_kmeans*`).
