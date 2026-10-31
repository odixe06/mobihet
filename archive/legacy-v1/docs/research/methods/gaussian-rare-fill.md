# A — Dồn mẫu Gaussian vào lớp hiếm local (water-filling)

**Trạng thái (17/10 ~22:40Z): FAIL** — `gpool_fill` s42 ep6 0,7162 (−0,0012 so gpool, benign +0,002, tệ nhất −0,001; proxy −0,004)
([kết quả](../../methods/full-drift/results.md)). Khớp probe 14: ~46 % khoảng private–proxy ở lớp đủ mẫu local, nên dồn mẫu vào lớp hiếm
không chạm nút thắt (tầng quyết định trên phân phối toàn cục). Owner duyệt 17/10 ~16Z (đề xuất A).

## Vấn đề (bằng chứng)
gpool s42 ep6: 56 % khoảng F1 private–proxy nằm ở lớp chiếm < 1 % dữ liệu local (recall private 0,57–0,60 vs proxy 0,73–0,76);
lớp ≥ 10 % recall bằng proxy, precision thấp hơn. gpool chia đều 16 mẫu/lớp bất kể client đã có bao nhiêu dòng thật.

## Cơ chế
Cùng 256 mẫu/bước, cùng trọng số CE 1; nhãn chia theo water-filling của prior local π_k: q = max(0, L − π_k), Σq = 1 (phân
bố nhãn real + Gaussian đều nhất có thể), làm tròn largest remainder. Với số đếm full: lớp lớn nhất local 0 mẫu, lớp hiếm
16 → 22–26 mẫu; 12/20 client không còn mẫu benign tổng hợp. 0 byte, không đổi kiến trúc.

## Dấu hiệu sai
Δ private gp < +0,010 so gpool; recall nhóm lớp < 1 % không tăng; benign giảm (mất mẫu benign gộp).

## Bằng chứng liên quan
- `gpool_gw2` (×2 liều mọi lớp, W&B chưa audit) −0,005 ⇒ thêm liều không giúp; A đổi phân bổ, giữ tổng liều.
- `replay` (dòng local cân bằng lớp) −0,033 so gpool ⇒ cân bằng lớp local không thay tri thức gộp.
- M1 oracle cũ: 32 mẫu thật/lớp từ client khác +0,07…+0,09 ([kế thừa](../../archive/legacy/kien-thuc-ke-thua.md)).
