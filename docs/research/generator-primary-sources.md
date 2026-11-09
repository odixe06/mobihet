# Generator từ thống kê cộng — nguồn và giới hạn

Nguồn đối chiếu ngày06/11/2026; tài liệu này không là bảng trạng thái triển khai. Xem [catalog](generators.md) và [runtime](../../archive/docs-20261109/research/results-generator-runtime.md). Giữ77feature scaled,16lớp, protocol/LA/Q/floor.
[Oracle r50](results-y50.md) tăng macro .899160 → .917581; chỉ chứng minh dư địa generator, không chứng minh ứng viên nào thắng.
Server chỉ dùng tổng toàn mạng; không raw row, model hoặc gradient.

## Nguồn chính đã đối chiếu

1. **Fed-TDA, Duan et al., 2023.** Mỗi cột liên tục học VB-GMM; cột rời rạc dùng frequency + inverse-CDF;
   sau đó lấy covariance của representation biến đổi, sinh Gaussian rồi decode. Appendix A trao thống kê responsibility,
   weighted sums và covariance qua nhiều bước; §Communication ghi VB-GMM chạy 100 vòng trong thí nghiệm.
   Chiều latent `l=2*n_cont+n_disc`; chi phí covariance bậc hai theo l, chưa nhân 16 lớp như Core.
   Theorem 1 chứng minh covariance của Gaussian latent sau Cholesky, không chứng minh joint distribution bất kỳ được khôi phục.
   Đây là prior art gần Core về synthesis từ marginal + covariance; không phải bằng chứng sửa dependence phi Gaussian.
   [Toàn văn, §Fed-TDA và Appendices A/C/D](https://arxiv.org/html/2211.13116).

2. **Private-PGM, McKenna/Sheldon/Miklau, ICML 2019.** §3.1 Definition 3 biểu diễn distribution bằng tổng clique potentials
   trong exponential; Theorem 1 chọn maximum-entropy distribution phù hợp marginals. §3.2 học potentials bằng marginal inference
   và mirror descent. Kết quả nói về fitting thống kê đã đo; marginal không đo có thể khác dữ liệu thật.
   Cần oracle inference khả thi; sparsity của measured pairs chưa bảo đảm inference rẻ nếu prior ban đầu đã dense.
   [Bài gốc, §3](https://arxiv.org/html/1901.09136).

3. **MST, McKenna/Miklau/Sheldon, 2021.** Quy trình select → measure → fit với Private-PGM; §6 Algorithm 6 tính discrepancy
   cho **mọi cặp** bằng true 2-way marginals rồi mới chọn `d−1` cạnh. Code chính thức `select` dòng 85–89 thực hiện điều này.
   Vì vậy `d−1` cạnh output không có nghĩa chỉ cần upload `d−1` marginals khi học cấu trúc từ dữ liệu phân tán.
   [Bài gốc, §4/§6](https://arxiv.org/html/2108.04978),
   [code tác giả](https://github.com/ryan112358/mbi/blob/master/mechanisms/mst.py).

Đối chiếu Chow–Liu gốc: tối ưu approximation trong họ tree bằng second-order distributions; không là proof tree giữ joint thật
ngoài họ đó. Có thể bỏ dependence đã có trong dense Gaussian copula nếu thay toàn bộ bằng cây.
[Chow/Liu 1968, trang IBM của tác giả](https://research.ibm.com/publications/approximating-discrete-probability-distributions-with-dependence-trees).

## Suy luận ứng dụng: giữ copula, hiệu chỉnh ít marginal

Đây là **suy luận thiết kế**, không phải thuật toán/utility đã được những bài trên kiểm chứng cho MobiHET.
Cho `q_c(x)` là generator hiện tại, `b_i(x_i)` là bin công khai/chung, `E` là danh sách cặp khóa trước:

```
p_c,λ(x) ∝ q_c(x) exp(Σ_(i,j)∈E λ_c,ij[b_i(x_i), b_j(x_j)])
```

Đây là dạng nghiệm Lagrange của `min KL(p || q_c)` dưới ràng buộc marginal khả thi; khác maximum entropy với base uniform.
Mỗi histogram `H_cijab = Σ_rows 1[y=c, b_i=a, b_j=b]` cộng được qua client; server fit λ chỉ từ tổng H và mẫu tự sinh.
Mỗi bước IPF chính xác nhân density bởi `target_pair/current_pair`; các bước sau có thể đổi lại marginal trước nên cần hội tụ.
Private-PGM nêu trên dùng mirror descent, **không tuyên bố code đó thực hiện IPF với dense-copula prior**.

Có thể fit/sampling xấp xỉ qua importance weights trên candidate mới từ q; phải đo ESS, max weight, bin coverage và sai số marginal
trên candidate độc lập. Candidate hữu hạn mất coverage; reweight không tạo được support mà q không sinh được, không bảo đảm hội tụ
IPF chính xác hoặc giữ univariate marginals. Muốn giữ marginal cũ cần thêm ràng buộc singleton từ histogram sẵn có.
Khóa luật clipping/shrinkage trước utility; clipping thay mục tiêu. Giữ streaming, không biến thành bank cố định đã thất bại F50.

## Dung lượng: tự tính, không phải số đo wire

`C=16`, `d=77`, 2.926 cặp; upload uint32 counts, download fp32 potential/table: `8*C*|E|*B²` byte hai chiều.
Chưa gồm header, bin edges, singleton support, selection/sketch, quantization hoặc secure-aggregation transport.

| thống kê mới | byte/xe |
|---|---:|
| mọi cặp, 8 bin: **uplink riêng** `4*16*2926*64` | 11.984.896 |
| mọi cặp, 16 bin: **uplink riêng** | 47.939.584 |
| cây định trước 76 cạnh, 8 bin, hai chiều | 622.592 |
| **12 cặp định trước, 16 bin, hai chiều** | **393.216** |
| 12 cặp, 16 bin: uint32 up + fp16 down | 294.912 |

[L50](results-l50.md) bootstrap 637.066 B/xe. Thêm 12 cặp fp32 → **1.030.282 B bootstrap**, hoặc **1.084.682 B/50 round** khi giữ Q;
fp16 down → 931.978 B bootstrap, 986.378 B/50 round. Chưa header/codec gate; fp16 potential cần kiểm động học weights/ESS,
không lấy argmax hoặc marginal gần nhau thay utility. Muốn dưới 1.000.000 B bootstrap ở fp32 có thể khóa ≤11 cặp16bin
(997.514 B trước header); muốn quanh 0,7 MB cần budget ít hơn hoặc codec đã đo, không tự gọi “cùng byte”.

**Chọn cặp không miễn phí.** Danh sách theo domain công khai: không selection upload. Nếu dùng train để chọn:
mọi residual/sketch được query phải cộng vào budget; statistic nonlinear như MI local không cộng thành MI pooled.
Có thể shortlist từ covariance sẵn có, nhưng correlation không phát hiện mọi dependence phi Gaussian; kiểm chứng giả thuyết bằng
chính marginals đã khóa. Giữ feature/bin mapping chung và dense fixed-shape counts nếu cần tương thích secure aggregation.
L50 dùng sparse uplink: số byte 637.066 B không phải proof SecAgg triển khai được hoặc chi phí SecAgg đầy đủ.

## Các hướng còn lại

| hướng | tương thích contract, giới hạn |
|---|---|
| Sparse residual calibration | Khả thi từ additive H; giữ dense copula base. Chỉ theo tiếp khi train-only residual cụ thể vượt sampling error, ESS/coverage dùng được; chưa utility. |
| Chow–Liu replacement | Additive pair tables khả thi khi cạnh khóa trước; học all-pair tree đắt. Mất cấu trúc ngoài cây, không ưu tiên thay dense copula. |
| Conditional copula | Histogram/moments theo condition khóa trước cộng được; K condition nhân khoảng K lần chi phí phần bị chia. Scenario8 mixture đã FAIL; đổi tên condition không thành cơ chế mới. |
| t-copula | Dùng shared radial scale để tạo tail dependence, vẫn họ elliptical. Cần dấu hiệu tail co-occurrence, không chỉ marginal nặng đuôi; không sửa dependence bất kỳ. |
| Flow học federated | Gradient/weight exchange trái contract. Flow fit server chỉ trên H/q khả thi về truyền thông nhưng không có thêm joint information ngoài H/q; flexible density không khôi phục quan hệ không được đo. |

t-copula dùng `t_ν` CDF, không dùng `Φ` lên latent t rồi coi marginal còn đúng; normal-score covariance hiện có không tự trở thành
ước lượng chính xác t-copula. Kendall tau pooled cần so cặp quan sát xuyên client, không bằng trung bình tau local.
Có thể fit likelihood qua scalar sums cho tham số đã broadcast, nhưng mỗi query phải tính cả downlink/contact.
[Demarta/McNeil, §§2–4, bản thảo tác giả](https://www.planchet.net/EXT/ISFA/1226.nsf/0/303eb11b4d617b79c1257b0800744575/$FILE/t%20copula%20demarta%20mcneil.pdf).

Nguồn xác lập tính biểu diễn/fitting và giới hạn bandwidth; chưa xác lập mức tăng F1, fairness hay khả năng tái đạt oracle.
