# Core — cơ chế đang thực thi

Chọn phương pháp: [proposal](proposal.md). Mọi cấu hình chung theo [protocol](protocol.md).
Implementation: [WLive](../../methods/mobihet_wstudy.py), [copula/bootstrap/Q](../../methods/mobihet.py),
[live teacher](../../methods/mobihet_next.py), [cấu hình Core](../../scripts/f50_common.py).
Các đối chứng M→X cũ không còn là mô tả mặc định.

## Bootstrap một lần

1. Mỗi xe đếm lớp và histogram 256 bin cho 77 feature trên miền scaled chung. Server chỉ dùng tổng để có prior/CDF toàn mạng theo lớp.
2. Xe dùng CDF chung biến đổi train local thành normal score theo mid-rank, gửi tổng `Σz`, `Σzzᵀ` theo lớp.
3. Server dựng mean/covariance có ridge; downlink CDF uint16 và mean/Cholesky fp16. Mỗi xe decode rồi tự sinh synthetic.

Đây là **Gaussian copula theo lớp**, không Gaussian trực tiếp trên mọi feature: marginal có histogram phi Gaussian,
nhưng dependence latent vẫn là một Gaussian mỗi lớp. Core hiện không tự chuẩn hóa mean0/variance1 sau decode;
đó là can thiệp riêng của ứng viên [standard latent](generators.md).

## Một round tại xe

- Dữ liệu thật: đúng một epoch, CE với logit adjustment từ prior local/global.
- Mỗi bước: thêm 64 dòng synthetic cân bằng 16 lớp. Round1 dùng nhãn cứng; các round sau dùng KL từ `Q[y]`, trọng số cố định1.
  Synthetic dùng logit thô; CEreal dùng logit đã adjustment theo thiết kế.
- Xe có ít hơn16 bước thật: thêm bước **chỉ synthetic** đến floor16; không thêm epoch thật.
- Online uplink: tổng xác suất model trên dòng thật theo true class và count. Server gộp theo số mẫu/lớp,
  encode/decode meanfp16 để có bảng `Q` chung. Q thu ở round r chỉ dùng từ r+1.
- Không realKD, input-bank, proxy server hoặc trao model trong Core. Mỗi xe giữ model riêng.

`Q[y]` là target có điều kiện theo nhãn thật, không classifier phụ thuộc input. Q đúng argmax/ít entropy hoặc synthetic agreement
không chứng minh teacher tổng quát hóa trên own-test.

## Byte, resume và đánh giá

Class Q: 544B mỗi chiều/xe/round. Tổng chi phí và codec theo [proposal](proposal.md); số đo ở kết quả từng batch.
Chỉ gọi SecAgg-compatible ở mức công thức tổng, không gọi sparse codec hoặc hệ thống hiện tại đã có SecAgg/DP.

Hai process/GPU giữ client riêng, không đồng bộ gradient. Checkpoint resume đầy đủ khác snapshot nhỏ dùng audit:
phải có private state, teacher, RNG của method/driver và torch CPU/CUDA của cả hai rank cùng round.
Optimizer local được khởi tạo lại mỗi round theo protocol. Không dùng W&B resume thay checkpoint resume.

Mỗi xe chấm own-test của nó; gộp đủ 5.343 CM và hỗ trợ từng xe/lớp để báo10metric, cohort và class results.
Headline r50, không chọn round/HP qua test. Audit hiện gồm source/config/CM/support/snapshot/teacher/finite;
chưa prediction→CM replay toàn mạng hoặc full-content fingerprint của dataset.

## Nguồn và phạm vi adaptation

- Class output exchange có quan hệ với [FD, Jeong et al.](https://arxiv.org/abs/1811.11479),
  soft target với [KD, Hinton et al.](https://arxiv.org/abs/1503.02531).
  Core dùng teacher chung gồm self, weighted counts, synthetic và LA; không phải reproduce đầy đủ bài gốc.
- Prior art synthesis và hạn chế marginal/covariance: [nguồn primary](generator-primary-sources.md).
  Không gán copula/floor/codec của dự án cho công thức paper khi không có đối chiếu.
