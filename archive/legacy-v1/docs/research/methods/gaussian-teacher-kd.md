# P1 — Teacher gộp dạy private trên support Gaussian (KD trên mẫu `gpool`)

**Trạng thái (17/10 ~22:40Z): FAIL** — `gpool_p1` s42 ep6 0,7175 (+0,0001 so gpool; p1self +0,0014, p1cm −0,0002) ([kết quả](../../methods/full-drift/results.md)). Trước đó (17:10Z): độ phủ gate đo trên moment full = 4,3 % mẫu Gaussian (benign 0,207, dataReplay 0,488) ([kết quả probe](../../methods/full-drift/results.md)) ⇒ dự báo hiệu ứng nhỏ. Vòng 13 đang chạy `fd_dml_m4b_gpool_p1` + hai control `p1self`, `p1cm` (s42 ep6 từ đầu,
[design §Vòng 13](../../methods/full-drift/design.md)). Độ phủ gate đo trong kernel probe. Đề xuất owner 17/10 (ưu tiên 1).

## Vấn đề (bằng chứng)
KD hiện chỉ trên dòng thật local ([`fd.py::private_epoch`](../../../methods/full-drift/builds/fd-probe/source/proj/fd.py)); 256 mẫu
Gaussian chỉ nhận CE cứng ở private. Tăng KD trên local không đóng khoảng: `gpool_kd2` +0,001, `gpool_kdpx` −0,008
([vòng 11](../../methods/full-drift/results.md)). Khoảng không đều giữa các lớp (s42, group ep6, private gp; proxy head cùng arm):

| Lớp | Private m4b | Private gpool | Proxy gpool | Đọc |
|---|---:|---:|---:|---|
| benign | 0,4994 | 0,5023 | **0,7773** | Teacher có tín hiệu private chưa hấp thụ |
| dataReplay | 0,4486 | 0,4714 | **0,7157** | Tương tự benign |
| positionMirroring | 0,2526 | 0,2429 | 0,2312 | Teacher còn yếu; không ép private bắt chước |
| timeDelayAttack | 0,1819 | 0,1776 | 0,2518 | Cả hai yếu; thiếu support hay thiếu thông tin? |
| suddenConstantSpeed | 0,5517 | 0,5674 | 0,6759 | Có khoảng truyền tri thức |
| suddenStop | 0,6662 | 0,6558 | 0,7555 | gpool không tăng mọi lớp |

Ba giả thuyết cần tách: teacher thiếu chỗ để dạy; synthetic không phản ánh ranh giới; teacher cũng chưa biết phân biệt.

## Cơ chế
1. Giữ toàn bộ CE/KD real của baseline. Teacher = proxy gộp `g^(r−1)` nhận được đầu epoch, eval mode, no-grad.
2. Cùng 256 Gaussian, cùng class balance và seed stream như gpool; CE với nhãn sinh mẫu giữ nguyên.
3. Thêm một chiều KL teacher → private trên synthetic; proxy đang train vẫn chỉ học real.
4. Chưa nhận snapshot gộp (round 1) thì tắt.

L = L_gpool + λ_s/m · Σ_j a_c a_j T² KL(q_teacher^T(x̃_j) ‖ p_k^T(x̃_j)), m = 256. Khóa: T = 1, λ_s = 0,25 (điểm khởi đầu
thiết kế, không quét thêm trên cùng tập chấm). a_j = 1 khi argmax teacher = nhãn Gaussian (lọc bảo thủ, không chứng minh nhãn
đúng). a_c = {benign, dataReplay} — lớp lịch sử cho thấy teacher mạnh; không tự bật positionMirroring. Chia cho m, không chia
số mẫu qua gate.

R1: 0 byte thêm (client vốn nhận proxy). R2: chỉ cần output 16 lớp của private. Proxy có thể đổi gián tiếp qua KL proxy←private.

## Controls (khóa vòng 13)
`p1self`: cùng công thức, teacher = private của chính client đầu epoch (tách lợi tự-chưng-cất). `p1cm`: cùng gate, target =
trung bình xác suất teacher các mẫu được gate cùng lớp trong batch (không phụ thuộc x). Cơ chế PASS chỉ khi p1 qua cổng gain và
p1 − p1self ≥ +0,005, p1 − p1cm ≥ +0,005.

## Dấu hiệu sai
Private benign/dataReplay không tăng; gap private–proxy ở lớp đã khóa không giảm; worst client giảm; gate gần rỗng; chỉ loss
synthetic giảm. Dừng nếu không qua cổng; không cứu bằng quét λ/T.

## Khác các bản đã FAIL / tiền lệ
- [P1 Gaussian-logit cũ](../../archive/methods/gaussian-logit-distillation/README.md): teacher local yếu, soft target không hơn nhãn
  Gaussian. Ở đây teacher là proxy đã học qua federation, giữ hard CE.
- `kdcap_frz` đổi teacher trên real cùng deployment prior ([kd-cap](kd-cap-frozen-teacher.md)); ở đây giữ KD real, chỉ thêm support mới.
- Hybrid cũ cho proxy học Gaussian; ở đây không.
- Build7 (KD từ teacher peer, đúng 63–85 %) không thêm gì so head ([kế thừa](../../archive/legacy/kien-thuc-ke-thua.md)).
- Đo local 17/10 (moment pilot): proxy gpool ep6 chỉ nhận 25 % mẫu benign, 3 % positionMirroring, 0 % timeDelay ⇒ gate
  {benign, dataReplay} phủ ít; độ phủ trên moment full đo trong probe.

## Nguồn
FML ([arXiv:2006.16765](https://arxiv.org/abs/2006.16765)) — private độc lập + model chung. Đề xuất owner 17/10 (file ưu tiên đã tách).
