# Y50 — ba ứng viên cải thiện Core, khóa trước launch

Owner 06/11: thử 3 ứng viên đầu của [pipeline-review](../../docs-20261107/research/pipeline-review.md), tạm chỉ cải thiện Core, được nối phiên khi thiếu quota.
Cell, luật và ngưỡng khóa ở `scripts/y50_common.py`. Control: F2 (s42), F3 (s43), cùng base source, không chạy lại Core.

## Ô

| arm | ô | seed | đổi đúng một thứ so với Core | account | đích |
|---|---|---|---|---|---|
| Y0 | oracle | 42 | `synth=oracle`: 64 dòng thật cân bằng lớp từ pool của rank, không triển khai được | outlookcheck | r50 |
| Y1, Y2 | prior | 42, 43 | `mobihet_prior`: shift s = log(1/16) − log π_g trên dòng synthetic và khi gom Q | heykewtiie, keocaosu | r20, qua luật mới tới r50 |
| Y3, Y4 | noq | 42, 43 | `no_kd=true`: synthetic nhãn cứng, không gửi Q, 0 byte/round | novahihi, nguyetquedo | r50 |
| Y5, Y6 | nofloor | 42, 43 | bỏ `synth_step_floor` | accanniez, outlookcheck | r50 |

Giữ protocol: 50 round cosine, batch 64, W1, 5.343 xe, checkpoint mỗi round. Không đổi HP khác.

**Vì sao Q cũng đổi tọa độ ở ô prior:** Q đo trên z thô đã mang π_g; khớp softmax(z+s) với Q đó sẽ tính prior hai lần và đẩy lớp benign.
Đây là một thay đổi cơ chế, không phải hai. Pipeline-review ghi thiếu phần Q; quyết định này khóa trước khi có số.

## Lịch quota

Owner 06/11: dùng ngay slot thứ hai của outlookcheck và các quota ngắn, nối phiên sau. Cả 7 kernel chạy từ 06/11.

| arm | session 1, cap | quota 06/11 | sau đó |
|---|---|---|---|
| Y0 | outlookcheck, 11 h | 19,79 h chia hai slot | budget stop thì `-c2` |
| Y6 | outlookcheck slot 2, 8 h | Y0 11 h + Y6 8 h < 19,79 h | `-c2` sau reset 07/11 |
| Y1, Y2 | heykewtiie, keocaosu, 5,3 h | 7,51; 6,60 | tới r20, qua luật thì `-c2` |
| Y3, Y4, Y5 | novahihi 5,6 h; nguyetquedo 5,4 h; accanniez 4,6 h | 6,49; 6,30; 5,33 | `-c2` sau reset 07/11 |

Mỗi session lưu state mỗi round. Session tới cap thì COMPLETE trước đích; worker đợi tới khi quota cùng owner đủ số round còn lại
(.21 h/round + 1 h, ×1,15) rồi push `-c2`, tối đa 3 session. ERROR không tự resume. Không chuyển account nên không cần dataset trung gian.

## Luật đọc

Ngưỡng ồn: accuracy, macro, other15, benign .0005; p10 .006; low256 .003; timeDelay .0035. Δ = ô − Core, ghép theo seed. Hai seed không phải significance.

| ô | đọc tại | kết luận |
|---|---|---|
| prior | r20 rồi r50 | PASS khi cả hai seed: other15 và benign tăng ≥.0005; p10, low256 không giảm quá ngưỡng; timeDelay giảm <.02; absent giảm <.01. Trượt r20 thì dừng, không chạy tiếp. Quyết định cuối chỉ ở r50 |
| noq, nofloor | r50 | `removable`: accuracy, macro, p10, low256, timeDelay không giảm quá ngưỡng ở seed nào. `needed`: một metric giảm quá ngưỡng ở cả hai seed. Còn lại `unresolved`, giữ nguyên Core |
| oracle | r50, một seed | `generator_worth_it`: macro ≥+.01 hoặc timeDelay ≥+.05. `stop_generator`: macro <+.0005 và timeDelay <+.0035. Còn lại `small_ceiling` |

`noq removable` thì Core thành one-shot, chỉ còn bootstrap. Oracle chỉ định vị trần, không phải phương pháp.
Không thêm ô, không đổi ngưỡng sau khi thấy số. Trượt thì không tinh chỉnh tiếp nếu không có giả thuyết mới.

## Byte/xe/50 round (bootstrap thường, chưa lossless CDF)

Core, prior, nofloor 1.054.839; noq 1.000.439 (−54.400). Oracle không có nghĩa về byte.

## Vận hành

Khởi động đo từ log F3: 10,5 phút/session, trong đó 80 s hash cache npy, ~8 phút compile (gate 110 s, thừa ở r1 train 110 s, eval 256 s).
Wave 2 giữ cache Inductor trong `inductor-cache.tar` của thư mục run để `-c2` dùng lại; local sm86 chạy ấm 36 s so với 140 s, trùng bit. Chưa đo trên T4.

`scripts/run_y50.py --wave N` (restart an toàn, `scripts/keep_y50.py` giữ worker sống), receipt `runs/y-study/wave<N>`, artifact `artifacts/y50`, audit `scripts/audit_y50.py`.
Preflight `scripts/preflight_y50.py`: mỗi ô chạy thẳng r4 so với dừng r2, resume, giả lập crash; local sm86, world 1, 2 xe, không chứng minh 2 rank T4.

## Điều chỉnh vận hành owner 06/11 buổi tối

Tận dụng quota ngắn ngay: cap = min(cap phần còn lại, gpu_left/1,15), tối thiểu1,21h; tối đa6session.
Audit endpoint partial trước nối; cùng owner để mount fullstate private, cache npy và Inductor tar.
Giữ nguyên round đích, lịch cosine50 và mọi luật khoa học phía trên. Prior đã FAIL tại r20, không nối.
