# H0–H4 — khóa sàng lọc 30/10/2026

Owner duyệt trong phiên 30/10: **chạy H0–H4; quyết định 50 round sau kết quả**.
Nền/cổng kế thừa [improvement-plan §5](improvement-plan.md); kết quả cũ: [screening-r10](screening-r10.md).

## Cấu hình khóa trước push

Full 5.343 xe, DAGSNet W=1, batch 64, mọi xe một epoch/round; AdamW và cosine **50 round**, dừng sau **r10**.
`arm=none`, `group_rows=4096`, W&B `21522798-uit/mobihet-v4`, 2×T4. Seed 42 cho H1–H4; H0 lặp seed 43.
Ngân sách 4 h/session, 6 session trên 6 account (quota từng account đã đọc; không đổi login toàn cục).
Không chạy F/50 round trong batch này. Không đổi HP theo kết quả test khi batch đang chạy.

| arm | slug | account | thay đổi so G4 seed 42 |
|---|---|---|---|
| H0a | `mh-h0-gauss-s43` | nguyenminhtran060113 | S4 gauss, seed 43 |
| H0b | `mh-h0-copula-s43` | thuyanh316 | G4 copula, seed 43 |
| H1 | `mh-h1-cluster` | annhannh331 | `synth=copula_cluster` |
| H2 | `mh-h2-copulamix` | minhtriethihi | `synth=copula_mix` |
| H3 | `mh-h3-dose` | annhtnt | `synth=copula synth_rows=64` (nền: 32) |
| H4 | `mh-h4-onehot` | takids1909 | `synth=copula synth_target=onehot` |

H1: histogram 256 bin trên [-4,4] và moment normal-score tách theo **lớp×8 nhóm scenario/run**.
Mỗi dòng synthetic lớp c chọn nhóm với xác suất n(c,g)/Σ_g n(c,g); nhóm rỗng xác suất 0. Nhóm lấy từ `clients.parquet`,
không dùng nhãn test. Giữ số dòng, phân bố nhãn cân bằng và phép nghịch đảo CDF của G4.

H2: sinh copula rồi mix với dòng thật cùng lớp **trong batch local**, λ~U(0,1); lớp không có trong batch giữ dòng copula.
H3: tăng 32→64 dòng synthetic/bước, vẫn lấy trung bình loss synthetic riêng; không nhân đôi trọng số loss.
H4: ở r≥2, thay KL(Q[y_g]‖p_g) bằng CE(p_g,y_g), **giữ hệ số adaptive raw và KD trên dòng thật**.
R1 vốn dùng synthetic CE nên H4 không đổi r1 về công thức. Không thêm logit adjustment cho synthetic.

## Bằng chứng → cơ chế → dấu hiệu bác bỏ

| arm | bằng chứng/vấn đề | cơ chế kỳ vọng | dấu hiệu bác bỏ |
|---|---|---|---|
| H0 | G4−S4 chỉ +0,00529, một seed | kiểm độ lặp của lợi bộ sinh | Δ(seed43)≤0: lợi copula không lặp dấu |
| H1 | oracle 0,8802 > G4 0,8732; thống kê gộp trộn nhiều chế độ | giữ tương quan và biên trong từng nhóm môi trường | không qua cổng pooled/p10; hoặc chi phí bootstrap không chấp nhận được |
| H2 | mix tăng p10, copula tăng pooled | thêm dòng gần dữ liệu xe vào copula | không qua cổng pooled/p10 |
| H3 | nền chỉ 32 dòng synthetic/64 dòng thật | thêm phủ lớp thiếu, giữ chuẩn hóa loss | không qua cổng hoặc tốc độ/AMP hỏng |
| H4 | teacher Q[y] chỉ tùy nhãn, có thể quá mềm | synthetic CE cung cấp đích nhãn rõ hơn | không qua cổng hoặc gây hại xe yếu |

Primary: private pooled F1-macro r10, **không chọn round tốt nhất**. H1–H4 so G4 seed42 từ artifact:
F1=0,873187514625593; p10=0,5737619047619047. PASS nếu ΔF1≥0,010 **và** Δp10≥−0,010.
H0 mô tả paired G4−S4 ở seed43, so dấu với seed42; hai seed không chứng minh ý nghĩa thống kê.
Kèm đủ 10 metric, mean/p10/min theo xe, benign F1, thời gian, skip, bootstrap + byte hai chiều.
Oracle là đối chứng tham khảo của một sampler, không phải cận trên toán học. Không hạ cổng nếu mọi arm FAIL.

## Chi phí phải tính cả bootstrap

`none` gửi 544 B lên **và** 544 B xuống mỗi round = **1.088 B/xe/round**, chưa tính bootstrap.
Từ `config.json` đã pull: S4 bootstrap=394.368 B hai chiều/xe; G4=3.311.744 B.
Tổng 10 round: S4=405.248 B, G4=3.322.624 B. Tổng 50 round nếu payload giữ nguyên: 448.768 B và 3.366.144 B.
H1 dùng bảng dense cho 8 nhóm nên bootstrap còn lớn hơn; báo theo cấu hình thật trước khi chọn arm.
Đây là kế toán payload của harness, chưa tính transport/secure aggregation; không dùng 544 B làm tổng chi phí phương pháp.

## Source và kiểm chứng

Không Git push. Notebook clone base SHA cố định, ghi đè `methods/mobihet.py` bằng byte local nhúng base64 đã kiểm SHA256.
Notebook ghi `provenance/source_manifest.json` và bản source không có credential vào output để audit đúng code chạy.
Notebook private vẫn nhúng W&B key theo quyền owner đã có: key tồn tại trong lịch sử version private; notebook ở thư mục ignored, mode 0600.
Các run độc lập bắt đầu r1; không tuyên bố exact resume từ các checkpoint hiện tại.

Trước push: 24 test CPU PASS; gauss/copula/mix sau sửa khớp từng bit với source cũ sau 2 round toy.
Smoke real-data compiled+AMP và trạng thái runtime ghi trong bàn giao; kiểm local một GPU không chứng minh hai T4.
Kết quả audit ở dưới; runtime hiện tại ở [bàn giao sống](../operations/continue.md).


## Kết quả audit 30/10 ~22:30

6 kernel version 1 COMPLETE; bulk pull (bỏ state*.pt) exit0, 47 file/run, 0 thiếu, ~367 MiB/run (~2,15 GiB tổng).
Selective pull metric/source riêng cũng exit0, 37 file/run, 0 thiếu; 60/60 snapshot tải được weights_only=True, mọi tensor hữu hạn.
Audit tái lập: `python scripts/audit_screening.py runs/h-screening/audit-input --output runs/h-screening/audit.json`.
Đủ r1–10, đúng HP/lịch LR, IDs 0…5342; support từng client×lớp khớp G4, tổng 9.347.679 test.
Manifest base/overlay và toàn bộ source khớp SHA. Mười metric mỗi round khớp JSONL từ CM.
Giới hạn: chưa dựng lại prediction→CM hoặc full private bank; chưa chứng minh bit-exact resume.

| arm | F1 r10 | Δ vs G4 | p10 | Δp10 | benign F1 | train s/r (r2–10) | bootstrap + 10r B/xe | cổng |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| H0a | 0.869048 | -0.004140 | 0.569274 | -0.004488 | 0.852507 | 529.5 | 405,248 | mô tả seed43 |
| H0b | 0.874508 | +0.001320 | 0.568757 | -0.005005 | 0.856487 | 551.7 | 3,322,624 | mô tả seed43 |
| H1 | 0.873943 | +0.000756 | 0.578691 | +0.004930 | 0.856952 | 565.1 | 23,744,256 | FAIL |
| H2 | 0.869818 | -0.003370 | 0.587524 | +0.013762 | 0.853464 | 551.6 | 3,322,624 | FAIL |
| H3 | 0.876895 | +0.003708 | 0.608332 | +0.034570 | 0.857620 | 631.0 | 3,322,624 | FAIL |
| H4 | 0.868698 | -0.004490 | 0.586652 | +0.012890 | 0.845952 | 550.4 | 3,322,624 | FAIL |

**H1–H4 đều FAIL cổng gain**, không đổi ngưỡng. H3 tốt nhất mô tả: +0,003708 pooled, +0,034570 p10.
H0 copula−gauss seed43 = +0,005460; seed42 = +0,005288 ⇒ lặp dấu, chưa đủ chứng minh ý nghĩa thống kê.
H1 tăng bootstrap lên 23.733.376 B/xe nhưng pooled chỉ +0,000756. H2/H4 giảm pooled.
H4 r10 skipped_frac=1,7053e-6 (rất nhỏ); các round/arm khác bằng 0. Không ghi nhầm mọi arm skip0.

### Mười metric r10 (từ CM)

| arm | accuracy | precision_macro | precision_micro | precision_weighted | recall_macro | recall_micro | recall_weighted | f1_macro | f1_micro | f1_weighted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H0a | 0.908384 | 0.918309 | 0.908384 | 0.907075 | 0.866660 | 0.908384 | 0.908384 | 0.869048 | 0.908384 | 0.887481 |
| H0b | 0.911680 | 0.926493 | 0.911680 | 0.910741 | 0.870176 | 0.911680 | 0.911680 | 0.874508 | 0.911680 | 0.890687 |
| H1 | 0.911936 | 0.925031 | 0.911936 | 0.910222 | 0.869476 | 0.911936 | 0.911936 | 0.873943 | 0.911936 | 0.890948 |
| H2 | 0.908954 | 0.915077 | 0.908954 | 0.905234 | 0.867059 | 0.908954 | 0.908954 | 0.869818 | 0.908954 | 0.888205 |
| H3 | 0.912905 | 0.925796 | 0.912905 | 0.911293 | 0.874049 | 0.912905 | 0.912905 | 0.876895 | 0.912905 | 0.891964 |
| H4 | 0.904802 | 0.878688 | 0.904802 | 0.881286 | 0.870816 | 0.904802 | 0.904802 | 0.868698 | 0.904802 | 0.886830 |

Mọi round kèm đủ 10 metric, per-class, mean/p10/median/min trong `runs/h-screening/audit.json` (artifact local ignored).
Payload steady-state 1.088 B/xe/round; bảng tính bootstrap hai chiều, chưa transport/secure aggregation.
Oracle G1 là đối chứng sampler, không phải cận trên toán học. Không kết luận bộ sinh tối ưu từ batch này.
