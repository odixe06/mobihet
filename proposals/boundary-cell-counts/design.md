# BoundaryCells — thiết kế trước pilot

## 1. Chủ dự án chốt 2026-10-13

- Cả direct target và teacher DAGSNet trung gian; mọi learned model DAGSNet.
- P=2/4, B=4, z-score edges −1/0/1, alpha=1, auxiliary weight=0,5.
- gp chính theo khối lượng target thực tế của CE; gp cũ và head báo phụ.
- Probe target hơn own/shuffle trước pilot. Không raw/perturbed record rời client.

Dictionary cố định: speed×implied-speed, distance×position-prediction-error,
sender-rate5s×sender-share5s, acceleration×acceleration-residual. P2 lấy hai cặp đầu.
Tên feature/tie convention ở [cells.py](../../papers/boundary-cells/proj/cells.py).
Không đổi dictionary/edges/alpha để cứu kết quả; bin ngoài không clamp x.

## 2. Cơ chế

Client tính n[i,r,b,y] và N[i,y] trên train trước SGD; input bins không phụ thuộc model.
RSU nhận tổng cohort/contributor totals; trusted-sum emulator chưa crypto/DP.
Cell suppression nếu bất kỳ nhãn count dương có contributor<5, hoặc tổng cell<32.
Không có eligible cell tại x thì skip auxiliary loss, giữ CE thật.

ell=(n+1)/(N+16); target=ell/sum_y ell; trung bình target các cặp eligible tại x.
Direct: CE hard +0,5 CE soft. Intermediate: local teacher DAGSNet fit bảng trên own x,
private CE hard +0,5 CE teacher, T=1. Không raw x tại teacher RSU.
Teacher có thể thêm approximation error; không mặc định hơn direct.

π_eff mỗi client tính target mass của **bước applied**, đúng trọng số/normalization từng CE.
gp=argmax(logits−logπ_eff+logπ_global); mass/smoothing lưu để audit.
Hiệu chỉnh vẫn có giả định label-shift, không bảo đảm posterior đúng. Không fit prior từ holdout/test.

## 3. Probe nhỏ đã đo

40.960 sampled train row, seed1729; 4.096 group holdout. Không official test, không SGD.
Sample không là full-train/cohort coverage proof.

| cặp | table F1 macro | Δ own | Δ shuffle | entropy/log16 | group coverage |
|---|---:|---:|---:|---:|---:|
| 2 | 0,08525 | +0,01550 | +0,06530 | 0,95496 | 92,87% |
| 4 | 0,22828 | +0,07243 | +0,20030 | 0,96268 | 100% |

P2 coverage randomPositionOffset11,56%, zeroSpeedReport35,23%; mean coverage che lớp khó.
P4 coverage đủ trên sample, log-score hơn own/uniform, nhưng table F1 vẫn thấp.
Đó là tri thức có điều kiện, chưa chứng minh cải thiện DAGSNet mạnh hơn bảng.
Nguồn số đầy đủ: [JSON](../../papers/boundary-cells/bounded-probe.json).

**Dự báo cập nhật:** P4 có cơ sở hơn P2; hơn local bất định, hơn Gaussian/DML ít cơ sở hơn.
Không tăng weight vì transfer dương. Không chốt thêm entropy/coverage threshold sau khi thấy sample.
Kaggle đo lại full pilot train/cohort, không chọn HP từ group holdout.

## 4. Controls/cổng đã khóa trước huấn luyện

Mỗi mode/P: pooled, own, cell-shuffle, single-axis-marginal; fresh local,
class-mean/prior-only và Gaussian-full references. Cùng hai-DAGS compute budget;
direct có teacher diagnostic không đưa output vào student.
Shuffle target giữa eligible cells, giữ mask: giữ multiset target **trên bảng**,
không tự giữ entropy/target mass khi lookup trên record với cell frequencies khác nhau.

Chủ dự án chốt group/private/gp r20, seed42/43: FL−own/shuffle≥0,010,
FL−reference mạnh hơn≥0,015, benign giảm≤0,005. Cost/channel báo riêng.
Cost≤2×freshlocal một DAGSNet; calibration2round trước, vượt thì giữoutput/dừng.
Không teacher fit/table F1/byte thay utility PASS.
[Contract source/calibration](../../papers/boundary-cells/contract.md) khóa teacherfit20batch/round,
CEtargetmass theo normalization của loss; r20 mới là cổng utility.

## 5. R1/R2

Packet thực serialize: P2 lên2.720/xuống2.208B; P4 lên5.280/xuống4.320B.
FP32 down, countuint32/contributoruint8/classuint64 up; header32B/CRC/version/hash.
Dictionary/bootstrap/scaler/ACK/retransmission/recovery cộng khi contact test.
P2 không vừa một contact4,5kB khi tổng hai chiều4.928B chưa overhead.
Chỉ áp coherent snapshot; private không align weights/latent. DAGSNet-only chưa chứng minh R2.
