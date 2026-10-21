# LESSONS — bài học chung về huấn luyện MobiHet

Đọc sau [CONTEXT](CONTEXT.md). Lỗi/quy tắc vận hành nằm ở CONTEXT và [sổ lỗi kernel](docs/operations/notebook-errors.md).
Bài học riêng còn dùng ở [docs/lessons](docs/lessons/README.md); bằng chứng chi tiết ở [archive](docs/archive/README.md); mọi con số phải trỏ báo cáo đã audit.
[Bản đầy đủ trước tổ chức](docs/archive/legacy/lessons-before-layout-20261014.md) giữ lịch sử, không xóa im lặng bài học bị bác bỏ.

## Tri thức nào cải thiện IDS

- Nút thắt là ranh giới đa receiver: private dễ học thuộc receiver của mình. Head-only/prototype trung bình lớp
  không chở đủ ranh giới; cần tác động backbone và so đối chứng quần thể/own. [Bằng chứng kế thừa](docs/archive/legacy/kien-thuc-ke-thua.md).
- Gaussian input covariance mang tri thức quần thể: own Gaussian chỉ giải thích phần nhỏ lợi; diag/lr/full giữ cấu trúc
  khác nhau. Input không trôi theo encoder. Mean lớp hoặc biến đổi mean nhiều lần không thay covariance.
  [Gaussian moments](docs/methods/gaussian-class-moments/README.md).
- Latent statistics cũ đi khi backbone còn học: offline probe tích cực không bảo đảm loss online có lợi.
  Snapshot/head đóng băng là phạm vi khác, phải kiểm riêng. [Bài học kế thừa](docs/archive/legacy/kien-thuc-ke-thua.md).
- Teacher phải đủ chất lượng. Ensemble local/teacher yếu có thể làm nhãn Gaussian miễn phí xấu đi;
  có teacher gate và class-mean control trước KD. Proxy DAGSNet có sức chứa tốt hơn proxy yếu trong P1.
- Ít byte không tự giữ sức chứa: seeded subspace có tối đaK bậc tự do chung. Compression contribution PASS
  không thay retention/utility PASS. [Subspace](docs/archive/methods/proxy-seeded-subspace/README.md), [carriers](docs/archive/methods/proxy-compression/README.md).
- Synthesis loss giảm, codec/đạo hàm PASS không chứng minh IDS tốt hơn. KME/DM/gradientbooks đã trượt utility;
  coverage cao có thể đi cùng soft target gần uniform. [Synthetic knowledge](docs/archive/methods/synthetic-knowledge/README.md).
- V9 M512 r20: 95–96% correction energy chung giữa anchors; bootstrap vẫn khoảng95%, kernel group entropy0,955
  dù coverage>99%. Coverage/teacher-fit loss thấp không bảo đảm chở ranh giới; không kéo dài adaptation này.
  Chẩn đoán saved-target không cô lập toàn bộ BN/fit/extrapolation, không bác cả họ residual.
  [Forensics hai seed](docs/archive/methods/anchor-function-residual/builds/anchor-residual/results/forensics-20261015.md).

- **Biên lớp của Gaussian moment lệch xa dữ liệu thật** (fixture 1M dòng, 53 feature m4b): KS trung vị 0,19, p90 0,49; 33 % cặp lớp×feature
  có atom > 20 %. Gaussian copula (histogram biên gộp + tương quan normal-score, đều SecureSum cộng tính) nâng độ đồng ý của snapshot gkd
  với nhãn mẫu 0,473 → 0,673 (thật 0,805), benign 0,27 → 0,51. Kênh mẫu tổng hợp nên khớp biên trước khi tinh chỉnh khác (vòng 18 kiểm utility).
- **Tri thức quần thể nằm trong proxy chung, không trong private** (khi đánh giá trên test toàn cục). DML proxy head hơn private gp
  +0,115/+0,122 group pilot và +0,056 full r20; trộn private vào proxy (ProxyFuse) **kém** proxy −0,032/−0,034,
  private đúng mà proxy sai chỉ ≈2% dòng. Owner15/10: proxy là headline DML/Hybrid. Cải tiến nên nhắm proxy
  (model triển khai) hoặc khả năng hấp thụ, không chỉ private. [ProxyFuse](docs/methods/proxy-fusion/results.md).
- **Kênh thống kê vô hướng ≈ local.** ClassTail, class-mass, CCR pooled/own/shuffle/fixed/classweight đều trong
  ±0,002 của CE local ở pilot group r20; Gaussian (+0,08) và proxy dense (+0,19–0,31) mới mang ranh giới.
  Không mở thêm kênh scalar/score-moment nếu không chở thông tin hàm/input mới.
- **Chống shortcut receiver giúp model chỉ thấy dữ liệu local; ở pilot không giúp proxy chung** (⚠ full vòng 3 bác bỏ phần proxy, xem bài dưới). AntiMem pilot r20 hai seed:
  Gaussian private gp m4b +0,025, swappos +0,016–0,018, swap +0,013, private_ema +0,035–0,039 (kết hợp +0,056–0,058);
  DML/Hybrid proxy head mọi kỹ thuật −0,012…+0,006, proxy EMA −0,028…−0,038. Proxy đã gộp 20 client nên shortcut
  không còn là nút thắt; muốn cải thiện headline phải nhắm tối ưu/tổng hợp proxy. [Kết quả](docs/methods/anti-memorization/results.md).
- **Proxy dense thừa độ chính xác.** QProxy q4 (int4 theo hàng, EF phía server) = dense ±0,001 proxy và private
  hai seed, byte nhỏ 7,65×. Nén theo tọa độ có chặn lỗi giữ được utility, khác sketch/subspace đã FAIL.
- **Phạm vi byte q4 — kiểm source 17/10 (đã sửa 17/10 chiều):** FullDrift giờ dùng codec thật `QPacket` (pack nibble, scale
  fp16, CRC, base/version), server/client chỉ dùng tensor decode từ bytes; 208.268 B/chiều = `len(packet)`, train bit-exact
  với q4 đã audit. Đoạn dưới là hồ sơ trước khi sửa: tỷ lệ7,65× khi đó là bộ đếm codec dự kiến, chưa phải byte wire end-to-end.
  FullDrift `quantize` đếm4bit nhưng `latest_packet` ghi `q` int8 chưa pack nibble; buffer/counter được đếm nhưng
  packet nhánh này chưa serialize đủ, upload cộng tensor decoded trực tiếp. Source khớp artifact vòng5 đã audit.
  Utility lượng tử hóa vẫn có bằng chứng; R1 cần codec pack/unpack thật, tính `len(packet)` cùng overhead và
  decode/commit qua contact. [Chi tiết và source](docs/archive/research/algorithm-improvement-plan-2026-10-17.md#31-q4-utility-đã-kiểm-wire-bytes-chưa-được-chứng-minh-đầy-đủ).
- **Kéo proxy về model chung không giúp ở full.** FullDrift seed42 ep6 group: FedProx μ0,01 giảm drift 5× mà F1
  không đổi (benign +0,098); FedAvgM β0,9 phân kỳ. FedAvg thuần ≈ DML proxy (+0,009).
  Gaussian swappos ở full +0,052, ngừng suy giảm sau r1. [Kết quả](docs/methods/full-drift/results.md).

- **Ở full, bỏ khối feature receiver giúp cả client lẫn proxy DML.** FullDrift vòng 3 seed42 ep6 group: m4b private gp
  +0,036, proxy +0,052, benign +0,11; swappos +0,033/+0,049. 0 byte thêm (R1), chỉ phía client (R2). Pilot r20
  từng FAIL proxy ⇒ học thuộc receiver chỉ lộ khi mỗi epoch ~4.200 bước. FedNTD hại private (−0,042): khoảng private–proxy
  không do lệch prior. q4 không thua. [Kết quả](docs/methods/full-drift/results.md).
- **Học thuộc receiver là lý do proxy DML chững và benign sụp theo epoch ở full** (vòng 5, ep15): `fd_dml` proxy phẳng
  0,763 từ ep4, benign proxy 0,74→0,36; m4b proxy còn tăng 0,815→0,851, benign giữ ~0,77, lợi lớn dần (+0,052 ep6 →
  +0,086 ep15; private +0,036). Plateau không phải hết capacity hay LR; kéo dài run không bỏ shortcut thì vô ích.
  Gaussian m4b +0,067. Kỹ thuật cộng được: m4b + q4 (±0,000). **Seed 43 lặp lại** (vòng 6–9): m4b private +0,031/
  proxy +0,039, swappos +0,034/+0,031; ep15 private +0,025/proxy +0,063, benign proxy 0,78 vs 0,48 ⇒ headline DML = m4b.
- **EMA private không cộng với m4b.** Gaussian-m4b private_ema −0,010 (s42), −0,021 (s43) dù EMA giúp Gaussian không
  mask (+0,024) và swappos (+0,036): khi đã bỏ shortcut, trung bình quỹ đạo chỉ làm chậm. Kiểm từng tổ hợp, không cộng lợi.
- **Đổi kênh KD không đóng khoảng private–proxy; đổi input phía client thì có.** FedNTD −0,042, hidden-state KD (FedKD,
  projector local) +0,006, cùng kênh KD m4b +0,036. Trước khi thêm kênh/loss chưng cất, kiểm private có học shortcut local không.
- **Tri thức lớp gộp giúp client khi đưa vào private dưới dạng dữ liệu phụ, không phải qua proxy.** FullDrift vòng 9–10:
  private DML-m4b thêm CE trên 256 mẫu Gaussian từ moment gộp (`gpool`) +0,016 (s42) / +0,024 (s43), hơn moment riêng
  +0,030 và hơn replay local cân bằng lớp cùng số forward +0,032 ⇒ lợi do thông tin của client khác, không do thêm CE.
  Proxy không đổi, +4,2 MiB/6 epoch. Hybrid cũ (proxy cũng học synthetic) thì interaction âm. Lợi FL thật: DML-m4b hơn
  proxy không gộp +0,040–0,043, Gaussian-m4b hơn moment riêng +0,05 (cả hai seed).
- **KD mạnh có lợi cho private, KL ngược mạnh hại proxy.** Trọng số `1/(CE+CE)` lên ~4 ở ep6 (‖w∇KL‖ ≈ ‖∇CE‖); trần 1
  làm private −0,025…−0,028 nhưng proxy +0,024/+0,033 (hai seed). Nếu sửa trọng số, tách hai chiều KL.
- **KD trên dòng local đã bão hòa; teacher tốt hơn không tự sang private** (vòng 11, gpool s42): KD private ×2 +0,001;
  chỉ cap KL proxy←private làm proxy +0,031 nhưng private −0,008. 56 % khoảng F1 private–proxy ở lớp < 1 % dữ liệu local —
  KD trên dòng local gần như không chạm các lớp đó. Không quét trọng số KD nữa; nhắm kênh mang tri thức lớp hiếm.
- **Mẫu Gaussian gộp lệch xa dữ liệu thật ở lớp khó.** Proxy ep6 chỉ đồng ý 53 % mẫu (benign 25 %, positionMirroring 3 %,
  timeDelay 0 %) trong khi private khớp 78 % ⇒ KD proxy trên mẫu Gaussian sẽ xung đột nhãn cứng. Đo độ đồng ý của proxy với
  mẫu tổng hợp (local, miễn phí) trước khi đề xuất kênh dùng mẫu đó.
- **Cổng ep6 đo tốc độ học, không đo mức hội tụ** (vòng 11 lc): gpool hơn m4b +0,054 ep1 → +0,016 ep6 → +0,002 ep15
  (horizon FAIL; test ep15 0,7486 vs 0,7478), proxy gpool thấp hơn từ ep9. Lợi tri thức dạng dữ liệu phụ có thể chỉ là
  khởi động nhanh mà KD từ proxy mạnh dần bù lại. Arm PASS ở ep6 phải lặp horizon ep15 trước khi nhận làm headline.
  gw2 (×2 liều Gaussian) −0,005, client tệ nhất −0,010. [Kết quả](docs/methods/full-drift/results.md).
- **Private gpool học prior gần π_k local; mẫu Gaussian không đổi prior của head trên dòng thật** (probe vòng 13): luật gpe theo
  prior trộn (π_k + q)/2 −0,094/−0,095 hai seed; BN tính lại chỉ trên dòng thật −0,03/−0,04 (BN trộn là một phần hàm đã học);
  hỗn hợp moment urban/highway không làm proxy nhận mẫu nhiều hơn; ridge RFF trên input thô M256 0,38 (M1024 0,48) ≪ DAGSNet 0,72.
  Khó ở positionMirroring/timeDelay không do mode trong lớp. [Probe](docs/methods/full-drift/results.md).
- **Rà kênh đã thử trước khi mở hướng mới; dùng lại weights đã lưu khi phép thử không cần train.** Đề xuất 17/10 P4 (ridge/RFF
  teacher tại RSU) trùng Build13 S13 (lợi ≈ làm mượt theo lớp, thua replay −0,09) ⇒ chỉ còn kiểm teacher-only. Luật triển khai
  (gpe), BN tính lại, độ phủ teacher chấm được trên weights ep6 đã audit. Với m4b, Gaussian 53 chiều = 66 chiều đặt 0 về phân phối
  ⇒ moment53 không cần train. Danh sách kênh cũ: [kế thừa §4](docs/archive/legacy/kien-thuc-ke-thua.md).
- **Private chững vì tầng quyết định, không vì backbone** (probe 14, 3 bộ weights ep15): đóng băng backbone private, fit lại riêng Linear
  cuối trên dữ liệu gộp → 0,84 (MLP head 0,87) vs private gp 0,73 / proxy 0,85, mọi client +0,05…+0,19; refit trên dòng own 0,68;
  bias oracle chỉ +0,045. Lỗi chính: gán tấn công cho benign ở receiver chưa thấy (benign recall 0,50 vs 0,82). Kênh mới phải nhắm
  ranh giới quyết định trên phân phối toàn cục, không thêm KD/biểu diễn. [Kết quả](docs/methods/full-drift/results.md).
  Probe 16: backbone ep6 không FL (Gaussian own) refit gộp 0,81/0,86 ≈ DML-m4b 0,83/0,87 ⇒ đặc trưng học được tại chỗ, lợi FL nằm ở
  tầng quyết định. Fit Linear cuối bằng dòng thật client khác **nhãn proxy** 0,831 ≈ nhãn thật 0,843 ⇒ proxy gán nhãn đủ, thiếu input
  thực tế; Gaussian gộp + nhãn proxy chỉ 0,767 (m4b) và hại trên backbone đã train bằng Gaussian (gpool 0,664).
- **Tầng quyết định private trôi về local trong mỗi epoch dài; teacher DML cũng trôi** (probe 15, 1 epoch từ m4b ep15, 20 client):
  `cont` đạt 0,758 ở bước 64 rồi về 0,727; private khởi tạo bằng proxy rơi 0,08 chỉ với CE local. Teacher = proxy gộp **cố định cả epoch**
  giữ ranh giới (head 0,770, +0,043); thêm input thật client khác → 0,842 ≈ proxy, Gaussian gộp → 0,792. Mẫu tinh chỉnh bằng proxy
  (CE + BN) chỉ +0,01…+0,02 so Gaussian (probe 17). ⇒ đòn bẩy: mục tiêu private (teacher cố định) và input phủ client khác, không phải
  kênh KD mới. Teacher cố định mang prior toàn cục ⇒ luật head (vòng 15).
  **Vòng 15 xác nhận từ đầu tới ep15 (s42):** frz +0,037, gpool_frz +0,038, gpool_frz_gkd **+0,069** (0,803; benign +0,20, tệ nhất +0,13),
  lợi không teo theo epoch; cùng weights đọc gp chỉ 0,64–0,68 ⇒ luật phải theo nguồn teacher.
  Khoảng còn lại (gkd 0,803 → refit gộp 0,855) chủ yếu là bias theo lớp **riêng từng private** (probe 18–19): bias proxy không chuyển
  sang private; bias CE/bCE federated chỉ ≤ +0,02; F1 oracle +0,045 cần nhãn toàn cục qua chính private ⇒ không làm được với R2.
- **Chọn luật quyết định bằng đo trên chính weights, theo mọi góc client** (owner 18/10): họ frz head hơn gp ở global, local F1/acc và client
  tệ nhất (gkd global +0,126, local acc +0,406); DML cũ gp hơn ở global, head hơn local acc. Đo rò prior local bằng độ dốc over-prediction
  theo log π_k − log π_g (`prior_leak.py`): m4b head +0,74, gkd +0,13, gp trên frz −0,41 ⇒ head gkd vẫn mang chút prior local — sửa ở mục
  tiêu huấn luyện (CE logit-adjusted) thay vì luật hậu kỳ. [Design §Luật](docs/methods/full-drift/design.md).
- **Sửa drift proxy (FedProx, FedDyn, SCAFFOLD) không giúp client ở full; hại client tệ nhất.** FedDyn −0,032/−0,034
  (tệ nhất −0,18/−0,15), SCAFFOLD LR tốt nhất −0,007…−0,010 (tệ nhất −0,13). Drift ở đây phần lớn là thích nghi local có ích.
- **Teacher đã mang prior toàn cục thì không áp luật gp lần nữa.** Frozen teacher (proxy gộp): private head 0,714/0,698 > gp
  m4b nhưng gp sập 0,587/0,577 vì cộng `log π_g − log π_k` hai lần. Luật triển khai phải khóa theo nguồn teacher.

## Những cơ chế làm train kém hoặc gây kết luận sai

- **20/10 final startup crash:** test `--smoke --final` vẫn đọc pilot fixture, không chứng minh đường production final-only.
  Gaussian legacy còn hash cứng pilot manifest ⇒ r24 zc/zi crash trước round1. Test bắt buộc: không tồn tại pilot,
  Gaussian đọc final manifest; đối chiếu context pilot với legacy. Notebook phải lan truyền exit code subprocess,
  không biến `rc=1, epochs=0` thành Kaggle COMPLETE. Source đã push giữ nguyên; sửa ở tag mới.
- **20/10 upload source:** `datasets create` có thể exit1/JSON parse error dù server đã tạo dataset ready.
  Đọc status, verify manifest names/sizes trước retry; không upload/push trùng từ exit code đơn lẻ.

- **Full ≠ pilot về động học.** Full mỗi round = 1 epoch ≈4.200 bước/client; Gaussian/local đạt đỉnh test ở r1 rồi giảm
  (học thuộc receiver), proxy bão hòa từ r3–r5 khi LR còn ~1e-3 (FullDrift vòng 5: cũng do học thuộc receiver, m4b bỏ được). Pilot2,5% (~100 bước) không tái hiện; kết luận về
  drift/hội tụ từ pilot (vd FedAvg > DML proxy +0,027) có thể đảo ở full (repo anh em NILM proxy 0,823 > AFPHA 0,775).
  Macro F1 còn bị khối {benign, dataReplay, positionMirroring, timeDelay} chặn; lỗi dịch chuyển trong khối.
  [Chẩn đoán](docs/archive/research/plateau-diagnosis-2026-10-15.md).

- **Trước khi đổi lượng/trọng số/độ thật của mẫu tổng hợp, đo nó bất đồng teacher ở cặp lớp nào** (probe 20, 21/10): Gaussian gkd bất đồng 12,7 %
  nhưng ở cặp tốc độ (cos 0,09 với lỗi thật của private, nằm ở khối benign) ⇒ giải thích g512/gk2/cop bão hòa. Ưu tiên **trong** Gaussian (top-KL,
  đối kháng, lớp/cặp) không đổi chỗ đó; nhiễu liên tục ở feature atom là thủ phạm. Ưu tiên chỉ có tác dụng với bộ sinh bám dữ liệu thật (lai hai dòng
  own theo cặp bất đồng: cos 0,55). Chỉ số này là điều kiện cần; utility vẫn phải train (vòng 24). [Probe 20](docs/methods/full-drift/results.md).
- **Mẫu tổng hợp "thật hơn" chưa chắc giúp client.** Gaussian copula (biên histogram gộp) nâng teacher đồng ý trên mẫu 0,48 → 0,67 suốt
  15 epoch nhưng private chỉ +0,003 (vòng 18 sa); gấp đôi số mẫu/bước (g512) +0,001, KL ×2 0,000. Đo dự báo bằng chính utility
  (F1 client) trên run ngắn, không suy từ độ khớp phân phối hay probe phủ input.
- **Teacher snapshot cố định (frz) khuếch đại nhiễu kênh nén.** Proxy gộp qua q2 dao động mạnh (DML thường: 0,48–0,78 giữa epoch) nhưng
  private DML vẫn ổn vì teacher là proxy local được train lại liên tục. Với frz, private học thẳng từ snapshot gộp bị lượng tử ⇒
  q2_frz_gkd private head 0,675 (ep6) → 0,577 (ep15), proxy 0,479 (−0,354 vs gkd dense). Dùng frz thì snapshot phải gần dense;
  kênh nén phải kiểm riêng với frz: **q4 (int4) qua** (Δ +0,001, byte proxy ÷7,6), q2 không.
  [Kết quả vòng 16–18](docs/methods/full-drift/results.md).
- **Lượng tử chiều xuống làm tròn gần nhất với scale theo max hàng là nút thắt của kênh ít bit, EF không cứu được.** q2 có EF cả hai
  chiều vẫn sụp (vòng 21 wc: private 0,313; phần dư server/|exact| 0,27 ep1 → 0,71 ep15, q4 chỉ 0,008): 3 mức {−1,0,1}·max đưa mọi phần
  tử < max/2 về 0, replica client tụt lại mãi. Chiều lên làm tròn ngẫu nhiên (không lệch, trung bình K client) chịu nhiễu tốt hơn. Theo dõi
  `q_residual_rel` từ ep1–2: tăng dần = kênh không theo kịp. Nén không mất mát (LZMA2) cho int4 lấy lại ~20 % byte miễn phí (trùng bit).
  **Sửa 20/10 (vòng 22):** chiều lên 2-bit (làm tròn ngẫu nhiên + EF client, xuống int4) cũng hỏng (proxy 0,45) ⇒ "chiều lên chịu ít bit hơn"
  sai. Nguyên nhân chung là **scale theo max hàng**: update proxy thật có crest max/RMS ~3,2 ⇒ 93 % phần tử < max/2; sai số tương đối q2 max
  0,83 (gần nhất) / 1,45 (ngẫu nhiên), q3 0,40 / 0,58 (PASS). 3-bit hai chiều PASS và rẻ hơn int4 33 %. Trước khi chọn số bit, đo sai số tương đối
  của bộ lượng tử trên update thật và so với mức đã PASS. [Kết quả vòng 20–22](docs/methods/full-drift/results.md).
- **q3_zl đã lặp đủ hai seed:** residual ep15 ~0,021–0,022, private head noninferior (s42 −0,0024; s43 +0,0032), byte proxy ~66,7–66,8 MB/15 epoch. 2-bit RMS ổn định nhưng mất F1 −0,0133 ⇒ giữ q3, không nới cổng. [Vòng 23](docs/methods/full-drift/results.md).
- **Logit adjustment cho client lệch nhãn: τ=1 với prior gộp là đủ.** τ 0,5/1,5, đích prior đều, la trên CE proxy, KD×2 đều nằm trong dải
  nhiễu quanh τ=1 (vòng 20). Đích prior đều tăng proxy nhưng làm head bỏ lớp benign phổ biến (−0,042) ⇒ đích phải là prior thật của tập đánh giá.

- Sketch-space error feedback không mặc định co: collision có thể khuếch đại residual và parameter rồi BN overflow.
  Đo decoded/residual norm, tính co và full-epoch stability; không suy toàn họ FetchSGD/EF bị bác bỏ từ adaptation này.
  Repro15/10 tối giản CPU FP32 cho residual2×, không cần BN/AMP. Line-search kiểm **wire-decoded** patch
  chặn overshoot trong80ca đại số. Repaired hai seed terminal: smallr20 vẫn qualityFAIL;
  mediumr17/r14 còn thiếu horizon, last5 group/gp phẳng/giảm và xa own. Decoded patch chỉ đổi0,638%/7,818%
  tọa độ small/medium, không đủ để quy lỗi duy nhất cho collision/ties. Co norm không chứng minh teacher
  fidelity hoặc giữ ranh giới. [Sửa, forensic và cổng cũ](docs/methods/proxy-countsketch-repair/builds/contractive-ef-pilot/results/report.md).
- Gaussian liên tục và real có feature atom/missing có thể khác support; ratio fit cao không chứng minh ratio hữu ích.
  Clip/class-normalize không sửa support hoặc tạo ESS tốt. Clip floor có thể triệt input score correction;
  finite burn-in không chứng minh MALA mixing. [Ratio](docs/archive/methods/gaussian-density-ratio/README.md), [MALA](docs/archive/methods/gaussian-mala-transport/README.md).
- Support/fidelity PASS không tự tăng F1. Joint mask có lợi tương tự control full66 chưa chứng minh geometry mechanism.
  Khoảng sai số đại số rộng không chứng minh faithful reconstruction, nhất là missing/coincident và denominator gần0.
  [Geometry](docs/archive/methods/geometry-primitive-moments/README.md).
- DAGSNet bounded-logit classifier trên R66 không tự là density chuẩn hóa: cần base/domain rõ ràng.
  Gaussian-base sửa tính hợp lệ nhưng không tự sửa sampler quality/runtime. [Energy](docs/archive/methods/dagsnet-joint-energy/README.md).
- gp theo prior local có thể sửa quá tay khi KD làm private gần global; benign F1 có thể giảm dù head tăng.
  Định nghĩa prior hiệu dụng từ loss trước run, primary rule cố định. BoundaryCells khóa target-mass gp;
  các full cũ giữ gp đã chốt, không đổi headline để cứu cổng.
- Tail-risk bằng histogram có lỗi lượng tử hóa: quantile ở mép bin có thể làm tail rỗng và chỉ đổi scale loss.
  Kiểm tail fraction/membership thực tế, overflow và cohort trước utility; weighted target mass cần audit nhưng
  x-dependent weighting không tự có bảo đảm label-shift. ClassTail cả hai seed: probe r2PASS nhưng utilityr20FAIL;
  pooled không hơn local/own/shuffle/class-mass, kém DML0,1967/0,1917. Probe chỉ loại tail rỗng/calibration hỏng,
  không chứng minh hard-example knowledge hữu ích. [Gates đã audit](docs/methods/class-tail-risk/builds/class-tail-pilot/results/gates.json).
- BoundaryCells P2 phủ95,65% group probe nhưng target entropy0,915/table F1≈0,129; coverage không thay
  discrimination. Target tĩnh không nhận tri thức mới khi thêm round. P4direct/P2-P4teacher r20 hai seed cũng FAIL
  so Gaussian; learned teacher head≈0,144–0,260 phù hợp table yếu. Không nối cả bốn adaptation;
  không suy cả họ bị bác. [Actual payload forensics](docs/methods/boundary-cell-counts/builds/boundary-cell-pilot/direct-target-forensics-20261015.md).
- Tổng CEreal+CEsynthetic cao chưa chứng minh real CE cao hoặc đã chống học thuộc. Tách loss và dùng holdout.
  Ngược lại, loss FL tăng khi LR gần0 có thể vì weights tải xuống không được fit thêm, không tự là trôi.
- EMA không gồm toàn head; BN phải khớp backbone EMA và head triển khai fit riêng. Recipe local có lợi chưa là lợi FL:
  so 2×2 +own, interaction báo riêng. [EMA thiết kế](docs/methods/ema-deployment-recipe/design.md).
- Hybrid tốt hơn thành phần mạnh nhất nhưng interaction âm; cộng hai cơ chế không có nghĩa synergy.
  Full dữ liệu sớm có tín hiệu tốt, chưa r50. Gaussian đã giảm metric qua một số round dù còn lợi so own;
  thêm epoch không tự tăng performance. [Hybrid full](docs/methods/gaussian-proxy-hybrid/builds/hybrid-full/results/feasibility-20261014.md).
- Máy local ngủ (WSL) làm poll nền dừng, Kaggle vẫn chạy. Sau khi thức đừng để watcher mở collector cho mọi session cùng
  lúc (mỗi audit đỉnh ~4 GB): chia nhóm session rời nhau cho 2–4 tiến trình `collect_fd.py --only`, pull ~20 s/session.
- **Sửa 20/10: CLI hiện tại đọc được log live bằng `kaggle kernels logs <owner/slug> --follow` (SSE).** Đã đọc W&B auth và
  `Prepared real pilot … old cut: 0` khi r24 vẫn RUNNING; lệnh không `--follow` có thể vẫn rỗng. Dùng account-scoped, đọc có timeout;
  exit 124 do hết thời gian đọc không phải kernel ERROR. Các ghi chú cũ "chỉ có W&B khi đang chạy" không còn đúng ở môi trường này.
  [Bằng chứng startup](docs/archive/operations/final-r24-2026-10-20/final-startup-2026-10-20.json). Log/status chưa thay terminal artifact audit.
- Collector có thể thoát 0 mà chưa pull nếu thiếu `PUSHED`. Kiểm `AUDITED.json` và completion cụ thể; chỉ dựng lại marker sau khi
  xác nhận đúng kernel từ status account-scoped, rồi vẫn đối chiếu executed source trước audit (ya-v2, 20/10).
- SHA của `.git/index` khác chưa có nghĩa nội dung staged đổi. Kiểm staged tree và HEAD riêng; ghi cả byte identity và content identity,
  không khôi phục index chỉ để ép SHA trùng snapshot cũ (20/10).
- **Harness Kaggle bị giới hạn bởi CPU, không phải GPU** (T4 ~12 ms compute vs 26–27 ms/bước gpool, 4 vCPU cho 2 arm).
  Mỗi `.item()`/`get_scale()`/`assert torch.isfinite(t)`/index-put bằng tensor là một lần đồng bộ host; deterministic mode
  mặc định lấp NaN mọi `empty` (~500 kernel/bước). Gỡ chúng: −20 %/bước, bit-exact. Mọi tối ưu harness phải kiểm bit-exact
  với source trước (test local) để đối chứng đã audit còn dùng được.
- **Chấm DAGSNet bị giới hạn bởi kernel conv1d, không phải FLOP** (21/10): 4,4 MFLOP/dòng, conv L=11 kênh hẹp ⇒ T4 chỉ ~1,5 TFLOPs,
  30–34 s/lượt 10,76M dòng. cuDNN autotune/batch lớn chỉ −5…−24 %; viết lại forward thành GEMM trên (dòng, vị trí, kênh), fold BN eval,
  conv k = một GEMM ra k×out rồi cộng dịch vị trí ⇒ 18–19 s (1,7×), khớp fp32 tới 4e-7, fp16 lệch 0,02 % dòng sát nút (lớp tolerance,
  owner duyệt). Đo trên T4 thật trước khi chọn: im2col nhanh như bản cộng dịch trên RTX 3050 nhưng chậm hơn 25 % trên T4.
  [fd_eval_fast.py](methods/full-drift/builds/fd-probe/source/proj/fd_eval_fast.py).
- Tiến trình khởi động độc lập không dùng `Connection.send` cho tensor: ForkingPickler của torch chia sẻ tensor qua
  resource_sharer với authkey của tiến trình gốc ⇒ AuthenticationError; dùng `pickle.dumps` + `send_bytes`.
  Đừng chạy job nặng khác cùng lúc với test nhiều tiến trình (guard RAM dừng test, không phải lỗi code).
- Cùng số model/SGD step không bảo đảm cùng walltime: dummy/eval-BN có overhead thật.
  Không dùng dummy-inflated denominator để gọi phương pháp rẻ; chưa profiling thì không chốt nguyên nhân duy nhất.

## Đánh giá để rút đúng bài học

- **Mẫu tổng hợp học KL từ teacher chỉ sửa được lỗi teacher đúng.** Trước khi thiết kế bộ sinh, tách lỗi private thành "dạy được"
  (teacher đúng) và "chung" (teacher cũng sai). Headline ep15: chung 77 %, dạy được 23 % (trần +0,036) và đã bị la/kdpx ăn một nửa ⇒ đổi
  vị trí/lượng/độ thật/ưu tiên mẫu (g512, gk2, cop, xo) chỉ còn ≤ +0,008. Muốn lợi lớn hơn phải làm nguồn tri thức tốt hơn, không phải bộ sinh.
- **Ưu tiên theo bất đồng hướng vào chỗ cả hai cùng yếu.** 50 % bất đồng là benign↔timeDelay, teacher đúng ~50 % ⇒ xo sửa 55 / làm hỏng 221
  dòng timeDelay; lợi chỉ ở cặp teacher đúng ≥ 90 %. Ưu tiên phải có độ tin cậy (nhãn own cho ρ, 0 byte), không chỉ khối lượng bất đồng.
- **Chỉ số phân bố (cos/align, syn_accept) không dự báo utility** (cop align ×3 → +0,003; xo align ×6 → +0,0076). Probe trước train phải
  đo trần reachable (refit/oracle trên chính tri thức sẽ truyền), không chỉ "mẫu có đúng chỗ không".
- **Feature đã đủ, thiếu ở tầng quyết định của teacher:** refit head teacher trên dòng thật có nhãn gộp 0,842 → 0,919 (timeDelay 0,27 → 0,50);
  thống kê đặc trưng theo lớp cộng được (CCVR) → 0,893, phần lỗi private teacher dạy được 21 → 47 %.
- **Private chững khi teacher còn tăng** (run cuối r12 → r27; gkd:50): screening ep15 đo ở vùng teacher ≈ private, đánh giá thấp kênh truyền.
  [Phân tích](docs/methods/full-drift/results.md).
- **So byte cùng đơn vị: mỗi client, cùng horizon, từ `result.json` đối chứng đã audit.** Hồ sơ cv (21/10) lấy byte một client so với kênh q3
  cộng 20 client ⇒ ghi cvf +48 % trong khi thật ×6,8 (15 epoch); owner chọn arm theo số sai. Báo tổng/client cả 15 epoch và 50 round, và theo round.
- Gate tolerance tuyệt đối trên logit vỡ khi logit lớn dần theo round: sai số fp16 tỉ lệ với độ lớn. Đặt ngưỡng tương đối
  (sai số / max|logit|) và hiệu chỉnh trên round muộn nhất sẽ chấm, không chỉ round đầu (r27 0,25 hiệu chỉnh r1–4, vỡ ze r34).
  [final-pipeline §Bộ chấm nhanh](docs/operations/final-pipeline.md).
- Fixture được fingerprint phải chính là fixture driver đọc. `--final` trước 20/10 chọn dữ liệu final ở Run nhưng main hash pilot; chưa có run final production. Dùng chung `fixture_path`, và evaluator từ chối manifest khác training. Smoke dùng pilot không phát hiện được sai đường dẫn production.

- Train/partition đã z-score, test raw áp scaler đúng một lần. Dữ liệu sắp theo lớp: sample ngẫu nhiên seeded,
  không prefix tùy tiện. Không chia lại partition hoặc dùng official test chọn HP/round.
- Holdout ngẫu nhiên theo dòng lạc quan hơn temporal/receiver-group. Receiver noise/profile và Sybil first-session
  là shortcut; benign từ runSybil có source bias. [Dataset](knowledge/dataset.md).
- positionMirroring chưa chạm trần thông tin; Gaussian/DML private chưa học được không có nghĩa dữ liệu vô ích.
  timeDelay thiếu dấu hiệu trong content theo generator: kiểm trần trước quy lỗi thuật toán.
- Probe phải dùng dữ liệu chưa train và đủ lớp/cohort. Local derivative/coverage/inference prefix là điều kiện cần,
  không là utility, full coverage, leakage hoặc physical consistency proof.
- **"Proxy hơn private" chỉ đúng khi mọi client đánh giá trên cùng test toàn cục.** Đánh trọng số lại confusion theo
  prior nhãn của từng client (full DML r23): private head local macro F1 0,677 ≈ proxy 0,675, local accuracy 0,818 so 0,771;
  gp tốt cho global nhưng kém cho local. Trước khi so phương pháp "cho client", chốt định nghĩa metric (global/local)
  và báo client tệ nhất. Mới mô phỏng lệch nhãn, chưa lệch receiver. [Bảng](docs/archive/research/client-model-directions-2026-10-16.md).
  gp có thể làm hỏng client tệ nhất ở góc local: m4b ep15 s43 private gp local acc min 0,642 (head 0,776). Góc local dùng head.
- Khóa gp/cổng/seed/controls trước chạy, so cùng round và một yếu tố mỗi arm. Mean-client metrics khác pooled confusion;
  lưu predictions, per-client CM, đủ10metric/per-class/benign để kiểm lại.
- CrossClientRank CE-r2: residual/delta macro cực lớn có thể do class delta gần0 và fit/test extrapolation;
  không mặc định SVD lỗi. Diagnostic cùng mẫu tái hiện energies, condition≈5,2; class5/client17 có coefficient lớn
  và vài test record chi phối. Probe gate PASS không định lượng tri thức hoặc utility; báo energy tuyệt đối,
  singular values và class influence trước khi diễn giải. [Probe/diagnostic đã audit](docs/methods/cross-client-ranking/builds/rank-probe/results/report.md).
- Reference cùng recipe/seed/data ở build khác là cùng phép tính: full DML/DML-own trùng bit-exact giữa hai build,
  Gaussian/local lệch ≤0,008. Tái dùng audited reference thay vì chạy lại; pilot DML không bit-exact giữa build (≈0,002)
  nên gate cần chênh ≥0,01 mới nên dựa reference khác build. [Sổ lỗi](docs/operations/notebook-errors.md).
- Pilot ngắn không dự báo độ bền; full thường giảm lợi hoặc đổi dấu. Không kéo dài adaptation FAIL để tìm điểm thắng.
  Full một seed là xác nhận mô tả theo protocol, không paired confidence interval.
- Exact-resume fixture phải cover state của cả local/control và FL arm. CrossClientRank pooled replay PASS
  đã bỏ sót CE empty-byte state, khiến production verifier dừng dù r3 hợp lệ;
  [repro và recovery](docs/operations/notebook-errors.md) giữ nguyên weights/source, không cần SGD lại.
- Lợi mutual learning khi chưa nhận snapshot RSU không được gọi là lợi FL. Tính đủ bootstrap/prior/header/ACK/retransmission;
  deadline-cut trả phí nhưng không commit, chỉ activate snapshot coherent. [Contact](docs/archive/methods/contact-fragmentation/README.md).
- **"FL" phụ thuộc giả định SecAgg khi chia sẻ thống kê lớp.** Bootstrap Gaussian gửi Σx, Σxxᵀ theo lớp; `SecureSum` chỉ giả lập. Có ô client×lớp
  chỉ 4–10 dòng ⇒ không SecAgg thật là gần lộ record. q4/q2 scale riêng từng client không cộng được dưới SecAgg. Mọi kênh mới phải ghi:
  ai thấy gì nếu không có SecAgg, ô nhỏ nhất, có cộng tính được không. [Kiểm toán](docs/research/federated-information-flow.md).
- R1 giới hạn giao tiếp và R2 khác kiến trúc phải kiểm riêng; unlimited-channel/DAGSNet-only PASS không giải quyết cả hai.
  Synthetic exception không tự là secure aggregation hoặc DP.

Khi bằng chứng mới bác bỏ bài học: sửa rõ phạm vi và dẫn artifact đã audit, giữ bản lịch sử. Cải tiến mới phải dự báo
performance từ bằng chứng, chỉ ra dấu hiệu có thể sai rồi kiểm ngắn trước khi tăng quy mô.
