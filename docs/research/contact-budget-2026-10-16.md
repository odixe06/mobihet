# Ngân sách contact R1 — bao nhiêu lần tổng hợp/epoch là thực tế (16/10/2026)

Owner 16/10 ~16:45Z: trước khi chọn tần suất tổng hợp (sub4/sub16), phải mô phỏng contact. Tài liệu này là bước 1:
phân tích không train, tách phần **đo được** và phần **giả định** (quét, không chốt). Tool: [contact_budget.py](../../scripts/contact_budget.py).

## Bằng chứng đo được

| đại lượng | giá trị | nguồn |
|---|---|---|
| tốc độ median theo xe | highway_2 18,5, highway_7 12,5, urban_2 12,7, urban_7 11,3 m/s (median chung 12,6) | InTAS ground truth (test files có `spd`) của VeReMi NextGen [1] |
| vùng mô phỏng / thời gian xe trong vùng | ~2×2 km; trace median 37 s (highway_2) – 130 s (urban_7) | như trên |
| 1 epoch local / client trên T4 | median 75,5 s (DML private + proxy, 20 client) | marker audited `fd_dml_m4b_q4` ep6 |
| một lần trao đổi (proxy xuống + lên) | q4 417 kB; dense ≈ 3,19 MB (×7,65) | `q_bytes_down` marker audited; AntiMem |

Trace InTAS chỉ phủ vài phút trong vùng 2 km nên **không đo được khoảng giữa hai contact** từ dữ liệu: số contact/epoch
phải dựa vào giả định mật độ RSU dọc tuyến.

## Cơ chế và kết quả (giả định được quét)

contacts/epoch = slowdown × T_epoch / (spacing / v); thời gian một contact = (π r / 2) / v (dây cung trung bình khi
tuyến cắt đĩa phủ sóng). Slowdown = epoch trên máy xe / epoch trên T4.

| RSU cách nhau | slowdown 1× | 5× | 20× |
|---|---:|---:|---:|
| 500 m | 1,9 | 9,5 | 38,0 |
| 1000 m | 1,0 | 4,8 | 19,0 |
| 2000 m | 0,5 | 2,4 | 9,5 |
| 5000 m | 0,2 | 1,0 | 3,8 |

| bán kính | giây/contact | rate | dung lượng | q4 vừa | dense vừa |
|---|---:|---|---:|---|---|
| 150 m | 19 | 80 kB/s | 1,50 MB | 3,6× | **0,5×** |
| 300 m | 37 | 80 kB/s | 2,99 MB | 7,2× | **0,9×** |
| 300 m | 37 | 750 kB/s (6 Mbps) | 28,1 MB | 67× | 8,8× |
| 700 m | 87 | 80 kB/s | 6,99 MB | 17× | 2,2× |

Giả định quét: bán kính 150–700 m (đo 802.11p ngoài thực địa ~138–700 m ở 6 Mbps [2], thiết kế tới ~1000 m [3]);
rate 80 kB/s (biên trên simulator cũ `mh/mobility.py`, [core/rsu_contact.py](../../core/rsu_contact.py)) và 750 kB/s
(PHY 6 Mbps); đo throughput/contact V2I thực tế ở [4]. Nghiên cứu FL xe cộ cũng coi thời gian lưu trú tại RSU là ràng buộc
chính và lọc/gán trọng số theo nó [5][6].

## Đọc

- **Byte không phải nút thắt khi dùng q4**: một lần trao đổi q4 vừa một contact trong mọi giả định (≥3,6×); dense có thể
  không vừa khi link chậm (80 kB/s, r ≤ 300 m). ⇒ q4 là điều kiện cần cho R1, lợi utility đã kiểm (noninferior).
- **Số contact/epoch mới là nút thắt**: sub4 cần ≥4 ⇒ RSU ≤1 km với slowdown ≥5×, hoặc ≤2 km với ≥20×; sub16 cần ≥16 ⇒
  chỉ RSU ≤1 km với slowdown ≈20× (hoặc 500 m, ≥~8×). Máy xe chậm hơn T4 làm **nhiều** contact/epoch hơn.
- Giới hạn: đường thẳng, không mất gói/MAC, không tính chờ đủ cohort 20 client cho tổng hợp đồng bộ (mỗi lần tổng hợp
  phải đợi client có khoảng contact dài nhất), không tính thời gian thu dữ liệu.

## Việc cần owner chốt (kiểm được)

Chọn kịch bản triển khai (mật độ RSU dọc tuyến, slowdown máy xe) ⇒ tần suất tối đa k = ⌊contacts/epoch⌋.
Sai nếu: với kịch bản đã chọn, mô phỏng cohort (bước 2: 20 client, contact ngẫu nhiên theo khoảng cách RSU/tốc độ,
tổng hợp khi đủ upload) cho tổng hợp/epoch < k ⇒ hạ k. Bước 2 dùng lại `core/rsu_contact.py` (phí frame/ACK, deadline).

## Tài liệu tham khảo

1. Lobo, Neumeier, Fernandez, Facchi. *InTAS — The Ingolstadt Traffic Scenario for SUMO*. SUMO 2020. [arXiv:2011.11995](https://arxiv.org/abs/2011.11995). VeReMi NextGen dùng InTAS (README nguồn `veremi_nextgen/docs`).
2. Báo cáo kỹ thuật GTA/UFRJ (ITSC 2018) về tầm 802.11p theo modulation, ~700 m ở 6 Mbps — chưa đọc full text (PDF không trích được). [itsc18.pdf](https://www.gta.ufrj.br/ftp/gta/TechReports/itsc18.pdf)
3. ComNets RWTH Aachen (EW 2007): PHY/MAC 802.11p thiết kế cho cự ly tới ~1000 m; thời gian contact phụ thuộc cự ly. [ew2007.pdf](https://www.comnets.rwth-aachen.de/downloads/publications/ew2007.pdf)
4. Lin et al. *802.11a vs 802.11p V2I field study* (2012): contact duration, loss, throughput theo khoảng cách OBU–RSU. [EUDL](https://eudl.eu/doi/10.1007/978-3-642-29222-4_39)
5. *Mobility-Aware Multi-Task Decentralized Federated Learning for Vehicular Networks*. [arXiv:2503.06468](https://arxiv.org/abs/2503.06468) (tổng quan sojourn-aware aggregation tại RSU).
6. *Mobility-Aware Decentralized FL with Joint Optimization of Local Iteration and Leader Selection*. [arXiv:2503.06443](https://arxiv.org/abs/2503.06443).
