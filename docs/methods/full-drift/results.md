# FullDrift — kết quả seed42 epoch1–6 (audited 15/10 ~21:30Z, sàng lọc một seed)

Contract khóa trước push: [design](design.md). Full 100% train80% trừ holdout nội bộ (train 40,0M dòng),
group/old 65.536 dòng; test chỉ epoch 6, mô tả. Sessions `annhannh331/mobihet-fd-s42{a,b}-20261015-v2`,
`anhtruong1909/mobihet-fd-s42c-20261015-v2`, 8/8 arm đủ epoch 6; audit executed source, input/fixture SHA,
holdout manifest (group overlap 0), marker chain, weights epoch 6 strict/finite, predictions → 10 metrics.
Bảng đủ: [report](results/report.md) · [gates.json](results/gates.json) · [10 metrics](results/metrics-all.csv) ·
[per-class](results/per-class-f1.csv).

## Epoch 6, group holdout (Δ so đối chứng cùng seed)

| arm | primary | Δ | benign | Δ benign | drift | test (mô tả) | cổng |
|---|---:|---:|---:|---:|---:|---:|---|
| fd_dml (đối chứng) | 0,7631 | — | 0,565 | — | 0,384 | 0,7715 | — |
| **fd_dml_sub4** | **0,7891** | **+0,026** | 0,595 | +0,030 | 0,179 | 0,7899 | **có triển vọng** |
| fd_dml_prox (μ=0,01) | 0,7624 | −0,001 | 0,662 | +0,098 | 0,076 | 0,7510 | FAIL primary |
| fd_dml_avgm (β=0,9) | 0,0537 | −0,709 | 0,000 | −0,565 | 0,295 | 0,0479 | FAIL, phân kỳ từ ep2 |
| fd_fedavg | 0,7724 | +0,009 vs DML | 0,525 | — | 0,385 | 0,7655 | mô tả |
| fd_fedavg_sub4 | 0,8059 (ep5 0,8197) | +0,034 vs FedAvg | 0,464 | −0,061 | 0,181 | 0,7877 | FAIL guard benign |
| fd_gaussian (private gp) | 0,5882 | — | 0,297 | — | — | 0,6074 | — |
| **fd_gaussian_swappos** | **0,6404** | **+0,052** | 0,427 | +0,130 | — | 0,6540 | **có triển vọng** |
| private_ema Gaussian / swappos | 0,6126 / 0,6768 | +0,024 / +0,036 | | | | | **có triển vọng** |
| private_ema DML arms | | −0,026…−0,040 | | | | | FAIL |

## Đọc

- **Drift là nút thắt thật ở full.** Ở epoch 1, đối chứng DML proxy chỉ đạt 0,187, còn sub4 đạt 0,751. Đến epoch 6,
  sub4 vẫn hơn +0,026 (FedAvg +0,034) với cùng compute và cùng LR. Drift tương đối giảm từ 0,38 xuống 0,18.
  Cái giá là byte gấp 4 lần mỗi epoch; QProxy q4 (pilot PASS, nhỏ hơn 7,65 lần) bù được, ghép lại còn ≈0,52 lần dense.
- FedProx giảm drift 5 lần nhưng macro F1 không đổi. Tuy vậy, benign giữ tốt hơn rõ (+0,098): đây là ứng viên để ghép với sub4.
- FedAvgM β=0,9, η=1 phân kỳ khi update local đã lớn (4.200 bước). Không dùng nếu chưa giảm server LR.
- FedAvg thuần ≈ DML proxy (+0,009). KD với private không cải thiện model triển khai ở full.
- Lỗi còn lại vẫn nằm trong khối {benign, dataReplay, positionMirroring, timeDelay}. FedAvg sub4 kéo positionMirroring
  0,068→0,510 nhưng benign 0,525→0,464: lỗi chỉ dịch chuyển trong khối.
- Gaussian ở full: swappos ngăn suy giảm sau epoch 1 (0,635→0,640, đối chứng 0,617→0,588); ghép EMA được 0,677.
  EMA hại private DML vì private DML còn đang tăng qua các epoch.
- Một seed, epoch 6/50: chỉ là sàng lọc. Lợi thế của sub4 có giữ tới r50 hay không thì chưa biết.

## Vòng 3 — mục tiêu client (audited 16/10 ~08:40Z)

Sessions `nguyenminhtran060113/mobihet-fd-s42{g,h}-20261015-r3-v1` COMPLETE, AUDITED (executed source, input/stage SHA,
data manifest, marker chain, ep6 weights strict/finite, predictions → confusion → 10 metrics). Seed 42, group ep6,
primary **private gp** (owner 16/10), đối chứng `fd_dml` vòng 1. Cổng khóa trước push ([design §Vòng 3](design.md)).

| arm | private gp | Δ | benign Δ | proxy head (Δ) | local F1 / acc private gp | client tệ nhất | test proxy | cổng |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fd_dml (đối chứng) | 0,6655 | — | — | 0,7631 | 0,637 / 0,794 | 0,595 | 0,7715 | — |
| **fd_dml_m4b** | **0,7010** | **+0,036** | **+0,113** | **0,8146 (+0,052)** | 0,683 / 0,854 | 0,617 | 0,8116 | **PASS** |
| **fd_dml_swappos** | **0,6985** | **+0,033** | **+0,107** | **0,8121 (+0,049)** | 0,681 / 0,845 | 0,613 | 0,8099 | **PASS** |
| fd_dml_ntd | 0,6236 | −0,042 | −0,034 | 0,7645 (+0,001) | 0,610 / 0,768 | 0,400 | 0,7695 | FAIL |
| fd_dml_q4 | 0,6613 | −0,004 | 0,000 | 0,7651 (+0,002) | 0,633 / 0,789 | 0,577 | 0,7716 | PASS noninferior |

Đọc:
- **Bỏ khối feature receiver (m4b/swappos) là kỹ thuật mạnh nhất ở full, cho cả client lẫn proxy**, 0 byte/contact thêm
  (R1) và chỉ là tiền xử lý phía client (R2). Private +0,036 ≈ sub4 (+0,035) nhưng sub4 tốn 4× contact; proxy +0,052
  gấp đôi sub4 (+0,026). Pilot r20 từng cho proxy FAIL (−0,012…+0,006): full khác pilot (học thuộc receiver chỉ lộ ở
  4.200 bước/epoch). Benign tăng mạnh (+0,11) — khối {benign, …} là nút thắt cũ.
- FedNTD hại private (−0,042) và client tệ nhất (0,400): bỏ lớp đúng khỏi KL làm mất tín hiệu thầy ở lớp đúng;
  lệch prior không phải nguyên nhân chính của khoảng private–proxy.
- q4 ở 1 contact/epoch không thua (private −0,004, proxy +0,002), byte nhỏ 7,65× ⇒ R1 dùng được.
- Một seed, ep6/50: sàng lọc. Cần horizon (ep15) và seed 43 trước khi đổi headline.

## Vòng 2 — sub4/sub16 + q4, prox, horizon ep15 (audited 16/10 ~16:40Z)

Sessions `anhtruong1909/…-fd-s42{d,e}-20261015-r2-v1`, `annhannh331/…-fd-s42f-20261015-r2-v1` COMPLETE, AUDITED như vòng 3.
Cổng proxy khóa 15/10 ~22:00Z, cổng phụ private gp khóa 16/10 ~04:00Z ([design §Vòng 2](design.md)). Group, seed 42.
Replay ep1–6 của bốn arm `@e15` trùng vòng 1 tuyệt đối (Δ 0,0000) ⇒ đối chứng tái dùng hợp lệ.

| so sánh | proxy head Δ (benign) | cổng proxy | private gp Δ (benign) | cổng private | tuyệt đối (proxy / private gp / test proxy) |
|---|---:|---|---:|---|---|
| `fd_dml_sub4_q4` vs `fd_dml_sub4` | −0,003 (−0,016) | FAIL (benign) | −0,004 (−0,004) | PASS noninferior | 0,7863 / 0,6969 / 0,7931 |
| `fd_fedavg_sub4_q4` vs `fd_fedavg_sub4` | −0,007 (−0,014) | FAIL (benign) | — | — | 0,7991 / — / 0,7800 |
| **`fd_dml_sub16_q4` vs `fd_dml_sub4_q4`** | **+0,041 (+0,163)** | **PASS** | **+0,017 (+0,027)** | **PASS** | 0,8270 / 0,7139 / 0,8118 |
| **`fd_fedavg_sub16_q4` vs `fd_fedavg_sub4_q4`** | **+0,054 (+0,173)** | **PASS** | — | — | 0,8527 / — / 0,8148 |
| `fd_dml_sub4_prox` vs `fd_dml_sub4` | −0,008 (−0,026) | FAIL | −0,016 (+0,002) | FAIL | 0,7806 / 0,6848 / 0,7760 |
| `fd_fedavg_sub4_prox` vs `fd_fedavg_sub4` | −0,027 (+0,026) | FAIL | — | — | 0,7792 / — / 0,7699 |
| **`fd_dml_sub4@e15` vs `fd_dml@e15`** (ep15) | **+0,027 (+0,047)** | **PASS** | **+0,025 (+0,027)** | **PASS** | 0,7926 / 0,7211 / 0,7980 |
| **`fd_fedavg_sub4@e15` vs `fd_fedavg@e15`** (ep15) | **+0,041 (+0,111)** | **PASS** | — | — | 0,8142 / — / 0,7891 |

Quỹ đạo ep6→ep15 (group): `fd_dml` proxy 0,763→0,765 (phẳng từ ep4), private gp 0,666→0,696, **benign proxy 0,741 (ep3)
→0,358**; `fd_dml_sub4` proxy 0,789→0,793 (phẳng từ ep3), private gp 0,700→0,721. Client view ep15 private gp:
local acc 0,788/0,790, client tệ nhất local acc 0,432/0,393.

Đọc:
- **Tổng hợp thường hơn vẫn là đòn bẩy proxy lớn nhất; nén q4 trả giá nhỏ ở benign.** sub16_q4 hơn sub4_q4 +0,041 proxy,
  +0,017 private; byte/epoch ≈ 2,1× `fd_dml` dense (16 contact × 1/7,65), contact 16×. q4 ở sub4 làm benign −0,014…−0,016 (cổng proxy FAIL,
  private không thua): lỗi lượng tử cộng dồn theo số lần tổng hợp.
- **FedProx không giúp gì khi đã tổng hợp thường** (sub4_prox FAIL cả proxy lẫn private): drift không còn là nút thắt
  ở sub4 nên kéo về θ chung chỉ làm chậm học local.
- **Horizon: lợi sub4 giữ nguyên tới ep15 nhưng không lớn thêm**; proxy cả hai arm phẳng từ ep3–4, benign proxy giảm đều
  theo epoch ⇒ kéo dài DML không mask không có tiềm năng ở proxy; private gp còn tăng chậm (+0,03 trong 9 epoch).
- sub16_q4 private gp: client tệ nhất local acc 0,288 (head 0,673) ⇒ gp theo prior global sửa quá tay cho vài client;
  góc nhìn local nên dùng head.

## Vòng 4 — hidden KD và Gaussian m4b (audited 16/10 ~16:40Z)

Session `odixe0502/mobihet-fd-s42i-20261015-r4-v1` COMPLETE, AUDITED. Cổng private gp group ep6 ([design §Vòng 4](design.md)).

| arm | private gp | Δ | benign Δ | proxy head | local F1 / acc private gp | client tệ nhất | test primary | cổng |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fd_dml (đối chứng) | 0,6655 | — | — | 0,7631 | 0,637 / 0,794 | 0,595 | 0,7715 (proxy) | — |
| fd_dml_hkd | 0,6710 | +0,006 | +0,001 | 0,7661 | 0,643 / 0,797 | 0,588 | 0,7745 (proxy) | FAIL |
| fd_gaussian (đối chứng) | 0,5882 | — | — | — | 0,614 / 0,790 | 0,458 | 0,6074 (private gp) | — |
| **fd_gaussian_m4b** | **0,6553** | **+0,067** | **+0,142** | — | 0,679 / 0,866 | 0,490 | 0,6671 | **PASS** |

Đọc:
- **Hidden-state KD (FedKD) gần như không có tác dụng** (+0,006 < +0,010): thêm kênh 256-d không đóng khoảng private–proxy.
  Cùng với NTD (−0,042) ⇒ khoảng private–proxy ở full không do kênh KD hẹp hay lệch prior, mà do private học shortcut
  receiver trên dữ liệu local (m4b đóng +0,036 với cùng kênh KD). Không nối/quét trọng số HKD.
- **m4b cho Gaussian lớn nhất từ trước tới nay** (+0,067, pilot +0,025; swappos full +0,052); vẫn dưới DML m4b (0,7010).

## Vòng 5 — m4b horizon + ghép (audited 16/10 ~17:00Z)

Session `nguyenminhtran060113/mobihet-fd-s42j-20261015-r5-v1` COMPLETE, AUDITED; replay ep1–6 của `fd_dml_m4b@e15` trùng
vòng 3 tuyệt đối. Cổng [design §Vòng 5](design.md). Group; test chỉ epoch cuối, mô tả.

| so sánh | private gp (Δ, benign Δ) | proxy head (Δ, benign Δ) | test private gp / proxy | cổng |
|---|---|---|---|---|
| **`fd_dml_m4b@e15` vs `fd_dml@e15`** (ep15) | **0,7323 (+0,036, +0,144)** | **0,8510 (+0,086, +0,409)** | 0,7478 / 0,8322 (đối chứng 0,7164 / 0,7769) | **PASS** horizon |
| `fd_dml_m4b_q4` vs `fd_dml_m4b` | 0,7010 (+0,000, −0,000) | 0,8157 (+0,001, +0,010) | 0,7234 / 0,8125 | PASS noninferior |
| **`fd_dml_m4b_sub4_q4` vs `fd_dml_m4b`** | **0,7319 (+0,031, +0,021)** | 0,8485 (+0,034) | 0,7462 / 0,8277 | **PASS** |

Quỹ đạo `fd_dml_m4b` ep6→ep15 (group): proxy 0,815→0,851 (còn tăng, `fd_dml` phẳng 0,763), private gp 0,701→0,732,
**benign proxy giữ 0,76–0,78** (`fd_dml` rơi 0,74→0,36). Client view ep15 private gp: local F1 0,708, local acc 0,867,
client tệ nhất local acc 0,756 (`fd_dml@e15` 0,432).

Đọc:
- **Học thuộc receiver là nguyên nhân proxy DML chững và benign sụp theo epoch ở full**: bỏ 13 feature receiver/vị trí thì
  proxy tiếp tục học tới ep15 và benign không rơi. Lợi m4b lớn dần theo horizon (proxy +0,052 ep6 → +0,086 ep15).
- So mô tả với full headline cũ (test, khác holdout/round): m4b ep15 private gp 0,748 = `h_dml` r41 0,748; proxy 0,832 > 0,801.
- Ba kỹ thuật cộng được: m4b (0 byte) + q4 (byte 7,65× nhỏ, không mất) + sub4 (4 contact/epoch, private +0,031).
  m4b_sub4_q4 ep6 private gp 0,732 ≈ m4b ep15. Còn chờ seed 43 (vòng 6) và mô phỏng contact (R1) trước khi chốt tần suất.
