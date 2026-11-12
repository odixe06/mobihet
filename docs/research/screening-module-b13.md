# Module screening b13 — khóa trước chạy 10/11/2026

Owner phiên này: mở mọi module/compute để cải thiện macro-F1; bắt buộc hỗ trợ khác kiến trúc,
thí nghiệm chỉ **DAGSNet W1**. Byte hạn chế tối đa nhưng **chưa cap cứng**; báo bootstrap,
hai chiều mỗi round và tổng. Cho local trước, Kaggle quota sẵn; PASS tự lên sim2seed → pilot20 → r50two.
Thay quyết định cũ "chưa train mới"; giữ protocol/split/optimizer/LR chung.

## Bằng chứng và phạm vi

[Rà code và lịch sử](../../archive/docs-20261113/research/pipeline-module-review-20261110.md) phân biệt cơ chế đã thử với phần chưa rõ;
[lessons](../../LESSONS.md) và [catalog cũ](improvement-proposals.md) giữ nguyên verdict cũ.
Gap CBB .9078 → APFed .9119 chủ yếu benign tương lai bị nhận thành tấn công; chưa có causal proof
module nào gây gap. Không suy gain từ density/teacher entropy/train fit.

Đợt đầu sàng **ba module, năm ứng viên** trên CBB cùng batch. Không tuyên bố liệt kê/chạy hết không gian
khả thi vô hạn. Nhánh thay thông tin uplink (cross-fit reliability, leave-one-out, disagreement) cần thiết kế
và evidence riêng; xem backlog trong bài rà. Không lặp input-bank, BNsplit, Q theo vùng, generator cũ bằng tên mới.

| arm | bằng chứng → cơ chế | giả thuyết và dấu hiệu bác bỏ | đổi compute/byte |
|---|---|---|---|
| C | CBB KM16 w0 .5, lite6, Q8 | đối chứng mới cùng batch, không dùng LT6/Q16 làm control | nền |
| SQ | sqrt-count cũ gain macro .0012–.0015, benign giảm → giảm chi phối của client nhiều dòng trong Q | tăng macro/giảm FP benign; bác bỏ khi không qua gain+guard trên CBB | không thêm forward/schema; byte thực có thể đổi do nội dung nén |
| RK | synthetic và Q hằng theo lớp có mismatch → `KL(p || Q)` thay `KL(Q || p)` | đổi phân bổ gradient; bác bỏ nếu chỉ dời timeDelay P/R hoặc hại rare/absent | vài phép toán 16 lớp; 0 payload mới |
| SK | cùng bằng chứng RK → `.5 KL(Q || p) + .5 KL(p || Q)` | cân hai áp lực; không cộng gain RK với control; falsifier như RK | 0 forward/payload mới |
| JS | cùng mismatch → **4JS** `2 KL(Q || m)+2 KL(p || m)`, `m=(Q+p)/2` | hạn chế loss trên mismatch lớn; bác bỏ nếu macro không tăng/guard trượt | 0 forward/payload mới |
| LS | train fit cao, own-test kém → **CEreal label smoothing .05** sau LA | giảm memorization; bác bỏ nếu train confidence giảm mà val macro không tăng, hoặc lớp hiếm/timeDelay giảm | 0 forward/payload mới |

RK/SK/JS: Q clamp `1e-6` rồi normalize để là phân phối hữu hạn khi Q8 có số 0;
forward control giữ chính xác clamp hiện có. Không đổi Q uplink/downlink, target class table, temperature hoặc outer weight1.
4JS chọn **trước dữ liệu mới**: độ cong bậc hai tại p=Q bằng forward KL; gradient xa Q vẫn khác,
không gọi đây là pure-shape causal proof. R1 hard synthetic CE giữ nguyên; từ r2 mới đổi divergence.
LS chỉ đổi CE trên **dòng thật**; Q vẫn đo trên đúng dòng thật, chưa smoothing probabilities.
Không aux classifier/proxy/model mới; giao thức dùng count và 16 xác suất nên không cần aligned model/hidden.

Định nghĩa toán là adaptation dự án, không attribution performance từ tác vụ khác.
Primary references: [Generalized Knowledge Distillation](https://arxiv.org/abs/2306.13649)
cho lựa chọn divergence; [Rethinking KL in KD](https://arxiv.org/abs/2404.02657)
cảnh báo không suy mode-seeking thành gain; label smoothing theo
[Rethinking the Inception Architecture](https://arxiv.org/abs/1512.00567).
Nguồn ngoài không chứng minh utility FL ở đây.

## Thiết lập và luật đã khóa

`methods/mobihet_refine.py` kế thừa CBB. Mặc định delegate stream cũ; SQ bật implementation sqrt-count
của Base **sau** init Live, giữ đúng denominator và Q8 codec. `refine_agg` lưu trong HP để audit.
`scripts/run_sim_kaggle.py` `SIM_BATCH=b13`: **12 kernel = 6 arm × seed42/43**,
200 xe phân tầng seed7, fold train→val, cosine20, r20, cap2.75h/kernel; W1, batch64, synthetic64/floor16.
Không official test ở screening. Dự trù cap33 kernel-giờ +15% margin; quota/slots sống trước push,
không coi 33 giờ là nhu cầu GPU-giờ đã đo trên hai T4.

Endpoint primary **r20**. Từng seed PASS đồng thời:

- Δmacro ≥ **+.002** so C cùng seed.
- Δp10 ≥ −.006; ΔtimeDelay F1 ≥ −.0035; Δbenign F1 ≥ −.0005.
- Δabsent accuracy ≥ −.01 nếu ≥2000 dòng ở cả arm/control.
- Audit PASS: source/HP/LR/CM support/10metric từng round, finite, skip ≤1%, mọi client có applied step.

Low256, per-class precision/recall/F1, rare/c11, TB r16–20, bootstrap byte và thời gian báo đầy đủ;
low256 không thành gate sim vì panel này ít xe và rất nhiễu (b12). Giữ guard full-scale.
Ứng viên chỉ lên khi **PASS cả hai seed**; các nhánh PASS lên riêng, không tự ghép combo.
FAIL dừng nhánh, không tune loss/ε/scale/weight/threshold sau số; hai seed không là significance.

Pilot full-scale r20 dùng cosine50, control CBB seed42 **cùng r20**, macro ≥+.002;
guard trên TB r16–20: p10≥−.006, low256≥−.003, timeDelay≥−.0035, benign≥−.0005,
absent≥−.01 (báo riêng endpoint vì absent chỉ1555 dòng toàn mạng). Pilot PASS → resume state2rank
tới r50 và seed43 r50 đồng thời, không restart pilot từ r1.
R50 từng seed: **endpoint r50 Δmacro≥+.002**; guard trên TB r46–50 như pilot,
control CBB cùng seed/round, báo đủ endpoint+TB. Chỉ PASS r50 hai seed mới đề xuất thay phương pháp chính.
Mọi ứng viên chưa có kết quả; CBB vẫn chính.

## Gate thực thi và provenance

- Unit: default loss/gradient/Q/RNG đúng CBB; gradient từng divergence, Q8 zero finite;
  Q lấy từ real, floor-only/padded mask; sqrt aggregate từ decoded packet, state roundtrip.
- Local smoke: 200 xe, W1, compile thật, 2 round để Q tham gia r2; bật đồng thời SQ+JS+LS
  **chỉ kiểm đường thực thi**, không là utility combo. Unit kiểm riêng RK/SK; T4 mỗi arm có compile gate riêng.
- Source hash trước/sau; chỉ push notebook dùng đúng overlay đã preflight; private, pinned image, 2T4 assert,
  không inline credential/W&B như sim hiện tại. Không Git/DVC push.
- Worker `scripts/run_b13_pipeline.py` qua codex-long-jobs: local → quota sống → generate/launch → collect1200s → audit → analysis.
  Source/intent lỗi thì dừng và báo về thread; không blind retry/restart. ERROR cũng pull để phân biệt lỗi vận hành với utility FAIL.
- Evidence: `runs/sim-kaggle-b13/{source-lock,preflight,quota,batch,audit,verdict-r20}.json`,
  `artifacts/sim-kaggle-b13/`; exact owner/slug/version trong batch; job metadata ở [continue](../operations/continue.md).
  Simulator không lưu full checkpoint/prediction: audit CM/metrics/source, chưa prediction→CM replay hoặc full state reconstruction T4.

## Trạng thái

**Đóng 11/11 00:08 UTC+7: 12/12 run đủ20 round, source/config/CM/10metric audit PASS; không ứng viên nào utility PASS.**
Đọc lại artifact và chạy lại audit/analyzer lúc completion, kiểm cả hai rank compile PASS/2T4 từ log.
22 unit PASS; local smoke W1/200xe/2round PASS. `runs/sim-kaggle-b13/{audit,verdict-r20,detail-r20}.json`;
[bảng đủ10metric + byte](../../runs/sim-kaggle-b13/report-r20.md), sinh bằng `scripts/report_module_screen.py b13`.
Không launch pilot hoặc tune tiếp các nhánh này. CBB vẫn phương pháp chính.

| arm | Δmacro s42 / s43 | Δp10 s42 / s43 | Δbenign | ΔtimeDelay | lý do FAIL |
|---|---:|---:|---:|---:|---|
| SQ | −.00037 / +.00070 | −.0038 / +.0009 | −.00362 / −.00050 | −.00481 / +.00579 | macro cả2; benign/td s42 |
| RK | −.00125 / −.00217 | −.0574 / −.0255 | +.00426 / +.00429 | −.02526 / −.02013 | macro/p10/td cả2 |
| SK | +.00083 / +.00164 | −.0139 / −.0055 | +.00511 / +.00409 | −.01604 / −.00797 | macro/td cả2; p10 s42 |
| JS | +.00059 / +.00083 | −.0093 / −.0056 | +.00185 / +.00145 | +.01270 / +.01118 | macro cả2; p10 s42 |
| LS | −.00118 / +.00040 | −.0193 / −.0073 | +.00173 / +.00161 | −.01336 / −.00944 | macro/p10/td cả2 |

Control C/CS43 r20 macro .913839/.913178. Absent chỉ **191 dòng**, nên không gate theo luật ≥2000;
không diễn giải ±.03–.06 absent như tín hiệu chắc. TB r16–20 không đảo kết luận macro: SK +.00047/+.00153,
JS +.00049/+.00076 (không dùng TB để thay endpoint đã khóa).

Đọc cơ chế trong đúng phạm vi:

- SQ không chuyển lợi recipe M20 sang CBB; đổi count weight không sửa được gap.
- RK/SK giảm benign→attack ngoài td (RK 1838→1299 ở s42, 1821→1251 ở s43) nhưng đổi lấy td/xe nhỏ;
  giảm FP benign riêng không đủ để nhận nhánh. RK low256 −.065/−.043.
- JS có dấu tăng benign+td ở cả seed nhưng gain macro dưới gate và p10 s42 trượt; không kéo dài/tune loss để tìm PASS.
- LS làm benign F1 tăng nhưng td giảm, macro không vững; train smoothing chưa sửa được generalization gap.

Chi phí **sim200**, không chuyển thành byte toàn mạng: bootstrap mọi arm = **1.196.032,46 B/xe**;
Q hai chiều control ≈305 B/round (151–152 lên +154 xuống), JS≈292, RK≈305, SK≈310, SQ≈312, LS≈341.
Tổng20r control≈1,202MB/xe; train khoảng3016–3169s/20round/2T4, chưa gồm setup/eval.
Runtime elapsed không là FLOP hoặc chứng minh speedup. Không có payload schema mới trong b13.

Chẩn đoán count cho bước sau (`detail-r20.json`): client lớn nhất đóng góp tối đa **9,01%** count/lớp ở sim,
nhưng chỉ **0,46%** ở full5343; mean effective clients/lớp≈48,2 vs1065. Đây là count concentration,
không đo covariance prediction; làm yếu ưu tiên leave-one-client-out như đòn bẩy full-scale, chưa chứng minh vô ích.
