# MobiHet L4 trên mọi kịch bản — port 20/11/2026

Phương pháp đề xuất MobiHet FB-L cấu hình **L4** (owner 19/11, repo `../mobihet`: [FB-L](../../../mobihet/docs/research/screening-fbl.md)):
mỗi xe giữ model riêng, chỉ trao thống kê bootstrap (copula/KM) + gói Q lớp, không trao tham số; FB N4 (dòng thật 64/37/37 theo
tầng cỡ xe, synthetic 64,64/16,16/16,16), chọn dòng thật theo CE cũ (`real_select=top`), engine `compact=true`.

## Quyết định owner 20/11

- **DAGSNet, BERT:** dùng lại chain r50 × 3 seed của `../mobihet` / `../fl_veremi_bert` (đúng protocol, audit ở repo đó); không chạy lại.
- **TabM, SAE, MLP, CNN-BiGRU:** chạy mới ở repo này, hp L4 khóa y nguyên (chỉ đổi seed), 3 seed. TabM: mọi số hạng loss tính theo
  từng member rồi lấy trung bình (như baseline TabM), điểm chọn dòng = CE trung bình member, Q = xác suất trung bình member.
- **Mixed case1–3:** làm sau đơn model (cần Q/bootstrap gộp xuyên các nhóm model).
- Push tự động khi có slot/quota (không dùng account MobiHet đang giữ slot, không đụng account teammate).

## Code

- `methods/mobihet*.py` (15 file) = byte copy `../mobihet/methods` (method `mobihet_budget` sha `a4147a1b…`);
  `scripts/run_p21_recovery.py`, `scripts/sim_fold_val.py` chép theo (import / harness local).
- `methods/mobihet_l4.py`: subclass `mobihet_budget`, chỉ đổi loss bước cho TabM (logit 4D) và SAE (thêm số hạng 0 cho decoder
  không dùng, như baseline SAE CE-only). DAGSNet/MLP/CNN-BiGRU đi thẳng `mobihet_budget`.
- `core/local.py`: nhánh MobiHet bật khi hp có `synth_step_floor` (bước floor chỉ synthetic; BN buffer + loss log chỉ từ bước có
  dòng thật; loss chia `ceil(n/b)`) — đúng `../mobihet/core/local.py`; baseline không có key này ⇒ không đổi.
- `scripts/run_fl.py`: hp không phải JSON giữ dạng chuỗi (`arm=none`); `--clients`.
- `scripts/build_l4_fl.py`: notebook 2×T4, nhúng source, hp L4 (= `../mobihet/runs/fbl-r50/batch.json`, 44 khóa) + seed,
  `--parent` nối phiên (copy thư mục run từ output phiên trước, kiểm round liền).

## Kiểm trước push (local, `runs/l4-port/`)

| kiểm | kết quả |
|---|---|
| L4 DAGSNet repo này vs `../mobihet` (panel 200 xe fold train→val, 2 round, compiled) | bit-exact: rounds.jsonl, CM r1/r2, 10/11 tensor state; `row_score` chỉ khác ô sink (vị trí đổ của dòng không train, không bao giờ đọc) |
| FedAvg DAGSNet/TabM/SAE trước vs sau port (20 xe, 2 round) | bit-exact |
| L4 TabM resume (stop r1 → resume r2) vs chạy liền | bit-exact (trừ ô sink) |
| Smoke L4 TabM/SAE/MLP (CPU) + CNN-BiGRU (GPU), compile gate | PASS, loss hữu hạn |
| `tests/test_{tabm,sae,dense_recurrent,aligned}_fl.py` | PASS |

## Mixed case1–3 (20/11)

Owner 20/11: model mỗi xe = **MixedAdapter** của backbone được gán (`core.mixed_backbones.config(m)`, đúng model của
FedProto/FedTGP/TinyProto/V2X-HtFL mixed); bảng gán `experiments/mixed-model-v1`, protocol, hp L4 (`L4_HP` + seed) giữ nguyên.

- `methods/mobihet_mixed_l4.py`: mỗi nhóm backbone = một `mobihet_l4` không đổi trên `GroupEnv` (dòng của xe nhóm đó trên rank,
  cfg/layout/lịch nhóm). Mọi trao đổi của L4 là tổng trên xe (`env.allreduce`) ⇒ **Lockstep**: các nhóm chạy như rank ảo
  (thread, lần lượt theo thứ tự nhóm, chỉ đổi tại allreduce; nhóm cuối cộng theo thứ tự nhóm rồi all-reduce qua GPU). Bootstrap
  copula/KM, Q lớp, loss/byte gộp **mọi xe của case**; mỗi nhóm giữ loss, dòng synthetic, chọn dòng thật, compact của nó.
  `local.local_epoch` (wrapper budget/sim/refine) và `local._round_steps` lưu/khôi phục theo thread ⇒ health riêng backbone.
  Collective rank-only duy nhất: compile guard round 1 của L4. Không trao tham số.
- `methods/mobihet_l4.py`: nhánh TabM nhận cả `backbone=tabm` của adapter (adapter tự cho decoder SAE gradient 0).
- Đóng gói engine (không phải hp): BERT `group_rows`/`eval_rows` 256 như `methods/mixed.py`; còn lại `group_rows` 4096 của L4.
- `scripts/run_mixed_l4.py` (checkpoint/resume như `run_mixed_fl.py`, gate health từng backbone mỗi round), `scripts/build_mixed_l4.py`
  (notebook; `--stop-after 2` = pilot, phiên sau `--parent` nối từ checkpoint r2, cùng cosine 50).

| kiểm (`tests/test_mixed_l4.py`, CPU, panel 20 xe) | kết quả |
|---|---|
| lockstep toy: tổng theo nhóm, globals riêng thread, lệch collective ⇒ lỗi, config dynamo chép vào thread | PASS |
| bootstrap gộp case2 = L4 đơn model trên hợp các xe | PASS: `cop_*` exact; KM bằng nhau **sau hoán vị cluster** (thứ tự cộng float64 lật dấu vector riêng lớp 14 ⇒ tâm `mu ∓ d`; sai khác sau hoán vị = 0) |
| gán xe: id nhóm = bảng gán; CM đặt đúng client, mỗi dòng test đếm 1 lần; health từng backbone r1 | PASS (skip 0) |
| một nhóm giữ mọi xe = L4 đơn model (state, Q, CM, 1 round) | PASS bit-exact |
| resume case2 r1 → r2 = liền (state + CM) | PASS bit-exact |

GPU smoke local (RTX 3050, compiled, 2 round, seed 42; `runs/mixed-l4/dbg-case*`): case2 panel 100 xe PASS (compile gate 3/3, skip 0,
`real_frac` .623); case1 panel 100 PASS (r2 F1 .737: DAGSNet .818 / BERT .809 / CNN-BiGRU .572); case3 panel 100 OOM 4 GB ⇒ panel
20 xe + `--test-set km2_frac=1.0` (20 xe không đủ dòng lớp 11 cho KM frac .25) PASS, skip 0 cả 5 backbone, không còn cảnh báo
recompile sau sửa config thread-local. Sửa trong lúc smoke: gate compile `G = min(4, K_nhóm)` (đệm client rỗng ⇒ BN NaN).

Compute lịch sử (panel 200, [compute](compute.md)): L4 DAGSNet .920×, BERT .917×, TabM 1,310×, SAE 2,445×,
MLP 1,952× FedAvg; mixed case2 212,2 Tops (FedProto 96,4) — xem LESSONS 20/11 (chi phí synthetic cố định; panel 200 đệm mixed).
CNN-BiGRU và mixed case1/3 đã pull trace từ `goilaaiudi/exp-count-l4-g4096/1`.

Kernel, kết quả: [continue](../operations/continue.md); số ghi `experiments/<kịch bản>/results.csv` (method `mobihet_l4`).

## Audit 20/11 sau phiên Claude

| Run | Seed | Endpoint | Macro-F1 |
|---|---:|---:|---:|
| CNN-BiGRU | 41 | r50 | .863450 |
| CNN-BiGRU | 42 | r50 | .863215 |
| CNN-BiGRU | 43 | r50 | .861055 |
| Mixed case1 | 42 | pilot r2 | .758572 |
| Mixed case2 | 42 | pilot r2 | .676016 |
| Mixed case2 | 41 | r50 | .853143 |
| Mixed case2 | 42 | r50 | .854014 |
| Mixed case2 | 43 | r50 | .854455 |
| Mixed case3 | 42 | pilot r2 | .717384 |

CNN-BiGRU đạt .86257 ± .00132 ở r50 (SD mẫu), từ CM đủ 9.347.679 dòng. Đủ 10 metric trong [results.csv](../../experiments/cnn-bigru/results.csv). Audit riêng kiểm source đóng băng, HP, seed, lịch LR và 50 round.

Pilot dùng 5.343 xe với gán backbone cố định. Ba case có hai vòng liền mạch, compile gate qua từng backbone, skip dưới 1%, loss hữu hạn và checkpoint r2 đủ hai rank. F1 pilot là endpoint sớm. Bảng baseline mixed công bố ở r50. Phiên nối dùng source notebook đã chạy pilot và giữ cosine 50 round. [Bàn giao](../operations/continue.md)

Mixed case2 đạt Macro-F1 .853871 ± .000668 ở r50 (SD mẫu, ba seed). Seed 42 nối từ r2. Seed 41 và 43 dùng source pilot với seed tương ứng. Tính lại 10 metric gộp và riêng từng backbone từ CM của cả 50 round cho ba seed, với cùng assignment và 9.347.679 dòng test. [10 metric pooled](../../experiments/mixed-case2/results.csv)

Mixed case3 đạt macro-F1 0,874446 ± 0,000175 và accuracy 0,906342 ± 0,000107 tại R50 (SD mẫu, ba seed). Seed 42 nối từ R2, seed 43 nối từ R49. Audit đủ CM của 50 round và checkpoint hai rank cho mỗi seed. [10 metric pooled](../../experiments/mixed-case3/results.csv)

Mixed case1 đạt macro-F1 0,894841 ± 0,000215 và accuracy 0,913857 ± 0,000118 tại R50 (SD mẫu, ba seed). Macro-F1 seed 41/42/43 lần lượt là 0,894902 / 0,895019 / 0,894603. Audit đủ 50 round và hai shard checkpoint cho mỗi seed. History và CM khớp parent tới R49/R44/R47 tương ứng. [10 metric pooled](../../experiments/mixed-case1/results.csv)
