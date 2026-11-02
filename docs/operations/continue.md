# Continue — bàn giao sống, 02/11/2026 — W20 đóng; X20 đang chạy

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [vận hành](lessons.md).
Bàn giao trước (P/Q/R/V chi tiết, callback cũ): [snapshot](../../archive/handoffs/2026-11-02-1700-before-w20-launch.md).

## Quyền owner đang có (02/11)

- Sau W20: chạy các kiểm nghiệm cần thiết cho phương pháp cuối → batch X20 sau preflight local PASS.
- Step floor hợp lệ khi bước thêm chỉ synthetic; 1 epoch trên dữ liệu local giữ như repo benchmark.
- Chuẩn FL: không chia sẻ dữ liệu local; server chỉ dùng tổng toàn mạng ([proposal §Quyền riêng tư](../research/proposal.md#quyền-riêng-tư)).
- Không đổi protocol/HP; không chọn round/HP bằng official test; không thêm run ngoài batch đã khóa.
- Credential Kaggle process-scoped, không đổi global login. W&B online, key chỉ trong notebook private. Commit không có trailer `Co-Authored-By`.

## Job sống

| job | việc | log |
|---|---|---|
| `kaggle-x20-pipeline-20m-20261102` | `scripts/run_w20.py --study x20`: quota → 8 notebook → push 2 kernel/owner → poll 1200 s → pull → audit → `X20_CLOSED` | `runs/x-study/pipeline.log` |

8 kernel đã push (02/11 ~22:40) từ `5f9f028`: keocaosu X0/X1 core, nguyetquedo X2/X3 bank,
novahihi X4/X5 live_real_nckd_local, emtrangtri X6/X7 full.
Cache npy dùng lại `odixe0502/mobihet-v4npy-cache/1` (manifest `runs/w-study/builder.json`). Manifest khóa
`docs/research/x20-launch.json`, batch `runs/x-study/batch.json`, owner theo `runs/x-study/quota.json` (quota reset 07/11).
State: `/home/odixe/.codex/long-jobs/jobs/<job>/state.json`; delivery event-only.
Theo dõi: `tail -n 200 -F -- /home/odixe/project/mobihet/runs/x-study/pipeline.log`.
Pipeline dừng ở stage lỗi, giữ receipt; chạy lại `python scripts/run_w20.py --study x20` tiếp stage còn thiếu,
`--collect-only` khi đã push. Không xóa intent để push lại mù.

## Khi job xong

1. Khi `X20_CLOSED`: kiểm `runs/x-study/closure.json`, chạy `scripts/analyse_r20_v20.py` rồi `scripts/analyse_w20.py`
   (đã nhận ô X và control), ghi `docs/research/results-x20.md`, áp quy tắc ở [screening-x20](../research/screening-x20.md)
   và [proposal](../research/proposal.md#sau-x20--quy-tắc-chọn). Ô ERROR: pull log, phân loại, recovery riêng (slug mới).
2. Combo thắng → thiết kế batch 50 round cùng baseline (cần owner duyệt quota/thời gian), cân nhắc ô trộn kiến trúc.
3. Backup R/V/W/X khi OneDrive còn chỗ (~30 GB cần).

## Kết quả mới nhất

[W20](../research/results-w20.md): 10/10 audit PASS. Giữ step floor + bootstrap nén; NCKD dòng thật cắt feedback nhưng đè lớp vắng;
synth hard+NCKD loại. Preflight X `runs/x-study/preflight.json` PASS.

## Việc tay cho owner

- Backup R/V/W/X (~30 GB): giải phóng hoặc mở rộng OneDrive; sau N/O/P/Q còn ~15 GB.
