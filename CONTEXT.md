# CONTEXT — MobiHet / federated IDS V2X

Đọc CONTEXT → [LESSONS](LESSONS.md) → [handoff hiện tại](docs/operations/continue.md).
[Mục lục](docs/README.md) · [Phương pháp](docs/proposals.md) · [Index file](docs/file-index.md).

## Bài toán

Xe học IDS cộng tác qua RSU với giao tiếp hữu hạn; private model độc lập, thiết kế hỗ trợ client khác kiến trúc.
Giữ 66 feature, dataset/partition/scaler gốc; mọi model đang chạy là DAGSNet. Không raw/perturbed record,
per-record activation/gradient rời client. Heterogeneous R2 thực nghiệm và contact/deadline R1 chưa chứng minh.

## Trạng thái và quyết định hiện hành — 21/10/2026 ~13Z (phiên dừng)

- Phương pháp chốt: `gkd + la + kdpx`, proxy q3_zl (PASS seed42/43). [Kết quả/cổng](docs/methods/full-drift/results.md), [thiết kế](docs/methods/full-drift/design.md).
- **Run cuối seed42** (q3_zl la_kdpx + đối chứng gkd × 20/50/100 client, 50 round × 1 epoch toàn train80%, chấm toàn test20% mỗi round,
  headline r50, không chọn HP/round bằng test): **za, zc, zg train xong r50**; ze r1–48, zi r1–26, zk r1–33 đã audit, S2 đang chạy.
  Chấm bằng bộ nhanh `fd_eval_fast.py` (lớp tolerance owner duyệt, khớp 99,96 %); owner chọn **song song thêm**, giữ EMA mọi round.
  Tag chấm **r29** từ ~10:40Z (gate tương đối ≤ 2 % max|logit|; gate tuyệt đối r27 vỡ ở round muộn, model không sai). Đã audit chấm:
  za r1–44, zc r1–27, ze r1–8 + r29–33, zg/zk r1–2; ze train đủ r1–50. Vòng 25 `cv` đã cài + test local PASS, chưa push — chờ owner xác nhận byte ([continue](docs/operations/continue.md)).
  Exact kernel/round/account: [continue](docs/operations/continue.md); timing/contract: [pipeline](docs/operations/final-pipeline.md).
- **Vòng 24 (owner 21/10: sinh mẫu có chủ đích): FAIL.** Probe 20: Gaussian gkd bất đồng sai chỗ; arm `xo` (lai ghép dòng own theo cặp
  bất đồng, 0 byte) đúng chỗ nhưng s42 ep15 Δ +0,0076 < +0,010 ⇒ dừng hướng ([results §Vòng 24](docs/methods/full-drift/results.md));
  xou (đối chứng) chỉ còn mô tả. Run cuối giữ phương pháp chốt. Phân tích sau FAIL (probe 21/22): 77 % lỗi private là lỗi teacher cũng sai
  ⇒ đề xuất vòng 25 `cv` hiệu chỉnh teacher bằng đặc trưng ảo theo lớp ([hồ sơ](docs/research/methods/teacher-virtual-feature-calibration.md)), chờ owner.
- Owner còn dùng Kaggle cho dự án khác: quota hụt bất thường là do dự án đó ([kaggle](docs/operations/kaggle.md)); đọc quota live trước mỗi push.
- odixeuit dùng chung với dự án khác của owner: không xếp chấm/train final lên đó. minhtrit06 là luồng chấm K50 chính (cache K50 sẵn sàng 10:55Z).
- `QUEUED` = chờ GPU. `RUNNING`/`COMPLETE`/W&B finished không chứng minh kết quả; cần collect + audit artifact. `kaggle datasets create` có thể báo
  lỗi dù dataset đã ready — đọc status trước khi thử lại. Log live: `kaggle kernels logs --follow` (SSE).
- Freeze mọi source/notebook đã PUSHED; sửa bằng tag mới. Resume từ round committed đã audit. Không nối adaptation FAIL y nguyên.
- Không Git/DVC push/backup/cleanup khi owner chưa yêu cầu. Credentials account-scoped, giữ global login thuyanh316; inline W&B chỉ ở
  private notebook 0600. Khi không rõ yêu cầu khoa học/budget, hỏi owner. Việc tay: báo exact bước và lỗi quan sát.

## Phiên sau

1. Mở [continue.md](docs/operations/continue.md): bảng train/chấm/screening, mục "Làm tiếp theo thứ tự".
2. Kernel terminal ⇒ collect + audit trước mọi kết luận; train S2 xong ⇒ chấm round mới (account khác train, có cache K).
3. Screening xo/xou xong ⇒ `report_fd.py`, đọc cổng vòng 24, báo owner.
4. Đủ eval r1–50 cả 6 chuỗi ⇒ `report_final.py`, cập nhật costs/10 metric từ artifact đã audit.

## Cần biết → mở

| câu hỏi | tài liệu |
|---|---|
| Tiếp tục ngay, kernel/checkpoint/quota/manual action? | [Handoff](docs/operations/continue.md) |
| Train/chấm tách, timing, source/tests, cost? | [Pipeline cuối](docs/operations/final-pipeline.md) |
| Kết quả nào PASS/FAIL, vì sao chọn phương pháp, vòng 24 sinh mẫu có chủ đích? | [Results](docs/methods/full-drift/results.md), [design](docs/methods/full-drift/design.md), [danh mục](docs/proposals.md) |
| Protocol, thông tin rời client, so sánh/cost? | [Protocol](docs/research/protocol.md), [luồng FL](docs/research/federated-information-flow.md), [so sánh](docs/research/comparison-fl-dagsnet-veremi.md), [cost](docs/research/computational-cost.md) |
| Dataset/kiến trúc cố định? | [Knowledge](knowledge/README.md) |
| Account/quota, lỗi kernel, layout, Git/DVC? | [Kaggle](docs/operations/kaggle.md), [sổ lỗi](docs/operations/notebook-errors.md), [layout](docs/operations/project-layout.md), [backup](docs/operations/backup.md), [artifact index](docs/operations/artifact-index.md) |
| Quyết định/kết quả lịch sử? | [Archive](docs/archive/README.md), [CONTEXT trước gọn tài liệu](docs/archive/operations/context-before-final-recovery-2026-10-20.md), [bàn giao17–20/10](docs/archive/operations/continuation-2026-10-17.md), [nhật ký 20–21/10](docs/archive/operations/continuation-2026-10-20-21.md) |

Hub giữ trạng thái/quyết định/route. Số đo và commands ở docs; dữ kiện cố định ở knowledge; procedure dùng lại ở skills.
