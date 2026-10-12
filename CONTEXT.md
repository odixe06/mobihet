# CONTEXT — MobiHet / federated IDS V2X

Đọc CONTEXT → [LESSONS](LESSONS.md) → [đề xuất theo phương pháp](proposals/README.md).
Tìm từng file: [README](README.md) → [index toàn dự án](docs/file-index.md).

## Bài toán và quyết định đang áp dụng

Xe học IDS cộng tác qua RSU với contact hữu hạn; private model có thể khác kiến trúc ở thiết kế.
Không raw/perturbed record rời xe; synthetic tối ưu từ noise được phép. Input66feature gốc,
protocol so sánh cố định; chọn phương pháp bằng holdout nội bộ, không officialtest.

- Hiện kiểm mọi learned model đều **DAGSNet**; R2 đa kiến trúc chưa có thực nghiệm.
- Local chỉ probe/smoke/collect/report; train nặng Kaggle, W&B online/key bắt buộc trước train.
- MởrộngH1/H2:20client/100%train80%/50round/seed42; Nyström/Hybrid pilotđãchốt.
  [Expansion/run](docs/operations/expansion-2026-10-12.md). W&Bround1–5: lỗi/bấtổn thì yêucầuchủdựándừng.
- Không tự Git/DVC add/push/backup/cleanup. Được restore DVC chọn lọc phục vụ collect.
- Kernel ERROR/COMPLETE không thay audit source/checkpoint/payload/prediction. Output đủ không rerun.
- Source, notebook, training manifest đã chạy giữ đường dẫn/hash. Tài liệu mới theo tên phương pháp.
- Q4 synthetic exception và SecureSum emulator chưa chứng minh crypto/DP.

## Trạng thái kiểm tại 2026-10-12

**18/18 kernel V3–V5 đã audit:** tái dùng6proof cũ khôi phụcmetadata exactMD5,
collect/audit12kernel mới; không rerun. [24nhóm/5hướng ưu tiên](proposals/status-summary.md).
[Status chính xác](docs/operations/kernel-status-2026-10-12.json),
[phiên collect](docs/operations/live-2026-10-12.md), [backup/restore](docs/operations/backup.md).

- P1: Gaussian moment/DAGSNet DML có lợi; P2 deadline/byte/prior có giới hạn.
- S-a Fourier/latent-mean tĩnh FAIL; không mở S-b.
- V2 CountSketch/subspace/density-ratio: paired FAIL; không rerun y nguyên/KD-B/ZO/contact.
- V3 bốn carrier: đủ48run, tất cả pairedFAIL; [report](papers/v3/report.md).
- V4 support vàMALA: paired FAIL, đủ20arm×20round; [report](papers/v4/report.md).
- V5 kernel-anchor small/medium và booksIPC1/4 pairedFAIL; cảIPC1/4 costPASS;
  [report](papers/v5/report.md).
- K5 geometry: chủ dự án chốt G51/G53 × marginal/joint32, giữ primitive nhánh missing
  không xác định; hai cổng quality/byte riêng, fidelity trước train. V6 G59/G61×M/J đã triển khai;
  localmath/15smoke/4exactresume/strict-inferencePASS;4kernel V6 đãpushv1 lúc11:11UTC+7.
  [Quyết định](proposals/geometry-primitive-moments/decisions.md).

## Bước tiếp theo

1. Đợt V3–V5 đã đủoutput/audit; dùng report10metric/perclass/cost/bytes để kết luận cổng.
2. Bài học theo phương pháp và giới hạn/giả thuyết ở [evidence-summary](proposals/evidence-summary.md).
3. K5: chốt eligibility/fidelity/codec/cổng trước bounded probe và Kaggle pilot. Không trùng arm nếu primitive thực tế giống nhau.
4. Chỉ utility PASS mới mở contact/privateKD thích hợp; full-scale/R2 theo thiết kế riêng.

## Muốn biết → mở

| cần biết | tài liệu |
|---|---|
| Cơ chế, code, kết quả, bài học từng phương pháp | [proposals/](proposals/README.md) |
| Bài học bắt buộc và quyết định mới | [LESSONS](LESSONS.md), [quyết định2026-10-12](docs/operations/decisions-2026-10-12.md) |
| Số đo/10metric/ledger từng đợt | [docs/](docs/README.md), `papers/<đợt>/report.md` |
| Dữ liệu/DAGSNet/máy, thông tin không phụ thuộc phương pháp | [knowledge/](knowledge/README.md) |
| Collect/restore/phiên đang hoạt động | [live](docs/operations/live-2026-10-12.md), [backup](docs/operations/backup.md) |
| Lịch sử CONTEXT chi tiết | [snapshot trước tổ chức](docs/operations/context-history-before-method-folders-2026-10-12.md) |
| Protocol/tri thức archive | [protocol](docs/research/protocol.md), [legacy](docs/legacy/kien-thuc-ke-thua.md) |

Số đo ở report/probe; lựa chọn từng phương pháp ở proposals/manifest; facts cố định ở knowledge/.
CONTEXT chỉ trạng thái, quyết định và đường dẫn; index toàn file cập nhật sau thay đổi.
