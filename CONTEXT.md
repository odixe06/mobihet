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
  [Expansion/run](docs/operations/expansion-2026-10-12.md). W&B banđầu; đãổnđịnh thìngừnghealthpoll theoquyếtđịnhmới. Lỗi/bấtổn quan sátđược thì yêucầuchủdựándừng.
- Không tự Git/DVC add/push/backup/cleanup. Được restore DVC chọn lọc phục vụ collect.
- Kernel ERROR/COMPLETE không thay audit source/checkpoint/payload/prediction. Output đủ không rerun.
- Source, notebook, training manifest đã chạy giữ đường dẫn/hash. Tài liệu mới theo tên phương pháp.
- Q4 synthetic exception và SecureSum emulator chưa chứng minh crypto/DP.
- Hướng/cải tiến mới phải có [dự báo utility từ bằng chứng](docs/research/performance-feasibility-2026-10-13.md)
  trước pilot. Chốt BoundaryCells direct/teacher, receiver stress local có fidelity gate, EMA fresh 2×2+own;
  không sửa run dài hiện tại.

## Trạng thái kiểm tại 2026-10-14

**18/18 kernel V3–V5 đã audit:** tái dùng6proof cũ khôi phụcmetadata exactMD5,
collect/audit12kernel mới; không rerun. [Danh mục và hướng ưu tiên](proposals/status-summary.md).
[Status chính xác](docs/operations/kernel-status-2026-10-12.json),
[phiên collect](docs/operations/live-2026-10-12.md), [backup/restore](docs/operations/backup.md).

- P1: Gaussian moment/DAGSNet DML có lợi; P2 deadline/byte/prior có giới hạn.
- S-a Fourier/latent-mean tĩnh FAIL; không mở S-b.
- V2 CountSketch/subspace/density-ratio: paired FAIL; không rerun y nguyên/KD-B/ZO/contact.
- V3 bốn carrier: đủ48run, tất cả pairedFAIL; [report](papers/v3/report.md).
- V4 support vàMALA: paired FAIL, đủ20arm×20round; [report](papers/v4/report.md).
- V5 kernel-anchor small/medium và booksIPC1/4 pairedFAIL; cảIPC1/4 costPASS;
  [report](papers/v5/report.md).
- V6 geometry:4kernel/30arm đãaudit.12G59overflow;18control/G61 đủ20round.
  G61quality/compressionpairedFAIL; joint mask cólợi tươngtựfull66, chưageometryproof.
  [Kết quả](docs/research/probes/pilot-v6.md). Khôngrerun/contact/full adaptation này.
- V7 Nyström:20arm×20round/hai seed đãaudit. Rank4/8 compressionPASS, qualitypairedFAIL;
  khôngcontact/full/rerun y nguyên. [Kết quả](papers/v7/report.md).
- V8 Hybrid cả hai seed đã audit/utility PASS: hơn thành phần mạnh nhất +0,04070/+0,04355;
  interaction âm, chưa bằng chứng synergy. [Cổng](papers/v8/gates.json).
  V9 residual đủ 26 arm-seed ×20 round: M64/M512 cost PASS, quality paired FAIL;
  không mở private KD/contact/full y nguyên. [Report](papers/v9/report.md).
- V10energy:2kernelCOMPLETE/14arm đãaudit. PairedcostFAIL vàr1samplerFLFAIL;
  candidatequalityr20 chưađo. Khôngrerun y nguyên/private/contact. [Report](papers/v10/report.md).
- Full H1/H2: 7 arm đã audit tới r7–r11/50 trên 100% train; chưa đủ r50.
  [10 metrics các round cuối](papers/full-validation/latest-rounds.md), [report](papers/full-validation/report.md).
  Audit source/weights/pred→CM/metric có cho round mới; sample-inference 64check tại r1/round cuối,
  prefix/client0, chưa full-inference/leakage proof.
  Chủdựán chốt chạytheoquota, nối checkpoint sau17/10; target/LR50 giữ nguyên.
  Cont01 terminal đã thu/audit; lần nối tiếp lấy parent cont01 và round riêng mỗi arm.
  Hybrid full 4 notebook terminal/pull/audit: 7 arm, 18 round; 3 arm r2 và 4 arm r3/50.
  Hai ERROR do budget/controller, checkpoint hợp lệ, không rerun từ r1.
  Hybrid có tín hiệu tốt đầu full-data; chưa utility r50/R1/R2. [Kết luận](papers/hybrid-full/feasibility-20261014.md),
  [10 metrics cuối](papers/hybrid-full/latest-rounds.md).
  [Ledger](papers/hybrid-full/rebuild.md).
  [Phiên mới/preflight/remaining](docs/operations/continuation-2026-10-13.md).

## Bước tiếp theo

Phiên sau bật lại máy: [collect/audit và việc còn thiếu 14/10](docs/operations/collection-2026-10-14.md).
BoundaryCells calibration hai seed đã audit, cả direct/teacher P2/P4 cost PASS (1,49–1,60× single-DAGS).
Pilot r20 đủ controls đang chuẩn bị, utility chưa đo. [Calibration](papers/boundary-cells/calibration-report.md).
ResponseRisk quyết định mới nhất: chỉ local augmentation trước, chưa shared-response pilot.
EMA cả nền Gaussian/DML và cổng r20 đã chốt; chưa triển khai/pilot utility.

1. Đợt V3–V5 đã đủoutput/audit; dùng report10metric/perclass/cost/bytes để kết luận cổng.
2. Bài học theo phương pháp và giới hạn/giả thuyết ở [evidence-summary](proposals/evidence-summary.md).
3. Tiếp checkpoint H1/H2/Hybrid tới r50 theo quota/refresh17Oct, parent đúng round đã audit;
   Hybrid cần budget đo riêng, không dùng chi phí H1/H2 thay thế.
   BoundaryCells paired r20 sau cost PASS; ResponseRisk local augmentation và EMA cần source/preflight.
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
