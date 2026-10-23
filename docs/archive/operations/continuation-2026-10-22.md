# Bàn giao 22/10/2026 — report final xong, synthetic P29–31, sẵn sàng compact

**Snapshot lịch sử (chuyển vào archive 22/10 ~10:45Z).** Trạng thái sống: [continue](../../operations/continue.md).

Đọc [CONTEXT](../../../CONTEXT.md) → [LESSONS](../../../LESSONS.md) → trang này.
Owner đã nối nghiên cứu synthetic, yêu cầu pull/chốt report, rồi tổng hợp ngữ cảnh trước compact.
Không mở thêm thí nghiệm khi đóng mốc này; toàn bộ job được mở trong phiên đã kết thúc.

## Run cuối seed42 — đủ train và chấm

**6 chuỗi ×50 round đã train/chấm/audit đủ**;300 round train,1.800 hàng metric (10 metric/hàng),
576 hàng precision/recall/F1 theo lớp ở r50. Mỗi model chấm toàn10.761.343 test rows; headline r50 cố định.

- [Report hoàn chỉnh](../../methods/full-drift/results/final/report.md): headline, per-class, mọi round,
  private/private_ema/proxy, chi phí, so sánh baseline, đồ thị; [CSV](../../methods/full-drift/results/final/metrics.csv).
- [Completion](../../methods/full-drift/results/final/completion.json),
  [provenance kernel/version/source/audit SHA](../../methods/full-drift/results/final/provenance.json),
  [kiểm export](../../methods/full-drift/results/final/VERIFIED.json).
- Phương pháp `gkd + la + kdpx`, proxy `q3_zl`: za20c/ze50c/zi100c. Đối chứng `gkd` dense: zc20c/zg50c/zk100c.
  Seed43 final chưa chạy; mọi private/proxy DAGSNet. Contact/deadline và mixed-architecture chưa proof.
- Bộ chấm nhanh `fd_eval_fast.py`, lớp tolerance owner duyệt; source r29 dùng gate≤2% max|logit|,
  chấm mọi private/EMA/proxy. Protocol/timing ở [final-pipeline](../../operations/final-pipeline.md).

**Vừa collect:** bốn kernel private version1, source/executed/input/weights/predictions/CM/10metric đã audit:

| ID notebook (sau `<owner>/mobihet-`) | owner | rounds |
|---|---|---|
| `fd-e34-38-s42zk04-20261015-r25-v1-f1` | anhtruong1909 |34–38|
| `fd-e39-43-s42zk04-20261015-r25-v1-f1` | minhtrit06 |39–43|
| `fd-e44-47-s42zk04-20261015-r25-v1-f1` | takids1909 |44–47|
| `fd-e48-50-s42zk04-20261015-r25-v1-f1` | minhtrit06 |48–50|

292 file,0 thiếu,~7,37GiB mới; marker `AUDITED.json` ở
`methods/full-drift/builds/fd-probe/artifacts/kaggle/<ID>/`. Private executed0600, bản redacted cạnh notebook.
**Không repull/re-audit các kernel này theo bảng cũ.** Source đã PUSHED giữ nguyên. Không launch thêm kernel trong phiên này.
MCP Kaggle không xuất hiện trong tool inventory; CLI account-scoped đã xác nhận/pull thành công, không switch global login.

## Nghiên cứu synthetic — tiến triển và ngõ đã đóng

Nguồn chuẩn mới: [P29–31: protocol, số đo, giới hạn](../../research/methods/synthetic-gradient-diagnosis-2026-10-22.md).
Đọc kèm [nhận diện update + primary sources/đề xuất tiếp](../../research/methods/synthetic-transfer-identifiability-2026-10-22.md),
[P23–28](../../research/methods/synthetic-learning-recovery-2026-10-22.md),
[build cũ/veto](../../research/methods/synthetic-build-history-audit-2026-10-22.md).

- P23 Project48 teacher PASS hai seed T4; P24 transfer/inversion FAIL; P26 weighting chưa đủ gain;
  P27 real support là oracle R3-ineligible; P28 gradient teacher FAIL. Không mở lại adaptation y nguyên.
- **P29:** đủ20 client,1.120 prediction→CM→10metric checks, exact CE/KL gradient và decomposition PASS.
  Synthetic không tái tạo hướng cập nhật private như real support, kể cả control cùng target theo lớp.
  Own CE/own KD hướng khác balanced-CE; đây là surrogate tại snapshot, không gradient macro-F1 hoặc toàn quỹ đạo Adam.
- **P30:** joint true-class×teacher-probability×input, đúng inclusion weights full train, codec thật;
  đủ20 client,200 prediction/metric checks. **FAIL** gain; joint không hơn mean-only/zero-cross.
  Không tune/launch Kaggle hướng này. Gaussian matched control được rerun, không claim bit-exact với P24 cũ.
- **P31:**16 train points×20 private, strict FP32 Jacobian và finite perturbations. Có hướng teacher48 không nhìn thấy,
  nhưng private Jacobian energy trong đó nhỏ trên mẫu đo. Chưa được kết luận mất chiều là nguyên nhân chính.
- **Hướng còn ở proposal:** conditional input-space density learner từ nhãn/dữ liệu thật tại client,
  classifier private vẫn độc lập. Chưa code/train/utility PASS. Cần capacity-oracle trước;
  atoms/dequantization và số contact học generator chưa chốt. Xem §7 ghi chú primary; không tự thêm jitter.

Script mới ở `methods/full-drift/builds/fd-probe/source/proj/fd_probe29.py`, `fd_probe30.py`, `fd_probe31.py`;
audit `scripts/check_gradient_synthesis_probes.py`. JSON/NPZ/heads của P29/30 ở
`docs/methods/full-drift/results/probe29-private-gradient-s42.*`, `probe30-joint-input-s42.*`;
P31 `probe31-private-observability-s42.json`. Không sửa production `fd.py`.

## Ràng buộc giữ nguyên sau compact

- R1 mới: ≤1,5× xa ở **cả15/50 round**, ≤231.000 B/contact kể cả bootstrap. Baseline/bootstrap còn vướng;
  budget nhỏ của một carrier không chứng minh toàn phương pháp đạt. Tính cả upload/downlink/header/ACK/crypto/contact wait.
- Test20 chỉ báo cáo; không chọn HP/round/recipe bằng test. Old/group đã dùng nhiều lần là development;
  không coi seed42 frozen-head probe là online ep15/r50 hoặc xác nhận đa seed.
- Mọi candidate cần changed mechanism→control→falsifier→stop gate; giữ benign≥−0,005,worst≥−0,010.
  Cổng method online ep15 +0,010 và xác nhận s43/horizon riêng; không kéo dài FAIL để tìm round thắng.
- Owner dùng Kaggle cho dự án khác: quota đọc live trước push. Credentials account-scoped;
  không in key, private inline notebook0600. User authorization launch cũ không thay cổng khoa học còn thiếu.
- Không Git/DVC push/backup/cleanup nếu owner chưa yêu cầu. Phiên này không stage/commit/push/xóa artifact.
  Có nhiều thay đổi của phiên khác trong skill backup/DVC/cloud; giữ nguyên. HEAD quan sát `f6ed8ed9ed5228237e1a8f303edcbe07c3b0a59a`.

## Sau compact

1. Đọc tuyến đầu trang; tiếp luồng synthetic từ kết luận P29–31. Report final đã xong, không chạy lại collectors.
2. Nếu tiếp synthetic: đọc kết luận P29–31 và §7 proposal; chốt atom policy trước triển khai density oracle.
   Oracle dùng pooled raw chỉ chẩn đoán local, không được biến thành payload R3. Khóa contrasts/gates trước thử.
3. Chỉ qua utility local mới xác nhận T4/seed43; chỉ qua transport/byte gates mới online. Không có run mới chờ launch tự động.

Lệnh tái sinh/check report (chỉ khi artifact/report đổi):

```bash
B=methods/full-drift/builds/fd-probe/scripts
conda run -n nckh python "$B/report_final.py"
conda run -n nckh python "$B/report_final_md.py"
conda run -n nckh python "$B/check_final_report.py"
```

Việc tay hiện tại: không có. Nếu owner muốn compact, dùng `/compact`; sau đó đọc lại tuyến đầu trang.
Kiểm bàn giao22/10 17:04 UTC+7:4.244 relative links/16 tài liệu không thiếu target;7 source Python AST PASS;
report/provenance SHA vẫn khớp `VERIFIED.json`;4 terminal audit/private0600/redacted PASS.
Không còn process P29–31/collector/report của phiên; staged diff rỗng. Hai file index đã tái sinh, không di chuyển artifact.
Lịch sử: [snapshot trước đủ report](continuation-before-final-report-2026-10-22.md),
[bàn giao trước compact nghiên cứu](continuation-synthetic-2026-10-22.md),
[20–21/10](continuation-2026-10-20-21.md). Không dùng snapshot lịch sử làm trạng thái live.
