# CONTEXT — MobiHet

Đọc file này → [LESSONS](LESSONS.md) → [bài học vận hành](docs/operations/lessons.md) →
[bàn giao sống](docs/operations/continue.md). Không nạp hoặc tìm kiếm `archive/` trong phiên thông thường.

## Bài toán và quyết định đang có hiệu lực

Federated IDS V2X, VeReMi NextGen v4: 5.343 xe, 16 lớp, 77 đầu vào. Mỗi xe giữ model riêng;
trao đổi thống kê gộp/output schema chung, không mẫu thật, feature/nhãn từng dòng hoặc private weights.
M/N/O dùng Q là bảng xác suất theo lớp; P đang kiểm thêm teacher theo input và proxy riêng tại server.
Chưa có bảo đảm DP/secure aggregation.

- Ưu tiên own-test tương lai của từng xe, cải thiện xe ít mẫu/non-IID và giảm chênh chất lượng bằng cách nâng xe yếu.
- Tối ưu ba khâu: thông tin upload → server xử lý/downlink → client kết hợp local và teacher.
- Chưa ngưỡng byte cứng; báo bootstrap + upload/download và thời gian. Schema hỗ trợ khác kiến trúc;
  thí nghiệm hiện giữ DAGSNet W1 để cô lập cơ chế, chưa chứng minh kiến trúc trộn.
- Protocol chung 50 round; batch chẩn đoán M/N/O chỉ r1–20, vẫn cosine50, seed42/43. Không tự kéo dài hoặc chọn best test round.
- Kaggle dùng credential theo process, giữ global login hiện tại; trạng thái tài khoản ở bàn giao. W&B online. Không cần MCP JSON.

## Trạng thái — 02/11/2026

M0–M15, N0–N7 và O0–O7 version1 đã collect/audit đủ20round; source/config/CM/Qlag/codec/snapshot PASS.
[O20](docs/research/results-o20.md) đã kiểm đủ2seed: teachertrajectory và cách nhận KD đều quan trọng;
fullQ có utility pooled nhưng lợi ích chưa đều cho xe yếu. Các job đã đóng, không restart/poll/collect lại.

Backup Git/DVC đã kiểm, archive và cleanup local hoàn tất. Owner đã yêu cầu chạy kiểm chứng N20.
Trạng thái backup/cleanup duy nhất ở [bàn giao](docs/operations/continue.md).
Artifact N/O mới chưa DVCbackup; giữ local. Chưa chốt phương pháp tổng.
Jobclosure/callback muộn chỉ ở bàn giao; không tự tăng round hoặc đổi protocol.

Owner01/11 đã ủy quyền P20 nhiều kiểm nghiệm song song: teacherinput/proxyserver/clientreception/v4missingness/localLA.
[Thiết kế32run](docs/research/screening-p20.md), [audit nguồn v1/features/benchmark](docs/research/prior-evidence-p20.md).
Giữ clientDAGSNetW1 như benchmark; v1 chỉ tham khảo khácdataset66feature/20–100client.
P0–P31 exactversion1 đã pull/audit đủ32:31PASS_BOUNDED đủ20round, P21 nonfinite trước round1.
[Auditclosure](docs/research/audit-p20.md), [sự cố/probe](docs/research/p20-numerical-incident.md).
Đã đóng gate audit và [phân tích P20 gốc](docs/research/results-p20.md).
[Recovery P21/C10seed43](docs/research/results-p21-recovery.md) đã audit đủ20round, bổ sung đủ32logicalarm/hai seed;
giữ originalP21FAIL và báo cáo gốc. Hybrid hơn proxythuần pooled cảhai seed nhưng vẫn dưới streamingclassQ,
xe nhỏ chưa hơn ensemble nhất quán; chưa chốt phương pháp tổng hoặc numericalrootcausefix.
Trạng thái job duy nhất ở bàn giao; không restart collector/launcher từ callback muộn.

[Q20](docs/research/results-q20.md) đã pull/audit/phân tích đủ16run/8cell×seed42/43,
DAGSNetW1/20round/cosine50/train+val; không trainingfailure. Controlquery qua tripwire, inputtarget nâng mean/p10 xe yếu;
sqrtinputweight giảm pooled cảhai seed. Margin nâng pooled/mean/p10 nhưng vẫn hại absent/timeDelay,
chi phí4.645504MB/xe so compactQ3.333504MB; chưa phương pháp tổng hoặc thắng benchmark.
Worker đều đã đóng; checker CPU/CUDA đã sửa, không trainlại. Owner02/11 đãủyquyền [R20](docs/research/screening-r20.md): phase8replay cùngQ/inputhistory đãpush đủ8kernel`-input1`version1 sau privateinput/preflightgates;
worker poll/pull/audit1200s. [V20](docs/research/screening-v20.md) split/localpreflightPASS, phase8validation đãpush đủ8kernel version1, receipt/SHA đãkiểm;
V0/V1ERROR saufirstepoch tại guardzero-skip; [hai probeT4](docs/research/v20-skip-incident.md) đãaudit3/5skip trên513368step, mọixeapplied/weightsBNfinite.
Worker CErecovery riêng chạyfresh20round với guardcommon+perclientfinite/applied, core/HPkhôngđổi;6kernel khác giữwatch, originalfailures giữnguyên.
chưa remotecompletion, không đổi headline own-test. Owner bổ sung tài khoản; phase4repeatA/B×2seed đãpush đủ4version1 sau inventory6owner mới.
Không modelpoll; trạng thái/actualpushreceipt ở bàn giao.
Memory/qualityweight/gradientprojection chưa triển khai.

## Cần biết → mở

| câu hỏi | tài liệu |
|---|---|
| Cơ chế hiện tại và cấu hình chung? | [method](docs/research/method.md), [protocol](docs/research/protocol.md) |
| Kết quả mới và giới hạn bằng chứng? | [Q20](docs/research/results-q20.md), [Q byte/time](docs/research/cost-q20.md), [P21 bổ sung](docs/research/results-p21-recovery.md), [P20 gốc](docs/research/results-p20.md), [O20](docs/research/results-o20.md) |
| Nguyên nhân nào đã biết, cần kiểm gì tiếp? | [diagnosis](docs/research/diagnosis.md) |
| Job nào sống, ai sở hữu file, bước tiếp? | [continue](docs/operations/continue.md) |
| Cách chạy công cụ, backup và khôi phục? | [operations](docs/operations/README.md), [scripts](scripts/README.md) |
| Dataset, model, máy, image? | [knowledge](knowledge/README.md) |

Số đo ở kết quả; quyết định ở diagnosis/protocol; dữ kiện bất biến ở knowledge; trạng thái job chỉ ở continue.
