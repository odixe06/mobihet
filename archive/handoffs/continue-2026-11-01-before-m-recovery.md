> Snapshot; trạng thái hiện tại tại [bàn giao sống](../operations/continue.md).

# Continue — bàn giao sống, 31/10/2026 ~18:10 (giờ VN)

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [lessons vận hành](../operations/lessons.md).
Snapshot trước M: [L đã báo cáo](../archive/continue-2026-10-31-before-m20.md).
Repo này nghiên cứu federated/MobiHet; baseline/NILM/Notion ở `../fl_dagsnet_veremi_v2`.

## Owner chốt và phạm vi

- **Chỉ chẩn đoán 20 round**, không50. Own-test tương lai từng xe là đánh giá chính; chưa benchmark/chốt method thắng.
- Tập trung ba khâu: client gửi thông tin cần thiết; server gộp/xử lý và gửi xuống; client cân bằng local và global knowledge.
  Xe ít mẫu/thiếu lớp cần tiến bộ, giảm chênh chất lượng giữa xe; không coi xe mạnh giảm là cải thiện fairness.
- Không raw local row/feature/nhãn từng mẫu/private weights trong thông điệp FL. Q là bảng16×16, không proxy neural;
  cùng Q mỗi round, cập nhật cho round kế tiếp. Aggregate statistics chưa có DP/secureaggregation guarantee.
- Ưu tiên F1, báo bootstrap+upload+download và thời gian; chưa ngưỡng byte cụ thể. Giữ DAGSNetW1 chung, mixedarchitecture sau.
- Owner chọn thêm chẩn đoán lớp từ xe khác, rồi nhấn mạnh own-test vẫn chính: chỉ evaluator mô phỏng34xe theo traincount,
  pool test cố định, loại source cùng xe, không vào training/bootstrap/Q hay kênh FL; không gọi là public data triển khai.
- Không Notion/Git/DVCpush/cleanup. CLIglobal **dinhan25** giữ nguyên; khôngMCPJSON/loginlại.
  HEAD099e95e8466811881fb71dcd7d3c44268ed082d6, stagedtree5cd0bcc611b032ca91743dc3e605a9c95d1911a3 không đổi.

## M0–M15 version1 đã push — watcher/collector nền đang chạy

[Spec khóa](../research/screening-m20.md).16notebook private, r1–20 mới/cosine50, full5343xe,2×T4,W1,batch64,1epoch.
Mỗi cell hai seed42/43:

| cặp | thay đổi được kiểm |
|---|---|
| M0/M1 | augmentation, khôngQ/khôngKD |
| M2/M3 | softQ fixed1, tắt realKD |
| M4/M5 | softQ fixed1 + realgate, control chính |
| M6/M7 | synthetic fallback teacher sai→onehot |
| M8/M9 | thu Q postepoch eval thay online |
| M10/M11 | server gộp sqrt(n_kc) thay n_kc |
| M12/M13 | fallback + uniformρ=.5 cho mixture và realKD |
| M14/M15 | fallback + localρ=64/(64+n_kc), cá nhân hóa theo lớp/xe |

Không cộng gain các cell/chốt cải tiến từ công thức. L hardfixedgate tốt hơn softfixedgate ở xe nhỏ là phản chứng cần
lưu ý trước khi tin tăng teacher luôn giúp. Cùng copula64/codecmeanfp16/outerlossweight1; không thay LR/model/optimizer.
Mốc cuối r20; đọc Δ10→20/15→20 và slope11–20, mean/p10/p90−p10/std giữa xe, lớp/absent/rare class; không best-round.
Q1088B/xe/r +3311744Bbootstrap =3333504B/xe/20round; augmentation3311744B. Bootstrap vẫn lớn, chưa giải quyết bandwidth rất thấp.

### Runtime và ownership

- Tất cả16push trả rc0/exactversion1, không ambiguous. Hai run đầu M8/M9 push17:58:50;14run còn lại18:01:41–18:02:10.
- M8/M9 initial read18:01:07: RUNNING,2×T4/source manifest/W&Bauth PASS, chưa thấycompilegate ở lần đó.
  Manifest remote cảhai=2a0ed1906fe31e0556fdf58f4d306c882bbdaed845baee5c0ed87783f8aa7cc1.
- Startup toàn batch lầnđầu18:04:55:7RUNNING/9QUEUED;7run xácnhận2×T4/source/W&Bauth,0logerror, chưa thấycompilegate.
  `runs/m-study/startup.json`, startup-all.log. W&Blookupđầu chưa thấy16run(CommError), không suy login lỗi/kernel chết;
  W&Binit sau data+compile. Những runQUEUED chưa cóbằngchứng phầncứng. Khônggọi toànbatchhealthy/progress từstatus.
- Longjob **kaggle-m-v1-watch-collect-20m-20261031**, supervisorPID**46308**, đãverifyrunning mộtlần.
  State `/home/odixe/.codex/long-jobs/jobs/kaggle-m-v1-watch-collect-20m-20261031/state.json`;
  log `/home/odixe/project/mobihet/runs/m-study/watch-collect.log`; delivery**queue(auto)** vềownerthread,
  viewernone. Lệnh log: `tail -n 200 -F -- /home/odixe/project/mobihet/runs/m-study/watch-collect.log`.
  Worker`watch_collect_m20.py`:status1200s; nếuERROR/CANCEL thìpullnodeđó→error-collection.json→jobfailed/queuecompletion,
  giữkernelkhác; nếu16terminalCOMPLETE thìpull4×8connections→auditM20→markerM20_PULL_AUDIT_DONE.
  Khôngmodelpolling, khôngautoGPUlaunch/cancel, khôngsurvivereboot; khôngstartwatcher/collectortrùng khi job này còn chạy.
- Ownerthread `01a18d65-e3b8-73f1-80b6-7b66531915da`; user yêu cầu pollnền1200s, khôngmodelpolling lúcchờ.
  `AGENTS.md` mục8 đã lưu rule skill global cho phiên khác; không ghi job ID vào AGENTS.
- Worker đang sở hữu `scripts/{watch_collect_m20,watch_kaggle_batch,audit_m20,audit_screening,run_bounded_python,kaggle_cli_bounded}.py`;
  giữ các file này read-only đến khi job kết thúc.
- Tôn trọng notebook/source đãpush; không thay bytes của5overlay để sửa run cũ, không pushversion2/restart khi chưa xác định lỗi.

### Local preflight và artifact

- 47tests PASS13,23s: independent loss+gradient fallback/uniform/local, maskedrows, classsupport, postepoch fullsupport,
  state/BN/RNG bảo toàn, explicit fp16upload/downlink, sqrtaggregation, external evaluator source-exclusion/hash.
- 3control augmentation/noReal/gate CPU2round BIT_EXACT với `runs/m-study/mobihet_l_frozen.py`.
- Real4xe [3338,3654,4793,5045],144306trainrows,2round W1 compiledAMP sm86: skip0, train/test10metric khớpCM,
  fullprivatebank strictrebuild, teacherlag/probes PASS. Smoke ghép nhánh mới để kiểm integration, không phải utility/performance.
  Local sm86 không chứng minh2×T4; compilegate mỗi rank vẫn phải đọc từ run.
- `runs/m-study/{tests.log,equivalence.json,smoke-audit.json,quota.json,quota-plan.json,batch.json}`.
  BatchSHA d0eef288592e50fa7b5cecdee4920fa37f20f39a1f2fb77d071be22d38834a48;16account đủ≥6,9h margin,cap6h/run,total96hcap.
- Source5overlay immutable: mobihet.py246b32790a85a1fe18017ad03375c7070cc88129ca8764993e2499d0862dc4a1;
  diagnostics8d32a58bb09baf6820352158453ea77aa958ce92ff75b747def7e9c1ed393f24;
  teacherf8c4df77c20b1d6fac9120fd0438448c4f993eb9e6ad582d987cac46cb9aa465;
  external38ca087bb05d64b545fc012e4ca82737b5f7bb24aa4cb080558cb75ed7358583;
  run_fl747fd50ba2a4ec7fbc3697e7a01f0b717e09e27c84884af0b7e626e1965bfe18.
- Helpers: `scripts/{prepare_m20,launch_m20,check_m20_startup,watch_kaggle_batch,watch_collect_m20,audit_m20}.py`.
  `watch_collect_m20.py` chỉstatus1200s→pull4process×8connections→audit20round; không gọi model/launch/cancel.
  Pull selective source/config/CM/probe/smallsnapshots/externalCM, **không fullprivatebank/executednotebook**;
  cần20GiBdiskfree. Pull cảERROR đểgiữbằngchứng; thiếuround/file thìjobfailed, không successclaim/restart.
- Chưa M kết quả. Train+val đãfit, không independentvalidation. Không formalprivacy/mixedarchitecture/contactdrop/50round proof.

## Bằng chứng đã biết, không làm lại vì quên

[L kết quả](../research/results-l10.md): L6/L7softfixedgateF1=.887030/.887433; noReal=.886315/.886861, fixed1 hơnadaptive;
softtarget vẫn hữu ích. Lợi riêngrealgate+.000715/+.000572 nhỏ, chưa ổn định. Nhóm<256 vẫn kémaugmentation vớisoftgate.
L r1cùngwarmupmath cóspreadsession.000516/.000588; chưaisolate, khôngCI/noisebound, khôngCPUbitexact→T4exact.
TimeDelayprior5,151→5,162%gầnnhưgiữ nguyên; traingap chưa xác định featurelimit/overfit/shift.

Fullsupportcohort audit `runs/m-study/support-cohorts.json`:848xe<256,median8/16trainclasses,
1555/34827testrows=4,465% absentlocal,9776testrowslocalclasscount1–16. Toànmạngabsent0,289% không phủ nhận vấn đề xe nhỏ.
[K](../research/results-k10.md), [H](../research/screening-h10.md), [J](../research/screening-j10.md) vàv1 giữ nguyên;
không mở lại bucket8/anchors/GMM/MADE thiếu bằng chứng mới.

L watch/collect jobs đều succeeded và completionđãconsumed; không restart/retrydeliveryhai jobL.
Không việc tay bắt buộc hiện tại. Bước tiếp: chờcompletion(workerđangchạy, khôngmodelpoll) → kiểmstate/log/artifact →
phân tích đường học/ba khâu theo đối chứng, rồi mới đề xuất phương án tổng; chưa tựlaunch batch khác hoặc50round.
