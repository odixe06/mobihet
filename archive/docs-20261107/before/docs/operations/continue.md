> Snapshot 07/11/2026; tài liệu hiện hành: [docs/operations/continue.md](../../../../../docs/operations/continue.md). Không đọc mặc định.

# Continue — bàn giao sống, 07/11/2026 02:41

Đọc [CONTEXT](../../../../../CONTEXT.md), [LESSONS](../../../../../LESSONS.md), [vận hành](../../../../../docs/operations/lessons.md).

## Quyền owner

- Owner07/11 **02:39 cậpnhật**: ưu tiênkernelcầnnốiphiên; **đượcpushngaymột sốthửnghiệmkhảthi bằngquota dư**, khôngcầnchờ07:00. BốnreceiverY50giữnguyên; forest42/43 dùng2accountdư>3h. Batchdài/ứngviênchưasẵnsàng vẫnchờquotareset07:00. Thayquyđịnhhoãntoànbộlúc02:28; kiểmquota/slotthật trướcpush.
- Tiếp tục tối ưu generator: thử → phân tích FAIL → cơ chế tiếp, không hỏi lại từng probe. Không đổi protocol/HP/loss/Q/floor; không chọn HP/round bằng official test, không nới verdict cũ.
- Được nối session bằng account khác, ưu tiên tài khoản còn >3h và slot trống. Chỉ chuyển checkpoint qua dataset private, verify SHA/config/state2rank/cùng round; không mount private kernel chéo account.
- Kaggle credential process-scoped, global login giữ nguyên. W&B online, private inline key đã được phép; remote history giữ key. Không Git/DVC push hoặc cleanup artifact nếu chưa được yêu cầu.
- Pull đa luồng nhanh: checkpoint 4 kernel×2file×4Range≈32 kết nối; exact owner/slug/version, size/Range/SHA trước import. Dataset verify dùng Range8, 2 stage song song.

## Kernel và round đã audit

Kiểm exact9refs lúc01:55: tất cả session trước COMPLETE (`runs/kernel-status-20261107.json`), không suy đủ đích từ COMPLETE.

| arm | round đã audit | đích | bước hiện tại |
|---|---:|---:|---|
| Y0 oracle42 | 50 | 50 | đóng audit; macro.917581, oracle diagnostic |
| Y1/Y2 prior42/43 | 20/20 | luật r20 | đóng, utilityFAIL, không nối |
| Y3/Y4 noQ42/43 | 27/26 | 50 | pull fullstate và chuyển owner >3h |
| Y5 nofloor42 | 20 | 50 | như trên |
| Y6 nofloor43 | 47 | 50 | như trên |
| Z0/Z1 standardized42/43 | 2/2 | 2 runtimegate | đã đóng runtimeauditPASS cả2seed |
| Z2/Z3 forest+quantile42/43 | chưa trainFL | 2 runtimegate | localpreflight/GPUfidelityPASS, dispatcherđãmởngay/freshquota trướcpush |

Z0/Z1 session1 dừng budget trước r2, compile2rankPASS/skip0. Audit1 PASS ở `runs/standard-t4/collection/Z{0,1}-r1`.
Kernel cuối: `trietbackup/mh-z0-stdcop-s42-t4r2-c2/1`, `odixeuit/mh-z1-stdcop-s43-t4r2-c2/1`; đãauditđủr2/skip0,
logresume2rank từr1 vàcachetarrestore0. `standard-t4-resume-20261107` succeeded02:28:58;
`runs/standard-t4-resume/{closure,artifact-recheck}.json` xácnhận. Chưautility50; batchutilitydài chờreset, forest2rđượcpushngay theoowner02:39.

Quota01:57 chưa reset: Yowner0.67–1.02h; còn anhtruong1909~4.96h,minhtriethihi4.91h,dinhan254.35h,emtrangtri3.97h,minhtrit063.42h,heykewtiie3.14h. Chọn lại live qua `runs/y-transfer/quota.json`/`plan.json`, không dùng số này như reservation.
Ytransfer đãkhóa receiver: Y5→anhtruong1909(cap4.26h),Y4→minhtriethihi(4.22h),Y3→dinhan25(3.73h),Y6→heykewtiie(1.63h).
Đây làallocationchờcheckpoint, chưareceiptGPU push. Freshquota02:40 ở`runs/quota-now-20261107/quota.json`: vẫn6account>3h,
4Yreceiver đãdành; emtrangtri3.966h/minhtrit063.416h dưchoforest. Exact6latestrefsY3–Y6/Z0/Z1đềuCOMPLETE,
0kerneltraintrongbatchtheodõi tạiquery (`kernel-status.json`). Forestworker loại4receiverY50, kiểm recentstatus/quota/access lại trướcpush.
Ypull32rangestream cònđangghi, khoảng19.94GB lúc02:40; vàiChunkedEncodingError đãretry từpartial, chưapull-complete.

## Worker sống và ownership

Ownerthread mọi job: `01a1b116-77ad-7d22-b83b-86020adaa518`, deliveryqueue. State dưới `/home/odixe/.codex/long-jobs/jobs/<job>/state.json`.

| job ID | supervisor PID | log | sở hữu |
|---|---:|---|---|
| `y50-checkpoint-pull-20261107` | 11264 | `runs/y-transfer/pull.log`, `Y{3,4,5,6}-pull.log` | `.range*`/state/cachetar trong4artifactY |
| `y50-crossaccount-resume-20261107` | 14247 | `runs/y-transfer/pipeline.log`, `<arm>/stage.log` | plan/owner-private datasets/notebook-x3+/Y50watch/collect/audit |
| `forest-quantile-t4-now-20261107` | 47001 | `runs/forest-t4/pipeline-now.log` | freshquota→push2seedT4ngaytrênquota dư→collect/audit2round |

Controller đãxácnhậnrunning. Không sửa source/ledger worker sở hữu khi còn chạy. Theo dõi:

```bash
tail -n 200 -F runs/y-transfer/pipeline.log
tail -n 200 -F runs/forest-t4/pipeline-now.log
```

WatcherY50cũ mất sau reboot; recoverywatcher `y50-wave2-resume-20261107` đãcancel bằngcontroller lúc01:58 đểhandovercrossaccount (cácKaggleparentđềuCOMPLETE). Worker mới giữ `runs/y-study/wave2/worker.lock`, không restart `keep_y50.py`/cron song song.
Notificationcancel đãxửlý: state xácnhận SIGTERM01:58:13, commandchỉ`--collect-only`; logY6audit47PASS trướcchờquota.
Kiểmartifactthực lại4endpoint27/26/20/47: contiguoushistory,CM→10metric/sourceSHA/finite snapshotPASS;
bằngchứng `runs/y-transfer/cancelled-watcher-artifact-check.json`. Hai workerpull/crossaccount cònprocessđúngidentity;
pullcomplete chưa có, chưaGPUreceipt mới. Không restart watcherđãcancel hoặccoi47là50.

Forestpreflightjob `forest-quantile-preflight-20261107` đãsucceeded02:27:49; artifactauditPASS, khôngcònsống.
Dispatcher `forest-quantile-t4-20261107` đãcancel02:28:42 đúngownerđổiưu tiên, sauGPUfidelityPASS/trướcbatch/push.
Khôngtrainingremote bịdừng. Log/statecũ giữnguyên; notificationcancelđếnmuộn chỉackhandover, khôngrestartdispatchertrước07:00.
Job0700 `forest-quantile-t4-0700-20261107` đãcancel02:39:59 theoownerchochạyngay; chỉhủytimerlocal, chưacóGPUkernel.
Jobnow47001thaythế, deliveryqueue cùngownerthread, ghi`FOREST_LAUNCH_AUTHORIZED_NOW`; khôngrestart cácdispatchercũ.

## Pipeline tiếp tục tự động

1. Parallel checkpointpull giữpartials, pin c2/version1; stages chỉ bắt đầu sau `runs/y-transfer/pull-complete.json`. Parent round/fingerprint trong `parents.json`.
2. Crossworker chọn4account idle >3h, source/state round đãaudit; tạo private resume.bin+manifest từngowner, downloadRange/SHAverify, import2rank/HP/cfg/RNG trước data decode. Giữ cache npy+Inductor. Temporary verifydownloads/range chunks dojobtạo được dọn sauvalidate; source artifact vàuploadarchive giữ.
3. Push các kernel`-x3`, giữout/W&BID gốc. Quota ngắn thì safe stop, collect+audit partial, nối cùngowner`-x4` sauquota đủ; pollKaggle1200s. Crossworker không nới gateutilityY50, chưa thêmarmY.
4. Y50full50 audit xong publish canonical `runs/y-study/wave2/closure.json`, chạy `analyse_y50.py`; cập nhật [resultsY50](../../../../../docs/research/results-y50.md), proposal vàlessons.
5. Standardresume đãđóngr2/source/CM/normalization/teacher/finite/resume2rank/cachePASS. Chỉruntimeproof;
   khóa utilitycontrol2seed/endpoint50 sau07:00, không chọnHP từr2. [Thiết kế](../../../../docs-20261109/research/screening-standard-t4.md).
6. [ForestT4](../../../../docs-20261109/research/screening-forest-t4.md): `mobihet_forest_quantile`;8testsPASS, localdriver2seed4round/crashresumeartifactPASS (`runs/forest-production/preflight-artifact-audit.json`). GPUfidelitypairTV .199355→.067299/singleton≤+.02/classPASS trênpaneltrain/val cũ. **Đượcchạyngay bằngquota dư**; worker tựchọn2idleaccount>3h/cap2.5h khôngtrùngYreceiver, pushZ2/Z3 đích2round. Khôngclaimđãpushkhichưareceipt. Source/hp/codec đãkhóa, khôngsửaworkerđangchạy. Bootstrap1704951B/xe (thêmselected19finecountdownlink đểkhớp12jointtables), Q1088B/rgiữCore. StaticSHA2rank/countconsistency/samplerbenchmark/compiledbackward/CMteacher audit; localPASSkhôngutilityFLproof.
7. Sau runtimegate, tiếp các thửnghiệm generator trongquyềnowner: khóa utilitycontrol/2seed/endpoint trước50round. [Kếtquả và hạnchế](../../../../docs-20261109/research/copula-marginal-diagnosis.md). Không tựgọiarmlocalFAIL đãPASS hoặc launch unchanged failedarms.

## Generator đã thử, tránh lặp

- [Pairreweight](../../../research/results-pair-generator.md): FAIL, ESScollapse.
- [Directforest](../../../research/results-direct-generator.md): pairTVtốt nhưngsingletonR xấu, FAIL.
- [Forest+quantile](../../../research/results-marginal-direct-generator.md): CPUlocalfeasibilityPASS; originaljobJSONfailed đãrecover từCM/source/plan/samplehash, khôngrefit. Torchport/8tests/localdriver2seed/GPUfidelityPASS; pipelineT4đãmởngaytheoowner02:39, chưaKagglereceipt.
- [Marginal-only](../../../research/results-marginal-copula.md): TSTR.8533 nhưngpairgateFAIL; chưaKaggle. Với continuous inverseCDF, lýthuyết transportmarginal tươngứngstandardizinglatent; tránh nhânarm tươngđương khi chưaisolatecodec/quantileapprox.
- [Standardlatent](../../../research/results-standard-copula.md): TSTR.8528 nhưngvalTVgateFAIL; diagnostictrainfitting−80% vàtemporalshift cóhồsơ. PyTorchpreflight2seedPASS; runtimeT4r2/2seed/resume2rank/cachePASS. Khôngutilityr50proof.

## Manual / blocker

Giữ máy bật cho localworker; remoteKaggle vẫn chạy khi máy tắt nhưng phải phục hồiworker/checkpid/lock sau reboot. Hiện không cần owner thao tác khác. Nếu quota/inputs/upload/pull lỗi: đọc state/log/receipts, phân biệt operationalfailure vàmethodFAIL; không blindlyretry push hoặc restart training.

Kiểmactive links: `python scripts/check_docs.py`. Không đọc archive mặc định; snapshot cũ chỉ phục hồi khi owner yêu cầu. Không commit/push trong phiên này.
