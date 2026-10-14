# Hybrid full — ledger

Owner chốt100%train80%/20client/50round/seed42, đủ7matchedcontrols;
điều kiệnlaunch:V8utilityPASS đủseed42/43 đãpull/audit.
[Contract nguồn đãfreeze](../protocol.md).

## 2026-10-14 — terminal collect/audit xong

4 kernel version1 đã pull/audit đủ 376 file, không thiếu; 7 arm/18 complete round.
`hybf1`: local/DML r2; `hybf2`: Gaussian/Hybrid r3;
`hybf3`: Gaussian-own r3/DML-own r2; `hybf4`: Hybrid-own r3.
Hai kernel ERROR do worker budget stop/parent assert3round, worker rc0, không model failure.
Không rerun r1; continuation lấy checkpoint r2/r3 riêng từng arm, target/LR50 giữ nguyên.
[Kết luận khả thi](feasibility-20261014.md), [10metrics](latest-rounds.md), [report](report.md).

Source/executed version/BN/state/codec/pred→CM/metrics và full-test-label/orderhash PASS.
42 bounded raw-test/client0 inference checks PASS; [proof](../../../../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/probes/inference-proof.json).
Budget theo pairing hiện tại/worst-round+dư15%≈314,70h phiên còn lại, chưa repeated setup/import;
[JSON điều kiện](../../../../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/probes/budget-remaining-20261014.json), [quota14/10](../../../../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/probes/quota-20261014.txt).
Không gọi calibration3/50 đủ hoặc r50 complete: 3 arm mới2/3 calibration rounds.
Các mốc RUNNING/collector bên dưới là lịch sử13/10, đã superseded.

## Chuẩn bị2026-10-13

**Cập nhật 15:24 UTC+7:** V8 cả hai seed đã audit/utility PASS; 4 notebook bên dưới
đã PUSHED version1 lúc 07:57 UTC (14:57 UTC+7). [Reservation](../../../../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/probes/reservation-20261013.json).
W&B exact fingerprint: 7/7 arm có heartbeat SGD round1 hữu hạn, applied>0,
skipped0–1 tại snapshot 08:25 UTC; không thấy lỗi guard. Đã ngừng health polling.
[Snapshot](../../../../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/probes/startup-health-latest.json). Collector terminal vẫn chạy; chưa có full round3/r50 audit.
Các ghi chú “chưa PUSHED/chờ seed42” bên dưới là mốc chuẩn bị cũ, đã superseded.

Source riêng `proj/`, mọi learned model DAGSNet; khôngsửaMH/fullH1H2 đãtrain.
7arm gồm local/Gaussian/DML/Hybrid/Gown/DMLown/Hybridown, luôn2DAGScompute;
CEsynthetic cho cảprivate/proxy, KD chỉbatchreal, dummy evalnoise giữBNcontrols.

Localbounded7arm×3round/5exactresume/strictBN/payload/pred→10metric/inferencePASS,
peak2913MiB, log `papers/hybrid-full-smoke-initial1-20261013.log`.
4notebook/8controllercases/keyboundary/metadataPASS; chưaT4compile/productionF1proof.
Private source dataset4owner `mobihet-hybrid-full-20261013-src` đãUPLOADED/READY.
12owner-scoped inputfilelistingchecks PASS; [proof](../../../../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/probes/input-access-proof.json).
`DatasetStatus` foreignslug404 khôngthayListFiles/mount/datahashgate.

| PUSHED version1 notebook | owner | arms |
|---|---|---|
| hybf1-s42-20261013-cal-v1 | minhtrit06 | local/DML |
| hybf2-s42-20261013-cal-v1 | odixeuit | Gaussian/Hybrid |
| hybf3-s42-20261013-cal-v1 | trietbackup | Gown/DMLown |
| hybf4-s42-20261013-cal-v1 | odixe0502 | Hybridown |

V8 seed42/43 paired PASS đã mở calibration full theo quyền chủ dự án chốt.
T4 đầu lấy3round theofixedLR50 đểđo compute/budget rồicheckpointcontinuation tới50.
Không dùng oldH1/H2 mộtDAGScontrols thayfullHybridcontrols.
Parent4,8h/worker4,2h; freshquota+dư15% trướclaunch, sauđóbudget theoactualT4.
W&Bproject `mobihet-hybrid-full`, fail trướctrain nếu thiếukey/online;
privateinlinekey0600/ignored giữchi phíversionhistory đãđượcownercho phép.

Git/DVC add/push/backup/cleanup khôngthực hiện. Khôngfullr50/R1/R2/DPclaim.
[Phiêncollect/quota/chốtremaining](../../../../../archive/operations/continuation-2026-10-13.md).
