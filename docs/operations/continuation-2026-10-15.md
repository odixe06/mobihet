# Bàn giao continuation và trial — 15/10/2026, UTC+7

## Quyết định chủ dự án trong phiên

- Ba account mới mỗi account30h tuần; không query lại quota. Dùng owner-scoped OAuth, không đổi login global.
- Ưu tiên DML/Hybrid và controls cần thiết; tạm hoãn standalone Gaussian diag/lr4. Không chọn best round/HP từ official test.
- Được sửa adaptation fail theo nguyên nhân rồi trial; không rerun fail nguyên trạng hoặc kéo dài để tìm win.
- Đề xuất mới phải triển khai thử. Đã chốt ClassTail α0,25/λ0,25, r20 hai seed, controls/gates;
  chốt riêng log-bin và probe gate trước khi thực hiện.
- Giữ DAGSNet/66feature/partition/R3, W&B online và finite-contact contract. Không Git/DVC add/push/backup/cleanup.

## Bằng chứng cũ đã cập nhật

10 kernel đợt14/10 terminal, pulled exit0; executed version1/source/fingerprint/checkpoint/BN/payload/
applied-step/LR/prediction→CM→10metrics đã audit. Status không thay bằng chứng này.

| full build | latest audited committed rounds |
|---|---|
| H1/H2 | local16; Gaussian14; diag18; lr4 17; Gaussian-own18; DML12; DML-own9 /50 |
| Hybrid | local4; DML4; Gaussian7; Hybrid7; Gaussian-own7; DML-own4; Hybrid-own6 /50 |

[H1/H2 10metrics](../methods/shared-baselines/builds/federated-baselines-full/results/latest-rounds.md),
[Hybrid 10metrics](../methods/gaussian-proxy-hybrid/builds/hybrid-full/results/latest-rounds.md).
Reports mới229/117evaluation rows đủ10metrics; mọi official-test score mô tả, không điểm dừng mới.
Các verifier inference14/10 chỉ bao phủ round cũ đã nêu trong proof; không tự mở rộng sang checkpoint15/10.

BoundaryCells cả hai output đã audit; seed42 11arm/148round, seed43 11arm/169round; chưa toàn20arm×20round.
Direct P2 r20 group/private/gp:0,504056/0,509389; Gaussian cùng controls0,630889/0,640571.
Chênh−0,126833/−0,131181 đã bác cổng hơn strongest-reference≥0,015 ở cả hai seed.
Δown+0,009266/+0,011509; seed42 còn trượt cổngown≥0,010. Seed42 shuffle mới19/marginal3,
không cần kéo dài chúng để gọi FAIL vì một điều kiện cần khác đã bị bác. Không suy nguyên nhân interaction từ marginal chưa đủ.
P2 direct đóng, không đổi weight/dictionary/head để cứu. P4 direct và teacher P2/P4 vẫn thiếu utility, tiếp12arm/seed.
Actual P2 target packet: group balanced2.048 probe phủ95,65%, entropy0,915, table diagnostic F1≈0,129;
P4 discrimination≈0,280/coverage100%, chưa DAGSNet utility. [Forensics/scope](../methods/boundary-cell-counts/builds/boundary-cell-pilot/direct-target-forensics-20261015.md).

V9 M512 saved-target forensics hai seed đã xong, chỉ restore4config/resume files, không SGD/test/tune.
95–96% correction energy chung giữa anchors, bootstrap≈95%, group kernel entropy≈0,955 dù coverage>99%.
Hỗ trợ thiếu spatial residual signal; không cô lập toàn BN/fit/extrapolation. Không nối adaptation cũ.
[Báo cáo và scope](../archive/methods/anchor-function-residual/builds/anchor-residual/results/forensics-20261015.md).

## Batch full đã chuẩn bị/push

[Plan ngân sách](continuation-plan-2026-10-15.json):6session×9h=54h; dư15%=62,1h.
Mỗi owner2session,20,7h gồm dư, dưới30h chủ dự án xác nhận; đây không là đủ r50 hoặc reservation service.

| owner / slug sau `mobihet-` | arm và audited parent round |
|---|---|
| annhtnt / hybf1-s42-20261015-q1-v1 | Hybrid local4 / DML4 |
| annhtnt / hybf2-s42-20261015-q1-v1 | Hybrid Gaussian7 / Hybrid7 |
| takids1909 / hybf3-s42-20261015-q1-v1 | Hybrid Gaussian-own7 / DML-own4 |
| takids1909 / hybf4-s42-20261015-q1-v1 | Hybrid-own6 |
| accanniez / full1-s42-20261015-q1-v1 | H1/H2 local16 / Gaussian14, reference cần thiết |
| accanniez / fulld-s42-20261015-q1-v1 | H1/H2 DML12 / DML-own9, gộp từ hai audited parents |

Mỗi notebook có private same-owner checkpoint dataset, full round history + frozen source package.
All-page names/sizes và CPU **production import** SHA/state/exact last round PASS trước GPU push.
Notebooks giữ lịchLR50 và scientific source byte; source40module mh không đổi. Parent checkpoint không từ W&B.
`CPU_MOUNT_VERIFIED.json`, `controller-preflight.json`, `checkpoint-dataset.json`, `resume-plan.json`
trong từng notebookfolder là proof; source/private/CPU output giữ đúng build.

## Các trial

- [ClassTail](../methods/class-tail-risk/builds/class-tail-pilot/README.md):8arm/seed, calibrationr2 của lịch20,
  minhtriethihi seed42/khanhmay0304 seed43, budget3h/seed. Local gradient/codec/4arm exact-resume/mass PASS.
  Cả hai terminal COMPLETE/pull/audit:16/16lớp và membership diff14,57%/13,61%; paired mechanism PASS.
  [Đủ10metrics208evaluation rows](../methods/class-tail-risk/builds/class-tail-pilot/results/report.md).
  Seed42 khởi động trễ; owner Console xác nhận2T4/own+pooled, sau đó W&B đủ8runr2 và terminal audit đã xác nhận.
  Không còn vấn đề locator hoặc yêu cầu owner đọc Console; không retry/cancel session đó.
  Utility exactr2→r20: `catbaochau/mobihet-ct-u-s42-20261015-v1` và
  `khanhmay0304/mobihet-ct-u-s43-20261015-v1`,2,5h/session, private checkpoint dataset+CPU import gate.
  Đã **PUSHED version1 cả hai**, dataset208file/seed PASS, CPU exactr2/all-model/SHA gate PASS,
  actual emitted controller8cases PASS. Giữ40mh+4proj source và fixed20schedule; chưa utility result.
  [Utility startup](class-tail-utility-startup-2026-10-15.json): cả hai seed local/fixed đã vượt warm-upr2 lênr5–7,
  no nonfinite trong snapshot; own/FL/controls/reference còn theo queue. Ngừng health polling run đã stable.
  Không gọi calibration là utility. Benign guard giữ so local và strongest reference như proposal.
- [CountSketch sửa](../methods/proxy-countsketch-repair/builds/contractive-ef-pilot/README.md):
  minimal CPU FP32 repro residual2×; decoded-patch line search,80production-size algebra cases và bounded
  medium exact-resume PASS. Hai seed7arm/r20,3h/session; same two owners như ClassTail, tối đa2GPU sessions/owner.
  Giữ original V2-A controls/gates; không chạy lại unstable original EF. Chờ terminal audit trước nhận xét IDS.
- BoundaryCells continuation: odixeuit seed42/minhtrit06 seed43,5h/session. Chỉ P4 direct +P2/P4 teachers,
  mỗi nhóm đủ own/shuffle/marginal. Import direct4fl từr2/r12, teacher2/4fl từr2; các own/shuffle/marginal chưa train bắt đầu fresh.
  Strong references giữ tại parent đã audit, không train lại. Source/HP/gp/target-mass/cổng cũ không đổi.
- EMA/ResponseRisk vẫn chưa triển khai trong phiên; không gọi hai recipe này thành phương pháp FL mới đã thử.

Hai trial đầu combined worst6h×1,15=6,9h/owner; snapshot cũ9,47h/7,19h đủ envelope.
Boundary5h×1,15=5,75h/owner, trong snapshot8,39h/8,88h.
[Quota cũ](quota-existing-2026-10-15.json), [owners Hybrid cũ](quota-hybrid-owners-2026-10-15.json).
Không dùng hai GPU trong một session như hai giờ quota account.

Utility mới: catbaochau2,9526h đủ2,875h gồm15%; khanhmay0304 snapshot6,4918h trừ tiếp toàn3h
repair+15% cho repair và utility:6,325h, vẫn vừa. minhtriethihi hiện0h free do reservation11,53h,
không đặt thêm utility ở đó. [Quota utility](quota-class-tail-utility-2026-10-15.json), [owner thay](quota-class-tail-alt-2026-10-15.json).
Ba account mới không query quota trong toàn phiên. Budget target không chứng minh mọi arm hoàn tấtr20.

## Trạng thái và thu kết quả

[Full startup](startup-batch-2026-10-15.json), [new-batch status](new-batch-status-2026-10-15.json)
và [pilot startup](pilot-startup-2026-10-15.json) có timestamp riêng; dữ liệu W&B là operational snapshot.
Không dùng state `finished` kế thừa hoặc round của parent để gọi resumed worker đã chạy.
Snapshot/follow-up mới:11/11full arms có fresh heartbeat; Hybrid DML-own tronghybf3 đã vượtparent4 lênr5,
applied1023, sau owner Console xác nhận2T4/require-resume/full-data PASS. Không còn yêu cầu Console pending.
CBC direct4FL hai seed đã chạy vượtparent, teacher/reference còn theo queue.
Repair hai seed đã thấy localr20/ownr10–11/denser1–2 finite; repaired EF còn chờ queue, chưa T4 utility proof.
Stable thì ngừng health polling, vẫn collect terminal. Observed instability → báo owner dừng.

Collectors/helpers, env `nckh`, chạy từ root:

```bash
python methods/shared-baselines/builds/federated-baselines-full/scripts/collect_full.py
python methods/gaussian-proxy-hybrid/builds/hybrid-full/scripts/collect_hybrid_full.py
python methods/class-tail-risk/builds/class-tail-pilot/scripts/collect_tail_calibration.py
python methods/class-tail-risk/builds/class-tail-pilot/scripts/collect_tail_calibration.py --utility
python methods/class-tail-risk/builds/class-tail-pilot/scripts/report_tail.py
python methods/proxy-countsketch-repair/builds/contractive-ef-pilot/scripts/collect_contract.py
```

Hai full collectors skip AUDITED cũ, dùng resumable Range, frozen verifier root đúng sau layout.
ClassTail collector single-pass, không tự launch utility; kiểm exact executed/source, toàn20model strict,
labels fixture/order và pred→10metrics, rồi ghi paired mechanism gate.
Boundary utility collector gọi `collect(folder, utility=True)`; continuation có12arm, không áp calibration-summary5arm.
CountSketch collector kiểm thêm non-increasing decoded residual ở mọi saved round kể cảr1; đã qua saved medium3round local.
Chỉ tính cổng original V2-A khi đủ7armr20, dense teacher và EF contrast tách riêng; hai seed/R1/cost chưa thay bằng algebra.
Mọi kernel ERROR phải pull/audit committed rounds trước quyết định retry; không restart r1.

## Bảo toàn và việc tiếp

Private W&B inline key theo quyền đã chốt, chỉ private_push ignored0600; platform private version history giữ key.
Không key trong source/checkpoint datasets hoặc docs. Restore fixture/V9 chọn lọc giữ MD5/.dir; không DVC snapshot mới.
Git HEAD đầu phiên `4a6893eeb8eaaeed799a2b59680a7b57cef13d4a`, staged content rỗng; không commit/push.

Tiếp theo: terminal collect/audit trial, chốt paired gates; chỉ nối utility/round mới
khi có bằng chứng và budget. Round50, R1 contact hữu hạn và R2 khác kiến trúc vẫn chưa đủ proof.
Thao tác tay: chưa có yêu cầu mới; account-scoped CLI dùng saved credentials, MCP native không callable trong phiên.

Chốt snapshot:14GPU notebooks mới version1,2calibration COMPLETE/audited +12RUNNING
(6full,2BoundaryCells,2repair,2ClassTail utility). RUNNING không thay training completion/quality.
Verification15/10 PASS:40module imports/SHA/copied-runtime,64DVC identities/remote manifests,
433frozen files,0missing links; private notebooks ignored0600. Git HEAD/index content giữ nguyên.
[Proof hiện tại](layout-verification-2026-10-15.json), [log](layout-check-2026-10-15.log).
Hai yêu cầu Console đã được owner trả lời, startup được kiểm tiếp; không còn thao tác tay pending.

[Layout trước](reorganization-2026-10-14.md) · [Đề xuất mới/nguồn](../research/new-directions-2026-10-15.md) ·
[Catalog](../proposals.md) · [Sổ lỗi](notebook-errors.md).
