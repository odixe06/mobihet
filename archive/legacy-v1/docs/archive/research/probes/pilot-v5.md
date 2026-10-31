# V5 — anchors và synthetic books

2026-10-12. Sốđo gốc: [report](../../methods/synthetic-knowledge/builds/anchor-synthetic-books/results/report.md), [gates](../../../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/probes/gates.json),
[figures](../../methods/synthetic-knowledge/builds/anchor-synthetic-books/results/figures.md). H/K2b teacher/head; G/J private/gp. Khôngđồngnhất teacher vớiprivateIDS.

## Kernel anchors

18run đủ20round cảhai seed đãaudit trướccleanup; khôi phụcmetadata/source/markers/results bằngMD5.
Small/mediumteacher pairedFAIL: FLthuaGaussianhardlabels vàlocal; khôngmởprivateKD/contact.
Nearuniformtarget làdấuhiệu từprobe; cầnactualartifactdiag, khônggánnguyênnhân duy nhất.

## Gradient books IPC1

14run đủ20round, source+allstate/encoderancestry/bookpayload/peermap/steps/prediction→CM10metrics
+sampledinference PASS. Actualbaseline/cost_observed/resultminutes kiểmđộc lập:
[COST_AUDITED](../../../../archive/tools/audit_book_cost.py). Cost2×PASS nhưngqualitypairedFAIL.

| seed | grad-trained−diag | −noisebook | −ownbook | −mean-trained | −grad-frozen |
|---|---:|---:|---:|---:|---:|
| 42 | -0.075148 | +0.002624 | +0.037565 | -0.001466 | -0.000442 |
| 43 | -0.079487 | -0.000104 | +0.036816 | +0.001946 | -0.000283 |

CommonguardsPASS; continuousGaussian tốt hơn1point/class. Gradientmatching khônghơnmean đều cảhai;
currentencoder khônghơnfrozen cảhai; interaction nhỏ/khôngổnđịnhdấu. Khôngđủbằngchứng hiệuứng
objective/encoder riêng theo2×2gate; fit/đạo hàmfinite khôngphảiutilityproof.
Khôngkếtluận toànhọcondensationFAIL hoặcgiớihạncompute lànguyênnhân: pilot này nằmtrongtrần2×.

## IPC4

14run đủ20round đãaudit; cost2×PASS, qualitypairedFAIL.

| seed | grad-trained−diag | −noisebook | −ownbook | −mean-trained | −grad-frozen |
|---|---:|---:|---:|---:|---:|
| 42 | -0.049552 | +0.000377 | +0.027322 | -0.003737 | -0.003588 |
| 43 | -0.056777 | +0.001465 | +0.019397 | -0.005186 | -0.001237 |

Objective/currentencoder contribution đềuFAIL. KhôngchọnIPC/round/head saukếtquả đểcứuIPC1;
IPC4gatekhóa riêng. Đủ6kernel/46run ởV5, không cònchờaudit.

## Phạm vi

Ownbook giữcommonGaussian/prior, khôngphảihoàntoàn khôngFL. Onlysynthetic noise-initbook gửiRSU
được phép; khôngraw/latent/greal export; exact-duplicateguard khôngDP.
OldH audit táidùng; G/J mới toànartifact. Ustage chưadeadline/R1/R2mixedarchitecture.
