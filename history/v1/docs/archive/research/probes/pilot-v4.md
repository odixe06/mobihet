# V4 — bằngchứng đangthu

2026-10-12 UTC+7. [MALAmanifest](../pilot-v4-mala.md), [supportmanifest](../pilot-v4-support.md),
[ledger](../../methods/gaussian-sampling/builds/support-mala/results/rebuild.md), [report](../../methods/gaussian-sampling/builds/support-mala/results/report.md).
Chỉartifactterminal đãAUDITED vào đây, không W&B/status nhưF1proof.

## Support cảhai seed đãverify, pairedFAIL

Kernelodixeuit/mobihet-v4s-s42-20261012-v1 vàminhtrit06/mobihet-v4s-s43-20261012-v1 COMPLETE,
10arm×20round; executedsource/fixture,
allcheckpoint/state/supportpacket/pred→clientCM→10metric vàsampledinference PASS.

| arm | group/private/gp macro-F1 | benignF1 |
|---|---:|---:|
| s_diag |0,590447|0,417028|
| s_marginal |0,576716|0,359656|
| s_structural |0,582204|0,362607|
| s_own |0,472941|0,314308|
| s_shuffle |0,514317|0,417024|

Seed42: structural0,579132 vsdiag0,581252;−diag−0,002120,−own+0,101527,
−shuffle+0,077441,−marginal+0,003466.
Seed43: Structural−diag −0,008243,−own +0,109262,−shuffle +0,067887,−marginal +0,005488.
Commonlocal/old/benignguardPASS nhưngdiagcontrastFAIL; **pairedFAIL, khôngcontact/rerun**.
Fidelity0invalid khôngđủ quality, marginalsupport bảnnày khônghơndiag. Firstconstraint cólợi contrast
trongseed43 nhưngseed42khôngđạt+0,005; khônggọi contributionpaired hoặccausalmọi feature.
Khôngsuy supportkhônghữuíchtoànhọ hoặcđây nguyênnhân duy nhất; khôngtune/rerun y nguyên đểcứu.
Bytebootstrap thật23568B/xe(sharedstats), own chỉprior200B; Ustage chưadeadline/frame/ACK.

## MALA đủ hai seed, paired FAIL

Hai kernel COMPLETE, pull/source/executednotebook/checkpoint/payload/prediction/10metric vàsampledinference PASS.
10arm×20round; source/runtime2×T4 pin đúngmanifest. Khôngrerun/contact.

| seed | FL MALA−weightedGaussian | −diag | −ownMALA | −shuffleMALA |
|---|---:|---:|---:|---:|
| 42 | +0.002930 | -0.013734 | -0.002061 | -0.003757 |
| 43 | -0.004897 | -0.025717 | -0.000932 | -0.000572 |

Commonlocal/old/benign guardPASS cảhai nhưng methodcontrastsFAIL. Acceptance cao khôngđảm bảo
MALA mangtri thức quầnthể; thuaown/shuffle cho thấychưa cóbằngchứng lợi vậnchuyểnratio gộp.
Khônggán clipfloor/support/mixing làmnguyênnhân duy nhất; chưa đoautocorrelation/ESS củachain.
HistoricalCscoreprobe chỉsnapshottrướcMALA, khôngcoi làscore thực tếteacherV4.

Startup đã rõ: terminaloutput đủ; RUNNING/chưaW&B/HTTP404 khiđợi khôngphải numericalfailure.
KhôngcầnGUI/manual/rerun. [Report](../../methods/gaussian-sampling/builds/support-mala/results/report.md), [figures](../../methods/gaussian-sampling/builds/support-mala/results/figures.md).

## Khôi phục saubackup

HaiS kernel audit trướccleanup: khôi phụcmetadata MD5, táidùngproof cũ, không kiểmweightslại.
HaiM kernel audit mới toànartifact trongphiênnày. [Restore/collect](../../operations/live-2026-10-12.md).
