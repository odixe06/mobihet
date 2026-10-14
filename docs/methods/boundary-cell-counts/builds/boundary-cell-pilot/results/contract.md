# BoundaryCells calibration/pilot contract — initial1

Frozen scientific source: `proj/`; inherited MH modules copied without changes into each input snapshot.
All learned models DAGSNet, no raw/perturbed records or per-record logits/gradients on FL wire.
FL wire: cell counts/contributor indicators, aggregate cell targets, paid priors/dictionary only.
Trusted sum emulator, no cryptographic secure aggregation or DP claim. R2 only design.

Owner locked P2/P4, B4, edges−1/0/1/right-tie, alpha1, lambda0,5; direct and intermediate.
Every learned model DAGSNet. Static statistics calculated before first SGD on existing pilot train.
Cell suppression: any positive label with<5contributors or total<32; suppress target values on wire too,
use uniform bytes with false eligibility, skip student auxiliary loss. Do not alter gates to recover failed arms.

Teacher intermediary: local DAGSNet fit20random batches/round, same LR20schedule; no server teacher trained
on real client rows. Direct/control arms fit identical diagnostic teacher but ignore its outputs in student.
Teacher missing eligible cell uses hard local CE; private still skips auxiliary at that x.
Private: one local epoch, batch512, AdamWwd0 reset/round, cosine1e−3→1e−5 over20round,
clip1, FP16autocast/FP32CE, applied>0/skipped≤16. Teacher T1, no inverse-CE KD weights.
Stats/teacher/student run locally inside each simulated client; no parameter averaging among clients.

gp primary: CE target mass averaged per applied optimizer batch, preserving each component's mean-loss
normalization and its coefficient. Smooth each class mass by1/N_local, normalize; subtract logπ_eff,
add paid population prior. This remains a label-shift assumption; legacygp/head diagnostic only.
Saved mass/π_eff enables audit; no holdout/test prior fitting or best-round choice.
Population gp prior is decoded from the paid FP32 downlink, not read from server count state.
Unused rare-cell targets are uniform on wire; mask alone must not expose their probabilities.
Own controls share the eligibility metadata to hold coverage fixed, with paid count uplink/mask/dictionary;
they do not consume population cell target vectors. Therefore own is a boundary-target control,
not a claim of zero communication. Local-single is the no-target/cost reference.

Calibration: seeds42/43, each single-DAGS baseline +directP2/P4FL +teacherP2/P4FL, two complete rounds
of fixed20roundschedule. Every round evaluate full existing old/group holdouts; all10mean-client metrics,
per-class/benign, per-client predictions/confusion, weights inclBN, separate resume and markerlast.
**No official test.** R2/finite-contact utility not proven by calibration.
Measure mean `algorithm_seconds` of2commits **before eval/commit**, candidate/baseline≤2.
Baseline`bc_single` has exactly one DAGSNet and no diagnostic-teacher work; no dummy-inflated denominator.
Candidate cost includes diagnostic teacher20steps and its inference; report actual implementation overhead.
Cross-T4 comparison recorded with physicalGPU; no inference from local RTX cost. No continue past2rounds
inside this calibration notebook. Keep FAIL outputs; fullpilot only if measuredcostPASS.

Pilot utility locked by owner: each seed42/43 private/group/gp r20:
FL−own≥0,010; FL−shuffle≥0,010; FL−strongest(local,class-mean,Gaussianfull)≥0,015;
benign F1 decrease≤0,005. Direct/teacher and P2/P4 gated separately.
Same-scientific-source resume from calibration is allowed only after audited exactsource/round/state proof.
Utility adds matched2DAGSlocal/classmean/Gaussian and own/shuffle/marginal; no R1claim before deadline test.

W&B online auth before preparation/train, key failfast; heartbeat loss/LR/applied/skipped and allmetrics.
Private inline key only ignoredprivate_push0600; never source dataset/public notebook/log/chat.
Private executed notebook/version history still retains inline secret under standing owner exception.
Source frozen before push; any correction needs newtag/preflight, never overwritePUSHED source.
