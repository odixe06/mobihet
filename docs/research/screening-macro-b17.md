# B17 — prior-coordinate / real-NCKD trên CBB, khóa 11/11/2026

Owner: tạm bỏ guard phụ, ưu tiên macro-F1; đã đồng ý thử tiếp hai hướng nếu cócơsở.
Chỉ trao thống kê/xác suất, **không gửi realrow**. DAGSNetW1 duy nhất; schema hỗ trợ kiến trúc khác.
[Review số/curves/privacy](../../archive/docs-20261113/research/macro-first-review-20261111.md) phân biệt cơ chế cũ và adaptation hiện tại.
CBB vẫn phươngphápchính; không ghép hai thay đổi hoặc SAM.

## Evidence → mechanism → falsifier

- Y50prior trên Core full5343/cos50r20: macro+.003345/+.002890, gain r15→20 giữ;
  bị loại theo TD/absent cũ. Hợp mục tiêu macro mới nhưng chưa chứng minh chuyển gain sangKM/Q8.
- W20real-NCKD trên mixed-input bank full5343/cos50r20: macro+.002461/+.002229 vsQ6/Q7;
  còn tăng ownmacro r15→20 nhưng relativegain co nhẹ. CBB stream bỏbank, nên phải kiểm adaptation riêng.
- B13–B16 không ứng viên nào đạt+.002macro cả2seed kểcả bỏguard. Không kéo dài/tune các loss đó.
  Pilotcos50 cóLRr20≈6,76e-4, còn dư địa; không ngoại suy tuyến tính hoặc cộng gain cũ vào CBB .9078.

## Arm khóa

`methods/mobihet_macro.py`, **C/P/NR × seed42/43** =6kernel.

**C** `macro_mode=cbb`: delegate CBB exact, không counter/loss mới.

**P** `macro_mode=prior`: giống công thức Y50, vector `s=log(1/16)−logπg`,
`πg=(global_count+1)/(global_rows+16)` dùng đúng Laplace convention.
RealCE(z+LA) giữnguyên; synthetic hardCEr1/KLr≥2 trên `(z_synth+s)`;
Qreal thu online `softmax(z_real+s)`, sauđó codecQ8/countaggregate/Q8 nhưCBB.
Q đo và student cùngcoordinate, không shift target lần nữa. Evalrawz giữnguyên.
Không là PHsuy luận b11. Globalcounts đã thuộc bootstrap; extra simulatorallreduce chỉ tính lại total,
không là clientquery/frame mới, không thêm field hoặc đếm byte mới cho cùngcounts đãtrao.

**NR** `macro_mode=real_nckd`: giữnguyên synthetic per-rowKL/rawQ/rawlogit, realCE+LA;
thêm trên realrow: `(1−Q[y,y]) KL(Q_non-y || softmax(z_non-y))`, mean theo realmasks từngclient,
coefficient1, temperature1. Trueclass không nhận gradient từ NCKD, CE vẫn phụ trách trueclass.
Giữ **binary validity gate của W20** `argmax Q[y]==y`; không margin/qualitythreshold/tuning.
Không restrict non-target set theo classlocal (đó làX20khác). R1 chưaQ delegate CBB loss exact.
Không bank/query/extraepoch/forward, không gửi model/gradient/BN, không realdata pool.

P thay logitcoordinate và measurement đồngbộ; NR thay reception trênreal. Haiarm tách để đọc mechanism.
Defaults CBB2round exact; NRr1exact trên CPU không claim wholeT4trajectoryexact; actualr1CM vẫn phải báo.

## Cấu hình / byte / budget

Sim200panelseed7, foldtrain1.476.064→val211.118; **r20cosine20** như thang sim hiện hành.
Fullpilot kế tiếp **r20cosine50**, mới nối đúngstate50; không nối trajectorycos20 thànhcos50.
Một epoch toànrealtrain/client/round, real64+synth64/floor16,
KM16w0.5 +lite6+Q8, AdamW/reset/LR/clip/dropout/AMP chung, DAGSNetW1395.408params.
Simkhông officialtest, không chọnbestround. Samecounts/Qschema; compressionbyte thực cóthểđổi theoQ.
Bootstrap+hai chiều+total+compute báo đủ. Không claim fullr50byte từ sim.

Cap**2,75h/kernel**, toàn6run16,5kernelh +15%margin từ quota sống;
CBBsim20train≈3100s, thêm16class phép tính, NR không thêm forward/backward.
W1local2round từngP/NR/compile trướcquota, không RAMprobe hai trainingjob đồngthời.
Pinnedprivate2T4, credentialprocessscope, khôngglobalcredential/Git/DVCpush.

## Utilitygate mới — khóa trước training

Theo owner: **p10/low256/TD/benign/absent report-only**, không veto hoặc dùng chọnarm.
Giữ ngưỡng practicalgain **Δmacro≥+.002 từngseed** và endpoint cốđịnh.
Simr20 PASShai seed→tựfullpilot20cos50s42 vsCBB cùngseed/round.
Fullpilotr20 macro≥+.002→resume chínhstate2rank từ20 tới50 và freshseed43r50 đồngthời.
R50 **endpoint** macro≥+.002 từngseed→promote; TB46–50 vàcácthoảihiệp báo đầyđủ,
không chọnbestround. NhiềuarmPASS lênriêng, khôngcombo. MacroFAIL dừngadaptation, khôngtune/extendđểtìmwin.
Luật này dùng B17/trialstheo scope mới; **không sửa verdict cũ** thànhPASS.
Hai seed vàpanelval đãđọc lặp khôngsignificance/freshconfirmation.

Audit vận hành/source/config/panel/CM/finite/skip/mọi clientapplied vẫn bắt buộc,
không là “guard phụ” được bỏ. Quota thiếu thì không pushpartial.

## Localproof / audit actual

- Reference loss/gradient P chohard/softQ; Qrealphảidịch đúngcoordinate, không doubleprior;
  NRexplicitnon-target KL, zero trueclassgradient, hardQ vàinvalidteacher binarygate.
- Tail/floor/padded finite, realmasks/Qcount/BN đúng; CBBrowmode2roundexact,
  NRr1exact, P/NRexactresume đầyđủTorch+driver+methodRNG.
- Guarded200xeW1/AMP/compile2round **P vàNR riêng**, r2tiêu thụQ;
  actualCM→10metric/support/statefinite/skip≤1%/mọiclientapplied.
- Actual mỗi round:23.692activeclientsteps,1.476.064realrows;
  Pshiftvector đúng globalcounts, NRQcoverage/keepvector/gatedrowmass/loss vàteacherlagr−1.
  Source/counter audit không private-gradient/predictionreplay hoặc actualserializedframe reconstruction.
- Sourcefreeze→localgates→freshquota→notebook/sourcehash→push6→collect1200s→audit/analyze/report.
  Ownerjob/paths vàownership ở [continue](../operations/continue.md).

## Phạm vi dữ liệu

Protocol chỉ chuyển thống kê/xác suất, không upload realrow mới. Owner chấpnhận phạm vi này để thử performance;
privacybootstrap hiệnđược hoãn, không tự pause. Hai loss không tạo privacyguarantee:
moments ônhỏ vàQ kếthừa rủiro hiện có, chưaSecAgg/DP. Modelweights/checkpoint là artifact private của simulator,
không trở thành message trao giữa client. Nguồn rawdataset/input mount theo protocol giữ nguyên.

Artifact `runs/sim-kaggle-b17/`, `artifacts/sim-kaggle-b17/`. Chưa utility khi khóa tài liệu này.

## Đóng 11/11/2026 14:57 — không ứng viên qua macro

Job `mobihet-b17-20261111` succeeded14:57:31/exit0/B17_AUDITED;6/6kernel version1 đủ20round.
Đã tái chạy source/config/panel/CM audit, macrocoordinate/counteraudit, all20CM→10metric/LR analysis vàreport.
2T4/cả2rankcompile PASS, skip0,mọi clientapplied;23.692activeclientsteps và1.476.064realrows/round đúngcount.
Pshift khớp globalcounts, NRbinarykeep/gatedrowmass đúng, Qlagr−1. Không replay privategradients/prediction→CM.
13sourcefiles đã kiểmSHA/copy `runs/sim-kaggle-b17/frozen-source/` trước cập nhật tài liệu;
`closure-source.json` giữ exactrefs/source-lock gốc, không sửa lock.

| delta so CBB cùngseed | P42 | P43 | NR42 | NR43 |
|---|---:|---:|---:|---:|
| macro r20 | −.000658 | +.000192 | −.000164 | +.001630 |
| macro TB16–20, mô tả | −.000858 | −.000176 | −.000318 | +.001401 |
| benign F1, report-only | +.008344 | +.009978 | +.001663 | +.001876 |
| TD F1, report-only | −.034854 | −.025627 | −.010385 | −.002038 |
| p10, report-only | −.016348 | −.001482 | −.004924 | +.001030 |
| low256,9xe mô tả | −.018618 | −.031197 | −.007212 | −.015136 |

**P/NR FAIL primarymacro cảhai seed**: không arm đạt+.002; không cần dùng bất kỳguardphụ nào để loại.
Không fullpilot, không tune/extend/combo. CBB giữ nguyên. Ngay cả chỉ đòi macro>0 cả2seed cũng chưa đạt.
Các Δdưới0 nhỏ không là proof damage cósignificance; hai seed không làCI, NR43dương không là robustgain.

CBB endpointmacro .914618/.912907; P .913960/.913100; NR .914455/.914538.
Bootstrapgiữ1.196.032,46B/xe; hai chiều/round C305,69/304,71B, P302,49/301,39B, NR311,33/311,06B.
Nội dungQ nén khác, không thêmframefield/realrow. Trainseconds20r C3065/3061, P3116/3061, NR3048/3074;
elapsedkhông làFLOPsproof. [10metric/byte](../../runs/sim-kaggle-b17/report-r20.md).

### Bài học từ CM/counter

- Pbenignrecall+.03202/+.03480, precision−.01026/−.00948. TDprecision+.09426/+.11990,
  recall−.05425/−.04929; TD→benign thêm645/590row. Class11F1−.00998/−.00637.
  PgiảmFPbenign→attack ngoàiTD532/493row nhưng vẫn không đủmacro; accuracy/weightedF1 tăng không thay mục tiêu.
- PgainmeanF1 trên15class ngoàiTD chỉ+.00162/+.00191, thấp hơn Y50Corecũ+.00534/+.00491;
  TD đóng góp vàoΔmacro−.00218/−.00160. Đây là decomposition sốhọc, không causalproof riêngKM hay Q8:
  oldfullCorecos50 vàsimCBBcos20 khác recipe/population/schedule.
- NRgiảmFPbenign ngoàiTD232/215row. NR42 class11−.00466/TD−.01039, NR43tăngnhiềulớp nhưnggainmacro+.00163<+.002;
  TB16–20 −.000318/+.001401 cũng không cho robustgainhai seed.
  Binarygate loạiclass13 r2–7 ở cảhai seed; r8trởđi16classđều đượcgiữ.
  R20NCKDlossmean trênrealrow≈.003727/.003742; counterđúng không chứng minh classQ giảiđược futureerror.
- NRr1 chưaQ vẫn kháccontrolT4 (Δmacro+.001641/+.000115, CML1 9256/9782), dù CPUr1exact đãPASS.
  Pđổi objective từr1, nên r1khác làđúngcơchế. Không dùngP r1chênh làm numericalbug.
  NRr1mismatch làgiớihạnpairing cònchưa isolate; không gán mọi laterdelta cho receptionloss.

[Verdict](../../runs/sim-kaggle-b17/verdict-r20.json), [CM/cohort](../../runs/sim-kaggle-b17/diagnostics.md),
[mechanismsummary](../../runs/sim-kaggle-b17/mechanism-summary.json),
[actualcoordinate/counters](../../runs/sim-kaggle-b17/macro-audit.json).
Không có đủ điều kiện từB17 để train50; không suy mọi prior/NCKD ở mọirecipe đều vôích.
