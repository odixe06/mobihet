# B18 — matched teacher source, khóa trước train 11/11/2026

Owner: “chạy thử nghiệm đi”, sau [khảo sát nguồn/domain](../../archive/docs-20261113/research/teacher-source-survey-20261111.md).
Main CBB giữ DAGSNetW1, KM16w0.5/lite6/Q8, fullfoldtrain, realCE+LA, per-rowKD,
64real+64synthetic/floor16/AdamWreset/cosine20/AMP như control. Không SAM, không thêm qualityweight/domainteacher.
Chỉ count/probability vào logicalpayload; không gửi realrow, model, BN, gradient hoặc hiddenvector.

## Evidence → cơ chế → dấu bác bỏ

B13–B17 đổi reception/confidence/prior chưa tăngmacro đủgate; V20 cho thấy trainquality khó dự báofuture.
Domainforecast từ donortrain chưa khai thác được association futurepeer. Nguồn Q là estimand còn mở.
Giả thuyết: Q đo ngoài fit chứa quan hệclass tổngquát hơn Qtrainconfidence khi chuyển sang KMsynthetic.
Không có định lý trực tiếp bảo đảm classmeanQ→KM; đây là thửnghiệm utility, không reproduction papercrossfit.

| phase | producer / source | vai trò |
|---|---|---|
| shadow S | DAGSNetW1 mỗi xe học earlyA; chỉA vào bootstrapKM/Qinternal/BN/weights | nguồn dùng chung E/H, không candidate |
| C | CBB bình thường, Qonline từ mainfulltrain | utilitycontrol |
| E | mainfulltrain, thay ôeligible bằng Qeval đo trên earlyrow đãfit của shadow | matchedsourcecontrol |
| H | mainfulltrain, thay ôeligible bằng Qeval đo trên heldoutrow chưafit của cùngshadow | candidate ngoài fit |

Mỗi seed42/43 có một notebook, tuần tự S→C→E→H trong process riêng; cảhai seed pushcùngbatch.
Giữ source/shadowweights cốđịnh giữa E/H, không shadow nhận feedback từ main.
Qshadowinternal riêng dùng A; probeE/H không update bất kỳmodel/BN/RNG hoặc Qinternal.
Không dùng simfoldval/officialtest trong shadow; shadow đánhgiá A chỉđể kiểm CM.

## Partition / estimand đã khóa

Chọn panel200/seed7 như lịch sử. Chỉ foldtrain của panel, 1.476.064row;
cutoff theo ceil(.8n) receive-time mỗi xe, giữ toàn bộ ties phíaearly.
Counts dựkiến A1.181.080/H294.984; không lấy80%prefix cache vì cache theo fileorder.
Sidecar scan410trainparquet, nối từngclient theo đúng canonicalfileorder, kiểm per-client labelsequenceSHA
và counts với loader trước train. Sidecar lưu roles/hash/count/cutoff, không lưu feature/labelrow.
Roles/probe membership là localdata instrumentation, không một phần packet gửiserver.

Eligible iff mỗi(client,class) có≥8row cảA/H. Match `m=min(nA,nH)` probe mỗi phía,
chọn smallestSplitMix64 hash của(client,canonicalrowordinal), cốđịnh qua seed.
E/H cùng client×classprobecount, batching/evalmode/rawlogits/no dropout và cùng Q8codec.
Weight đóng góp server vẫn `n_main(client,class)`, không `m`; ôineligible giữ Qonline của chínhmainarm.
Đây là **hybrid**, không pureOOF; eligiblemass98,3685%, c11 chỉ81,8160%.
Q thu shadowroundr dùng tạo teachermain sau roundr, chỉ tác động mainr+1; r1main chưasoftQ.
Fixedv4scaler chứa H: chỉ model-out-of-fit conditional onfixedpreprocessing; chưaembargo/eventindependence.

## Codec / server / communication

Giữ Q8clientcodes và đường decodeCBB: code/255 fp32→fp16, weightedfloat64sum,
servermeanfp16→rowrenormalize→Q8downlink→renormalize. Không average meanlog hoặc sharpen.
Server nhận sourceE/Hcodes của eligiblecell từshadow, thay đúng đóng góp ấy trong classsum;
online main chỉ cần gửi ôineligible. Không đẩy sourcecodes vềmain rồi reupload.

Packet mới:32byteheader+2byteclassmask+1byte raw/zlibflag; bodyprob16uint8/ô.
Measurement source thêm4byteint32probecount/ô; bootstrapmaincounts đãcó weightn_main.
Downlink16×16Q dùng cùngframe; khác overheadCBB ítbyte, không đổi Qnumeric.
Ghi actualframe/offset của source và mainfallback, mixed/onlinecodes/aggregate/Q/hash từngrank/round.
Mỗi E/H báo tổng **shadowQuplink + sourceprobeuplink + mainfallbackuplink**,
downlink **shadowQinternal + mainQ**; bootstrap **mainKM + shadowKM**.
Sourcecounts truyền lại mỗiround trongframe theo accounting bảo thủ; không coi metadata/count miễnphí.
Thựcnghiệm đo cảE/H, report actual sharedshadowcost riêng; deployment mộtcandidate chỉ cần mộtsourceprobe.

## Gate / budget

Sim endpoint r20cosine20, pooledmacro16class: **H−C≥+.002 và H−E>0 ở cảseed42/43**.
E là mechanisticcontrol, không tự promote nếu E thắng. Guard p10/low256/benign/TD/absent report-only.
Đọc thêm TB16–20/trajectory/cost nhưng không chọnround/HP. FAIL thì dừng adaptation, không tune/extend.
PASS tự lên pilotfullr20cos50 seed42; pilot và r50 gate phải khóa trước push bậc ấy,
giữcùngmeasurementcontrol và r50mainheadline; fullphases táchsession nếu state vượt20GB.

Local: unitproof Cexact2round, r1mainweights/E/Hfallback, codec/replay/corruptframe,
partitionties/canonicalorder; real200xe/W1/compiledAMP đủ2round **mọiphase** và source/CMaudit.
Compileguard phảiPASS, mọi clientapplied, finite/skip≤1%; local1GPU không chứng minh2T4.
Remote: đúng2T4/rank, source/config/partitionSHA cảseed, đủ20round mọiphase,
20CM×10metric từngphase, teacherlag/hash và actualwire/server/cost replay.

Compute bound main+shadowactive steps23.692+19.211=42.903/round≈1.811×C,
còn inference2×matchedproberows và bootstrap/eval/setup/I/O. Mỗi notebook chạy3main+1shadow;
từ B17control train+eval≈3222s/20round, bound train cảbatch≈6.9h hai notebook chưa extra.
Cap **7h/notebook**, reserve15% tại accountlaunch + notebookreserve0.4h, tổng16.1quota-hours reserved.
Đây cap/budget, không measuredruntime hoặc GPU-FLOPs. Quota sống sau localgate, ưu tiên2owner khácnhau.
Nếu compileprobe làm capkhông đủ, dừng trước push và cập nhật budget từ sốđo, không cắtround/seed.

## Trạng thái

Local hoàn tất32tests +4phase×2round/W1/compileAMP, source/CM/packet auditPASS.
Hai kernelversion1 đãpush nhưng đều **ERROR trướcround1**: thiếu `runs/sim/core_hp.txt` trongoverlay,
không cócheckpoint hoặcutility. Source/artifactgốc đãpull/verifySHA,11filelock giữ tại
`runs/sim-kaggle-b18/frozen-source/`, proof `runs/sim-kaggle-b18/packaging-failure.json`.

Recovery11/11: thêm **resourceHP vàooverlay**, không đổi8trainingmodule/code/HP/data/loss/round.
Regression `scripts/check_b18_bundle.py` dựng cloneGitHEAD+declaredoverlay, import actualphase/method/auditor:
đãreproduceFileNotFound rồiPASS saufix. Không dùng localcheckout đầyđủ thay deploymentgate.
Recovery cóslug/batch mới **b18r1**, version1; freshround1 vì0checkpoint.
Tái audit actualsmoke cũ vàHP/sourceSHA, khôngtrainlocal lại vìlỗi đónggói.
Worker mới vẫn unit→deploymentgate→reusedsmokeaudit→freshquota→2seed→collect1200s/audit.
Cap/gate khoa học giữ nguyên; đây khôngmethodFAIL hoặc tune. Job/refs/quota thực: [continue](../operations/continue.md).

## Kết quả đã audit — đóng 12/11/2026

Hai recoverykernel version1 **COMPLETE**, shadow/C/E/H mỗi phase đủ20round, tổng8phase.
Localcontrollerfailed02:08:59 ởanalysis: auditor yêu cầu bảncopyHP `.txt` nhưng provenance cũ chỉcopyPython.
Đã phục hồi **analysis**, không train/push lại:8executedPythonoverlay khớpSHA;
HPresource kiểm bằng sealedpushednotebookSHA+payloadSHA+remoteoverlaymanifest và4effectiveconfig/seed.
Không tạo fileHPgiả vào `provenance/source`, không claim giữ executedHPcopy.
Regression sửaPython/configHP/notebookhash đều bịtừchối; futuregenerator đãcopy mọioverlayresource.
Writerverdict cũng sửa NumPybool→Pythonbool sau khi CM/packet audit đãPASS.
13source banđầu giữ `runs/sim-kaggle-b18r1/frozen-source/`, proof `closure-source.json`;
controllerfailedstate/log giữnguyên, scientificclosure riêng.

Tái chạy actualaudit đủ8phase×20CM→10metric, source/config/panel/partition/count/SHA,
Q8source/fallback/mixedframes/tworanksums/downlinkQ/hash/lag/bootstrap/bytes PASS.
2T4/2rankcompilemọiphasePASS, khôngxe mấtupdate; không privategradient/weight hoặc prediction→CM replay toànmạng.
Endpoint chỉfoldvalpanel200, không freshconfirmation, hai seed khôngsignificance.

| seed | C macro | E macro | H macro | H−C | H−E | verdict H |
|---:|---:|---:|---:|---:|---:|---|
|42|.91415140|.91458920|.91308569|−.00106570|−.00150351|FAIL|
|43|.91451459|.91401839|.91338226|−.00113233|−.00063613|FAIL|

**H FAIL cả2seed**, cảgainmacro+.002 vàmatchedsourcecontrol. Khôngguardphụveto;
khôngpilot/tune/extend/ghépvaliditygate đểtìmwin. E cũngkhônggainmacro ổnđịnh, chỉmechanisticcontrol.
TB16–20 H−C−.00083834/−.00111387, không đổi endpoint bằng chọnround.
CBB giữnguyên phươngpháp chính.

[Đủ10metric/byte](../../runs/sim-kaggle-b18r1/report-r20.md),
[CSV](../../runs/sim-kaggle-b18r1/metrics-r20.csv),
[verdict](../../runs/sim-kaggle-b18r1/verdict-r20.json),
[source/CMdiagnostics](../../runs/sim-kaggle-b18r1/diagnostics.md),
[teacher và số đầyđủ](../../runs/sim-kaggle-b18r1/diagnostics.json).

### Cơ chế thấtbại quan sát được

Mainr1CM C/E/H **khớpđúng cảhai seed**, khi chưa dùngQ; không suy thànhwholetrajectorybitexact.
Matchedprobe291.220row/nguồn/round, cùngshadow/evalmode/probecount vàpopulationcountweight.
Shadowr19 rawprobehardmacro E .99092/.99127, H .90444/.90566;
populationweightedmeanp(trueclass) E .97861/.97869, H .89399/.89405.
Không suy softBrier/NLL/per-rowvariance từ moment1; nguồn H đo thực lỗi/dịchchuyển thời gian,
không tự là targettri thức tốt hơn cho synthetic.

TeacherH **đã decode/gửi** r19 (tácđộng mainr20) có hàngtrueTD:
`Q[TD,benign]=.65748/.64706`, `Q[TD,TD]=.31102/.32157`; chỉTD cóargmaxkháctrueclass.
Đây khác lỗi codec: packet/aggregate replaykhớp, source thật ngoài fit đãnhầm TDthànhbenign.
R20 H benignF1+.008142/.008172, nhưng TDF1−.047154/−.038045;
TD→benign thêm783/686row, benign→TD giảm1255/1325row.
15classkhác meanΔF1+.002007/.001329 chỉlà decomposition: TD vẫn1/16macro,
không bỏTD hoặc chọnsubsetclass đểgọiwin.

H byte/round≈722.48/723.36 vsC305.67/305.03, E≈678.02/677.63, đãcộngshadowQinternal+sourceprobe+mainfallback.
Bootstrap C1.196032MB/xe, E/H2.378452MB/xe (up646040.705+down1732411.2B).
Main+sharedshadow measuredtrainelapsed≈1.92×C; shadow đo cảE/Hprobe nên đây experimentalcost,
không FLOPs hoặc estimate deploymentmộtprobe. Mỗi notebook cả4phase≈3,6h, không cần7hcap thực.

### Bài học cho hướng sau

Nguồnngoàifit hữuích để **đo reliability/shift**, không đồngnghĩa lấy nguyên classconfusion làm KDtarget.
Countweightedpopulationmean khôngbug. B18 không bác mọi qualityweight/recipientteacher;
nhưng muốn thử estimand khác phải cùngsource/control, thêm evidence/khóa rule riêng,
không sửa H bằng temperature/gate/rounds saukhi đãFAIL đểtìmwin.
Sauclosure chưa có trainingjob hoặc candidate mới; chưacần thao tác tay.
