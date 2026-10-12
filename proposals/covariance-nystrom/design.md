# S1 — covariance Nyström cộng được tại xe

Ngày2026-10-12 UTC+7. **S1 đã có bounded math/codec probe; chưa DAGSNet utility pilot.**
S1 là carrier mới của H2 input moments, đã thuộc danh mục24hướng; không thêm family mới.
Mọi learned model DAGSNet; analytic Gaussian sinh freshnoise tại xe. Input luôn66feature gốc.
Không record, latent hay gradient theo record rời xe. Không sửa source/notebook/manifest V3–V6 đã đóng băng.
[Đề xuất S1 ban đầu](../../docs/research/de-xuat-bo-sung-2026-10-11.md#2-s1--sketch-covariance-cộng-được--phát-từng-lớp),
[probe](../../scripts/probe_nystrom_math.py), [proof](../../papers/nystrom/math-probe.json).

## 1. Bằng chứng → cơ chế → bác bỏ

P1: Gaussianfull gộp hơnown vàdiag; lr4 giữ phần lớn lợifull. Nhưng P2lr4 chỉ giảm download,
uploadfullcov vẫn≈145,8kB/xe, B0 không kích hoạt tri thức trong20round.
[P1](../../docs/research/probes/pilot-p1.md), [P2](../../docs/research/probes/pilot-p2.md).

S1 gửi **sketch covariance ngay tại client** bằng tổng cộng được, không gửi covariance đầy đủ để
server mới truncate. Diagonal residual giữ marginalvariance; lowrank giữ một phầncrosscovariance.
Nếu numeric/codec ổn nhưngF1 thua matchedexactlr4/lr8 quá mức hoặc khônghơnown/shuffle,
dừng adaptation; không mặc định “moment gần hơn” đồng nghĩa decisionboundary tốt hơn.

Representation ởinput cốđịnh, không drift theo backbone như CCVR-like latentstats.
S1 vẫn là Gaussian approximation, không sửa support/phi tuyến mà V4/K5 đang kiểm.
R2 phù hợp interface input ởthiết kế; pilot đồngnhấtDAGSNet chưa thực nghiệm khác kiến trúc.

## 2. Nguồn phương pháp và phần adaptation

Tropp/Yurtsever/Udell/Cevher xây dựng linear sketch `Y=AΩ`, cập nhật cộng được;
Nyström là `Y(ΩᵀY)†Yᵀ`. Paper nêu randomized orthogonal Ω bằng thinQR,
và có stable shifted/SVD implementation. Relative-error theorem dùng sketch rộng hơn outputrank;
không áp bảo đảm ấy tự động cho k=r=4/8 của pilot này.
[Paper sơ cấp §2–4](https://arxiv.org/pdf/1706.05736),
[bản NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2017/file/4558dbb6f6f8bb2e16d03b85bde76e2c-Paper.pdf).

**Adaptation MobiHet riêng:** classconditional centeredmoments; explicitdiagonalresidual;
FP32uploaded sums/sketch, FP64server arithmetic, FP16decodedfactors; fixedfloor1e−4;
FL/cohort/contact/utilitygates. Không gọi adaptation này là tái lập toànAlgorithm3 hoặc theorem của paper.
P1 source [Synth](../../mh/pilot.py#L281) là referenceGaussian/full/eigenrank +diagonal residual;
P1/P2 historicalbyte counter chưa codecproof, phải dùng freshwirecontrols.

## 3. Thống kê local và centering sauaggregate

Public orthogonal `Ω∈R^(66×k)`, k=4hoặc8. Đề nghị cùngPCG64seed7601,
thinQR66×8, cốđịnhsignRdiagonal≥0; k4 lấy4cột đầu. FrozenΩhash/version/schema/featureorder
phải khớp. Seed một mình chưa đảm bảo QR bitwise giữa mọibackend; phân phối ma trận frozen hoặc
pin generator/hash; bootstrapbyte trảphí nếu xe chưa có. Matrix không được fit từ holdout.

Client i, class y, cácrow z-score nằmlocal:

```text
n_iy = số row
s_iy = Σx                         [66]
q_iy = Σx²                        [66]
C_iy = X_iyᵀ(X_iy Ω)             [66×k]
W_iy = (X_iy Ω)ᵀ(X_iy Ω)         [k×k, uppertriangle]
```

FP64accumulate tại xe, chunkbounded. Không cần giữ fullQ66×66 để tính C/W.
C vàW là aggregate thống kê theo lớp, không latent theo record. RSU chỉ thấySecureSum của cohort≥5;
không individualmean/factor/clientC/W. Trustedemulator, chưacrypto/DP.

RSU cộng n,s,q,C,W **trước** center; không cộng localcovariances rồi quên between-client means.

```text
n=Σ_i n_i; μ=Σ_i s_i/n; a=Ωᵀμ
v=Σ_i q_i/n−μ²
Y=Σ_i C_i/n−μ aᵀ                 =Σ_global Ω, lý tưởng
B=sym(Σ_i W_i/n−a aᵀ)            =ΩᵀΣ_global Ω, lý tưởng
```

W có thể derive `sym(ΩᵀY)` không thêm bytes trong exactarithmetic.
Bản đượcprobe gửi explicitWupper10/36scalar/lớp để kiểm consistency sauwire;
**không** coi W miễnphí. Chọn bỏW là codec/manifest khác phải khóa trướctrain.
IndependentFP32rounding làm B khácΩᵀY nhẹ; đo sai khác, không claim exactidentity sauwire.

Không chưng cất backbone/modelphụ. PrivateDAGSNet học CEreal+CEfreshsynthetic toànbackbone,
nhãn16class balanced nhưH2. Mean/proto alone không thay lowrank mechanism.

## 4. PSD/residual và sample dùng actualdecodedfactors

FP64server eigendecomposition `B=V diag(λ)Vᵀ`; numericrule của boundedprobe:

```text
λ_safe=max(λ,1e−4)
U=Y V diag(λ_safe^(−1/2))         [66×k]
residual_j=max(v_j−Σ_l U_jl²,1e−4)
sd_j=sqrt(residual_j)
Σ_hat=UUᵀ+diag(sd²)
x_synthetic=μ+U ε_k+sd⊙ε_66
```

LuônPSD khi factorsfinite vàresidual≥0; đó là algebra của sampler.
Flooredλ là **regularized inverse**, không pseudoinverse cutoff hay shifted-SVDAlgorithm3 nguyênpaper.
Floor cố định1e−4 trướcprobe; không tunejitter/rank/seed dựa vào residual hoặcF1 để cứu.
NegativeB eigen/floorcount/residualovershoot phải report. Nếu distortion lớn dùfinite thì
đừng dùng PSDPASS che failure sứcchứa hoặc unstablecentering.

Packet xuống chỉμ,U,sd; FP16serialize/decode thật. Privatekhông giữ shadowFP64factor khi train.
Decodedcovariance=`U_wire U_wireᵀ+diag(sd_wire²)`; PSD vẫn có vì đâyfactor representation,
nhưng marginalvariance không exactdoquantization/floor. Không gửi covariance denseFP16 rồi giảPSDsẵn.

Sampler sinh fresh từng minibatch, khôngbank tĩnh; không projectionmask/heading/distanceextra trongS1plain.
Nếu kết hợp vớiG59/G61 sau utility, cần manifest2×2riêng; không gọi cải thiện củabundle là lợi S1 riêng.

## 5. Codec thực đãprobe và byte

Codec trongprobe riêng, không thêm `mh` module: littleendian32Bheader,16Bcontexthash trongbody,
fixed16class order. Upload16×uint64counts +FP32[s,q,C,Wupper]; downloadFP16[μ,U,sd].
Header magic/schema/kind/dtype/flags/base/version/length/CRC được kiểm;
context bindΩhash/rank/feature/classorder/cohort/version/fixture/numericrule.
CRC/context làerror/bindingcheck, **không authentication hoặcSecureSum thật**.

Với D66,C16,T=2D+Dk+k(k+1)/2:

```text
up   =32+16context+8*C+4*C*T
down =32+16context+2*C*D*(k+2)
prior=200B/client tổngup+down, objectriêng theoV2codec hiệnhành
```

| knowledge | measured up/client B | measured down/client B | +paidprior tổng B |
|---|---:|---:|---:|
|Nyström4|26.160|12.720|39.080|
|Nyström8|44.720|21.168|66.088|
|matchedFP32full66/FP16chol|145.904|72.912|219.016|

Nyström4/8 applicationbytes≈17,84%/30,17% fullcontrol; chưa F1 hoặcR1.
Nếu bỏ explicitW, up giảm640/2304B/client; chưaprobe variant đó nên không gộpclaim.
Header/context/scalerΩbootstrap/frameACK/securitysetup/retry phải trảledger riêng trongcontactstage.
Prior không miễn phí chỉ vì packetmoments đãcó classcounts.

Globalobject B0cap4500B, bỏ overhead thêm: r4 ít nhất6up+3downcontacts; r8 10up+5down.
B2cap231000B: mỗi chiều vừa1contact theoapplicationbytes; chưadeadline/cohort/grantproof.
Không chia tổngbytes cho20round để gọi khảthi bootstrap: knowledgechỉ kích hoạt khiđã nhận coherentobject.

Per-class framing là **estimate chưa implemented**: r4 up1680/down840B/lớp;
r8 up2840/down1368B/lớp,16class total+prior40.520/67.528B.
Trảthêm15header+context mỗi chiều thaywholeobject. Cohort theo từngclass≥5contributors;
không phát singletonclassmoment dùwholecohort≥5. Cursorfixedround-robin nhưH6,
version(cohort/schema/Ω/class) phảiatomic. Lớp chưađủ object chưa dùng.

## 6. Boundedmathproof và bài học thực

[Script](../../scripts/probe_nystrom_math.py), [aggregateJSON](../../papers/nystrom/math-probe.json).
20client×2048uniform seededrow/client: **40.960row**, mmaptrain only,
seed7602+cid; không prefixvìfixture sắptheolớp. NoGPU/model/train/holdout.
Latestguard peak78MiB,exit0. Classes309–8481rows,19–20contributingclients.

LocalFP64stats → actualFP32bytepacket → decode → FP64cohortaggregate → centeredfactor
→ actualFP16download →1024freshsample/class/rank. Khôngexportrow/indices/factor/table;
chỉaggregatecounts/errors/bytes/sourcehash. Testbadcontext/CRCcorruption/truncationrejectionPASS.

Numericalgates đãđịnh trước:

- finitepacket/factors/synthetic, classcohort≥5;
- decodedminλ≥0,5×1e−4;
- FP16factor-induced covariance relativeFrobeniuserror≤1% so predownloadapprox;
- đúngroundtripshape/header/context và rejecttamper/truncatedpayload.

**r4/r8 đềuPASS.** Fullsamplecovariance error làdiagnostic, không numericgatechọnrank hoặcF1gate.

| meanrelativeFrobeniuserror vsfullfloat64sampleΣ |rank4|rank8|
|---|---:|---:|
|Nyström+diagonal, sauFP16|0,49324|0,43165|
|exacteigenrank+diagonal, từmatchedFP32fullmoments|0,32915|0,25773|
|diag only|0,65827|0,65827|
|FP16quantizationerror thêm, maximumclass|0,000434|0,000449|
|residualfloorfraction|3,8826%|3,8826%|

Nyström giữcrosscov hơndiag trênmẫu nhưng k=r hẹp kémexacteigenrank đángkể.
FP16error≈0,045%worstclass rất nhỏ so lossapproximation≈43–49%; chưa có cơsở đổlỗi quantization
cho capacity thiếu của sketch trênmẫu này. Không suy tấtcả futurefulltrainclass/Ωseed.

Wminimumλ0,51528/0,28164;0eigenvalue floorởcảhai ranks.
Residualnegative tới≈4,15e−9, chủyếurounding/constantfeature; fixedfloorclip3,88% coordinates.
Worstdecodedvarabsoluteerror≈0,00704/0,00579 standardizedvariance; decodedΣminimumλ≈1,000427e−4.
FullFP32uploadcovariance vs exactfloat64sample maximumrelativeerror≈2,13e−8.
Quantization/cross-CWconsistency đượcđo; không claimwhole43M hoặc coefficient thậtngoàisample.

## 7. Utilitypilot cần chốt trước triển khai

Chưalaunch. Bộsmallestproposed11arms/seed:

- localreference đãaudit, freshdiag, full66, exactlr4, exactlr8;
- nys4FL/own/signshuffle vànys8FL/own/signshuffle.

Nếu reuseauditedlocal, còn10fresharms/seed. Seed42/43,20round,fixture/protocolDAGSNet20c/alpha.5/2,5%,
privatebatch512/aux256,λsynthetic1,optimizer/cosineLR/BN/evalold/group vàr20gp giốngcontrol.
Đocostactual, mọiapplied/skipped/CEreal/CEsynthetic, priors/payloads/source/weights/pred→10metric/perclass.
NặngKaggle; local boundeddriver/smoke/exactresume/inferenceaudit. W&Bonline/key trướctrain bắtbuộc.

FreshV6full66plain là reference gần đây vềmodel/sampler; **uplinkV6FP64, S1FP32**.
Đểcausalclaimsketch, freshS1full/exactlr controls phải sameFP32momentupload/FP16download,
chỉ thaycovreconstruction; V6F64 có thể làadditionalreference, khôngswapâmthầmkhácprecision.
Residualfloor1e−4 chungS1/exactlr. FullCholesky dùngΣ+1e−4I nhưP1; báoregularizationmapping,
không gọi marginalequalabsolute trêncoordinates sátfloor.

Ownstats giữlocal,FP16roundtriplocal; commonpaidprior vẫngộp. ClassN0 nếu có phải lockfallback
trướcrun; không dùngpopulationmomentlấpown. Populationclassmoment chỉrelease≥5clients;
own không áp5contributorreleaseguard vìkhông phátdata.

**Signshuffle proposal sạch:** public±1percoordinate, `U_shuffle=diag(sign)U`;
mean/sd giữ nguyên. Marginalvariances vàeigenvalues giữ vìcovariancecongruence bởiorthogonalsigns;
FP16signflipexact. Phácorrelation-sign alignment, giữmagnitudes; khôngclaimxóa toànbộcovknowledge.
Coordinate-rowpermU trực tiếp thaydiagvariance/floor nênkhông làm matchedshuffle mạnh.
Đây scientificchoice cầnchốt, khôngprivate-recordshuffle.

ProposedUgate từngrank độc lập, cả2seed, trướctrain:

- nys_k−matchedexactlr_k≥−0,010, từS1contractbanđầu;
- nys_k−own≥0,010 và−signshuffle≥0,010 nếu chọncontrastorientation;
- group/gp−local≥0,030;old/gp>local;benignF1≥local−0,030;
- actualappbytes≤0,65×matchedfullcontrol, nhưngbytePASS không cứuqualityFAIL;
- numeric/source/payload/step/resume/inferenceauditPASS.

Reportfull/diag/exactlrcontrast,positionMirroring/timeDelay vàmọiclass; không pickbesthead/round/Ω/hp.
Costgate hoặctrần2×freshdiag nếumuốn khóa phảichốt trướclaunch, không coi cheaplinearstats làchi phítrain.

ChỉUqua mới mởcontact: `{diag,nys}×{whole,classobject}` tách đónggópearlyactivation,
bootstrapprior/cohort/cursor/ACK/deadline/lostframecost/versioncoherent; proposedretain≥70%Ugain
vàFLgain≥0,010 undercontact. KhôngghépH6 rồi quy lợiactivationchoNyström.

## 8. Điều còn phải chốt, không thao tác tay lúc này

1. Utility kiểm cảk4/8, k=rkhôngoversampling nhưprobe, hay thêmk8→outputr4? Probe khôngtự chọnvarianttốt.
2. ExplicitWupper đãtínhbyte hayderiveW từΩᵀC; không bỏblock sauđọcsốđểbáobytenhỏhơn.
3. Full/exactlrFP32freshcontrols, signshuffle vàutility/costthreshold ở§7.
4. WholeobjectU trước, classobject/contact chỉsauU; publicΩ/scaler bootstrapownership.

Numericimplementation cóbằngchứng khảthi; scientificutility/cost/contact chưa có.
GiữS1 trongnhómchưautility, khôngthêm vào2hướngpositivepilot hoặcgọimathPASS làIDSthắng.
Khôngcần thao tácmanual choprobe/doc; code/notebookKaggleS1 chưa được tạo/push trongtasknày.
