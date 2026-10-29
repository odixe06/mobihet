# V4-S — support moments gộp, Gaussian sinh mới tại xe

2026-10-12 UTC+7. AdaptationH2, chưa F1feasibility. Mọi learnedmodel DAGSNet; analytic sampler khôngclassifier học được.
[Bằng chứng/schema](../../../archive/methods/proxy-compression/builds/proxy-carriers/probes/support-probe.json), [thiết kế nguồn](v3-knowledge-sources.md),
[ledger](../methods/gaussian-sampling/builds/support-mala/results/rebuild.md). Không dựa supportfidelity để tuyên bố IDS cải thiện.

## Bằng chứng → cơ chế → bác bỏ

Feature binary/profileone-hot và13zero-imputed đầu phiên không khớp supportGaussian liên tục.
Schema sample40960trainrow/7337firstrow đãPASS; fullfixture local schemaassert trước tínhmoment
ở mỗi worker làgate, không suy sample thànhfullscan. Atomraw0 phảizscore đúngscaler vàfp16 nhưfixture.
GaussianmomentgộpP1 có lợi; giữ moment non-atom có thể tránh sinh điểm thiếu support.
Nếu binary/firstfidelity tốt nhưng privateF1 không hơndiag/own/shuffle thì dừng adaptation này.
Không khẳng định support là nguyên nhân duy nhất của C/KME/DM thất bại.

## Payload và thống kê

Mỗi xe/lớp y giữ `n_y`, mỗi66feature j: `a_yj=count[x_j=atom0_j]`,
`s_yj=sum_{nonatom}x_j`, `v_yj=sum_{nonatom}x_j²`.
Không export mask/record/latent; trustedaggregate cohort20 ởU, gate≥5 ởcontactstage.
Uploadfp32 `16*(1+3*66)=3184` values+32Bheader+8Bcount = **12776B**/xe, một lần.
Count nguyên trongfixture≤2²⁴, không mất integerprecision khi fp32 trướcaggregate.
Gộp rồi derive `rho=a/n`, `nonmu=s/(n−a)`, `nonsd=sqrt(max(v/(n−a)−nonmu²,1e−4))`.
Unconditionalmean/variance từ `s+a*atom0`, `v+a*atom0²`.
Nonatomcount0: divisorclamp1, Gaussianbranch cóprob0; emptyownclass fallback deterministicfinite,
không mượn globalclassstat để lấpownknowledge.

Down table5array`[mu,sd,rho,nonmu,nonsd]`,16×66,fp16 actualdecode:
**10592B**/xe một lần; prior codec200B/xe thêm ⇒ **23568B** tổngapplication/xe.
Không thêm categorypacket: binary`p1=1−rho`; profile categorical probabilities từ3p1,
unknownprob=max(1−sum(p1),0), normalize4categories để chịu fp16rounding.
Atom/schema/codeversion/bootstrap/frame/ACK cầntrảphí ởcontactstage, không gọi seed miễn phí.
B0 ít nhất6contact từpayload hai chiều; B2 vừa mộtcontact theoapplicationbytes, chưadeadline/ACKproof.

## Sampler tại xe

Mỗi privatebatch sinh256freshpoint, nhãn cân bằng16class, khôngbank tĩnh.
Continuousfeature: atomraw0 vớiprobρ; còn lạiGaussiannon-atom. Raw0 hợp lệ không gọi là missingness.
Firstflag sinhraw0/raw1exact. Profilemỗibên sinh categorical4category normal/cautious/aggressive/unknown,
luônone-hot hoặcall-zero; không independentBernoulli gâymulti-hot.
**Structural sampler:** nếufirst=1,13feature schema zero-imputed đềuatom0.
Nếufirst=0, atomprob13feature=`(rho_atom−p_first)/(1−p_first)`,clamp[0,1] chịuquantization;
continuousconditionalmoments giữ từ thống kênon-atom. Không cầnconditionalmoment packet phụ:
schema đãassert firstrows13featureatom0, nên mọi non-atom nằm ởfirst=0.
Chưa giữ dependence giữaprofile/mask vàmọifeaturecontinuous, covariance/multi-featurealgebra.

Shuffle: xoayrho giữa13structuralfeatures riêng, continuousfeatures khác riêng, trong3category mỗi profile riêng;
firstprobgiữ nguyên, nonmu/nonsd vẫn địa chỉ gốc. Categorical/firststructuralconstraints vẫnđúng;
phá thông tinatomprofile đúngfeature, không shuffle record/nhãn thật.

## Arm, protocol, cổngkhóa

- `s_diag`: unconditionalGaussian từcùng thốngkê/codec, strongfreshcontrol;
- `s_marginal`: categorical/binaryexact, independentatom chưaràngbuộcfirst13;
- `s_structural`: first13constraint, phươngánchính;
- `s_own`: structural sampler từ ownmoments (fp16roundtriplocal), khônggộp/downloadsupportstats;
- `s_shuffle`: structuralvaliditygiữ, rhoaddressshuffle.

Prior gộp chung tấtcảarm, kểcảown; không gọi own là hoàn toànno-communication.
PrivateCEreal+CEsyntheticλ1, DAGSNettoànbackbone. Khôngteacher/proxy/learnedgenerator.
Fixture/protocol/numeric/BN/optimizer nhưV2diag:20clients,2,5%,alpha.5,66features,seed42/43,
20round×1epoch,privatebatch512/aux256,AdamWwd0reset,lr1e-3→1e-5 cosine,clip1,
AMP/deterministic/backendpin nhưV2. Evalr5/10/15/20 old/group,head/gp,10metric/perclass.

Hai seed đều: structural−diag≥0,015; −own≥0,010; −shuffle≥0,010;
group/gp−localV2≥0,03; old/gp>localV2; benignF1≥localV2−0,03.
Structural−marginal≥0,005 đểclaim first-sessioncoupling; thiếucontrast không phủđịnhutility nếuqualityqua.
Syntheticstructural/own/shuffle binary/profile/firstconstraint invalidfraction0 theoatomschema;
marginalfirstinvalid vàdiagbinaryinvalid làdiagnostic đốichứng, khôngfailchủđích.
Không tune atomrule từ positionMirroring/holdout; báo lớp yếu riêng.

## Audit và giới hạn

Freeze scaler/schema hash vàatomarrays trong`mh/v4_schema.py`; khôngfit model phụ hoặc đổi featureorder.
Actualsupportpacket lưu trongalgorithmstate; table cốđịnhsauone-shot; byteledger actualcodec.
5arm×3roundsmall, structural/own exactresume, fidelity65536syntheticdraw/arm,
weights/predictions→clientconfusion→10metric/sampledinference trước Kaggle.
W&B online trướctrain, privatekeyboundary/source/metadata/2T4 preflight nhưV3.
ChỉqualityPASS mớicontactprotocol mới vớibootstrapcoherent/ACK/deadline; chưaR1/R2 thựcnghiệm, chưaDP/crypto.
