# V3 — nguồn tri thức mới cho DAGSNet riêng tại xe

Ngày2026-10-11. **Đề xuất nghiên cứu, chưa code/chạy; không phải kết quả.** Đọc cùng
[CONTEXT](../../../CONTEXT.md), [LESSONS](../../../LESSONS.md),
[V2](de-xuat-v2/README.md), [S-a đã trượt](probes/pilot-new-fl.md).
Không sửa cấu hình B/C đang chạy. Những cổng dưới đây là **đề nghị cần đóng vào manifest mới trước pilot**.

**Cập nhật preflight trongphiên:** K1schema sample40960row PASS:profile phảicategoricalone-hot,
first-session buộc13feature vềraw0 atom đãzscore. [Supportprobe](../../../archive/methods/proxy-compression/builds/proxy-carriers/probes/support-probe.json).
K4unbased JEM trênR66 **FAILnormalizability** doheadLayerNorm làmDAGSlogits bounded;
không heavytrain bảnnày. [Headboundprobe](../../../archive/methods/proxy-compression/builds/proxy-carriers/probes/energy-probe.json), [LESSONS](../../../LESSONS.md).
Gaussianbase/compactdomain làadaptationkhác phảichốt; không tựxóaLayerNorm đểcứuphươngán.

## 1. Vấn đề cần giải, không mở lại giả thuyết đã thua

P1 chứng minh tham số DAGSNet toàn cục và covariance input mang tri thức receiver khác.
S-a chứng minh giảm synthesis fit của bank KME/DM tĩnh chưa tạo lợi phân loại.
V2-A đang cho thấy adaptation sketch-EF có thể khuếch đại residual; chưa lấy carrier ấy làm nền mặc định.
V2-B/C chưa có kết quả artifact-verify tại thời điểm viết. Không giả định chúng thắng.

Đợt mới giữ20client, fixture2,5%, seed42/43,20round; group/gp làm headline private, old và benign làm guard.
Mọi mạng học được là DAGSNet. Thống kê/fixed kernel/codec không phải classifier phụ học được.
Mỗi hướng phải có đối chứng `own` không trao đổi; hơn local do thêm auxiliary loss chưa đủ gọi lợi FL.
Không raw, perturbed record, latent hay gradient từng record rời xe. Thống kê/gradient mục tiêu gộp
chỉ xuất hiện sau trusted secure-sum simulation cohort≥5; chưa cryptographic secure aggregation/DP.
Synthetic được tối ưu từ noise có thể đọc từng producer theo ngoại lệ hiện hành.

| hướng | cơ chế khác các build | sẵn sàng thiết kế | chưa biết lớn nhất |
|---|---|---|---|
| **K1 support-aware Gaussian** | giữ atom zero/binary và moment điều kiện, sinh mới liên tục | ưu tiên nhỏ, sau audit schema | mất dependence giữa mask và feature khác |
| **K2 residual hàm tại anchor** | gửi thống kê label-residual thật quanh public synthetic anchor; không lấy vote teacher yếu | pilot sau khóa bandwidth/anchor | anchor không phủ ranh giới receiver mới |
| **K3 condensation gradient động** | mục tiêu gradient tại DAGSNet đang học, book tái sinh; khác encoder ngẫu nhiên tĩnh | khả thi compute có điều kiện | synthesis quá tốn, hiệu lực gradient không chuyển thành F1 |
| **K4 joint-energy DAGSNet** | DAGSNet vừa phân loại vừa học năng lượng input, SGLD sinh điểm mới | nghiên cứu rủi ro cao, trần U trước | support, BN và chuỗi sampling không ổn định; chưa giải byte |

K1 là cải tiến H2 theo cơ chế support, không họ FL hoàn toàn mới. K2–K4 là adaptation đề nghị của dự án,
không khẳng định novelty hoặc mượn định lý hội tụ ngoài giả định nguồn.

## 2. K1 — thống kê điều kiện bảo toàn atom support

### Bằng chứng → cơ chế

Dataset có feature binary/zero-fill; Gaussian liên tục có thể tạo mẫu thiếu cấu trúc support.
Điều này hiện là rủi ro V2-C, chưa phải nguyên nhân đã được cô lập của S-a.
Đề nghị thay marginal Gaussian ở feature schema xác định bằng hỗn hợp atom và continuous conditional moments.
Không EM, không cây hay generator học được. Private classifier vẫn DAGSNet.

**Không coi standardized value0 là raw zero.** Với scaler, atom raw0 nằm tại
`a_j=(0-mean_j)/std_j`; binary raw0/1 nằm tại hai atom tương ứng. Chỉ dùng feature được schema/generator
xác định trước, không chọn theo holdout. Cột zero-fill conditional cần phân biệt zero hợp lệ với missingness;
nếu schema không phân biệt được thì ghi hạn chế, không tự gọi là missing.

Tại xe, cho từng lớp y/feature j:

\[
n_y,\quad n^{atom}_{yj}=\sum 1[x_j=a_j],\quad
s^+_{yj}=\sum_{x_j\ne a_j}x_j,\quad v^+_{yj}=\sum_{x_j\ne a_j}x_j^2.
\]

Gộp tuyến tính, suy ra `rho=n_atom/n_y`, `mu+=s+/(n_y-n_atom)`, `var+=v+/(n_y-n_atom)-mu+²`.
Sinh mới mỗi minibatch: `x_j=a_j` với xác suất rho; còn lại lấy Gaussian điều kiện non-atom.
Binary sinh Bernoulli theo count, không Gaussian rồi round. Continuous feature khác giữ H2 diag.
Nhãn là y đã điều kiện, không teacher vote. Case non-atom count0 dùng atom chắc chắn; count quá nhỏ
không phát riêng producer, gộp đủ cohort; variance floor/finite checks khóa trước.

**Mức tiếp theo có điều kiện:** joint mask cho tối đa4binary/zero indicators đã định trước.
16mask ×16class counts và conditional moment cho từngmask giúp tránh độc lập giả giữa mask.
Không mở cấp này nếu marginal support không mang lợi; count các cell gộp có thể nhỏ, cần suppress/merge
theo rule từ train đã khóa, không dựa holdout. Đây là thống kê analytic, không learned histogram classifier.

### Byte thực, BN, privacy

Ví dụ marginal full66features,16class, upload `n_y` uint64 + từngfeature atomcount uint32,
sum/sumsq fp32: `128 +16×66×12 +32header =12.832B` một lần/client.
Download `rho,mu+,sd+` fp16: `16×66×6+32=6.368B` một lần/client.
Với schema chỉ m feature atom, có thể giữ Gaussian diag baseline cho66features, cộng riêng
atomcount uint32 và conditional sum/sumsq fp32 cho m; báo tổng, không chỉ overhead.
Prior/frame/ACK/cohort setup trả phí riêng, không nằm trong các số payload trên.
B0 khoảng4.500B/contact cần≥3uplink và≥2downlink, cộng overhead có thể cần thêm contact.
B2 đủmột contact mỗi chiều. Snapshot chỉ kích hoạt khi nhận đủ prior+stat+schema version.

Private BN dùng local train policy như mốc diag; synthetic CE tác động toàn backbone.
Đối chứng BN phải giống nhau. Thống kê giữ atom không bảo đảm privacy: tổng hiếm có thể lộ,
cohort≥5 không thay DP; không gửi pattern theo receiver/vehicle ID.

### Arm và cổng đề nghị

`local`, `diag`, `support_own`, `support_fl`, `support_shuffle`.
Shuffle hoán atomprob giữa feature cùng kiểu theo permutation công khai; giữ continuous moments
và tổng compute, phá đúng thông tin atom phù hợp feature. Không shuffle raw records.
Nếu schema không có nhóm≥2feature cùng kiểu thì bỏ shuffle ấy và nêu thiếu đối chứng.

Hai seed finalr20: FL−diag≥0,015; FL−own≥0,010; FL−shuffle≥0,010;
groupgp−local≥0,03; oldgp>local; benignF1≥local−0,03.
Đo zero/binary invalid fraction từsynthetic phải0 theo định nghĩaschema; positionMirroring F1 báo riêng,
không chọn kiểu atom theo lớp đó trên holdout. Nếu chỉ support fidelity tăng mà F1 không tăng: dừng.

## 3. K2 — residual hàm từ nhãn thật trên anchor synthetic công khai

### Bằng chứng → cơ chế

DS-FL thua nhãn Gaussian vì teacher local yếu. Đề nghị **không hỏi local teacher vote**.
Mỗi client dùng nhãn thật local để đo hướng sửa cho hàm chung ở neighborhood anchor;
chỉ thống kê tổng mục tiêu được gộp. Nguồn cảm hứng là tối ưu functional FL, không tuyên bố triển khai FFGB nguyên bản.
[FFGB](https://arxiv.org/abs/2103.06972) nghiên cứu tối ưu trong không gian hàm và phân biệt
giả định marginal đồng nhất/khác nhau; kết quả ấy không trực tiếp bảo đảm adaptation kernel hữu hạn này.

Anchor `z_m` sinh từGaussian H2/seed chung, độc lập record thật; M=64nhắmB0 hoặc512nhắmB2.
Private DAGSNet tại mỗi client có cùng input/16logits, kiến trúc tương lai có thể khác.
Anchor target `t_m∈R¹⁶` là **thống kê giám sát gộp**, không model deployment phiDAGSNet.
Client i giữ DAGSNet companion tại local, fit vào `t_m`; đối chứng own cũng có companion cùng compute.
**Bản gradient mục tiêu chung phải dùng cùng DAGSNet `theta_t` trước tính loss;**
không lấy gradient từ20snapshot private khác nhau rồi gọi gradient chung. Bản dưới chỉ hợp lệ nếu
`p_i(y|x)=p_theta_t(y|x)` cho mọi client trong cohort; khóa hash theta/BN trước tínhG.
Companion private đểKD có thể khácweights, nhưng không được dùng thay probe chung khi tínhG.
Không truyền weights companion. R2 khả thi giao diện, chưa kiểm khác model.

Với shared anchor-version t, dùng kernel input cố định:

\[
k_m(x)=\exp(-\|D^{-1}(x-z_m)\|^2/(2h^2)),\quad
f_i(x)=\operatorname{DAGSNet}_{\theta_t}^{shared}(x),
\]
\[
G_{imy}=\sum_{(x,l)\in D_i}k_m(x)[p_i(y\mid x)-1(l=y)],\qquad
N_{im}=\sum_{x\in D_i}k_m(x).
\]

G là batch/dataset aggregate local, tuyệt đối không export `k(x)`, `p_i(x)` hay residual từng row.
RSU secure-sum G,N rồi `t_m←t_m−eta G_m/(N_m+eps)` và broadcast target mới.
Companion fitMSE centered logits hoặc KL(T=2) vào targets; private học CEraw+KDcompanion(real local+freshGaussian).
Chỉ dùng companion sau đã fit đủ và teacher quality qua gate. Nhãn Gaussian hard là đối chứng mạnh.
Khóa centering `t_m−mean_y(t_m)` để loại gauge; clip/update trust radius cần khóa trước.

**Không có bảo đảm residual này là gradient hàm chính xác** nếu kernelprojection không đại diện
ngoàianchor. N là kernel coverage, không nhãn density riêngrecord. Đây là surrogate phải đo bias;
không gọi kernelregression DAGSNet hoặc coi targettable là learnedclassifier độc lập.

### Byte, bootstrap và giới hạn

Uplink32header + M×(16+1)×4 fp32: **4.384B(M64)** hoặc**34.848B(M512)**.
Downlink32header + M×16×2 fp16: **2.080B** hoặc**16.416B**.
Nếu denominator N gửi một lần thay vì mọiround, phải khóa dataset/cohort invariant; chưa mặc định miễn phí.
B0 M64 payloaduplink rất sát4.500B, overheadsecure-cohort+ACK có thể làm không fitmột contact.
Khoảnh thiếu byte cần chọnM nhỏ hơn bằng ledger trước pilot contact, không lược bỏ overhead.
M512 B2 đủmột contact. H2diag bootstrap cóupload~12,8kB/down~6,4kB riêng;
anchor được tái dựng đúng từdecodedstat+RNGcode/seed/hash, không gửi M×66input nếuđãtáidựng.
Coldstart trả toànbootstrap+targetsnapshot. Missedtargetversion không tính delta khi basekhác.
**Tổng byte phải cộng probe DAGSNet chung:**395.024weights fp16+32header=790.080B
cho mỗi lần sync snapshot `theta_t`. Nếu RSU fit DAGSNet rồi đổi snapshot mỗi round thìdownlink
790.080B/round, do đó K2 chưa đạtB0/B2 một contact dùG nhỏ.
Một thiết kế có thể tái dựngtheta_t từsameinit/targets/publicoptimizerstate tạimọixe; chỉ được
miễn dense snapshot nếu numeric reconstruction đã chứng minh hashcoherent, kể cảBN vàresume.
Device/kernelkhác có thể lệchbit; hash mismatch cầnresyncdense trảphí hoặcdiscardcohort, khôngâmthầm
cố dùng. Chưa proof ấy thìledger dùngdense sync. Frozenrandomprobe không gọi dynamicknowledge,
có nguycơ lặpstaticKME; nếu thử phải thành arm riêng, không thaybảnchính để giảtạothấpbăngthông.

BN companion canonical recalibrate bằng4096sharedGaussian draw/round, freeze khifitanchor và tínhG;
private BNlocal. D tính từdecodedglobalSD, h khóa từtrain statistics, không holdout.
66D Gaussian anchor có thể nằmngoàimanifold; k underflow/rareclass không coverage là **dấu hiệu loại**,
không cứu bằngteacher-confidence trên điểmao.
G/N có thể tiết lộ record qua repeated-query nếucohort thay đổi; trustedaggregate khôngDP;
khóa publicanchors vàcohort/querypolicy, không choRSU chọn anchor tại record nghi ngờ.

### Arm và cổng đề nghị

Teacher first: `anchor_own`, `anchor_fl64`, `anchor_fl512`, `anchor_label_gauss`, `dense_proxy`.
Thêm residual-shuffle giữaanchor phá địa chỉ thông tin, giữnorm/compute; chưa đủquota thì giảmM arm
trước khóamanifest, không bỏđối chứng sau nhìnkết quả.
Mọilearnedteacher DAGSNet, khônganalytic kernelclassifier triển khai.

Coverage gate pretrain:≥90%train rows có `max_m k_m(x)≥1e-3`, mọiclass≥80%; counts chỉ thống kê gộp.
Nếu không đạt, dừngK2 bảnbandwidthđó trước20 round. Cổng này chưa phải F1gate.
Teacherhead final≥localprivategp+0,05 và≥anchor_label_gausshead+0,01 cả hai seed.
Sau teacherpass mớiprivateKD; FL−own≥0,01,FL−residualshuffle≥0,01,FL−diag≥0,015,
commonprivategate nhưK1. Nếuteacher yếu: không gọithêmKD làthửkhảthiđã đạt.

## 4. K3 — synthetic mới theo gradient của DAGSNet đang học

### Bằng chứng → cơ chế

DM S-a khớp meanlatent encoder ngẫu nhiênđóngbăng, bankcốđịnh; cònchưa côlập nguyênnhân.
KhácbiệtK3 là **gradient mục tiêu toànDAGSNet đanghọc**, vàsynthetic từ noise được tái sinh mỗi5round.
Nguồn [GradientMatching ICLR2021](https://arxiv.org/abs/2006.05929)
đềxuất tốiưu dữ liệutổnghợp bằngkhớp gradientmodel; không paperFL vàkhông bảođảm riêng tư.
[Codechính thức](https://github.com/VICO-UoE/DatasetCondensation) lànềntham khảo,
không saochép raw-init/imageaugmentation sang66feature.

Produceri có DAGSNetlocal `theta_i` snapshot; giữraw batch tại xe.
Đây là **local condensation**, không gộp gradienttargets giữaproducer; không gọimatch mộtglobal
gradient khi20theta_i khác nhau. Nếu đổi sangglobalgradient-target condensation, phải dùngsame
theta_t/BN trước tínhgreal và trảdense-sync790.080B mỗi snapshot trừkhiđãproof seedreconstruction:

\[
g^{real}_{iy}=\nabla_{\theta_i}\frac1{|B_{iy}|}\sum CE(f_{\theta_i}(x),y),\quad
g^{syn}_{iy}(S)=\nabla_{\theta_i}\frac1{|S_{iy}|}\sum CE(f_{\theta_i}(s),y).
\]

TốiưuSnoisethrough `sum_layer cosine_distance(greal,gsyn)`;
greal khôngupload hoặcchia sẻ. SyntheticbeginGaussian decodedglobalmoments, khôngrawinit/nearestrecordcopy.
Mọiteacher/encoder dùng DAGSNet. RSUchỉđọcSnhận từ noise theo ngoại lệ đã chốt.
S labelcốđịnh16class, mỗi1point/class/cyclehoặc4point/class/cycle; sinhthaythế book,
không tăngbankkíchthước cứuS-a. Neighborproducerbook riêngchoxe đượcchọn bằngseed,
khôngreceiverfingerprint. RSUphát4producerbooks/cycle ởB2;B0phát1book/cycle+priorobject.

**Rủiroteacher local yếu:** gradientmatch vẫn có thể mãhóa shortcutreceiver của producer;
không tựgiảiquyết ranhgiớiđa receiver. Cần đối chứngbook Gaussian vàcross-producer
heldoutgradientalignment; khôngdùngtrainmatchloss làmđích cuối.

### Byte, BN và compute

1point/class:16×66×2 +32 = **2.144B/book/uplink**.
4point/class:64×66×2+32=**8.480B/book/uplink**.
Download4books nguyên vẹn =8.576B hoặc33.920B, cộng manifestproducer/version vàprior.
B0 nhỏ chỉfit1book/contact mỗi chiều trướcoverhead;B2đủcả4book. CohortaggregatekhôngápchoS
theo ngoại lệsynthetic; prior/moment vẫnaggregate≥5.
Khônggửiθ_i xuốngxe khác; snapshot localkhông bootstrap tham sốmạngshared.
Causalcontrast globalgradient-target nếu thêm sau phải tính cảtheta_t sync; không so partial-byte
local-gradientbook với globalprobe rồi kết luậncarrier thắng.
Codebook cũ/mớitrộn0,5 chỉnếumanifestkhóa trước; không gọi bankrotation làcơchếmới khikhông đối chứng.

Gradientmatching cómixedsecondderivative theoS vàθ, AMPkhông mặcđịnhan toàn:
FP32synthesis, freezeBN/dropout tạiproducer, BNstatslocaltraincalibrated,
privateBNpolicy đồngnhấtđối chứng. TiềnkiểmKagglebatch32 chỉ5synthesisstep đopeakVRAM/time;
không localpilotnặng. Giới hạn≤2×thời gianbaseline/cycle, ít hơn15% tổngKaggleGPUbudget
cho synthesis? **Đềnghịbudget cần chốt**, chưatựlấy đây làmHP.

### Đối chứng đủ để biết vì sao

2×2 `objective meanlatent / fullgradient` × `static / refresh5round`, cùngpoints/producer/computeaccounting;
thêm`diag`, `gauss_refresh`, `gradient_own`, `local`. MeanlatentDAGSsnapshot
**phảiđúngdefinition khớpS-a hoặc ghiadaptation**, không đổiencoder rồi gọiđãcôlập.
NếuchỉchạygradientrefreshvàDMstatic thìkhông phânbiệt refresh/objective.
Teacherindependentalignment đo trênlocaltrainbatch chưađượcdùngsynthesis:
cosine(gsyn,greal)−gauss≥0,10 trênmedianclient/lớp, benignkhông giảm; chỉconditionnecessary.
FinalFL−max(diag,gauss_refresh)≥0,015,FL−own≥0,010,gradientrefresh−meanrefresh≥0,010
cả hai seed +commonprivategate. Refresh−static≥0,005 đểclaimrefresh đónggóp.
Khôngqua thìđóngadaptation, không tăngsteps/memory saukết quả.

## 5. K4 — DAGSNet joint-energy, trần tri thức trước carrier

### Bằng chứng → cơ chế

Gaussian chưa mang quan hệphi tuyến; staticbook không đủ.
[JEM](https://arxiv.org/abs/1912.03263) diễn giải logitclassifier thành jointenergy,
cho vai trò sinh/phân loại cùngmạng. [Codechính thức](https://github.com/wgrathwohl/JEM)
cho sampling/training image; chưa chứng minh DAGSNet/tabularV2X hoặc FL.

VớiDAGS16logits `f_theta(x)`:

\[
E_\theta(x,y)=-f_\theta(x)_y,\quad
\nabla_\theta\log p_\theta(x\mid y)
=\nabla_\theta f_\theta(x)_y-E_{s\sim p_\theta(\cdot\mid y)}\nabla_\theta f_\theta(s)_y.
\]

Mỗiclient giữpositive raw local; negativechainnoise-init local/sharedseed,
SGLD `s←s+(eta/2)∇_s f_y(s)+sqrt(eta)epsilon`; khôngrawreplayinit.
LocaltrainCE+lambdaenergy positive/negative; gộp dense parameterupdate cohort≥5.
Privatexe dùngGaussianinit chuyểnqua10SGLDsteps của sharedenergyDAGS trướcCEsynthetic,
khôngxuất per-record score gradient. Khôngthêmscoremodel67/66outputs khácDAGSclassifier.

Đây khácV2-C: thayimportanceweights trênq cốđịnh bằngchuỗiinputtransport,
không phụthuộcESSimportanceweight; nhưng sampling bias/mixing/normalizer vẫnchưa biết.
BN ởenergytrain/sampling phảifreezecanonicalGaussian, dropoutoff; không đểsample batch đổienergy.
Supportdiscrete cầnhurdle/conditionalcontinuouschain vớimaskfixed; không perturbbinaryrồi gọiđúngsupport.
Nếuđưaunbounded66D lênSGLD, phát-sinh runaway hoặc artificialspikes làfailure,
không chỉclipmetric. Domainconstraints phảiđến từschema/publicbounds hoặctrainaggregatestats
đãcodec, chưatự claim physicallyvalid.

### Byte và vì sao chưa ready R1

Dense395.024parameterfp16+32 = **790.080B mỗi chiều/round**, chưaBN buffers,
prior/stat/bootstrap/ACK. B0 cần≥176contacts/chiều;B2cap231.000B cần≥4.
KhôngđượcgọiK4 low-byte. ChỉUqualitypilot trước, sauđó carrierđãpass mớighép.
Syntheticexporttừsharedenergymodel cóthểgiảmdownload vềbook K3 nhưnguploaddense vẫnlớn;
không cheupload bằngchỉbáodownloadbook.
Compute10step inputgradient/mỗi aux batch cao hơnGaussian; smoke đoVRAM/runbudget,
không claim“littleoverhead” nguồnvisionápDAGSNet66D.

### Cổng chất lượng, scientific stop

Teacherstage `dense_CE`, `dense_CE_energy`, `energy_own`; giữsamplercostdummyGaussianđối chứng
khiđánhgiáprivate. Energyteacherhead≥dense_CEhead−0,01,
energyFLprivate−max(diag,CEproxyKD)≥0,015,energyFL−energyown≥0,010 cả hai seed +commongates.
Samplerdiagnosticheldoutenergy/chainvariance/autocorrelation/invalidfeaturefraction
đến từinputsynthetic hoặcscalaraggregate, khôngexportrecord thật.
Finiteallparameters/chains, ≤1%samplinginvalid theođịnh nghĩaschemađãkhóa,
benignguard. Nếuteacher/samplerfail, không mởR1/KDfull.
10step đây làkhởitạothiết kế; cần chốteta/lambda/clipdomain từtrain-onlypreflight
trước pilot, không tựlaunch vớiHPchưađịnh nghĩa.

## 6. Quyết định đề nghị và điều loại

ƯutiênK1 vìstat analytic/chi phíthấp;K2 saucoveragecheck đểbỏnhánhanchor rỗngsớm;
K3táchrefresh/objective, giới hạncompute;K4xếpsau vìdensecarrier+samplingrủiro.
“Thửmọikhảthi” nghĩa mỗigiảthuyết cóđối chứng/cổng/dừng, không ma trậntuning vôhạn.
ChờB/Cartifact đểkhông thiết kếdựavàoK chưaqua hoặcweightedGaussian chưađượckiểm.

Loại khỏiđợt này:

- LearnedGMM/tree/flow/diffusiongenerator ngoàiDAGSNet: vi phạmràngbuộcmọilearnedmodel.
- Chia sẻlocalfeature/nearestrealrecord/latent/per-recordgradient: vi phạmR3;
  gradientmatching chỉtínhlocal vàsyntheticnoise-init đượcphép.
- FedGen gốc latentgenerator kháckiếntrúc: không triển khai; [nguồn chính](https://proceedings.mlr.press/v139/zhu21b.html)
  chỉ làmđối chiếucơchếgenerator/KD, không bằngchứngK4đãthắng.
- Vote localteacher trênGaussian khôngqualitygate: lặprủiroDS-FL đãbiết.
- Randomizedresponse per-record rồiupload: vẫnexportthông tinper-record;
  không tựđượcphép vìtênDP. DP/cohortsecuritythiết kế riêng, hiệnchưaclaim.
- Kernelanchorgradient nhưbảnsaoCountSketch parameterEF: không phảicùngpayload/objective;
  nhưngnếukhông coverage, vẫnchỉmean smoothing, phảiloại.

Khôngthao tác taybắtbuộcchoMarkdown này. K1 cầnđọcschema/generatorlocal;
K2/K3/K4 cònHP/BN/computephảikhóa trongmanifest riêng trướcKaggle. OnlineW&B/key,
sourceimmutable, fail-before-trainauth, pullverifyall10metric/prediction/weights/codec bắtbuộc.
KhôngGit/DVC. Khôngsửa/notebookB/C đangchạy.

## 7. Nguồn sơ cấp, phạm vi đã kiểm

Đãmở/tracứu nguồn sơ cấp ngày2026-10-11; khẳngđịnh nguồn ởđây chỉ mô tảcơchế,
khôngdịch kết quảvision/théory thànhIDS. Cácphương trìnhK1/K2vàcổng/byte làthiết kế/sốhọcdựán.

| nguồn | phần dùng | phần không được suy |
|---|---|---|
| [Federated Functional Gradient Boosting,2021](https://arxiv.org/abs/2103.06972) | độngcơ tốiưuhàm, giảđịnhheterogeneity | địnhlýkhông choDAGS+finiteanchor+CEsurrogate này |
| [Dataset Condensation with Gradient Matching,ICLR2021](https://arxiv.org/abs/2006.05929), [officialcode](https://github.com/VICO-UoE/DatasetCondensation) | mục tiêugradient thaymeanlatent | khôngFL/DP; không cho phéprawinit |
| [Distribution Matching,WACV2023](https://openaccess.thecvf.com/content/WACV2023/html/Zhao_Dataset_Condensation_With_Distribution_Matching_WACV_2023_paper.html) | đối chiếumeanembedding/S-a | không phảiK3 đã đạt; nguồn dùngmanyembeddings khácS-a |
| [JEM,ICLR2020](https://arxiv.org/abs/1912.03263), [officialcode](https://github.com/wgrathwohl/JEM) | logitenergy/SGLD, cùngclassifier | khôngFL/contact/tabularsupportguarantee |
| [FedGen,ICML2021](https://proceedings.mlr.press/v139/zhu21b.html) | data-freeKD/generatorđối chiếu | learnedgeneratorlatent không hợpchốtDAGS-only |
| [SecureAggregation,Bonawitzet al.](https://research.google/pubs/practical-secure-aggregation-for-federated-learning-on-user-held-data/) | tổngnhiềubên, thiết kếprivacy | trusted-sumlocal không cryptography/DP; overheadpaper không copy20client |

Khôngđặtlinkpaper phụkhông đã kiểm làmnguồn. Khôngtrích nguyênvăn nguồn.
