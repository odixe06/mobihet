# K5 — Gaussian trên primitive, tái dựng quan hệ feature bằng đại số

Ngày2026-10-12 UTC+7. **Nghiên cứu thiết kế; chưa code/train/push.** Không sửa source/manifest
V3/V4 đã đóng băng. Mọi model học được vẫn DAGSNet; đầu vào model vẫn đủ66feature gốc.
Primitive/mask/Gaussian dưới đây chỉ là thống kê analytic và bộ sinh từ noise, không mạng phụ.
Không khẳng định khả thi F1, R1, DP hoặc novelty. Chỉ dùng nguồn sơ cấp local trong §9.

## 1. Bằng chứng → giả thuyết có thể bác bỏ

P1 `h2_full` groupgp0,5697/0,5727; `h2_diag`0,5425/0,5443;
`h2_own`0,4875/0,4887. Lợi Gaussian gộp vượt làm mượt riêng xe, nhưng positionMirroring
chỉ tăng khoảng0,01 trong khi FedAvg tăng khoảng0,26. Đây là động cơ tìm cấu trúc Gaussian
không giữ được; chưa chứng minh đại số là nguyên nhân thiếu.
[P1:L14–20,46–58](probes/pilot-p1.md#21-h2--thống-kê-bậc-hai-ở-input).

Gaussian trực tiếp66cột có thể sinh `dist`, `dx`, `bearing`, `velocity` mâu thuẫn nhau,
dù covariance giữ quan hệ tuyến tính. **K5:** lấy moment trên primitive, sinh primitive mới
từ noise, tái dựng feature phụ thuộc bằng đúng công thức extractor. Tái dựng áp dụng cho mọi lớp,
kể cả attack; không sửa claim của attacker cho giống benign.

Đây là cải tiến H2 theo support/DAG analytic, không họ FL mới. R2 phù hợp giao diện input
nhưng pilot vẫn DAGSNet-only; không lấy kết quả ấy làm bằng chứng đa kiến trúc.

## 2. DAG chính xác từ extractor

Nguồn ghi feature chỉ dùng CAM nhận được, trạng thái receiver và map; không đọc nhãn/attacker.
[vnx_features.py:L1–10](../../../../dataset/script/preprocessed/vnx_features.py#L1).
Các identity dưới đây mô tả **feature đã tính từ claim**, không quan hệ giữa claim và thực tế.

Trong bảng, giá trị là **raw scale sau inverse z-score**, trước standardize lại output66cột.
Ký hiệu `sx,sy,rx,ry` là bốn `*_x_rel/*_y_rel`; origin chung triệt tiêu trong phép trừ.

| primitive → feature phụ thuộc | công thức / branch | nguồn sơ cấp |
|---|---|---|
| position4 → `f_dx,f_dy` | `dx=sx-rx; dy=sy-ry` | [L245–256](../../../../dataset/script/preprocessed/vnx_features.py#L245) |
| `dx,dy` → `f_dist` | `hypot(dx,dy)` | [L251–257](../../../../dataset/script/preprocessed/vnx_features.py#L251) |
| `dx,dy,dist` → bearing | **COMPASS:** `sin=dx/dist; cos=dy/dist`; `dist=0` → hai NaN → impute0 | [L259–262](../../../../dataset/script/preprocessed/vnx_features.py#L259), [L369–381](../../../../dataset/script/preprocessed/vnx_features.py#L369) |
| heading pairs → `f_hed_diff_cos` | `c_s*c_r+s_s*s_r` khi heading hợp lệ | [L264–270](../../../../dataset/script/preprocessed/vnx_features.py#L264) |
| speed + heading → velocities4 | `vx=spd*sin; vy=spd*cos`;0° hướng+y, góc tăng clockwise | [L272–280](../../../../dataset/script/preprocessed/vnx_features.py#L272) |
| velocities + position → `f_rel_speed,f_closing_speed` | `hypot(v_s-v_r)`; `closing=−dot(v_s-v_r,[dx,dy])/dist`;dist0 → closing0 sau impute | [L282–285](../../../../dataset/script/preprocessed/vnx_features.py#L282) |
| speeds → `f_spd_diff` | `snd_spd-rcv_spd` | [L286](../../../../dataset/script/preprocessed/vnx_features.py#L286) |
| position-noise coordinates → noise magnitudes2 | `hypot(noise_x,noise_y)` | [L287–288](../../../../dataset/script/preprocessed/vnx_features.py#L287) |
| session dt/dt_send → skew | `dt-dt_send` nếu hai primitive nguyên thủy hữu hạn; first-session →0 | [L311–330](../../../../dataset/script/preprocessed/vnx_features.py#L311) |
| dpos/dt/speed → implied/spd_residual | `implied=dpos/dt`; `residual=implied-snd_spd` **chỉdt>0**;dt≤0 → cả hai0 do NaN/impute | [L312–316,331–333](../../../../dataset/script/preprocessed/vnx_features.py#L312) |
| dspd/dt/claimed acceleration → acl_residual | `dspd/dt-snd_acl` chỉdt>0;dt≤0 →0 | [L313,317,335](../../../../dataset/script/preprocessed/vnx_features.py#L313) |
| rates → share | `sender_rate_5s/max(rx_rate_5s,1)` | [L355–360](../../../../dataset/script/preprocessed/vnx_features.py#L355) |

### Những feature không tái dựng được chỉ từ một row66cột

- `f_delay_s` cần hai timestamp; metadata không được dùng làm model input, nên giữ delay như primitive.
  [L241–254](../../../../dataset/script/preprocessed/vnx_features.py#L241), [meta:L89–110](../../../knowledge/meta.json#L89).
- `sess_dpos,dspd,dhed,dmsgid,ddist,ddre,pos_pred_err,alias_age` cần predecessor/session timeline.
  Không có predecessor trong mộtrow66feature; giữ chúng làm primitive, không suy từ claim hiện tại.
  [L290–344](../../../../dataset/script/preprocessed/vnx_features.py#L290).
- Rate1s/5s đếm toàn stream, không suy từ mộtrow.
  [L205–212,346–360](../../../../dataset/script/preprocessed/vnx_features.py#L205).
- `snd_dist_road_edge` cần map và claimed position; không map thì không tái dựng từ x/y.
  Âm là off-road claim hợp lệ trong dữ liệu attack, không clamp0.
  [L14–16,67](../../../../dataset/script/preprocessed/vnx_features.py#L14).

## 3. Bảo toàn attack: identity ≠ physical assumption

Được áp dụng cho attack rows:

- Dist/bearing/velocity là phép tính của extractor từ CAM claim, nên reversedHeading/positionMirroring/
  speedOffset vẫn phải có các feature derived đúng **claim đã bị sửa**, không claim benign.
- `first=1` đi với13session feature0; `sess_idx=0,alias_age=0` tại session start là rule cấu trúc.
  [L168–177,301–341](../../../../dataset/script/preprocessed/vnx_features.py#L168).
- Profile có3indicator mỗi side, tối đa một1; **all-zero được phép** khi giá trị unknown/null/unrecognized.
  Không ép exactly-one nếu nguồn không bảo đảm enum luôn known.
  [L49,362–367](../../../../dataset/script/preprocessed/vnx_features.py#L362).
- Receiver stream sorttime làmdt≥0; tiesdt0 được phép. Đây là thứ tự dữ liệu, không tốc độ vật lý.
  [L295,304,311–313](../../../../dataset/script/preprocessed/vnx_features.py#L295).

**Cấm thêm trong adaptation chính:**

- `implied_speed=claimed_speed`, `acl_residual=0`, `pos_pred_err=0` hoặcclampresidual nhỏ:
  xóa thông tin tấn công. Residual phải sinh từ primitive bất nhất và công thức gốc, không ép bằng0.
- Đặt claimed heading theo displacement/road tangent; ép sender cùngreceiver heading; ép vị tríon-road.
- Ép `snd_dist_road_edge≥0`; ép noise là measurement-error thật hoặc ép `snd_spd=rcv_spd`.
- Chọn/loại row vì nhãn attack, sửaprimitive đểloss benign nhỏ, dùng raw record làm syntheticinit.
- Ép `sess_dt_send>0`: replay/timestamp manipulation có thể có signed send-time differences;
  extractor không sorttheosendtime.
- Thiết lập rate từ labelbenign: nguồn đã có caveat benignstream không receiveduringflood;
  không gọi shortcut rate là quy luật V2X phổ quát.
  [L36–40](../../../../dataset/script/preprocessed/vnx_features.py#L36).

## 4. Primitive subset và mức triển khai đề nghị

### G51 — geometry trước

Loại **15feature derived geometry**:

`f_dx,f_dy,f_dist,f_bearing_sin,f_bearing_cos,f_hed_diff_cos,f_rcv_vx,f_rcv_vy,f_snd_vx,f_snd_vy,f_rel_speed,f_closing_speed,f_spd_diff,f_rcv_noise_mag,f_snd_noise_mag`.

Primitive51 gồm raw15 + position4 + `f_delay_s` + headingsin/cos4 + session16 + rate5 + profile6.
Thứ tự lấy từ `knowledge/meta.json.feature_cols`, không reorder theo dictionary/scaler.
[meta:L21–88](../../../knowledge/meta.json#L21), [schema:L13–80](../../../../dataset/centralized/feature_schema.json#L13).

Hai heading pairs vẫn được fitGaussian trên sin/cos rồi **normalize analytic** trước dựng geometry.
Đây là projected-normal generator, không Gaussian chính xác sauprojection; covariance của sinh66cột
không nhất thiết bằng covariance thật. Atancircularangle fitGaussian một chiều dễ lỗi quanh−π/+π,
nên không dùng làm thay thế mặc định. Pair gầnzero của synthetic cần redraw bằng RNGnoise,
không tự gọi finiteheading0°; valid-mask quyết định headingimputed branch.

### GS46 — mở rộng session/rate, chỉ sau auditbranch

Loại thêm `f_sess_dt_skew,f_sess_implied_spd,f_sess_spd_residual,f_sess_acl_residual,f_sender_share_5s`.
Giữdt,dt_send,dpos,dspd,snd_spd,snd_acl vàrate4 nhưprimitive. Tái dựng theo bảng §2;
first/non-first vàdt0 cầnmask. Không ép inferredspeed khớp claimed speed.

GS46 chỉ giảm dimension của moment; không còn đủ51Gaussianvariable. Đối chứng phải fit cùng46primitive
và đổi duy nhất tái dựng vs không tái dựng hoặc dùng mapshuffle (§7), tránh quy hết lợi cho giảm dimension.

### Mask/schema không được ẩn hoặc bỏ miễn phí

Đề nghị gói counts gộp riêng theo16class:

1. first-session2states;
2. joint receiver/senderprofile4×4=16states, mỗi side3known+unknown;
3. joint headingvalid2×2=4states;
4. positioncoincidence2states (`dist=0` trước bearingdivision).

Tổng24counts/class. **Đây là4marginal nhóm mask**, chưa jointfirst/profile/head/geometry.
Sinh chúng độc lập có thể bỏdependence vớicontinuous/classcondition. Chưa gọi sinh đúngjointdistribution.
Nếu conditionalmoments theojointmask16/32states, byte tăng nhân sốstate; phải khóa thống kê riêng,
không dùng bảng §6 rồi truyền thêmconditionalmoments miễn phí.

Đối chứngschema phải cùng first/profile sampler vàmoment continuous; soK5 vớiGaussian66không schema
chưa tách riêng identity vs support (K1/V4 đang thửsupport). Khuyến nghị `full66_support`
là control mạnh bên cạnh `full66_plain`.

**Branch fidelity chưa được chứng minh trên mọi attack:**
extractor impute **mọi**nonfinitecell về0, không chỉ13sessioncolumns.
[L369–395](../../../../dataset/script/preprocessed/vnx_features.py#L369),
[schema:L172–199](../../../../dataset/centralized/feature_schema.json#L172).
Một tốc độ NaN trướcimpute có thể thànhspeed0 nhưng velocity cũngimpute0;
heading NaN làmheaddiff0 vàclosing0 dùimputedvelocity0 có vẻ hữu hạn.
Không suy missingness đầyđủ từrowđãimpute. Cần audit nguyên thủynonfinite/featureflags tạilocal;
nếu không xác định đượcbranch thì **không claimidentity phổquát** và giữderived đó làmprimitive.
Không loại attack rows để đạtfidelitygate. Đọcmetadata/schema không thay auditbranchdataset.

## 5. Giao thức FL, moment và transform

Client giữtoàn bộrecord tạilocal; lấyprimitivez-score vector `u`, maskcounts,
và từng classcount/sum/secondmoment:

\[
n_{iy},\quad s_{iy}=\sum_{x\in D_{iy}}u(x),\quad
Q_{iy}=\sum_{x\in D_{iy}}u(x)u(x)^T.
\]

Fullcov chỉ gửi uppertriangle gồm diagonal; diag chỉ gửisum-square từng coordinate.
Gộp cohort≥5 rồi `mu=s/n; Sigma=Q/n-mu mu^T`, đối xứngPSD theo rule đã khóa.
Momentuploadfp32,countuint64; RSU derivefactor L rồi downloadmu/Lfp16 thật, đọcbytesdecode,
không giữshadowfp32 khitrain. Diagdownloadmu/sd. Gaussianprimitive `u=mu+L epsilon`, `epsilon~N(0,I)`.

Inverse scaler từng primitive:
`raw_j=u_j*scaler[j].std_used+scaler[j].mean`.
Applymask/rules/schema, geometryDAG, cuối `z_j=(raw_j-mean_j)/std_used` cho đủ66cột,
theo đúngmetaorder. **Không áp công thứchypot/trigonometry trực tiếp trênz-score**.
Atomraw0 thành `−mean/std_used`, raw1 thành `(1−mean)/std_used`.
[scaler:L69–100](../../../knowledge/scaler.json#L69), [meta:L150–153](../../../knowledge/meta.json#L150).

Scaler min/max ởfile là statistic train, không tự biến thành physical bounds hoặc truncation.
Không tăng feature, không đổi input66; privateDAGS dùngCEreal+CEfreshsynthetic vàBNpolicy giống đối chứng.
Thống kê tĩnh tính một lần, auxiliarydraw mới từngminibatch; không bank cốđịnh đểtránh lặpS-a.
Priorobject200B trả phí riêng; counts trongmoment không làm prior free theo ledger hiện hành.
Đâytrusted secure-sum simulation, chưacrypto/DP; raremasks vẫncó leakage risk.

## 6. Byte thực đề nghị, chưa có codec chạy

16classes, dprimitive, `T_full=d+d(d+1)/2`, `T_diag=2d`.

- Uplinkmoment `32header +16×(8count+4T)`B.
- Downlink `32header +16×2T`B cho mean+triangularfactor/SD.
- Maskobject24uint64/class: uplink `32+16×24×8=3.104B`.
- Maskdownload24fp16prob/class: `32+16×24×2=800B`.
- Prior200B là **tổnguplink+downlink bổ sung/client**, không200B mỗi chiều.
  Nếu protocol200B đượcđịnh nghĩakhác phải sửaledger trước launch.

| moment | upB | downB | tổng+prior200B | tổng+maskobject+priorB |
|---|---:|---:|---:|---:|
| full66 control |145.888|72.896|218.984|222.888|
| diag66 control |8.608|4.256|13.064|16.968|
| fullG51 |88.288|44.096|132.584|136.488|
| diagG51 |6.688|3.296|10.184|14.088|
| fullGS46 |72.288|36.096|108.584|112.488|
| diagGS46 |6.048|2.976|9.224|13.128|

Full51 giảmmomentpayload khoảng39,5% sofull66, **chưa chứng minh giữF1**.
Moment51 vẫn gồm first/profile Gaussian coordinates dù schema dùngmask sampler riêng; đây là
ledgerbản đơn giản córedundancy, khôngclaimnén tốiưu. Projection/mask có thể thaymean/cov sau sinh;
phải đo synthetic momentdistortion và giữsamplerpolicy giốngcontrol, khônggọi Gaussian51 output
là Gaussian66 giữsamecovariance.
Controlplain không cầnmaskobject; controlsupport trả cùngmaskbyte.
ACK/frame/retransmit/securitysetup/manifest ngoàiheader32 đều phải tínhledgercontact riêng.
Khôngfit B0 bằng chia tổngpayload cho20round: bootstrap coherent phải nhận thật trướcactivation.

Ởcap4.500B B0, bỏoverhead/priorchiachiều, lowerboundfragmentmoment+mask:
fullG51 up`ceil(91.392/4.500)=21`,down`ceil(44.896/4.500)=10`contacts;
fullGS46up17,down9;diagG51up3,down1;diagGS46up3,down1.
Đây lowerboundbyte, không thời gian: session13firstmask bắtđầutrainlocal dù chưa nhậntri thức,
onecontact/round thìfullG51uplink≥21round vượtU20round, chưa cộngdownload/cohortwait.
**Không chọnfullG51 làmB0 phươngánđãkhảthi.** Diag có thểtestdeadline20round riêng sauUqualitypass.
ỞB2cap231.000B, mỗi object/mỗi chiều đềufitmộtcontact ởpayload; deadline thực/cohort/grant
vàACK cònphải kiểm. B2không tựđảm bảo mọi20 clientnhận đủ.

Scaler/schema/code seed có trênxe từdatasetprotocol làbootstrappublic hiện hành;
nếu deployment chưa có thì bytes/hashes/coldstart trả riêng, không gọifree.
PSDfloor,factorizationdecodedfp16/stablemaskprob normalization rule cần khóa vàtestresumable,
đặc biệtill-conditionedcov dofeaturecorrelation/headingnormalization.

## 7. Đối chứng mạnh và cổng bác bỏ

Không dùng P1 số cũ làmcontrolruntime mới. **Fresh fullGaussian66** cùng fixture, DAGSNet,
FP32/AMPpolicy/lr/steps/privatebatch/auxbatch, lossweight, BN, sampler budget,
seed42/43,20round,group/oldholdout vàhead/gp nhưarmK5. Giữrawlocal không xuất cho mốc trên.

Nhómgeometry tốithiểu:

- `local`, `diag66`, `full66_plain`, `full66_support`;
- `G51_full_own`, `G51_full_fl`, `G51_full_shuffle`;
- `G51_diag_fl` nếu muốnkiểmB0 sauUquality;
- `GS46_full_fl` chỉ sau source/branchfidelity; thêmown/shuffleGS46 nếu muốnclaim chínhGS46.

Shufflephá **alignmentprimitive→derived** màgiữsampler budget vàmarginalđối chứng:
chỉ hoán blockgeometryderived15 giữaindependentnoise synthetic cùnglabel trongminibatch;
raw record không thamgia. Headingpairs vẫn valid,first/profile masks giữ nguyên,
vànhãn/covmomentdraw sốlượngkhông đổi. Khôngswapdx/dy rồi vôtình đổiCOMPASS.
Shuffle giữmarginalderived nhưng làmprimitive/derivedmâu thuẫn; khácfull66 khác cảsampler,
nênchỉ contrastidentityK5−shuffle cócausal interpretation hẹp.

Fidelitygate trướctrain:

1. Dùnglocaltrain rows gồm16class, reconstructfromprimitives; kiểm từng formula vớiFP16input/scaler
   quantization tolerance xácđịnh trước. Allclass/branch coverage; không loạiattackrecords.
2. Float64toyrows geometrycompass,coincidentpositions,first/nonfirstdt0,negativeclaimedacceleration,
   unknownprofile,missingheadingbranch; sameextractorformulas/impute0.
3. Numericallfinite; syntheticatom/profile/first-sessionviolations0 theo branch đãkhóa;
   không claimrate/sessiontrajectoryconsistent chỉtừsingle-rowprojection.
4. Báorealrecord residualdistribution theoformula **tạilocal**, chỉgộpscalarcounts/max/quantilehist,
   khônguploadrecord hayresidualvectorper-row. Fidelitykhông chứng minhF1.

Đề nghịUgate finalr20, cả hai seed:

- `G51_full_fl−max(full66_plain,full66_support)≥0,015` đểclaim **thêmquality**;
- `FL−own≥0,010`, `FL−shuffle≥0,010` đểclaimtri thức gộp/identityalignment;
- groupgp−local≥0,03;oldgp>local;benignF1≥local−0,03;
- BáopositionMirroring/timeDelayF1 riêng, không selectHP từofficialtest.

Cổng **carrier nén** là claim khác: FLfull51−full66support≥−0,005 vàactualbootstrapbyte≤0,65×control,
FL−own/shuffle/commonguard vẫnpass. Mốc0,65 chỉđềnghị; maskedfull51tổng136.488/222.888≈0,6124.
Nếuqualitygatefail nhưngcarrierpass, không gọiqualityimprovement. Khôngselectgiữacổngsau results;
chốt trước claim/cổngtừngarm trongmanifest.
Contactchỉsau hướngUqua: decodedpayloadactual,B0/B2,priorpaid,cohort≥5,timeout/version,
activation20round vàgainretention≥0,70 soU +FLgain≥0,01 soown/localundercontact.
Khôngkéo dàiđểcứubootstrapB0 hoặc tăngsamplersteps sauđọcsố.

## 8. Điểm phải hỏi chủ dự án trước triển khai

1. **Phạm vi:** geometryG51 trước haycảGS46session/rate? KhuyếnnghịG51trước; GS46 branchaudit phức tạp.
2. **Mục tiêuclaim:** quality≥+0,015 haycarrier≤−0,005/byte≤0,65; cócho phép2cổng riêngđãkhóakhông?
3. **Mask:** dùng4marginalmaskgroups bảnnhẹ vàcontrol cùng schema, hayconditionalmoments theojointmask
   (tăng byte)? Không tựgiả địnhjointindependence làđúng.
4. **Missing/nonfinite:** nếu không thể suy nguyên thủybranch từimputed66features, giữderived bịảnhhưởng
   làmprimitive haykhông triển khai nhóm đó? Khôngcóquyềnloạiattack rows đểtái dựngđúng.
5. **Rule support:** cho integer/nonnegative ratecount vàsessionidx bằnground/projection analytic theo extractor?
   Cầnphânbiệtcounts structural vớispeed/acceleration physical; không clampclaimsigneds vôcớ.
6. **DùngsameBN/sampler/computecontrols vàfreshfull66** dùtốnthêmarm; protocol50round/full scale
   chưa được yêu cầu choK5, pilot20roundtrước.

Khôngthao táctaybắtbuộcchofile nghiên cứunày. Các câu hỏi làscientificchoices trướcimplementation,
không requestpermission đọcfile. K1/V4đãpush giữ nguyên; chờresults nhưngkhông tuneK5theoofficialtest.

## 9. Nguồn đã đọc, không nguồn thứ cấp/web

| file sơ cấp | phần đã dùng |
|---|---|
| [vnx_features.py](../../../../dataset/script/preprocessed/vnx_features.py) | L245–288geometry;L290–344session;L205–212,346–360rate;L362–381profile/impute |
| [feature_schema.json](../../../../dataset/centralized/feature_schema.json) | L13–80featureorder;L172–199imputation/13columns |
| [knowledge/meta.json](../../../knowledge/meta.json) | L21–88order16classes/66features;L89–110droppedmetadata;L150–153scalertrain-only |
| [knowledge/scaler.json](../../../knowledge/scaler.json) | L69–100dx/dy/firstmeans/std_used;L553–584senderheadingmeans/scales; toànfeaturesmapping choinverse |
| [P1](probes/pilot-p1.md) | L14–20full/diag/own;L46–59cơchếvàpositionMirroringgap |
| [LESSONS](../../../LESSONS.md) | L98–105support/receiverboundary;L114–122codec/prior/activation; quyếtđịnhDAGS-only/local-smoke/noGit |

Sốbyte §6 đã đối chiếu số học với16class/uppertriangle; chưa cócodec/KaggleF1 thực thi K5.

## 10. Bounded local algebra preflight —2026-10-12

[Script](../../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/scripts/probe_synthetic_book_algebra.py), [aggregate evidence](../../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/probes/algebra-probe.json).
Không model/train. Uniform sample seed`5501+cid`, tối đa2.048trainrow/client,20client:
**40.960rows**, peakRSS259MiB qua RAMguard, exit0. Không scan officialtest hoặc toàntrain.
Output chỉ count/residual quantile/fraction/sourcehash; không row/index/label/latent/gradient export.
Source/scaler/fixturemanifest vàscriptSHA lưu trongJSON.

Inversezscore dùng `std_used`/mean từngfeature. Mỗi giá trịfp16 được mở thành interval giữa
midpoint với neighboring representables (`np.nextafter`); thêmallowance
`8×float32eps×max(1,abs(rawinterval))` cho extractor/zscore float32rounding.
Interval arithmetic được propagate qua subtraction, product, hypot vàpositive-distance division;
derivedcolumn cũng có intervalquantization. Vi phạm = haiinterval không giao nhau.
Không so rawidentity bằngexact equality, không chọn tolerance từresidual đođược.
Interval này bảo thủ, bỏcorrelation giữa variables; PASS không chứng minh center reconstruction chính xác.

**17identity/heading-unit checks:0intervalviolations trên branches được kiểm.**
First-session7.432rows,non-first33.528rows. Không row nào cóheadingpair consistentraw(0,0),
không row nào có`dist`consistentraw0. **6rows** cóposition-derived distance interval chứa0:
bearing/closing loại khỏifraction denominator, vẫn giữcount; không gọi6rows PASS hoặc missingbranch.
Bearing/closing kiểm40.954rows; identities khác kiểm40.960rows.

| identity | abs residual raw p50 | p99 | max | propagated tolerance p99 |
|---|---:|---:|---:|---:|
| dx từrelativepositions |0,09296|0,74669|1,64816|1,70425|
| dist từrelativepositions |0,21264|1,12339|1,66985|1,74613|
| bearing_sin COMPASS |0,000753|0,025839|0,373289|0,161620|
| bearing_cos COMPASS |0,000480|0,030268|0,781645|0,265191|
| closing_speed |0,003518|0,248099|20,068772|2,908539|
| rel_speed |0,001848|0,024197|0,062373|0,060619|
| hed_diff_cos |0,000183|0,000802|0,001122|0,001233|

Hai p99 trong mộtrow bảng không nhất thiết cùngrecord; maxresidual có thể thấp hơn tolerance của
đúngrecord dù lớn hơn p99tolerance. JSON giữintervalgapquantiles vàsubgroupfirst/zero-pattern counts.
Nearzero-distance bounds lớn: maxbearingcos tolerance20,48175; maxclosing tolerance330,53452.
**Không lấy0violations vớibound rộng làmbằngchứng tái dựngfaithful tại center.**

Kết luận hẹp:

- Sample hợp vớigeometryalgebra nguồn trong quantizationbounds; không thấybằngchứng phá identity
  ởbranches cócoverage. Không chứng minhmọiattack/sourcebranch.
- Rawpositionsfp16 rồisubtraction có thể mất độ chính xác tương đối lúcsender/receiver gần nhau.
  G51dựngdx/dy từpositions không tái tạo centerderived originallycomputed-before-quantization.
- Headingimputation/coincidentpositions không xuất hiệntrongsample, nên khôngidentifiable/validated
  bằngsample này. Speed/noise0 cóthể làvalidraw0 hoặcimputednonfinite;66imputedfeatures không giữflaggốc.
- G51 **chưa đủ bằngchứng faithful toànbranch/toànrow**; chưa đổimask hoặc thêmrepairtheokếtquả.

Cần chốt trước triểnkhai saupreflight:

1. Chấp nhận quantizationconsistent nhưngcenter-error lớn ởsmall-distance làmthiết kếG51,
   hay giữ`f_dx,f_dy` làmprimitive riêng (**G53**, cầnmanifest/byte/đối chứng mới)? G53 chỉ làđềnghị,
   chưa kiểm. Giữdx/dy giảmsubtraction-error nhưngkhông cònép position-derived displacement exact.
2. Với originalmissingflags khôngquan sát được, giữ cácderived bịảnhhưởng làmprimitive,
   haycho phép đọcauditflagnguyên thủy tại local? Khôngexportflagper-record vàkhôngloạiattackrows.
3. Cầnsynthetictoy branch test độc lập/coincident/missing đểchốtimplementationfidelity;
   không cứu6quantization-ambiguous rows bằngrule mới saukếtquả.
4. Nếu giữG51, phải khóa cảfidelitycriterion center-error/quantizationbound hợp lýtrướcpilot;
   không chỉintervaloverlapgate rộng. Chưa F1/R1/DPclaim.
