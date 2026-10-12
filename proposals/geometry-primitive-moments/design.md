# Gaussian primitive hình học — K5

Ngày 2026-10-12, UTC+7. **Thiết kế cập nhật theo quyết định chủ dự án; chưa triển khai phương pháp hoặc train.**
Đầu vào IDS luôn đủ 66 feature gốc. Mọi mô hình học được đều DAGSNet. Gaussian, mask,
phép đại số chỉ là bộ sinh analytic tại xe từ nhiễu. Không gửi record, latent hay gradient theo record.
Source/notebook/manifest V3–V5 đã chạy giữ nguyên.

## 1. Điều đã chốt và kết luận mới

Chủ dự án yêu cầu kiểm cả G51/G53; hoãn projection session/rate; thử hai cách giữ mask:
nhóm marginal và joint32 `first × receiver_profile4 × sender_profile4`.
Trong mỗi joint32 cell, giữ thống kê atom của 13 feature đầu phiên; **không** dùng bảng
joint đầy đủ gồm `2^13` tổ hợp atom. Cell không đủ ≥5 xe đóng góp hoặc đủ mẫu cho covariance
phải dùng moment toàn lớp **và mask của cell**. Giữ derived bị ảnh hưởng khi nhánh missing/nonfinite không xác định
từ 66 feature đã impute. Chốt hai cổng riêng: tăng F1 ≥0,015; hoặc giữ F1 trong −0,005 với byte ≤65% control.
Đối chứng fresh full66-plain, full66-support, own, shuffle; cùng BN/sampler/compute.
Fidelity trước train; không qua thì dừng, không loại attack rows để cứu cổng.

**Toy sơ cấp đã bác bỏ tái dựng nguyên bản G51 và G53 trên mọi nhánh.**
[Probe](../../scripts/probe_k5_missing.py) gọi trực tiếp extractor/imputer gốc, không dùng dữ liệu thật;
[bằng chứng](fidelity-missing.json): 47 case/94 toy rows, peak RAM 112 MiB, exit0.
Giữ bảo thủ tám derived bị ảnh hưởng khiến cả hai nominal design thành **cùng một primitive59**.
Không chạy hai arm giống nhau rồi gọi là so G51/G53. Đây chưa phải chứng minh dimension tối thiểu;
chỉ là kết quả áp đúng chính sách giữ feature chưa xác định missingness.

**Sau proof, chủ dự án chọn kiểm cả G59 và G61 cho quality**, marginal và joint32;
full-cov carrier65% ghi FAIL từ estimate đại số, không hạ ngưỡng để cứu.
G61 giữ thêm bearings2, chỉ tái dựng velocity4+heading difference.
G59 giữ đúng inverse `dist>0` của extractor; distance/bearing support sai báo diagnostic,
nonfinite fail. Không claim cả geometry manifold đúng.
[Quyết định đã chốt](decisions.md). Chưa notebook G59/G61 được push tại lúc ghi thiết kế này.

Đọc nền: [nguồn và probe algebra trước](../../docs/research/v5-algebra-sources.md),
[P1](../../docs/research/probes/pilot-p1.md), [V4-support](../../docs/research/probes/pilot-v4.md).

## 2. Bằng chứng → cơ chế → dấu hiệu có thể bác bỏ

P1 Gaussian full-cov gộp hơn own Gaussian và diag; covariance input mang tri thức quần thể.
Nhưng positionMirroring chỉ tăng khoảng0,01 trong H2, trong khi FedAvg tăng khoảng0,26.
Đây là động cơ giữ cấu trúc feature; chưa chứng minh phá cấu trúc là nguyên nhân duy nhất.
[Kết quả P1](../../docs/research/probes/pilot-p1.md).

V4 giữ binary/profile/13 atom đầu phiên đúng, nhưng structural−diag −0,00212/−0,00824,
trượt cả hai seed. **Fidelity support không tự tạo lợi IDS.** K5 phải kiểm riêng lợi phép tái dựng,
lợi covariance theo nhóm mask và chi phí chúng gây ra.
[Kết quả V4](../../docs/research/probes/pilot-v4.md).

Giả thuyết K5: fit moment trên primitive thật sự đủ thông tin; sinh primitive từ nhiễu;
tái dựng derived bằng DAG extractor; private DAGSNet train CE real + CE synthetic toàn backbone.
Joint32 có thể giữ phụ thuộc profile/first/continuous bị marginal mất.
Nếu fidelity không qua, joint quá thưa, hoặc F1 không vượt đối chứng đã khóa thì dừng adaptation đó.
Không sửa claim attacker để giống benign, không chọn best round/head thay `gp`.

## 3. Source sở hữu định nghĩa, không suy từ vật lý lý tưởng

Thứ tự feature: [meta](../../knowledge/meta.json), [schema gốc](../../../dataset/centralized/feature_schema.json).
Công thức: [extractor](../../../dataset/script/preprocessed/vnx_features.py).
Imputer gốc thay **mọi** cell nonfinite thành raw0, không chỉ 13 session columns.
[Extractor:369–395](../../../dataset/script/preprocessed/vnx_features.py#L369).

Phép tính luôn trên raw scale sau inverse z-score:
`raw_j = z_j * std_used_j + mean_j`; output đủ66 cột rồi standardize lại đúng một lần.
Không lấy hypot, trừ tọa độ, hay sin/cos trực tiếp trên z-score.
[Scaler](../../knowledge/scaler.json), [meta](../../knowledge/meta.json).

| Nhóm | Công thức nguồn | Nhánh quan trọng |
|---|---|---|
| displacement | `dx=snd_x_rel−rcv_x_rel`, `dy=snd_y_rel−rcv_y_rel` | position nguyên thủy missing có thể làm derived NaN→0 |
| distance | `hypot(dx,dy)` | computed trước impute; imputed dx/dy không đủ suy missingness |
| bearing | COMPASS: `sin=dx/dist`, `cos=dy/dist` | `dist≤0` nguồn đặt inverse NaN; imputer→both0 |
| heading difference | `cos(snd_angle−rcv_angle)`; tương đương dot hai sin/cos pair trên nhánh hữu hạn | missing heading làm pair và difference0 |
| velocity | `vx=spd*heading_sin`, `vy=spd*heading_cos` | missing operand làm product NaN→0; product của zero-imputed operands cũng0 |
| relative speed | `hypot(v_s−v_r)` | một velocity nguyên thủy missing làm toàn phép tính NaN→0 |
| closing | `−dot(v_s−v_r,[dx,dy])/dist` | missing speed/heading/position và coincidence đều có thể→0 |
| speed difference | `snd_spd−rcv_spd` | trừ các speed đã impute không tương đương impute của phép trừ |
| noise magnitude | `hypot(noise_x,noise_y)` | một coordinate missing khiến magnitude0, không phải norm coordinate còn lại |

Nguồn [geometry:245–288](../../../dataset/script/preprocessed/vnx_features.py#L245).
Velocity/heading nói về CAM **claim**, không vận tốc thật. Không ép sender heading theo displacement,
không ép residual0, speed bằng implied speed, vị trí on-road, hay road-edge distance không âm.
Session/rate luôn primitive trong đợt này; giữ delay, session predecessor summaries và road-edge feature.
Không thêm timestamp/rawflag/new dataset vào đầu vào hoặc yêu cầu phục hồi dữ liệu nguyên thủy.

## 4. Fidelity-missing: vì sao tên51/53 không còn đúng

G51 cũ loại15 derived:

`f_dx, f_dy, f_dist, f_bearing_sin, f_bearing_cos, f_hed_diff_cos,
f_rcv_vx, f_rcv_vy, f_snd_vx, f_snd_vy, f_rel_speed,
f_closing_speed, f_spd_diff, f_rcv_noise_mag, f_snd_noise_mag`.

G53 cũ giữ thêm dx/dy, loại13 còn lại. Vẫn không giữ missingness gốc.

Probe dựng hai thế giới chỉ khác một raw input: NaN và **giá trị raw0 hợp lệ**; chạy cùng extractor,
imputer. Một số cặp có **primitive nominal giống hệt** nhưng derived đích khác nhau:

| Thay đổi raw toy | Derived khác dù G51 primitive giống | G53 còn counterexample? |
|---|---|---|
| speed NaN so0, xe kia đang chuyển động | relative speed, closing, speed difference | Có |
| noise coordinate NaN so0, coordinate kia khác0 | noise magnitude | Có |
| position coordinate NaN so0, receiver coordinate khác0 | dx/dy, distance, bearings, closing | Một số cặp dx/dy giúp phân biệt |
| position coordinate NaN so0, receiver coordinate0, axis kia cách xa | distance, bearing, closing; dx/dy giống | Có |

Đây là không đủ thông tin, không phải lỗi optimizer. Một phép tái dựng deterministic từ primitive
giống nhau không thể trả hai target khác nhau. Không cứu bằng tolerance rộng, loại row hoặc đọc flag bị cấm.
[Bằng chứng từng case](fidelity-missing.json).

Theo chính sách bảo thủ **không suy missingness từ residual giữa những derived khác**, giữ:

`f_dx, f_dy, f_dist, f_rel_speed, f_closing_speed, f_spd_diff,
f_rcv_noise_mag, f_snd_noise_mag`.

Chỉ còn7 derived được tái dựng:

- velocities4 từ speed và heading sin/cos đã impute;
- heading difference từ dot hai heading pair;
- bearings2 từ **dx/dy/dist được giữ**, branch inverse như extractor.

Nominal G51:51+8=59. Nominal G53:53+6=59. Cùng thứ tự và cùng primitive list.
Sau bỏ7 tọa độ binary/profile đã được mask sinh exact, covariance thực có thể dùng **52 coordinate**;
con số52 là dimension thống kê, không đầu vào model. Full66-support tương ứng59 coordinate thống kê.
Phải báo cả nominal name, actual primitive count và actual covariance dimension.

**G61 đã được chọn kiểm cùngG59:** giữ bearings2; loại chỉ5 velocity/heading difference.
Actual primitive61, covariance continuous54; first1 cell có41 active coordinate sau bỏ13structuralconstant.
G59 tương ứng52/39; full66-support59/46. Chênh G59/G61 kiểm lợi/hại riêng của tái dựngbearing bằnginverse,
không gọi G61 là triển khai G53. [Toy proof bổ sung](fidelity-missing.json) lưu5formulaPASS.

47 toy case gồm finite, COMPASS0/90/180/270 mỗi side, coincidence, unknown/null profile,
12 raw field×NaN/+∞/−∞, first/nonfirst trong từng case.
Tất cả output sau impute hữu hạn; first13 atom đúng;7 phép tái dựng còn lại qua `rtol=atol=1e−12`
ở float64. **Không** kiểm FP16 codec, dữ liệu thật, branch frequency, Gaussian fidelity hoặc F1.
Không chứng minh7 phép này bitwise đúng giữa backend. Tái dựng heading/dot có sai số số học nhỏ.

## 5. Mask: marginal và joint32 đúng phạm vi đã chốt

V4 không có bốn mask `first/profile-joint/heading-valid/coincidence` như bản K5 nghiên cứu ban đầu.
V4 thực tế dùng first binary, receiver profile4, sender profile4, và atom theo từng coordinate;
13 session atom có ràng buộc first. Không nhầm bảng24count cũ với phương án mới.
[Source V4-support](../../mh/v4_support.py), [manifest V4](../../docs/research/pilot-v4-support.md).

Đợt K5 giữ bốn nhóm **quan sát được sau impute**:

1. First:2 trạng thái raw0/raw1.
2. Receiver profile:4 category normal/cautious/aggressive/unknown.
3. Sender profile:4 category tương tự.
4. Atom raw0 trên13 structural session columns; raw0 không tự được gọi missing.

Profile unknown là cả3 indicator0; không ép exactly-one.
First=1 bắt buộc13 structural coordinate atom0.
[Nguồn profile/session](../../../dataset/script/preprocessed/vnx_features.py#L301),
[schema frozen làm đối chiếu](../../mh/v4_schema.py).

### M — marginal theo lớp

Gộp class moments liên tục; counts first2, receiver4, sender4,13 atom counts.
Sinh first/profile independently **given class**, giữ ràng buộc first13.
Nonfirst atomprob của coordinate j:
`(atom_count_yj−first1_count_y)/first0_count_y`, clipped[0,1] chỉ để chịu quantization.
13 atom draw độc lập given class/nonfirst; không giữ joint atom pattern.
Đây approximation explicit, không giữ dependence profile↔first↔continuous.

### J32 — moment theo `first×profile_rcv×profile_snd`

State key công khai `s=16*first+4*rcv_category+snd_category`,0…31.
Thống kê theo từng `(y,s)`: record count, số xe có record trong cell, sum và upper-triangle second moment
của continuous primitives,13 atom counts. First/profile coordinates không cần Gaussian moment.
First=1:13 structural coordinate fixed raw0; được bỏ khỏi covariance active của cell.
First=0:13 atomprob/nhánh nonatom riêng trong cell.
Giữ exact categorical combination và dependence continuous theo32state này; **chưa** giữ joint13atom,
chưa giữ heading-valid/coincidence mask hay trajectory/session dependence.

**Không sử dụng full joint atom catalogue.** Full enumeration first×profiles×13atom có
`16*(2^13+1)=131088` structurally possible states/class:8192 nonfirst pattern +1 first pattern,
nhân16 profile combinations. Counts `n:uint64 + contributors:uint32` đã khoảng25.17MB/xe,
chưa moment. Đây thiết kế khác, chủ dự án đã chọn J32.

## 6. FL, rare cell và fallback không lộ record

Mỗi xe chỉ tính thống kê tại local. SecureSum mô phỏng cohort≥5; production pilot cohort20.
RSU chỉ đọc **tổng cohort**, không individual mean/cov/state keys. Không dùng synthetic exception
để hợp thức hóa đọc individual empirical moments. Không có learned GMM hoặc model phụ.

J32 dùng hai phase knowledge, cả hai trả byte/RTT:

1. SecureSum fixed32cell counts,13 atom counts và contributor indicators. Contributor indicator mỗi xe/cell chỉ0/1.
   RSU xác định cell eligible từ tổng số xe và tổng N, broadcast eligibility cùng class/state weights/mask.
   Không broadcast danh sách contributor/cid hoặc individual occupancy.
2. Chỉ cell eligible gửi sum/Q qua SecureSum; cell ineligible không xuất cell continuous moments.
   Luôn có globalclass moment/mask đã gộp≥5 để fallback. Cell eligible đọc đủ coherent snapshot,
   RSU derive/decode actual factors và broadcast.

Điều kiện đã chốt: ≥5 contributor và đủN. Threshold chung mọi population/control/candidate:
`N_min=max(32,d_control_active+1)`; dùng dimension full66-support, không giảm threshold cho candidate.
Nonfirst d_control59 ⇒N≥60; first1 d_control46 ⇒N≥47. Own dùng cùng N rule tại local.
N≥d+1 chỉ là guard tối thiểu cho full covariance, không bảo đảm estimator có chất lượng.
Dimension bỏ7 binary/profile; first1 cell bỏ13 fixed atom.
Covariance singular do correlation vẫn có thể xảy ra ngay khi N≥d+1; phải cùng PSD policy với control.
Nếu class global cũng thiếu≥5 contributor/đủN, dừng arm trước train; không lấy một xe làm fallback.

Fallback đã được chủ dự án chọn: toàn lớp `mu_y,Sigma_y` **và mask cell**; không lấy mean/cov của rare cell,
không điền từ một donor vehicle. Giữ sampled first/profile state, áp schema first13;13atom dùng
probabilities của `(y,s)` từ SecureSum phase1. Báo mass/count cell fallback theo class, không gọi
fallback sample là sample từ conditional density của rare cell.

N=1 cell moments sẽ trực tiếp mã hóa record; vì vậy không gửi nó rồi chỉ “không dùng”.
Một cohort20 không đảm bảo từng cell có≥5 xe. Fixed catalogue/counts không tiết lộ cid occupancy,
nhưng rare cell cóN=1 làm13atom counts lộ một zero-pattern gộp. Chính sách đã cho phép giữmaskcell,
không đồng nghĩa có DP hoặc chặn mọi leakage. Không gửi continuous mean/Q của cell ấy.
Trust model và limitation phải ghi trong report.

## 7. Moment, sampler và numerics cần đóng băng

Với continuous primitive vector u của cell/class, moment trên tọa độ z-score hiện hành:

`n = Σ1`, `sum = Σu`, `Q = Σu uᵀ`.

Chỉ uppertriangle Q được truyền. Local sums/Q dùngfloat64, covariance từ aggregate làmfloat64 để hạn chế cancellation;
wire precision là lựa chọn codec phải được đo thật, không ghi bytefp32 nhưng dùng shadowfp64.
Ledger §8 tính proposedfp32 sums/Q và countuint64; hợp đồng pilot khóa actualcodec trước train.
Server covariance `Q/n−mu*muᵀ`, symmetrize, factorizationfp32 chung mọiarm,
Cholesky với ridge cố định `1e−4*I` như P1full, chung control. Đây numerical policy đã có nguồn,
không eigenspectrum floor đảm bảo mọi uploadfp32 luônPSD.
Cholesky fail ⇒ numerical fidelity FAIL; không tự tăng ridge sau đọc số.
[P1source:295–314](../../mh/pilot.py#L295).
Factor/mean xuốngfp16 thật; train từ decoded values, không shadowfp32.
FP16 factor overflow/nonfinite ⇒ fidelity FAIL, không lặng lẽ dùngfp32.

Continuous sample mới từng private minibatch; không bank tĩnh. **Recipe đã chọn: full-cov Gaussian
từ moment toàn cell/class + overlay13atom**, không fit conditional-nonatom covariance hoặc thêmcopula.
Atom=1 thì overwrite coordinate bằng raw0 standardized đúng; atom=0 giữ Gaussian draw gốc.
First/profile exact từ maskstats; first1 ép13atom. J32 rarecell dùng globalmoment +cellmask.
**Không headingpair renormalization**, không distanceprojection/truncation mới, giống control/candidate.
Overlay thay moment của sample cuối; báo distortion, không gọi output giữ đúng covariance gốc.
Moment mới là approximation: chưa giữ joint13atom hoặc mọi nonfinite/geometrybranch distribution.

Giữ mọi session/rate value primitive, không projection integer/rate/session extra trong đợt này.
Không gọi independent point là nhất quán với trajectory chưa quan sát.
Raw0 atom standardize bằng scaler, lượng tử đúng như fixture; inverse scaler không khôi phục raw64 precision.

**G59 có vấn đề support đã chấp nhận để kiểm:** Gaussian primitive giữ distance có thể sinh âm; retained distance có thể
không bằng norm dx/dy. Bearings dựng từ chúng có thể không unit. Việc giữ derived để không mất branch
không tự làm Gaussian generator đúng. Fold `abs(dist)` vẫn có density gần0, khiến `dx/dist` có heavy tail;
không bảo đảm numeric vì mọi denominator đều dương. Ép `dist=hypot(dx,dy)` ghi đè feature đã chốt giữ;
unit-normalize bearing thay công thức extractor. **G59 dùng nguyên inverse `dist>0`, còn lạiNaN→0**
cho bearing giống extractor; không tự clampdenominator. Gaussian draw dist≤0 làm bearing0 nhưng
distance input vẫngiữ giá trị đãdraw. Báo fraction negative distance, bearingnorm sai, denominatornear0,
max/quantile bearing vàsamplevariance distortion; đây diagnostic, không gọi synthetic support hoàn toànđúng.
G61 giữ bearingssampled, không dùnginverse;5formula product/dot vẫn áp y hệt.
Mọi outputnonfinite⇒FAIL trước train/commit, không clipinput cứu run.
Fidelity synthetic bắt buộc chỉfinite+binary/profile/first13 đúng và5/7function decoderđúng;
headpairunit/distancepositive/fullgeometry consistency làdiagnostic đãnêu, không âmthầmđổi gate.

## 8. Byte: estimate minh bạch, codec thực mới quyết định cổng

Gói proposed đơn giản `T(d)=d+d(d+1)/2`,16classes,32B header/object:

- global moment up `32+16*(8+4*T)`;
- mean/factor down `32+16*2*T`;
- marginal mask23count/class:2first+4receiver+4sender+13atom ⇒ up2976B,down768B;
- prior200B/client **tổng hai chiều** thêm riêng theo ledger hiện hành.

| actual Gaussian d, chưa bỏ binary/profile redundancy | up moment B | down moment B | tổng moment+mask+prior B |
|---|---:|---:|---:|
| nominal51, fidelity FAIL |88288|44096|136328|
| nominal53, fidelity FAIL |95136|47520|146600|
| conservative59 |117216|58560|179720|
| conservative61 |125088|62496|191528|
| full66 |145888|72896|222728|

Đây **estimate schema proposed**, không codec V4 đã chạy và không byte đã đo.
Bảng24count cũ trong nguồn nghiên cứu trả136488B cho51; bảng23count mới khác vì mask khác.
Không so cổng65% qua hai ledger mask khác nhau.

G59/full66 theo bảng≈80,7%;G61≈86,0%; không đạt65%. Nếu bỏ7 fixed categorical coordinate chung cho support arms,
moment d52/d59 cũng gần78%, trước mask overhead. Không hứa compressionPASS bằng đổi tên “G51”.
Cổng quality vẫn giả thuyết có thể kiểm sau recipe/fidelity qua.

J32 không tự truyền32×full66 nếu biết fixed coordinate. Với e eligible cells, moment payload:

`4*Σ_(y,s eligible) T(d_s)` up và `2*Σ T(d_s)` down, cộng global fallback packet,
cell IDs/flags, means/factors và headers.
Count/mask phase fixed512cells: `32+512*(8+4+8*13)=59424B` up;
eligibility bitmap64B + stateprob table1024B +13atomprob table13312B + header32B =14432B down,
trước cohort/setup/ACK. Cảrare cell cũng trả13atomcounts/probabilities; không chỉ eligible cell.
First1 cell atom deterministic nên có thể bỏ13counts theo schema frozen, tất cả arm cùng policy,
nhưng bảng fixed đang tính conservative đủ32state và ghi rõ actualcodec sau tối ưu.
Tổng phải đo thực với occupied/eligible count, fallback global packet và slot directory.
Không chỉ báo “average d” bỏ cell bootstrap.

B0≈4500B/contact: global full59 moment up ít nhất27contact/down14contact,
chưa masks/prior/ACK. U20round không chứng minh kích hoạt đượcB0.
B2≈231000B/contact: global59 mỗi chiều vừa payload1contact; J32 có thể vượt do nhiều cell.
R1 chỉ sau contact ledger coherent/bootstrap/deadline/ACK/retry, không từ utility pilot.
Schema/scaler/code seed nếu xe chưa có phải trả phí cold-start riêng.

## 9. Ma trận kiểm nghiệm nhỏ nhất và thứ tự

**Bước0 đã hoàn tất:** primary missing/nonfinite toy → nominal51/53 FAIL,
retention59 collision; chủ dự án chọn G59/G61quality. Chưa train; không đưa duplicate59 lên Kaggle.

**Bước1 trước train:** chốt recipe actual primitive và support; bounded train-only probe
FP16 center reconstruction từng feature/class/nearzero/zeroheading; counts32state và contributor≥5,
N threshold; actual codec byte + decode sampling. Không đọc officialtest.
Toy decoder không đạt hoặc factor/atoms khôngfinite ⇒ dừng trước pilot.
Center-error audit không dùng interval-overlap rộng để gọi tái dựng nguyên bản toàn row faithful.
G59/G61 chỉ5/7formula được phépdựng, không claim tương đương các15formula nominal.

**Ma trận15arms/seed đã chọn:**

- shared fresh `full66_plain`, `full66_support_M`, `full66_support_J32`:3;
- `G59 × M/J32 × FL/own/shuffle`:6;
- `G61 × M/J32 × FL/own/shuffle`:6.

Dùng auditedV2 local reference cho common guard, phải match fixture/source/protocol/seed;
không thêm freshlocal chỉ để lặp control đãverify. Full66own/shuffle không cần cho claimcandidate−own/shuffle;
joint contribution luôn đo J−M ở cảfull66 vàcandidate, không quy genericjoint lợi cho geometry.
Tổng30run×20round; chạy4job/kernel như pilot trước, ngân sách lấy từ runtime audited control+15%.
Không lấy quota RUNNING để báo cost cuối. Train Kaggle, local chỉ bounded smoke/report.
Notebook mới W&B online trước train, APIkey ởprivate_push; missingkey/connection fail trước train.

Own: own moment/mask, fp16 roundtrip local, cùng shared paid prior; không nhận peer statistics.
J32 own cell có moment hoàn toàn giữ local: áp cùng N guard, thiếuN dùng **own-class** fallback.
Guard≥5contributors áp khi **phát moment ra RSU**, không ép own cell local có5 xe rồi làm mọi cell fallback.
Own không broadcast cell counts/moments hoặc nhận population mean để lấp cell hiếm.
Counts/N local khác population là khác biệt dữ liệu vốn có của control; báo fallback mass riêng,
không gọi tri thức own và population cùng effective capacity nếu N chênh nhiều.

Shuffle: mỗi minibatch có **hai batch synthetic độc lập từnoise cùnglabel/first/profilestate**;
swap toàn5derived block G61 hoặc7derived block G59 của batchB sangprimitive batchA.
Không raw record hoặc nhãn thật khácxe. Giữ marginalderived/category/schema, phá alignment.
Mọi FL/own/control arm cũng tạo cảhai batch và chạy phép dựng tương ứng để match extraRNG/compute;
chỉ FL/own/control dùng batchA coherent, shuffle dùng blockB. Không resample do label hoặc holdout.
Singletonstate vẫn có independentpair; nếu derived block degenerate thì shuffle có thể trùng,
báo effectivechangedfraction/coverage chứ không gọi mọi shuffle hiệu quả tự động.

Ràng buộc chung: fixture20c/2,5%/alpha.5,seed42/43,20round×1epoch,
privatebatch512/aux256,CEreal+CEsyntheticλ1,optimizer/cosineLR/BN/eval giống freshcontrol.
Không lên fullscale50round để cứu pilot. Lưu checkpoint/decoded packet/prediction mỗiclient,
confusion/10metric/perclass/actual steps/bytes/sourcehash, exact-resume và sampled inference.

## 10. Hai cổng và giới hạn kết luận

Finalr20, group/gp chính, từngseed phải qua; report old/group/head/gp đủ10metric.
Mỗi retained recipe có cổng riêng theo M hoặc J32, không chọn best scheme sau results rồi gộpclaim.

**Quality:** candidate_FL−max(full66_plain,full66_support_cùngmask)≥0,015.
**Compression:** candidate_FL−full66_support_cùngmask≥−0,005;
actual application bootstrap hai chiều/client≤0,65×control cùngmask/cùng fallback/count phases.
Hai cổng độc lập, không đổi ngưỡng sau số. Compression estimateFAIL không được gọi qualityFAIL.

Các guard cho cả claim: FL−own≥0,010, FL−shuffle≥0,010;
group/gp−local≥0,03; old/gp>local; benignF1≥local−0,03;
finite/checkpoint/payload/resume/sourceaudit PASS. Perclass positionMirroring/timeDelay báo riêng.
J32−M≥0,005 cảhai seed là proposed contribution gate để claim conditional masking có ích;
khóa trước launch, không thay utility gate chính. Báo interaction
`(candidate_J32−candidate_M)−(full66support_J32−full66support_M)` để tách lợi mask generic khỏigeometrycoupling.
Nhiều fallback làm contrast kém rõ,
không phải bằng chứng joint đầy đủ vô ích.

ArtifactPASS chưa chứng minh no-leak/DP/crypto; source audit kiểm luồng không export raw.
DAGSNet-only kết quả chưa thực nghiệm R2. U utility khôngdeadline chưa R1.
Chỉ sau quality/compression utility gatePASS mới đề nghị contact, không mở fullscale/tune tự động.

## 11. Bài học có thể dùng ngay và điểm còn phải chốt

1. Impute là phép nhiều-về-một. Identity đúng trước impute chưa đủ cho decoder dùng feature đã impute.
   Bằng chứng: toy hai world identical nominalprimitive/distinctderived, không phải đoán về dữ liệu.
2. Giữ derived bảo thủ có thể làm hai thiết kế collapse và mất lợi byte. Deduplicate đúngdimension;
   không đánh tráo nominal tênG51/G53 với carrier51/53 thật.
3. V4supportFAIL cảnh báo valid support chỉ là điều kiện cần. Conditional covariance phải có control và cost riêng.
4. Cohort20 không đảm bảo từngconditionalcell có≥5contributors. Rare-cell mean/Q phải chặn **trước gửi**.
5. Full joint atom table và joint32 categorical là hai mô hình thống kê khác nhau; không nhầm byte/claim.
6. Quantization intervalPASS với bound rộng chưa faithfulcenter. Không chọn rule bỏnearzero rows sau probe.
7. Giữ original feature bị thiếu branch không tự làm Gaussian sampler giữ manifold; G59 có vấn đề distance/bearing riêng.

Scientific đã chốt sau proof; numerical/code preflight còn phải thực hiện:

- dừng nominal51/53; kiểm G59 vàG61 quality; giữ carrier65% FAILanalytic;
- G59 inverse đúngextractor, supportinvalid báodiagnostic/nonfiniteFAIL;G61 giữbearings; không fullmanifoldclaim;
- classfullcov+13atomoverlay, noheadingrenormalize/distanceprojection; ridge1e−4 chung;
- N≥60nonfirst/≥47first chungcontrol/candidate; populationcell≥5xe,owncell chỉlocalN;
- pairedindependentnoisedraw shuffle/matchedcompute toànarm;
- kiểm actualwire precision/Cholesky/decodefiniteness/tolerance và source/byteledger trước push;
- schemaatoms+partialdecoder scope rõ; chưa có F1, R1, DP hoặccrypto evidence.

Không thao tác tay bắt buộc cho research/toy. Các quyết định trên ảnh hưởng phương pháp/cổng,
không xin quyền read-only hoặc tự launch thêm để lấy nhiều số.

## 12. Nguồn sơ cấp và evidence

- [vnx_features.py](../../../dataset/script/preprocessed/vnx_features.py): CAM-derived geometry245–288,
  session301–344, profile362–367, general nonfinite/impute369–395.
- [feature_schema.json](../../../dataset/centralized/feature_schema.json),
  [meta](../../knowledge/meta.json), [scaler](../../knowledge/scaler.json):66order, atom13, preprocessing/scales.
- [v4_support.py](../../mh/v4_support.py), [v4_schema.py](../../mh/v4_schema.py),
  [V4frozenmanifest](../../docs/research/pilot-v4-support.md): thực tế mask/supportcodec, không sửa source.
- [K5algebra source/probe](../../docs/research/v5-algebra-sources.md): bounded40960row/17identity,
  FP16 position-derived nearzero center error, không fullbranch/fidelityproof.
- [P1](../../docs/research/probes/pilot-p1.md), [V4auditedresults](../../docs/research/probes/pilot-v4.md):
  động cơ thống kê và failure của support-only adaptation.
- [Primarytoyscript](../../scripts/probe_k5_missing.py), [toyaggregate](fidelity-missing.json):
  scientific gate đã thực thi, không realrecord/modeltrain.

Không cần nguồn web cho các công thức/counterexample này; nguồn sở hữu là extractor local đã đọc và chạy.
