# Macro-first và dư địa pilot r20 — 11/11/2026

Owner: tạm bỏ các guard phụ, tập trung pooled **macro-F1**; hỏi pilot nào còn cơ sở nếu train dài hơn,
và hai hướng ưu tiên hoạt động/trao dữ liệu ra sao. Đây là review theo mục tiêu mới, không sửa verdict lịch sử.
P10/low256/benign/TD/absent chuyển thành số báo cáo cho hướng nghiên cứu này;
audit source/round/data/finite/CM vẫn cần. Giữ mốc tham chiếu gain+.002/hai seed hiện hành khi sàng lọc,
chưa có yêu cầu đổi ngưỡng macro hoặc bỏ đối chứng. Chưa launch/push kernel mới từ câu hỏi này.

## Hai pilot có cơ sở

| pilot | đối chứng đúng | Δmacro r20 s42 / s43 | Δmacro r15 s42 / s43 | macro của arm tăng r15→20 | đánh giá |
|---|---|---:|---:|---:|---|
| **Y1/Y2 prior-coordinate** | F2/F3 Core, full5343 | **+.003345 / +.002890** | +.003203 / +.002746 | +.002953 / +.002912 | ưu tiên1: gain tương đối còn giữ ở cuối pilot |
| **W2/W3 real-NCKD** | Q6/Q7 mixed-input bank, full5343 | **+.002461 / +.002229** | +.002787 / +.002455 | +.002197 / +.002429 | ưu tiên2: arm còn học, nhưng lợi thế với control đang co nhẹ |
| X6/X7 restricted real-NCKD + floor/compact | X2/X3 bank cùngfloor/compact | +.002224 / +.002267 | +.002039 / +.002228 | +.002228 / +.002513 | bổ sung lịch sử; không coi là CBB stream, không mở thêm gridlocal-class mask |

Tất cả trên **cosine50**, tại r20 LR=`.0006759557`, còn67,6%LRmax `.001`.
Có cơ sở train tiếp, **chưa bằng chứng r50 chắc giữ gain hoặc vượt CBB/APFed**.
Prior Core cũ bị loại TD−.02651/−.02744, absent−.01543/−.03023; p10 gần như giữ.
W2/W3 bị đánh đổi TD≈−.00933/−.00954 và absent xe nhỏ−.02958/−.01543.
Theo mục tiêu macro mới, các đánh đổi đó không tự veto; vẫn phải công khai.

Đã đọc đúng `methods/mobihet_prior.py`, `methods/mobihet_wstudy.py`, `scripts/y50_common.py`,
`scripts/w20_common.py`; prior r1–20 CM/control được tính lại và khớp đủ10metric trong audit lịch sử.
NCKD dùng sampledcurve được giữ + endpoint audit, không gọi đã replay privateweights.
[JSON/SHA/scope](../../../runs/macro-first-review-20261111/analysis.json),
[hình PNG](../../../runs/macro-first-review-20261111/pilot-curves.png),
[SVG](../../../runs/macro-first-review-20261111/pilot-curves.svg).
Tái tạo: `python scripts/analyse_macro_first.py`.

## Hai cơ chế làm gì; có gửi data thật không

**Prior-coordinate lúc train:** CEreal hiện đã có LA hướng globalpriorπg, còn synthetic cân bằng16class.
Y50 dùng `s=log(1/16)−logπg`: thêm s vào syntheticlogits, và đo realQ bằng `softmax(zreal+s)`;
realCE giữ nguyên, suy luận vẫnrawz. Student/teacher cùngcoordinate tránh tính prior hai lần.
Không phải PH của b11 chỉ cộng prior lúc inference. Cần đổi Qmeasurement và KD cùng nhau,
không chỉ dịchlogit mộtphía. [Code](../../../methods/mobihet_prior.py).

**Real-NCKD:** giữ CE nhãn đúng trên realrow local; thêm KL chỉ trên15class ngoài trueclass,
teacher là Qglobal[trueclass] đã gộp. Teacher/student bỏtrueclass, normalize phần còn lại;
loss nhân teacherwrongclassmass `(1−Q[y,y])`. Học cấu trúc nhầm lẫn toànmạng, CE vẫn phụ trách nhãn đúng.
Không phải mean-KD b16 hoặc hard+NCKD **synthetic** W4/W5.
W2/W3 lịch sử còn mixed-input bank1024synthetic/query trong cảarm vàcontrol; chuyển real-NCKD sang
CBB stream64synthetic là **adaptation mới cần kiểm**, không bê nguyên gain cũ.
[Code NCKD và reception](../../../methods/mobihet_wstudy.py).

Xét **giao thức client–server**, hai hướng không cần gửi từngrealrow/labelrow/model/gradient/BN.
Prior cần classcounts đã có trong bootstrap; NCKD có thể nhận cùngclassQ16×16.
Realrow chỉ được forward/loss tại client; bank W20 là inputsynthetic, không realdatapool như oracle.
Giữ schema xác suất/thống kê, dùng được với kiến trúc khác; experiment vẫnDAGSNetW1.
Adaptation lên CBB phải đo lại frame/byte, không claim byteint8 kế thừa nếu chưa audit.

**Không đồng nghĩa đã bảo đảm không lộ dữ liệu.** Core/KM hiện trao bootstrap moments theo client/class/cluster;
recount `evidence/research-data/train_counts.npy`:510ôclient×class chỉ1row tại442xe, trên37.361.273rowtrainFL.
Ở ô1row, trướcquantization `sum(z)=mean(z)=z` và covariance population không che được vector đó.
Sau codec và CDF, server có thể suyngược xấp xỉ feature ở các ônhỏ; không hứa exactrawrecord recovery toàn77feature.
Chưa triển khai secureaggregation/DP. Đây là rủi ro cósẵn của carrier, hai loss không giải quyết nó.
Q là xác suất gộp, không rawfeature, nhưng cũng không tự tạo privacyguarantee.
Secureaggregation có thể che contribution riêng; không gọi nó tự động thành DP hoặc bảo đảm chống mọi inference.
Owner trước đó hoãn xử lý privacy; câu hỏi này được trả lời kỹ thuật, chưa suy thành lệnh triển khai hoặc pause training.
[Phân tích đã có](../../../docs/research/improvement-proposals.md).

## Vì sao không kéo dài mọi run r20

- **B13–B16:** ngay cả bỏ tất cả guard, không arm nào đạt+.002 cả2seed. SymmetricKL có+.000834/+.001639;
  SAM+.000531/+.001196; MK âm2seed. Những run này kết thúc **cosine20**, LRr20=`1e-5`.
  Nối30round không biến trajectory đó thành protocolcosine50; so protocol50 cần lịch đãkhóa từr1.
- **B12:** floor64 +.001753/+.001482, lợi nhỏ hơnmốc+.002; gộpphiên16 +.002018/+.002318 trên hai control,
  nhưng là inferencewrapper, không distillation/server improvement, muốn so APFed phải áp cùngwrapper.
  Owner09/11 đã đóng cảhai; chỉ đưa vào bảng evidence theo mục tiêu mới, chưa tựlaunch lại.
- **Bank F50 là phảnchứng:** bank−Core seed42 từmacro+.00023 ởr20 thành−.00165 r50,
  seed43−.00013 thành−.00137. “Arm còn tăng” không đủ: control cũng tăng và có thể bắt kịp.
- Prior/NCKD đang dùng recipeCore/bank cũ, không KM16+lite6+Q8 hiện tại. Không cộng gain r20 vào CBB .9078,
  không so simval với APFedfulltest, không chọn best round hoặc coi hai seed là significance.

## Ưu tiên triển khai nếu tiếp tục thử theo mục tiêu này

1. **CBB + prior-coordinate**: cùngKM16/lite6/Q8, đổi synthetic/Qcoordinate đồngbộ, controlCBB mới.
2. **CBB + real-NCKD**: giữ64synthetic stream, dùng Q hiện có; tách khỏiprior, không khôi phụcbank chỉ để giốngW20.
3. Mỗi adaptation sim2seed → fullpilot20 **cosine50** → r50two nếu macro đạt luật khóa;
   các guard phụ report-only theo steering mới. Giữ10metric/byte/compute và audit đầy đủ.
   Chỉ resume pilot khi có fullstate+RNG hai rank; chưa kiểm khả năng resume các sessioncũ trong review này.

Quality ngoài fit vẫn là hướng mở, nhưng đắt hơn và chưa có utility; xếp sau hai tín hiệu macro đãđo.
Không có batchB17 hoặc source/method mới được launch từ tài liệu này.

**Owner sau review:** chỉ cần không gửi realrow, cho tiếp tục thử. [B17](../../../docs/research/screening-macro-b17.md) đã khóa C/P/NR×hai seed,
localgates trướcquota/push; trạng thái thật ởcontinue. Các câu “chưalaunch” ở trên mô tả thời điểm review, không currentjobstatus.

**Kết quả adaptation B17:** P/NRđềuFAILmacrohai seed, khôngpilot/tune/extend.
[Kết quả](../../../docs/research/screening-macro-b17.md). Evidencepilotcũ vẫn làkết quảcũ, không currentrecommendation launchlại hai adaptationnày.
