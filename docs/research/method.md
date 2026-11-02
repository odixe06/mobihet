# Cơ chế đang chẩn đoán

Mục tiêu: từng xe học tốt trên own-test, nhất là ít dữ liệu/non-IID; giữ schema trao đổi chung cho model khác nhau.
Đây là họ đối chứng, chưa chọn phương pháp tổng. Cấu hình chung tại [protocol](protocol.md).

## Luồng thông tin

1. Bootstrap thống kê gộp theo lớp để dựng Gaussian copula/prior; không truyền mẫu local.
2. Client học model riêng bằng CE trên dữ liệu thật và64synthetic/bước, sinh tại client.
3. Client gửi mean probability theo true class (16×16); server gộp theo n_kc hoặc sqrt(n_kc).
4. Server gửi Q chung cho mọi xe. Q giữ cố định trong round, cập nhật cho round kế tiếp.
5. Client distill từ Q trên synthetic, tùy đối chứng thêm realKD có gate. CEreal có logit adjustment;
   KD dùng logit chưa adjustment. Không tự gọi hai hệ xác suất khác nhau là bug.

## Các khâu M kiểm riêng

- Upload: xác suất online so với full real pass sau epoch, eval BN/dropoutoff.
- Server: n_kc so sqrt(n_kc), cùng packet/schema.
- Client: augmentation, softtarget, realgate, fallback, uniform/local support mixture; loss outer coefficient1.
  Localρ=64/(64+n_kc), lớp ít mẫu nhận Q nhiều hơn. Đây là giả thuyết chẩn đoán, không bảo đảm có ích.
- Chi tiết thiết kế đã khóa: [screening-m20](../../archive/docs/research/screening-m20.md); kết quả ở [results-m20](../../archive/docs/research/results-m20.md).
- N giữ nguyên source/loss/protocol, hoàn tất factorial realKDoff/gate × syntheticsoft/fallback:
  [screening-n20](../../archive/docs/research/screening-n20.md), [results-n20](../../archive/docs/research/results-n20.md). Reference FD/KD bên dưới áp dụng cùng thành phần trong N0–N7.
- [O20](../../archive/docs/research/screening-o20.md) giữ code client/codec N, thay downlink bằng lịch Q train-only N đã audit để chẩn đoán.
  Uplink live vẫn collect; Q_live không dùng feedback. [Kết quả](../../archive/docs/research/results-o20.md) đủ8×20; replay chưa là phương pháp liveFL hoặc bằng chứng giảm byte.

## Chi phí và giới hạn

Q codec meanfp16:544B/chiều,1088B/xe/round; copula bootstrap3311744B/xe hai chiều.
20round:3333504B/xe; augmentation3311744B và0learnedB/round. Bootstrap vẫn chi phối chi phí.
Số byte chưa gồm transport/ACK/secureaggregation end-to-end. Aggregate statistics chưa có DP guarantee.
Client model khác kiến trúc và contactdrop chưa được kiểm; batch hiện cùng DAGSNetW1.
Snapshot nhỏ không chứa fullprivatebank, không dùng làm checkpoint resume.

## Reference gắn với phần thực sự đã thử

Chỉ reference cơ chế đã có code và run; không coi trích dẫn là bằng chứng utility trên VeReMi.

| nguồn gốc | thành phần đã chạy | khác biệt/phạm vi |
|---|---|---|
| [Jeong et al., FD, 2018; §2/Algorithm1](https://arxiv.org/html/1811.11479v2#S2) | M2–M15 gửi mean outputsoftmax theo trueclass, nhận teacherrow cùng lớp | Bài gốc gọi output đãsoftmax là logitvector, teacher loại chính client và trung bình peer; M dùng Q chung gồm self, weighted n/sqrt(n), fp16, LA và syntheticcopula. Là adaptation, chưa reproduce FD gốc. |
| [Hinton et al., KD, 2015; §2](https://arxiv.org/html/1503.02531v1#S2) | CEreal + KL từ softtarget trong M2–M15; augmentation M0/M1 là noKDcontrol | M dùng T=1 và teacher theo lớp; bài gốc distill phân phối theo từng input. KL có cùng studentgradient với softCE khi target cố định; không tái tạo toàn bộ teacher/model của bài. |

Copula bootstrap/gate/fallback/supportmixture/server sqrt(n) là thiết kế đang kiểm của dự án; kết quả ở M20 và N20.
FAug cùng bài FD yêu cầu upload seed samples để train GAN; **không áp dụng** trong phương pháp hiện tại.
DAGSNet thực sự dùng cho mọi M, nguồn và deviation tại [đặc tả kiến trúc](../../knowledge/dagsnet.md).
Bài báo/ý tưởng chưa triển khai không đưa vào bảng reference phương pháp đã thử; đề xuất giữ ở [diagnostic-next](../../archive/docs/research/diagnostic-next.md).

P20 gốc31run đủ20round/P21r0; [kết quả gốc](results-p20.md) và [transfer từngxe](../../archive/docs/research/p20-client-transfer.md).
[Recovery C10seed43](../../archive/docs/research/results-p21-recovery.md) đã audit20round: hybrid hơn proxythuần pooled cảhai seed nhưng vẫn dưới streamingclassQ.
[Đề xuất pipeline](../../archive/docs/research/proposed-method-p20.md) giữ streamingclassQ, kiểm thêm
support/inputtargets/serverreliability/reception từngkhâu; chưa triển khai hoặc nhận method tổng.
[Thiết kế](../../archive/docs/research/screening-p20.md) gắn reference đúng thành phần:

| nguồn | thành phần P đã chạy và audit | deviation |
|---|---|---|
| [FedMD, Li & Wang2019, §2.2](https://arxiv.org/html/1910.03581v1#S2.SS2) | C7 trao đổi output trên sharedsyntheticbank thay classmean | copula refresh/r thay publicdata, không publicpretraining, classcountweights, probabilityfp16 thay logitmean; không reproduce FedMD |
| [FedDF, Lin et al.2020, §3](https://arxiv.org/html/2006.07242v3#S3) | C8/C9/C10/C14 serverproxy học aggregateoutput | privateweights không upload, seedcopula thay public/unlabeledpool, warmstartserver/BNsynthetic; C10hybridlabel là giả thuyết riêng, không công thức paper |

Clientmodel mọi P giữ DAGSNetW1 như repo benchmark. Serverproxy riêng không ràng buộc kiến trúc client về mặt schema;
batch chưa kiểm modeltrộn. ClassQ và proxy là các đối chứng song song, chưa chọn teacher cuối.

[Q20](screening-q20.md) tiếp nối C2/C7: giữ onlineclassQ, mixed32stream/32bank, inputensemble count/sqrt
và realKD/margin/strength controls. [Kết quả Q](results-q20.md) đủ16×20: inputtarget nâng weakcohort,
margin nâng pooled/mean/p10 nhưng vẫn hại absent/timeDelay; sqrtinputweight chưa giúp. Chưa method tổng.
Không serverproxyfit hoặc modeldownlink Q, không gán FedDFproxyfit cho Q; inputconsensus vẫn là adaptation FedMD ở trên.

[R20/V20](results-r20-v20.md) tách độ ồn phiên, tiếp nhận realKD với teacher cố định và validation sạch: realKD bị loại khỏi
mặc định. Hướng hiện hành và byte theo tầng: [proposal](proposal.md); ô đang chạy: [W20](screening-w20.md)
(non-target KD theo [Decoupled KD, Zhao et al. 2022](https://arxiv.org/abs/2203.08679), step floor synthetic, bootstrap nén).
