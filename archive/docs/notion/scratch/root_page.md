Here is the result of "fetch" for the Page with URL https://app.notion.com/p/409730cee68d81f19259d14bea45ea61 as of 2026-11-02T10:13:00.477Z:
<page url="https://app.notion.com/p/409730cee68d81f19259d14bea45ea61">
<ancestor-path>
<parent-page url="https://app.notion.com/p/319730cee68d80c3ace9eaaf52bc2cb4" title="NCKH"/>
<ancestor-2-data-source url="collection://2fc730ce-e68d-80e9-ae46-000b14541d7a" name="Tự học"/>
<ancestor-3-database url="https://app.notion.com/p/2fc730cee68d80b0afa5cb740b8f1540" title=""/>
</ancestor-path>
<properties>
{"title":"50 round with new dataset 5000 client"}
</properties>
<iconMetadata>null</iconMetadata>
<content>
<b>VeReMi NextGen v4 — 5.343 client</b>, 77 đầu vào, 16 lớp. Train 37.361.273 dòng (train+val); test 9.347.679 dòng, theo timeline từng xe.

DAGSNet W=1, seed 42; FL 50 round × 1 epoch local, batch 64, mọi xe tham gia theo protocol. Đây là bản dựng chung trên VeReMi, không phải kết quả trên dataset gốc của từng bài báo.

Chỉ hiển thị các run hoàn tất. Bấm tên phương pháp trong bảng để mở page con chứa đầy đủ lịch sử 50 round/epoch, các view được đánh giá và PDF bài báo.

## Bảng tổng — điểm cuối

<table header-row="true">
<colgroup>
<col />
<col width="67" />
<col width="63" />
<col />
<col />
<col />
<col />
<col />
<col />
<col />
<col />
<col />
<col />
<col />
<col />
</colgroup>
<tr>
<td>
Phương pháp
</td>
<td>
View
</td>
<td>
Round cuối
</td>
<td>
accuracy
</td>
<td>
precision\_macro
</td>
<td>
precision\_micro
</td>
<td>
precision\_weighted
</td>
<td>
recall\_macro
</td>
<td>
recall\_micro
</td>
<td>
recall\_weighted
</td>
<td>
f1\_macro
</td>
<td>
f1\_micro
</td>
<td>
f1\_weighted
</td>
<td>
Client F1 mean
</td>
<td>
Client F1 p10
</td>
</tr>
<tr>
<td>
<mention url="https://app.notion.com/p/409730cee68d8189b63ec73d9405a1cb">Centralized DAGSNet</mention>
</td>
<td>
global
</td>
<td>
ep50
</td>
<td>
0.957975
</td>
<td>
0.968431
</td>
<td>
0.957975
</td>
<td>
0.956038
</td>
<td>
0.953124
</td>
<td>
0.957975
</td>
<td>
0.957975
</td>
<td>
0.959267
</td>
<td>
0.957975
</td>
<td>
0.955551
</td>
<td>
—
</td>
<td>
—
</td>
</tr>
<tr>
<td>
<mention url="https://app.notion.com/p/409730cee68d8195b142c34df4a02579">FedAvg</mention>
</td>
<td>
global
</td>
<td>
50
</td>
<td>
0.922828
</td>
<td>
0.935397
</td>
<td>
0.922828
</td>
<td>
0.914712
</td>
<td>
0.895710
</td>
<td>
0.922828
</td>
<td>
0.922828
</td>
<td>
0.904240
</td>
<td>
0.922828
</td>
<td>
0.908533
</td>
<td>
0.866743
</td>
<td>
0.753971
</td>
</tr>
<tr>
<td>
<mention url="https://app.notion.com/p/409730cee68d8165afaece45a2f2d8ff">FedProx</mention>
</td>
<td>
global
</td>
<td>
50
</td>
<td>
0.921429
</td>
<td>
0.930698
</td>
<td>
0.921429
</td>
<td>
0.911186
</td>
<td>
0.894065
</td>
<td>
0.921429
</td>
<td>
0.921429
</td>
<td>
0.899792
</td>
<td>
0.921429
</td>
<td>
0.904940
</td>
<td>
0.865762
</td>
<td>
0.752392
</td>
</tr>
<tr>
<td>
<mention url="https://app.notion.com/p/409730cee68d81eaaaebf961e6ee8a71">SCAFFOLD</mention>
</td>
<td>
global
</td>
<td>
50
</td>
<td>
0.922531
</td>
<td>
0.901121
</td>
<td>
0.922531
</td>
<td>
0.884220
</td>
<td>
0.885855
</td>
<td>
0.922531
</td>
<td>
0.922531
</td>
<td>
0.891069
</td>
<td>
0.922531
</td>
<td>
0.900289
</td>
<td>
0.874401
</td>
<td>
0.773810
</td>
</tr>
<tr>
<td>
<mention url="https://app.notion.com/p/409730cee68d811d9da1de38dae9fc6d">AFPHA</mention>
</td>
<td>
global
</td>
<td>
50
</td>
<td>
0.921095
</td>
<td>
0.930021
</td>
<td>
0.921095
</td>
<td>
0.910649
</td>
<td>
0.893181
</td>
<td>
0.921095
</td>
<td>
0.921095
</td>
<td>
0.899023
</td>
<td>
0.921095
</td>
<td>
0.904529
</td>
<td>
0.865150
</td>
<td>
0.752286
</td>
</tr>
<tr>
<td>
<mention url="https://app.notion.com/p/409730cee68d81998959db030f875664">FD-IDS</mention>
</td>
<td>
global
</td>
<td>
50
</td>
<td>
0.921977
</td>
<td>
0.937296
</td>
<td>
0.921977
</td>
<td>
0.915501
</td>
<td>
0.891403
</td>
<td>
0.921977
</td>
<td>
0.921977
</td>
<td>
0.899241
</td>
<td>
0.921977
</td>
<td>
0.905426
</td>
<td>
0.866438
</td>
<td>
0.755232
</td>
</tr>
<tr>
<td>
<mention url="https://app.notion.com/p/409730cee68d811cbcbefc0d04ac6bb0">APFed</mention>
</td>
<td>
client
</td>
<td>
50
</td>
<td>
0.927752
</td>
<td>
0.937669
</td>
<td>
0.927752
</td>
<td>
0.921673
</td>
<td>
0.903184
</td>
<td>
0.927752
</td>
<td>
0.927752
</td>
<td>
0.911900
</td>
<td>
0.927752
</td>
<td>
0.916856
</td>
<td>
0.866284
</td>
<td>
0.753254
</td>
</tr>
<tr>
<td>
<mention url="https://app.notion.com/p/409730cee68d817888b5ea2ad68e5461">LwFedNIDS</mention>
</td>
<td>
global
</td>
<td>
50
</td>
<td>
0.916238
</td>
<td>
0.932028
</td>
<td>
0.916238
</td>
<td>
0.911105
</td>
<td>
0.883039
</td>
<td>
0.916238
</td>
<td>
0.916238
</td>
<td>
0.890402
</td>
<td>
0.916238
</td>
<td>
0.898587
</td>
<td>
0.847640
</td>
<td>
0.729127
</td>
</tr>
<tr>
<td>
<mention url="https://app.notion.com/p/409730cee68d81a28505d8b134d9824f">V2X-HtFL</mention>
</td>
<td>
client
</td>
<td>
50
</td>
<td>
0.897366
</td>
<td>
0.886461
</td>
<td>
0.897366
</td>
<td>
0.890973
</td>
<td>
0.855393
</td>
<td>
0.897366
</td>
<td>
0.897366
</td>
<td>
0.866140
</td>
<td>
0.897366
</td>
<td>
0.890667
</td>
<td>
0.764178
</td>
<td>
0.551908
</td>
</tr>
<tr>
<td>
<mention url="https://app.notion.com/p/409730cee68d8164be84ed3105f3c795">CANPerFL</mention>
</td>
<td>
client
</td>
<td>
50+10 FT
</td>
<td>
0.923271
</td>
<td>
0.928065
</td>
<td>
0.923271
</td>
<td>
0.917080
</td>
<td>
0.898640
</td>
<td>
0.923271
</td>
<td>
0.923271
</td>
<td>
0.908773
</td>
<td>
0.923271
</td>
<td>
0.915945
</td>
<td>
0.877907
</td>
<td>
0.768399
</td>
</tr>
<tr>
<td>
<mention url="https://app.notion.com/p/409730cee68d81cabe4bf47b16eb4b47">pFedES</mention>
</td>
<td>
client
</td>
<td>
50
</td>
<td>
0.886407
</td>
<td>
0.869161
</td>
<td>
0.886407
</td>
<td>
0.879986
</td>
<td>
0.840485
</td>
<td>
0.886407
</td>
<td>
0.886407
</td>
<td>
0.851079
</td>
<td>
0.886407
</td>
<td>
0.880252
</td>
<td>
0.744913
</td>
<td>
0.526380
</td>
</tr>
<tr>
<td>
<mention url="https://app.notion.com/p/409730cee68d81859f80f97502914260">FedTGP</mention>
</td>
<td>
client\_proto
</td>
<td>
50
</td>
<td>
0.902049
</td>
<td>
0.892430
</td>
<td>
0.902049
</td>
<td>
0.894466
</td>
<td>
0.860958
</td>
<td>
0.902049
</td>
<td>
0.902049
</td>
<td>
0.871860
</td>
<td>
0.902049
</td>
<td>
0.894656
</td>
<td>
0.779431
</td>
<td>
0.571160
</td>
</tr>
<tr>
<td>
<mention url="https://app.notion.com/p/409730cee68d81b5bdfcc2b6d29cbc2a">PerFed-SKD</mention>
</td>
<td>
client
</td>
<td>
50
</td>
<td>
0.899367
</td>
<td>
0.890316
</td>
<td>
0.899367
</td>
<td>
0.893091
</td>
<td>
0.859805
</td>
<td>
0.899367
</td>
<td>
0.899367
</td>
<td>
0.870976
</td>
<td>
0.899367
</td>
<td>
0.892788
</td>
<td>
0.813308
</td>
<td>
0.657291
</td>
</tr>
<tr>
<td>
<mention url="https://app.notion.com/p/409730cee68d81de89cef37e2a5f5b31">FedProto</mention>
</td>
<td>
client\_proto
</td>
<td>
50
</td>
<td>
0.897805
</td>
<td>
0.877790
</td>
<td>
0.897805
</td>
<td>
0.892327
</td>
<td>
0.865114
</td>
<td>
0.897805
</td>
<td>
0.897805
</td>
<td>
0.869797
</td>
<td>
0.897805
</td>
<td>
0.893212
</td>
<td>
0.787324
</td>
<td>
0.591068
</td>
</tr>
<tr>
<td>
<mention url="https://app.notion.com/p/409730cee68d81ec955ffa94fae0f171">NILM-FL</mention>
</td>
<td>
client
</td>
<td>
50
</td>
<td>
0.915583
</td>
<td>
0.929072
</td>
<td>
0.915583
</td>
<td>
0.914630
</td>
<td>
0.871662
</td>
<td>
0.915583
</td>
<td>
0.915583
</td>
<td>
0.884596
</td>
<td>
0.915583
</td>
<td>
0.902018
</td>
<td>
0.785394
</td>
<td>
0.577689
</td>
</tr>
<tr>
<td>
<mention url="https://app.notion.com/p/409730cee68d8143aec3c852e56a6677">TinyProto</mention>
</td>
<td>
client\_proto
</td>
<td>
50
</td>
<td>
0.886593
</td>
<td>
0.870208
</td>
<td>
0.886593
</td>
<td>
0.875921
</td>
<td>
0.845711
</td>
<td>
0.886593
</td>
<td>
0.886593
</td>
<td>
0.853375
</td>
<td>
0.886593
</td>
<td>
0.877354
</td>
<td>
0.760675
</td>
<td>
0.552137
</td>
</tr>
<tr>
<td>
<mention url="https://app.notion.com/p/40a730cee68d81fba3fcfb532fb66a8d">MobiHet v4 — 50 round</mention>
</td>
<td>
private
</td>
<td>
50
</td>
<td>
0.921213
</td>
<td>
0.944686
</td>
<td>
0.921213
</td>
<td>
0.922069
</td>
<td>
0.887272
</td>
<td>
0.921213
</td>
<td>
0.921213
</td>
<td>
0.895298
</td>
<td>
0.921213
</td>
<td>
0.903596
</td>
<td>
0.846776
</td>
<td>
0.716339
</td>
</tr>
</table>

accuracy = precision\_micro = recall\_micro = recall\_weighted = f1\_micro trong phân loại single-label; vẫn giữ đủ 10 metric. Client F1 là trung bình/p10 macro-F1 theo xe trên lớp có trong test riêng của xe.

## Cách đọc và giới hạn

- MobiHet v4: đủ50round, session1 r1–25 + session2 r26–50; đã pull và audit PASS\_BOUNDED từ50CM×2view. Bảng chi tiết có đủ10metric mỗi round, source37file khớp manifest, parent25CM khớp byte. Chưa replay inference từ full private bank/predictions. MobiHet là phương pháp đề xuất, chưa đính kèm PDF; RelKD r10 không nằm trong kết quả final.
- Global: một model chấm toàn test; client/private: model mỗi xe chấm test xe đó; client\_proto: luật prototype của phương pháp.
- Centralized: <b>50 epoch</b>, batch 4.096; không gọi đây là 50 round FL.
- CANPerFL: <b>50 round FedAvg pretrain + 10 epoch fine-tune local</b>. Page con có cả bảng 50 round pretrain và 10 epoch fine-tune. Headline là kết quả sau fine-tune, không giả thành một run 50 round fine-tune.
- APFed: metric r50 đủ; kernel ERROR lúc ghi state cuối vì RAM, bank private r50 không lưu được.
- SCAFFOLD: train\_loss NaN r15–16, sau đó hồi phục; không coi run này ổn định suốt quá trình.
- NILM-FL: nối phiên thành công: session 1 (r1–36) + session 2 (fl2-nilm-r2s2, r37–50); parent preflight r1–36 byte-identical.
- TinyProto: μ=1,5e-4 theo cấu hình tác giả (deviation so với VeReMi); đã khắc phục bùng nổ scale của r3; prototype F1 đạt 0,8534.
- Metric tính lại từ confusion matrix đã pull, đối chiếu đủ 5.343 xe và support nhãn của toàn test. Không lưu prediction từng dòng và thiếu một số bank model, nên chưa replay được toàn bộ suy luận.

Dataset có benign/timeDelay khó phân biệt, shortcut theo world và Sybil lệch prior theo thời gian. Một seed; headline là điểm cuối, không chọn round tốt nhất trên test.

[Protocol và cấu hình chung](https://github.com/odixe06/fl_dagsnet_veremi_v2/blob/main/docs/research/protocol.md).

## Các page chi tiết

<page url="https://app.notion.com/p/409730cee68d8189b63ec73d9405a1cb">
Centralized DAGSNet
</page>

<page url="https://app.notion.com/p/409730cee68d8195b142c34df4a02579">
FedAvg
</page>

<page url="https://app.notion.com/p/409730cee68d8165afaece45a2f2d8ff">
FedProx
</page>

<page url="https://app.notion.com/p/409730cee68d81eaaaebf961e6ee8a71">
SCAFFOLD
</page>

<page url="https://app.notion.com/p/409730cee68d811d9da1de38dae9fc6d">
AFPHA
</page>

<page url="https://app.notion.com/p/409730cee68d81998959db030f875664">
FD-IDS
</page>

<page url="https://app.notion.com/p/409730cee68d811cbcbefc0d04ac6bb0">
APFed
</page>

<page url="https://app.notion.com/p/409730cee68d817888b5ea2ad68e5461">
LwFedNIDS
</page>

<page url="https://app.notion.com/p/409730cee68d81a28505d8b134d9824f">
V2X-HtFL
</page>

<page url="https://app.notion.com/p/409730cee68d8164be84ed3105f3c795">
CANPerFL
</page>

<page url="https://app.notion.com/p/409730cee68d81cabe4bf47b16eb4b47">
pFedES
</page>

<page url="https://app.notion.com/p/409730cee68d81859f80f97502914260">
FedTGP
</page>

<page url="https://app.notion.com/p/409730cee68d81b5bdfcc2b6d29cbc2a">
PerFed-SKD
</page>

<page url="https://app.notion.com/p/409730cee68d81de89cef37e2a5f5b31">
FedProto
</page>

<page url="https://app.notion.com/p/409730cee68d81ec955ffa94fae0f171">
NILM-FL
</page>

<page url="https://app.notion.com/p/409730cee68d8143aec3c852e56a6677">
TinyProto
</page>

<page url="https://app.notion.com/p/40a730cee68d81fba3fcfb532fb66a8d">
MobiHet v4 — 50 round
</page>
</content>
</page>