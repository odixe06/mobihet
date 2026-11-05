# V20 — CE zero-skip guard, 02/11/2026

V0`dinhan25/mh-v0-val-real-s42/1` và V1`minhtriethihi/mh-v1-val-real-s43/1` ERROR.
Output đãpull đủselected56file/run, source đối chiếu base+overlay; foldreceipt train-only/val đúng.
Log hai run xác nhận compile gate rank0/1PASS, rồi assert`info["skipped_frac"]==0` tại GuardedBasic.round.
Không roundCM/snapshot/resume đãcommit trong evidence; chưa có checkpoint tương thích để resume20round.
Giữ nguyên failedartifact`artifacts/v20/{slug}`, originalbatch/manifest và6kernel chưa báoERROR.

## Biết và chưa biết

P0/P1 CE gốc đãaudit có skippedfraction~1.7e-6–1.53e-5, mọiround; P2/P3augmentation có skip ởroundmuộn.
Đường train chung cho phép skip (bỏ optimizer/BNupdate của bước gradientnonfinite); auditorcommon chặn fraction>0.01.
GuardVzero-skip mới nghiêm hơn đường cũ. Không kết luận skipfractionV nhỏ từ P: giá trịV bị mất khi assert trướclog.
Core dùng fixedscale4096; không dynamicGradScaler hoặc tự động giảm scale, nên chưa gọi warm-up/overflowrootcause đãchứngminh.
Không đổi AMP/scale/optimizer/participation hoặc nới gateVchỉ để tìmPASS.

## Kiểm chứng kế tiếp đãkhóa

Hai probeCE seed42/43, mỗi probe fresh1epoch/full5343/originalfoldtrain, đúngbatch64/group4096/AMPfp16/W1.
Quan sát `local_epoch` sau khi nó trả về: applied/skipped từngxe, finiteweights/BN/loss từnggroup.
Mọi outputstat trả nguyên lại caller; không chạm RNG, loss, model update hoặc data. Không eval/teacherselection;
không exportprivateweights/BN, fullcheckpoints hoặc localrows. Metadataobserver trong artifact nghiên cứu riêng.
Đối chiếu applied+skipped với ceil(n_train/64) cho mọi xe, aggregatefraction với info toànmạng.
Nếu zero-applied hoặc finiteFAIL/fractioncao: điều tra số học/xe cụ thể trước retryutility.
Nếu finite/allclientapplied và skipnhỏ: kiểm sửa guard cho nhất quán đườngchung, vẫn báo đầyđủskip và giữ failuregốc.
Trước probe chưa chốt thay gate; quyết định sau audit ở mục dưới, không chọn theo F1/val.

Localobserverpreflight2seed×3round trên150train/23val: privateweights/BN/RNG/info bitexact với Basic,
2unitcase bắt nonzeroskip/zeroapplied và khôi phục hook khi exception. Không proofT4/fullnetwork.
Hai ownerdựphòng ngoài activeR/V/repeat, cap1.2h/probe +15%reserve, phase2.76h;
freshquota/serveridentity/access/ownedslug trước push. PrivateW&Binlinekey theo quyền đãcó.
Manifest`v20-numerics-launch.json` saufreeze; `runs/v-study/numerics-guard` giữreceipt/watch1200s/pull/audit.
Completion chỉ chứng minh hai numericalobservations đãthu/audit, không20roundutility hoặc guardfix.

## Hai probe T4 đã audit — 11:10/02Nov

| seed | applied | skipped / tổngstep | skip % | zero-applied xe | weights/BN/loss |
|---|---:|---:|---:|---:|---|
|42|513365|3/513368|0.000584376|0|finite|
|43|513363|5/513368|0.000973960|0|finite|

Sourcebase+14overlay/foldreceipt/5343client/ceil(n/64)/aggregatefraction khớp; compile2T4PASS.
Seed43 xe716 có193trainrow, applied3/skip1: mất25%step của xe này dù globalfraction rất nhỏ.
Không xác định vì sao gradientnonfinite hoặc gán F1harm; chỉ chứng minh stoptoànrun khi có bất kỳskip nào quá nghiêm
so đường training/audit chung. Không gọi tất cả skip là vôhại, warm-up, hoặc cam kết mọiround sau cũngnhỏ.
OriginalV0/V1fraction vẫnunknown: probe là sessionfresh có runtimevariation, không thay failedartifact.

## CE recovery khóa sau bằng chứng

Chỉ V0/V1 thiếu20round: fresh20round/cosine50/seed42–43/W1/full5343/originalfoldtrain→val.
Giữ nguyên core, data, HP, loss, optimizer, AMP, loss_scale4096 và maskedstep behavior. Không resume vì chưa có checkpoint tươngthích.
Guard mới áp dụng trần commonaudit`skipped_frac<=0.01`, allclientspositiveapplied và finiteweights/BN/loss;
đồng thuận2rank, lưu từngxe và countergroup trước khi fail. Trần lấy từauditcommon cósẵn, không fit theo số3/5 hoặc metric.
Không sửa guard/source của V2–V7 đangchạy, không sửa originalV0/V1manifest hay gọi bảncũPASS.
`methods/mobihet_vrecover.py` kế thừa Basic qua observer đãchứngminh không đổi training;
local2seed×3round privateweights/BN/RNG/info bitexact, state/load/strictW1 và diagnosticsRNGPASS.
14test guard/data +5testtelemetryaudit PASS; auditor recovery riêng kiểm CM→10metric/cleanval/support/source/20snapshot,
perclientapplied+skipped đúngceil(n/64), guardfinite và class/train diagnostics đủ20round.
Cap4.1h/run +15%reserve→9.43account-hour; freshquota/slot/access trên2owner numericalprobe đãterminal trước push.
Manifest`v20-ce-recovery-launch.json`; `runs/v-study/recovery-ce` và `artifacts/v20-recovery` riêng.
Sau full20audit mới ghép hai recovery vào diagnosticV theo logicalparent, giữ mọiattempt/error gốc.
