# V20 — validation tách biệt, chẩn đoán generalization

Owner02/11 ủy quyền nhánh validation sau [Q20](results-q20.md)/[R20](screening-r20.md).
**Nhánh chẩn đoán riêng**, không thay headline train+val→own-test của protocol và không chọn HP bằng officialtest.

## Câu hỏi và cell khóa trước launch

| cell | seed42/43 | học | đối chiếu |
|---|---|---|---|
| CE+LA | V0/V1 | realCE, prior train-only | control: gap có tồn tại khi chưa augmentation/KD? |
| augmentation | V2/V3 | realCE +64stream copulahard | augmentation−CE: lợi synthetic có tổng quát sang val chưafit? |
| mixedinput | V4/V5 | realCE +32streamclassQ/32bankinputKD, noReal | KD−augmentation là tổng hiệu ứng support+target, **không isolate targetresolution** |
| margin | V6/V7 | mixedinput +realKD marginclassQ | margin−mixedinput: tradeoff timeDelay/absent còn trên validation chưafit? |

Fresh20round/cosine50/full5343/DAGSNetW1/395408params, seed42/43/batch64/1epoch/resetAdamW/clip1/AMPfp16.
Mọi cell cùng foldtrain32.685.166rows → foldval4.676.107rows (12.5% cuối phần train/xe).
Bootstrap/prior/copula chỉ fit foldtrain, giữ scaler gốc được fitfoldtrain; không chuẩn hóa lại.
Cache chỉ đọc originaltrainpart, giữ foldcode; Env dùng sizesfoldtrain để phân rank/nhóm, evalval thay officialtest.
Không externaltest diagnostic. Không teacher/qualityweight theo test hoặc val, chưa có adaptive selection.
Teacher là live train-only aggregate như Q, lag1; không Q frozen của R (vì nguồn đó đãfitval).
Không serverproxy/privateweights/BN hoặc rawlocalrows trong FLmessages; chưaDP/SecAgg/mixedarchitectureproof.

## Bằng chứng và dấu hiệu bác bỏ

P/Q có matchedtrain→testgap timeDelay cảxe nhiều mẫu. Nếu gap train→val xuất hiện cảCE, không quy toàn bộ teacherKD.
Nếu chỉKD/margin làm gap lớn/absent giảm trên val so cùngsplitcontrol, có tradeoff reception/livefeedback cần tách sâu.
Valgap vẫn không phân rã overfit, temporalshift, labelnoise hoặc featureoverlap; cần tiếp feature/time analysis.
V không isolated fixedteacher replay; R mới giữ teachertrajectory. Không cộng gain R/V hoặc so delta hai split như causal effect.
Không dùng V chọn policy từngxe rồi báo officialtest như freshconfirmatory; V vẫn exploratorydiagnostic.
Không gọi endpoint20 là plateau: báo slope15→20, class/support/low256/rare/absent cùng mean/p10/std.

## Gates và tài nguyên

Split audit local410trainpartSHA, tr+valcount khớp đầy đủ, tất cảxe train trướcval; scalerfitfoldtrain.
Preflight localsm86 2seed×4cell×3round/150train23val: exacttraincounts, valCMsupport, finite/skip0,
mixedinput/margin r1bitexact, diagnostics không đổi state/RNG, state/load và strictW1rebuildPASS.
Unit test cache không hề có officialtestarray vẫn route đúng train/val; scalerfittrain+val phảifail.
Local không full5343/T4/utility/fullresumeproof. Sourcehash trước/cuối không đổi; khung train shared giữ nguyên.

Chỉ dùng8owner ngoài Ractive. CE cap4.1h/run, các cell có synthetic cap4.6h/run;
35.8session-hour +15%reserve→41.17h; từng owner đủcap×1.15 và slot/access/identity trước push.
P CEfulltrain+val max3.379h, augmax4.454h; Q mixedinputmax4.385h, marginmax4.079h.
V trainrows giảm12.5%, eval4.676M so9.348M; đây **dự toán**, không đo speedup.
Notebook reserve0.4h/driver worst-round guard giữ nguyên. Budgetstop trước20 là incomplete, không báoPASS đủround.
Nếu quota không đủ wholephase, dừng trướcpush; không hạround/reserve hoặc bỏseed. Nối task không resetquota.

CE bootstrap128B/xe,0roundpayload; augmentation3311744B/xe,0roundpayload;
mixedinput/margin bootstrap3311744B +2×33344B/r =4645504B/xe/20r theo codec Q.
Báo10metric từnground trên toànval, trainCM tại1/2/5/10/15/20, cohort theo foldtraincount, paireddelta/magnitude/support,
timeDelay/other15, absent/rare, fullbootstrap/two-waybytes/time; không gọitime2T4 là latency xe.
No rawpredictionreplay/fullprivatebank snapshot/fullcontactproof. Audit bounded aggregate/source/CM/guard.
Sourcebase+overlay/notebookSHA, foldreceipt từngrank, traincounts/valsupport/teacherlag/finite snapshots khóa trước nhận utility.
codex-long-jobs poll1200s, collect cảERROR; failing stage recover riêng, giữ healthy kernels đượcsupervise.

V0/V1ERROR tại guardzero-skip trước round1commit, đãpull/source/foldreceipt audit; chưa utilityFAIL của CE.
[Sự cố/probe/recovery](v20-skip-incident.md): hai probeT4audit xác nhận3/5skip trên513368step, mọixeapplied, weights/BN/lossfinite.
CErecovery riêng sửa guard về common≤1% +positiveapplied từngxe/finite, core/HPkhôngđổi; fresh20round trên ownerprobe đãterminal.
Giữ originalguard/source/batch và6kernel còn lại; originalfailures vẫnFAIL, không gọi probe hoặc guardfix là full20utilityPASS.

## References theo thành phần

CE/LA/copula/classQ kế thừa [method](../../../docs/research/method.md); sharedinputconsensus thích nghi từ
[FedMD](https://arxiv.org/abs/1910.03581). Margin/validationbranch là chẩn đoán dự án;
không triển khai FedDFproxyfit, FedAUXqualityweight hoặc PCGrad ở V.
