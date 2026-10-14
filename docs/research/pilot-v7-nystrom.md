# V7 — Nyström covariance: hợp đồng pilot

Khóa trước push,2026-10-12. Chủ dự án đãchốtrank4/8,explicitWupper,
FP32uplink/FP16factor-downlink,cácđối chứng vàcổngdesign§7.
[Thiết kế](../../proposals/covariance-nystrom/design.md).

10arm/seed42/43:diag/full/exactlr4/exactlr8 +Nyström4/8×FL/own/sign-shuffle.
20clientDAGSNet,fixturepilot2,5%,20round×1epoch,batch512,aux256,λCEsynthetic1;
AdamWwd0reset,cosine1e−3→1e−5,clip1,AMPfp16/lossfp32 nhưdriverV2.
Real/syntheticforward riêng nhưV2;BNpolicy giốngnhau trongmọicontrolS1.
Evalold/group r5/10/15/20,privatehead/gp;headlinegroup/gp r20.
V2local đãaudit làmcommonguard; freshGaussian controls đúngsource/runtime/codec.

## Wire và numeric choices

StatsFP64local → FP32serialize → FP64cohortsum; countsuint64;
**16contributorbitsFP32 gửi thật**, thêm64B/uplink so mathprototype.
Chỉreleasepopulationclass khi≥5contributors;cohort20,trustedemulator,chưacrypto/DP.
Full/exactlr dùngsum/upperQ;Nyström dùngsum/diagQ/C/Wupper;diag dùngsum/diagQ.
Fullridge1e−4I; eigenrank residualfloor1e−4;
NyströmregularizedW eigenfloor1e−4 +diagresidualfloor1e−4,khôngoversampling.
FactorsFP16serialize/decode thật,khôngshadowhigherprecision trongsampling.
Factorquantizationcovariance-relativeerror≤1%;finiteFP32/fp16synthetic bắtbuộc.
Khôngtunefloor/rank/Ωseed hoặcđổihead/round đểcứu.

PublicΩ66×8 FP64 đãfrozenhash trong `mh/v7_basis.py`,rank4 lấy4cộtđầu.
**Coldstart gửiΩ thật** dướipacketkind3,4.272B/xe,không giảQRcrossdevice miễnphí.
Header32B +context16B +CRC trongheader; contextbindarm/seed/fixture/Ω/numericrule.
Prior200B/xe riêng. Classstats sumcohort vàfactorpacket nằmresumestate;
khôngcần phátlạistaticfactor mỗiround. Source/architecture/scaler đãpreinstall chung;
ACK/frame/securitysetup/retry/deadline chưa nằmapplicationledger,phảiđoởcontactstage.

| arm | up B/xe | down B/xe | total+prior B/xe |
|---|---:|---:|---:|
| diag | 8.688 | 4.272 | 13.160 |
| full | 145.968 | 72.912 | 219.080 |
| exactlr4 | 145.968 | 12.720 | 158.888 |
| exactlr8 | 145.968 | 21.168 | 167.336 |
| Nyström4FL/shuffle | 26.224 | 16.992 | 43.416 |
| Nyström8FL/shuffle | 44.784 | 25.440 | 70.424 |

Ownstatskhôngexport;localFP32/FP16roundtrip giữprecision;commonpaidprior/Ω vẫnnhận.
Absentownclass:mean0/identitycovariance,declaredfallback;khôngpopulationmoments.
Signshuffle đổiU→diag(public±1)U,mean/sd/diagonalvariance/eigenvalues giữ;
khôngclaim xóa mọicovknowledge. Stats/tables/bootstrapstatic,version0.

## Cổng từngrank, cảhai seed

- FL−matchedexactlr≥−0,010.
- FL−own≥0,010;FL−signshuffle≥0,010.
- groupgp−local≥0,030;oldgp>local;benign≥local−0,030.
- Actualtwo-waybootstrap gồmprior/Ω≤0,65×freshfullcontrol.
- Source/codec/cohort/finite/weights/state/pred→10metrics/resume/inferencePASS.

Reportfull/diag/exactlr/perclass/positionMirroring/timeDelay vàactualcost;
khôngdùngnumericPSD hoặcbytePASS thayquality. StageU chưaR1/R2.
ChỉqualityPASS mới mởcontact vớimanifest/control riêng.

## Vận hành

Local512uniformrow/clientmath,1024row/client×3roundsmoke;
4representatives exactresume,actualartifactstrict/sampledinference.
Kaggle2×T4/pinnedTorch2.10+cu128;W&Bonline/key trướcdata/model/train,
loground/loss/realCE/auxCE/lr/applied/skipped/time/10metrics,giámsátr1–5.
Worker1,6h;parentdừng/finalize trước3,8h,khôngnhậnarmmới sau2,7h.
Leastremainingquotaowner đủbudget+15%;đoT4không suy từlocalGPU.
Privateinlinekey theoquyềnđãchốt,private_push0600/publicsecretfree;
source/notebookPUSHEDimmutable. Khôngraw/perturbed/latent/recordgradientupload.
