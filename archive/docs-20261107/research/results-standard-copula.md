# Standard copula — localFAIL, 06/11

[Plan](screening-standard-copula.md), `runs/standard-copula/result.json`: workerexit0/marker, đủ16class. SingletonTVval 0.051059583→0.046991119: giảm8%, khôngqua20%gate. TSTRmacro0.852839726, hơnbaseline.761809; singletonguard/classbenign-timeDelayguardPASS. OverallFAILgiữnguyên, khôngpromotechuẩnhoálatenttừTSTRalone.

Đây làcode/mechanismđúngtheodefinitionGaussianCopula nhưngutilitykhôngtừđịnhnghĩa. Hypothesis cầnphântích: empiricalmarginaltrain khácvaltheothờigian; guardvalTV cóthểchứatemporalshiftngoàifittingerror. ChưađokhônggọilàlýdoFAILđãxáclập. Chẩnđoántiếpgiữgatecũ, đotrain→val, base→train, standard→train vàvalriêng mỗi77feature×16class. KhôngđổigateđểnhậnarmđãFAIL.

Hybridforest+transport trướcđóqua3cổnglocal vẫn chỉlàcandidate, khôngphươngphápFLđãthắng. TạithờiđiểmCPUprobe chưa cóFLgeneratorrunmới; runtimegate bổsung07/11 ởdưới, verdictlocalFAIL giữnguyên.


## T4 runtime gate — 07/11

[Plan](../../docs-20261109/research/screening-standard-t4.md). Seed42/43 đủ2round/cosine50, toàn5343xe,2T4sm75/compiled2rank.
Session1 budgetstopr1 đãauditpartial; session2 importfullstate2rankr1, logresumeđúngr2, Inductorcachetarrestore0.
Source/config/IDs/support/CM→10metric/teacher/snapshot/finite/latentnormalization/bootstrap1000439B đãauditPASS.
Cảhai seed r2 skipped_frac0. Worker đóng02:28:58; recheck từdisk ở
`runs/standard-t4-resume/artifact-recheck.json`, audits tại `runs/standard-t4/collection/Z{0,1}-r2/`.
Đây làruntime/resumeproof, chưaT4bitexactresumevsstraight, chưautilityr50. Khôngtuyênbốcachehit/speedup từchỉtarrestore.

| metric r2 | seed42 | seed43 |
|---|---:|---:|
| accuracy | 0.894971682 | 0.893994220 |
| precision_macro | 0.886286719 | 0.888865477 |
| precision_micro | 0.894971682 | 0.893994220 |
| precision_weighted | 0.886281866 | 0.885887921 |
| recall_macro | 0.842883228 | 0.841983282 |
| recall_micro | 0.894971682 | 0.893994220 |
| recall_weighted | 0.894971682 | 0.893994220 |
| f1_macro | 0.853074615 | 0.853092993 |
| f1_micro | 0.894971682 | 0.893994220 |
| f1_weighted | 0.880014452 | 0.878759678 |

KhôngchọnHP/bestround từr2; mọi thửnghiệmmới chờ07:00UTC+7 theoowner, ưu tiên nốiY50.
