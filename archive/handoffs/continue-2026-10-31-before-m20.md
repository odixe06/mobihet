> Snapshot trước M20; trạng thái hiện tại tại [bàn giao sống](../operations/continue.md).

# Continue — bàn giao sống, 31/10/2026 ~17:12 (giờ VN)

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [lessons vận hành](../operations/lessons.md).
Snapshot trước báo cáoL: [16:55](../archive/continue-2026-10-31-before-l-report.md).
Repo này nghiên cứu federated/MobiHet; baseline/NILM/Notion ở `../fl_dagsnet_veremi_v2`.

## Quyền và ràng buộc

- Khônggửi rawlocalrow/featuretừngmẫu/nhãntừngdòng/privateweights. Trao đổi thốngkêgộp vàoutputschema16lớp; model cóthểkhác kiếntrúc.
- Ưu tiên F1, chưa trầnbyte; báo bootstrap+upload+download vàthời gian. GiữDAGSNet chung trước, mixedarchitecture sau.
- Screening10round/full5343xe/cosine50đãduyệt. **Chưa launch/chốt50round hoặc batch sauL.** KhôngNotion/Git/DVCpush/cleanup.
- GlobalCLI **dinhan25** giữ nguyên. Credentialaccountscoped, khôngMCPJSON/loginlại. HEAD099e95e8466811881fb71dcd7d3c44268ed082d6,
  stagedtree5cd0bcc611b032ca91743dc3e605a9c95d1911a3 giữ nguyên.

## Trạng thái hiện tại — không còn job L/watch/collect đang chạy

L0–L11 exactversion1 COMPLETE. [SpecL](../research/screening-l10.md), [kết quảL](../research/results-l10.md).
12pull exit0,660file/4.331GiB, source+metrics+CM+probe+smallsnapshots. Fullprivatebank vàexecutednotebook khôngpull.
Audit120testCM→10metric/perclass/clientdistribution,48trainCM→10metric,1632clientproberecord/teacherlag,
120snapshotfinite/Qchuẩnhóa. Source3overlay/base/HP/LR/IDs/support/77featureorder/labelmap và2×T4/compilegate2rank PASS.
Chưa prediction→CM/fullbankrebuild/exactresume/independenttemporalheldout/50round/mixedarchitecture/contactevaluation.

### Job nền đã kết thúc, completion đã consumed

- `kaggle-l-v1-20m-20261031`: succeeded16:50:53,queueaccepted16:50:54. Tick16:10=12RUNNING,
  16:30=4COMPLETE/8RUNNING,16:50=12COMPLETE. `runs/l-screening/watch-20m/terminal.json`, `watch-20m.log`.
- `kaggle-l-v1-collect-audit-20261031`: succeeded/exit0/marker16:59:20, queuecompletion đãconsumed đểphântích/báocáoL.
  `runs/l-screening/collect-audit.log`; state `/home/odixe/.codex/long-jobs/jobs/kaggle-l-v1-collect-audit-20261031/state.json`.
- Ownerthread `01a18d65-e3b8-73f1-80b6-7b66531915da`, deliveryqueue, viewernone. Khôngretrydelivery/startwatchertrùng.
- Owner muốnpollnền20phút, **khôngmodelpolling lúcchờ**. Khi cóbatchđượcduyệtmới mớiđăngkýjobwatch mới;
  chưa cówatcher/collector nàotiếptụcsauphiênnày. Localwatch khôngsurvivereboot.

## Kết luận L — dùng khi quyết định tiếp

| cell | F1 seed42/43 | p10 seed42/43 | cổng42/43 |
|---|---|---|---|
| softfixed/noReal L0/L1 | .886315/.886861 | .616754/.600000 | PASS/PASS |
| hardfixed/gate L4/L5 | .884547/.884579 | .624465/.617117 | PASS/PASS |
| softfixed/gate L6/L7 | .887030/.887433 | .620423/.606002 | PASS/PASS |
| softadaptive/gate L10/L11 | .884016/.884720 | .608968/.596747 | PASS/PASS |

L2/L3hardadaptive vàL8/L9comboadaptive FAILgain. Khôngnớicổng.
Fixed1 hơnadaptive ởcáccellcùngtarget/realKD. Softtarget vẫn cógiátrị: +.002967/+.002527 vsaugmentation khi scale1.
L6/L7gainvsG4+.013842/+.012925, vsaugmentation+.003682/+.003099; realgatechỉthêm+.000715/+.000572 trênsoftfixednoReal.
L4/L5tốt hơntail/ítmẫu, L6/L7xe<256 vẫnkémaugmentationmean−.007724/−.007348. Khôngcellthắngmọimục tiêu.

Probe34clientpostepochfp32/dropout0 khôngactualAMPgradient; adaptiveweightmedian2,38–3,88,
weightedsyntheticgradientmedian2,35–4,89×realCE. Hỗtrợhypothesisauxquámạnh, khôngcausalproofduynhấttừcosine.
Train+valđãfit, khônggọitrainF1validation. TimeDelayprior5,151→5,162%gầnnhưgiữ nguyên nhưng gaptrain/testlớn.
SửaSybilfulltrain+val8,80%vs17,67%test trongknowledge/LESSONS (8,0%cũkhôngđúngsplitFLhiệntại).
R1cùngwarmupmath nhưngF1spreadgiữasessioncùngnominalseed.000516/.000588; chưaisolatenguyênnhân,
khôngCI/noisebound/bitexactT4proof. Lợirealgate nhỏ cầnlặpcontrol.

## Chi phí và artifact

MọicellL1088B/xe/round +3311744Bbootstrap =3322624B/xe/10round; augmentation0learnedB/r sau bootstrap.
L6/L7trainwallr2–10mean621/625s/job2×T4, eval~17s/r; diagnostic295/298s tổng4probe/fulltrainpasses.
Khônggọithờigianaccountkhácnhau làcausalspeedup hoặc latencyhardwarexe; bootstrap/transport chưaend-to-endcodecproof.

- `runs/l-screening/batch.json`:12account/slug/HP/source/notebookSHA, base099e95e8466811881fb71dcd7d3c44268ed082d6.
- Source_manifest12/12=5c6e9f9c4e0923b1a28a3b76a393e481018397168228f69fdc61743cd56f3b1b; overlay3file giữ nguyên:
  methods/mobihet.py1ffaae2b824711df4798b1fc5f8b0c89ae9666f0b4fb770fd10911e33e57bf89,
  methods/mobihet_diagnostics.pyebe4b1e6bc76ac75acb990e3f3ced4253ee568ea8804670327dc167d773d7504,
  scripts/run_fl.py747fd50ba2a4ec7fbc3697e7a01f0b717e09e27c84884af0b7e626e1965bfe18.
- `runs/l-screening/{collection,audit,diagnostic-audit,analysis}.json`, metrics-r10.csv; perrun `runs/pulls/SLUG/SLUG/`.
- Reproduceaudit: `python scripts/audit_l_diagnostics.py`; analysis: `python scripts/analyse_l_screening.py`.
  BáocáoL sinh bởi `runs/l-screening/write_report.py` từauditedJSON; khôngchọnround/HPtrêntest.

## Bước tiếp — đề xuất, chưa launch

1. Giữsofttarget+fixed1 làmnền; giữsoftfixed/noReal vàaugmentation làmcontrol. Lặpcấuhìnhgatefixed đểxétđộnhạysession.
2. So syntheticfallback (teacherargmaxsai→hardlabel riêngsynthetic) vàpostepochteacher vớicontrolcùngcodec/scale/generator;
   khônghardcodeclass13, khôngmởlạibucketB8/anchor/GMM/MADEđãFAIL thiếugiảthuyếtmới.
3. Nhánhdiagnostic fitmới foldtrain→foldval chưađượchọc/statfit đểxétshift/teacher/featureoverlap; khôngđổi protocolheadline.
4. Chỉ50round/kiếntrúctrộn/contactdrop khi chốtphạmvi; nếuresume cầnkiểmfullstate/RNGtrước, không dùngsmallsnapshot.

Kbaseline/history: [Kresults](../research/results-k10.md), [H](../research/screening-h10.md), [J](../research/screening-j10.md).
Nền42G4=.873187514625593/p10=.5737619047619047; nền43H0b=.8745079727196574/.5687566137566138.
Cổnggain≥.010,p10Δ≥−.010 từngseed; giữ dùsátngưỡng. Không việc tay bắt buộc hiện tại.
Phiênnày chỉphântích/báocáo/lessons/handoff/datasetnote; khôngđổifrozenmethod/traincode, khônglaunch/Git/DVCpush/cleanup.
