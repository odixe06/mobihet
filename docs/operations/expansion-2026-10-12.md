# Kiểm nghiệm mở rộng — 2026-10-12 UTC+7

Chủ dự án yêu cầu mở rộng hướng pilot có lợi và triển khai hướng chưa kiểm chứng;
không tạo run trùng nếu kernel tương ứng đang chạy.

## Quyết định mới đã chốt

- H1/H2:20client,100%train80%,50round,**chỉseed42**. Batch512,cosine1e−3→1e−5,
  DAGSNet-only; test20% descriptive/full mỗi round, headline r50, không chọn HP từ test.
- Nyström:rank4/8,explicitWupper có tính byte,FP32upload/FP16factor-download;
  freshfull/lr4/lr8/diag +own/sign-shuffle; cổng tại design§7 đã được chủ dự án đồng ý.
  Pilotseed42/43; seed42-only chỉ áp dụng đợt full H1/H2.
- Hybrid:pilot2×2 local/Gaussianfull/DML/DML+Gaussian, thêmown controls;
  syntheticCE vào cảprivate/proxy, KD trênreal, BN batchtrộn. Seed42/43,20round.
- Mọi notebookW&B online trướctrain; theo dõiround1–5. Có lỗi/bất ổn đang chạy:
  báokernel/link/bằngchứng và yêu cầu chủ dự án dừng; không tựcancelkernel.
  Numerical/deadline guard trong notebook vẫn dừng theo manifest.
- KhôngGit/DVCadd/push/cleanup; localchỉboundedprobe/smoke/collect/report.

## Run đã mở, không mở trùng

V6đã qua15smoke/4exactresume/strictinference;4privateinputREADY.
Pushversion1 lúc11:11UTC+7 ngày2026-10-12:

| family/seed | kernel | quota trước push h |
|---|---|---:|
| A42 | [minhtran0601/mobihet-v6a-s42-20261012-v1](https://www.kaggle.com/code/minhtran0601/mobihet-v6a-s42-20261012-v1) | 5,77 |
| A43 | [dinhan25/mobihet-v6a-s43-20261012-v1](https://www.kaggle.com/code/dinhan25/mobihet-v6a-s43-20261012-v1) | 6,24 |
| B42 | [trietbackup/mobihet-v6b-s42-20261012-v1](https://www.kaggle.com/code/trietbackup/mobihet-v6b-s42-20261012-v1) | 13,95 |
| B43 | [odixe0502/mobihet-v6b-s43-20261012-v1](https://www.kaggle.com/code/odixe0502/mobihet-v6b-s43-20261012-v1) | 14,94 |

Tài khoảnítquota ưu tiênpilot. CollectorfamilyA/B jobs6, tổng12connections;
log `papers/v6/collection-{a,b}-20261012.log`, GPUauditlock dùngchung.
W&Bregistry30identity tại [wandb-runs](../../papers/v6/wandb-runs.json);
monitor [live-status](../../papers/v6/live-status.json). Snapshot11:12UTC+7 chưa cóW&Bhistory,
không chứng minh failure. Source/version đã push giữ nguyên.

## Phần đang triển khai

1. H1/H2full:driver50round, actualcodec, source/datahash, checkpointresumable,
   fulltest/10metrics/clientCM/perclass. Đo3roundT4/budget trước launch chuỗirun dài.
2. Nyström:source/smoke/resume/pilotU theo cổng đãchốt; chưacontact trướcU.
3. Hybrid:contract2×2/own, smoke/resume/pilot; không đưa lênfull trướcutility.
4. K2/K4:nghiêncứu cách sửa thiết kế hợp lệ rồi chốt scientificchoices trướctraining.
   K4unbasedR66 đãFAIL vềnormalizability; khôngrerun y nguyên.

W&Binline key nằm ignoredprivate_push0600 và private Kaggle versionhistory lâu dài;
không vào publicsource/log/chat. Không cần thao tácmanual cho bước đã thực hiện.

## Collect lúc12:51UTC+7

APIowner-scoped: A42/A43ERROR, B42/B43COMPLETE; khôngV6đangchạy cầnmanualstop.
W&B30identity cóhistory:18complete20round; G59trướccommit hoặc1round, chưaheadlineartifact.
Oldcollectorprocess khôngcòn sau phiên bịtạmngắt; khởi động lạifamilyA/B jobs6,
Range-resume qua helper; logs `papers/v6/collection-{a,b}-20261012-terminal.log`.
Seed42A333file/0missing đãthu; source/payload/model/prediction auditđanghoàntất,
seed43 vàfamilyB đangthu. Khôngrerunkernel.

Seed42G59cả6arm lỗi `V6 synthetic fp16 input overflow`; parentERROR bởiassertallcomplete.
Bafull66control đềucomplete20round. Giữbằngchứngnumericalpartial vàcontrol hợp lệ.
Đã có `scripts/report_v6.py` tínhquality/compressionriêng vàjointinteraction;
chỉAUDITED được vào report. W&Bfinished không thay complete-roundgate.
