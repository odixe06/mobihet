# Error register — đọc trước tạo notebook

Giữ lỗi còn tác động tới workflow hiện tại. Chi tiết kernel cũ ở [snapshot](../../archive/docs-20261107/before/docs/operations/notebook-errors.md).
Run sống tại [continue](continue.md); xử lý theo [lessons](lessons.md). Không sửa scientific source đang chạy.

| lỗi/dấu hiệu | kiểm đúng trước khi hành động | cách phục hồi |
|---|---|---|
| Zero GPU/sai T4 | accelerator cảnotebook+metadata, image, device count/capability | giữ assert; runtime đúng/account khác theo quyền, không reinstall torch |
| RUNNING nhưng chưa cell/W&B | exact version, bounded live log, allocation/quota | QUEUED/chưa cấp máy không training FAIL; không push trùng khi trạng thái chưa rõ |
| Parent mount mất dù push success | refused-input reply, actual mount/import round | private checkpoint dataset theo receiver; require-resume trước decode |
| Dataset create403/parse reply | intent/result, owned listing/status, all-page files | retry read bounded; không create/upload mù |
| State một rank thiếu/lệch round | state.pt + state_rank1.pt, source/config/RNG | dùng last verified compatible boundary; không snapshot thayfullstate |
| Artifact count thiếu/partial pull | exact version, hếtpage, sizes/Range headers | resume partial download; nonzero khi còn thiếu, không kết luận từexit0wrapper |
| COMPLETE nhưng thiếu đích | rounds/CM/state thực, worst-round budget | audit partial rồi nối; không nới đích hoặc gọi methodFAIL |
| Compile forward tốt, backward NaN | gradients/BN/padded slots, seed/rank/shape, applied từngxe | giữ failure; sửa nguyênnhân rồi gate; không bỏclient/hạguard |
| Float probability auditor falseFAIL | stored aggregate/body, CPU/CUDA bound quaquantizer | sửa checker vàverify artifactcũ; không train lại |
| Hybrid LG auditor falseFAIL trước push | ô nhỏ dùng Q8→fp32 log/center, không phải fp16 logscore; replay packet cũ GPU vs CPU | bound theo tỷ trọng count từng nhánh; regression corruption; reuse smoke khi training SHA/HP giữ nguyên, không retrain |
| W&B stale/crashed | exact Kaggle session + timestamp/heartbeat + quota | đọc đa nguồn; không dừng chỉ từ W&Bstate |
| Source thay giữa preflight | hash đầu/cuối, module import, frozenoverlay | giữ evidence cũ; preflight trênsourcecốđịnhtrướclaunch |
| JSON numpy.bool_ | CM/log/source/sampleSHA đã lưu chưa | recover JSONtừartifact, castbool ởwriter; khôngrefit |
| Resume trùnghistory/step | checkpoint boundary vs rows/W&BID | rollback rows sau boundary; không advance W&B trong preflight |
| Watcher chết/reboot | PID/start identity, lock, receipt và remote state | phục hồi collector, không restart training |
| Cap ngắn dừngr1 | coldcompile/eval/diagnostic/finalsave | đủbudget+margin, restorecompilecache, nối2rank |

Guard chỉ phát hiện lỗi, không chứng minh numerical root cause đã sửa. Recovery giữ failure gốc và source/slug mới nếu trainingcode đổi.
Kiểm full source/config/CM/round/byte thực; localfixture hoặc healthy W&B không thay whole T4 audit.
