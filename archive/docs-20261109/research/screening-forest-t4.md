# Forest + quantile — triển khai GPU, khóa 07/11 trước push

Owner yêu cầu triển khai các thử nghiệm trên Kaggle bằng quota còn lại. [CPU feasibility](../../docs-20261107/research/results-marginal-direct-generator.md)
PASS chưa chứng minh FL. Giữ 12 pair, 16 coarse bins, 256 fine bins, 32768 synthetic calibration/class seed9200,
4097 quantiles; roots độc lập, conditional Gaussian cho58feature R, transport chỉR. Không chọn cấu hình qua official test.

## Port và codec

`methods/mobihet_forest_quantile.py` kế thừa Core WLive, thay bootstrap/sampler. Core Q/LA/floor16/model/batch/steps/optimizer/cosine50
giữ nguyên. Hai rank chỉ allreduce tổng singleton/pair count và các thống kê Core; không chia sẻ dòng/model/gradient.

Core CDF uint16 có thể xóa mass coarsebin rất hiếm trong khi pair count vẫn dương. Vì vậy downlink bổ sung **count nguyên
uint32 cho19 selected marginals**, cùng12 joint tables; kiểm coarse margins của cả hai đầu mỗi cạnh bằng integer equality.
Không tính lại singleton uplink: nó đã nằm trong Core sparse histogram. Joint uplink196608B, downlink joint196608B
+ selected fine counts311296B. Tổng bootstrap **1704951B/xe**, thêm704512B so Core; Q1088B/round giữ nguyên.
Đây là codec bổ sung đã công bố, không gọi cost393216B của proposal cũ là tổng thật.

Torch dùng fp32 và RNG riêng của sampler, class-major searchsorted tránh nhân bảng4097quantile theo từngdraw.
CPU calibration vẫn NumPy/scipy từ **thống kê đã decode**, RNG9200 độc lập; không consume training stream.
Zero-probability sampling tại uniform0 dùng right insertion. Source CDF của production là Coredecodedfloat32,
khác CPUprototype float64 ở rounding; không tuyên bố CPU/Torch bitexact. Không tối ưu hay thay đường train chung/control.
Static sampler SHA lưu trong fullresume và kiểm khi khởi tạo lại; torch gen_g và RNGglobal do driver hiện tại lưu.

## Cổng trước launch

1. Tests nonlinear pair, marginalR, quantile ties/endpoints, không mutate CPU statistics, sampler RNG replay.
2. LocalGPU DAGSNet realfixture613/2029,2seed: compile/backward và4round straight vs stop2→3→restore2→4;
   fullstate/history/CM/snapshot bitexact **trong cùng phương pháp**, không so bitexact với Core.
3. GPU sampler trên bounded train→val panel cũ,16class: pairTV giảm≥20%, singletonmeanTV/class không tăng>.02.
   Khôngfitmodel lại hoặc nới gate theo kết quả. Đây không fresh validation.

## Kaggle runtime gate

Hai kernel mới `mh-z2-forestq-s42-t4r2`, `mh-z3-forestq-s43-t4r2`,2round cosine50, toàn5343xe/own-test như protocol.
Chọn2 tài khoản idle còn>3h, loại4receiver đã dành choY50 và cácsessionactive/unknown. Cap2.5h/account, reserve×1.15,
quota và source/access checks mới trước push. Không chiếm receiverY50. Account thực ghi trong launchmanifest.
Owner07/11 02:39 cậpnhật: ưu tiên kernel cần nối phiên, **cho pushngaythửnghiệmkhảthi bằngquota dư**.
Thay lệnhhoãntoànbộ02:28; timer0700 đãcancel trướcpush. Worker mới inventoryquota/slot/access ngay,
giữ4receiverY50 vàchọn2accountdư>3h. Batchdàichờreset; quota luôn đọc thật.

Public npycache+SHA, compilethreads1, Inductorcachetar và privateW&Bonline. Keyprivateinline đã đượcownercho phép,
lưu vĩnh viễn trong Kaggle private versionhistory; không Git/DVC push notebook. Worker poll1200s, pullCOMPLETE/ERROR,
auditroundthật; partial budgetstop đượcghi partial, chưaPASSđích2, phải nốicheckpointcompatible sauđó.

Gate:2T4sm75,compilePASS2rank/compiled,2roundcommitted,skip≤1%/positiveapplied+finite,source/config/IDs/support/LR/CM→10metric,
teacherlag/snapshot/wirebyte theo auditorchung. Forestrecord2rank phải cùngstaticSHA, countsconsistent, đúngcalibration/budget;
ghi benchmark samplerG1/G32 soCore bằnggeneratorriêng, không tiêu thụ RNGtrain. Chậm vẫn reportthật, không bỏsteps đểtăngspeed.

PASS chỉ runtimeproof. Trước utility50 phải đọc tradeoff/khóa phépso sánh vớiCore2seed; không chọn theo own-test r2,
không mở lại pairreweight/directforest đãFAIL. FAIL phân biệt code/runtime/budget/audit và methodutility.

Artifacts: `runs/forest-production/{preflight,sampler-gate}.json`; launcher `scripts/run_forest_t4.py`, receipts
`runs/forest-t4/`, artifact `artifacts/forest-t4/`. Chưa có utilityFL hoặc bằng chứng phục hồi gainoracle.
