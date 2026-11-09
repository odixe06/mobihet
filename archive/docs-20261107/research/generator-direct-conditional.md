# Sinh trực tiếp từ conditional statistics — sau pair-probe FAIL

06/11/2026. Owner hỏi phương án tiếp; đây là đề xuất dựa trên [probeFAIL](results-pair-generator.md), chưa implement/launch. Ràng buộc và protocol giữ như [proposal](../../../docs/research/proposal.md). Nguồn tree/marginal fitting: [primary](../../../docs/research/generator-primary-sources.md).

## Vấn đề thật vừa đo

Reweight32768candidate/class rồi evaluate16384candidate độc lập tạo ESS chỉ1–8mẫu; pairTV .199→.751. Không đủ evidence kết luận Gaussian có zero support: covariance+ridge Gaussian continuous thường có fullsupportlatent; vấn đề có thể là xác suất rất nhỏ/overfitfinitepool. Không chạyFL phương án này, không tăng candidate/temper/clipping mù để tìmwin.

## Ưu tiên: forest histogram cho nhóm feature, Gaussian conditional cho phần còn lại

Giữ12cặp đãcôngbố, không chọn lại theo officialtest. Danhsách đó tạo **forest12cạnh/19feature/7component**, không cycle. ChoS là19feature trongforest, R là58feature cònlại.

1. Mỗi xe đếm bảng chung H[c,i,j,a,b] và singleton; chỉ tổng toànmạng. Fixededges,16bin,16class, uint32count up/fp32table down ~393216B/xe mộtlần, chưaheader/edges/SecAgg; cóthể dùng histogram256bin sẵncó cho singleton khi decode trongbin.
2. Chọn root mỗi component bằng quy tắc tên/index cố định. Lấy binroot theo singletonclass, rồi ancestral sampling từngchild từ H(childbin|parentbin,class). Zero-parentcount không được gặp nếu root/conditional đều derive từ count nhấtquán; codec phải giữ quy tắc này. Singleton/edgepair coarsebin khớp empirical pooledcounts nếu tables nhấtquán và samplingexact.
3. Trong coarsebin đãchọn, lấy finebin theo mass256bin class cắt trongcoarsebin; cuối cùng nội suy giốngCore. Histogram256 vàpair16 phải align biênchung[-4,4]; 256chia16 nên poolingfine→coarse nhấtquán trướcquantization. Count bảngsingleton/edge phải đốichiếu để tránh codec làmchúng mâu thuẫn. Không ép quan hệ vật lý trên scaledvalues.
4. Từ xS lấy zS=Phi^-1(F_c,S(xS)) theo CDFclass, rồi sinh zR theo Gaussianconditional của copula gốc:

```
A = Sigma_RS @ inverse(Sigma_SS)
mu_R_given_S = mu_R + A @ (zS - mu_S)
Sigma_R_given_S = Sigma_RR - A @ Sigma_SR
```

Dùng solve/Cholesky, không tính inverse tườngminh; kiểm finite/PSD và sai sốridge. DecodezR bằngcdf nhưCore. Triểnkhai phải thốngnhất CDFcodec/saturation/quantile ties; cần test identity conditional against directjoint Gaussian trongfixture trước data thật.

Phânphối thiết kế là `p_forest(xS|c) * q_copula(xR|xS,c)`. Đây là construction mới: exact tínhchấtcoarseforest không có nghĩa mọijoint77dim đúng. Sinh vùngforest bằngtabletrực tiếp, không cần tìm vùnghiếm trongcandidatecopula; không còn ESS/SIRcollapse cho xS. Covariance conditional giữ một phần liênhệ S↔R và R↔R; quan hệ trongS ngoàicạnhforest bị mất.

**Giới hạn cần nói rõ:** lấyroot độc lập làm7component độclập trongS, cóthể bỏ liênhệcopula hữuích. Không gọi fullypreserve densecovariance. Một biếnthể rootcopula cóthể nối cácroot nhưng cần thiết kế/gate riêng; không tự gộp vào armđầu. Các77marginal không tựgiữ vì conditionalGaussian ởR chịu xS mới. Đo singletonR cũng nhưpairngoàiforest trênheldoutval.

## Phương án đốichiếu: tree histogram toàn77feature

Cây76cạnh,8bin:622592B/xe hai chiều32bit, chưaheader; cộng L50 thành1.314058B/xe/50round. Ancestral sampling trực tiếp, dễkiểm count/support và không matrixconditional. Nhưng bỏdependence ngoàicây,8bin coarsening mạnh. Học cây từallpairmarginals đắt (~12MBuplink/classconditioned77features/8bin toàn16class), không được gọi selectionmiễnphí. Nếu làm, chọnbackbone public hoặc từcovariance sẵncó; không suy Pearson chọnđược nonlinear tốt nhất. Ưu tiên foresthybrid trước vì giữ nhiềucopula hơn và budgetnhỏ hơn, chưa proof utility.

## Phương án có điều kiện khác: copula theo trạng thái quan sát

Dùngcondition thật tạirow (vd flag`ses_dt_nan` hoặc vùngfeature được khóa), thu histogram/moment riêng mỗi class×state, rồi sample state theo pooledcount. Đây khác scenario8 vìcondition tácđộng trựctiếp tới cơchế thiếu lịch sử, và khác projection vì fit lại jointconditional chứ không chỉ sửa cờ sau sinh.

Chỉ mở khi train/val chứngminh copularesidual tập trungtheostate và mỗi state đủsupport; không có evidence đó thì không launch. Kstate nhân chi phíphầnthốngkê bịchia khoảngK; 16class×77dim fullsplit cóthể vượtbyte. Không chọn bin/condition quaofficialtest. Mixture từngFAIL nên oraclePASS không đủ để mởlại mọi mixture.

## Cổng đềxuất trước GPU dài

- Khóa12edge/16bin và samplingrule trướcfit; metadata/countsource cốđịnh, không officialtest.
- Test known nonlinear/disconnected distributions, tablecounts từclientpartition cộng đúng, coarsejoint sinh khớpconfidence sampling, codeccount consistency, Gaussianconditional identity, RNG/resume.
- Bounded train→val data giốngprobeđầu; đo12pairTV và allsingletonTV, thêmpairngoàiforest đểphát hiện mấtdependence. Bổsung TSTRmacro/classprecisionrecall, không chỉ pairfittrênedges đãthu.
- SoCopulacontrol: pairTV giảm≥20%, không classsingletonmeanTV tăng>.02; TSTRmacro≥+.003. Kiểmval từngclass, đọcbenefit/loss ởbenign/timeDelay, không chọnbestHP. Utilitythreshold từscreening đầu, không hạ theoFAIL.
- ĐoT4throughput, memory và compile/backward gate; không tuyênbốxnhanh hoặc cachetăngtốc khi chưađo. Hầu hếtlookup/matrix sampler cóthể batchGPU, nhưng sequentialedgecost làthực.
- Chỉ sauPASS mớiFL2seed cùngharness, cosine50/CoreQ/LA/floor và64synthetic/step. Controlrerun nếuexecution/RNGpath thayđổi; không lấyTSTR thaypersonalizedown-test utility.

Đây là hướng sinhtrực tiếp để khắc phục limitation của finitecandidate reweight, không tuyênbốđãđạt oracle hoặc tốiưu thốngkê.

Owner đãduyệt tiếp tục vòngthử; [directprobe đangchạy](screening-direct-generator.md), chưaFL.

[Directprobe đãFAILsingletonR](results-direct-generator.md); [marginaltransport đangthử](screening-marginal-direct-generator.md). KhôngcoipairTVPASSlàutilityPASS.
