# Standard-normal copula — feasibility06/11

Problem: [marginalonly](results-marginal-copula.md) tăngTSTR hơnforesthybrid màpairTV khôngtăng; khôngcần2Dtableđểgiải thíchgain. CodeCore hiệnfitmu/cov của mid-ranknormalscores rồi `Phi(mu+L*eps)`. Khi nhiều ties/discretevalues, scoredistribution khôngN(0,1); PhiGaussianmean/variancekhácchuẩn khôngUniformmarginal, nêninverseCDF khônggiữhistogramtarget.

Gaussian copula dùng latent mean0/correlationmatrix: [statsmodels official GaussianCopula](https://www.statsmodels.org/devel/generated/statsmodels.distributions.copula.api.GaussianCopula.html). Đây làđối chiếucơchế, khôngutilityproof hoặcrootcausemọilỗi. Standardizingcovariance khôngước lượng latentcorrelation tốiưu cho dữliệurời rạc; projection/fractionalflags vẫnchưađượcsửa.

Arm mới **khôngquantilecalibration,khôngpairtables**: từLđãdecodefp16, đặtmu=0, rownormalizeL bằngnorm2từngrow. `Lnorm Lnorm.T` códiag1, giữcorrelation củaGaussian hiện tại. CDFuint16/hist/mass/featureorder giữcũ; đây làmethodchange generator, khôngcodeoptimizationbitexact.

CPUprobe `scripts/probe_standard_copula.py`, sampler `scripts/standard_copula_generator.py`; samepanel/seed16384draw/class/4096TSTRrow,classifierHGB40/31/seed42. ControlreferenceCM đãrecoververify; actualsampleSHAphảikhớp. Unit: diag1/marginalbinary recoveryPASS.

Gate mới gắnmechanism trướckếtquả: meansingletonTV val giảm≥20%, khôngclassmeanTV tăng>.02, TSTRmacro≥+.003; benign/timeDelayTSTRclassF1 khônggiảm>.02. PairTV diagnostic, khôngprimary vìkhôngnhắmnonlinearpairs. Gate/verdict mọiarmtrướckhôngđổi. KhôngHPsweep, khôngofficialtest, reusedval khôngfreshconfirmation.

PASS→GPU streaming/resume/finite/codec/T4timing rồiFL2seedendpoint50 cùngcontrol/Q/LA/floor/cache/compile. FAIL→phântíchlatent/ties/classfailure vàcơchếmớitrongquyềnowner, khônglặp unchanged.
