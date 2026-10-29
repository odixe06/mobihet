# Bài học — Synthetic book khớp latent mean

Chưa gán thất bại cho toàn họ condensation. Refresh và encoder phải kiểm riêng bằng 2×2, cùng số bước synthesis.

S-a staticFAIL; V5 đãđủIPC1/4×2seed cùng frozen/trained encoder và gradient/mean objectives.
Mean-trained vẫn thuaGaussiancontinuous; currentencoder không cho lợi đủcổng tronggradientcontrast.
Không còn pendingaudit cho adaptation này; khôngsuy toàn họDM bịbácbỏ.
[ReportV5](../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/results/report.md), [gates](../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/probes/gates.json).

Bằng chứng và giới hạn: [README](../archive/methods/latent-mean-synthetic-books/README.md). Kết luận chỉ áp dụng adaptation/cấu hình đã thử; không phủ định toàn họ phương pháp.
