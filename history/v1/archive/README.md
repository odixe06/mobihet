# Archive — build và công cụ đã kết thúc

Source, probes, calibration, pilot, artifact, logs và DVC metadata giữ cùng build. Không xóa history hoặc sửa source đã chạy. Thiết kế/kết quả đọc tại [docs/archive](../docs/archive/README.md); bài học dùng tiếp tại [docs/lessons](../docs/lessons/README.md).

| build | căn cứ lưu trữ |
|---|---|
| [pilot-baselines](methods/shared-baselines/builds/pilot-baselines/README.md) | Đợt pilot: pilot/probe/đối chứng đã kết thúc; không có launch tiếp y nguyên |
| [fragmented-snapshots](methods/contact-fragmentation/builds/fragmented-snapshots/README.md) | Đợt p2: pilot/probe/đối chứng đã kết thúc; không có launch tiếp y nguyên |
| [static-synthetic-books](methods/synthetic-distribution-matching/builds/static-synthetic-books/README.md) | Đợt new-fl: pilot/probe/đối chứng đã kết thúc; không có launch tiếp y nguyên |
| [countsketch-subspace-ratio](methods/shared-comparisons/builds/countsketch-subspace-ratio/README.md) | Đợt v2: pilot/probe/đối chứng đã kết thúc; không có launch tiếp y nguyên |
| [proxy-carriers](methods/proxy-compression/builds/proxy-carriers/README.md) | Đợt v3: pilot/probe/đối chứng đã kết thúc; không có launch tiếp y nguyên |
| [support-mala](methods/gaussian-sampling/builds/support-mala/README.md) | Đợt v4: pilot/probe/đối chứng đã kết thúc; không có launch tiếp y nguyên |
| [anchor-synthetic-books](methods/synthetic-knowledge/builds/anchor-synthetic-books/README.md) | Đợt v5: pilot/probe/đối chứng đã kết thúc; không có launch tiếp y nguyên |
| [geometry-g59-g61](methods/geometry-primitive-moments/builds/geometry-g59-g61/README.md) | Đợt v6: pilot/probe/đối chứng đã kết thúc; không có launch tiếp y nguyên |
| [nystrom-rank4-rank8](methods/covariance-nystrom/builds/nystrom-rank4-rank8/README.md) | Đợt v7: pilot/probe/đối chứng đã kết thúc; không có launch tiếp y nguyên |
| [hybrid-pilot](methods/gaussian-proxy-hybrid/builds/hybrid-pilot/README.md) | Đợt v8: pilot/probe/đối chứng đã kết thúc; không có launch tiếp y nguyên |
| [anchor-residual](methods/anchor-function-residual/builds/anchor-residual/README.md) | Đợt v9: pilot/probe/đối chứng đã kết thúc; không có launch tiếp y nguyên |
| [gaussian-base-energy](methods/dagsnet-joint-energy/builds/gaussian-base-energy/README.md) | Đợt v10: pilot/probe/đối chứng đã kết thúc; không có launch tiếp y nguyên |

[Input-statistics probe](methods/gaussian-class-moments/builds/input-statistics-probe/README.md) cũng lưu tại đây.
[Công cụ lịch sử](tools/README.md) · [Index từng file](../docs/archive/file-index.md).
Một số code trong archive vẫn cần cho snapshot frozen hiện tại; compatibility map giữ dependency, không nhân đôi source.
