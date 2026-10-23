# Khôi phục checkpoint cũ từ Kaggle

Owner cho phép dọn tối đa checkpoint ít giá trị: ưu tiên adaptation đã đóng FAIL, kể cả pilot ngắn PASS nhưng kiểm dài hơn FAIL. Nhóm dưới là các build lưu trữ đã chốt cổng FAIL; checkpoint trung gian của đối chứng cũ cùng build cũng thuộc phạm vi. Giữ checkpoint cuối từng run, source/config, predictions/metrics, input, smoke/reference và object được manifest khác tham chiếu.

[Danh sách đầy đủ MD5 → kernel/version/output/manifest](kaggle-retired-weights.json) · [Script khôi phục](../../scripts/restore_kaggle_retired_weights.py) · [Trạng thái dọn hiện tại](backup.md).

Index có cả ứng viên đang kiểm. **Chỉ `recovery_verified: true` được phép dọn.** Ghi index trước xóa; metadata `.dvc` và `.dir` giữ nguyên. Một số object cũ chuyển sang Kaggle-only: `dvc pull` nguyên manifest lịch sử có thể báo thiếu; dùng script dưới để nạp lại object được chọn vào cache trước.

Kiểm phục hồi: listing bằng đúng tài khoản + HTTP Range đọc 1 byte/đúng tổng size; một mẫu/kernel tải đầy đủ đối chiếu MD5. Đây không phải MD5 toàn bộ output Kaggle. Mỗi lần khôi phục luôn kiểm MD5 toàn file trước đưa vào cache.

| build | ứng viên | GB | đã kiểm link | bằng chứng đóng hướng |
|---|---:|---:|---:|---|
| anchor-function-residual | 988 | 35.11 | 988 | [report](../archive/methods/anchor-function-residual/builds/anchor-residual/results/report.md) |
| synthetic-knowledge | 1064 | 18.60 | 1064 | [report](../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/results/report.md) |
| synthetic-distribution-matching | 228 | 7.53 | 228 | [report](../archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/results/report.md) |
| covariance-nystrom | 760 | 13.28 | 760 | [report](../archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/results/report.md) |
| geometry-primitive-moments | 684 | 13.18 | 684 | [report](../archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/results/report.md) |
| proxy-compression | 1216 | 83.80 | 1216 | [report](../archive/methods/proxy-compression/builds/proxy-carriers/results/report.md) |
| shared-comparisons | 942 | 29.61 | 942 | [report](../archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/results/report.md) |
| gaussian-sampling | 380 | 11.93 | 380 | [report](../archive/methods/gaussian-sampling/builds/support-mala/results/report.md) |
| dagsnet-joint-energy | 100 | 3.48 | 100 | [report](../archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/results/report.md) |
| boundary-cell-counts | 1410 | 45.46 | 1410 | [report](../methods/boundary-cell-counts/builds/boundary-cell-pilot/results/utility-report.md) |
| class-tail-risk | 544 | 11.53 | 544 | [report](../methods/class-tail-risk/builds/class-tail-pilot/results/report.md) |
| proxy-countsketch-repair | 514 | 16.20 | 514 | [report](../methods/proxy-countsketch-repair/builds/contractive-ef-pilot/results/report.md) |

## Output từng notebook

Link web mở output; version cố định trong cột/index được script gửi bằng `version_label=vN`. Không lưu signed URL hoặc token.

| notebook | version đã kiểm | file đã kiểm |
|---|---:|---:|
| [catbaochau/mobihet-ct-u-s42-20261015-v1](https://www.kaggle.com/code/catbaochau/mobihet-ct-u-s42-20261015-v1/output) | 1 | 272/272 |
| [catbaochau/mobihet-v5j-s42-20261012-v1](https://www.kaggle.com/code/catbaochau/mobihet-v5j-s42-20261012-v1/output) | 1 | 266/266 |
| [dinhan25/mobihet-bc-u-s43-20261014-stage01-v1](https://www.kaggle.com/code/dinhan25/mobihet-bc-u-s43-20261014-stage01-v1/output) | 1 | 282/282 |
| [dinhan25/mobihet-sa-s43-20261011](https://www.kaggle.com/code/dinhan25/mobihet-sa-s43-20261011/output) | 1 | 114/114 |
| [dinhan25/mobihet-v2a-s43-20261011-v1](https://www.kaggle.com/code/dinhan25/mobihet-v2a-s43-20261011-v1/output) | 1 | 204/204 |
| [dinhan25/mobihet-v2b-s43-20261011-v1](https://www.kaggle.com/code/dinhan25/mobihet-v2b-s43-20261011-v1/output) | 1 | 114/114 |
| [dinhan25/mobihet-v2c-s43-20261011-v1](https://www.kaggle.com/code/dinhan25/mobihet-v2c-s43-20261011-v1/output) | 1 | 152/152 |
| [dinhan25/mobihet-v3r-s43-20261011-v1](https://www.kaggle.com/code/dinhan25/mobihet-v3r-s43-20261011-v1/output) | 1 | 266/266 |
| [dinhan25/mobihet-v4m-s43-20261012-v1](https://www.kaggle.com/code/dinhan25/mobihet-v4m-s43-20261012-v1/output) | 1 | 190/190 |
| [dinhan25/mobihet-v6a-s43-20261012-v1](https://www.kaggle.com/code/dinhan25/mobihet-v6a-s43-20261012-v1/output) | 1 | 114/114 |
| [dinhan25/mobihet-v7n-s43-20261012-v1](https://www.kaggle.com/code/dinhan25/mobihet-v7n-s43-20261012-v1/output) | 1 | 380/380 |
| [khanhmay0304/mobihet-cef-s43-20261015-v1](https://www.kaggle.com/code/khanhmay0304/mobihet-cef-s43-20261015-v1/output) | 1 | 254/254 |
| [khanhmay0304/mobihet-ct-u-s43-20261015-v1](https://www.kaggle.com/code/khanhmay0304/mobihet-ct-u-s43-20261015-v1/output) | 1 | 272/272 |
| [khanhmay0304/mobihet-v5g-s42-20261012-v1](https://www.kaggle.com/code/khanhmay0304/mobihet-v5g-s42-20261012-v1/output) | 1 | 266/266 |
| [khanhmay0304/mobihet-v5j-s43-20261012-v1](https://www.kaggle.com/code/khanhmay0304/mobihet-v5j-s43-20261012-v1/output) | 1 | 266/266 |
| [minhtran0601/mobihet-bc-u-s42-20261014-stage01-v1](https://www.kaggle.com/code/minhtran0601/mobihet-bc-u-s42-20261014-stage01-v1/output) | 1 | 260/260 |
| [minhtran0601/mobihet-sa-s42-20261011](https://www.kaggle.com/code/minhtran0601/mobihet-sa-s42-20261011/output) | 1 | 114/114 |
| [minhtran0601/mobihet-v2a-s42-20261011-v1](https://www.kaggle.com/code/minhtran0601/mobihet-v2a-s42-20261011-v1/output) | 1 | 206/206 |
| [minhtran0601/mobihet-v2b-s42-20261011-v1](https://www.kaggle.com/code/minhtran0601/mobihet-v2b-s42-20261011-v1/output) | 1 | 114/114 |
| [minhtran0601/mobihet-v2c-s42-20261011-v1](https://www.kaggle.com/code/minhtran0601/mobihet-v2c-s42-20261011-v1/output) | 1 | 152/152 |
| [minhtran0601/mobihet-v3r-s42-20261011-v1](https://www.kaggle.com/code/minhtran0601/mobihet-v3r-s42-20261011-v1/output) | 1 | 266/266 |
| [minhtran0601/mobihet-v4m-s42-20261012-v1](https://www.kaggle.com/code/minhtran0601/mobihet-v4m-s42-20261012-v1/output) | 1 | 190/190 |
| [minhtran0601/mobihet-v6a-s42-20261012-v1](https://www.kaggle.com/code/minhtran0601/mobihet-v6a-s42-20261012-v1/output) | 1 | 114/114 |
| [minhtran0601/mobihet-v7n-s42-20261012-v1](https://www.kaggle.com/code/minhtran0601/mobihet-v7n-s42-20261012-v1/output) | 1 | 380/380 |
| [minhtriethihi/mobihet-cef-s42-20261015-v1](https://www.kaggle.com/code/minhtriethihi/mobihet-cef-s42-20261015-v1/output) | 1 | 260/260 |
| [minhtriethihi/mobihet-v5g-s43-20261012-v1](https://www.kaggle.com/code/minhtriethihi/mobihet-v5g-s43-20261012-v1/output) | 1 | 266/266 |
| [minhtrit06/mobihet-bc-u-s43-20261015-cont02-v1](https://www.kaggle.com/code/minhtrit06/mobihet-bc-u-s43-20261015-cont02-v1/output) | 1 | 424/424 |
| [minhtrit06/mobihet-v9a-s43-20261012-v1](https://www.kaggle.com/code/minhtrit06/mobihet-v9a-s43-20261012-v1/output) | 1 | 494/494 |
| [odixe0502/mobihet-v10e-s43-20261012-v1](https://www.kaggle.com/code/odixe0502/mobihet-v10e-s43-20261012-v1/output) | 1 | 50/50 |
| [odixe0502/mobihet-v3o-s43-20261011-v1](https://www.kaggle.com/code/odixe0502/mobihet-v3o-s43-20261011-v1/output) | 1 | 228/228 |
| [odixe0502/mobihet-v3t-s43-20261011-v1](https://www.kaggle.com/code/odixe0502/mobihet-v3t-s43-20261011-v1/output) | 1 | 228/228 |
| [odixe0502/mobihet-v6b-s43-20261012-v1](https://www.kaggle.com/code/odixe0502/mobihet-v6b-s43-20261012-v1/output) | 1 | 228/228 |
| [odixeuit/mobihet-bc-u-s42-20261015-cont02-v1](https://www.kaggle.com/code/odixeuit/mobihet-bc-u-s42-20261015-cont02-v1/output) | 1 | 444/444 |
| [odixeuit/mobihet-v9a-s42-20261012-v1](https://www.kaggle.com/code/odixeuit/mobihet-v9a-s42-20261012-v1/output) | 1 | 494/494 |
| [trietbackup/mobihet-v10e-s42-20261012-v1](https://www.kaggle.com/code/trietbackup/mobihet-v10e-s42-20261012-v1/output) | 1 | 50/50 |
| [trietbackup/mobihet-v3o-s42-20261011-v1](https://www.kaggle.com/code/trietbackup/mobihet-v3o-s42-20261011-v1/output) | 1 | 228/228 |
| [trietbackup/mobihet-v6b-s42-20261012-v1](https://www.kaggle.com/code/trietbackup/mobihet-v6b-s42-20261012-v1/output) | 1 | 228/228 |

## Pull một object

Chạy từ gốc repo; dùng đúng owner ghi trong index. Chọn MD5 cần dùng, kiểm dung lượng trước. Lệnh chỉ tải object đã chọn, không tải cả nhóm.

```bash
HELPER=.agents/skills/kaggle-training-notebook/scripts/kaggle_as.py
conda run -n nckh python "$HELPER" odixeuit -- python scripts/restore_kaggle_retired_weights.py --md5 b8882ad3816de58bcdf1d57bf4765e26
```

Mặc định đích `.dvc/cache/files/md5/<2 ký tự>/<phần còn lại>`. Có thể chọn `--cache <thư-mục-cache>`. Chọn riêng từng owner khi cần nhiều object. Cache có file khác MD5 sẽ được giữ và báo lỗi; lỗi tải/hash không xuất bản object. Sau đó dùng `dvc checkout <manifest>` khi đủ object cần thiết; không `dvc push` nguyên build đã nghỉ để tránh đẩy lại checkpoint vừa dọn.
