# Chỉ mục artifact / pull lại

Môi trường `nckh`; remote `gdrive` trên G:. [Backup](backup.md). Chạy từ root repo.
Dùng manifest canonical bên dưới; `papers/` chỉ facade runtime, bị `.dvcignore` loại khỏi scan.
Không `dvc add` workspace partial; giữ entry remote-only. Kiểm dung lượng trước pull.
Đổi layout chỉ đổi vị trí manifest và `outs.path`; MD5/size/nfiles và relpath bên trong payload giữ nguyên.
[Kiểm đồng nhất](dvc-relocation-verification-2026-10-14.json).

| Manifest | Output tương đối từ manifest | Files | GiB logical | Local | Khôi phục |
|---|---|---:|---:|---|---|
| `archive/methods/anchor-function-residual/builds/anchor-residual/input.dvc` | `artifacts/input` | 222 | 0.298 | remote-only | `dvc pull archive/methods/anchor-function-residual/builds/anchor-residual/input.dvc` |
| `archive/methods/anchor-function-residual/builds/anchor-residual/kaggle.dvc` | `artifacts/kaggle` | 2798 | 34.855 | remote-only | `dvc pull archive/methods/anchor-function-residual/builds/anchor-residual/kaggle.dvc` |
| `archive/methods/anchor-function-residual/builds/anchor-residual/resume_reference.dvc` | `artifacts/resume_reference` | 153 | 1.811 | remote-only | `dvc pull archive/methods/anchor-function-residual/builds/anchor-residual/resume_reference.dvc` |
| `archive/methods/anchor-function-residual/builds/anchor-residual/smoke.dvc` | `artifacts/smoke` | 684 | 7.825 | remote-only | `dvc pull archive/methods/anchor-function-residual/builds/anchor-residual/smoke.dvc` |
| `archive/methods/contact-fragmentation/builds/fragmented-snapshots/kaggle.dvc` | `artifacts/kaggle` | 52 | 0.001 | remote-only | `dvc pull archive/methods/contact-fragmentation/builds/fragmented-snapshots/kaggle.dvc` |
| `archive/methods/contact-fragmentation/builds/fragmented-snapshots/runs.dvc` | `artifacts/runs` | 44 | 0.001 | remote-only | `dvc pull archive/methods/contact-fragmentation/builds/fragmented-snapshots/runs.dvc` |
| `archive/methods/contact-fragmentation/builds/fragmented-snapshots/source_audit.dvc` | `artifacts/source_audit` | 32 | 0.000 | remote-only | `dvc pull archive/methods/contact-fragmentation/builds/fragmented-snapshots/source_audit.dvc` |
| `archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/input.dvc` | `artifacts/input` | 204 | 0.298 | remote-only | `dvc pull archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/input.dvc` |
| `archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/kaggle.dvc` | `artifacts/kaggle` | 1618 | 13.132 | remote-only | `dvc pull archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/kaggle.dvc` |
| `archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/resume_reference.dvc` | `artifacts/resume_reference` | 60 | 0.394 | remote-only | `dvc pull archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/resume_reference.dvc` |
| `archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/smoke.dvc` | `artifacts/smoke` | 150 | 0.981 | remote-only | `dvc pull archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/smoke.dvc` |
| `archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/input.dvc` | `artifacts/input` | 222 | 0.298 | remote-only | `dvc pull archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/input.dvc` |
| `archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/kaggle.dvc` | `artifacts/kaggle` | 568 | 4.213 | remote-only | `dvc pull archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/kaggle.dvc` |
| `archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/resume_reference.dvc` | `artifacts/resume_reference` | 102 | 1.195 | remote-only | `dvc pull archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/resume_reference.dvc` |
| `archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/smoke.dvc` | `artifacts/smoke` | 246 | 2.787 | remote-only | `dvc pull archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/smoke.dvc` |
| `archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/input.dvc` | `artifacts/input` | 210 | 0.298 | remote-only | `dvc pull archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/input.dvc` |
| `archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/kaggle.dvc` | `artifacts/kaggle` | 1242 | 19.010 | remote-only | `dvc pull archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/kaggle.dvc` |
| `archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/resume_reference.dvc` | `artifacts/resume_reference` | 51 | 0.618 | remote-only | `dvc pull archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/resume_reference.dvc` |
| `archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/smoke.dvc` | `artifacts/smoke` | 111 | 1.422 | remote-only | `dvc pull archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/smoke.dvc` |
| `archive/methods/gaussian-sampling/builds/support-mala/input.dvc` | `artifacts/input` | 376 | 0.595 | remote-only | `dvc pull archive/methods/gaussian-sampling/builds/support-mala/input.dvc` |
| `archive/methods/gaussian-sampling/builds/support-mala/kaggle.dvc` | `artifacts/kaggle` | 1784 | 18.371 | remote-only | `dvc pull archive/methods/gaussian-sampling/builds/support-mala/kaggle.dvc` |
| `archive/methods/gaussian-sampling/builds/support-mala/resume_reference.dvc` | `artifacts/resume_reference` | 89 | 0.845 | remote-only | `dvc pull archive/methods/gaussian-sampling/builds/support-mala/resume_reference.dvc` |
| `archive/methods/gaussian-sampling/builds/support-mala/runs.dvc` | `artifacts/runs` | 0 | 0.000 | remote-only | `dvc pull archive/methods/gaussian-sampling/builds/support-mala/runs.dvc` |
| `archive/methods/gaussian-sampling/builds/support-mala/smoke.dvc` | `artifacts/smoke` | 243 | 2.241 | remote-only | `dvc pull archive/methods/gaussian-sampling/builds/support-mala/smoke.dvc` |
| `archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/input.dvc` | `artifacts/input` | 396 | 0.595 | remote-only | `dvc pull archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/input.dvc` |
| `archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/kaggle.dvc` | `artifacts/kaggle` | 1666 | 13.091 | remote-only | `dvc pull archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/kaggle.dvc` |
| `archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/resume_reference.dvc` | `artifacts/resume_reference` | 60 | 0.408 | remote-only | `dvc pull archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/resume_reference.dvc` |
| `archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/smoke.dvc` | `artifacts/smoke` | 240 | 1.629 | remote-only | `dvc pull archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/smoke.dvc` |
| `archive/methods/proxy-compression/builds/proxy-carriers/input.dvc` | `artifacts/input` | 534 | 0.892 | remote-only | `dvc pull archive/methods/proxy-compression/builds/proxy-carriers/input.dvc` |
| `archive/methods/proxy-compression/builds/proxy-carriers/kaggle.dvc` | `artifacts/kaggle` | 4556 | 125.595 | remote-only | `dvc pull archive/methods/proxy-compression/builds/proxy-carriers/kaggle.dvc` |
| `archive/methods/proxy-compression/builds/proxy-carriers/resume_reference.dvc` | `artifacts/resume_reference` | 216 | 4.434 | remote-only | `dvc pull archive/methods/proxy-compression/builds/proxy-carriers/resume_reference.dvc` |
| `archive/methods/proxy-compression/builds/proxy-carriers/runs.dvc` | `artifacts/runs` | 0 | 0.000 | remote-only | `dvc pull archive/methods/proxy-compression/builds/proxy-carriers/runs.dvc` |
| `archive/methods/proxy-compression/builds/proxy-carriers/smoke.dvc` | `artifacts/smoke` | 888 | 20.426 | remote-only | `dvc pull archive/methods/proxy-compression/builds/proxy-carriers/smoke.dvc` |
| `archive/methods/shared-baselines/builds/pilot-baselines/fixture.dvc` | `artifacts/fixture` | 67 | 0.149 | remote-only | `dvc pull archive/methods/shared-baselines/builds/pilot-baselines/fixture.dvc` |
| `archive/methods/shared-baselines/builds/pilot-baselines/runs.dvc` | `artifacts/runs` | 32 | 0.002 | remote-only | `dvc pull archive/methods/shared-baselines/builds/pilot-baselines/runs.dvc` |
| `archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/input.dvc` | `artifacts/input` | 170 | 0.297 | remote-only | `dvc pull archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/input.dvc` |
| `archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/kaggle.dvc` | `artifacts/kaggle` | 2422 | 29.499 | remote-only | `dvc pull archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/kaggle.dvc` |
| `archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/resume_reference.dvc` | `artifacts/resume_reference` | 108 | 1.150 | remote-only | `dvc pull archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/resume_reference.dvc` |
| `archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/runs.dvc` | `artifacts/runs` | 0 | 0.000 | partial; không add | `dvc pull archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/runs.dvc` |
| `archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/smoke.dvc` | `artifacts/smoke` | 612 | 6.392 | remote-only | `dvc pull archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/smoke.dvc` |
| `archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/input.dvc` | `artifacts/input` | 162 | 0.297 | remote-only | `dvc pull archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/input.dvc` |
| `archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/kaggle.dvc` | `artifacts/kaggle` | 622 | 7.450 | remote-only | `dvc pull archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/kaggle.dvc` |
| `archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/resume_reference.dvc` | `artifacts/resume_reference` | 13 | 0.093 | remote-only | `dvc pull archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/resume_reference.dvc` |
| `archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/runs.dvc` | `artifacts/runs` | 0 | 0.000 | remote-only | `dvc pull archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/runs.dvc` |
| `archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/smoke.dvc` | `artifacts/smoke` | 79 | 0.558 | remote-only | `dvc pull archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/smoke.dvc` |
| `archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/input.dvc` | `artifacts/input` | 291 | 0.446 | remote-only | `dvc pull archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/input.dvc` |
| `archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/kaggle.dvc` | `artifacts/kaggle` | 3802 | 41.338 | remote-only | `dvc pull archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/kaggle.dvc` |
| `archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/resume_reference.dvc` | `artifacts/resume_reference` | 135 | 1.253 | remote-only | `dvc pull archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/resume_reference.dvc` |
| `archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/runs.dvc` | `artifacts/runs` | 0 | 0.000 | remote-only | `dvc pull archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/runs.dvc` |
| `archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/smoke.dvc` | `artifacts/smoke` | 711 | 6.368 | remote-only | `dvc pull archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/smoke.dvc` |
| `knowledge/ckpt_round_005.pt.dvc` | `ckpt_round_005.pt` | 1 | 0.005 | remote-only | `dvc pull knowledge/ckpt_round_005.pt.dvc` |
| `methods/boundary-cell-counts/builds/boundary-cell-pilot/controller-fixture.dvc` | `artifacts/controller-fixture` | 76 | 0.000 | remote-only | `dvc pull methods/boundary-cell-counts/builds/boundary-cell-pilot/controller-fixture.dvc` |
| `methods/boundary-cell-counts/builds/boundary-cell-pilot/input.dvc` | `artifacts/input` | 230 | 0.298 | remote-only | `dvc pull methods/boundary-cell-counts/builds/boundary-cell-pilot/input.dvc` |
| `methods/boundary-cell-counts/builds/boundary-cell-pilot/kaggle.dvc` | `artifacts/kaggle` | 338 | 1.155 | partial; không add | `dvc pull methods/boundary-cell-counts/builds/boundary-cell-pilot/kaggle.dvc` |
| `methods/boundary-cell-counts/builds/boundary-cell-pilot/smoke.dvc` | `artifacts/smoke` | 210 | 2.040 | remote-only | `dvc pull methods/boundary-cell-counts/builds/boundary-cell-pilot/smoke.dvc` |
| `methods/gaussian-proxy-hybrid/builds/hybrid-full/diagnostics.dvc` | `artifacts/diagnostics` | 20 | 0.001 | remote-only | `dvc pull methods/gaussian-proxy-hybrid/builds/hybrid-full/diagnostics.dvc` |
| `methods/gaussian-proxy-hybrid/builds/hybrid-full/input.dvc` | `artifacts/input` | 212 | 0.001 | remote-only | `dvc pull methods/gaussian-proxy-hybrid/builds/hybrid-full/input.dvc` |
| `methods/gaussian-proxy-hybrid/builds/hybrid-full/kaggle.dvc` | `artifacts/kaggle` | 384 | 2.640 | partial; không add | `dvc pull methods/gaussian-proxy-hybrid/builds/hybrid-full/kaggle.dvc` |
| `methods/gaussian-proxy-hybrid/builds/hybrid-full/resume_reference.dvc` | `artifacts/resume_reference` | 81 | 1.012 | remote-only | `dvc pull methods/gaussian-proxy-hybrid/builds/hybrid-full/resume_reference.dvc` |
| `methods/gaussian-proxy-hybrid/builds/hybrid-full/smoke.dvc` | `artifacts/smoke` | 111 | 1.420 | remote-only | `dvc pull methods/gaussian-proxy-hybrid/builds/hybrid-full/smoke.dvc` |
| `methods/shared-baselines/builds/federated-baselines-full/input.dvc` | `artifacts/input` | 200 | 0.001 | remote-only | `dvc pull methods/shared-baselines/builds/federated-baselines-full/input.dvc` |
| `methods/shared-baselines/builds/federated-baselines-full/kaggle.dvc` | `artifacts/kaggle` | 968 | 8.641 | partial; không add | `dvc pull methods/shared-baselines/builds/federated-baselines-full/kaggle.dvc` |
| `methods/shared-baselines/builds/federated-baselines-full/mount-output.dvc` | `artifacts/mount-output` | 18 | 0.000 | remote-only | `dvc pull methods/shared-baselines/builds/federated-baselines-full/mount-output.dvc` |
| `methods/shared-baselines/builds/federated-baselines-full/resume_reference.dvc` | `artifacts/resume_reference` | 63 | 0.613 | remote-only | `dvc pull methods/shared-baselines/builds/federated-baselines-full/resume_reference.dvc` |
| `methods/shared-baselines/builds/federated-baselines-full/smoke.dvc` | `artifacts/smoke` | 221 | 1.224 | remote-only | `dvc pull methods/shared-baselines/builds/federated-baselines-full/smoke.dvc` |

Fresh checkout: `python3 scripts/restore_compat_aliases.py`; tạo facade source, không pull artifact.
Sau `dvc pull` target cần thiết, khôi phục alias run canonical từ [map](artifact-aliases.json):

```bash
python3 - <<'PYALIAS'
from pathlib import Path
import json, os
for a in json.loads(Path('docs/operations/artifact-aliases.json').read_text())['aliases']:
    p, target = Path(a['path']), Path(a['target'])
    if target.exists() and not p.exists() and not p.is_symlink():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.symlink_to(os.path.relpath(target, p.parent), target_is_directory=True)
PYALIAS
```

Map cũ trong backup-delta giữ nguyên làm bằng chứng lịch sử; không dùng trực tiếp với layout mới.
Một số `runs.dvc` snapshot rỗng lịch sử: cần pull `kaggle.dvc` cùng build rồi tạo run alias, không dùng snapshot rỗng thay output Kaggle.
Metadata chọn lọc: `python scripts/restore_dvc_metadata.py <canonical.dvc> --proof <proof.json>`;
sau restore vẫn partial, không gọi `dvc add`. Private_push/runtimecache không backup.
