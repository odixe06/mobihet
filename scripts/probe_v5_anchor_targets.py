#!/usr/bin/env python3
"""CPU-only aggregate diagnostics of four audited V5-H FL target tables."""
import configparser
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mh.v2_math import unpack

torch.set_num_threads(1)
OUTPUT = ROOT / 'papers/v5/anchor-target-diagnostics.json'


def digest(path, algorithm='sha256'):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def describe(values):
    values = values.double().flatten()
    return dict(mean=float(values.mean()), minimum=float(values.min()),
                p10=float(torch.quantile(values, .1)), median=float(values.median()),
                p90=float(torch.quantile(values, .9)), maximum=float(values.max()))


def entropy(probabilities):
    return -(probabilities * probabilities.clamp_min(1e-300).log()).sum(-1)


def tv(a, b):
    return float((a - b).abs().sum() / 2)


def main():
    config = configparser.ConfigParser()
    config.read(ROOT / '.dvc/config')
    remote = Path(config['\'remote "gdrive"\'']['url']) / 'files/md5'
    manifest = ROOT / 'papers/v5/kaggle.dvc'
    directory_hash = yaml.safe_load(manifest.read_text())['outs'][0]['md5']
    directory = remote / directory_hash[:2] / directory_hash[2:]
    assert digest(directory, 'md5') == directory_hash.removesuffix('.dir')
    entries = {item['relpath']: item for item in json.loads(directory.read_text())}
    names = json.loads((ROOT / 'knowledge/meta.json').read_text())['class_names']
    rows = []
    across_seeds = {}
    selected_bytes = 0
    for seed in (42, 43):
        kernel = f'v5h-s{seed}-20261012-v1'
        base = ROOT / 'papers/v5/kaggle' / kernel
        audit = json.loads((base / 'AUDITED.json').read_text())
        assert audit['family'].lower() == 'h' and audit['seed'] == seed and audit['complete']
        source_proof = {}
        for name in ('v5_anchor.py', 'v2_math.py', 'v2_methods.py', 'v2_run.py'):
            path = base / 'source' / name
            assert digest(path) == audit['source'][name]
            source_proof[name] = digest(path)
        assert digest(ROOT / 'mh/v2_math.py') == source_proof['v2_math.py']
        for size in ('small', 'medium'):
            arm = f'b_anchor_{size}_fl'
            run = base / 'runs' / f'{arm}_s{seed}'
            path = run / 'resume/round_020.pt'
            entry = entries[str(path.relative_to(ROOT / 'papers/v5/kaggle'))]
            assert digest(path, 'md5') == entry['md5']
            selected_bytes += path.stat().st_size
            marker = json.loads((run / 'complete/round_020.json').read_text())
            assert digest(path) == marker['resume_sha']
            identity = json.loads((run / 'config.json').read_text())
            assert any(item['arm'] == arm and item['complete'] and item['rounds'] == 20
                       for item in audit['audits'])
            saved = torch.load(path, map_location='cpu', weights_only=True)
            assert saved['round'] == 20 and saved['fingerprint'] == identity['fingerprint']
            algorithm = saved['algorithm']
            assert algorithm['diag']['target_mode'] == 'fl'
            m = 32 if size == 'small' else 512
            assert algorithm['M'] == m and algorithm['anchors'].shape == (m, 66)
            raw = unpack(algorithm['target_packet'])['values'].reshape(m, 16)
            decoded = raw / raw.sum(1, keepdim=True).clamp_min(1e-12)
            assert torch.equal(decoded, algorithm['target'])
            p = decoded.double()
            assert torch.isfinite(p).all() and (p >= 0).all()
            labels = algorithm['labels'].long()
            assert torch.equal(labels, torch.arange(m) % 16)
            q = p.mean(0)
            uniform = torch.full((16,), 1 / 16, dtype=torch.float64)
            prior = saved['ratio_pi'].double()
            prior = prior / prior.sum()
            coverage = algorithm['coverage_per_class'].double()
            normalized_entropy = entropy(p) / math.log(16)
            mean_kl_to_class_mean = entropy(q) - entropy(p).mean()
            variation_from_class_mean = (p - q).square().sum(1).mean()
            variation_from_uniform = (p - uniform).square().sum(1).mean()
            ordered = p.sort(1, descending=True).values
            own_label_probability = p[torch.arange(m), labels]
            label_ranks = (p > own_label_probability[:, None]).sum(1) + 1
            aggregate_per_class = []
            for c, name in enumerate(names):
                mask = labels == c
                aggregate_per_class.append(dict(
                    class_name=name, anchor_count=int(mask.sum()),
                    mean_target_mass=float(q[c]), decoded_train_prior=float(prior[c]),
                    train_coverage=float(coverage[c]),
                    mean_generating_label_probability=float(own_label_probability[mask].mean()),
                    argmax_alignment_fraction=float((p[mask].argmax(1) == c).double().mean()),
                ))
            final = marker['eval']['proxy']['group']['head']['mean_metrics']
            gaussian_run = base / 'runs' / f'b_anchor_{size}_gauss_s{seed}'
            gaussian_marker = json.loads((gaussian_run / 'complete/round_020.json').read_text())
            gaussian_final = gaussian_marker['eval']['proxy']['group']['head']['mean_metrics']
            row = dict(
                seed=seed, arm=arm, anchors=m, round=20,
                selected_bandwidth=float(algorithm['h']),
                queried_pilot_train_rows=int(algorithm['diag']['queried_real_rows']),
                coverage_minimum=float(coverage.min()),
                coverage_prior_weighted=float((coverage * prior).sum()),
                coverage_gate_pass=bool((coverage * prior).sum() >= .9 and coverage.min() >= .8),
                coverage_limits='Chosen bandwidth/coverage recorded for full pilot T only; rejected candidate h coverage is not saved, and original full43M train was not queried.',
                normalized_entropy=describe(normalized_entropy),
                maximum_probability=describe(ordered[:, 0]),
                top_two_margin=describe(ordered[:, 0] - ordered[:, 1]),
                total_variation_from_uniform=describe((p - uniform).abs().sum(1) / 2),
                kl_from_uniform_nats=describe(math.log(16) - entropy(p)),
                generating_label_probability=describe(own_label_probability),
                generating_label_rank=describe(label_ranks),
                argmax_gaussian_label_alignment=float((p.argmax(1) == labels).double().mean()),
                class_mean_target_entropy_normalized=float(entropy(q) / math.log(16)),
                mean_kl_to_class_mean_nats=float(mean_kl_to_class_mean),
                mean_kl_to_class_mean_normalized=float(mean_kl_to_class_mean / math.log(16)),
                varying_anchor_fraction_of_squared_uniform_deviation=float(variation_from_class_mean / variation_from_uniform),
                target_mean_vs_uniform_tv=tv(q, uniform),
                target_mean_vs_decoded_train_prior_tv=tv(q, prior),
                decoded_train_prior_vs_uniform_tv=tv(prior, uniform),
                mean_squared_deviation_from_uniform=float((p - uniform).square().sum(1).mean()),
                round20_mean_optimizer_soft_ce=float(marker['loss']),
                target_mean_entropy_nats=float(entropy(p).mean()),
                round20_soft_ce_minus_target_entropy_nats=float(marker['loss'] - entropy(p).mean()),
                optimizer_ce_limit='Average of applied soft-CE steps on fixed anchors in round20; gap near float32/AMP precision, not new final-model inference.',
                per_class_aggregates=aggregate_per_class,
                group_head_macro_f1=float(final['f1_macro']),
                gaussian_hard_label_group_head_macro_f1=float(gaussian_final['f1_macro']),
                difference_vs_gaussian_hard_labels=float(final['f1_macro'] - gaussian_final['f1_macro']),
                proof=dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                           dvc_md5=entry['md5'], sha256=digest(path),
                           marker_resume_sha_match=True, fingerprint=saved['fingerprint'],
                           source_hashes=source_proof,
                           target_packet_decode_exact=True,
                           prior_source='ratio_pi from aggregate decoded class counts; no smoothing'),
            )
            rows.append(row)
            across_seeds.setdefault(size, []).append(decoded.clone())
            del saved, algorithm
    assert selected_bytes <= 100_000_000
    agreement = {size: dict(
        decoded_targets_bitwise_equal=torch.equal(pair[0], pair[1]),
        maximum_absolute_target_difference=float((pair[0] - pair[1]).abs().max()),
    ) for size, pair in across_seeds.items()}
    result = dict(
        diagnostic='V5-H executed kernel-anchor target tables; retrospective aggregate only',
        script_sha256=digest(Path(__file__)), directory_manifest=directory_hash,
        directory_manifest_md5_pass=True, exactly_four_resume_files_restored=True,
        selected_resume_bytes=selected_bytes,
        seeds=[42, 43], sizes=['small', 'medium'], rows=rows,
        across_seed_target_agreement=agreement,
        read_real_records=False, inference_or_training=False, device='cpu',
        limits=[
            'Descriptive diagnostics after failed locked pilot; no h, target sharpening, head or round selection.',
            'High entropy supports weak target contrast hypothesis; does not isolate sole cause from finite-anchor representation, teacher training or OOD anchors.',
            'Targets divide kernel sums by each class count before row normalization; class balance is intentional, not necessarily a class-prior bug.',
            'Both seeds use same pilot data and fixed anchor RNG; equal targets are not independent replications of knowledge estimation.',
            'Generating Gaussian label is a synthetic control label, not true IDS label of an observed real point.',
            'Restore MD5/SHA/packet checks extend prior AUDITED evidence; no new full checkpoint/prediction/model-inference audit.',
            'No individual anchor table, record, real gradient, API key or signed URL in output.',
        ],
    )
    OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
    for row in rows:
        print(json.dumps({key: row[key] for key in (
            'arm', 'seed', 'selected_bandwidth', 'coverage_minimum',
            'coverage_prior_weighted', 'argmax_gaussian_label_alignment',
            'target_mean_vs_uniform_tv',
        )} | dict(entropy_mean=row['normalized_entropy']['mean'],
                  maxprob_mean=row['maximum_probability']['mean'])))
    print(json.dumps(dict(restored_resume_MiB=selected_bytes / (1 << 20),
                          across_seed_target_agreement=agreement)))


if __name__ == '__main__':
    main()
