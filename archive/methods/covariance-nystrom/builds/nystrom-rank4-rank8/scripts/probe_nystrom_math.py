#!/usr/bin/env python3
"""Bounded CPU Nyström covariance probe; encoded moments, no models/record export."""
import hashlib
import json
from pathlib import Path
import struct
import zlib

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])
OUT = ROOT / 'papers/nystrom/math-probe.json'
HEADER = struct.Struct('<4sBBBBIIIIII')
D, CLASSES = 66, 16
RANKS = (4, 8)
OMEGA_SEED = 7601
SAMPLE_SEED = 7602
EPS = 1e-4
QUANTIZATION_GATE = .01


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def packet(kind, counts, values, context, half=False):
    assert len(context) == 16 and kind in (1, 2)
    a = np.asarray(values, dtype='<f2' if half else '<f4')
    assert np.isfinite(a).all()
    flag = int(counts is not None)
    body = context + (np.asarray(counts, dtype='<u8').tobytes() if flag else b'') + a.tobytes()
    return HEADER.pack(b'NYS1', 1, kind, 2 if half else 4, flag, 0, 0,
                       a.size, 0, zlib.crc32(body), len(body)) + body


def decode(blob, context, kind, count_values, expected_size):
    assert len(blob) >= HEADER.size
    magic, schema, got_kind, dtype, flag, base, version, n, reserved, crc, size = HEADER.unpack_from(blob)
    assert magic == b'NYS1' and schema == 1 and got_kind == kind
    assert dtype == (4 if kind == 1 else 2) and flag == int(count_values)
    assert base == version == reserved == 0 and n == expected_size
    body = blob[HEADER.size:]
    assert len(body) == size and zlib.crc32(body) == crc
    assert body[:16] == context, 'wrong schema/cohort/Omega context'
    offset = 16
    counts = None
    if flag:
        counts = np.frombuffer(body, dtype='<u8', count=CLASSES, offset=offset).copy()
        offset += 8 * CLASSES
    values = np.frombuffer(body, dtype='<f4' if dtype == 4 else '<f2', offset=offset).astype(np.float64)
    assert len(values) == n and np.isfinite(values).all()
    return counts, values


def rejected(function):
    try:
        function()
    except AssertionError:
        return True
    return False


def upper(matrix):
    return matrix[np.triu_indices(len(matrix))]


def from_upper(values, rank):
    matrix = np.zeros((rank, rank), dtype=np.float64)
    i, j = np.triu_indices(rank)
    matrix[i, j] = values
    matrix[j, i] = values
    return matrix


def summarize(values):
    return dict(minimum=float(np.min(values)), median=float(np.median(values)),
                maximum=float(np.max(values)), mean=float(np.mean(values)))


def main():
    fx = ROOT / 'papers/pilot/fixture'
    manifest = json.loads((fx / 'manifest.json').read_text())
    assert len(manifest['clients']) == 20 and len(manifest['features']) == D
    q, r = np.linalg.qr(np.random.default_rng(OMEGA_SEED).standard_normal((D, 8)), mode='reduced')
    omega8 = q * np.where(np.diag(r) >= 0, 1.0, -1.0)
    assert np.linalg.norm(omega8.T @ omega8 - np.eye(8)) < 1e-12
    batches = []
    local_counts, local_sums, local_diagonal, local_full = [], [], [], []
    contributors = np.zeros(CLASSES, dtype=np.int64)
    for cid in range(20):
        x = np.load(fx / f'train_{cid}_X.npy', mmap_mode='r')
        y = np.load(fx / f'train_{cid}_y.npy', mmap_mode='r')
        indices = np.random.default_rng(SAMPLE_SEED + cid).choice(len(y), min(2048, len(y)), replace=False)
        values, labels = np.asarray(x[indices], dtype=np.float64), np.asarray(y[indices], dtype=np.int64)
        assert np.isfinite(values).all()
        n = np.bincount(labels, minlength=CLASSES)
        s = np.zeros((CLASSES, D)); d = np.zeros((CLASSES, D)); full = np.zeros((CLASSES, D, D))
        for c in range(CLASSES):
            selected = values[labels == c]
            s[c] = selected.sum(0); d[c] = np.square(selected).sum(0)
            full[c] = selected.T @ selected
        contributors += n > 0
        batches.append((values, labels))
        local_counts.append(n); local_sums.append(s); local_diagonal.append(d); local_full.append(full)
        del x, y
    n_reference = np.sum(local_counts, axis=0)
    s_reference = np.sum(local_sums, axis=0)
    full_reference = np.sum(local_full, axis=0)
    mu_reference = s_reference / n_reference[:, None]
    covariance = full_reference / n_reference[:, None, None] - mu_reference[:, :, None] * mu_reference[:, None, :]
    covariance = (covariance + covariance.transpose(0, 2, 1)) / 2
    assert (n_reference > 0).all() and (contributors >= 5).all()
    full_size = D + D * (D + 1) // 2
    full_context = hashlib.sha256(b'nystrom-full-fp32-control-v1' + bytes.fromhex(sha(fx/'manifest.json'))).digest()[:16]
    full_aggregate = np.zeros((CLASSES, full_size), dtype=np.float64)
    full_counts = np.zeros(CLASSES, dtype=np.uint64)
    for cid in range(20):
        entries = np.concatenate((local_sums[cid], np.stack([upper(m) for m in local_full[cid]])), axis=1)
        full_upload = packet(1, local_counts[cid], entries, full_context)
        n, values = decode(full_upload, full_context, 1, True, CLASSES * full_size)
        full_counts += n; full_aggregate += values.reshape(CLASSES, full_size)
    assert np.array_equal(full_counts, n_reference)
    full_encoded_covariance = np.zeros_like(covariance)
    full_factors = []
    for c in range(CLASSES):
        mean = full_aggregate[c, :D] / full_counts[c]
        q = from_upper(full_aggregate[c, D:], D)
        cov = q / full_counts[c] - np.outer(mean, mean)
        cov = (cov + cov.T) / 2
        full_encoded_covariance[c] = cov
        chol = np.linalg.cholesky(cov + EPS * np.eye(D))
        full_factors.append(np.concatenate((mean, chol[np.tril_indices(D)])))
    full_download = packet(2, None, np.stack(full_factors), full_context, half=True)
    _, full_decoded = decode(full_download, full_context, 2, False, CLASSES * full_size)
    assert np.isfinite(full_decoded).all()
    full_wire_error = [float(np.linalg.norm(full_encoded_covariance[c] - covariance[c]) /
                             max(np.linalg.norm(covariance[c]), EPS)) for c in range(CLASSES)]
    results = []
    for rank in RANKS:
        omega = omega8[:, :rank]
        tri = rank * (rank + 1) // 2
        size_per_class = 2 * D + D * rank + tri
        identity = dict(schema='nystrom-full-input-v1', rank=rank, features=manifest['features'],
                        classes=manifest['classes'], cohort=list(range(20)), version=0,
                        omega_seed=OMEGA_SEED, omega_sha256=hashlib.sha256(omega.tobytes()).hexdigest(),
                        fixture_sha256=sha(fx / 'manifest.json'), epsilon=EPS,
                        reconstruction='W eigenvalues floored1e-4; residual variance floored1e-4')
        context = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).digest()[:16]
        aggregate = np.zeros((CLASSES, size_per_class), dtype=np.float64)
        counts = np.zeros(CLASSES, dtype=np.uint64)
        wire = None
        for cid, (values, labels) in enumerate(batches):
            cblock = np.zeros((CLASSES, D, rank)); wblock = np.zeros((CLASSES, rank, rank))
            for c in range(CLASSES):
                selected = values[labels == c]
                projected = selected @ omega
                cblock[c] = selected.T @ projected
                wblock[c] = projected.T @ projected
            entries = np.concatenate((local_sums[cid], local_diagonal[cid], cblock.reshape(CLASSES, -1),
                                      np.stack([upper(wblock[c]) for c in range(CLASSES)])), axis=1)
            wire = packet(1, local_counts[cid], entries, context)
            count_decoded, values_decoded = decode(wire, context, 1, True, CLASSES * size_per_class)
            aggregate += values_decoded.reshape(CLASSES, size_per_class)
            counts += count_decoded
        assert np.array_equal(counts, n_reference)
        wrong_context = bytes([context[0] ^ 1]) + context[1:]
        corrupted = wire[:-1] + bytes([wire[-1] ^ 1])
        assert rejected(lambda: decode(wire, wrong_context, 1, True, CLASSES * size_per_class))
        assert rejected(lambda: decode(corrupted, context, 1, True, CLASSES * size_per_class))
        assert rejected(lambda: decode(wire[:-1], context, 1, True, CLASSES * size_per_class))
        by_class, down_values = [], []
        for c in range(CLASSES):
            s, d = aggregate[c, :D], aggregate[c, D:2*D]
            c_raw = aggregate[c, 2*D:2*D+D*rank].reshape(D, rank)
            w_raw = from_upper(aggregate[c, 2*D+D*rank:], rank)
            mean = s / counts[c]
            mu_o = mean @ omega
            c_centered = c_raw / counts[c] - np.outer(mean, mu_o)
            w_centered = w_raw / counts[c] - np.outer(mu_o, mu_o)
            w_centered = (w_centered + w_centered.T) / 2
            eigenvalues, eigenvectors = np.linalg.eigh(w_centered)
            low = (c_centered @ eigenvectors) / np.sqrt(np.maximum(eigenvalues, EPS))[None, :]
            var = d / counts[c] - np.square(mean)
            low_diagonal = np.square(low).sum(1)
            residual_raw = var - low_diagonal
            residual = np.maximum(residual_raw, EPS)
            sd = np.sqrt(residual)
            approximate = low @ low.T + np.diag(residual)
            assert np.isfinite(approximate).all()
            down_values.append(np.concatenate((mean, low.flatten(), sd)))
            denominator = max(np.linalg.norm(covariance[c]), EPS)
            exact_eigenvalues, exact_eigenvectors = np.linalg.eigh(full_encoded_covariance[c])
            best = exact_eigenvectors[:, -rank:] * np.sqrt(np.maximum(exact_eigenvalues[-rank:], 0))[None, :]
            best_residual = np.maximum(np.diag(full_encoded_covariance[c]) - np.square(best).sum(1), EPS)
            exact_rank = best @ best.T + np.diag(best_residual)
            diag = np.diag(np.maximum(np.diag(covariance[c]), EPS))
            by_class.append(dict(
                class_name=manifest['classes'][c], rows=int(counts[c]), contributors=int(contributors[c]),
                centered_w_minimum_eigenvalue=float(eigenvalues.min()),
                centered_w_floored_eigenvalues=int((eigenvalues < EPS).sum()),
                centered_w_condition_number_after_floor=float(np.maximum(eigenvalues, EPS).max() / np.maximum(eigenvalues, EPS).min()),
                independent_w_vs_omega_c_relative_error=float(np.linalg.norm(w_centered - omega.T @ c_centered) / max(np.linalg.norm(w_centered), EPS)),
                mu_fp32_wire_maximum_error=float(np.max(np.abs(mean - mu_reference[c]))),
                centered_c_fp32_wire_relative_error=float(np.linalg.norm(c_centered - covariance[c] @ omega) / max(np.linalg.norm(covariance[c] @ omega), EPS)),
                covariance_frobenius_relative_error=float(np.linalg.norm(approximate - covariance[c]) / denominator),
                exact_eigenrank_residual_frobenius_relative_error=float(np.linalg.norm(exact_rank - covariance[c]) / denominator),
                diagonal_frobenius_relative_error=float(np.linalg.norm(diag - covariance[c]) / denominator),
                residual_floor_coordinates=int((residual_raw < EPS).sum()),
                residual_negative_coordinates=int((residual_raw < 0).sum()),
                maximum_negative_residual_magnitude=float(max(-residual_raw.min(), 0)),
                maximum_variance_error_vs_full_sample=float(np.max(np.abs(np.diag(approximate) - np.diag(covariance[c])))),
                approximate_minimum_eigenvalue=float(np.linalg.eigvalsh(approximate).min()),
                approximate=approximate,
            ))
        down_values = np.stack(down_values)
        down = packet(2, None, down_values, context, half=True)
        _, downloaded = decode(down, context, 2, False, CLASSES * D * (rank + 2))
        downloaded = downloaded.reshape(CLASSES, D * (rank + 2))
        assert rejected(lambda: decode(down, wrong_context, 2, False, CLASSES * D * (rank + 2)))
        for c in range(CLASSES):
            mean = downloaded[c, :D]
            low = downloaded[c, D:D+D*rank].reshape(D, rank)
            sd = downloaded[c, D+D*rank:]
            decoded_cov = low @ low.T + np.diag(np.square(sd))
            reference_approximate = by_class[c].pop('approximate')
            error = np.linalg.norm(decoded_cov - reference_approximate) / max(np.linalg.norm(reference_approximate), EPS)
            minimum = float(np.linalg.eigvalsh(decoded_cov).min())
            rng = np.random.default_rng(7603 + rank * 16 + c)
            generated = mean + rng.standard_normal((1024, rank)) @ low.T + rng.standard_normal((1024, D)) * sd
            assert np.isfinite(generated).all() and error <= QUANTIZATION_GATE and minimum >= .5 * EPS
            by_class[c].update(
                fp16_factor_relative_covariance_error=float(error), decoded_covariance_minimum_eigenvalue=minimum,
                decoded_maximum_variance_error_vs_full_sample=float(np.max(np.abs(np.diag(decoded_cov) - np.diag(covariance[c])))),
                decoded_covariance_frobenius_relative_error=float(np.linalg.norm(decoded_cov - covariance[c]) / max(np.linalg.norm(covariance[c]), EPS)),
                generated_synthetic_points=1024, sampling_finite=True,
            )
        results.append(dict(
            rank=rank, sketch_width=rank, no_oversampling=True,
            public_omega_seed=OMEGA_SEED, public_omega_sha256=identity['omega_sha256'],
            orthogonality_error=float(np.linalg.norm(omega.T @ omega - np.eye(rank))),
            coefficients_per_class=size_per_class, wire_context_hex=context.hex(),
            actual_upload_packet_bytes_per_client=len(wire), actual_download_packet_bytes_per_client=len(down),
            paid_prior_bytes_per_client=200, actual_total_application_bytes_per_client=len(wire)+len(down)+200,
            wrong_context_rejected=True, corruption_crc_rejected=True, truncation_rejected=True,
            fp32_upload_roundtrip_and_fp64_aggregate=True, fp16_down_roundtrip=True,
            math_gate_pass=True, per_class=by_class,
            covariance_error_summary=summarize([row['decoded_covariance_frobenius_relative_error'] for row in by_class]),
            fp16_covariance_quantization_error_summary=summarize([row['fp16_factor_relative_covariance_error'] for row in by_class]),
            residual_floor_fraction=float(sum(row['residual_floor_coordinates'] for row in by_class)/(CLASSES*D)),
        ))
    payload = dict(
        probe='S1 additive covariance Nyström bounded CPU math; no training',
        rows=int(n_reference.sum()), clients=20, maximum_train_rows_per_client=2048,
        sample_seed_recipe=f'{SAMPLE_SEED}+cid; uniform without replacement; mmap train only',
        covariance_reference='Exact float64 covariance of same bounded sample, ddof0, no original fulltrain/holdout',
        epsilon=EPS, fp16_covariance_quantization_gate=QUANTIZATION_GATE,
        numeric_gates='Finite encoded/decoded packets; >=5contributors/class; decoded eigmin>=0.5e-4; factor quantization relativeFrobenius<=0.01; finite1024draws/class; CRC/context/truncation reject',
        matched_full_fp32_control=dict(
            upload_packet_bytes_per_client=len(full_upload),
            download_packet_bytes_per_client=len(full_download),
            prior_bytes_per_client=200,
            total_application_bytes_per_client=len(full_upload)+len(full_download)+200,
            upload_covariance_relative_error_vs_float64_sample=summarize(full_wire_error),
            full_cholesky_ridge=EPS, full_factor_fp16_finite=True,
            exact_eigenrank_reference='Computed from actual decoded fullFP32 moments, with same1e-4residualfloor; no learnedmodel',
        ),
        script_sha256=sha(Path(__file__)), fixture_manifest_sha256=sha(fx/'manifest.json'),
        source_sha256={'mh/pilot.py':sha(ROOT/'mh/pilot.py'),
                       'docs/research/de-xuat-bo-sung-2026-10-11.md':sha(ROOT/'docs/research/de-xuat-bo-sung-2026-10-11.md')},
        result=results,
        limits=[
            'Probe exports only aggregate error/count/byte metrics; no record/index/label vector/factor or learnedmodel.',
            'CRC/context are binding/error detection, not cryptographic authentication or SecureSum.',
            'Sums/C/W are serializedfp32 independently; exact PSD/Loewner identities need not survive wire rounding.',
            'The spectral floor and residual diagonal are an adaptation, not exact original paper pseudoinverse or theorem configuration.',
            'Nooversampling ranks4/8 lack strong paper relative-error guarantee.',
            'Covariance approximation error is diagnostic; no F1/R1/R2/DP acceptance from mathPASS.',
            'Fresh full/diag/eigenrank controls and utility gates must be locked before a Kaggle pilot.',
        ],
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(rows=payload['rows'], ranks=[dict(
        rank=row['rank'], math_gate_pass=row['math_gate_pass'],
        bytes=row['actual_total_application_bytes_per_client'],
        covariance_error=row['covariance_error_summary'],
        quantization_error=row['fp16_covariance_quantization_error_summary'],
        residual_floor_fraction=row['residual_floor_fraction'],
    ) for row in results])))


if __name__ == '__main__':
    main()
