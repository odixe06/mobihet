"""Locked train->val feasibility probe. No official-test reads or FL launch."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
from scipy.special import ndtr, ndtri

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "mobihet-features/data/v4-scaled"
OUT = ROOT / "runs/pair-generator"
PAIRS = [("msg_acl", "msg_abs_acl"), ("msg_spd", "rel_spd"), ("msg_spd", "rel_spd_diff"),
         ("rel_bearing_sin", "rel_bearing_cos"), ("ses_dt", "ses_dt_send"), ("ses_dt", "ses_dt_skew"),
         ("ses_implied_spd", "ses_spd_resid"), ("ses_spd_resid", "win_spd_resid_mean5"),
         ("win_spd_resid_mean5", "win_spd_resid_max5"), ("win_pos_pred_mean5", "win_pos_pred_max5"),
         ("rel_dist", "nbr_min_dist"), ("msg_spd", "p4_std_msg_spd")]
BINS, FIT_N, EVAL_N = 16, 32768, 16384


def bins(x, n=16):
    return np.clip(((x + 4) * (n / 8)).astype(np.int64), 0, n - 1)


def hist(code, size, weights=None):
    h = np.bincount(code, weights=weights, minlength=size).astype(np.float64)
    return h / h.sum()


def codes(x, pairs, singleton=True):
    b = bins(x)
    result = [(b[:, i] * BINS + b[:, j], BINS * BINS) for i, j in pairs]
    if singleton:
        result += [(b[:, i], BINS) for i in range(x.shape[1])]
    return result


def fit_potentials(real, synthetic, pairs, sweeps=20):
    rc, sc = codes(real, pairs), codes(synthetic, pairs)
    targets = [hist(c, n) for c, n in rc]
    potentials = [np.zeros(n) for _, n in sc]
    logw = np.zeros(len(synthetic))
    for _ in range(sweeps):
        for k, (code, size) in enumerate(sc):
            w = np.exp(logw - logw.max()); w /= w.sum()
            current = hist(code, size, w)
            # Fixed regularization. Impossible cells remain measured, never hidden by target projection.
            update = .5 * np.log(np.clip((targets[k] + 1e-6) / (current + 1e-6), .1, 10.))
            new = np.clip(potentials[k] + update, -6., 6.)
            logw += (new - potentials[k])[code]
            potentials[k] = new
    return potentials, targets


def evaluate(real, synthetic, pairs, potentials=None):
    rc, sc = codes(real, pairs), codes(synthetic, pairs)
    logw = np.zeros(len(synthetic))
    if potentials is not None:
        for potential, (code, _) in zip(potentials, sc):
            logw += potential[code]
    w = np.exp(logw - logw.max()); w /= w.sum()
    tv = [.5 * np.abs(hist(a, n) - hist(b, n, w)).sum() for (a, n), (b, _) in zip(rc, sc)]
    missing = [hist(a, n)[hist(b, n) == 0].sum() for (a, n), (b, _) in zip(rc, sc)]
    return {"ess_fraction": float(1 / (w @ w) / len(w)), "pair_tv": float(np.mean(tv[:len(pairs)])),
            "singleton_tv": float(np.mean(tv[len(pairs):])), "unsupported_pair_mass_max": float(max(missing[:len(pairs)]))}, w


def copula(real, rng, n):
    d, nb = real.shape[1], 256
    b = bins(real, nb)
    mass = np.stack([np.bincount(b[:, j], minlength=nb) for j in range(d)]).astype(np.float64)
    mass /= mass.sum(1, keepdims=True)
    cdf = mass.cumsum(1)
    score = ndtri(np.clip(cdf - mass / 2, 1e-4, 1 - 1e-4)).astype(np.float32)
    z = score[np.arange(d), b].astype(np.float64)
    mu = z.mean(0)
    centered = z - mu
    cov = centered.T @ centered / len(z) + np.eye(d) * 1e-4
    chol = np.linalg.cholesky(cov).astype(np.float32).astype(np.float16).astype(np.float32)
    mu = mu.astype(np.float32).astype(np.float16).astype(np.float32)
    cdf = np.rint(cdf * 65535) / 65535
    cdf[:, -1] = 1.
    mass = np.diff(cdf, axis=1, prepend=np.zeros((d, 1)))
    # CPU approximation to Core's compact sampler; not exact torch/CUDA replay.
    u = np.clip(ndtr(rng.standard_normal((n, d)).astype(np.float32) @ chol.T + mu), 1e-6, 1-1e-6)
    result = np.empty((n, d), np.float32)
    for j in range(d):
        ix = np.searchsorted(cdf[j], u[:, j]).clip(0, nb-1)
        fraction = (u[:, j] - (cdf[j, ix] - mass[j, ix])) / np.maximum(mass[j, ix], 1e-12)
        result[:, j] = -4 + (ix + fraction) * (8 / nb)
    assert np.isfinite(result).all()
    return result


def load():
    order = json.loads((ROOT / "knowledge/input_order.json").read_text())["inputs"]
    rng = np.random.default_rng(6006)
    train, val, records = [], [], []
    for folder in sorted((DATA / "train").glob("scenario=*/world=*")):
        files = sorted(folder.glob("*.parquet"))
        path = files[len(files) // 2]  # one predetermined file per scenario/world, all classes covered
        table = pq.ParquetFile(path).read(columns=order + ["fold", "label"]).to_pandas()
        fold = table["fold"].astype(str).to_numpy()
        records.append({"path": str(path), "bytes": path.stat().st_size})
        for name, dest, cap in (("train", train, 4096), ("val", val, 2048)):
            ii = np.flatnonzero(fold == name)
            ii = rng.choice(ii, min(cap, len(ii)), replace=False)
            dest.append((table.iloc[ii][order].to_numpy(np.float32), table.iloc[ii]["label"].to_numpy(np.int64)))
    return order, [(np.concatenate([v[0] for v in group]), np.concatenate([v[1] for v in group])) for group in (train, val)], records


def main():
    OUT.mkdir(exist_ok=True)
    source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plan = {"source_sha256": source, "pairs": PAIRS, "bins": BINS, "fit_candidates_per_class": FIT_N,
            "independent_eval_candidates_per_class": EVAL_N, "sweeps": 20, "damping": .5, "potential_clip": 6,
            "gates": {"mean_pair_tv_relative_improvement": .20, "singleton_tv_max_increase": .02,
                      "min_class_ess_fraction": .05, "tstr_macro_gain": .003},
            "scope": "bounded CPU train->val; no official test, no production sampler or 2-rank proof"}
    lock = OUT / "plan.json"
    if lock.exists():
        assert json.loads(lock.read_text()) == json.loads(json.dumps(plan)), "Plan changed; preserve old evidence"
    else:
        lock.write_text(json.dumps(plan, indent=2) + "\n")
    start = time.time()
    order, ((xt, yt), (xv, yv)), records = load()
    pairs = [(order.index(a), order.index(b)) for a, b in PAIRS]
    print("PAIR_DATA", xt.shape, xv.shape, flush=True)
    result, base_x, calibrated_x, synth_y = [], [], [], []
    for c in range(16):
        real, val = xt[yt == c], xv[yv == c]
        assert len(real) >= 256 and len(val) >= 128, (c, len(real), len(val))
        rng = np.random.default_rng(7000 + c)
        samples = copula(real, rng, FIT_N + EVAL_N)
        potential, _ = fit_potentials(real, samples[:FIT_N], pairs)
        baseline, _ = evaluate(val, samples[FIT_N:], pairs)
        corrected, w = evaluate(val, samples[FIT_N:], pairs, potential)
        row = {"class": c, "train_rows": len(real), "val_rows": len(val), "baseline": baseline, "corrected": corrected}
        result.append(row)
        take = rng.choice(EVAL_N, 4096, replace=False)
        base_x.append(samples[FIT_N:][take])
        calibrated_x.append(samples[FIT_N:][rng.choice(EVAL_N, 4096, p=w)])
        synth_y.append(np.full(4096, c))
        print("PAIR_CLASS", json.dumps(row), flush=True)
    before = np.mean([r["baseline"]["pair_tv"] for r in result])
    after = np.mean([r["corrected"]["pair_tv"] for r in result])
    gates = {"pair_improvement": bool(after <= .8 * before),
             "singleton": bool(all(r["corrected"]["singleton_tv"] <= r["baseline"]["singleton_tv"] + .02 for r in result)),
             "ess": bool(all(r["corrected"]["ess_fraction"] >= .05 for r in result))}
    report = {"classes": result, "gates": gates, "pair_tv_before": float(before), "pair_tv_after": float(after),
              "input_files": records, "train_rows": len(yt), "val_rows": len(yv), "plan_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()}
    if all(gates.values()):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import f1_score
        scores = {}
        for name, x, y in (("real", xt, yt), ("copula", np.concatenate(base_x), np.concatenate(synth_y)),
                           ("corrected", np.concatenate(calibrated_x), np.concatenate(synth_y))):
            model = HistGradientBoostingClassifier(max_iter=40, max_leaf_nodes=31, early_stopping=False, random_state=42)
            pred = model.fit(x, y).predict(xv)
            scores[name] = float(f1_score(yv, pred, labels=list(range(16)), average="macro", zero_division=0))
            print("PAIR_TSTR", name, scores[name], flush=True)
        report["tstr"] = scores
        gates["tstr"] = scores["corrected"] >= scores["copula"] + .003
    report.update(pass_feasibility=all(gates.values()), seconds=time.time()-start)
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == source
    (OUT / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PAIR_PROBE_DONE", json.dumps(gates), flush=True)


if __name__ == "__main__":
    main()
