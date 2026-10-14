"""Bounded train-only statistics probe; no SGD, official test or raw output."""
import hashlib
import json
from pathlib import Path
import numpy as np
from cells import C, PAIRS, count, targets, lookup, packets

ROOT = Path(__file__).resolve().parents[3]
FX = ROOT / "papers/pilot/fixture"


def diagnostic(prob, y, ok):
    pred = prob.argmax(axis=1)
    cm = np.bincount(y*C+pred, minlength=C*C).reshape(C, C)
    tp = np.diag(cm).astype(float)
    f1 = np.divide(2*tp, cm.sum(0)+cm.sum(1), out=np.zeros(C), where=cm.sum(0)+cm.sum(1)>0)
    nll = -np.log(np.maximum(prob[np.arange(len(y)), y], 1e-12))
    return dict(f1_macro=float(f1.mean()), f1_per_class=f1.tolist(),
                balanced_nll=float(np.mean([nll[y==c].mean() for c in range(C) if (y==c).any()])),
                entropy_normalized=float((-prob*np.log(np.maximum(prob, 1e-12))).sum(1).mean()/np.log(C)),
                coverage=float(ok.mean()),
                coverage_per_class=[float(ok[y==c].mean()) if (y==c).any() else None for c in range(C)],
                class_rows=np.bincount(y, minlength=C).tolist())


def main():
    manifest = json.loads((FX/"manifest.json").read_text())
    names = manifest["features"]
    provenance = dict(fixture_manifest_sha=hashlib.sha256((FX/"manifest.json").read_bytes()).hexdigest(),
                      source_sha={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob("*.py")},
                      note="Manifest referenced; full fixture file hashes not reverified by this bounded probe.")
    rng = np.random.default_rng(1729)
    own = {2: [], 4: []}
    for k in range(20):
        x = np.load(FX/f"train_{k}_X.npy", mmap_mode="r")
        y = np.load(FX/f"train_{k}_y.npy", mmap_mode="r")
        idx = rng.choice(len(y), min(2048, len(y)), replace=False)
        xx, yy = x[idx].astype(np.float32), y[idx].astype(np.int64)
        for p in own:
            own[p].append(count(xx, yy, names, p))
    gx = np.load(FX/"group_X.npy", mmap_mode="r")
    gy = np.load(FX/"group_y.npy", mmap_mode="r")
    idx = rng.choice(len(gy), min(4096, len(gy)), replace=False)
    gx, gy = gx[idx].astype(np.float32), gy[idx].astype(np.int64)
    results = {}
    for p, clients in own.items():
        ns = np.stack([item[0] for item in clients])
        n = np.stack([item[1] for item in clients]).sum(axis=0)
        pooled = ns.astype(np.uint64).sum(axis=0)
        contributors = (ns > 0).sum(axis=0)
        # Strict cell suppression if ANY observed class has fewer than 5 contributors.
        eligible = ~((pooled > 0) & (contributors < 5)).any(axis=-1)
        eligible &= pooled.sum(axis=-1) >= 32
        table, eligible = targets(pooled, n, eligible)
        prob, ok = lookup(gx, names, table, eligible)
        global_result = diagnostic(prob, gy, ok)
        own_results = []
        for counts, local_n in clients:
            local_t, _ = targets(counts, local_n, eligible)
            pp, oo = lookup(gx, names, local_t, eligible)
            own_results.append(diagnostic(pp, gy, oo))
        shuffled = table.copy()
        srng = np.random.default_rng(701)
        for r in range(p):
            cells = np.flatnonzero(eligible[r])
            shuffled[r, cells] = table[r, srng.permutation(cells)]
        sp, so = lookup(gx, names, shuffled, eligible)
        shuffle_result = diagnostic(sp, gy, so)
        identity = bytes.fromhex(provenance["fixture_manifest_sha"])[:16]
        up, down = packets(clients[0][0], clients[0][1], table, eligible, (n/n.sum()).astype(np.float32), identity)
        own_f1 = float(np.mean([d["f1_macro"] for d in own_results]))
        own_nll = float(np.mean([d["balanced_nll"] for d in own_results]))
        results[str(p)] = dict(pairs=PAIRS[:p], eligible_cells=int(eligible.sum()), total_cells=int(eligible.size),
            global_targets=global_result, own_mean_f1=own_f1, own_mean_balanced_nll=own_nll,
            shuffle_targets=shuffle_result, delta_f1_own=global_result["f1_macro"]-own_f1,
            delta_f1_shuffle=global_result["f1_macro"]-shuffle_result["f1_macro"],
            balanced_nll_gain_own=own_nll-global_result["balanced_nll"],
            balanced_nll_gain_uniform=float(np.log(C))-global_result["balanced_nll"],
            upload_serialized_bytes=len(up), download_serialized_bytes=len(down),
            note="Table argmax is a diagnostic; no DAGSNet trained and no deployment classifier.")
        assert len(up) == 32+5*p*16*C+8*C
        assert len(down) == 32+4*p*16*C+p*16*C//8+4*C
    output = dict(stage="BOUNDED_PROBE_ONLY", seed=1729, training_rows=40960, group_rows=len(gy),
                  official_test_read=False, model_training=False, provenance=provenance, results=results,
                  gate_status="NOT_LOCKED; no pilot authorization inferred from diagnostic values")
    path = ROOT / "papers/boundary-cells/bounded-probe.json"
    path.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(dict(stage=output["stage"], results=results), indent=2), flush=True)


if __name__ == "__main__":
    main()
