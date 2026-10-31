"""Offline K teacher/gate diagnostics; no training or test-based selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--counts", type=Path, required=True)
    ap.add_argument("--pulls", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    n = np.load(args.counts)
    assert n.shape == (5343, 16) and n.sum() == 37361273
    tot = n.sum(1)
    mass = n.sum(0)
    steps = (tot + 63) // 64
    low = tot < 256
    classes = []
    for c in range(16):
        w = n[:, c] / mass[c]
        classes.append({"class": c, "clients": int((w > 0).sum()),
                        "effective_clients": float(1 / (w @ w)),
                        "top10_mass": float(np.sort(w)[-10:].sum()),
                        "low256_train_mass": float(w[low].sum())})
    runs = []
    for slug in ("mh-k8-gate-s42", "mh-k9-gate-s43", "mh-h3-dose", "mh-k2-dose-s43"):
        root = args.pulls / slug / slug
        records = []
        for r in range(1, 11):
            path = root / "models" / f"r{r:03}.pt"
            q = torch.load(path, map_location="cpu", weights_only=True)["Q"].double().numpy()
            assert q.shape == (16, 16) and np.isfinite(q).all() and (q >= 0).all()
            np.testing.assert_allclose(q.sum(1), 1, atol=1e-4, rtol=0)
            reject = q.argmax(1) != np.arange(16)
            records.append({"round_saved": r, "teacher_for_round": r + 1,
                            "sha256": fingerprint(path),
                            "rejected_classes": np.flatnonzero(reject).tolist(),
                            "rejected_train_mass": float(mass[reject].sum() / mass.sum()),
                            "diagonal": q.diagonal().tolist(), "timeDelay_to_benign": float(q[13, 0])})
        runs.append({"slug": slug, "snapshots": records})
    pairs = []
    for seed in (42, 43):
        matrices = {}
        for name, slug in (("gate", f"mh-k{8 if seed == 42 else 9}-gate-s{seed}"),
                           ("augmentation", f"mh-k{4 if seed == 42 else 5}-aug-s{seed}")):
            path = args.pulls / slug / slug / "cms_r010.npz"
            with np.load(path) as z:
                assert np.array_equal(z["ids"], np.arange(5343))
                cm = z["private"].sum(0)
            assert cm.sum() == 9347679
            matrices[name] = {"sha256": fingerprint(path), "support": cm.sum(1).tolist(),
                              "benign_to_timeDelay": int(cm[0, 13]),
                              "timeDelay_to_benign": int(cm[13, 0]),
                              "correct_benign": int(cm[0, 0]), "correct_timeDelay": int(cm[13, 13])}
        assert matrices["gate"]["support"] == matrices["augmentation"]["support"]
        pairs.append({"seed": seed, **matrices})
    out = {"scope": "Descriptive full-count and frozen Q/CM analysis; not causal or gradient evidence.",
           "counts_sha256": fingerprint(args.counts), "clients": int(len(tot)),
           "low256_clients": int(low.sum()), "low256_client_fraction": float(low.mean()),
           "low256_train_mass": float(tot[low].sum() / tot.sum()),
           "local_steps_min_max": [int(steps.min()), int(steps.max())],
           "synth64_rows_per_real_row_min_max": [float((64 * steps / tot).min()),
                                                  float((64 * steps / tot).max())],
           "max_client_class_count": int(n.max()), "class_weight_concentration": classes,
           "teacher_history": runs, "paired_confusions_r10": pairs}
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("teacher_history", "class_weight_concentration")}, indent=2))


if __name__ == "__main__":
    main()
