"""Reject fold leaks or corrupted receipts using actual global v4 fold support."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_v20 import check_fold_receipts


@pytest.mark.parametrize("bad", (None, "training_fold", "evaluation_fold", "local_val_rows", "scaler_sha256"))
def test_real_fold_receipt_rejects_leaks_and_corruption(tmp_path, bad):
    counts, support = (np.load(f"runs/v-study/{f}_counts.npy") for f in ("train", "val"))
    scaler_sha = json.loads(Path("runs/v-study/split-audit.json").read_text())["scaler_sha256"]
    order = np.argsort(-counts.sum(1), kind="stable")
    for rank in (0, 1):
        mine = np.sort(order[rank::2])
        record = {"training_fold": "train", "evaluation_fold": "val", "officialtest_read": False,
            "local_train_rows": int(counts[mine].sum()), "local_val_rows": int(support[mine].sum()),
            "scaler_fit_on": "fold == train", "scaler_sha256": scaler_sha}
        if rank == 1:
            if bad == "training_fold": record[bad] = "train+val"
            if bad == "evaluation_fold": record[bad] = "test"
            if bad == "local_val_rows": record[bad] += 1
            if bad == "scaler_sha256": record[bad] = "wrong"
        (tmp_path / f"fold_rank{rank}.json").write_text(json.dumps(record))
    if bad is None:
        ranks = check_fold_receipts(tmp_path, counts, support, scaler_sha)
        assert len(np.unique(np.concatenate(ranks))) == 5343
    else:
        with pytest.raises(AssertionError): check_fold_receipts(tmp_path, counts, support, scaler_sha)
