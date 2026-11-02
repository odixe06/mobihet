"""A fold cache contains no officialtest, yet Env must select train/val and correct sizes."""
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from core import data
from core.dagsnet import width
from scripts.run_v20 import DiagnosticEnv
from scripts.run_fl import PROTOCOL


def test_fold_cache_sizes_and_routing(tmp_path, monkeypatch):
    # No test_X or test_Y exist. A test-route access would immediately fail.
    for key, value in {"X": np.zeros((8, 77), np.float16), "Y": np.arange(8, dtype=np.int8),
        "ids": np.array([3, 7]), "off": np.array([0, 4, 8]), "F": np.array([0, 0, 0, 1, 0, 0, 1, 1], np.int8)}.items():
        np.save(tmp_path / f"train_{key}.npy", value)
    (tmp_path / "done").write_text("ok")
    (tmp_path / "scaler.json").write_text(json.dumps({"fit_on": "fold == train", "train_rows": 32685166}))
    monkeypatch.setenv("V4_NPY", str(tmp_path))
    sizes = DiagnosticEnv._sizes(str(tmp_path), None)
    assert sizes[3] == 3 and sizes[7] == 2 and sizes.sum() == 5
    original = data.load
    with patch("torch.cuda.is_available", return_value=False):
        env = DiagnosticEnv(width(1), PROTOCOL, clients=[3, 7], root=str(tmp_path))
    assert data.load is original and env.train.sizes.tolist() == [3, 2] and env.test.sizes.tolist() == [1, 2]
    assert env.train.Y.tolist() == [0, 1, 2, 4, 5] and env.test.Y.tolist() == [3, 6, 7]
    assert not env.fold_receipt["officialtest_read"]
    assert env.n_all.tolist() == [3, 2]
    (tmp_path / "scaler.json").write_text(json.dumps({"fit_on": "train+val", "train_rows": 37361273}))
    with patch("torch.cuda.is_available", return_value=False), pytest.raises(AssertionError):
        DiagnosticEnv(width(1), PROTOCOL, clients=[3, 7], root=str(tmp_path))
    assert data.load is original
