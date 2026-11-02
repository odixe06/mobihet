"""Real frozen teacher data must fail audit after a changed target, source or bank."""
import json
from pathlib import Path

import pytest
import torch
from methods.mobihet_next import signature
from methods.mobihet_rstudy import load_schedule
from methods.mobihet_rstudy_data import SHA256
from scripts.audit_r20 import check_replay


@pytest.mark.parametrize("bad", (None, "Q", "bank_target", "bank_body", "bank_X", "source", "live_signature"))
def test_actual_teacher_audit_rejects_corruption(bad):
    spec = json.loads(Path("runs/q-study/batch.json").read_text())[6]
    s = torch.load(Path("artifacts/q20") / spec["slug"] / spec["slug"] / "models/r010.pt", weights_only=True, map_location="cpu")
    s["Q_live"], s["bank_target_live"] = s["Q"].clone(), s["bank_target"].clone()
    hp = {"next_mode": "mixed_input", "r_source_sha256": SHA256}
    row = {"replay_source_sha256": SHA256,
           "live_teacher_produced_sha256": signature(s["Q_live"], s["bank_X"], s["bank_target_live"], "mixed_input")}
    history = load_schedule(42, "cpu", SHA256)
    if bad in {"Q", "bank_target", "bank_body", "bank_X"}: s[bad] = s[bad].roll(1, -1)
    if bad == "source": row["replay_source_sha256"] = "wrong"
    if bad == "live_signature": row["live_teacher_produced_sha256"] = "wrong"
    if bad is None: check_replay(s, row, hp, history, 10)
    else:
        with pytest.raises(AssertionError): check_replay(s, row, hp, history, 10)
