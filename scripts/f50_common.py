"""F50 batch: the method chosen after X20 (docs/research/results-x20.md) at the protocol's 50 rounds, locked before launch.

bank = core + input-bank (the proposed method), core = the bank ablation. Same X20 hp; only the round count changes
(rounds 1..50 of the cosine50 schedule) and the full resume state is written every round.
"""
import json
from pathlib import Path

from scripts.x20_common import CELLS as X_CELLS, SOURCES as X_SOURCES

ROOT = Path(__file__).resolve().parents[1]
CELLS = {cell: X_CELLS[cell] for cell in ("bank", "core")}
# owner 03/11: run the proposed method first. Quota 03/11 fits three 50-round runs (one per owner with >= 15 h);
# core seed 43 follows after the 07/11 quota reset as its own launch of this same locked recipe.
PLAN = [("bank", 42), ("bank", 43), ("core", 42)]
DEFERRED = [("core", 43)]
ENDPOINT = 50
SOURCES = [*X_SOURCES, "scripts/f50_common.py", "scripts/preflight_f50.py"]


def hp_for(cell, seed):
    """X20 hp of the cell (same construction as scripts/run_w20.py prepare)."""
    from scripts.q20_common import settings
    spec = CELLS[cell]
    parent = next(r for r in json.loads((ROOT / "runs/q-study/batch.json").read_text()) if r["arm"] == ("Q0" if seed == 42 else "Q1"))
    hp = [h for h in parent["hp"] if not h.startswith(("next_mode=", "real_kd=", "real_kd_gate="))]
    hp += [f"{k}={json.dumps(v) if isinstance(v, bool) else v}" for k, v in settings(spec["mode"]).items()
           if k in {"next_mode", "real_kd", "real_kd_gate"}]
    return hp + [f"r_kind={spec['r_kind']}"] + [f"{k}={v}" for k, v in spec["w"].items()]
