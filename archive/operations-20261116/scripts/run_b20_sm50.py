"""B20 SM r50 (ladder step after SM42P PASS_TRAJECTORY, 13/11): resume SM42 r20 -> r50 (same owner) + fresh SM43 r50.

    python scripts/run_b20_sm50.py launch    # build notebooks, live quota gate, lock runs/b20-sm50/batch.json, push
    python scripts/run_b20_sm50.py collect   # one shot: status; pull terminal kernels into artifacts/b20-sm50/<kernel>
    python scripts/run_b20_sm50.py audit     # same checks/verdict as run_b20_full.py audit (r50: macro > .9048 per seed)

Resume row = SM42P notebook bytes + verified two-rank import cell, --until 50 and the session budget (pattern of
scripts/run_codec_combo50.py). Fresh row = scripts/run_b20_full.py rows() for SM43 with the same base SHA, overlays and hp.
Rule locked before push: docs/research/screening-compute-b20-addendum.md §4.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_b20_full as B  # noqa: E402
from scripts import run_w20 as W  # noqa: E402
from scripts.resume_checkpoint_cells import kernel_import  # noqa: E402
from scripts.watch_kaggle_batch import atomic_json, query  # noqa: E402

PILOT = ROOT / "runs/b20-full"
OUT, ART = ROOT / "runs/b20-sm50", ROOT / "artifacts/b20-sm50"
RESUME_CAP = 7.0          # 30 rounds; SM42P measured 3.57 h train / 20 rounds (CBB resume r20->r50 used 7.0)
FRESH = ("SM43", 43, ["synth_grad=mix16"], 50, 11.5, "minhtran0601")   # live quota 13/11 02:40: 17.98 h, no active run
B.OUT, B.ART = OUT, ART
W.OUT, W.ART, W.TAG = OUT, ART, "B20SM"


def resume_row(orig):
    owner, run, cap = orig["account"], orig["slug"], RESUME_CAP
    kernel = "mh-k-b20-sm42-r50-c2"
    nb_path = ROOT / "notebooks" / run / (run + ".ipynb")
    assert W.sha(nb_path) == orig["notebook_sha256"], "pilot notebook changed"
    nb = json.loads(nb_path.read_text())
    idx = next(i for i, c in enumerate(nb["cells"]) if "NPY, =" in "".join(c.get("source", [])))
    nb["cells"].insert(idx, dict(cell_type="code", execution_count=None, metadata={}, outputs=[],
                                 source=kernel_import(run, 20, orig["method"], orig["seed"]).splitlines(keepends=True)))
    n_budget = n_until = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        n_budget += src.count(f"budget = {orig['cap_hours']} -")
        n_until += src.count("--until 20 ")
        src = src.replace(f"budget = {orig['cap_hours']} -", f"budget = {cap} -").replace("--until 20 ", "--until 50 ")
        compile(src, kernel, "exec")
        c["source"] = src.splitlines(keepends=True)
    assert n_budget == 1 and n_until == 1, (n_budget, n_until)
    folder = ROOT / "notebooks" / kernel
    folder.mkdir(exist_ok=True)
    path = folder / (kernel + ".ipynb")
    path.write_text(json.dumps(nb, indent=1) + "\n")
    meta = json.loads((ROOT / "notebooks" / run / "kernel-metadata.json").read_text())
    meta.update(id=f"{owner}/{kernel}", title=kernel, code_file=kernel + ".ipynb", kernel_sources=[f"{owner}/{run}", orig["npy_kernel"]])
    assert meta["is_private"]
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
    return dict(orig, arm="SM42", slug=kernel, run_slug=run, version=1, cap_hours=cap, endpoint=50, resume_from=f"{owner}/{run}",
                resume_round=20, notebook_sha256=W.sha(path), metadata_sha256=W.sha(folder / "kernel-metadata.json"))


def prepare():
    if (OUT / "batch.json").exists():
        return json.loads((OUT / "batch.json").read_text())
    pilot = {r["arm"]: r for r in json.loads((PILOT / "batch.json").read_text())}
    assert json.loads((PILOT / "audit.json").read_text())["SM42P"]["verdict"] in {"PASS", "PASS_TRAJECTORY"}
    batch = [resume_row(pilot["SM42P"])]
    arm, seed, knobs, until, cap, owner = FRESH
    B.ARMS = [(arm, seed, knobs, until, cap)]
    fresh = B.rows({"accounts": [{"user": owner, "gpu_left": 99.0, "active_recent": []}]})[0]
    assert fresh["account"] == owner and fresh["overlay_sha256"] == pilot["SM42P"]["overlay_sha256"]
    assert fresh["hp"] == [x for x in pilot["SM42P"]["hp"] if x != "seed=42" and x != "synth_grad=mix16"] + ["seed=43", "synth_grad=mix16"]
    batch.append(fresh)
    k = W.ka()
    need = {}
    for r in batch:
        need[r["account"]] = need.get(r["account"], 0) + r["cap_hours"] * 1.15
    for o, hrs in need.items():                                     # live quota gate before locking
        q = k.account_quota(o)
        assert q["gpu_left"] >= hrs, (o, q["gpu_left"], hrs)
    atomic_json(OUT / "batch.json", batch)
    print("B20SM_LOCKED", [(r["arm"], r["account"], r["slug"], r["cap_hours"]) for r in batch], flush=True)
    return batch


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "prepare":
        prepare()
    elif cmd == "launch":
        W.launch(prepare())
    elif cmd == "collect":
        batch = json.loads((OUT / "batch.json").read_text())
        states = [query(s) for s in batch]
        atomic_json(OUT / "status.json", states)
        for spec, st in zip(batch, states):
            print(spec["arm"], spec["slug"], st.get("status"), flush=True)
            if st.get("terminal") and not (ART / spec["slug"] / "pulled.ok").exists() and W.collect(spec) == 0:
                (ART / spec["slug"] / "pulled.ok").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        print("ALL_TERMINAL" if all(s.get("terminal") for s in states) else "WAITING", flush=True)
    elif cmd == "audit":
        B.audit()


if __name__ == "__main__":
    main()
