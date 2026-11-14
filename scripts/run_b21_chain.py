"""B21 same-owner continuation of any finished partial session (transfer x-sessions, seed43 session 1, earlier chain parts).

    python scripts/run_b21_chain.py launch   # every terminal COMPLETE head below r50 -> next session on the same owner
    python scripts/run_b21_chain.py collect  # status + pull light outputs of chain sessions

Head = latest session of an arm across runs/b21-transfer/*/batch.json, runs/b21-s43/batch.json, runs/b21-chain/*/batch.json.
Next notebook = head notebook minus its import cell + kernel_import(run, last committed round), kernel_sources = [head, npy],
checkpoint datasets dropped, budget from live quota. Rounds come from the pulled rounds.jsonl; the import cell asserts the
two-rank state round before any decode.
"""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_w20 as W  # noqa: E402
from scripts.watch_kaggle_batch import atomic_json, query  # noqa: E402
from scripts.resume_checkpoint_cells import kernel_import  # noqa: E402

OUT, ART = ROOT / "runs/b21-chain", ROOT / "artifacts/b21-chain"
W.ART, W.TAG = ART, "B21CHAIN"
W.COLLECT = r'(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz)$)'
SOURCES = [("runs/b21-transfer/*/batch.json", "artifacts/b21-transfer"), ("runs/b21-s43/batch.json", "artifacts/b21-s43"),
           ("runs/b21-chain/*/batch.json", "artifacts/b21-chain")]
IMPORT_MARK = ("resumed from verified two-rank checkpoint", "resumed from verified private dataset")


def heads():
    best = {}
    for pattern, art in SOURCES:
        for f in sorted(ROOT.glob(pattern)):
            for r in json.loads(f.read_text()):
                rows = list((ROOT / art / r["slug"]).rglob(f'{r.get("run_slug", r["slug"])}/rounds.jsonl'))
                last = json.loads(rows[0].read_text().splitlines()[-1])["round"] if rows else 0
                r = dict(r, run_slug=r.get("run_slug", r["slug"]), last=last, pulled=(ROOT / art / r["slug"] / "pulled.ok").exists())
                if r["arm"] not in best or r.get("part", 1) >= best[r["arm"]].get("part", 1):
                    best[r["arm"]] = dict(r, part=r.get("part", 1))
    return best


def notebook(h, cap):
    src = ROOT / "notebooks" / h["slug"]; nbp = src / (h["slug"] + ".ipynb")
    assert W.sha(nbp) == h["notebook_sha256"]
    nb = json.loads(nbp.read_text()); meta = json.loads((src / "kernel-metadata.json").read_text())
    cells = [c for c in nb["cells"] if not any(m in "".join(c.get("source", [])) for m in IMPORT_MARK)]
    assert len(nb["cells"]) - len(cells) <= 1
    idx = next(i for i, c in enumerate(cells) if "NPY, =" in "".join(c.get("source", [])))
    code = kernel_import(h["run_slug"], h["last"], h["method"], h["seed"])
    cells.insert(idx, dict(cell_type="code", execution_count=None, metadata={}, outputs=[], source=code.splitlines(True)))
    n = 0
    for c in cells:
        if c["cell_type"] != "code": continue
        s = "".join(c["source"]); n += s.count(f'budget = {h["cap_hours"]} -')
        s = s.replace(f'budget = {h["cap_hours"]} -', f"budget = {cap} -"); compile(s, h["slug"], "exec"); c["source"] = s.splitlines(True)
    assert n == 1 and "--until 50 " in json.dumps(cells)
    nb["cells"] = cells
    slug = f'{h["run_slug"]}-s{h["last"]}'; d = ROOT / "notebooks" / slug; d.mkdir(exist_ok=True)
    p = d / (slug + ".ipynb"); p.write_text(json.dumps(nb, indent=1) + "\n")
    meta.update(id=f'{h["account"]}/{slug}', title=slug, code_file=p.name, kernel_sources=[f'{h["account"]}/{h["slug"]}', h["npy_kernel"]],
                dataset_sources=[x for x in meta["dataset_sources"] if "-ckpt-" not in x]); assert meta["is_private"]
    mp = d / "kernel-metadata.json"; mp.write_text(json.dumps(meta, indent=1) + "\n")
    keep = {k: h[k] for k in ("arm", "seed", "method", "hp", "base_sha", "overlay_sha256", "lr_schedule_rounds", "control",
                              "npy_kernel", "npy_manifest_sha256", "run_slug", "account")}
    return dict(keep, slug=slug, version=1, part=h["part"] + 1, endpoint=50, cap_hours=cap, resume_round=h["last"],
                resume_from=f'{h["account"]}/{h["slug"]}', notebook_sha256=W.sha(p), metadata_sha256=W.sha(mp))


def launch():
    k = W.ka()
    for arm, h in sorted(heads().items()):
        if not h["pulled"] or h["last"] >= 50: continue
        st = query(h)
        if st.get("status") != "COMPLETE": print("SKIP", arm, st.get("status"), flush=True); continue
        d = OUT / arm / f'p{h["part"] + 1}'; d.mkdir(parents=True, exist_ok=True)
        if (d / "launch-complete.json").exists(): continue
        if not (d / "batch.json").exists():
            q = k.account_quota(h["account"]); cap = min(11., int((q["gpu_left"] / 1.15 - .1) * 100) / 100)
            if cap < 1.5: print("DEFER", arm, q["gpu_left"], flush=True); continue
            atomic_json(d / "batch.json", [notebook(h, cap)])
        W.OUT = d; W.launch(json.loads((d / "batch.json").read_text()))
    print("B21CHAIN_LAUNCH_DONE", flush=True)


def collect():
    rows = [r for f in sorted(OUT.glob("*/*/batch.json")) for r in json.loads(f.read_text())]
    states = [query(r) for r in rows]; atomic_json(OUT / "status.json", states)
    for r, st in zip(rows, states):
        print(r["arm"], r["account"], r["slug"], st.get("status"), flush=True)
        if st.get("terminal") and not (ART / r["slug"] / "pulled.ok").exists():
            W.OUT = OUT / r["arm"]
            if W.collect(r) == 0: (ART / r["slug"] / "pulled.ok").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    print("ALL_TERMINAL" if states and all(s.get("terminal") for s in states) else "WAITING", flush=True)


if __name__ == "__main__":
    {"launch": launch, "collect": collect}[sys.argv[1]]()
