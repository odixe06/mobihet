"""Owner-authorized seed42/10-round compression screen; immutable launch receipts and local 1200s watcher."""
import concurrent.futures
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_w20 as w
from scripts.watch_kaggle_batch import atomic_json, query

OUT = ROOT / "runs/c10-study"
ART = ROOT / "artifacts/c10"
ARMS = {"core": (256, False), "boot64": (64, False), "q8": (256, True), "combo": (64, True)}


def prepare():
    if (OUT / "batch.json").exists():
        return json.loads((OUT / "batch.json").read_text())
    frozen = json.loads((ROOT / "runs/core-compression/source.json").read_text())
    assert all(w.sha(ROOT / n) == h for n, h in frozen.items())
    smoke = json.loads((ROOT / "runs/core-compression/smoke.json").read_text())
    assert len(smoke) == 5 and all(x["finite"] and x["skipped_frac"] == 0 for x in smoke)
    assert next(x for x in smoke if x["arm"] == "lossless")["bitexact_control"]
    quota = w.inventory()
    slots = []
    for q in sorted(quota["accounts"], key=lambda x: -x.get("gpu_left", 0)):
        if q.get("inventory_error") or q["active_recent"] or q["unknown_recent"]:
            continue
        slots += [q["user"]] * min(2, int(q["gpu_left"] // (3.3 * 1.15)))
    assert len(slots) >= 4, "Need four full allocations; do not reduce batch silently"
    base = json.loads((ROOT / "docs/research/f50-launch.json").read_text())["batch"]
    parent = next(x for x in base if x["cell"] == "core")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/main"], text=True).split()[0]
    assert head == remote, "Base source must already exist remotely; no automatic Git push"
    build = json.loads((ROOT / "runs/w-study/builder.json").read_text())
    overlay = ["methods/mobihet.py", "methods/mobihet_compression.py"]
    batch = []
    for i, (cell, (bins, quant)) in enumerate(ARMS.items()):
        owner, slug = slots[i], f"mh-c{i}-{cell}-s42-r10"
        hp = [x for x in parent["hp"] if not x.startswith(("seed=", "diagnostic_rounds="))]
        hp += ["seed=42", "diagnostic_rounds=[1,5,10]", f"compression_bins={bins}", f"compression_q8={str(quant).lower()}"]
        args = shlex.join(["scripts/run_q20.py", "--method", "mobihet_compression", "--out", f"/kaggle/working/{slug}",
                           "--width", "1", "--rounds", "50", "--until", "10", "--compile", "--wandb", "mobihet-v4",
                           "--max-hours", "BUDGET", "--cache", "", "--set", *hp])
        subprocess.run([sys.executable, "scripts/gen_notebook.py", "--owner", owner, "--slug", slug,
                        "--sha", head, "--max-hours", "3.3", "--args", args, "--npy-kernel", build["ref"],
                        "--npy-manifest-sha256", build["manifest_sha256"], "--overlay", *overlay], check=True, capture_output=True)
        folder = ROOT / "notebooks" / slug
        notebook, metadata = folder / (slug + ".ipynb"), folder / "kernel-metadata.json"
        nb = json.loads(notebook.read_text())
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                c["source"] = "".join(c["source"]).replace('TORCHINDUCTOR_COMPILE_THREADS="2"', 'TORCHINDUCTOR_COMPILE_THREADS="1"').splitlines(keepends=True)
        notebook.write_text(json.dumps(nb, indent=1) + "\n")
        subprocess.run([sys.executable, str(w.SKILL / "embed_wandb_key.py"), str(notebook), "--metadata", str(metadata)], check=True, capture_output=True)
        meta = json.loads(metadata.read_text())
        runtime = json.loads((ROOT / "knowledge/runtime.json").read_text())
        assert meta["is_private"] and meta["id"] == f"{owner}/{slug}"
        assert all(meta[k] == runtime[k] for k in ("machine_shape", "docker_image"))
        for c in json.loads(notebook.read_text())["cells"]:
            if c["cell_type"] == "code": compile("".join(c["source"]), str(notebook), "exec")
        batch.append(dict(arm=f"C{i}", cell=cell, account=owner, slug=slug, version=1, hp=hp, cap_hours=3.3,
                          endpoint=10, base_sha=head, source_sha256=w.sha(ROOT / "methods/mobihet.py"),
                          overlay_sha256={n: w.sha(ROOT / n) for n in overlay}, notebook_sha256=w.sha(notebook),
                          metadata_sha256=w.sha(metadata)))
    atomic_json(OUT / "batch.json", batch)
    atomic_json(ROOT / "docs/research/c10-launch.json", dict(batch=batch, batch_sha256=w.sha(OUT / "batch.json"),
                authorization="Owner 03/11: Kaggle 10-15 rounds, seed42 only, preserve Core performance with fewer bytes"))
    return batch


def watch(batch):
    progress = OUT / "progress.json"
    results = json.loads(progress.read_text()) if progress.exists() else {}
    (OUT / "collection").mkdir(exist_ok=True)
    while True:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            states = list(pool.map(query, [s for s in batch if not results.get(s["arm"], {}).get("terminal")]))
        for state in states:
            results.setdefault(state["arm"], {}).update(state=state, terminal=state["terminal"])
        atomic_json(progress, results)
        for spec in batch:
            r = results[spec["arm"]]
            if r["terminal"] and not r.get("pulled"):
                r["pulled"] = w.collect(spec) == 0
                atomic_json(progress, results)
        print("C10_PROGRESS", {k: v["state"]["status"] for k, v in results.items()}, flush=True)
        if all(r["terminal"] and r.get("pulled") for r in results.values()):
            subprocess.run([sys.executable, "scripts/audit_c10.py"], check=True)
            print("C10_CLOSED", flush=True)
            return
        startup = OUT / "startup/startup.json"
        if not startup.exists() or not json.loads(startup.read_text()).get("all_ready"):
            subprocess.run([sys.executable, "scripts/check_screening_startup.py", "--batch", str(OUT / "batch.json"),
                            "--out", str(OUT / "startup"), "--entity", "21522798-uit"], cwd=ROOT)
        time.sleep(1200)


def main():
    OUT.mkdir(exist_ok=True)
    lock = (OUT / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    w.OUT, w.ART, w.TAG = OUT, ART, "C10"
    w.COLLECT = r"(^provenance/|\.log$|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz|train_cms_r\d{3}\.npz|diagnostics_rank[01]\.jsonl|bootstrap_codec_rank[01]\.json|qcodec_rank[01]_r\d{3}\.pt)$|/models/r\d{3}\.pt$|/recovery/rank[01]\.jsonl$)"
    batch = prepare()
    w.launch(batch)
    watch(batch)


if __name__ == "__main__": main()
