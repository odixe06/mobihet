"""B21 cross-account continuation: parent checkpoint -> receiver-private dataset -> resumed notebook push.

    python scripts/run_b21_transfer.py plan      # lock arm -> target account (live quota)
    python scripts/run_b21_transfer.py run       # per arm, sequential: pull, verify, tar, stage, verify, push
    python scripts/run_b21_transfer.py collect   # pull light outputs of pushed sessions

Disk: one run is ~9.6 GB; staging is sequential and deletes only its own scratch pull/archive after the
dataset download-SHA matches. Parent outputs stay on Kaggle untouched.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_w20 as W  # noqa: E402
from scripts.watch_kaggle_batch import atomic_json, query  # noqa: E402
from scripts.resume_checkpoint_cells import dataset_import  # noqa: E402

OUT, ART = ROOT / "runs/b21-transfer", ROOT / "artifacts/b21-transfer"
SKILL = ROOT / ".agents/skills/kaggle-training-notebook/scripts"
W.ART, W.TAG = ART, "B21XFER"
W.COLLECT = r'(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz)$)'
TARGETS = {"T143": "outlookcheck", "T4FP": "minhtran0601", "T142": "dinhan25", "T3P": "annhtnt"}  # most rounds left -> most quota


def parents():
    rows = []
    for spec in json.loads((ROOT / "runs/b21-resume/batch.json").read_text()):
        rows.append(dict(spec, parent_ref=f'{spec["account"]}/{spec["slug"]}/1', parent_nb=spec["slug"]))
    for spec in json.loads((ROOT / "runs/b21-t1/batch.json").read_text()):
        if spec["arm"] == "T143":
            rows.append(dict(spec, run_slug=spec["slug"], parent_ref=f'{spec["account"]}/{spec["slug"]}/1', parent_nb=spec["slug"]))
    for r in rows:
        f = next((ROOT / "artifacts").glob(f'b21-*/{r["parent_ref"].split("/")[1]}/{r["run_slug"]}/rounds.jsonl'))
        rs = [json.loads(x)["round"] for x in f.read_text().splitlines()]
        assert rs == list(range(1, rs[-1] + 1)), r["arm"]
        r["round"] = rs[-1]
    return rows


def plan():
    path = OUT / "plan.json"
    if path.exists(): return json.loads(path.read_text())
    k = W.ka(); out = []
    for r in parents():
        owner = TARGETS[r["arm"]]; q = k.account_quota(owner)
        out.append(dict(r, target_account=owner, quota_at_plan=q["gpu_left"], new_slug=r["run_slug"] + "-x" + str(r["round"])))
    OUT.mkdir(parents=True, exist_ok=True); atomic_json(path, out)
    print("B21XFER_PLAN", [(r["arm"], r["round"], r["target_account"], r["quota_at_plan"]) for r in out], flush=True)
    return out


def sha(path):
    with path.open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def pull(spec, dest):
    log = OUT / spec["arm"] / "pull.log"
    with log.open("a") as fh:
        p = subprocess.run([*W.cli(spec["account"])[:4], sys.executable, str(SKILL / "pull_kernel_output.py"), spec["parent_ref"],
                            "-p", str(dest), "-j", "8", "--pattern", "^" + re.escape(spec["run_slug"]) + r"/(?!state(_rank1)?\.pt$)"],
                           stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    assert p.returncode == 0, (spec["arm"], "parent pull failed; partials kept")
    # 4.3 GB state files: one stream each is slow; Range-parallel 16 pieces per file, under the parent's credentials.
    with log.open("a") as fh:
        p = subprocess.run([*W.cli(spec["account"])[:4], sys.executable, __file__, "fetchstate", spec["arm"], str(dest)],
                           stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    assert p.returncode == 0, (spec["arm"], "state range pull failed; pieces kept")


def fetchstate(spec, dest):
    sys.path.insert(0, str(SKILL)); import pull_kernel_output as K
    from scripts.pull_checkpoint_parallel import fetch_parallel
    from concurrent.futures import ThreadPoolExecutor
    names = [f'{spec["run_slug"]}/{n}' for n in ("state.pt", "state_rank1.pt")]
    for attempt in range(10):  # broken connections: refresh signed URLs, resume the kept range pieces
        files = K.listing(spec["parent_ref"])[1]
        try:
            with ThreadPoolExecutor(2) as pool: list(pool.map(lambda n: fetch_parallel(files[n], dest / n, 16), names)); break
        except Exception as e:
            print("retry", attempt, type(e).__name__, flush=True); time.sleep(20)
    else: raise RuntimeError("state pull failed after retries")
    print("STATE_PULLED", [(dest / n).stat().st_size for n in names], flush=True)


def verify_state(spec, source):
    import torch
    cfg = json.loads((source / "config.json").read_text())
    for rank, name in enumerate(("state.pt", "state_rank1.pt")):
        st = torch.load(source / name, map_location="cpu", weights_only=True, mmap=True)
        assert st["round"] == spec["round"] and "rng_cpu" in st and "rng_cuda" in st, (spec["arm"], name)
        if rank == 0:
            assert st["method"] == spec["method"] and st["hp"] == cfg["hp"]
            assert st["hp"]["seed"] == spec["seed"] and st["hp"]["rounds"] == 50
            hp = hashlib.sha256(json.dumps(st["hp"], sort_keys=True).encode()).hexdigest()
            cf = hashlib.sha256(json.dumps(st["cfg"], sort_keys=True).encode()).hexdigest()
        del st
    rs = [json.loads(x)["round"] for x in (source / "rounds.jsonl").read_text().splitlines()]
    assert rs == list(range(1, spec["round"] + 1))
    return hp, cf


def wait_ready(api, ref):
    for _ in range(240):  # 9.6 GB processing; read-only retries, never recreate
        try:
            if api.dataset_status(ref).lower() == "ready": return
        except Exception: pass
        time.sleep(15)
    raise RuntimeError("dataset readiness unknown; no recreate, no push")


def stage(spec):
    """Runs under kaggle_as <target>. Writes ready.json."""
    d = OUT / spec["arm"]; ready = d / "ready.json"
    if ready.exists(): return
    upload = d / "upload"; archive = upload / "resume.bin"; manifest_p = d / "manifest.json"
    if not archive.exists():
        pulled = d / "pull"; pull(spec, pulled); source = pulled / spec["run_slug"]
        hp, cf = verify_state(spec, source)
        upload.mkdir(exist_ok=True)
        with tarfile.open(upload / "resume.bin.tmp", "w") as tar: tar.add(source, arcname=spec["run_slug"])
        (upload / "resume.bin.tmp").rename(archive)
        atomic_json(manifest_p, dict(hp_sha256=hp, cfg_sha256=cf))
        shutil.rmtree(pulled)  # scratch copy of the Kaggle parent output, now inside the archive
    m = json.loads(manifest_p.read_text())
    owner = spec["target_account"]; slug = f'mh-b21-{spec["arm"].lower()}-ckpt-r{spec["round"]}'; ref = f"{owner}/{slug}"
    digest = m.get("archive_sha256") or sha(archive)
    manifest = dict(ref=ref, owner=owner, run=spec["run_slug"], round=spec["round"], archive_sha256=digest,
                    archive_bytes=archive.stat().st_size, hp_sha256=m["hp_sha256"], cfg_sha256=m["cfg_sha256"],
                    overlay_sha256=spec["overlay_sha256"], parent_ref=spec["parent_ref"])
    atomic_json(manifest_p, manifest)
    (upload / "resume-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (upload / "dataset-metadata.json").write_text(json.dumps(dict(id=ref, title=f'MobiHet B21 {spec["arm"]} ckpt r{spec["round"]}',
                                                                 licenses=[dict(name="CC0-1.0")]), indent=2) + "\n")
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetMetadataRequest, ApiDownloadDatasetRequest
    api = KaggleApi(); api.authenticate(); intent = d / "create-intent.json"
    if not intent.exists():
        intent.write_text(json.dumps(manifest, indent=2) + "\n")
        r = api.dataset_create_new(str(upload), public=False, quiet=True, convert_to_csv=False, dir_mode="skip")
        atomic_json(d / "create-result.json", dict(ref=r.ref, status=r.status, error=r.error))
        assert not r.error and str(r.status).lower() != "error", r.error
    wait_ready(api, ref)
    req = ApiGetDatasetMetadataRequest(); req.owner_slug, req.dataset_slug = owner, slug
    with api.build_kaggle_client() as c: meta = c.datasets.dataset_api_client.get_dataset_metadata(req)
    assert not meta.error_message and meta.info.is_private
    files = {f.name: f.total_bytes for f in api.dataset_list_files(ref, page_size=100).dataset_files}
    assert files.get("resume.bin") == archive.stat().st_size, files
    req = ApiDownloadDatasetRequest(); req.owner_slug, req.dataset_slug, req.file_name = owner, slug, "resume.bin"
    with api.build_kaggle_client() as c: resp = c.datasets.dataset_api_client.download_dataset(req)
    url = resp.request.url; resp.close(); h = hashlib.sha256(); n = 0
    with requests.get(url, stream=True, timeout=(30, 300)) as r:  # streamed hash, nothing written to disk
        r.raise_for_status()
        for chunk in r.iter_content(8 << 20): h.update(chunk); n += len(chunk)
    assert h.hexdigest() == digest and n == archive.stat().st_size, "download SHA mismatch"
    atomic_json(ready, dict(manifest, private=True, download_sha_verified=True))
    archive.unlink()  # scratch upload; verified copy lives in the private dataset
    print("B21XFER_READY", spec["arm"], ref, flush=True)


def notebook(spec, ready, cap):
    src = ROOT / "notebooks" / spec["parent_nb"]; nb = json.loads((src / (spec["parent_nb"] + ".ipynb")).read_text())
    meta = json.loads((src / "kernel-metadata.json").read_text())
    assert W.sha(src / (spec["parent_nb"] + ".ipynb")) == spec["notebook_sha256"]
    cells = [c for c in nb["cells"] if "resumed from verified two-rank checkpoint" not in "".join(c.get("source", []))]
    assert len(nb["cells"]) - len(cells) == (0 if spec["arm"] == "T143" else 1)
    nb["cells"] = cells
    idx = next(i for i, c in enumerate(cells) if "NPY, =" in "".join(c.get("source", [])))
    code = dataset_import(spec["run_slug"], spec["round"], spec["method"], spec["seed"], ready["archive_sha256"], ready["hp_sha256"], ready["cfg_sha256"])
    cells.insert(idx, dict(cell_type="code", execution_count=None, metadata={}, outputs=[], source=code.splitlines(True)))
    n = 0
    for c in cells:
        if c["cell_type"] != "code": continue
        s = "".join(c["source"]); n += s.count(f'budget = {spec["cap_hours"]} -')
        s = s.replace(f'budget = {spec["cap_hours"]} -', f"budget = {cap} -")
        compile(s, spec["new_slug"], "exec"); c["source"] = s.splitlines(True)
    assert n == 1, n
    assert "--until 50 " in json.dumps(nb)
    owner = spec["target_account"]; slug = spec["new_slug"]; d = ROOT / "notebooks" / slug; d.mkdir(exist_ok=True)
    p = d / (slug + ".ipynb"); p.write_text(json.dumps(nb, indent=1) + "\n")
    meta.update(id=f"{owner}/{slug}", title=slug, code_file=p.name, kernel_sources=[spec["npy_kernel"]],
                dataset_sources=meta["dataset_sources"] + [ready["ref"]]); assert meta["is_private"]
    mp = d / "kernel-metadata.json"; mp.write_text(json.dumps(meta, indent=1) + "\n")
    keep = {k: spec[k] for k in ("arm", "seed", "method", "hp", "base_sha", "overlay_sha256", "endpoint", "lr_schedule_rounds", "control",
                                 "npy_kernel", "npy_manifest_sha256", "run_slug")}
    return dict(keep, account=owner, slug=slug, version=1, cap_hours=cap, resume_round=spec["round"], resume_from=ready["ref"],
                parent_ref=spec["parent_ref"], endpoint=50, notebook_sha256=W.sha(p), metadata_sha256=W.sha(mp))


def run():
    k = W.ka()
    for spec in sorted(plan(), key=lambda s: s["round"]):  # most rounds left first
        d = OUT / spec["arm"]; d.mkdir(parents=True, exist_ok=True)
        if (d / "launch-complete.json").exists(): continue
        if not (d / "ready.json").exists():
            p = subprocess.run([*W.cli(spec["target_account"])[:4], sys.executable, __file__, "stage", spec["arm"]], cwd=ROOT)
            assert p.returncode == 0, (spec["arm"], "staging failed; inspect, no push")
        ready = json.loads((d / "ready.json").read_text())
        q = k.account_quota(spec["target_account"]); cap = min(11., int((q["gpu_left"] / 1.15 - .1) * 100) / 100)
        if cap < 1.5: print("B21XFER_DEFER", spec["arm"], q["gpu_left"], flush=True); continue
        if not (d / "batch.json").exists(): atomic_json(d / "batch.json", [notebook(spec, ready, cap)])
        W.OUT = d; W.launch(json.loads((d / "batch.json").read_text()))
    print("B21XFER_RUN_DONE", flush=True)


def collect():
    rows = [r for a in sorted(OUT.glob("*/batch.json")) for r in json.loads(a.read_text())]
    states = [query(r) for r in rows]; atomic_json(OUT / "status.json", states)
    for r, st in zip(rows, states):
        print(r["arm"], r["account"], r["slug"], st.get("status"), flush=True)
        if st.get("terminal") and not (ART / r["slug"] / "pulled.ok").exists():
            W.OUT = OUT / r["arm"]
            if W.collect(r) == 0: (ART / r["slug"] / "pulled.ok").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    print("ALL_TERMINAL" if states and all(s.get("terminal") for s in states) else "WAITING", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "plan": plan()
    elif cmd == "run": run()
    elif cmd == "stage": stage(next(s for s in plan() if s["arm"] == sys.argv[2]))
    elif cmd == "fetchstate": fetchstate(next(s for s in plan() if s["arm"] == sys.argv[2]), Path(sys.argv[3]))
    elif cmd == "collect": collect()
