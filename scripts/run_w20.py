"""W20/X20 pipeline: public npy-cache builder -> quota inventory -> frozen notebooks -> all-owner gates + one-shot push ->
local 20-min watch, early pull, 20-round audit. Each stage writes a receipt under runs/<w|x>-study and is skipped when its
receipt exists, so a rerun resumes the failing stage instead of relaunching anything.

    python scripts/run_w20.py [--study x20]           # whole pipeline (owner 02/11: run the batch after local preflight PASS)
    python scripts/run_w20.py [--study x20] --collect-only
"""
import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.launch_repeat20 import owned_refs
from scripts.watch_kaggle_batch import atomic_json, query, status_from_cli
from scripts import w20_common, x20_common

STUDIES = {  # tag -> receipts, cells, artifact folder, launch authorization
    "w20": ("runs/w-study", w20_common, "artifacts/w20", "Owner 02/11: run the whole W batch after local preflight PASS; public npy-cache builder",
            "One component per cell vs audited Q/R controls; 20 round/cosine50; analysis reads session spread from R20"),
    "x20": ("runs/x-study", x20_common, "artifacts/x20", "Owner 02/11 (after W20): run the tests needed for the final method; X batch after local preflight PASS",
            "W20-passed components combined (core/bank), nckd_local one change vs W2/W3, full read only if nckd_local passes; 20 round/cosine50"),
}
OUT, CELLS, SOURCES, ART, TAG = ROOT / "runs/w-study", w20_common.CELLS, w20_common.SOURCES, ROOT / "artifacts/w20", "W20"
SKILL = ROOT / ".agents/skills/kaggle-training-notebook/scripts"
BUILDER_OWNER, BUILDER_SLUG = "odixe0502", "mobihet-v4npy-cache"
CAP = 5.3                       # h/run: Q/R measured <= 4.8 h incl. setup; notebook keeps its own 0.4 h reserve
COMPACT_BOOTSTRAP = 4 * 16 + 73847 + 4 * 16 * (77 + 77 * 78 // 2) + 4 * 16 + 2 * 16 * 77 * 256 + 2 * 16 * (77 + 77 * 78 // 2)
COLLECT = (r'(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz|train_cms_r\d{3}\.npz|diagnostics_rank[01]\.jsonl|'
           r'external_cms_rank[01]_r\d{3}\.npz|teacher_probe_rank[01]_r\d{3}\.json)$|/models/r\d{3}\.pt$|/recovery/rank[01]\.jsonl$)')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ka():
    spec = importlib.util.spec_from_file_location("ka", SKILL / "kaggle_account.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def cli(owner):
    return [sys.executable, str(SKILL / "kaggle_as.py"), owner, "--", sys.executable, str(ROOT / "scripts/kaggle_cli_bounded.py")]


def git_head():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/main"], cwd=ROOT, text=True).split()[0]
    assert head == remote, "Notebooks clone GitHub at HEAD: push it first"
    for name in SOURCES:   # executed code must be the committed bytes, no overlay
        assert subprocess.check_output(["git", "show", f"{head}:{name}"], cwd=ROOT) == (ROOT / name).read_bytes(), name
    return head


def builder(head):
    """Public CPU kernel that runs core.data.build_npy once; its output mounts into every W kernel."""
    receipt = OUT / "builder.json"
    if receipt.exists(): return json.loads(receipt.read_text())
    shared = ROOT / "runs/w-study/builder.json"   # one cache for every study; each kernel re-verifies its manifest SHA256
    if shared.exists(): return json.loads(shared.read_text())
    ref = f"{BUILDER_OWNER}/{BUILDER_SLUG}"
    if not (OUT / "builder-push.json").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts/gen_notebook.py"), "--owner", BUILDER_OWNER, "--slug", BUILDER_SLUG,
                        "--build-npy", "--sha", head], check=True, capture_output=True, cwd=ROOT)
        folder = ROOT / "notebooks" / BUILDER_SLUG
        text = (folder / f"{BUILDER_SLUG}.ipynb").read_text()
        assert "INLINE WANDB CREDENTIAL" not in text and "wandb.login" not in text and "import wandb" not in text and json.loads((folder / "kernel-metadata.json").read_text())["is_private"] is False
        p = subprocess.run(["timeout", "--foreground", "240s", *cli(BUILDER_OWNER), "kernels", "push", "-p", str(folder)],
                           capture_output=True, text=True)
        (OUT / "builder-push.log").write_text(p.stdout + p.stderr)
        assert p.returncode == 0 and "successfully pushed" in p.stdout, "builder push failed; inspect builder-push.log"
        version = int(re.search(r"Kernel version (\d+) successfully pushed", p.stdout).group(1))
        atomic_json(OUT / "builder-push.json", {"ref": ref, "version": version, "base_sha": head,
                                                 "notebook_sha256": sha(folder / f"{BUILDER_SLUG}.ipynb")})
    push = json.loads((OUT / "builder-push.json").read_text())
    spec = {"arm": "builder", "account": BUILDER_OWNER, "slug": BUILDER_SLUG, "version": push["version"]}
    while True:
        state = query(spec)
        print(f"{TAG}_BUILDER", state["status"], flush=True)
        if state["terminal"]: break
        time.sleep(1200)
    assert state["status"] == "COMPLETE", "builder ERROR; pull its log before retry"
    dest = OUT / "builder"; dest.mkdir(exist_ok=True)
    p = subprocess.run([*cli(BUILDER_OWNER)[:4], sys.executable, str(ROOT / "scripts/run_bounded_python.py"),
                        str(SKILL / "pull_kernel_output.py"), f'{ref}/{push["version"]}', "-p", str(dest), "-j", "4",
                        "--pattern", r"(^v4npy/manifest\.json$|^provenance/source_manifest\.json$|\.log$)"], cwd=ROOT)
    assert p.returncode == 0
    manifest = dest / "v4npy/manifest.json"
    files = json.loads(manifest.read_text())["files"]
    assert set(files) == {"done", *(f"{s}_{k}.npy" for s in ("train", "test") for k in ("X", "Y", "off", "ids", "F"))}
    # train X: 37,361,273 rows x 77 fp16 + 128 B npy header
    assert files["train_X.npy"]["bytes"] == 37361273 * 77 * 2 + 128 and files["test_X.npy"]["bytes"] == 9347679 * 77 * 2 + 128
    out = {"ref": ref, "version": push["version"], "manifest_sha256": sha(manifest), "files": files, "base_sha": head}
    atomic_json(receipt, out)
    print(f"{TAG}_BUILDER_PASS", out["manifest_sha256"], flush=True)
    return out


def inventory():
    receipt = OUT / "quota.json"
    if receipt.exists(): return json.loads(receipt.read_text())
    k = ka()
    owners = sorted({q["user"] for q in json.loads((ROOT / "runs/r-study/quota.json").read_text())["accounts"]}
                    | set(json.loads((ROOT / "runs/r-repeat/authorized-owners.json").read_text())["owners"]))
    assert set(owners) <= set(k.known_users())

    def one(owner):
        q = k.account_quota(owner)
        p = subprocess.run(["timeout", "90s", *cli(owner), "kernels", "list", "--mine", "--sort-by", "dateRun", "--page-size", "10",
                            "--csv"], capture_output=True, text=True)
        if p.returncode: return dict(q, user=owner, inventory_error=True)
        states = []
        for row in csv.DictReader(io.StringIO(p.stdout)):
            ref = row.get("ref", "")
            if not ref.startswith(owner + "/"): continue
            s = subprocess.run(["timeout", "45s", *cli(owner), "kernels", "status", ref], capture_output=True, text=True)
            states.append({"ref": ref, "status": status_from_cli(s.returncode, s.stdout)})
        return dict(q, recent_sessions=states, active_recent=[r["ref"] for r in states if r["status"] in {"QUEUED", "RUNNING"}],
                    unknown_recent=[r["ref"] for r in states if r["status"] == "QUERY_ERROR"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: rows = list(pool.map(one, owners))
    out = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "accounts": rows, "global_cli": k.active_user(),
           "scope": "Owner-authorized saved accounts (R16 + repeat6); scoped identity/quota, latest10 owned sessions"}
    atomic_json(receipt, out)
    return out


def prepare(head, build, quota):
    if (OUT / "batch.json").exists(): return json.loads((OUT / "batch.json").read_text())
    smoke = json.loads((OUT / "preflight.json").read_text())
    assert smoke["source_unchanged"] and all(sha(ROOT / n) == h for n, h in smoke["source_sha256"].items())
    assert {(r["cell"], r["seed"]) for r in smoke["reports"] if r["check"] == "cell"} == {(c, s) for c in CELLS for s in (42, 43)}
    suites = ET.parse(OUT / "tests.xml").getroot().findall("testsuite")
    assert all(int(s.attrib[k]) == 0 for s in suites for k in ("failures", "errors", "skipped"))
    # up to two concurrent kernels per owner (as P20), each owner covering the sum of its caps x 1.15
    slots = []
    for q in sorted(quota["accounts"], key=lambda q: (-q.get("gpu_left", 0), q["user"])):
        if q.get("inventory_error") or q["active_recent"] or q["unknown_recent"]: continue
        slots += [q["user"]] * min(2, int(q["gpu_left"] // (CAP * 1.15)))
    plan = [(cell, seed) for cell in CELLS for seed in (42, 43)]
    assert len(slots) >= len(plan), f"{len(plan)} full allocations required, {len(slots)} slots; do not cut rounds/seeds"
    # replay needs the frozen Q6/Q7 blob as an owner-private dataset (never Git/notebook); both replay seeds share one owner
    replay = [(c, s) for c, s in plan if CELLS[c]["r_kind"] == "replay"]
    first = slots[0]
    assert slots.count(first) >= len(replay)
    owners = {key: first for key in replay}
    for _ in replay: slots.remove(first)
    for key in plan:
        if key not in owners: owners[key] = slots.pop(0)
    ready = ROOT / "runs/r-study/input-datasets" / (first + ".ready.json")
    if replay and not ready.exists():
        subprocess.run([*cli(first)[:4], sys.executable, str(ROOT / "scripts/run_bounded_python.py"),
                        str(ROOT / "scripts/stage_r20_input.py"), "--owner", first], check=True, cwd=ROOT)
    inputs = {first: json.loads(ready.read_text())} if replay else {}
    if replay: assert inputs[first]["private"] and inputs[first]["download_sha_verified"]
    parents = {r["arm"]: r for r in json.loads((ROOT / "runs/q-study/batch.json").read_text())}
    runtime = json.loads((ROOT / "knowledge/runtime.json").read_text())
    from methods import mobihet_rstudy_data as frozen
    from scripts.q20_common import settings
    batch = []
    for index, (cell, seed) in enumerate(plan):
        spec = CELLS[cell]
        parent = parents["Q0" if seed == 42 else "Q1"]          # protocol hp of the audited stream control
        hp = [h for h in parent["hp"] if not h.startswith(("next_mode=", "real_kd=", "real_kd_gate="))]
        hp += [f"{k}={json.dumps(v) if isinstance(v, bool) else v}" for k, v in settings(spec["mode"]).items()
               if k in {"next_mode", "real_kd", "real_kd_gate"}]
        hp += [f"r_kind={spec['r_kind']}"] + [f"{k}={v}" for k, v in spec["w"].items()]
        if spec["r_kind"] == "replay": hp.append(f"r_source_sha256={frozen.SHA256}")
        owner, arm = owners[(cell, seed)], f"{TAG[0]}{index}"
        slug = f"mh-{arm.lower()}-{cell.replace('_', '-')}-s{seed}"
        args = shlex.join(["scripts/run_q20.py", "--method", "mobihet_wstudy", "--out", f"/kaggle/working/{slug}", "--width", "1",
                           "--rounds", "50", "--until", "20", "--compile", "--wandb", "mobihet-v4", "--max-hours", "BUDGET",
                           "--cache", "", "--set", *hp])
        subprocess.run([sys.executable, str(ROOT / "scripts/gen_notebook.py"), "--owner", owner, "--slug", slug, "--sha", head,
                        "--max-hours", str(CAP), "--args", args, "--npy-kernel", build["ref"],
                        "--npy-manifest-sha256", build["manifest_sha256"], "--overlay", "methods/mobihet.py"],
                       check=True, capture_output=True, cwd=ROOT)
        folder = ROOT / "notebooks" / slug; notebook = folder / (slug + ".ipynb"); metadata = folder / "kernel-metadata.json"
        nb = json.loads(notebook.read_text())
        for c in nb["cells"]:   # same compile-thread setting as R/V launches
            if c["cell_type"] == "code": c["source"] = "".join(c["source"]).replace(
                'TORCHINDUCTOR_COMPILE_THREADS="2"', 'TORCHINDUCTOR_COMPILE_THREADS="1"').splitlines(keepends=True)
        notebook.write_text(json.dumps(nb, indent=1) + "\n")
        if spec["r_kind"] == "replay":
            meta = json.loads(metadata.read_text()); meta["dataset_sources"].append(inputs[owner]["ref"])
            metadata.write_text(json.dumps(meta, indent=1) + "\n")
        subprocess.run([sys.executable, str(SKILL / "embed_wandb_key.py"), str(notebook), "--metadata", str(metadata)],
                       check=True, capture_output=True)
        meta = json.loads(metadata.read_text())
        assert meta["is_private"] is True and meta["id"] == f"{owner}/{slug}" and meta["kernel_sources"] == [build["ref"]]
        assert meta["machine_shape"] == runtime["machine_shape"] and meta["docker_image"] == runtime["docker_image"]
        for c in json.loads(notebook.read_text())["cells"]:
            if c["cell_type"] == "code": compile("".join(c["source"]), str(notebook), "exec")
        batch.append({"arm": arm, "cell": cell, "mode": spec["mode"], "r_kind": spec["r_kind"], "account": owner, "slug": slug,
                      "version": 1, "hp": hp, "base_sha": head, "source_sha256": sha(ROOT / "methods/mobihet.py"),
                      "overlay_sha256": {"methods/mobihet.py": sha(ROOT / "methods/mobihet.py")}, "notebook_sha256": sha(notebook),
                      "metadata_sha256": sha(metadata), "cap_hours": CAP, "endpoint": 20, "lr_schedule_rounds": 50,
                      "npy_kernel": build["ref"], "npy_manifest_sha256": build["manifest_sha256"],
                      "bootstrap_total": COMPACT_BOOTSTRAP if spec["w"].get("w_boot") == "compact" else 3311744})
    atomic_json(OUT / "batch.json", batch)
    atomic_json(ROOT / f"docs/research/{TAG.lower()}-launch.json", {"batch_sha256": sha(OUT / "batch.json"), "batch": batch,
        "preflight_sha256": sha(OUT / "preflight.json"), "quota_sha256": sha(OUT / "quota.json"), "builder": build,
        "authorization": STUDIES[TAG.lower()][3], "scope": STUDIES[TAG.lower()][4]})
    print(f"{TAG}_NOTEBOOKS_LOCKED", len(batch), flush=True)
    return batch


def launch(batch):
    if (OUT / "launch-complete.json").exists(): return
    raw = (OUT / "batch.json").read_bytes()
    k = ka(); before = k.active_user()
    groups = {}
    for row in batch:
        groups.setdefault(row["account"], []).append(row)
        assert not (OUT / (row["slug"] + ".push-intent.json")).exists(), "Existing push intent; inspect before retry"

    def gate(item):
        owner, runs = item
        q = k.account_quota(owner)
        assert q["gpu_left"] >= sum(r["cap_hours"] for r in runs) * 1.15, (owner, "quota changed")
        refs = owned_refs(cli(owner), owner)
        assert all(f'{owner}/{r["slug"]}' not in refs for r in runs), (owner, "slug exists")
        for source in ("datasets files -d odixe0502/veremi-nextgen-v4-scaled-per-vehicle", f"kernels status {BUILDER_OWNER}/{BUILDER_SLUG}"):
            p = subprocess.run(["timeout", "90s", *cli(owner), *source.split()], capture_output=True, text=True)
            assert p.returncode == 0, (owner, "input access unavailable", source)
        return {"owner": owner, "quota": q, "planned_cap_hours": sum(r["cap_hours"] for r in runs)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: gates = list(pool.map(gate, groups.items()))
    atomic_json(OUT / "launch-gates.json", {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "accounts": gates})

    def push(item):
        owner, runs = item
        for row in runs:
            assert (OUT / "batch.json").read_bytes() == raw
            slug = row["slug"]; folder = ROOT / "notebooks" / slug
            assert sha(folder / (slug + ".ipynb")) == row["notebook_sha256"]
            ref = f"{owner}/{slug}/1"
            with (OUT / (slug + ".push-intent.json")).open("x") as f: json.dump({"ref": ref, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, f)
            p = subprocess.run(["timeout", "--foreground", "--kill-after=10s", "240s", *cli(owner), "kernels", "push", "-p", str(folder)],
                               capture_output=True, text=True)
            log = OUT / (slug + ".push.log"); log.write_text(p.stdout + p.stderr); log.chmod(0o600)
            assert p.returncode == 0 and re.search(r"Kernel version 1 successfully pushed", p.stdout), f"Push failed/ambiguous {slug}"
            atomic_json(OUT / (slug + ".push.json"), {"ref": ref, "notebook_sha256": row["notebook_sha256"], "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            print(f"{TAG}_PUSHED", row["arm"], ref, flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(push, groups.items()))
    atomic_json(OUT / "launch-complete.json", {"batch_sha256": sha(OUT / "batch.json"), "runs": len(batch),
                                               "global_cli_before": before, "global_cli_after": k.active_user()})
    print(f"{TAG}_ALL_PUSHED", len(batch), flush=True)


def collect(spec):
    dest = ART / spec["slug"]; dest.mkdir(parents=True, exist_ok=True)
    with (OUT / "collection" / (spec["slug"] + ".collect.log")).open("w") as log:
        p = subprocess.run([*cli(spec["account"])[:4], sys.executable, str(ROOT / "scripts/run_bounded_python.py"),
                            str(SKILL / "pull_kernel_output.py"), f'{spec["account"]}/{spec["slug"]}/{spec["version"]}',
                            "-p", str(dest), "-j", "8", "--pattern", COLLECT], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    return p.returncode


def watch(batch):
    raw = (OUT / "batch.json").read_bytes()
    progress = OUT / "progress.json"
    results = json.loads(progress.read_text())["runs"] if progress.exists() else {}
    known = {a: r["state"] for a, r in results.items() if r.get("terminal")}
    (OUT / "collection").mkdir(exist_ok=True)
    while True:
        started = time.monotonic()
        active = [r for r in batch if not known.get(r["arm"], {}).get("terminal")]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for state in pool.map(query, active): known[state["arm"]] = state
        snapshot = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runs": [known[r["arm"]] for r in batch]}
        atomic_json(OUT / "status.json", snapshot)
        with (OUT / "status-history.jsonl").open("a") as f: f.write(json.dumps(snapshot) + "\n")
        for spec in batch:
            name, result = spec["arm"], results.setdefault(spec["arm"], {})
            if known[name]["terminal"] and not result.get("pulled"):
                assert shutil.disk_usage(ROOT).free > 4 * 2**30
                rc = collect(spec)
                result.update(state=known[name], terminal=True, pulled=rc == 0, audited=False)
            if not result.get("pulled") or result.get("audited") or known[name]["status"] != "COMPLETE": continue
            armout = OUT / "collection" / name; armout.mkdir(exist_ok=True)
            atomic_json(armout / "batch.json", [spec])
            p = subprocess.run([sys.executable, str(ROOT / "scripts/audit_w20.py"), "--study", TAG.lower(), "--batch",
                                str(armout / "batch.json"), "--root", str(ART), "--out", str(armout)], cwd=ROOT)
            result["audited"] = p.returncode == 0
            if result["audited"]:
                result["report_sha256"] = {n: sha(armout / n) for n in ("audit.json", "diagnostic-audit.json")}
                print(f"{TAG}_AUDITED", name, flush=True)
            else:
                print(f"{TAG}_ATTENTION audit failed", name, flush=True)
        atomic_json(progress, {"batch_sha256": hashlib.sha256(raw).hexdigest(), "runs": results})
        failed = [a for a, r in results.items() if r.get("terminal") and not r.get("audited")]
        audited = sum(r.get("audited", False) for r in results.values())
        print(f"{TAG}_PROGRESS", audited, "/", len(batch), "audited; failed", failed, flush=True)
        if all(known.get(r["arm"], {}).get("terminal") for r in batch):
            atomic_json(OUT / "closure.json", {"batch_sha256": hashlib.sha256(raw).hexdigest(), "runs": results,
                                               "full20_pass": audited, "all_classified": True})
            print(f"{TAG}_CLOSED", audited, "/", len(batch), flush=True)
            return
        startup = OUT / "startup/startup.json"
        if not startup.exists() or not json.loads(startup.read_text())["all_ready"]:
            subprocess.run([sys.executable, str(ROOT / "scripts/check_screening_startup.py"), "--batch", str(OUT / "batch.json"),
                            "--out", str(OUT / "startup"), "--entity", "21522798-uit"], cwd=ROOT)
        time.sleep(max(1, 1200 - (time.monotonic() - started)))


def main():
    global OUT, CELLS, SOURCES, ART, TAG
    ap = argparse.ArgumentParser(); ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--study", choices=sorted(STUDIES), default="w20"); a = ap.parse_args()
    out, common, art = STUDIES[a.study][:3]
    OUT, CELLS, SOURCES, ART, TAG = ROOT / out, common.CELLS, common.SOURCES, ROOT / art, a.study.upper()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "worker.lock").open("a"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not a.collect_only:
        head = git_head()
        build = builder(head)
        batch = prepare(head, build, inventory())
        launch(batch)
    batch = json.loads((OUT / "batch.json").read_text())
    assert (OUT / "launch-complete.json").exists(), "Partial push: inspect receipts, never launch blindly"
    watch(batch)
    print(f"{TAG}_PIPELINE_DONE", flush=True)


if __name__ == "__main__": main()
