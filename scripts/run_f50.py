"""F50 pipeline: the X20-chosen method at 50 rounds, full resume state every round, automatic resume sessions.

Reuses the W/X stages (shared npy cache, quota inventory, push gates, collector) with F50 receipts under runs/f-study.
A session that ends COMPLETE before round 50 (budget stop: Kaggle's 12 h cap) gets a resume session on the same owner
(`<slug>-c2`, ...) that mounts the previous output, continues from its last saved round and carries the whole run
directory; the last session is pulled and audited over rounds 1..50. An ERROR session is pulled and left for manual
classification (owner rule), never auto-resumed.

    python scripts/run_f50.py                 # owner 03/11: run the proposed method at 50 rounds, checkpoint every round
    python scripts/run_f50.py --collect-only
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import f50_common, run_w20 as W
from scripts.launch_repeat20 import owned_refs
from scripts.watch_kaggle_batch import atomic_json, query

OUT, ART, TAG = ROOT / "runs/f-study", ROOT / "artifacts/f50", "F50"
CAP = 11.0              # h, session 1: measured X20 ~740 s/round + per-round state save; Kaggle hard cap 12 h incl. output save
RESUME_CAP = 2.5        # h, a resume session: the few rounds left after the budget stop + setup and the 10 GB copy-in
NEED = (CAP + RESUME_CAP) * 1.15
MAX_SESSIONS = 3
AUTH = "Owner 03/11: run the proposed method (core+bank) at 50 rounds with core as bank ablation; no mixed architecture; checkpoint every round"
SCOPE = "X20 core/bank recipe unchanged, rounds 1..50 of cosine50, --save-every 1; core seed 43 deferred to the 07/11 quota reset"


def configure():
    W.OUT, W.ART, W.TAG, W.CELLS, W.SOURCES = OUT, ART, TAG, f50_common.CELLS, f50_common.SOURCES


def args_for(run, hp):
    return shlex.join(["scripts/run_q20.py", "--method", "mobihet_wstudy", "--out", f"/kaggle/working/{run}", "--width", "1",
                       "--rounds", "50", "--save-every", "1", "--compile", "--wandb", "mobihet-v4", "--max-hours", "BUDGET",
                       "--cache", "", "--set", *hp])


def notebook(owner, kernel, run, hp, head, build, cap, resume=""):
    subprocess.run([sys.executable, str(ROOT / "scripts/gen_notebook.py"), "--owner", owner, "--slug", kernel, "--run", run,
                    "--sha", head, "--max-hours", str(cap), "--args", args_for(run, hp), "--npy-kernel", build["ref"],
                    "--npy-manifest-sha256", build["manifest_sha256"], "--overlay", "methods/mobihet.py",
                    *(["--resume-kernel", resume] if resume else [])], check=True, capture_output=True, cwd=ROOT)
    folder = ROOT / "notebooks" / kernel; nb_path = folder / (kernel + ".ipynb"); metadata = folder / "kernel-metadata.json"
    nb = json.loads(nb_path.read_text())
    for c in nb["cells"]:   # same compile-thread setting as R/V/W/X launches
        if c["cell_type"] == "code": c["source"] = "".join(c["source"]).replace(
            'TORCHINDUCTOR_COMPILE_THREADS="2"', 'TORCHINDUCTOR_COMPILE_THREADS="1"').splitlines(keepends=True)
    nb_path.write_text(json.dumps(nb, indent=1) + "\n")
    subprocess.run([sys.executable, str(W.SKILL / "embed_wandb_key.py"), str(nb_path), "--metadata", str(metadata)],
                   check=True, capture_output=True)
    meta, runtime = json.loads(metadata.read_text()), json.loads((ROOT / "knowledge/runtime.json").read_text())
    assert meta["is_private"] is True and meta["id"] == f"{owner}/{kernel}"
    assert meta["kernel_sources"] == [k for k in (resume, build["ref"]) if k]
    assert meta["machine_shape"] == runtime["machine_shape"] and meta["docker_image"] == runtime["docker_image"]
    text = nb_path.read_text()
    assert "--save-every 1" in text and "--until" not in text and ("resumed from" in text) == bool(resume)
    for c in json.loads(text)["cells"]:
        if c["cell_type"] == "code": compile("".join(c["source"]), str(nb_path), "exec")
    return W.sha(nb_path), W.sha(metadata)


def prepare(head, build, quota):
    if (OUT / "batch.json").exists(): return json.loads((OUT / "batch.json").read_text())
    smoke = json.loads((OUT / "preflight.json").read_text())
    assert smoke["source_unchanged"] and all(W.sha(ROOT / n) == h for n, h in smoke["source_sha256"].items())
    assert {r["cell"] for r in smoke["reports"] if r["history_cms_snapshots_state_bitexact"]} == set(f50_common.CELLS)
    import xml.etree.ElementTree as ET
    assert all(int(s.attrib[k]) == 0 for s in ET.parse(OUT / "tests.xml").getroot().findall("testsuite")
               for k in ("failures", "errors", "skipped"))
    owners = [q["user"] for q in sorted(quota["accounts"], key=lambda q: (-q.get("gpu_left", 0), q["user"]))
              if not (q.get("inventory_error") or q["active_recent"] or q["unknown_recent"]) and q.get("gpu_left", 0) >= NEED]
    plan = f50_common.PLAN
    assert len(owners) >= len(plan), f"{len(plan)} owners with >= {NEED:.1f} h required, have {owners}; do not cut rounds"
    batch = []
    for index, ((cell, seed), owner) in enumerate(zip(plan, owners)):
        spec = f50_common.CELLS[cell]; hp = f50_common.hp_for(cell, seed)
        arm = f"F{index}"; slug = f"mh-{arm.lower()}-{cell}-s{seed}-r50"
        nb_sha, meta_sha = notebook(owner, slug, slug, hp, head, build, CAP)
        batch.append({"arm": arm, "cell": cell, "mode": spec["mode"], "r_kind": spec["r_kind"], "account": owner, "slug": slug,
                      "version": 1, "hp": hp, "base_sha": head, "source_sha256": W.sha(ROOT / "methods/mobihet.py"),
                      "overlay_sha256": {"methods/mobihet.py": W.sha(ROOT / "methods/mobihet.py")}, "notebook_sha256": nb_sha,
                      "metadata_sha256": meta_sha, "cap_hours": CAP, "resume_cap_hours": RESUME_CAP, "endpoint": 50,
                      "lr_schedule_rounds": 50, "save_every": 1, "npy_kernel": build["ref"],
                      "npy_manifest_sha256": build["manifest_sha256"], "bootstrap_total": W.COMPACT_BOOTSTRAP})
    atomic_json(OUT / "batch.json", batch)
    atomic_json(ROOT / "docs/research/f50-launch.json", {"batch_sha256": W.sha(OUT / "batch.json"), "batch": batch,
        "preflight_sha256": W.sha(OUT / "preflight.json"), "quota_sha256": W.sha(OUT / "quota.json"), "builder": build,
        "deferred": f50_common.DEFERRED, "authorization": AUTH, "scope": SCOPE})
    print(f"{TAG}_NOTEBOOKS_LOCKED", len(batch), flush=True)
    return batch


def sessions_of(batch):
    path = OUT / "sessions.json"
    return json.loads(path.read_text()) if path.exists() else {r["arm"]: [r["slug"]] for r in batch}


def collect(spec, kernel):
    dest = ART / spec["slug"]; dest.mkdir(parents=True, exist_ok=True)
    with (OUT / "collection" / (kernel + ".collect.log")).open("w") as log:
        p = subprocess.run([*W.cli(spec["account"])[:4], sys.executable, str(ROOT / "scripts/run_bounded_python.py"),
                            str(W.SKILL / "pull_kernel_output.py"), f'{spec["account"]}/{kernel}/1', "-p", str(dest), "-j", "8",
                            "--pattern", W.COLLECT], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    return p.returncode


def push_resume(spec, sessions, build):
    """Resume session n+1 on the same owner from session n's output (state.pt of its last saved round)."""
    owner, prev = spec["account"], sessions[-1]
    kernel = f'{spec["slug"]}-c{len(sessions) + 1}'
    assert W.sha(ROOT / "methods/mobihet.py") == spec["source_sha256"], "overlay changed since launch"
    q = W.ka().account_quota(owner)
    assert q["gpu_left"] >= RESUME_CAP * 1.15, (owner, "quota for the resume session")
    assert f"{owner}/{kernel}" not in owned_refs(W.cli(owner), owner), (kernel, "slug exists")
    nb_sha, _ = notebook(owner, kernel, spec["slug"], spec["hp"], spec["base_sha"], build, RESUME_CAP, resume=f"{owner}/{prev}")
    with (OUT / (kernel + ".push-intent.json")).open("x") as f:
        json.dump({"ref": f"{owner}/{kernel}/1", "resume_from": f"{owner}/{prev}", "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, f)
    p = subprocess.run(["timeout", "--foreground", "--kill-after=10s", "240s", *W.cli(owner), "kernels", "push", "-p",
                        str(ROOT / "notebooks" / kernel)], capture_output=True, text=True)
    log = OUT / (kernel + ".push.log"); log.write_text(p.stdout + p.stderr); log.chmod(0o600)
    assert p.returncode == 0 and re.search(r"Kernel version 1 successfully pushed", p.stdout), f"Push failed/ambiguous {kernel}"
    atomic_json(OUT / (kernel + ".push.json"), {"ref": f"{owner}/{kernel}/1", "notebook_sha256": nb_sha, "resume_from": prev,
                                                 "quota": q, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    print(f"{TAG}_RESUME_PUSHED", spec["arm"], kernel, flush=True)
    return kernel


def last_round(spec):
    rows = (ART / spec["slug"] / spec["slug"] / "rounds.jsonl").read_text().splitlines()
    return json.loads(rows[-1])["round"] if rows else 0


def watch(batch, build):
    raw = (OUT / "batch.json").read_bytes()
    sessions, progress = sessions_of(batch), OUT / "progress.json"
    results = json.loads(progress.read_text())["runs"] if progress.exists() else {}
    (OUT / "collection").mkdir(exist_ok=True)
    while True:
        started = time.monotonic()
        live = [dict(r, slug=sessions[r["arm"]][-1], version=1) for r in batch if not results.get(r["arm"], {}).get("final")]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: states = {s["arm"]: s for s in pool.map(query, live)}
        snapshot = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "sessions": sessions, "runs": list(states.values())}
        atomic_json(OUT / "status.json", snapshot)
        with (OUT / "status-history.jsonl").open("a") as f: f.write(json.dumps(snapshot) + "\n")
        for spec in batch:
            name, kernel = spec["arm"], sessions[spec["arm"]][-1]
            result = results.setdefault(name, {})
            state = states.get(name)
            if result.get("final") or not state or not state["terminal"]: continue
            assert W.shutil.disk_usage(ROOT).free > 6 * 2**30
            rc = collect(spec, kernel)
            result.setdefault("sessions", {})[kernel] = {"state": state, "pulled": rc == 0}
            if rc or state["status"] != "COMPLETE":
                result.update(final=True, audited=False, reason=f"{kernel} {state['status']} pulled={rc == 0}")
                print(f"{TAG}_ATTENTION", name, result["reason"], flush=True); continue
            done = last_round(spec)
            result["sessions"][kernel]["last_round"] = done
            if done < 50:
                if len(sessions[name]) >= MAX_SESSIONS:
                    result.update(final=True, audited=False, reason=f"round {done} after {MAX_SESSIONS} sessions")
                    print(f"{TAG}_ATTENTION", name, result["reason"], flush=True); continue
                sessions[name].append(push_resume(spec, sessions[name], build)); atomic_json(OUT / "sessions.json", sessions)
                continue
            armout = OUT / "collection" / name; armout.mkdir(exist_ok=True)
            atomic_json(armout / "batch.json", [dict(spec, sessions=sessions[name])])
            p = subprocess.run([sys.executable, str(ROOT / "scripts/audit_f50.py"), "--batch", str(armout / "batch.json"),
                                "--root", str(ART), "--out", str(armout)], cwd=ROOT)
            result.update(final=True, audited=p.returncode == 0)
            if result["audited"]:
                result["report_sha256"] = {n: W.sha(armout / n) for n in ("audit.json", "diagnostic-audit.json")}
            print(f"{TAG}_AUDITED" if result["audited"] else f"{TAG}_ATTENTION audit failed", name, flush=True)
        atomic_json(progress, {"batch_sha256": hashlib.sha256(raw).hexdigest(), "runs": results})
        audited = sum(r.get("audited", False) for r in results.values())
        print(f"{TAG}_PROGRESS", audited, "/", len(batch), "audited; sessions", {a: len(s) for a, s in sessions.items()}, flush=True)
        if all(results.get(r["arm"], {}).get("final") for r in batch):
            atomic_json(OUT / "closure.json", {"batch_sha256": hashlib.sha256(raw).hexdigest(), "runs": results,
                                               "sessions": sessions, "full50_pass": audited, "all_classified": True})
            print(f"{TAG}_CLOSED", audited, "/", len(batch), flush=True)
            return
        startup = OUT / "startup/startup.json"
        if not startup.exists() or not json.loads(startup.read_text())["all_ready"]:
            subprocess.run([sys.executable, str(ROOT / "scripts/check_screening_startup.py"), "--batch", str(OUT / "batch.json"),
                            "--out", str(OUT / "startup"), "--entity", "21522798-uit"], cwd=ROOT)
        time.sleep(max(1, 1200 - (time.monotonic() - started)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--collect-only", action="store_true"); a = ap.parse_args()
    configure(); OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "worker.lock").open("a"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    build = json.loads((ROOT / "runs/w-study/builder.json").read_text())   # shared npy cache; every kernel re-verifies it
    if not a.collect_only:
        batch = prepare(W.git_head(), build, W.inventory())
        W.launch(batch)
    batch = json.loads((OUT / "batch.json").read_text())
    assert (OUT / "launch-complete.json").exists(), "Partial push: inspect receipts, never launch blindly"
    watch(batch, build)
    print(f"{TAG}_PIPELINE_DONE", flush=True)


if __name__ == "__main__": main()
