"""Y50 pipeline (docs/research/screening-y50.md), one detached worker per wave, receipts in runs/y-study/wave<N>.

Wave 1: oracle s42 r50, prior s42/s43 to r20; the locked prior rule then either stops prior or resumes it to r50 on the same owner.
Wave 2: Core-Q and Core-floor, 2 seeds r50, started at once on what quota is left (owner 06/11).
A budget stop gets a resume session on the same owner once its quota covers the rounds left, so short sessions continue after the
07/11 reset; an ERROR session is pulled and left for manual classification.

    python scripts/run_y50.py --wave 1 [--collect-only]
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
from scripts import f50_common, run_f50 as F, y50_common as Y
from scripts.launch_repeat20 import owned_refs
from scripts.watch_kaggle_batch import atomic_json, query

W = F.W
STUDY, ART = ROOT / "runs/y-study", ROOT / "artifacts/y50"
CAP, ROUND_H, SETUP_H = 11.0, .21, 1.0       # h: Kaggle-safe session; F3 measured ~740 s/round; setup + 10 GB copy-in + state save
MAX_SESSIONS = 6
AUTH = "Owner 06/11: test the first 3 candidates of pipeline-review to improve Core; session chaining allowed when quota is short"
OUT = None


def resume_cap(done, target): return min(CAP, (target - done) * ROUND_H + SETUP_H)


def overlays(cell):
    return ["methods/mobihet.py"] + (["methods/mobihet_prior.py"] if Y.CELLS[cell]["method"] == "mobihet_prior" else [])


def args_for(run, hp, method, until):
    return shlex.join(["scripts/run_q20.py", "--method", method, "--out", f"/kaggle/working/{run}", "--width", "1", "--rounds", "50",
                       *(["--until", str(until)] if until < 50 else []), "--save-every", "1", "--compile", "--wandb", "mobihet-v4",
                       "--max-hours", "BUDGET", "--cache", "", "--set", *hp])


def notebook(owner, kernel, run, cell, hp, head, build, cap, until, resume=""):
    subprocess.run([sys.executable, str(ROOT / "scripts/gen_notebook.py"), "--owner", owner, "--slug", kernel, "--run", run, "--sha", head,
                    "--max-hours", str(cap), "--args", args_for(run, hp, Y.CELLS[cell]["method"], until), "--npy-kernel", build["ref"],
                    "--npy-manifest-sha256", build["manifest_sha256"], "--overlay", *overlays(cell),
                    *(["--resume-kernel", resume] if resume else [])], check=True, capture_output=True, cwd=ROOT)
    folder = ROOT / "notebooks" / kernel; path = folder / (kernel + ".ipynb"); metadata = folder / "kernel-metadata.json"
    nb = json.loads(path.read_text())
    tar = f"/kaggle/working/{run}/inductor-cache.tar"   # one output file instead of ~6.6k; local warm start was bit-exact
    warm = f'''if os.path.exists("{tar}"): print("inductor cache restored", subprocess.run(["tar", "-xf", "{tar}", "-C", "/kaggle/temp"]).returncode)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/kaggle/temp/inductor"
rc = subprocess.run(cmd, shell=True).returncode
subprocess.run(["tar", "-cf", "{tar}", "-C", "/kaggle/temp", "inductor"])
'''
    for c in nb["cells"]:   # same compile-thread setting as F50
        if c["cell_type"] == "code": c["source"] = "".join(c["source"]).replace(
            'TORCHINDUCTOR_COMPILE_THREADS="2"', 'TORCHINDUCTOR_COMPILE_THREADS="1"').replace(
            'rc = subprocess.run(cmd, shell=True).returncode\n', warm).splitlines(keepends=True)
    path.write_text(json.dumps(nb, indent=1) + "\n")
    subprocess.run([sys.executable, str(W.SKILL / "embed_wandb_key.py"), str(path), "--metadata", str(metadata)], check=True, capture_output=True)
    meta, runtime = json.loads(metadata.read_text()), json.loads((ROOT / "knowledge/runtime.json").read_text())
    assert meta["is_private"] is True and meta["id"] == f"{owner}/{kernel}"
    assert meta["kernel_sources"] == [k for k in (resume, build["ref"]) if k]
    assert meta["machine_shape"] == runtime["machine_shape"] and meta["docker_image"] == runtime["docker_image"]
    text = path.read_text()
    assert "TORCHINDUCTOR_CACHE_DIR" in text and "inductor-cache.tar" in text
    assert "--save-every 1" in text and ("--until" in text) == (until < 50) and ("resumed from" in text) == bool(resume)
    for c in json.loads(text)["cells"]:
        if c["cell_type"] == "code": compile("".join(c["source"]), str(path), "exec")
    return W.sha(path), W.sha(metadata)


def prepare(wave, build):
    path = OUT / "batch.json"
    if path.exists(): return json.loads(path.read_text())
    smoke = json.loads((STUDY / "preflight.json").read_text())
    assert smoke["source_unchanged"] and all(W.sha(ROOT / n) == h for n, h in smoke["source_sha256"].items())
    assert {r["cell"] for r in smoke["reports"] if r["history_cms_snapshots_state_bitexact"]} == set(Y.CELLS)
    f2 = next(r for r in json.loads((ROOT / "runs/f-study/batch.json").read_text()) if r["arm"] == "F2")
    assert W.sha(ROOT / "methods/mobihet.py") == f2["source_sha256"]
    batch = []
    for arm, cell, seed, owner, until, cap in Y.WAVES[wave]:
        slug = f"mh-{arm.lower()}-{cell}-s{seed}-r50"; hp = Y.hp_for(cell, seed)
        nb, meta = notebook(owner, slug, slug, cell, hp, f2["base_sha"], build, cap, until)
        total = 4 * 16 * (1 + 77 + 77 * 78 // 2) + 4 * 16 * (77 + 77 * 78 // 2) + 4 * 16 if cell == "oracle" else W.COMPACT_BOOTSTRAP
        batch.append({"arm": arm, "cell": cell, "seed": seed, "mode": "stream", "r_kind": "live", "method": Y.CELLS[cell]["method"],
                      "account": owner, "slug": slug, "version": 1, "hp": hp, "base_sha": f2["base_sha"],
                      "source_sha256": f2["source_sha256"], "overlay_sha256": {n: W.sha(ROOT / n) for n in overlays(cell)},
                      "notebook_sha256": nb, "metadata_sha256": meta, "cap_hours": cap, "until": until, "endpoint": 50,
                      "lr_schedule_rounds": 50, "save_every": 1, "npy_kernel": build["ref"], "npy_manifest_sha256": build["manifest_sha256"],
                      "bootstrap_total": total})
    atomic_json(path, batch)
    atomic_json(ROOT / f"docs/research/y50-wave{wave}-launch.json", {"batch_sha256": W.sha(path), "batch": batch, "builder": build,
        "preflight_sha256": W.sha(STUDY / "preflight.json"), "authorization": AUTH, "scope": "screening-y50.md"})
    print("Y50_NOTEBOOKS_LOCKED", wave, len(batch), flush=True)
    return batch


def quota_ok(owner, need):
    return W.ka().account_quota(owner)["gpu_left"] >= need


def short_cap(left, wanted):
    cap = min(wanted, left / 1.15)
    return cap if cap >= SETUP_H + ROUND_H else 0.0


def push_resume(spec, sessions, build, cap, until):
    owner, prev = spec["account"], sessions[-1]
    kernel = f'{spec["slug"]}-c{len(sessions) + 1}'
    if (OUT / (kernel + ".push.json")).exists(): return kernel   # pushed before a worker crash: adopt, never push twice
    assert quota_ok(owner, cap * 1.15), "quota changed before push"
    assert all(W.sha(ROOT / n) == h for n, h in spec["overlay_sha256"].items()), "overlay changed since launch"
    assert f"{owner}/{kernel}" not in owned_refs(W.cli(owner), owner), (kernel, "slug exists")
    nb, _ = notebook(owner, kernel, spec["slug"], spec["cell"], spec["hp"], spec["base_sha"], build, cap, until, resume=f"{owner}/{prev}")
    with (OUT / (kernel + ".push-intent.json")).open("x") as f:
        json.dump({"ref": f"{owner}/{kernel}/1", "resume_from": f"{owner}/{prev}", "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, f)
    p = subprocess.run(["timeout", "--foreground", "--kill-after=10s", "240s", *W.cli(owner), "kernels", "push", "-p",
                        str(ROOT / "notebooks" / kernel)], capture_output=True, text=True)
    log = OUT / (kernel + ".push.log"); log.write_text(p.stdout + p.stderr); log.chmod(0o600)
    assert p.returncode == 0 and re.search(r"Kernel version 1 successfully pushed", p.stdout), f"Push failed/ambiguous {kernel}"
    atomic_json(OUT / (kernel + ".push.json"), {"ref": f"{owner}/{kernel}/1", "notebook_sha256": nb, "resume_from": prev,
                                                 "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    print("Y50_RESUME_PUSHED", spec["arm"], kernel, flush=True)
    return kernel


def run_audit(spec, sessions, endpoint):
    out = OUT / "collection" / f'{spec["arm"]}-r{endpoint}'; out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "batch.json", [dict(spec, sessions=sessions)])
    p = subprocess.run([sys.executable, str(ROOT / "scripts/audit_y50.py"), "--batch", str(out / "batch.json"), "--root", str(ART),
                        "--out", str(out), "--endpoint", str(endpoint)], cwd=ROOT)
    return p.returncode == 0


def run_dir(spec): return ART / spec["slug"] / spec["slug"]


def screen_prior(batch, results, sessions):
    """Both prior arms at r20: audit, then the locked rule. Returns True when prior goes on to r50."""
    arms = [r for r in batch if r["cell"] == "prior"]
    if not all(run_audit(s, sessions[s["arm"]], Y.SCREEN) for s in arms):
        for s in arms: results[s["arm"]].update(final=True, audited=False, reason="r20 audit failed")
        print("Y50_ATTENTION prior r20 audit failed", flush=True); return False
    deltas = [Y.delta(run_dir(s), s["seed"], Y.SCREEN) for s in arms]
    rule = Y.prior_rule(deltas)
    atomic_json(OUT / "prior-r20.json", {"deltas": {s["arm"]: d for s, d in zip(arms, deltas)}, "rule": rule})
    print("Y50_PRIOR_R20", json.dumps(rule), flush=True)
    return rule["pass"]


def watch(batch, build):
    raw = (OUT / "batch.json").read_bytes()
    path, progress = OUT / "sessions.json", OUT / "progress.json"
    sessions = json.loads(path.read_text()) if path.exists() else {r["arm"]: [r["slug"]] for r in batch}
    results = json.loads(progress.read_text())["runs"] if progress.exists() else {r["arm"]: {} for r in batch}
    (OUT / "collection").mkdir(exist_ok=True)
    while True:
        started = time.monotonic()
        live = [dict(r, slug=sessions[r["arm"]][-1], version=1) for r in batch
                if not results[r["arm"]].get("final") and not results[r["arm"]].get("resume") and not results[r["arm"]].get("screened")]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: states = {s["arm"]: s for s in pool.map(query, live)}
        snapshot = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "sessions": sessions, "runs": list(states.values())}
        atomic_json(OUT / "status.json", snapshot)
        with (OUT / "status-history.jsonl").open("a") as f: f.write(json.dumps(snapshot) + "\n")
        for spec in batch:
            name, kernel = spec["arm"], sessions[spec["arm"]][-1]
            result, state = results[name], states.get(name)
            if not state or not state["terminal"]: continue
            assert W.shutil.disk_usage(ROOT).free > 6 * 2**30
            rc = F.collect(spec, kernel)
            result.setdefault("sessions", {})[kernel] = {"state": state, "pulled": rc == 0}
            if rc or state["status"] != "COMPLETE":
                result.update(final=True, audited=False, reason=f"{kernel} {state['status']} pulled={rc == 0}")
                print("Y50_ATTENTION", name, result["reason"], flush=True); continue
            done = F.last_round(spec)
            result["sessions"][kernel]["last_round"] = done
            target = 50 if result.get("past_screen") else spec["until"]
            if done < target:
                if not run_audit(spec, sessions[name], done):
                    result.update(final=True, audited=False, reason=f"partial r{done} audit failed")
                    continue
                result["partial_audited_round"] = done
                if len(sessions[name]) >= MAX_SESSIONS:
                    result.update(final=True, audited=False, reason=f"round {done} after {MAX_SESSIONS} sessions")
                    print("Y50_ATTENTION", name, result["reason"], flush=True); continue
                result["resume"] = {"cap": resume_cap(done, target), "until": target, "after": kernel}
            elif target == Y.SCREEN:
                result["screened"] = True
            else:
                result.update(final=True, audited=run_audit(spec, sessions[name], 50))
                print("Y50_AUDITED" if result["audited"] else "Y50_ATTENTION audit failed", name, flush=True)
        prior = [r for r in batch if r["cell"] == "prior"]
        if any(results[r["arm"]].get("final") for r in prior):      # a lost partner leaves no two-seed rule to read
            for r in prior:
                if results[r["arm"]].pop("screened", None):
                    results[r["arm"]].update(final=True, audited=False, reason="partner prior arm failed before r20")
                    print("Y50_ATTENTION", r["arm"], "partner prior arm failed", flush=True)
        if prior and all(results[r["arm"]].get("screened") for r in prior):
            go = screen_prior(batch, results, sessions)
            for r in prior:
                result = results[r["arm"]]; result.pop("screened")
                if go: result.update(past_screen=True, resume={"cap": resume_cap(Y.SCREEN, 50), "until": 50, "after": sessions[r["arm"]][-1]})
                elif not result.get("final"): result.update(final=True, audited=True, reason="prior rule failed at r20; stopped by the locked rule")
        for spec in batch:
            pending = results[spec["arm"]].get("resume")
            if pending and sessions[spec["arm"]][-1] != pending["after"]:
                results[spec["arm"]].pop("resume")                      # already resumed before a worker crash
            elif pending:
                done = F.last_round(spec)
                if results[spec["arm"]].get("partial_audited_round") != done:
                    if not run_audit(spec, sessions[spec["arm"]], done):
                        results[spec["arm"]].update(final=True, audited=False, reason=f"partial r{done} audit failed")
                        results[spec["arm"]].pop("resume")
                        continue
                    results[spec["arm"]]["partial_audited_round"] = done
                cap = short_cap(W.ka().account_quota(spec["account"])["gpu_left"], pending["cap"])
                if not cap: continue
                sessions[spec["arm"]].append(push_resume(spec, sessions[spec["arm"]], build, cap, pending["until"]))
                atomic_json(path, sessions); results[spec["arm"]].pop("resume")
                atomic_json(progress, {"batch_sha256": hashlib.sha256(raw).hexdigest(), "runs": results})
        atomic_json(progress, {"batch_sha256": hashlib.sha256(raw).hexdigest(), "runs": results})
        print("Y50_PROGRESS", time.strftime("%H:%M"), {a: len(s) for a, s in sessions.items()},
              {a: ("final" if r.get("final") else "wait_quota" if r.get("resume") else "live") for a, r in results.items()}, flush=True)
        if all(results[r["arm"]].get("final") for r in batch):
            atomic_json(OUT / "closure.json", {"batch_sha256": hashlib.sha256(raw).hexdigest(), "runs": results, "sessions": sessions})
            print("Y50_CLOSED", flush=True); return
        startup = OUT / "startup/startup.json"
        if not startup.exists() or not json.loads(startup.read_text())["all_ready"]:
            subprocess.run([sys.executable, str(ROOT / "scripts/check_screening_startup.py"), "--batch", str(OUT / "batch.json"),
                            "--out", str(OUT / "startup"), "--entity", "21522798-uit"], cwd=ROOT)
        time.sleep(max(1, 1200 - (time.monotonic() - started)))


def main():
    global OUT
    ap = argparse.ArgumentParser(); ap.add_argument("--wave", type=int, required=True, choices=(1, 2))
    ap.add_argument("--collect-only", action="store_true"); ap.add_argument("--lock-only", action="store_true"); a = ap.parse_args()
    OUT = STUDY / f"wave{a.wave}"; OUT.mkdir(parents=True, exist_ok=True)
    F.OUT, F.ART, F.TAG = OUT, ART, "Y50"; W.OUT, W.ART, W.TAG, W.CELLS, W.SOURCES = OUT, ART, "Y50", Y.CELLS, f50_common.SOURCES
    lock = (OUT / "worker.lock").open("a"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    build = json.loads((ROOT / "runs/w-study/builder.json").read_text())
    if not a.collect_only:
        batch = prepare(a.wave, build)
        if a.lock_only: return
        W.launch(batch)
    assert (OUT / "launch-complete.json").exists(), "Partial push: inspect receipts, never launch blindly"
    watch(json.loads((OUT / "batch.json").read_text()), build)
    print("Y50_PIPELINE_DONE", a.wave, flush=True)


if __name__ == "__main__": main()
