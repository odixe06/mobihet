"""Logit-adjustment sim batch, DAGSNet (24/11/2026, docs/research/screening-la.md): L4 with la_mode full / off / absent / mask,
seeds 42 and 43; fold train -> fold val, 200 vehicles, cosine20, compact engine. Reuses scripts/run_sim_budget.py (notebook
generation, launch, collect); only the arms, the seeds and the gate differ.

    python scripts/run_sim_la.py quota | launch | collect | gate
    python scripts/run_sim_la.py relaunch <owner> ...   # rows Kaggle refused (2-session limit) -> other owners
    python scripts/audit_sim_batch.py la      # before gate

Rule locked before push (screening-la.md): a candidate (A3m, A3x) PASSES when its pooled Macro-F1 on fold val, mean of
rounds 16-20 averaged over seeds 42 and 43, is >= that of A3 (la_mode=off) in this batch. Every other number is report-only.
"""
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sim_budget as B  # noqa: E402

N4 = ([[64, 64], [16, 16], [16, 16]], [64, 37, 37])
L4 = ["real_select=top", "compact=true"]
ARMS = {"L4": ("l4", "control", []),                                   # arm: (tag, role, extra hp on top of L4)
        "A3": ("la-off", "control", ["la_mode=off"]),
        "A3m": ("la-absent", "candidate", ["la_mode=absent"]),
        "A3x": ("la-mask", "candidate", ["la_mode=mask"])}
SEEDS = (42, 43)
OUT, ART = ROOT / "runs/sim-kaggle-la", ROOT / "artifacts/sim-kaggle-la"
B.OUT, B.ART = OUT, ART
B.W.OUT, B.W.ART, B.W.TAG = OUT, ART, "LA"


def base_sha():
    """origin/main (local main is ahead with docs/tooling commits, not pushed): every runtime file the notebook takes from
    the clone must equal its bytes at that SHA; the only differing runtime file travels as an overlay."""
    head = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/main"], cwd=ROOT, text=True).split()[0]
    runtime = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", head, "core", "methods"], cwd=ROOT,
                                      text=True).split() + B.W.SOURCES + ["scripts/sim_fold_val.py"]
    for name in sorted(set(runtime) - set(B.OVERLAYS)):
        assert subprocess.check_output(["git", "show", f"{head}:{name}"], cwd=ROOT) == (ROOT / name).read_bytes(), name
    return head


def notebook(owner, slug, h, head, builder, runtime):
    """Generate the run's notebook for owner; returns (notebook sha256, metadata sha256)."""
    args = shlex.join(["scripts/sim_fold_val.py", "run", "--n", "200", "--method", "mobihet_budget", "--out",
                       f"/kaggle/working/{slug}", "--rounds", "20", "--compile", "--max-hours", "BUDGET", "--cache", "",
                       "--set", *h])
    subprocess.run([sys.executable, "scripts/gen_notebook.py", "--owner", owner, "--slug", slug, "--sha", head, "--run", slug,
                    "--max-hours", str(B.CAP), "--args", args, "--npy-kernel", builder["ref"],
                    "--npy-manifest-sha256", builder["manifest_sha256"], "--overlay", *B.OVERLAYS],
                   cwd=ROOT, check=True, capture_output=True)
    folder = ROOT / "notebooks" / slug
    nb, meta = folder / (slug + ".ipynb"), folder / "kernel-metadata.json"
    doc = json.loads(nb.read_text())
    for cell in doc["cells"]:                                         # no W&B in the simulator: no credential embedded
        src = "".join(cell["source"])
        if "BEGIN INLINE WANDB CREDENTIAL" in src:
            cell["source"] = ['print("W&B off: simulator sweep")\n']
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(nb), "exec")
    nb.write_text(json.dumps(doc, indent=1) + "\n")
    metadata = json.loads(meta.read_text())
    assert metadata["is_private"] and metadata["id"] == f"{owner}/{slug}" and metadata["kernel_sources"] == [builder["ref"]]
    assert all(metadata[k] == runtime[k] for k in ("machine_shape", "docker_image"))
    return B.W.sha(nb), B.W.sha(meta)


REFUSED = "Maximum batch GPU session count of 2 reached"


def relaunch(owners):
    """Rows whose push Kaggle refused outright (REFUSED in the push log, no push.json: no kernel was created) move to the
    next owner of `owners` with quota for the cap; hp, base SHA, overlays and slug stay locked. An owner that refuses too is
    dropped. Every move is appended to relaunch.jsonl; the old intent/log keep their bytes as *.failed-<owner>."""
    import re
    import time
    W = B.W
    batch = json.loads((OUT / "batch.json").read_text())
    builder = json.loads((ROOT / "runs/w-study/builder.json").read_text())
    runtime = json.loads((ROOT / "knowledge/runtime.json").read_text())
    k = W.ka()
    free = []
    for o in owners:
        q = k.account_quota(o)
        used = sum(r["cap_hours"] for r in batch if r["account"] == o and (OUT / (r["slug"] + ".push.json")).exists())
        free += [o] * max(0, min(2, int((q["gpu_left"] - used - 1 / 60) // B.CAP)))
    for row in batch:
        slug = row["slug"]
        if (OUT / (slug + ".push.json")).exists():
            continue
        intent, log = OUT / (slug + ".push-intent.json"), OUT / (slug + ".push.log")
        if intent.exists():
            assert log.exists() and REFUSED in log.read_text(), ("inspect before retry", slug)
            for f in (intent, log):
                f.rename(f.with_name(f.name + f".failed-{row['account']}"))
        while True:
            assert free, ("no free owner left", slug)
            owner = free.pop(0)
            refs = W.owned_refs(W.cli(owner), owner)
            assert f"{owner}/{slug}" not in refs, (owner, "slug exists")
            nb_sha, meta_sha = notebook(owner, slug, row["hp"], row["base_sha"], builder, runtime)
            old = row["account"]
            row.update(account=owner, notebook_sha256=nb_sha, metadata_sha256=meta_sha)
            B.atomic_json(OUT / "batch.json", batch)
            with (OUT / "relaunch.jsonl").open("a") as f:
                f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "slug": slug, "from": old, "to": owner,
                                    "notebook_sha256": nb_sha}) + "\n")
            ref = f"{owner}/{slug}/1"
            with intent.open("x") as f:
                json.dump({"ref": ref, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, f)
            p = subprocess.run(["timeout", "--foreground", "--kill-after=10s", "240s", *W.cli(owner), "kernels", "push", "-p",
                                str(ROOT / "notebooks" / slug)], capture_output=True, text=True)
            log.write_text(p.stdout + p.stderr)
            log.chmod(0o600)
            if p.returncode == 0 and re.search(r"Kernel version 1 successfully pushed", p.stdout):
                B.atomic_json(OUT / (slug + ".push.json"), {"ref": ref, "notebook_sha256": nb_sha,
                                                            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
                print("LA_PUSHED", row["arm"], ref, flush=True)
                break
            assert REFUSED in log.read_text(), ("push failed/ambiguous: inspect", slug, owner)
            print("LA_REFUSED", owner, slug, flush=True)
            free = [o for o in free if o != owner]
            for f in (intent, log):
                f.rename(f.with_name(f.name + f".failed-{owner}"))


def prepare(quota):
    if (OUT / "batch.json").exists():
        return json.loads((OUT / "batch.json").read_text())
    W, head = B.W, base_sha()
    builder = json.loads((ROOT / "runs/w-study/builder.json").read_text())
    runtime = json.loads((ROOT / "knowledge/runtime.json").read_text())
    runs = [(arm, seed) for seed in SEEDS for arm in ARMS]
    slots = []
    for q in sorted(quota["accounts"], key=lambda q: (-q.get("gpu_left", 0), q["user"])):
        if q.get("inventory_error") or q.get("unknown_recent") or q.get("gpu_reserved", 0):
            continue
        slots += [q["user"]] * max(0, min(2 - len(q.get("active_recent", [])), int((q.get("gpu_left", 0) - 1 / 60) // B.CAP)))
    assert len(slots) >= len(runs), f"need {len(runs)} slots, have {len(slots)}"
    cost = B.costs()["N4"][0]
    batch = []
    for (arm, seed), owner in zip(runs, slots):
        tag, role, extra = ARMS[arm]
        slug = f"mh-sim-la-{tag}-s{seed}-r20"
        h = B.arm_hp(*N4, L4 + extra + [f"seed={seed}"])
        nb_sha, meta_sha = notebook(owner, slug, h, head, builder, runtime)
        batch.append(dict(arm=f"{arm}-s{seed}", variant=arm, role=role, seed=seed, method="mobihet_budget", account=owner, slug=slug, version=1, hp=h,
                          base_sha=head, overlay_sha256={p: W.sha(ROOT / p) for p in B.OVERLAYS},
                          notebook_sha256=nb_sha, metadata_sha256=meta_sha, cap_hours=B.CAP, endpoint=20,
                          lr_schedule_rounds=20, cost_x_fedavg=cost, npy_kernel=builder["ref"],
                          npy_manifest_sha256=builder["manifest_sha256"]))
    B.atomic_json(OUT / "batch.json", batch)
    print("LA_LOCKED", [(r["arm"], r["seed"], r["account"], r["slug"]) for r in batch], flush=True)
    return batch


def gate():
    """Locked rule on audited arms only; absent-class recall, p10 and per-class F1 are report-only."""
    batch = json.loads((OUT / "batch.json").read_text())
    audit = json.loads((OUT / "audit.json").read_text())["arms"]
    rows = {}
    for spec in batch:
        assert audit.get(spec["arm"], {}).get("pass"), ("audit", spec["slug"])
        rows.setdefault(spec["variant"], {})[spec["seed"]] = B.rounds(ART / spec["slug"])
    mean = {a: {k: sum(s[k] for s in rows[a].values()) / len(SEEDS) for k in ("macro_tb", "macro_r20", "p10_r20")} for a in ARMS}
    ctl = mean["A3"]["macro_tb"]
    out = {}
    for arm, (tag, role, _) in ARMS.items():
        m = mean[arm]
        verdict = role.upper() if role != "candidate" else "PASS" if m["macro_tb"] >= ctl else "FAIL"
        out[arm] = dict(m, d_tb_vs_a3=m["macro_tb"] - ctl, d_tb_vs_l4=m["macro_tb"] - mean["L4"]["macro_tb"], verdict=verdict,
                        seeds={s: dict(macro_tb=v["macro_tb"], macro_r20=v["macro_r20"], p10_r20=v["p10_r20"],
                                       f1_per_class_r20=v["f1_per_class_r20"]) for s, v in rows[arm].items()})
        print(f'{arm:4s} TB {m["macro_tb"]:.5f} r20 {m["macro_r20"]:.5f} dTB-A3 {out[arm]["d_tb_vs_a3"]:+.5f} '
              f'dTB-L4 {out[arm]["d_tb_vs_l4"]:+.5f} {verdict}', flush=True)
    passed = sorted((a for a in out if out[a]["verdict"] == "PASS"), key=lambda a: -out[a]["macro_tb"])
    B.atomic_json(OUT / "verdict-r20.json", {"rule": {"metric": "pooled Macro-F1 fold val, mean r16-20, mean seeds 42/43",
                                                      "pass": "candidate >= A3 (la_mode=off), same batch",
                                                      "select": "PASS candidate with the highest metric; none -> A3"},
                                             "selected": passed[0] if passed else "A3", "arms": out})
    print("SELECTED", passed[0] if passed else "A3 (no candidate passed)", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    assert cmd in ("quota", "launch", "relaunch", "collect", "gate"), cmd
    if cmd == "quota":                                                 # clean reads only: a failed owner gets no slot
        (OUT / "quota.json").unlink(missing_ok=True)
        q = B.W.inventory()
        if q.get("partial"):
            good = json.loads((OUT / "quota.partial.json").read_text())
            B.atomic_json(OUT / "quota.json", dict(good, failed=[a["user"] for a in q["accounts"]
                                                                  if a["user"] not in {g["user"] for g in good["accounts"]}]))
        print(sorted((a["user"], round(a.get("gpu_left", 0), 2), len(a.get("active_recent", [])))
                     for a in json.loads((OUT / "quota.json").read_text())["accounts"]), flush=True)
    elif cmd == "launch":
        B.W.launch(prepare(json.loads((OUT / "quota.json").read_text())))
    elif cmd == "relaunch":                                           # relaunch <owner> ...
        relaunch(sys.argv[2:])
    elif cmd == "gate":
        gate()
    else:
        B.main()
