"""Two-seed, two-round full-network T4 runtime gate; not a r50 utility verdict."""
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
from scripts import run_w20 as W, f50_common
from scripts.watch_kaggle_batch import atomic_json, query

OUT, ART = ROOT / "runs/standard-t4", ROOT / "artifacts/standard-t4"
CAP = 1.25
OVERLAYS = ["methods/mobihet.py", "methods/mobihet_copula_standard.py"]


def prepare():
    if (OUT/"batch.json").exists():
        return json.loads((OUT/"batch.json").read_text())
    local = json.loads((ROOT/"runs/standard-copula-production/preflight.json").read_text())
    assert all(W.sha(ROOT/n)==h for n,h in local["sources"].items())
    assert {r["seed"] for r in local["reports"] if r["state_history_cm_snapshot_crash_resume_bitexact"]} == {42,43}
    head = W.git_head()
    quota = W.inventory()
    eligible = sorted([q for q in quota["accounts"] if q.get("gpu_left",0)>=CAP*1.15 and
                       not(q.get("inventory_error") or q.get("active_recent") or q.get("unknown_recent") or q.get("gpu_reserved",0))],
                      key=lambda q:(q["gpu_left"],q["user"]))
    assert len(eligible)>=2, "Need two idle accounts each >=1.4375h; preserve quota evidence, do not push blindly"
    builder = json.loads((ROOT/"runs/w-study/builder.json").read_text())
    batch=[]
    for index, (seed, account) in enumerate(zip((42,43),eligible)):
        owner, slug = account["user"], f"mh-z{index}-stdcop-s{seed}-t4r2"
        hp = f50_common.hp_for("core",seed)
        args = shlex.join(["scripts/run_q20.py","--method","mobihet_copula_standard","--out",f"/kaggle/working/{slug}",
                           "--width","1","--rounds","50","--until","2","--save-every","1","--compile","--wandb","mobihet-v4",
                           "--max-hours","BUDGET","--cache","","--set",*hp])
        subprocess.run([sys.executable,"scripts/gen_notebook.py","--owner",owner,"--slug",slug,"--sha",head,"--run",slug,
                        "--max-hours",str(CAP),"--args",args,"--npy-kernel",builder["ref"],
                        "--npy-manifest-sha256",builder["manifest_sha256"],"--overlay",*OVERLAYS],cwd=ROOT,check=True,capture_output=True)
        folder=ROOT/"notebooks"/slug; notebook=folder/(slug+".ipynb"); meta=folder/"kernel-metadata.json"
        doc=json.loads(notebook.read_text())
        warm=f'''os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/kaggle/temp/inductor"
rc = subprocess.run(cmd, shell=True).returncode
subprocess.run(["tar", "-cf", "/kaggle/working/{slug}/inductor-cache.tar", "-C", "/kaggle/temp", "inductor"])
'''
        for c in doc["cells"]:
            if c["cell_type"]=="code":
                c["source"]="".join(c["source"]).replace('TORCHINDUCTOR_COMPILE_THREADS="2"','TORCHINDUCTOR_COMPILE_THREADS="1"').replace(
                    'rc = subprocess.run(cmd, shell=True).returncode\n',warm).splitlines(keepends=True)
                compile("".join(c["source"]),str(notebook),"exec")
        notebook.write_text(json.dumps(doc,indent=1)+"\n")
        subprocess.run([sys.executable,str(W.SKILL/"embed_wandb_key.py"),str(notebook),"--metadata",str(meta)],check=True,capture_output=True)
        metadata=json.loads(meta.read_text()); runtime=json.loads((ROOT/"knowledge/runtime.json").read_text())
        assert metadata["is_private"] and metadata["id"]==f"{owner}/{slug}" and metadata["kernel_sources"]==[builder["ref"]]
        assert all(metadata[k]==runtime[k] for k in ("machine_shape","docker_image"))
        batch.append(dict(arm=f"Z{index}",cell="standard",seed=seed,mode="stream",r_kind="live",method="mobihet_copula_standard",
            account=owner,slug=slug,version=1,hp=hp,base_sha=head,source_sha256=W.sha(ROOT/"methods/mobihet.py"),
            overlay_sha256={p:W.sha(ROOT/p) for p in OVERLAYS},notebook_sha256=W.sha(notebook),metadata_sha256=W.sha(meta),
            cap_hours=CAP,endpoint=2,lr_schedule_rounds=50,save_every=1,bootstrap_total=W.COMPACT_BOOTSTRAP,
            npy_kernel=builder["ref"],npy_manifest_sha256=builder["manifest_sha256"]))
    atomic_json(OUT/"batch.json",batch)
    atomic_json(ROOT/"docs/research/standard-t4-launch.json",dict(batch=batch,quota_sha256=W.sha(OUT/"quota.json"),
        preflight_sha256=W.sha(ROOT/"runs/standard-copula-production/preflight.json"),
        authorization="Owner: continue generator experiments, analyse failures and try next; runtime gate only, no r50 utility claim"))
    print("STANDARD_T4_LOCKED",[(r["account"],r["slug"]) for r in batch],flush=True)
    return batch


def audit(spec, endpoint=2):
    import numpy as np
    from scripts.audit_screening import audit as common
    from scripts.audit_r20 import extra
    counts=np.load(ROOT/"evidence/research-data/train_counts.npy")
    support=np.load(ROOT/"evidence/research-data/test_support.npy")
    report=common(ART,spec,support,endpoint=endpoint)
    detail=extra(ART,spec,counts,method=spec["method"],bootstrap_total=spec["bootstrap_total"],endpoint=endpoint)
    folder=ART/spec["slug"]/spec["slug"]
    for rank in (0,1):
        r=json.loads((folder/f"recovery/standard_copula_rank{rank}.json").read_text())
        assert r["rank"]==rank and r["latent_mu_max_abs"]==0 and r["latent_variance_max_error"]<1e-6
        assert sum(r["bootstrap_bytes"].values())==spec["bootstrap_total"]
    dest=OUT/"collection"/(spec["arm"]+f"-r{endpoint}");dest.mkdir(exist_ok=True)
    atomic_json(dest/"audit.json",report);atomic_json(dest/"diagnostic-audit.json",detail)
    print("STANDARD_T4_ARTIFACT_PASS",spec["arm"],flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/"worker.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    W.OUT,W.ART,W.TAG=OUT,ART,"STANDARD_T4"
    W.COLLECT=W.COLLECT+r'|/recovery/standard_copula_rank[01]\.json$'
    batch=prepare();W.launch(batch)
    path=OUT/"progress.json";progress=json.loads(path.read_text()) if path.exists() else {}
    (OUT/"collection").mkdir(exist_ok=True)
    while True:
        pending=[s for s in batch if not progress.get(s["arm"],{}).get("final")]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            states=list(pool.map(query,pending))
        atomic_json(OUT/"status.json",states)
        for spec,state in zip(pending,states):
            if not state["terminal"]:continue
            rc=W.collect(spec)
            progress[spec["arm"]]=dict(state=state,pulled=rc==0,final=True,audited=False)
            atomic_json(path,progress)
            if rc or state["status"]!="COMPLETE":continue
            try:
                audit(spec);progress[spec["arm"]]["audited"]=True
            except Exception as error:
                progress[spec["arm"]]["audit_error_type"]=type(error).__name__
                print("STANDARD_T4_ATTENTION",spec["arm"],repr(error),flush=True)
            atomic_json(path,progress)
        if all(progress.get(s["arm"],{}).get("final") for s in batch):
            atomic_json(OUT/"closure.json",progress)
            assert all(r["audited"] for r in progress.values()),"Runtime gate failed: inspect actual outputs, not utilityFAIL"
            print("STANDARD_T4_GATE_DONE",flush=True);return
        print("STANDARD_T4_WAIT",states,flush=True)
        startup=OUT/"startup/startup.json"
        if not startup.exists() or not json.loads(startup.read_text()).get("all_ready"):
            subprocess.run([sys.executable,"scripts/check_screening_startup.py","--batch",str(OUT/"batch.json"),
                            "--out",str(startup.parent),"--entity","21522798-uit"],cwd=ROOT)
        time.sleep(1200)


if __name__=="__main__":main()
