"""Cross-account Y50 continuation, parallel checkpoint staging, same-owner follow-up sessions."""
import concurrent.futures
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_w20 as W,run_f50 as F,run_y50 as Y
from scripts.watch_kaggle_batch import atomic_json,query
from scripts.resume_checkpoint_cells import dataset_import,kernel_import

OUT=ROOT/"runs/y-transfer";ART=ROOT/"artifacts/y50";LIVE=ROOT/"runs/y-study/wave2"


def plan():
    path=OUT/"plan.json"
    if path.exists():return json.loads(path.read_text())
    quota=W.inventory()
    owners=sorted([q for q in quota["accounts"] if q.get("gpu_left",0)>3 and not(q.get("inventory_error") or q.get("active_recent") or q.get("unknown_recent") or q.get("gpu_reserved",0))],key=lambda q:q["gpu_left"],reverse=True)
    assert len(owners)>=4,"Need four idle accounts >3h for current cross-account handoff"
    parents=json.loads((OUT/"parents.json").read_text())
    result=[]
    for spec in sorted(parents,key=lambda s:s["round"]):
        owner=owners.pop(0 if spec["round"]<40 else -1)
        cap=min(11.,(50-spec["round"])*.21+1.,owner["gpu_left"]/1.15-.05)
        result.append(dict(spec,target_account=owner["user"],cap_hours=cap,quota_at_plan=owner))
    atomic_json(path,result)
    print("Y_TRANSFER_ACCOUNTS",[(s["arm"],s["target_account"],round(s["cap_hours"],2)) for s in result],flush=True)
    return result


def notebook(spec,session,cap,ready=None):
    assert all(W.sha(ROOT/n)==h for n,h in spec["overlay_sha256"].items()),"Frozen training overlay changed"
    run=spec["slug"];owner=spec["target_account"];kernel=run+f"-x{session}"
    builder=json.loads((ROOT/"runs/w-study/builder.json").read_text())
    Y.notebook(owner,kernel,run,spec["cell"],spec["hp"],spec["base_sha"],builder,cap,50)
    folder=ROOT/"notebooks"/kernel;path=folder/(kernel+".ipynb");meta_path=folder/"kernel-metadata.json"
    nb=json.loads(path.read_text());meta=json.loads(meta_path.read_text())
    if ready:
        code=dataset_import(run,spec["round"],spec["method"],spec["seed"],ready["archive_sha256"],ready["hp_sha256"],ready["cfg_sha256"])
        meta["dataset_sources"].append(ready["ref"])
    else:
        code=kernel_import(run,spec["resume_round"],spec["method"],spec["seed"])
        meta["kernel_sources"].insert(0,f'{owner}/{spec["previous_kernel"]}')
    index=next(i for i,c in enumerate(nb["cells"]) if "NPY, =" in "".join(c.get("source",[])))
    nb["cells"].insert(index,dict(cell_type="code",execution_count=None,metadata={},outputs=[],source=code.splitlines(keepends=True)))
    for c in nb["cells"]:
        if c["cell_type"]=="code":compile("".join(c["source"]),kernel,"exec")
    path.write_text(json.dumps(nb,indent=1)+"\n");meta_path.write_text(json.dumps(meta,indent=1)+"\n")
    return dict(spec,account=owner,slug=kernel,run_slug=run,version=1,cap_hours=cap,wandb_run_id=run,endpoint=50,
                notebook_sha256=W.sha(path),metadata_sha256=W.sha(meta_path))


def stage(spec):
    folder=OUT/spec["arm"];folder.mkdir(exist_ok=True)
    with (folder/"stage.log").open("a") as log:
        p=subprocess.run([*W.cli(spec["target_account"])[:4],sys.executable,"scripts/stage_y50_checkpoint.py","--arm",spec["arm"]],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert p.returncode==0,(spec["arm"],"Private staging failed; inspect stage log, no GPU push")
    return json.loads((folder/"ready.json").read_text())


def audit(spec,sessions,endpoint):
    out=OUT/spec["arm"]/f"audit-r{endpoint}";out.mkdir(exist_ok=True)
    path=out/"batch.json";atomic_json(path,[dict(spec,sessions=[s["slug"] for s in sessions])])
    subprocess.run([sys.executable,"scripts/audit_y50.py","--batch",str(path),"--root",str(ART),"--out",str(out),"--endpoint",str(endpoint)],check=True,cwd=ROOT)


def publish(progress,ledger):
    atomic_json(OUT/"progress.json",progress);atomic_json(OUT/"sessions.json",ledger)
    atomic_json(LIVE/"progress.json",dict(batch_sha256=W.sha(LIVE/"batch.json"),runs=progress))
    atomic_json(LIVE/"sessions.json",{arm:[s["slug"] for s in ss] for arm,ss in ledger.items()})


def main():
    OUT.mkdir(exist_ok=True)
    lock=(LIVE/"worker.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    W.OUT,W.ART,W.TAG=OUT,ART,"Y_TRANSFER";F.OUT,F.ART=OUT,ART
    # The pull worker is deterministic and owns only checkpoint download files.
    while not (OUT/"parents.json").exists():time.sleep(2)
    batch=plan()
    while not (OUT/"pull-complete.json").exists():
        state=json.loads((Path.home()/".codex/long-jobs/jobs/y50-checkpoint-pull-20261107/state.json").read_text())
        assert state["status"] not in {"failed","cancelled"},"Checkpoint pull failed; preserve partials"
        time.sleep(15)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:ready=list(pool.map(stage,batch))
    launch_path=OUT/"batch.json"
    if launch_path.exists():launch=json.loads(launch_path.read_text())
    else:
        launch=[notebook(s,len(s["sessions"])+1,s["cap_hours"],r) for s,r in zip(batch,ready)]
        atomic_json(launch_path,launch)
    W.launch(launch)
    ledger_path=OUT/"sessions.json"
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        s["arm"]:[*[dict(account=s["account"],slug=k,version=1) for k in s["sessions"]],l] for s,l in zip(batch,launch)}
    progress_path=OUT/"progress.json"
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else json.loads((LIVE/"progress.json").read_text())["runs"]
    for row in progress.values():row.pop("resume",None)
    publish(progress,ledger);(OUT/"collection").mkdir(exist_ok=True)
    while True:
        pending=[s for s in batch if not progress[s["arm"]].get("final") and not progress[s["arm"]].get("resume")]
        with concurrent.futures.ThreadPoolExecutor(4) as pool:states=list(pool.map(lambda s:query(ledger[s["arm"]][-1]),pending))
        atomic_json(OUT/"status.json",states)
        terminal=[(s,t) for s,t in zip(pending,states) if t["terminal"]]
        def collect(item):
            spec,state=item;current=ledger[spec["arm"]][-1]
            rc=F.collect(dict(spec,account=current["account"]),current["slug"])
            assert rc==0,(spec["arm"],"Output pull failed")
            return spec,state
        with concurrent.futures.ThreadPoolExecutor(4) as pool:collected=list(pool.map(collect,terminal))
        for spec,state in collected:
            assert state["status"]=="COMPLETE",(spec["arm"],"ERROR outputs collected; inspect training failure")
            done=F.last_round(spec);audit(spec,ledger[spec["arm"]],done)
            row=progress[spec["arm"]];current=ledger[spec["arm"]][-1]
            row.setdefault("sessions",{})[current["slug"]]=dict(state=state,pulled=True,last_round=done)
            row["partial_audited_round"]=done
            if done==50:row.update(final=True,audited=True)
            else:row["resume"]=dict(round=done,after=current["slug"])
        publish(progress,ledger)
        for spec in batch:
            row=progress[spec["arm"]];resume=row.get("resume")
            if not resume:continue
            owner=spec["target_account"];q=W.ka().account_quota(owner)
            cap=min(11.,(50-resume["round"])*.21+1.,q["gpu_left"]/1.15-.05)
            if cap<1.21:continue
            child=notebook(dict(spec,resume_round=resume["round"],previous_kernel=resume["after"]),len(ledger[spec["arm"]])+1,cap)
            receipt=OUT/spec["arm"]/f'session{len(ledger[spec["arm"]])+1}';receipt.mkdir(exist_ok=True)
            atomic_json(receipt/"batch.json",[child]);W.OUT=receipt
            try:W.launch([child])
            finally:W.OUT=OUT
            ledger[spec["arm"]].append(child);row.pop("resume");publish(progress,ledger)
        if all(progress[s["arm"]].get("final") for s in batch):
            closure=dict(batch_sha256=W.sha(LIVE/"batch.json"),runs=progress,sessions={a:[s["slug"] for s in ss] for a,ss in ledger.items()})
            atomic_json(OUT/"closure.json",closure);atomic_json(LIVE/"closure.json",closure)
            subprocess.run([sys.executable,"scripts/analyse_y50.py"],check=True,cwd=ROOT)
            print("Y50_TRANSFER_PIPELINE_DONE",flush=True);return
        print("Y_TRANSFER_PROGRESS",{a:r.get("partial_audited_round") for a,r in progress.items()},flush=True)
        current=[ledger[s["arm"]][-1] for s in batch if not progress[s["arm"]].get("final")]
        health=OUT/("startup-"+str(sum(len(v) for v in ledger.values())))
        if current and (not (health/"startup.json").exists() or not json.loads((health/"startup.json").read_text()).get("all_ready")):
            atomic_json(OUT/"current-batch.json",current)
            subprocess.run([sys.executable,"scripts/check_screening_startup.py","--batch",str(OUT/"current-batch.json"),
                            "--out",str(health),"--entity","21522798-uit"],cwd=ROOT)
        time.sleep(1200)


if __name__=="__main__":main()
