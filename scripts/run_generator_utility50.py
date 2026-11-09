"""Continue audited generator runtime parents to r50 after quota refresh; never restart r1."""
import concurrent.futures
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_w20 as W,run_f50 as F,run_standard_t4 as S,run_forest_t4 as T,y50_common as Y
from scripts.resume_checkpoint_cells import kernel_import
from scripts.watch_kaggle_batch import atomic_json,query

OUT=ROOT/"runs/generator-utility50"
NOT_BEFORE=datetime.fromisoformat("2026-11-07T07:00:00+07:00")
GUARDS={"p10":-.006,"low256":-.003,"timeDelay":-.0035,"benign":-.0005,"absent":-.01}


def originals():
    rows=[]
    for study,runner in (("standard",S),("forest",T)):
        for spec in json.loads((runner.OUT/"batch.json").read_text()):
            parent=spec["slug"]+("-c2" if study=="standard" else "")
            rows.append(dict(spec,study=study,parent_kernel=parent,round=2,next_session=3 if study=="standard" else 2,
                             artifact_root=str(runner.ART)))
    return rows


def lock_plan():
    specs=originals()
    assert len(specs)==4 and len({s["account"] for s in specs})==4
    receivers={s["target_account"] for s in json.loads((ROOT/"runs/y-transfer/plan.json").read_text())}
    assert not receivers & {s["account"] for s in specs}
    for spec in specs:
        assert all(W.sha(ROOT/n)==h for n,h in spec["overlay_sha256"].items())
        runner=S if spec["study"]=="standard" else T
        audit=runner.OUT/"collection"/(spec["arm"]+"-r2")/"audit.json"
        report=json.loads(audit.read_text())
        assert [r["round"] for r in report["rounds"]]==[1,2]
        spec["parent_audit_sha256"]=W.sha(audit)
        manifest=json.loads((Path(spec["artifact_root"])/spec["slug"]/"provenance/source_manifest.json").read_text())
        shared={n:h for n,h in manifest["files"].items() if n.startswith(("core/","methods/")) or n in
                {"scripts/run_fl.py","scripts/run_q20.py","scripts/run_p21_recovery.py","knowledge/input_order.json","knowledge/label_mapping.json","knowledge/runtime.json"}}
        assert all(W.sha(ROOT/n)==h for n,h in shared.items()),"Shared training source differs from audited parent"
        spec["shared_training_sha256"]=shared
    plan=dict(specs=specs,not_before=NOT_BEFORE.isoformat(),endpoint=50,macro_gain=.003,guards=GUARDS,
              control="F2 seed42/F3 seed43, source-matched common harness; no fresh control rerun",
              rule="Both seeds macro >= +.003 and all guard deltas >= thresholds; no test-driven HP or best round",
              source_sha256=W.sha(Path(__file__)),authorization="Owner: continue generator experiments/failure analysis; prioritize Y50, long batches after quota refresh")
    path=OUT/"plan.json"
    if path.exists():assert json.loads(path.read_text())==plan,"Frozen utility plan changed"
    else:atomic_json(path,plan)
    return specs


def notebook(spec,previous,done,session,cap):
    assert all(W.sha(ROOT/n)==h for n,h in spec["overlay_sha256"].items())
    assert all(W.sha(ROOT/n)==h for n,h in spec["shared_training_sha256"].items())
    owner,run=spec["account"],spec["slug"];kernel=run+f"-u50-c{session}"
    args=shlex.join(["scripts/run_q20.py","--method",spec["method"],"--out",f"/kaggle/working/{run}","--width","1",
                    "--rounds","50","--until","50","--save-every","1","--compile","--wandb","mobihet-v4",
                    "--max-hours","BUDGET","--cache","","--set",*spec["hp"]])
    subprocess.run([sys.executable,"scripts/gen_notebook.py","--owner",owner,"--slug",kernel,"--run",run,
                    "--sha",spec["base_sha"],"--max-hours",str(cap),"--args",args,"--npy-kernel",spec["npy_kernel"],
                    "--npy-manifest-sha256",spec["npy_manifest_sha256"],"--overlay",*spec["overlay_sha256"]],
                   cwd=ROOT,check=True,capture_output=True)
    folder=ROOT/"notebooks"/kernel;path=folder/(kernel+".ipynb");meta_path=folder/"kernel-metadata.json"
    nb=json.loads(path.read_text());meta=json.loads(meta_path.read_text())
    index=next(i for i,c in enumerate(nb["cells"]) if "NPY, =" in "".join(c.get("source",[])))
    nb["cells"].insert(index,dict(cell_type="code",execution_count=None,metadata={},outputs=[],
        source=kernel_import(run,done,spec["method"],spec["seed"]).splitlines(keepends=True)))
    cache=f'''cache_tar = "/kaggle/working/{run}/inductor-cache.tar"
if os.path.exists(cache_tar): print("inductor cache restored", subprocess.run(["tar","-xf",cache_tar,"-C","/kaggle/temp"]).returncode)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/kaggle/temp/inductor"
rc = subprocess.run(cmd, shell=True).returncode
subprocess.run(["tar","-cf",cache_tar,"-C","/kaggle/temp","inductor"])
'''
    for cell in nb["cells"]:
        if cell["cell_type"]!="code":continue
        source="".join(cell["source"]).replace('TORCHINDUCTOR_COMPILE_THREADS="2"','TORCHINDUCTOR_COMPILE_THREADS="1"')
        source=source.replace('rc = subprocess.run(cmd, shell=True).returncode\n',cache)
        compile(source,kernel,"exec");cell["source"]=source.splitlines(keepends=True)
    meta["kernel_sources"].insert(0,f"{owner}/{previous}")
    assert meta["is_private"] and meta["id"]==f"{owner}/{kernel}"
    runtime=json.loads((ROOT/"knowledge/runtime.json").read_text())
    assert all(meta[k]==runtime[k] for k in ("machine_shape","docker_image"))
    path.write_text(json.dumps(nb,indent=1)+"\n");meta_path.write_text(json.dumps(meta,indent=1)+"\n")
    subprocess.run([sys.executable,str(W.SKILL/"embed_wandb_key.py"),str(path),"--metadata",str(meta_path)],check=True,capture_output=True)
    return dict(spec,slug=kernel,run_slug=run,wandb_run_id=run,version=1,restore_round=done,cap_hours=cap,endpoint=50,
                notebook_sha256=W.sha(path),metadata_sha256=W.sha(meta_path))


def launch(specs,progress):
    assert datetime.now(NOT_BEFORE.tzinfo)>=NOT_BEFORE,"New long experiments wait until 07:00 UTC+7"
    waiting=[]
    for spec in specs:
        row=progress[spec["arm"]]
        if row.get("current") or row.get("final"):continue
        q=W.ka().account_quota(spec["account"])
        done=row["round"]
        # R2 duration + measured-save allowance; setup per new session is budgeted separately.
        history=[json.loads(v) for v in (Path(spec["artifact_root"])/spec["slug"]/spec["slug"]/"rounds.jsonl").read_text().splitlines()]
        worst=max(v["t_train"]+v["t_eval"]+v.get("t_diagnostics",0)+90 for v in history[1:])
        whole=(50-done)*worst/3600+1.5
        if q["gpu_left"]<whole*1.15:
            waiting.append(dict(arm=spec["arm"],quota=q,required_hours_with_margin=whole*1.15));continue
        parent=dict(spec,slug=row["previous"],version=1)
        state=query(parent)
        assert state["status"]=="COMPLETE",(spec["arm"],"Parent not COMPLETE; no blind push")
        cap=min(11.,whole,q["gpu_left"]/1.15-.05)
        child=notebook(spec,row["previous"],done,row["next_session"],cap)
        folder=OUT/spec["arm"]/f'session{row["next_session"]}';folder.mkdir(parents=True,exist_ok=True)
        atomic_json(folder/"batch.json",[child]);W.OUT=folder
        try:W.launch([child])
        finally:W.OUT=OUT
        row["current"]=child;row["next_session"]+=1
        atomic_json(OUT/"progress.json",progress)
    atomic_json(OUT/"waiting-quota.json",waiting)


def analyse(specs):
    results={}
    for study in ("standard","forest"):
        arms=[]
        for spec in (s for s in specs if s["study"]==study):
            d=Path(spec["artifact_root"])/spec["slug"]/spec["slug"]
            arms.append(dict(arm=spec["arm"],seed=spec["seed"],value=Y.read(d,50),delta=Y.delta(d,spec["seed"],50)))
        passed=all(a["delta"]["f1_macro"]>=.003 and all(a["delta"][k]>=v for k,v in GUARDS.items()) for a in arms)
        results[study]=dict(arms=arms,pass_utility=passed,scope="exploratory reused official test, two seeds not significance")
    atomic_json(OUT/"analysis.json",results)


def main():
    OUT.mkdir(exist_ok=True);lock=(OUT/"worker.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    W.OUT,W.TAG=OUT,"GENERATOR_UTILITY50"
    W.COLLECT+=r'|/recovery/(standard_copula|forest)_rank[01]\.json$'
    specs=lock_plan()
    print("GENERATOR_UTILITY50_WAIT_UNTIL",NOT_BEFORE.isoformat(),flush=True)
    while (seconds:=(NOT_BEFORE-datetime.now(NOT_BEFORE.tzinfo)).total_seconds())>0:time.sleep(min(60,seconds))
    path=OUT/"progress.json"
    progress=json.loads(path.read_text()) if path.exists() else {s["arm"]:dict(round=2,previous=s["parent_kernel"],next_session=s["next_session"],final=False) for s in specs}
    (OUT/"collection").mkdir(exist_ok=True)
    # Fresh full-owner inventory after the deadline: reject active/unknown owners before initial starts.
    inventory=W.inventory();owners={q["user"]:q for q in inventory["accounts"]}
    assert all(s["account"] in owners and not(owners[s["account"]].get("active_recent") or owners[s["account"]].get("unknown_recent") or owners[s["account"]].get("inventory_error")) for s in specs)
    while True:
        launch(specs,progress)
        pending=[s for s in specs if progress[s["arm"]].get("current")]
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            states=list(pool.map(lambda s:query(progress[s["arm"]]["current"]),pending))
        atomic_json(OUT/"status.json",states)
        for spec,state in zip(pending,states):
            if not state["terminal"]:continue
            row=progress[spec["arm"]];child=row["current"]
            F.OUT,F.ART=OUT,Path(spec["artifact_root"])
            assert F.collect(spec,child["slug"])==0,"Output pull failure: recover collector only"
            assert state["status"]=="COMPLETE","ERROR outputs pulled; inspect before retry"
            history=[json.loads(v) for v in (Path(spec["artifact_root"])/spec["slug"]/spec["slug"]/"rounds.jsonl").read_text().splitlines()]
            done=history[-1]["round"];assert done>row["round"] and done<=50
            runner=S if spec["study"]=="standard" else T;runner.audit(spec,done)
            log=(Path(spec["artifact_root"])/spec["slug"]/(child["slug"]+".log")).read_text()
            assert all(f'resumed rank{rank} at round {row["round"]+1}' in log for rank in (0,1))
            assert "inductor cache restored 0" in log
            row.setdefault("sessions",[]).append(dict(kernel=child["slug"],round=done,state=state))
            row.update(round=done,previous=child["slug"],final=done==50);row.pop("current")
            assert row["next_session"]<=9,"Too many continuation sessions; inspect budget evidence"
            atomic_json(path,progress)
        if all(r["final"] for r in progress.values()):
            analyse(specs);atomic_json(OUT/"closure.json",progress)
            print("GENERATOR_UTILITY50_DONE",flush=True);return
        current=[r["current"] for r in progress.values() if r.get("current")]
        if current:
            atomic_json(OUT/"current-batch.json",current)
            health=OUT/"startup"/str(sum(r["next_session"] for r in progress.values()))
            if not (health/"startup.json").exists() or not json.loads((health/"startup.json").read_text()).get("all_ready"):
                subprocess.run([sys.executable,"scripts/check_screening_startup.py","--batch",str(OUT/"current-batch.json"),"--out",str(health),"--entity","21522798-uit"],cwd=ROOT)
        print("GENERATOR_UTILITY50_PROGRESS",{a:r["round"] for a,r in progress.items()},flush=True)
        time.sleep(1200)


if __name__=="__main__":main()
