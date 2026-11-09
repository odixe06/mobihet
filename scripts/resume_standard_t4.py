"""Resume the two audited r1 T4 sessions to the original r2 gate; no restart or method change."""
import concurrent.futures
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_standard_t4 as S, run_w20 as W, run_f50 as F
from scripts.watch_kaggle_batch import atomic_json,query
from scripts.resume_checkpoint_cells import kernel_import

OUT=ROOT/"runs/standard-t4-resume"


def prepare():
    if (OUT/"batch.json").exists():return json.loads((OUT/"batch.json").read_text())
    batch=[]
    for original in json.loads((S.OUT/"batch.json").read_text()):
        S.audit(original,1)
        owner,run=original["account"],original["slug"];kernel=run+"-c2"
        q=W.ka().account_quota(owner);cap=min(1.,q["gpu_left"]/1.15-.01)
        assert cap>=.8,(owner,"Insufficient quota for warm one-round continuation")
        assert all(W.sha(ROOT/n)==h for n,h in original["overlay_sha256"].items())
        folder=ROOT/"notebooks"/kernel;folder.mkdir(exist_ok=True)
        nb=json.loads((ROOT/"notebooks"/run/(run+".ipynb")).read_text())
        index=next(i for i,c in enumerate(nb["cells"]) if "NPY, =" in "".join(c.get("source",[])))
        nb["cells"].insert(index,dict(cell_type="code",execution_count=None,metadata={},outputs=[],source=kernel_import(run,1,original["method"],original["seed"]).splitlines(keepends=True)))
        for c in nb["cells"]:
            if c["cell_type"]=="code":
                source="".join(c["source"]).replace("budget = 1.25 -",f"budget = {cap!r} -")
                if 'os.environ["TORCHINDUCTOR_CACHE_DIR"]' in source:
                    source=source.replace('os.environ["TORCHINDUCTOR_CACHE_DIR"]',f'cache_tar = "/kaggle/working/{run}/inductor-cache.tar"\nif os.path.exists(cache_tar): print("inductor cache restored", subprocess.run(["tar","-xf",cache_tar,"-C","/kaggle/temp"]).returncode)\nos.environ["TORCHINDUCTOR_CACHE_DIR"]',1)
                compile(source,kernel,"exec");c["source"]=source.splitlines(keepends=True)
        path=folder/(kernel+".ipynb");path.write_text(json.dumps(nb,indent=1)+"\n")
        meta=json.loads((ROOT/"notebooks"/run/"kernel-metadata.json").read_text())
        meta.update(id=f"{owner}/{kernel}",title=kernel,code_file=kernel+".ipynb",kernel_sources=[f"{owner}/{run}",original["npy_kernel"]])
        (folder/"kernel-metadata.json").write_text(json.dumps(meta,indent=1)+"\n")
        batch.append(dict(original,slug=kernel,run_slug=run,wandb_run_id=run,cap_hours=cap,
                          notebook_sha256=W.sha(path),metadata_sha256=W.sha(folder/"kernel-metadata.json")))
    atomic_json(OUT/"batch.json",batch);return batch


def main():
    OUT.mkdir(exist_ok=True);lock=(OUT/"worker.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    W.OUT,W.ART,W.TAG=OUT,S.ART,"STANDARD_RESUME"
    W.COLLECT+=r'|/recovery/standard_copula_rank[01]\.json$'
    F.OUT,F.ART=OUT,S.ART
    batch=prepare();W.launch(batch)
    path=OUT/"progress.json";progress=json.loads(path.read_text()) if path.exists() else {}
    (OUT/"collection").mkdir(exist_ok=True)
    while True:
        pending=[s for s in batch if not progress.get(s["arm"],{}).get("final")]
        with concurrent.futures.ThreadPoolExecutor(2) as pool:states=list(pool.map(query,pending))
        atomic_json(OUT/"status.json",states)
        for spec,state in zip(pending,states):
            if not state["terminal"]:continue
            original=dict(spec,slug=spec["run_slug"])
            rc=F.collect(original,spec["slug"])
            assert rc==0 and state["status"]=="COMPLETE",(spec["arm"],"Outputs collected; inspect before retry")
            S.audit(original,2)
            log=(S.ART/spec["run_slug"]/(spec["slug"]+".log")).read_text()
            assert "resumed rank0 at round 2" in log and "resumed rank1 at round 2" in log
            assert "inductor cache restored" in log
            progress[spec["arm"]]=dict(final=True,audited=True,state=state,round=2)
            atomic_json(path,progress)
        if len(progress)==2:
            atomic_json(OUT/"closure.json",progress);print("STANDARD_RESUME_DONE",flush=True);return
        subprocess.run([sys.executable,"scripts/check_screening_startup.py","--batch",str(OUT/"batch.json"),"--out",str(OUT/"startup"),"--entity","21522798-uit"],cwd=ROOT)
        time.sleep(1200)


if __name__=="__main__":main()
