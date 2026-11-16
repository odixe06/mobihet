"""Pull four audited Y50 parent checkpoints concurrently (up to 32 range streams)."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_w20 import cli


def main():
    out=ROOT/"runs/y-transfer";out.mkdir(exist_ok=True)
    batch=json.loads((ROOT/"runs/y-study/wave2/batch.json").read_text())
    sessions=json.loads((ROOT/"runs/y-study/wave2/sessions.json").read_text())
    progress=json.loads((ROOT/"runs/y-study/wave2/progress.json").read_text())["runs"]
    assert __import__('shutil').disk_usage(ROOT).free>110*2**30,"Insufficient space for state/archive/download-verification copies"
    plan=[]
    for spec in batch:
        row=progress[spec["arm"]];kernel=sessions[spec["arm"]][-1]
        assert row["partial_audited_round"]==row["sessions"][kernel]["last_round"]
        plan.append(dict(spec,parent_kernel=kernel,round=row["partial_audited_round"],sessions=sessions[spec["arm"]]))
    path=out/"parents.json"
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2)+"\n")
    def one(spec):
        log=out/(spec["arm"]+"-pull.log")
        with log.open("a") as f:
            p=subprocess.run([*cli(spec["account"])[:4],sys.executable,"scripts/pull_checkpoint_parallel.py",
                f'{spec["account"]}/{spec["parent_kernel"]}/1',"--dest",str(ROOT/"artifacts/y50"/spec["slug"]),"--segments","4"],
                cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        assert p.returncode==0,(spec["arm"],"checkpoint pull failed; read per-arm log")
        print("Y50_CHECKPOINT_PULLED",spec["arm"],flush=True)
    with ThreadPoolExecutor(4) as pool:list(pool.map(one,plan))
    (out/"pull-complete.json").write_text(json.dumps({"parents_sha256":__import__('hashlib').sha256(path.read_bytes()).hexdigest()},indent=2)+"\n")
    print("Y50_CHECKPOINT_PULL_DONE",flush=True)


if __name__=="__main__":main()
