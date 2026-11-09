"""Range-parallel checkpoint pull, strict immutable-version/range/size gates; signed URLs stay in memory."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
from pathlib import Path
import re
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / ".agents/skills/kaggle-training-notebook/scripts/pull_kernel_output.py"
spec = importlib.util.spec_from_file_location("kernel_pull", HELPER)
pull = importlib.util.module_from_spec(spec); spec.loader.exec_module(pull)


def range_piece(url, start, end, total, path):
    offset = path.stat().st_size if path.exists() else 0
    assert offset <= end-start+1, "Oversized range partial"
    if offset == end-start+1:
        return
    first = start+offset
    with requests.get(url, headers={"Range": f"bytes={first}-{end}"}, stream=True, timeout=(30,120)) as r:
        r.raise_for_status()
        assert r.status_code == 206 and r.headers.get("Content-Range") == f"bytes {first}-{end}/{total}", "Range not honored"
        with path.open("ab" if offset else "wb") as f:
            for chunk in r.iter_content(1<<20):
                f.write(chunk)
    assert path.stat().st_size == end-start+1, "Incomplete range"


def fetch_parallel(url, path, segments=4):
    path.parent.mkdir(parents=True,exist_ok=True)
    with requests.get(url,headers={"Range":"bytes=0-0"},stream=True,timeout=(30,120)) as response:
        response.raise_for_status()
        match = re.fullmatch(r"bytes 0-0/(\d+)",response.headers.get("Content-Range",""))
        if response.status_code != 206 or not match:
            with requests.Session() as session:
                pull.fetch(session,url,path,120)
            return
        size=int(match[1])
    if path.exists() and path.stat().st_size==size:
        return
    count=min(segments,size)
    pieces=[(i*size//count,(i+1)*size//count-1,path.with_name(path.name+f".range{i}")) for i in range(count)]
    with ThreadPoolExecutor(count) as pool:
        list(pool.map(lambda part:range_piece(url,part[0],part[1],size,part[2]),pieces))
    temp=path.with_name(path.name+".assembled")
    with temp.open("wb") as target:
        for _,_,piece in pieces:
            with piece.open("rb") as source:
                while block:=source.read(8<<20):target.write(block)
        target.flush();os.fsync(target.fileno())
    assert temp.stat().st_size==size
    os.replace(temp,path)
    for _,_,piece in pieces:piece.unlink()  # only this download's verified scratch pieces


def main():
    ap=argparse.ArgumentParser();ap.add_argument("ref");ap.add_argument("--dest",type=Path,required=True);ap.add_argument("--segments",type=int,default=4)
    a=ap.parse_args();assert len(a.ref.split("/"))==3 and a.ref.rsplit("/",1)[1].isdigit(),"Pin exact version"
    a.dest.mkdir(parents=True,exist_ok=True)
    missing={"state.pt","state_rank1.pt"};start=time.monotonic()
    for attempt in range(4):
        _,files,_=pull.listing(a.ref)
        selected={n:u for n,u in files.items() if Path(n).name in {"state.pt","state_rank1.pt","inductor-cache.tar"}}
        assert {Path(n).name for n in selected} >= {"state.pt","state_rank1.pt"}
        def one(item):
            n,url=item
            try:
                fetch_parallel(url,a.dest/n,a.segments)
                return n,True
            except Exception as error:
                print("CHECKPOINT_RANGE_RETRY",n,type(error).__name__,flush=True)
                return n,False
        # 2 files x 4 ranges x 4 concurrent kernels = 32 streams; cache follows states.
        states=[item for item in selected.items() if Path(item[0]).name!="inductor-cache.tar"]
        with ThreadPoolExecutor(2) as pool:outcomes=list(pool.map(one,states))
        missing={n for n,ok in outcomes if not ok}
        if not missing:
            for item in selected.items():
                if Path(item[0]).name=="inductor-cache.tar":
                    _,ok=one(item)
                    assert ok,"Cache pull incomplete"
            break
        time.sleep(2**attempt)
    assert not missing,"Checkpoint ranges incomplete; preserve partials, no training push"
    print("CHECKPOINT_PULL_DONE",a.ref,round(time.monotonic()-start),flush=True)


if __name__=="__main__":main()
