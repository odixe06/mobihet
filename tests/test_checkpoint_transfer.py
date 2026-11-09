import re
from pathlib import Path

import pytest
import torch

from scripts import pull_checkpoint_parallel as P
from scripts.resume_checkpoint_cells import kernel_import,dataset_import


class Response:
    def __init__(self, body, first, last, total, status=206):
        self.status_code=status
        self.headers={"Content-Range":f"bytes {first}-{last}/{total}"}
        self.body=body
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def raise_for_status(self):pass
    def iter_content(self,*args):yield self.body


def test_parallel_download_assembles_ranges_and_removes_only_own_scratch(tmp_path,monkeypatch):
    body=b"checkpoint-immutable-content"
    def get(url,headers,**kwargs):
        first,last=map(int,re.fullmatch(r"bytes=(\d+)-(\d+)",headers["Range"]).groups())
        return Response(body[first:last+1],first,last,len(body))
    monkeypatch.setattr(P.requests,"get",get)
    dest=tmp_path/"state.pt";other=tmp_path/"unrelated.part";other.write_bytes(b"keep")
    P.fetch_parallel("unused",dest,4)
    assert dest.read_bytes()==body and other.read_bytes()==b"keep"
    assert not list(tmp_path.glob("*.range*"))


def test_range_resume_and_bad_range_rejection(tmp_path,monkeypatch):
    dest=tmp_path/"segment";dest.write_bytes(b"cd")
    def good(url,headers,**kwargs):
        assert headers["Range"]=="bytes=4-7"
        return Response(b"efgh",4,7,10)
    monkeypatch.setattr(P.requests,"get",good);P.range_piece("unused",2,7,10,dest)
    assert dest.read_bytes()==b"cdefgh"
    monkeypatch.setattr(P.requests,"get",lambda *a,**k:Response(b"bad",0,2,10,status=200))
    with pytest.raises(AssertionError,match="Range"):
        P.range_piece("unused",0,2,10,tmp_path/"bad")
    assert not (tmp_path/"bad").exists()


def test_parent_gate_checks_both_ranks_before_copy(tmp_path):
    import shutil
    source=tmp_path/"input"/"run";source.mkdir(parents=True)
    working=tmp_path/"working";working.mkdir()
    state=dict(round=1,rng_cpu=torch.zeros(2),rng_cuda=torch.zeros(2))
    torch.save(state|dict(method="method",hp=dict(seed=42,rounds=50)),source/"state.pt")
    torch.save(state|dict(round=2),source/"state_rank1.pt")
    code=kernel_import("run",1,"method",42).replace('/kaggle/input',str(tmp_path/"input")).replace('/kaggle/working',str(working))
    with pytest.raises(AssertionError):exec(code,dict(Path=Path,torch=torch,shutil=shutil))
    assert not (working/"run").exists()
    torch.save(state,source/"state_rank1.pt")
    exec(code,dict(Path=Path,torch=torch,shutil=shutil))
    assert (working/"run/state_rank1.pt").is_file()


def test_dataset_gate_checks_archive_and_config_before_training(tmp_path):
    import hashlib,json,tarfile
    source=tmp_path/"source";source.mkdir()
    state=dict(round=27,rng_cpu=torch.zeros(2),rng_cuda=torch.zeros(2))
    hp=dict(seed=42,rounds=50);cfg=dict(width=1)
    torch.save(state|dict(method="method",hp=hp,cfg=cfg),source/"state.pt")
    torch.save(state,source/"state_rank1.pt")
    inputs=tmp_path/"input";inputs.mkdir();archive=inputs/"resume.bin"
    with tarfile.open(archive,"w") as tar:tar.add(source,arcname="run")
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    hp_sha=hashlib.sha256(json.dumps(hp,sort_keys=True).encode()).hexdigest()
    cfg_sha=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    working=tmp_path/"working";working.mkdir()
    code=dataset_import("run",27,"method",42,digest,hp_sha,cfg_sha).replace('/kaggle/input',str(inputs)).replace('/kaggle/working',str(working))
    exec(code,dict(Path=Path,torch=torch,hashlib=hashlib,json=json))
    assert (working/"run/state_rank1.pt").is_file()
    archive.write_bytes(b"corrupted")
    with pytest.raises(AssertionError):exec(code,dict(Path=Path,torch=torch,hashlib=hashlib,json=json))
