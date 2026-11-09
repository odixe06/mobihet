"""Create and download-verify one receiver-owned private checkpoint archive; no training push."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stage_r20_input import wait_ready
from scripts.pull_checkpoint_parallel import fetch_parallel


def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--arm",required=True);a=ap.parse_args()
    plan=json.loads((ROOT/"runs/y-transfer/plan.json").read_text());spec=next(s for s in plan if s["arm"]==a.arm)
    out=ROOT/"runs/y-transfer"/a.arm;out.mkdir(exist_ok=True)
    if (out/"ready.json").exists():return
    import torch
    source=ROOT/"artifacts/y50"/spec["slug"]/spec["slug"]
    cfg=json.loads((source/"config.json").read_text())
    for rank,name in enumerate(("state.pt","state_rank1.pt")):
        state=torch.load(source/name,map_location="cpu",weights_only=True,mmap=True)
        assert state["round"]==spec["round"] and "rng_cpu" in state and "rng_cuda" in state
        if rank==0:
            assert state["method"]==spec["method"] and state["hp"]==cfg["hp"]
            assert state["hp"]["seed"]==spec["seed"] and state["hp"]["rounds"]==50
            hp_sha=hashlib.sha256(json.dumps(state["hp"],sort_keys=True).encode()).hexdigest()
            cfg_sha=hashlib.sha256(json.dumps(state["cfg"],sort_keys=True).encode()).hexdigest()
        del state
    rows=[json.loads(x) for x in (source/"rounds.jsonl").read_text().splitlines()]
    assert [x["round"] for x in rows]==list(range(1,spec["round"]+1))
    upload=out/"upload";upload.mkdir(exist_ok=True);archive=upload/"resume.bin"
    if not archive.exists():
        with tarfile.open(upload/"resume.bin.tmp","w") as tar:tar.add(source,arcname=spec["slug"])
        (upload/"resume.bin.tmp").rename(archive)
    digest=sha(archive);owner=spec["target_account"];ref=f'{owner}/mh-{a.arm.lower()}-checkpoint-20261107-r{spec["round"]}'
    manifest=dict(ref=ref,owner=owner,run=spec["slug"],round=spec["round"],archive_sha256=digest,archive_bytes=archive.stat().st_size,
                  hp_sha256=hp_sha,cfg_sha256=cfg_sha,overlay_sha256=spec["overlay_sha256"],parent_ref=f'{spec["account"]}/{spec["parent_kernel"]}/1')
    (upload/"resume-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (upload/"dataset-metadata.json").write_text(json.dumps(dict(id=ref,title=f'MobiHet {a.arm} private checkpoint',licenses=[dict(name="CC0-1.0")]),indent=2)+"\n")
    from kaggle.api.kaggle_api_extended import KaggleApi,_resolve_download_path
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetMetadataRequest,ApiDownloadDatasetRequest
    api=KaggleApi();api.authenticate();intent=out/"create-intent.json"
    if not intent.exists():
        assert all(str(getattr(d,"ref",""))!=ref for d in api.dataset_list(mine=True,search=ref.split("/")[1]))
        intent.write_text(json.dumps(manifest,indent=2)+"\n")
        r=api.dataset_create_new(str(upload),public=False,quiet=True,convert_to_csv=False,dir_mode="skip")
        (out/"create-result.json").write_text(json.dumps(dict(ref=r.ref,status=r.status,error=r.error))+"\n")
        assert not r.error and r.status.lower()!="error"
    else:assert json.loads(intent.read_text())==manifest
    wait_ready(api,ref)
    req=ApiGetDatasetMetadataRequest();req.owner_slug=owner;req.dataset_slug=ref.split("/")[1]
    with api.build_kaggle_client() as client:meta=client.datasets.dataset_api_client.get_dataset_metadata(req)
    assert not meta.error_message and meta.info.is_private
    listing=api.dataset_list_files(ref,page_size=100)
    assert not listing.next_page_token
    assert {f.name:f.total_bytes for f in listing.dataset_files}=={"resume.bin":archive.stat().st_size,"resume-manifest.json":(upload/"resume-manifest.json").stat().st_size}
    verify=out/"download-verify";verify.mkdir(exist_ok=True)
    req=ApiDownloadDatasetRequest();req.owner_slug=owner;req.dataset_slug=ref.split("/")[1];req.file_name="resume.bin"
    with api.build_kaggle_client() as client:response=client.datasets.dataset_api_client.download_dataset(req)
    url=response.request.url;destination=Path(_resolve_download_path(str(verify),"resume.bin",url));response.close()
    try:fetch_parallel(url,destination,8)
    except Exception as error:raise RuntimeError("Verification download failed: "+type(error).__name__) from None
    downloaded=destination
    if destination.suffix==".zip":
        import zipfile
        with zipfile.ZipFile(destination) as z:z.extractall(verify)
        downloaded=verify/"resume.bin"
    assert sha(downloaded)==digest
    (out/"ready.json").write_text(json.dumps(dict(manifest,private=True,download_sha_verified=True),indent=2)+"\n")
    # Verification-only downloads are scratch; source run and upload archive remain intact.
    downloaded.unlink()
    if destination!=downloaded:destination.unlink()
    print("Y50_PRIVATE_CHECKPOINT_READY",a.arm,ref,flush=True)


if __name__=="__main__":main()
