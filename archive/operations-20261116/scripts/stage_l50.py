"""Stage an owner-private, immutable cross-account L50 resume archive with raw-SHA verification."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stage_r20_input import wait_ready


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--owner',required=True);p.add_argument('--run',required=True);p.add_argument('--round',type=int,required=True)
    a=p.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetMetadataRequest
    out=ROOT/'runs/l50-study/transfer';out.mkdir(exist_ok=True)
    ready=out/'ready.json'
    if ready.exists():
        r=json.loads(ready.read_text());assert r['owner']==a.owner and r['round']==a.round
        return
    source=ROOT/'artifacts/l50'/a.run/a.run
    assert (source/'state.pt').is_file() and (source/'state_rank1.pt').is_file()
    import torch
    for name in ('state.pt','state_rank1.pt'):
        s=torch.load(source/name,map_location='cpu',weights_only=True,mmap=True)
        assert s['round']==a.round
        del s
    rows=[json.loads(x) for x in (source/'rounds.jsonl').read_text().splitlines()]
    assert [x['round'] for x in rows]==list(range(1,a.round+1))
    folder=out/'upload';folder.mkdir(exist_ok=True)
    archive=folder/'l50-resume.bin'
    if not archive.exists():
        temp=folder/'l50-resume.bin.tmp'
        with tarfile.open(temp,'w') as tar:tar.add(source,arcname=a.run)
        temp.rename(archive)
    digest=sha(archive)
    ref=f'{a.owner}/mh-l50-checkpoint-20261104-r{a.round}'
    manifest=dict(owner=a.owner,round=a.round,run=a.run,archive_sha256=digest,archive_bytes=archive.stat().st_size)
    (folder/'resume-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (folder/'dataset-metadata.json').write_text(json.dumps(dict(id=ref,title=f'MobiHet L50 private checkpoint round {a.round}',licenses=[dict(name='CC0-1.0')]),indent=2)+'\n')
    api=KaggleApi();api.authenticate()
    intent=out/'create-intent.json'
    if not intent.exists():
        owned=api.dataset_list(mine=True,search='mh-l50-checkpoint-20261104')
        assert all(str(getattr(x,'ref',''))!=ref for x in owned),'Dataset exists without matching receipt; inspect before retry'
        intent.write_text(json.dumps(dict(ref=ref,**manifest))+'\n')
        response=api.dataset_create_new(str(folder),public=False,quiet=True,convert_to_csv=False,dir_mode='skip')
        (out/'create-result.json').write_text(json.dumps(dict(ref=response.ref,status=response.status,error=response.error))+'\n')
        assert not response.error and response.status.lower()!='error'
    else:
        old=json.loads(intent.read_text());assert old['ref']==ref and old['archive_sha256']==digest
    wait_ready(api,ref)
    req=ApiGetDatasetMetadataRequest();req.owner_slug=a.owner;req.dataset_slug=ref.split('/')[1]
    with api.build_kaggle_client() as client:meta=client.datasets.dataset_api_client.get_dataset_metadata(req)
    assert not meta.error_message and meta.info.is_private
    listing=api.dataset_list_files(ref,page_size=100)
    files={f.name:f.total_bytes for f in listing.dataset_files}
    assert not listing.next_page_token and files=={'l50-resume.bin':archive.stat().st_size,'resume-manifest.json':(folder/'resume-manifest.json').stat().st_size}
    verify=out/'download-verify';verify.mkdir(exist_ok=True)
    api.dataset_download_file(ref,'l50-resume.bin',path=str(verify),quiet=True,force=True)
    downloaded=verify/'l50-resume.bin'
    if not downloaded.exists():
        import zipfile
        with zipfile.ZipFile(verify/'l50-resume.bin.zip') as z:z.extractall(verify)
    assert sha(downloaded)==digest
    ready.write_text(json.dumps(dict(ref=ref,private=True,download_sha_verified=True,**manifest),indent=2)+'\n')
    print('L50_TRANSFER_READY',ref,flush=True)


if __name__=='__main__':main()
