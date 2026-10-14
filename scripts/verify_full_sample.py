"""Bounded actual-test inference against stored full-run predictions, without changing frozen source."""
import argparse
import fcntl
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mh.dagsnet import build_model,CFG


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--build',choices=('full-validation','hybrid-full'),default='full-validation')
    build=parser.parse_args().build
    base=ROOT/'papers'/build
    meta=json.loads((ROOT/'knowledge/meta.json').read_text())
    scaler=json.loads((ROOT/'knowledge/scaler.json').read_text())
    columns=meta['feature_cols'];raw=ROOT.parent/'dataset/centralized/test/part-00000.parquet'
    batch=next(pq.ParquetFile(raw).iter_batches(batch_size=16384,columns=columns+['label']))
    x=np.stack([batch.column(c).to_numpy() for c in columns],1).astype('float64')
    mean=np.array([scaler['features'][c]['mean'] for c in columns])
    sd=np.array([scaler['features'][c]['std_used'] for c in columns])
    x=((x-mean)/sd).astype('float16');labels=batch.column('label').to_numpy().astype('uint8')
    assert np.isfinite(x).all()
    audits=[];torch.set_num_threads(1)
    with (ROOT/'papers/.artifact-verification.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True).cuda().eval()
        assert sum(p.numel() for p in model.parameters())==395024
        tensor=torch.from_numpy(x).cuda().float()
        for marker in sorted((base/'kaggle').glob('*/AUDITED.json')):
            record=json.loads(marker.read_text())
            manifest=json.loads((marker.parent/'source/data_manifest.json').read_text())
            counts=torch.tensor([manifest['counts'][f'train_{k}'] for k in range(20)],device='cuda',dtype=torch.float64)
            log_pi=((counts+1)/(counts+1).sum(1,keepdim=True)).log().float()
            for item in record.get('audits',[]):
                run=marker.parent/'runs'/f'{item["arm"]}_s42'
                for r in sorted({1,item['rounds']}):
                    state=torch.load(run/f'weights/round_{r:03d}.pt',map_location='cpu',weights_only=True)
                    resume=torch.load(run/f'resume/round_{r:03d}.pt',map_location='cpu',weights_only=True)
                    with np.load(run/f'predictions/round_{r:03d}.npz') as stored:
                        assert np.array_equal(labels,stored['test_y'][:len(labels)])
                        for role in ('private','proxy'):
                            if not state[role]:continue
                            model.load_state_dict(state[role][0],strict=True)
                            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):z=model(tensor)[0].float()
                            assert torch.isfinite(z).all()
                            for rule in (('head','gp') if role=='private' else ('head',)):
                                logits=z+(resume['log_pg'].cuda()-log_pi[0] if rule=='gp' else 0.)
                                decisive=logits.topk(2,1).values.diff(dim=1).abs().squeeze(1)>.02
                                original=torch.from_numpy(stored[f'{role}_test_{rule}'][0,:len(labels)].astype('int64')).cuda()
                                assert torch.equal(logits.argmax(1)[decisive],original[decisive]),('full inference mismatch',item['arm'],r,role,rule)
                                audits.append(dict(arm=item['arm'],round=r,role=role,rule=rule,rows=len(labels),decisive=int(decisive.sum()),
                                    ambiguous=int((~decisive).sum()),weights_sha=hashlib.sha256((run/f'weights/round_{r:03d}.pt').read_bytes()).hexdigest()))
    proof=dict(evidence='actual local raw-test prefix, scaler once, fp16 storage, saved predictions; client0 sampled only',
               raw_source=str(raw),parameter_count=395024,audits=audits)
    (base/'inference-proof.json').write_text(json.dumps(proof,indent=2))
    print('PASS actual full-test sample inference',len(audits),'checks; not full-data inference or a leakage audit')


if __name__=='__main__':main()
