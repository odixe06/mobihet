"""Hybrid committed-state audit, coherent full-state payload and Gaussian tables."""
import json
import sys
from pathlib import Path
import torch

ROOT = next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from mh.dagsnet import build_model, CFG
from mh.model_packet import Codec
from mh.v2_math import Layout
from check_v3_driver import equal
from verify_v2 import verify as common


def verify(path, complete=True, infer=False):
    path = Path(path)
    cfg = json.loads((path/'config.json').read_text())['config']
    model = build_model(dict(CFG,n_features=66,num_classes=16),latent=True)
    codec = Codec(model,f'v8:{cfg["arm"]}:{cfg["seed"]}')
    def unpack(packet,expected_base=None):
        _,base,version = codec.decode(packet,expected_base=expected_base)
        return dict(base=base,version=version,kind=254)
    audit = common(path,complete,infer,payload_unpack=unpack,proxy_count=20,smoke_train_rows=1024)
    gaussian = None
    for p in sorted((path/'complete').glob('round_*.json')):
        row = json.loads(p.read_text()); r = row['round']
        weights = torch.load(path/f'weights/round_{r:03d}.pt',map_location='cpu',weights_only=True)
        state = torch.load(path/f'resume/round_{r:03d}.pt',map_location='cpu',weights_only=True)['algorithm']
        if 'payload_sha' in row:
            decoded,_,version = codec.decode((path/f'knowledge/round_{r:03d}.bin').read_bytes(),expected_base=r-1)
            assert version == r and equal(decoded,state['global_state'])
            assert all(equal(decoded,s) for s in weights['proxy'])
            model.load_state_dict(decoded,strict=True)
            assert torch.equal(Layout(model).flatten(model),state['global_vector'])
        if state['gaussian_state'] is not None:
            if gaussian is not None:
                assert equal(gaussian,state['gaussian_state'])
            gaussian = state['gaussian_state']
            assert len(gaussian['tables']) == (20 if cfg['arm'].endswith('_own') else 1)
            assert all(torch.isfinite(v).all() for t in gaussian['tables'] for v in t.values())
    print('VERIFIED V8',path.name,flush=True)
    return audit
