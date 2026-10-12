"""V4 saved-state invariants plus independent V2 model/payload/prediction audit."""
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from verify_v2 import verify as common
from mh.v2_math import unpack


def verify(out,complete=True,infer=False):
    out=Path(out); cfg=json.loads((out/'config.json').read_text())['config']; arm=cfg['arm']
    support=arm.startswith('s_')
    result=common(out,complete,infer,proxy_count=0 if support else None)
    rows=[json.loads(p.read_text()) for p in sorted((out/'complete').glob('round_*.json'))]
    first_tables=None
    for row in rows:
        r=row['round']; state=torch.load(out/f'resume/round_{r:03d}.pt',map_location='cpu',weights_only=True)
        algorithm=state['algorithm']
        if support:
            table=algorithm['tables']; assert torch.isfinite(table).all()
            assert table.shape==((20,5,16,66) if arm=='s_own' else (5,16,66))
            if first_tables is None: first_tables=table
            else: assert torch.equal(table,first_tables),'support knowledge changed after one-shot bootstrap'
            if arm!='s_own':
                assert torch.equal(unpack(algorithm['support_packet'])['values'].reshape(5,16,66),table)
                assert len(algorithm['support_packet'])==10592
            assert not row['proxy_clients']
        elif 'mala' in arm and r>1:
            assert len(algorithm['chains'])==20
            for client in row['diagnostics']['mala']['clients']:
                k=client['client']; batches=(row['clients'][k]['rows']+cfg['batch']-1)//cfg['batch']
                aux=cfg['aux_batch']; assert client['transitions']==20+batches
                assert client['attempted']==(20+batches)*aux and 0<=client['accepted']<=client['attempted']
                assert client['acceptance']==client['accepted']/client['attempted'] and client['esjd']>=0
                assert algorithm['chain_round'][k]==r
                assert algorithm['chains'][k].shape==(aux,66) and torch.isfinite(algorithm['chains'][k]).all()
    print('VERIFIED V4',out.name)
    return result
