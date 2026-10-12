"""Read committed A states; quantify sketch/residual growth without retraining."""
import json
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from mh.v2_math import CountSketch


def main():
    torch.set_num_threads(1); records=[]
    for run in sorted((ROOT/'papers/v2/runs').glob('a_*')):
        arm=run.name.rsplit('_s',1)[0]
        if 'small' not in arm and 'medium' not in arm: continue
        depth,width=(3,256) if 'small' in arm else (5,2048)
        sketch=CountSketch(395024,depth,width,1701,'cpu'); previous=None
        for marker in sorted((run/'complete').glob('round_*.json')):
            row=json.loads(marker.read_text()); r=row['round']
            state=torch.load(run/'resume'/f'round_{r:03d}.pt',map_location='cpu',weights_only=True)['algorithm']
            vector=state['global_vector']; residual=state['residual']; mean=state['last_mean']
            record=dict(run=run.name,round=r,parameter_max=float(vector.abs().max()),
                        parameter_norm=float(vector.norm()),mean_sketch_norm=float(mean.norm()),
                        residual_norm=float(residual.norm()))
            if previous is not None:
                target=mean+previous['residual']; decoded=sketch.encode(vector-previous['global_vector'])
                denom=float(target.norm())
                record.update(decoded_sketch_norm=float(decoded.norm()),
                    residual_fraction=float(residual.norm())/denom if denom else None,
                    decoded_fraction=float(decoded.norm())/denom if denom else None)
            records.append(record); previous=state
    out=ROOT/'papers/v2/sketch-diagnostics.json'; out.write_text(json.dumps(records,indent=2))
    print('wrote',out,'rows',len(records))


if __name__=='__main__': main()
