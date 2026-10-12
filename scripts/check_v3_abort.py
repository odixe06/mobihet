"""Actual saved-update PowerSGD stage-two fault: no model/residual/version commit."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from mh.dagsnet import build_model,CFG
from mh.v3_methods import Method
from mh.v3_math import ContextSum


def main():
    torch.set_num_threads(1); torch.manual_seed(4901)
    source=ROOT/'papers/v2/runs/a_dense_s42/resume/round_001.pt'
    delta=torch.load(source,map_location='cpu',weights_only=True)['algorithm']['last_mean']
    def create():
        torch.manual_seed(4902)
        run=SimpleNamespace(init=build_model(dict(CFG,n_features=66,num_classes=16),latent=True),
            K=5,dev=torch.device('cpu'),n=[17]*5,bytes=[],a=SimpleNamespace(arm='a_p_medium_ef'))
        return Method(run)
    targets=[delta+.001*torch.randn_like(delta) for _ in range(5)]
    def stage(method):
        method.start_round(); method.staged=[v.clone() for v in targets]; method.local_delta=method.staged
        for k,v in enumerate(targets):
            packet,_=method.packet(method.project(v),k,1,'P'); method.aggregate_sum.add(k,packet)
    method=create(); stage(method); initial=method.global_vector.clone()
    residual=[v.clone() for v in method.client_residual]; finish=ContextSum.finish
    def injected(instance):
        if instance is method.aggregate_sum: return finish(instance)
        raise RuntimeError('injected second-stage failure')
    with patch.object(ContextSum,'finish',injected):
        try: method.aggregate(1)
        except RuntimeError as error: assert str(error)=='injected second-stage failure'
        else: raise AssertionError('fault not reached')
    assert method.version==0 and torch.equal(method.global_vector,initial)
    assert all(torch.equal(a,b) for a,b in zip(residual,method.client_residual))
    assert not hasattr(method,'latest_packet')
    failed_bytes=sum(v[-1] for v in method.R.bytes)
    stage(method); method.aggregate(1)
    fresh=create(); stage(fresh); fresh.aggregate(1)
    assert torch.equal(method.global_vector,fresh.global_vector)
    assert all(torch.equal(a,b) for a,b in zip(method.client_residual,fresh.client_residual))
    result=dict(evidence='verified dense r1 update, actual two-stage production-rank8 codec, CPU five-client math fault; no training/R1 claim',
        fault_stage='Q aggregate finish',version_not_committed=True,model_not_committed=True,residual_not_committed=True,
        restart_round_matches_fresh=True,paid_application_bytes_before_fault=failed_bytes,
        limit='U driver crashes before checkpoint on fault; finite-contact durable failed-frame accounting still needs separate contact-stage implementation')
    (ROOT/'papers/v3/abort-probe.json').write_text(json.dumps(result,indent=2))
    print('PASS V3 actual PowerSGD stage-two abort/whole-round recovery; failed bytes recorded in probe')


if __name__=='__main__': main()
