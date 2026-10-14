"""Actual MALA math, five-arm smoke and bitwise interrupted-round recovery."""
import json
import argparse
import subprocess
import sys
from pathlib import Path
import torch
ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1]); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from mh.v4_mala import transition,accept,potential
from mh.v4_run import FAMILIES
from mh.dagsnet import build_model,CFG
from check_v3_driver import equal
from verify_v4 import verify


def math_check():
    torch.set_num_threads(1); dev='cuda'; g=torch.Generator(device=dev).manual_seed(4701)
    z=torch.randn(65536,2,device=dev,generator=g); accepted=0
    for _ in range(32):
        lp=-.5*z.square().sum(1); score=-z
        proposal=transition(z,lp,score,g)
        new_lp=-.5*proposal.square().sum(1)
        # Gaussian MALA exact log acceptance is eta/4*(||z||²-||proposal||²).
        forward=(proposal.double()-z.double()-.01*score.double()).square().sum(1)
        reverse=(z.double()-proposal.double()+.01*proposal.double()).square().sum(1)
        actual=new_lp.double()-lp.double()+(forward-reverse)/.04
        expected=.01/4*(z.double().square().sum(1)-proposal.double().square().sum(1))
        assert torch.allclose(actual,expected,atol=3e-6,rtol=2e-3)
        z,_,_,take=accept(z,proposal,lp,score,new_lp,-proposal,g); accepted+=int(take.sum())
    assert float(z.mean().abs())<.015 and abs(float(z.square().mean())-1)<.02
    model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True).to(dev).eval()
    mu=torch.zeros(16,66,device=dev); sd=torch.ones_like(mu)
    labels=torch.tensor([0,1],device=dev); zz=torch.randn(2,66,device=dev,generator=g)
    before=[p.requires_grad for p in model.parameters()]
    lp,score=potential(model,zz,labels,mu,sd)
    assert before==[p.requires_grad for p in model.parameters()]
    plus=zz.clone(); minus=zz.clone(); plus[0,4]+=.005; minus[0,4]-=.005
    a,_=potential(model,plus,labels,mu,sd); b,_=potential(model,minus,labels,mu,sd)
    assert abs(float((a[0]-b[0])/.01-score[0,4]))<.02
    record=dict(evidence='Gaussian stationary-chain null and DAGSNet finite difference; no fitted density/mixing claim',
        gaussian_mean=float(z.mean()),gaussian_second_moment=float(z.square().mean()),
        acceptance=accepted/(32*len(z)),input_gradient_finite_difference=True)
    (ROOT/'papers/v4/mala-math.json').write_text(json.dumps(record,indent=2))
    del model; torch.cuda.empty_cache()


def run(arm,out,extra=()):
    subprocess.run([sys.executable,'-m','mh.v4_run','--arm',arm,'--smoke','--out',str(out),*extra],cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--tag',default='mala'); args=parser.parse_args()
    math_check(); base=ROOT/'papers/v4/smoke'/args.tag; resumed=ROOT/'papers/v4/resume_reference'/args.tag
    for arm in FAMILIES['m']:
        assert not (base/f'{arm}_s42').exists(),'fresh smoke path required'
        run(arm,base)
    for arm in ('c_mala_fl','c_mala_own','c_mala_shuffle'):
        run(arm,resumed,('--stop_after','1')); run(arm,resumed,('--resume',))
        for r in range(1,4):
            for kind in ('weights','resume'):
                a=torch.load(base/f'{arm}_s42/{kind}/round_{r:03d}.pt',weights_only=True)
                b=torch.load(resumed/f'{arm}_s42/{kind}/round_{r:03d}.pt',weights_only=True)
                assert equal(a,b),(arm,r,kind)
    for arm in FAMILIES['m']: verify(base/f'{arm}_s42',infer=True)
    for arm in ('c_mala_fl','c_mala_own','c_mala_shuffle'):
        identity=json.loads((base/f'{arm}_s42/config.json').read_text())
        for r in (2,3):
            path=base/f'{arm}_s42'; row=json.loads((path/f'complete/round_{r:03d}.json').read_text())
            state=torch.load(path/f'resume/round_{r:03d}.pt',weights_only=True)['algorithm']
            for client in row['diagnostics']['mala']['clients']:
                k=client['client']; assert client['transitions']==21
                assert 0<=client['acceptance']<=1 and client['esjd']>=0
                assert state['chain_round'][k]==r and torch.isfinite(state['chains'][k]).all()
    print('PASS MALA null/sign/score, five arms x3rounds, three exact resumes and artifact inference')


if __name__=='__main__': main()
