"""Noise synthesis from true-class x teacher-probability x input joint moments.

Matched mean-only control, correct full-count inclusion weights. Frozen-head screening only;
no raw records communicated, no official test, no online/contact/R2 proof.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fd_probe23 import C, P, basis, decode_stats, encode_stats, fd, fit, forward, report, sha, virtual
from fd_probe24 import adapt
from mh.v2_math import pack_vector, unpack


@torch.no_grad()
def probabilities(model, x):
    return torch.cat([model.head(model.combined(x[s:s+2048])).softmax(1) for s in range(0,len(x),2048)])


def synthesize(teacher, initial, labels, active, mean, covariance, qmean, cross, joint):
    result = initial.clone(); diagnostics = {}
    for c in range(C):
        slots = torch.where(labels==c)[0]
        x0 = initial[slots][:,active]; x = x0.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([x],lr=.03)
        scale = covariance[c].diagonal().sqrt().float().clamp_min(.1)
        qscale = (qmean[c]*(1-qmean[c])).clamp_min(0).sqrt().float().clamp_min(.05)
        cov_target = covariance[c].float()/scale[:,None]/scale[None,:]
        cross_target = cross[c].float()/qscale[:,None]/scale[None,:]
        def losses():
            full = torch.zeros(len(x),66,device=x.device); full[:,active]=x
            q = teacher.head(teacher.combined(full)).softmax(1)
            z = (x-mean[c].float())/scale
            qcenter = q-q.mean(0); zcenter=z-z.mean(0)
            return dict(input_mean=z.mean(0).square().mean(),
                        input_cov=(zcenter.T@zcenter/len(x)-cov_target).square().mean(),
                        probability_mean=((q.mean(0)-qmean[c].float())/qscale).square().mean(),
                        cross=((qcenter/qscale).T@zcenter/len(x)-cross_target).square().mean(),
                        trust=((x-x0)/scale).square().mean())
        for _ in range(80):
            terms=losses(); loss=terms['input_mean']+terms['input_cov']+terms['probability_mean']+.01*terms['trust']
            if joint:loss=loss+terms['cross']
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        with torch.no_grad():
            terms=losses();diagnostics[c]={k:float(v) for k,v in terms.items()}
            assert torch.isfinite(x).all()
            result[slots[:,None],active[None,:]]=x.detach()
    return result,diagnostics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--weights',required=True);parser.add_argument('--fixture',required=True);parser.add_argument('--out',required=True)
    a=parser.parse_args();start=time.monotonic();fd.deterministic();torch.set_num_threads(2);dev=torch.device('cuda')
    log=lambda *v:print(round(time.monotonic()-start,1),*v,flush=True)
    fx=Path(a.fixture);manifest=json.loads((fx/'manifest.json').read_text())
    for name,expected in manifest['files'].items():assert sha(fx/name)==expected
    assert manifest['train_frac']==.025 and manifest['group_overlap']==0
    population_path=Path(a.weights).parents[3]/'source/data_manifest.json'
    population=json.loads(population_path.read_text())
    zero=[manifest['features'].index(name) for name in fd.BLOCK+fd.POSITION]
    active=torch.tensor([j for j in range(66) if j not in zero],device=dev);assert len(active)==53
    def read(name):
        x=torch.from_numpy(np.load(fx/f'{name}_X.npy').astype('float32')).to(dev);x[:,zero]=0
        return x,torch.from_numpy(np.load(fx/f'{name}_y.npy').astype('int64')).to(dev)
    weights=torch.load(a.weights,map_location=dev,weights_only=True)
    teacher=P.make('dagsnet').to(dev);teacher.load_state_dict(weights['proxy'][0],strict=True);teacher.eval().requires_grad_(False)
    projection=basis(teacher.head[5].weight,48);generator=torch.Generator().manual_seed(22)
    own={};total=0;raw_input=[];weighted_input=[];carriers=[];raw_carriers=[];up_bytes=[];weights_log=[]
    for k in range(20):
        x,y=read(f'train_{k}');idx=torch.randperm(len(y),generator=generator)[:16384].to(dev);x,y=x[idx],y[idx]
        z,h,_=forward(teacher,x,feats=True);own[k]=(x,y,h.float(),z)
        total=total+encode_stats(h,y,'project48',projection)[0]
        sampled=torch.bincount(y,minlength=C).double()
        full=torch.tensor(population['counts'][f'train_{k}'],device=dev,dtype=torch.float64)
        assert not ((full>0)&(sampled==0)).any();factor=full/sampled.clamp_min(1)
        probability=probabilities(teacher,x).double();xx=x[:,active].double()
        one=F.one_hot(y,C).double();s=one.T@x.double()
        qq=torch.stack([x[y==c].double().T@x[y==c].double() for c in range(C)])
        raw_input.append((sampled,s,qq));weighted_input.append((full,s*factor[:,None],qq*factor[:,None,None]))
        psum=one.T@probability
        px=torch.stack([probability[y==c].T@xx[y==c] for c in range(C)])
        values=torch.cat([full,(psum*factor[:,None]).flatten(),(px*factor[:,None,None]).flatten()])
        packet=pack_vector('moments',values,base=15,version=15,count=len(y))
        raw_carriers.append(values)
        carriers.append(unpack(packet,dev,expected_base=15)['values'].double());up_bytes.append(len(packet))
        weights_log.append(dict(client=k,sample_counts=sampled.tolist(),population_counts=full.tolist()))
    n,mu,cov=decode_stats(total,'project48',projection)
    hbank,ybank=virtual(mu,cov,n/n.sum(),200000,23)
    fc=torch.nn.Linear(48,C).to(dev)
    with torch.no_grad():fc.weight.copy_(teacher.head[5].weight@projection.float());fc.bias.copy_(teacher.head[5].bias)
    fc=fit(fc,hbank,ybank,3000,23);del hbank,ybank
    calibrated=copy.deepcopy(teacher)
    with torch.no_grad():calibrated.head[5].weight.copy_(fc.weight@projection.float().T);calibrated.head[5].bias.copy_(fc.bias)
    def input_moments(parts):
        nn=sum(p[0] for p in parts);ss=sum(p[1] for p in parts);qq=sum(p[2] for p in parts)
        mm=ss/nn[:,None];cc=(qq-nn[:,None,None]*mm[:,:,None]*mm[:,None,:])/(nn[:,None,None]-1)
        return nn,mm,(cc+cc.transpose(1,2))/2
    _,im,ic=input_moments(raw_input);wn,wm,wc=input_moments(weighted_input)
    share=torch.ones(C,device=dev)/C
    original,labels=virtual(im,ic,share,4096,24);original[:,zero]=0
    initial,wl=virtual(wm,wc,share,4096,24);initial[:,zero]=0;assert torch.equal(labels,wl)
    aggregate=torch.stack(carriers).sum(0)
    down=pack_vector('moments',aggregate,base=15,version=15,count=20*16384)
    decoded=unpack(down,dev,expected_base=15)['values'].double()
    bn=decoded[:C];bq=decoded[C:C+C*C].reshape(C,C)/bn[:,None]
    bqx=decoded[C+C*C:].reshape(C,C,53)/bn[:,None,None]
    cross=bqx-bq[:,:,None]*wm[:,None,active]
    raw=torch.stack(raw_carriers).sum(0);raw_q=raw[C:C+C*C].reshape(C,C)/raw[:C,None]
    raw_cross=raw[C+C*C:].reshape(C,C,53)/raw[:C,None,None]-raw_q[:,:,None]*wm[:,None,active]
    cross_codec_error=float((cross-raw_cross).norm()/raw_cross.norm().clamp_min(1e-12))
    assert cross_codec_error<1e-3,('centered carrier distorted by codec',cross_codec_error)
    # Quantization must not turn the class counts or conditional probability masses into a different estimator.
    assert torch.equal(bn,wn) and torch.allclose(bq.sum(1),torch.ones(C,device=dev,dtype=torch.float64),atol=1e-6)
    supports=dict(gaussian_original=original,gaussian_weighted=initial);generation={}
    for mode in ('mean_only','zero_cross','joint'):
        target_cross=torch.zeros_like(cross) if mode=='zero_cross' else cross
        supports[mode],generation[mode]=synthesize(teacher,initial,labels,active,wm[:,active],wc[:,active][:,:,active],bq,target_cross,mode!='mean_only')
        log('generated',mode,{k:round(float(np.mean([v[k] for v in generation[mode].values()])),5) for k in generation[mode][0]})
    evals={s:read(s) for s in ('old','group')}
    targets={s:forward(calibrated,x).log_softmax(1) for s,x in supports.items()}
    target_mass={s:targets[s].exp().mean(0).tolist() for s in supports}
    out=dict(scope='local frozen-head mechanism screen; full-count inclusion weights; no online FL or official test',
             weights_sha=sha(a.weights),fixture_sha=sha(fx/'manifest.json'),population_sha=sha(population_path),source_sha=sha(__file__),
             dependencies={name:sha(Path(__file__).with_name(name)) for name in ('fd_probe23.py','fd_probe24.py','fd_probe14.py','fd.py')},
             protocol=dict(clients=20,bank=4096,generator_steps=80,head_steps=1000,joint_weight=1.,teacher_for_B='original FP32',teacher_for_KD='Project48 AMP'),
             sampling=weights_log,carrier=dict(up_bytes_per_client=up_bytes,down_bytes_per_client=len(down),
             raw_values=len(decoded),centered_cross_relative_codec_error=cross_codec_error,
             scope='B only; existing input moments/bootstrap, calibration, proxy, SecAgg and contact overhead excluded'),
             input_scaled_cross_energy=float((cross/wc[:,active][:,:,active].diagonal(dim1=1,dim2=2).sqrt().clamp_min(.1)[:,None,:]).square().mean()),
             generation=generation,target_mass=target_mass,clients={})
    predictions={s+'_y':yy.cpu().numpy().astype('uint8') for s,(_,yy) in evals.items()};heads={}
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    global_counts=torch.tensor([sum(manifest['counts'][f'train_{j}'][c] for j in range(20)) for c in range(C)],device=dev)
    pg=((global_counts+1)/(global_counts.sum()+C)).log()
    for k in range(20):
        private=P.make('dagsnet').to(dev);private.load_state_dict(weights['private'][k],strict=True);private.eval().requires_grad_(False)
        x,y,ht,zt=own[k];hp=forward(private,x,feats=True)[1].float()
        sh={s:forward(private,xx,feats=True)[1].float() for s,xx in supports.items()}
        tests={s:(yy,forward(private,xx,feats=True)[1].float()) for s,(xx,yy) in evals.items()}
        counts=torch.tensor(manifest['counts'][f'train_{k}'],device=dev);shift=((counts+1)/(counts.sum()+C)).log()-pg
        with torch.no_grad():real_target=calibrated.head[5](ht).log_softmax(1)
        proxy_ce=F.cross_entropy(zt,y).detach();out['clients'][k]={}
        for s in supports:
            layer=adapt(private.head[5],hp,y,sh[s],targets[s],real_target,proxy_ce,shift,1000,2400+k)
            heads[f'{k}_{s}']={name:value.detach().cpu() for name,value in layer.state_dict().items()}
            out['clients'][k][s]={}
            for split,(yy,hh) in tests.items():
                with torch.no_grad():pred=layer(hh).argmax(1)
                predictions[f'{k}_{s}_{split}']=pred.cpu().numpy().astype('uint8');out['clients'][k][s][split]=report(yy,pred)
        path.write_text(json.dumps(out,indent=1,allow_nan=False)+'\n')
        log('client',k,{s:round(v['group']['metrics']['f1_macro'],5) for s,v in out['clients'][k].items()})
        del private,hp,sh,tests
    np.savez_compressed(path.with_suffix('.npz'),**predictions);torch.save(heads,path.with_suffix('.pt'))
    out.update(predictions_sha=sha(path.with_suffix('.npz')),heads_sha=sha(path.with_suffix('.pt')),seconds=time.monotonic()-start)
    path.write_text(json.dumps(out,indent=1,allow_nan=False)+'\n');log('saved',str(path))


if __name__=='__main__':main()
