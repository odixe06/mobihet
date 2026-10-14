"""Positive H1/H2 long validation; actual coherent proxy and Gaussian packets."""
import copy
from types import SimpleNamespace
import torch
from mh import pilot as P
from mh.v2_math import Layout
from mh.v2_methods import Method as Prior
from mh.v7_nystrom import Method as Gaussian
from mh.model_packet import Codec,Sum

ARMS=['h_local','h_gaussian','h_gaussian_diag','h_gaussian_lr4',
      'h_gaussian_own','h_dml','h_dml_own']


class Method(Prior):
    def __init__(self,run,initialize=True):
        self.R,self.arm,self.diag=run,run.a.arm,{}
        self.role,self.version='private',0
        self.kd='dml' in self.arm;self.aux='gaussian' in self.arm
        self.own=self.arm.endswith('_own')
        self.layout=Layout(run.init);self.global_vector=self.layout.flatten(run.init).clone()
        self.proxy=[];self.global_state={};self.gaussian_state=None;self.gaussian=None
        if self.kd:
            torch.manual_seed(run.a.seed+1000);initial=P.make('dagsnet').to(run.dev)
            self.proxy=[copy.deepcopy(initial) for _ in range(run.K)]
            self.global_state={name:value.clone() for name,value in initial.state_dict().items()}
            self.codec=Codec(initial,f'full:{self.arm}:{run.a.seed}')
        if self.aux:
            form='diag' if self.arm.endswith('diag') else 'lr4' if self.arm.endswith('lr4') else 'full'
            args=SimpleNamespace(arm='n_'+form,seed=run.a.seed)
            self.gaussian=Gaussian(SimpleNamespace(init=run.init,dev=run.dev,a=args),initialize=False)
            if initialize:
                self.gaussian.R=run;self.gaussian.mode='own' if self.own else 'fl'
                self.gaussian.make_tables();self.gaussian_state=self.gaussian.state()

    def prior(self):
        super().prior()
        if self.kd and not self.own:
            packet=self.codec.encode(self.global_state,0,0)
            self.global_state,_,_=self.codec.decode(packet,self.R.dev)
            for k in range(self.R.K):self.record(packet,0,k,'down','proxy_bootstrap')

    def prepare(self,k,r):
        if self.kd and not self.own:self.proxy[k].load_state_dict(self.global_state,strict=True)
        return self.proxy[k]

    def start_round(self):
        self.logs={}
        if self.kd and not self.own:self.aggregate_sum=Sum(self.codec,self.version,self.R.dev)

    def proxy_epoch(self,k,r):
        packet=None
        if not self.own:
            packet=self.codec.encode(self.proxy[k].state_dict(),r-1,r)
            self.record(packet,r,k,'up','proxy_state')
        return packet,self.logs[k]

    def aggregate(self,r):
        if self.kd and not self.own:
            state=self.aggregate_sum.finish(self.R.K)
            packet=self.codec.encode(state,r-1,r)
            self.global_state,_,_=self.codec.decode(packet,self.R.dev,r-1)
            for k in range(self.R.K):
                self.record(packet,r,k,'down','proxy_state')
                self.proxy[k].load_state_dict(self.global_state,strict=True)
            self.global_vector=self.layout.flatten(self.proxy[0]).clone();self.latest_packet=packet
        self.version=r

    def state(self):
        return dict(global_vector=self.global_vector,global_state=self.global_state,version=self.version,
                    gaussian_state=self.gaussian_state)

    def draw(self,labels,g,k):
        self.gaussian.R=self.R;self.gaussian.mode='own' if self.own else 'fl'
        for name,value in self.gaussian_state.items():setattr(self.gaussian,name,value)
        return self.gaussian.draw(labels,g,k)
