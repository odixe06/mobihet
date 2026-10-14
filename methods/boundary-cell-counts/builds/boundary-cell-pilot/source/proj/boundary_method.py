"""Private DAGSNet learns cohort cell targets on its own records; no row carrier."""
import copy
import hashlib
import json
import numpy as np
import torch
import torch.nn.functional as F
from mh import pilot as P
from mh.v2_methods import Method as PaidPrior, finite
from cells import C, PAIRS, EDGES, count, targets, packets, packet, decode

ARMS = ['bc_single', 'bc_local', 'bc_mean', 'bc_gaussian'] + [
    f'bc_{mode}{p}_{scope}' for mode in ('direct', 'teacher')
    for p in (2, 4) for scope in ('fl', 'own', 'shuffle', 'marginal')]
CALIBRATION = ['bc_single', 'bc_direct2_fl', 'bc_direct4_fl', 'bc_teacher2_fl', 'bc_teacher4_fl']


class Method:
    def __init__(self, run, initialize=True):
        self.R, self.arm = run, run.a.arm
        self.role, self.version, self.diag = 'private', 0, {}
        self.mode = self.arm.split('_')[1]
        self.p = 2 if '2' in self.mode else 4
        self.scope = self.arm.split('_')[-1]
        self.is_cells = self.mode.startswith(('direct', 'teacher')) or self.mode == 'mean'
        self.names = json.loads((P.ROOT/'papers/pilot/fixture/manifest.json').read_text())['features']
        self.axes = torch.tensor([[self.names.index(a), self.names.index(b)] for a,b in PAIRS[:self.p]], device=run.dev)
        self.edges = torch.from_numpy(EDGES.copy()).to(run.dev)
        self.mass = torch.zeros(run.K, C, dtype=torch.float64, device=run.dev)
        self.log_pi_effective = run.log_pi.clone()
        self.proxy = []
        if self.mode != 'single':
            torch.manual_seed(run.a.seed+1000)
            initial = P.make('dagsnet').to(run.dev)
            self.proxy = [copy.deepcopy(initial) for _ in range(run.K)]
        self.tables, self.mask = [], None
        self.identity = hashlib.sha256(json.dumps(dict(pairs=PAIRS[:self.p],edges=EDGES.tolist(),
            fixture=json.loads((P.ROOT/'papers/pilot/fixture/manifest.json').read_text())['source_id']),sort_keys=True).encode()).digest()[:16]
        if self.is_cells:
            local = [count(x.float().cpu().numpy(), y.cpu().numpy(), self.names, self.p)
                     for x,y in zip(run.X, run.Y)]
            # Serialize then decode every uplink before the trusted sum.
            decoded = []
            for ns,ny in local:
                body = ns.astype('<u4').tobytes()+(ns>0).astype('u1').tobytes()+ny.astype('<u8').tobytes()
                body = decode(packet(1,self.p,body,self.identity),1,self.p,self.identity)
                size = ns.size
                got = np.frombuffer(body,dtype='<u4',count=size).reshape(ns.shape).copy()
                bits = np.frombuffer(body,dtype='u1',offset=4*size,count=size).reshape(ns.shape)
                total_n = np.frombuffer(body,dtype='<u8',offset=5*size,count=C).copy()
                assert np.array_equal(bits,got>0) and np.array_equal(got.sum(1),np.broadcast_to(total_n,got.sum(1).shape))
                decoded.append((got,total_n))
            local = decoded
            totals = sum((n.astype(np.uint64) for n,_ in local), np.zeros_like(local[0][0], dtype=np.uint64))
            n = sum((ny for _,ny in local), np.zeros(C, dtype=np.uint64))
            contributors = sum((ns>0 for ns,_ in local), np.zeros_like(totals, dtype=np.uint8))
            eligible = ~((totals>0)&(contributors<5)).any(-1) & (totals.sum(-1)>=32)
            self.mask = torch.from_numpy(eligible).to(run.dev)
            table, _ = targets(totals, n, eligible)
            population_prior = ((n+1)/(n.sum()+C)).astype('float32')
            if self.scope == 'shuffle':
                rng = np.random.default_rng(701)
                for r in range(self.p):
                    cells = np.flatnonzero(eligible[r])
                    table[r,cells] = table[r,rng.permutation(cells)]
            if self.scope == 'marginal':
                shaped = totals.reshape(self.p, 4, 4, C)
                margins = []
                for axis in (2,1):
                    likelihood = (shaped.sum(axis=axis)+1.)/(n+4)
                    margins.append(likelihood/likelihood.sum(-1,keepdims=True))
                table = ((margins[0][:,:,None]+margins[1][:,None,:])/2).reshape(self.p,16,C).astype('float32')
                table[~eligible] = 1./C
            for k,(ns,ny) in enumerate(local):
                own_table, _ = targets(ns, ny, eligible)
                self.tables.append(torch.from_numpy(own_table if self.scope=='own' else table.copy()).to(run.dev))
                if self.scope != 'own':
                    up, down = packets(ns,ny,table,eligible,population_prior,self.identity)
                    body = decode(down,2,self.p,self.identity)
                    used = np.frombuffer(body,dtype='<f4',count=table.size).reshape(table.shape).copy()
                    assert np.array_equal(used,table)
                    self.tables[k] = torch.from_numpy(used).to(run.dev)
                    self.population_prior = torch.from_numpy(np.frombuffer(body,dtype='<f4',offset=len(body)-4*C,count=C).copy()).to(run.dev)
                    packed_mask = body[table.size*4:len(body)-4*C]
                    wire_mask = np.unpackbits(np.frombuffer(packed_mask,dtype='u1'),bitorder='little').reshape(self.p,16,C)
                    assert np.array_equal(wire_mask,np.repeat(eligible[:,:,None],C,axis=-1))
                    self.mask = torch.from_numpy(wire_mask[:,:,0].astype(bool)).to(run.dev)
                    if initialize:
                        self.record(up,0,k,'up','cell_counts')
                        self.record(down,0,k,'down','cell_targets')
                    self.latest_packet = down
                elif initialize:
                    up,_ = packets(ns,ny,table,eligible,population_prior,self.identity)
                    self.record(up,0,k,'up','cell_counts_for_common_mask')
                    mask_packet = packet(5,self.p,np.packbits(eligible.ravel(),bitorder='little').tobytes(),self.identity)
                    self.record(mask_packet,0,k,'down','common_eligibility')
            # Label-conditioned mean retains class knowledge, discards cell location.
            weights = totals*eligible[:,:,None]
            mean = np.einsum('rby,rbz->yz', weights.astype('float64'), table.astype('float64'))
            denom = weights.sum((0,1))
            mean = np.divide(mean,denom[:,None],out=np.full((C,C),1./C),where=denom[:,None]>0)
            self.class_mean = torch.from_numpy(mean.astype('float32')).to(run.dev)
            if self.mode == 'mean':
                mean_packet = packet(4,self.p,mean.astype('<f4').tobytes()+population_prior.astype('<f4').tobytes(),self.identity)
                used = np.frombuffer(decode(mean_packet,4,self.p,self.identity),dtype='<f4',count=C*C).reshape(C,C).copy()
                self.class_mean = torch.from_numpy(used).to(run.dev)
                self.latest_packet = mean_packet
                if initialize:
                    for k in range(run.K): self.record(mean_packet,0,k,'down','class_mean')
            self.diag.update(eligible_cells=int(eligible.sum()),total_cells=int(eligible.size),
                             teacher_steps=20,prototype_or_rule_model=False)
        self.gaussian = None
        if self.mode == 'gaussian':
            from mh.v7_nystrom import Method as Gaussian
            from types import SimpleNamespace
            holder = SimpleNamespace(init=run.init,dev=run.dev,a=SimpleNamespace(arm='n_full',seed=run.a.seed))
            self.gaussian = Gaussian(holder, initialize=False)
            self.gaussian.R = run
            self.gaussian.form, self.gaussian.mode, self.gaussian.rank = 'full','fl',0
            self.gaussian.make_tables()

    def record(self, packet, r, k, direction, kind):
        self.R.bytes.append((r,k,direction,kind,len(packet)))

    def prior(self):
        if self.is_cells and self.scope != 'own':
            self.R.log_pg = self.population_prior.log()
            self.R.ratio_pi = self.population_prior.clone()
            body = self.axes.cpu().numpy().astype('<u2').tobytes()+np.tile(EDGES,self.p*2).astype('<f4').tobytes()
            meta = packet(3,self.p,body,self.identity)
            for k in range(self.R.K): self.record(meta,0,k,'down','bin_dictionary')
        else:
            PaidPrior.prior(self)
            if self.is_cells:
                body = self.axes.cpu().numpy().astype('<u2').tobytes()+np.tile(EDGES,self.p*2).astype('<f4').tobytes()
                meta = packet(3,self.p,body,self.identity)
                for k in range(self.R.K): self.record(meta,0,k,'down','bin_dictionary')

    def start_round(self):
        self.logs = {}

    def prepare(self, k, r):
        return self.proxy[k].eval()

    def proxy_epoch(self, k, r):
        return None, self.logs[k]

    def aggregate(self, r):
        self.version += 1

    def target(self, x, y, k):
        if self.mode == 'mean':
            return self.class_mean[y], torch.ones(len(y),dtype=torch.bool,device=x.device)
        ids = torch.bucketize(x[:,self.axes].contiguous(),self.edges,right=True)
        ids = ids[:,:,0]*4+ids[:,:,1]
        rows = torch.arange(self.p,device=x.device)[None]
        valid = self.mask[rows,ids]
        tt = self.tables[k][rows,ids]
        return (tt*valid[:,:,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None], valid.any(1)

    def state(self):
        return dict(version=self.version,mass=self.mass,log_pi_effective=self.log_pi_effective)


def step(model, optimizer, scaler, loss):
    assert torch.isfinite(loss), 'BoundaryCells nonfinite loss'
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    before = scaler.get_scale()
    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
    return scaler.get_scale() >= before


def private_epoch(run, method, k, r):
    x,y = run.X[k],run.Y[k]
    teacher = method.proxy[k] if method.proxy else None
    generator = run.gen(r,k,0)
    teacher_applied = teacher_skipped = 0
    teacher_loss = 0.
    if teacher is not None:
        teacher.train()
        opt = torch.optim.AdamW(teacher.parameters(),lr=run.lr(r),weight_decay=0.)
        scaler = torch.amp.GradScaler()
        torch.manual_seed(int(np.random.SeedSequence([run.a.seed,r,k,900]).generate_state(1)[0]))
        for _ in range(20):
            idx = torch.randperm(len(y),device=run.dev,generator=generator)[:run.a.batch]
            with torch.autocast('cuda',dtype=torch.float16):
                logits = teacher(x[idx].float())[0].float()
                if method.is_cells:
                    target,valid = method.target(x[idx].float(),y[idx],k)
                    loss = -(target[valid]*F.log_softmax(logits[valid],1)).sum(1).mean() if valid.any() else F.cross_entropy(logits,y[idx])
                else:
                    loss = F.cross_entropy(logits,y[idx])
            if step(teacher,opt,scaler,loss):
                teacher_applied += 1; teacher_loss += float(loss.detach())
            else:
                teacher_skipped += 1
        assert teacher_applied>0 and teacher_skipped<=16
        finite(teacher); teacher.eval()
    model = run.priv[k].train()
    opt = torch.optim.AdamW(model.parameters(),lr=run.lr(r),weight_decay=0.)
    scaler = torch.amp.GradScaler()
    torch.manual_seed(int(np.random.SeedSequence([run.a.seed,r,k,7]).generate_state(1)[0]))
    order = torch.randperm(len(y),device=run.dev,generator=run.gen(r,k,77))
    totals = torch.zeros(2,device=run.dev)
    mass = torch.zeros(C,dtype=torch.float64,device=run.dev)
    applied = skipped = eligible_rows = 0
    for start in range(0,len(y),run.a.batch):
        idx = order[start:start+run.a.batch]
        xx,yy = x[idx].float(),y[idx]
        with torch.autocast('cuda',dtype=torch.float16):
            logits = model(xx)[0].float()
            real = F.cross_entropy(logits,yy)
            aux = logits.sum()*0
            batch_mass = F.one_hot(yy,C).double().mean(0)
            with torch.no_grad():
                tp = F.softmax(teacher(xx)[0].float(),1) if teacher is not None else None
            if method.is_cells:
                target,valid = method.target(xx,yy,k)
                if method.mode.startswith('teacher'):
                    target = tp
                if valid.any():
                    aux = .5*(-target[valid]*F.log_softmax(logits[valid],1)).sum(1).mean()
                    batch_mass += .5*target[valid].double().mean(0)
            elif method.gaussian is not None:
                labels = torch.arange(run.a.aux_batch,device=run.dev)%C
                synthetic = method.gaussian.draw(labels,generator,k)
                aux = F.cross_entropy(model(synthetic)[0].float(),labels)
                batch_mass += F.one_hot(labels,C).double().mean(0)
                valid = torch.ones(len(yy),device=run.dev,dtype=torch.bool)
            else:
                valid = torch.zeros(len(yy),device=run.dev,dtype=torch.bool)
            loss = real+aux
        if step(model,opt,scaler,loss):
            applied += 1; mass += batch_mass
            totals += torch.stack([real.detach(),aux.detach()]); eligible_rows += int(valid.sum())
        else:
            skipped += 1
        if getattr(run,'wb',None) is not None and (start//run.a.batch)%32==0:
            run.wb.log({'progress/round':r,'progress/client':k,'progress/applied':applied,
                        'progress/skipped':skipped,'progress/real_ce':float(real.detach()),
                        'progress/lr':run.lr(r)},step=r,commit=False)
    assert applied>0 and skipped<=16
    finite(model)
    method.mass[k] = mass
    # Laplace mass 1/N_epoch-step: same units as mean-per-applied-step CE target mass.
    prior = mass/applied + 1./max(len(y),1)
    method.log_pi_effective[k] = (prior/prior.sum()).log().float()
    run.log_pi[k] = method.log_pi_effective[k]
    if teacher is not None:
        method.logs[k] = dict(client=k,rows=len(y),applied=teacher_applied,skipped=teacher_skipped,
                              loss=teacher_loss/teacher_applied,positive_ce=teacher_loss/teacher_applied)
    real_ce,auxiliary = (totals/applied).tolist()
    return dict(client=k,rows=len(y),applied=applied,skipped=skipped,loss=real_ce+auxiliary,
                real_ce=real_ce,auxiliary=auxiliary,eligible_rows=eligible_rows)
