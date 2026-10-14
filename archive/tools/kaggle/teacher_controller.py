"""Notebook source: measure two committed rounds, cost-gate, then resume same schedule."""
COST_RUN = r'''
T0=time.time(); OUT.mkdir(exist_ok=True)
baseline=METHODS[0]; pending=list(METHODS[1:]); ready=[]; active={}; summary={}; baseline_cost=None
def identity(arm,reason):
    folder=OUT/f'{arm}_s{SEED}';folder.mkdir(exist_ok=True)
    cfg=dict(arm=arm,seed=SEED,private='dagsnet',rounds=20,batch=512,lr=.001,lr_min=.00001,
             clip=1.,eval_every=5,aux_batch=256,aux_weight=1.,temperature=2.,tau=.05,weight_cap=10.,smoke=False)
    data=dict(config=cfg,source={name:manifest[name] for name in CODE},
              fixture=hashlib.sha256((SRC/'manifest.json').read_bytes()).hexdigest())
    data['fingerprint']=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
    (folder/'config.json').write_text(json.dumps(data,indent=2))
    (folder/'log.txt').write_text(reason+'\n')
def start(arm,gpu,phase):
    folder=OUT/f'{arm}_s{SEED}';folder.mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    command=[sys.executable,'-m',MODULE,'--arm',arm,'--seed',str(SEED),'--out',str(OUT),
             '--wandb_online','--max_hours','2.0']
    if phase=='probe':command+=['--stop_after','2']
    else:command+=['--resume']
    log=open(folder/'log.txt','a' if phase=='continue' else 'w')
    process=subprocess.Popen(command,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
    active[gpu]=(arm,phase,process,log,time.time())
    print('start',arm,'gpu',gpu,'phase',phase,flush=True)
start(baseline,0,'probe')
while active or pending or ready:
    for gpu in (0,1):
        if gpu in active or time.time()-T0>=4.0*3600:continue
        if ready:
            arm,target_gpu=ready[0]
            if target_gpu!=gpu:continue
            ready.pop(0);start(arm,gpu,'continue')
        elif baseline_cost is not None and pending:
            start(pending.pop(0),gpu,'probe')
    for gpu,(arm,phase,process,log,started) in list(active.items()):
        if process.poll() is None and time.time()-T0>=4.7*3600:process.terminate()
        if process.poll() is None and time.time()-T0>=4.75*3600:process.kill()
        if process.poll() is None:continue
        log.close();del active[gpu]
        folder=OUT/f'{arm}_s{SEED}';path=folder/'result.json'
        result=json.loads(path.read_text()) if path.exists() else {}
        if phase=='probe':
            rows=result.get('history',[])
            if process.returncode or len(rows)!=2:
                summary[arm]=dict(rc=process.returncode or -996,complete=False,cost_pass=None,
                                  reason='cost preflight failed before two complete rounds')
                if arm==baseline:
                    for skipped in pending:identity(skipped,'COST BASELINE MISSING')
                    for skipped in pending:summary[skipped]=dict(rc=-998,complete=False,cost_pass=None)
                    pending.clear()
                continue
            seconds=[row['algorithm_seconds'] for row in rows]
            measured=sum(seconds)/2
            if arm==baseline:baseline_cost=measured
            ratio=measured/baseline_cost;passed=ratio<=2.
            gate=dict(measured_algorithm_seconds=seconds,mean_seconds=measured,
                      baseline_mean_seconds=baseline_cost,ratio=ratio,pass_cost=passed,
                      physical_gpu=gpu,backend='same pinned 2T4; one arm per GPU')
            (folder/'cost_gate.json').write_text(json.dumps(gate,indent=2))
            print('COST',arm,ratio,passed,flush=True)
            if passed:ready.append((arm,gpu))
            else:
                with (folder/'log.txt').open('a') as f:f.write('COST PREFLIGHT REJECT: ratio exceeds 2x fresh dense CE\n')
                summary[arm]=dict(rc=-997,complete=False,cost_pass=False,cost_stopped=True,cost=gate)
        else:
            gate=json.loads((folder/'cost_gate.json').read_text())
            summary[arm]=dict(rc=process.returncode,complete=result.get('complete',False),cost_pass=gate['pass_cost'],cost=gate)
    if not active and time.time()-T0>=4.0*3600:break
    if not active and baseline_cost is None:break
    time.sleep(2)
for arm in METHODS:
    if arm in summary:continue
    folder=OUT/f'{arm}_s{SEED}'
    if not (folder/'config.json').exists():identity(arm,'NOT STARTED: parent budget')
    summary[arm]=dict(rc=-998,complete=False,cost_pass=None,reason='parent budget before arm/continuation')
(OUT/f'summary_s{SEED}.json').write_text(json.dumps(summary,indent=2))
print(summary,flush=True)
shutil.rmtree(WORK)
assert all((v['complete'] and v['rc']==0) or v.get('cost_stopped') for v in summary.values()),'incomplete training outside declared cost stop'
'''
