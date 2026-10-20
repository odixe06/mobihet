"""Generate FullDrift probe notebooks (per round plan) and the flat source dataset stage. Nothing is pushed here.

Public notebook keeps an empty W&B key; the private_push copy (git-ignored, 0600) carries the owner's key.

    python methods/full-drift/builds/fd-probe/scripts/gen_fd.py [--round r2] [--suffix v1]
"""
import argparse
import hashlib
import json
import netrc
import os
from pathlib import Path
import shutil
import subprocess
import nbformat

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
FULL = ROOT / 'methods/shared-baselines/builds/federated-baselines-full/source/proj'
TAG = '20261015'
RUNTIME = json.loads((ROOT / 'knowledge/runtime.json').read_text())
# (owner, seed, session, per-GPU queues, hours). Owner 15/10: one shared seed screens methods; seeds come later.
# Queue item 'arm' runs epochs 1-6; 'arm:15' runs epochs 1-15 into <arm>_s42_e15 (round 2 horizon).
# Owner 17/10: fastest code on 2xT4 without changing results -> queues may be {'workers': W, 'arms': [...]}: arms one after
# another, each spread over both GPUs by fd.py --workers W (bit-exact with the per-GPU layout, test_fd_local.py --part par).
PLANS = {'r1': [('annhannh331', 42, 'a', [['fd_dml'], ['fd_dml_sub4']], 7.0),
                ('annhannh331', 42, 'b', [['fd_dml_prox'], ['fd_gaussian', 'fd_gaussian_swappos']], 8.5),
                ('anhtruong1909', 42, 'c', [['fd_fedavg', 'fd_fedavg_sub4'], ['fd_dml_avgm']], 7.5)],
         # round 2 (owner 15/10 ~22Z); measured round 1: DML 31 min/epoch (+24 test), prox 48, FedAvg 16
         'r2': [('anhtruong1909', 42, 'd', [['fd_dml:15'], ['fd_dml_sub4:15']], 9.5),
                ('anhtruong1909', 42, 'e', [['fd_fedavg:15', 'fd_fedavg_sub4_q4', 'fd_fedavg_sub4_prox'],
                                            ['fd_fedavg_sub4:15', 'fd_fedavg_sub16_q4']], 9.0),
                ('annhannh331', 42, 'f', [['fd_dml_sub4_q4', 'fd_dml_sub4_prox'], ['fd_dml_sub16_q4']], 10.5)],
         # round 3 (owner 16/10: client private gp; R1/R2-safe arms); DML epoch 1-6 measured 217 min + setup 5 min
         'r3': [('nguyenminhtran060113', 42, 'g', [['fd_dml_m4b'], ['fd_dml_swappos']], 4.5),
                ('nguyenminhtran060113', 42, 'h', [['fd_dml_ntd'], ['fd_dml_q4']], 4.5)],
         # round 4 (owner 16/10: short-quota accounts); odixe0502 4.0h live, Gaussian epoch 1-6 measured 150 min
         'r4': [('odixe0502', 42, 'i', [['fd_dml_hkd'], ['fd_gaussian_m4b']], 3.95)],
         # round 5 (owner 16/10 ~08:20Z): m4b horizon + combinations; nguyenminhtran060113 9.27h live; DML 32.5 min/epoch (r2)
         'r5': [('nguyenminhtran060113', 42, 'j', [['fd_dml_m4b:15'], ['fd_dml_m4b_q4', 'fd_dml_m4b_sub4_q4']], 8.95)],
         # round 6 (owner 16/10 ~16:45Z): seed 43 confirmation; anhtruong1909 9.36h live; DML epoch 1-6 measured 205-217 min
         'r6': [('anhtruong1909', 43, 'k', [['fd_dml'], ['fd_dml_m4b']], 4.2),
                ('anhtruong1909', 43, 'l', [['fd_dml_sub4_q4'], ['fd_dml_m4b_sub4_q4']], 4.2)],
         # round 7 (owner 16/10 ~17:25Z): seed 43 for swappos and Gaussian m4b; annhannh331 5.29h live
         'r7': [('annhannh331', 43, 'm', [['fd_dml_swappos'], ['fd_gaussian_m4b']], 4.2)],
         # round 8 (owner 17/10, algorithm-improvement-plan): accanniez 10.47h / annhtnt 8.96h / takids1909 8.70h live before the
         # 00:00Z reset (owner-supplied); DML ep6 205-217 min, Gaussian ep6 150 min (fd_gaussian s43 needs ~4.85h), cons = 2 views
         'r8': [('accanniez', 42, 'n', [['fd_dml_m4b_kdcap'], ['fd_dml_m4b_kdcap_frz']], 5.0),
                ('accanniez', 42, 'o', [['fd_dml_m4b_kdcap_wu'], ['fd_dml_m4b_dyn']], 5.0),
                ('annhtnt', 42, 'p', [['fd_dml_m4b_own'], ['fd_gaussian_m4b_own']], 4.6),
                ('annhtnt', 42, 'q', [['fd_dml_swappos_cons'], ['fd_dml_swappos_cons0']], 8.5),
                ('takids1909', 43, 'r', [['fd_gaussian'], ['fd_gaussian_swappos']], 5.4)],
         # round 9 (same plan): SCAFFOLD-proxy at three SGD LRs (paired with the c=0 control) + Hybrid private-only + seed-43
         # m4b horizon; small accounts pushed only when hours-to-reset x sessions < GPU left (owner: no mid-run quota stop)
         'r9': [('takids1909', 42, 's', [['fd_dml_m4b_sgd3'], ['fd_dml_m4b_scaf3']], 5.0),
                ('khanhmay0304', 42, 'u', [['fd_dml_m4b_sgd1'], ['fd_dml_m4b_scaf1']], 5.0),
                ('minhtriethihi', 42, 'v', [['fd_dml_m4b_sgd10'], ['fd_dml_m4b_scaf10']], 5.0),
                ('trietbackup', 42, 't', [['fd_dml_m4b_gpool'], ['fd_dml_m4b_gown']], 5.4),
                # fd_dml_m4b rerun: KD weight/gradient logs of the uncapped baseline (plan §3.2) + cross-source replication
                ('trietbackup', 42, 'x', [['fd_dml_m4b_replay'], ['fd_dml_m4b']], 5.4),
                ('annhannh331', 43, 'w', [['fd_dml_m4b:15'], ['fd_dml:15']], 10.0)],
         # round 10 (owner 16/10 ~22:10Z: push every account that still fits before the 00:00Z reset); bytes = r9.
         # 'y' replaces trietbackup 'x' (never pushed); seed 43 replicates paired with the audited round-6/7 s43 controls
         'r10': [('anhtruong1909', 42, 'y', [['fd_dml_m4b_replay'], ['fd_dml_m4b']], 5.4),
                 ('nguyenminhtran060113', 43, 'z', [['fd_dml_m4b_own'], ['fd_gaussian_m4b_own']], 5.0),
                 ('catbaochau', 43, 'ka', [['fd_dml_m4b_kdcap'], ['fd_dml_m4b_kdcap_wu']], 5.0),
                 ('dinhan25', 43, 'kb', [['fd_dml_m4b_kdcap_frz'], ['fd_dml_m4b_q4']], 5.0),
                 ('minhtran0601', 43, 'ga', [['fd_dml_m4b_gpool'], ['fd_dml_m4b_gown']], 5.4),
                 ('odixe0502', 43, 'gb', [['fd_dml_m4b_replay'], ['fd_dml_m4b_dyn']], 5.4)],
         # round 11 (owner 17/10 ~05:10Z: improve headline m4b+gpool before 50 rounds). New week: 30h each (all accounts
         # queried, owner ~05:30Z); gpool ep6 measured 241 min, ep15 ~10 h
         'r11': [('accanniez', 42, 'la', [['fd_dml_m4b_gpool_kd2'], ['fd_dml_m4b_gpool_kdpx']], 5.4),
                 ('takids1909', 42, 'lc', [['fd_dml_m4b_gpool:15'], ['fd_dml_m4b_gpool_gw2']], 11.0)],
         # round 12 (owner 17/10 ~14Z: limited communication budget): headline over the real int4 QPacket. Source adds the
         # sync-free loop and --workers (bit-exact: test_fd_local --part r12/par); one arm over both GPUs, one worker each
         'r12': [('odixe0502', 42, 'qa', {'workers': 2, 'arms': ['fd_dml_m4b_gpool_q4']}, 4.0)],
         # round 13 (owner 17/10 ~16Z: run A, B, C + the owner's proposals, each from scratch ep1-6; 30h live on every account)
         'r13': [(owner, 42, session, {'workers': 2, 'arms': [f'fd_dml_m4b_gpool_{arm}']}, 5.0) for owner, session, arm in
                 [('annhannh331', 'ma', 'fill'), ('anhtruong1909', 'mb', 'rej'), ('catbaochau', 'mc', 'p1'),
                  ('dinhan25', 'md', 'p1self'), ('khanhmay0304', 'me', 'p1cm'), ('minhtran0601', 'mf', 'q2')]],
         # round 14 (design §Vòng 14): repeat the round-13 PASS arms -- horizon ep15 + seed 43; source bytes = r13
         'r14': [('trietbackup', 42, 'na', {'workers': 2, 'arms': ['fd_dml_m4b_gpool_rej:15']}, 7.0),
                 ('minhtriethihi', 43, 'nb', {'workers': 2, 'arms': ['fd_dml_m4b_gpool_rej']}, 5.0),
                 ('accanniez', 42, 'nc', {'workers': 2, 'arms': ['fd_dml_m4b_gpool_q2:15']}, 7.0)],
         # round 15 (design §Vòng 15): frozen aggregated snapshot as the private's teacher (frz), Gaussian KL to it (gkd); ep1-15
         'r15': [('thuyanh316', 42, 'oa', {'workers': 2, 'arms': ['fd_dml_m4b_frz:15']}, 7.0),
                 ('annhtnt', 42, 'ob', {'workers': 2, 'arms': ['fd_dml_m4b_gpool_frz:15']}, 7.0),
                 ('khanhmay0304', 42, 'oc', {'workers': 2, 'arms': ['fd_dml_m4b_gpool_frz_gkd:15']}, 7.0)],
         # round 16 (design §Vòng 16): seed-43 horizon of the round-15 PASS arms + seed-43 gpool control; gkd over the q2 wire
         'r16': [('anhtruong1909', 43, 'pa', {'workers': 2, 'arms': ['fd_dml_m4b_gpool_frz_gkd:15']}, 7.0),
                 ('dinhan25', 43, 'pb', {'workers': 2, 'arms': ['fd_dml_m4b_gpool:15']}, 7.0),
                 ('catbaochau', 42, 'pc', {'workers': 2, 'arms': ['fd_dml_m4b_gpool_q2_frz_gkd:15']}, 7.0),
                 ('minhtran0601', 43, 'pd', {'workers': 2, 'arms': ['fd_dml_m4b_frz:15']}, 7.0)],
         # round 17 (owner 18/10 ~06Z, design §Vòng 17): small improvements of the round-15 headline (ep1-15 from scratch) and the
         # headline itself continued ep16-50 from the audited round-15 epoch-15 checkpoint (exact resume, --parent); gkd ~18 min/epoch
         'r17': [('odixe0502', 42, 'ra', {'workers': 2, 'arms': ['fd_dml_m4b_gpool_frz_gkd_la:15']}, 7.0),
                 ('annhannh331', 42, 'rb', {'workers': 2, 'arms': ['fd_dml_m4b_gpool_frz_gkd_kdpx:15']}, 7.0),
                 ('minhtriethihi', 42, 'rc', {'workers': 2, 'arms': ['fd_dml_m4b_gpool_frz_gkd_la_kdpx:15']}, 7.0),
                 ('trietbackup', 42, 'rd', {'workers': 2, 'arms': ['fd_dml_m4b_gpool_frz_gkd:50'],
                                            'parent': dict(dataset='trietbackup/mobihet-fd-parent-gkd-s42-e15', session='fd-s42oc-20261015-r15-v1',
                                                           run='fd_dml_m4b_gpool_frz_gkd_s42_e15', round=15)}, 11.75)],
         # round 18 (owner 18/10 ~07Z: try every improvement with evidence; design §Vòng 18): one change each on the gkd headline,
         # ep1-15 from scratch; g512 has 2x synthetic rows per step (more compute)
         'r18': [(owner, 42, session, {'workers': 2, 'arms': [f'{arm}:15']}, hours) for owner, session, arm, hours in
                 [('accanniez', 'sa', 'fd_dml_m4b_gpool_frz_gkd_cop', 7.0), ('takids1909', 'sb', 'fd_dml_m4b_gpool_frz_gkd_pce', 7.0),
                  ('khanhmay0304', 'sc', 'fd_dml_m4b_gpool_frz_gkd_t2', 7.0),
                  ('nguyenminhtran060113', 'sd', 'fd_dml_m4b_gpool_frz_gkd_g512', 8.5),
                  ('annhtnt', 'se', 'fd_dml_m4b_gpool_frz_gkd_gk2', 7.0), ('thuyanh316', 'sf', 'fd_dml_m4b_gpool_q4_frz_gkd', 7.0)]],
         # round 19 (owner 18/10 ~09:15Z: --fast only for new builds, compared only with --fast controls; design §Vòng 19): gkd controls
         # re-run in the new code (s42, s43), the round-17 best la_kdpx (s42 replicate + s43), la_kdpx over the int4 wire (R1)
         'r19': [(owner, seed, session, {'workers': 2, 'arms': [f'{arm}:15'], 'fast': True}, 7.0) for owner, seed, session, arm in
                 [('minhtran0601', 42, 'ta', 'fd_dml_m4b_gpool_frz_gkd'), ('dinhan25', 42, 'tb', 'fd_dml_m4b_gpool_frz_gkd_la_kdpx'),
                  ('annhannh331', 42, 'tc', 'fd_dml_m4b_gpool_q4_frz_gkd_la_kdpx'), ('anhtruong1909', 43, 'td', 'fd_dml_m4b_gpool_frz_gkd'),
                  ('odixe0502', 43, 'te', 'fd_dml_m4b_gpool_frz_gkd_la_kdpx')]]
                + # round 19 continuations ep16->50 (--fast, same tag/source; each from the audited ep15 of its own --fast run): la_kdpx and
                  # its gkd control, seeds 42/43 (design §Vòng 19); generated per batch with --sessions once the parent is audited
                  [(owner, seed, session, {'workers': 2, 'arms': [f'{arm}:50'], 'fast': True,
                                           'parent': dict(dataset=f'{owner}/mobihet-fd-parent-{slug}-fast-s{seed}-e15', session=parent,
                                                          run=f'{arm}_s{seed}_e15', round=15)}, 11.75)
                   for owner, seed, session, arm, slug, parent in
                   [('dinhan25', 42, 'ua', 'fd_dml_m4b_gpool_frz_gkd_la_kdpx', 'lakdpx', 'fd-s42tb-20261015-r19-v1'),
                    ('anhtruong1909', 43, 'ub', 'fd_dml_m4b_gpool_frz_gkd', 'gkd', 'fd-s43td-20261015-r19-v1'),
                    ('minhtran0601', 42, 'uc', 'fd_dml_m4b_gpool_frz_gkd', 'gkd', 'fd-s42ta-20261015-r19-v1'),
                    ('odixe0502', 43, 'ud', 'fd_dml_m4b_gpool_frz_gkd_la_kdpx', 'lakdpx', 'fd-s43te-20261015-r19-v1')]],
         # round 20 (owner 19/10: exhaust the la/kdpx direction at probe scale before any final 50-epoch run; design §Vòng 20): one
         # change each on la_kdpx, s42 ep1-15 --fast, compared with tb (la_kdpx --fast s42)
         'r20': [(owner, 42, session, {'workers': 2, 'arms': [f'{arm}:15'], 'fast': True}, 7.0) for owner, session, arm in
                 [('minhtriethihi', 'va', 'fd_dml_m4b_gpool_frz_gkd_la15_kdpx'), ('catbaochau', 'vb', 'fd_dml_m4b_gpool_frz_gkd_la05_kdpx'),
                  ('thuyanh316', 'vc', 'fd_dml_m4b_gpool_frz_gkd_lau_kdpx'), ('annhtnt', 'vd', 'fd_dml_m4b_gpool_frz_gkd_la_lap_kdpx'),
                  ('khanhmay0304', 've', 'fd_dml_m4b_gpool_frz_gkd_la_kdpx_kd2')]]
                + # round 21 (owner 19/10: smallest communication budget without losing performance; same source tag, parallel)
                  [(owner, 42, session, {'workers': 2, 'arms': [f'{arm}:15'], 'fast': True}, 7.0) for owner, session, arm in
                   [('dinhan25', 'wa', 'fd_dml_m4b_gpool_q4_frz_gkd_la_kdpx_zl'), ('minhtran0601', 'wb', 'fd_dml_m4b_gpool_q4_frz_gkd_la_kdpx_ef_zl'),
                    ('anhtruong1909', 'wc', 'fd_dml_m4b_gpool_q2_frz_gkd_la_kdpx_ef_zl')]],
         # round 22 (design §Vòng 22: q2 collapsed through its downlink; q4_zl chosen): 3-bit both ways, 2-bit uplink only with client
         # error feedback (int4 or 3-bit downlink), and the q4_zl replicate on seed 43 (vs te)
         'r22': [(owner, seed, session, {'workers': 2, 'arms': [f'{arm}:15'], 'fast': True}, 7.0) for owner, seed, session, arm in
                 [('annhannh331', 42, 'xa', 'fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl'),
                  ('odixe0502', 42, 'xb', 'fd_dml_m4b_gpool_q4_u2_frz_gkd_la_kdpx_ef_zl'),
                  ('minhtriethihi', 42, 'xc', 'fd_dml_m4b_gpool_q3_u2_frz_gkd_la_kdpx_ef_zl'),
                  ('catbaochau', 43, 'xd', 'fd_dml_m4b_gpool_q4_frz_gkd_la_kdpx_zl')]],
         # round 23 (design §Vòng 23: q3 PASS, 2-bit failed through the row-max step): q3 replicate on seed 43 (vs te); the 2-bit wire
         # with the RMS (Lloyd-Max) step both ways (vs tb)
         'r23': [(owner, seed, session, {'workers': 2, 'arms': [f'{arm}:15'], 'fast': True}, 7.0) for owner, seed, session, arm in
                 [('minhtran0601', 43, 'ya', 'fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl'),
                  ('dinhan25', 42, 'yb', 'fd_dml_m4b_gpool_q2_rs_frz_gkd_la_kdpx_zl')]]}

WANDB = r'''import time
SESSION_T0=time.monotonic()

import os, subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','--quiet','wandb==0.29.0'],check=True)
import wandb
_INLINE_WANDB_KEY = ""
try:
    from kaggle_secrets import UserSecretsClient
    _wandb_key=_INLINE_WANDB_KEY or UserSecretsClient().get_secret('wandb_key')
except Exception:
    _wandb_key=_INLINE_WANDB_KEY
assert _wandb_key, 'W&B key required before training'
assert wandb.login(key=_wandb_key,anonymous='never',relogin=True,verify=True)
os.environ['WANDB_API_KEY']=_wandb_key
del _wandb_key
assert wandb.Api(timeout=30).default_entity, 'W&B online authentication failed'
print('W&B authenticated online',flush=True)
'''

SETUP = r'''
import hashlib,json,shutil
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import torch
assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) for i in range(2))
assert torch.__version__=='2.10.0+cu128'
hits=[p.parent for p in Path('/kaggle/input').rglob('input_manifest.json') if json.loads(p.read_text())==INPUTS]
assert len(hits)==1,('missing/ambiguous inputs',hits)
SRC=hits[0];WORK=Path('/kaggle/temp/fd');SOURCE=Path('/kaggle/working/source');OUT=Path('/kaggle/working/runs')
PROJ=WORK/'papers/full-drift/proj'
for folder in (WORK/'mh',WORK/'knowledge',PROJ,SOURCE,OUT):folder.mkdir(parents=True,exist_ok=True)
for name,digest in INPUTS.items():
    assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==digest,name
    target=WORK/'mh'/name if name in MH else PROJ/name if name.endswith('.py') else WORK/'knowledge'/name
    shutil.copyfile(SRC/name,target);shutil.copyfile(SRC/name,SOURCE/name)
shutil.copyfile(SRC/'input_manifest.json',SOURCE/'input_manifest.json')
sys.path.insert(0,str(WORK));sys.path.insert(0,str(PROJ))
t=time.monotonic()
from fd_prepare import prepare_fd
data=prepare_fd(WORK,final=FINAL)   # FINAL (owner 19/10): every train80% row trains, test20% every round
(SOURCE/'data_manifest.json').write_text(json.dumps(data,indent=2))
print('verified source; prepared','final' if FINAL else 'full holdout','fixture',data['source_rows'],{k:sum(v) for k,v in data['counts'].items()},
      'test',data['test_rows'],'sec',round(time.monotonic()-t),flush=True)
'''

RUN = r'''
pending={gpu:list(q) for gpu,q in enumerate(QUEUES)};active={};summary={}
while active or any(pending.values()):
    for gpu in (0,1):
        if gpu in active or not pending[gpu]:continue
        remaining=SESSION_HOURS-(time.monotonic()-SESSION_T0)/3600
        if remaining<1.0:pending[gpu]=[];continue
        item=pending[gpu].pop(0);arm,_,ep=item.partition(':');ep=int(ep or 6)
        folder=OUT/(f'{arm}_s{SEED}'+('' if ep==6 else f'_e{ep}'));folder.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=f'{PROJ}:{WORK}',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        command=[sys.executable,str(PROJ/'fd.py'),'--arm',arm,'--seed',str(SEED),'--backend','compiled','--wandb_online',
                 '--wandb_project','mobihet-fulldrift','--max_hours',str(round(remaining-.3,3)),'--out',str(OUT),'--stop_after',str(ep)]
        log=open(folder/'log.txt','a');proc=subprocess.Popen(command,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
        active[gpu]=(item,folder,proc,log,time.monotonic())
        print('start',item,SEED,'gpu',gpu,'remaining_h',round(remaining,2),flush=True)
    for gpu,(arm,folder,proc,log,t) in list(active.items()):
        if proc.poll() is None and (time.monotonic()-SESSION_T0)/3600>=SESSION_HOURS-.05:proc.terminate()
        if proc.poll() is None:continue
        log.close();del active[gpu]
        rp=folder/'result.json'
        result=json.loads(rp.read_text()) if rp.exists() else dict(history=[])
        summary[arm]=dict(rc=proc.returncode,epochs=len(result['history']),complete=result.get('complete',False),
                          minutes=round((time.monotonic()-t)/60,1))
        (OUT/'fd_summary.json').write_text(json.dumps(summary,indent=2))
        print('finish',arm,summary[arm],flush=True)
        if proc.returncode:print(open(folder/'log.txt').read()[-3000:],flush=True)
    time.sleep(5)
(OUT/'fd_summary.json').write_text(json.dumps(summary,indent=2))
print('FullDrift session; gates only after collector audit',json.dumps(summary),flush=True)
'''


RUN_POOL = r'''
summary={};PARENT_DIR=None
if PARENT:   # committed epoch of an audited run (flat dataset files, SHA-checked here and by fd.py): exact resume
    PARENT_DIR=Path('/kaggle/temp/parent')/PARENT['run']
    for rel,digest in PARENT['files'].items():
        flat=('resume_' if rel.startswith('resume/') else '')+Path(rel).name   # upload_parent.py flat names
        hits=[p for p in Path('/kaggle/input').rglob(flat) if hashlib.sha256(p.read_bytes()).hexdigest()==digest]
        assert hits,('parent file',rel);(PARENT_DIR/rel).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(hits[0],PARENT_DIR/rel)
    print('parent',PARENT['run'],'round',PARENT['round'],flush=True)
for item in ARMS:
    remaining=SESSION_HOURS-(time.monotonic()-SESSION_T0)/3600
    if remaining<1.0:break
    arm,_,ep=item.partition(':');ep=int(ep or 6)
    folder=OUT/(f'{arm}_s{SEED}'+('' if ep==6 else f'_e{ep}'));folder.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=f'{PROJ}:{WORK}',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('CUDA_VISIBLE_DEVICES',None)
    command=[sys.executable,str(PROJ/'fd.py'),'--arm',arm,'--seed',str(SEED),'--backend','compiled','--wandb_online',
             '--wandb_project','mobihet-fulldrift','--max_hours',str(round(remaining-.3,3)),'--out',str(OUT),'--stop_after',str(ep),
             '--workers',str(WORKERS)]+(['--parent',str(PARENT_DIR),'--parent_round',str(PARENT['round'])] if PARENT else [])+(['--fast'] if FAST else [])+(['--final'] if FINAL else [])
    t=time.monotonic();log=open(folder/'log.txt','a')
    proc=subprocess.Popen(command,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
    print('start',item,SEED,'workers',WORKERS,'remaining_h',round(remaining,2),flush=True)
    while proc.poll() is None:
        if (time.monotonic()-SESSION_T0)/3600>=SESSION_HOURS-.05:proc.terminate()
        time.sleep(5)
    log.close();rp=folder/'result.json'
    result=json.loads(rp.read_text()) if rp.exists() else dict(history=[])
    summary[item]=dict(rc=proc.returncode,epochs=len(result['history']),complete=result.get('complete',False),
                       minutes=round((time.monotonic()-t)/60,1))
    (OUT/'fd_summary.json').write_text(json.dumps(summary,indent=2))
    print('finish',item,summary[item],flush=True)
    if proc.returncode:
        print(open(folder/'log.txt').read()[-3000:],flush=True)
        for f in sorted(folder.glob('worker*.log')):print(f.name,f.read_text()[-2000:],flush=True)
(OUT/'fd_summary.json').write_text(json.dumps(summary,indent=2))
print('FullDrift session; gates only after collector audit',json.dumps(summary),flush=True)
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parent_files(parent):
    """Round 17 --parent: the audited session's config, epoch marker and weights for the notebook (relative path -> SHA)."""
    if not parent: return None
    run = BASE / 'artifacts/kaggle' / parent['session'] / 'runs' / parent['run']; r = parent['round']
    assert (BASE / 'artifacts/kaggle' / parent['session'] / 'AUDITED.json').exists()
    rels = ['config.json', f'complete/round_{r:03d}.json', f'weights/round_{r:03d}.pt']
    if (run / f'resume/round_{r:03d}.pt').exists(): rels.append(f'resume/round_{r:03d}.pt')   # round 21: codec state (q4/q2, ef)
    files = {rel: sha(run / rel) for rel in rels}
    assert json.loads((run / rels[1]).read_text())['weights_sha'] == files[rels[2]]
    return dict(run=parent['run'], round=r, files=files)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--round', choices=sorted(PLANS), default='r2')
    parser.add_argument('--suffix', default='v1')
    parser.add_argument('--sessions', help='comma list: generate only these sessions of the plan (later batches of one source tag)')
    args = parser.parse_args()
    tag = TAG if args.round == 'r1' else f'{TAG}-{args.round}'   # round 1 stage/dataset kept as pushed
    stage = BASE / 'artifacts/input' / f'fd-src-{tag}'
    mh = sorted(p.name for p in (ROOT / '.compat/mh').glob('*.py')); assert len(mh) == 40
    files = {**{n: ROOT / '.compat/mh' / n for n in mh}, **{n: FULL / n for n in ('full_run.py', 'full_method.py', 'full_driver.py')},
             'fd.py': BASE / 'source/proj/fd.py', 'fd_prepare.py': BASE / 'source/proj/fd_prepare.py',
             'meta.json': ROOT / 'knowledge/meta.json', 'scaler.json': ROOT / 'knowledge/scaler.json'}
    if stage.exists():   # reuse the uploaded dataset; its manifest must still match the sources
        inputs = json.loads((stage / 'input_manifest.json').read_text())
        assert inputs == {n: sha(p) for n, p in files.items()}, 'source changed after staging: new tag'
    else:
        stage.mkdir(parents=True)
        for name, path in files.items(): shutil.copyfile(path, stage / name)
        inputs = {n: sha(stage / n) for n in sorted(files)}
        (stage / 'input_manifest.json').write_text(json.dumps(inputs, indent=1) + '\n')
    secret = netrc.netrc().authenticators('api.wandb.ai')[2]
    for owner, seed, session, queues, hours in PLANS[args.round]:
        if args.sessions and session not in args.sessions.split(','): continue
        dataset = f'{owner}/mobihet-fulldrift-src-{tag}'   # same bytes uploaded under each running account
        rid = f'fd-s{seed}{session}-{tag}-{args.suffix}'
        public, private = BASE / 'notebooks' / rid, BASE / 'private_push' / rid
        assert not public.exists() and not private.exists()
        public.mkdir(parents=True); private.mkdir(parents=True, mode=0o700)
        head = f'MH={mh!r}\nINPUTS={inputs!r}\nFINAL={isinstance(queues, dict) and queues.get("final", False)!r}\n'
        if isinstance(queues, dict):
            run = (f'SEED={seed}\nSESSION_HOURS={hours}\nWORKERS={queues["workers"]}\nARMS={queues["arms"]!r}\nFAST={queues.get("fast", False)!r}\n'
                   f'PARENT={parent_files(queues.get("parent"))!r}\n' + RUN_POOL)
        else:
            run = f'SEED={seed}\nSESSION_HOURS={hours}\nQUEUES={queues!r}\n' + RUN
        cells = [nbformat.v4.new_markdown_cell(
                     f'# FullDrift probe seed{seed} session {session}\n'
                     'Full 100% train80% minus internal holdout (temporal2% + receiver-group5%), epochs 1-6 (or 1-15) of cosine T50. '
                     'Decisions on group holdout; official test once after the last epoch, descriptive. '
                     'Contract: docs/methods/full-drift/design.md'),
                 nbformat.v4.new_code_cell(WANDB), nbformat.v4.new_code_cell(head + SETUP), nbformat.v4.new_code_cell(run)]
        nb = nbformat.v4.new_notebook(cells=cells)
        nb.metadata = dict(kaggle=dict(accelerator='nvidiaTeslaT4', isGpuEnabled=True, isInternetEnabled=True,
                                       language='python', sourceType='notebook'),
                           kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                           language_info=dict(name='python', version='3.12'))
        for c in nb.cells:
            if c.cell_type == 'code': compile(c.source, rid, 'exec')
        nbformat.validate(nb)
        parents = [queues['parent']['dataset']] if isinstance(queues, dict) and queues.get('parent') else []
        meta = dict(id=f'{owner}/mobihet-{rid}', title=f'mobihet-{rid}', code_file=f'{rid}.ipynb', language='python',
                    kernel_type='notebook', is_private=True, enable_gpu=True, enable_internet=True,
                    machine_shape=RUNTIME['machine_shape'], docker_image=RUNTIME['docker_image'],
                    dataset_sources=[dataset, 'trietbackup/veremi-fl-20client', 'trietbackup/veremi-nextgen2026-test', *parents],
                    kernel_sources=[], competition_sources=[])
        assert len(meta['title']) <= 50
        nbformat.write(nb, public / meta['code_file'])
        (public / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
        (public / 'source_manifest.json').write_text(json.dumps(dict(inputs=sha(stage / 'input_manifest.json'), fd=inputs['fd.py'],
                                                                     queues=queues, session_hours=hours, seed=seed), indent=2) + '\n')
        (public / 'checkpoint-dataset.json').write_text(json.dumps(dict(
            dataset=dataset, files={n: dict(size=(stage / n).stat().st_size) for n in [*inputs, 'input_manifest.json']},
            uploaded=False), indent=2) + '\n')
        subprocess.run(['git', 'check-ignore', '-q', str((private / meta['code_file']).resolve())], cwd=ROOT, check=True)
        pushed = nbformat.reads(nbformat.writes(nb), as_version=4)
        pushed.cells[1].source = pushed.cells[1].source.replace('_INLINE_WANDB_KEY = ""', '_INLINE_WANDB_KEY = ' + json.dumps(secret))
        assert pushed.cells[1].source != nb.cells[1].source
        for name, value in [(meta['code_file'], nbformat.writes(pushed)), ('kernel-metadata.json', json.dumps(meta, indent=2))]:
            fd = os.open(private / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w') as f: f.write(value)
        print(meta['id'], 'prepared', queues, hours, 'h')


if __name__ == '__main__':
    main()
