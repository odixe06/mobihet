"""Small isolated fixtures: detached evaluation parity, K=20/50/100, codec resume and tamper rejection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]


def same(a, b):
    if isinstance(a, torch.Tensor): return torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list): return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def main():
    p = argparse.ArgumentParser(); p.add_argument('--work', type=Path, required=True)
    p.add_argument('--clients', type=int, choices=(20, 50, 100), default=20)
    p.add_argument('--backend', choices=('eager', 'compiled'), default='eager')
    a = p.parse_args(); work = a.work.resolve(); work.mkdir(parents=True, exist_ok=True)
    proj = work / 'papers/full-drift/proj'; proj.mkdir(parents=True, exist_ok=True)
    for src in (ROOT / '.compat/mh').glob('*.py'):
        dst = work / 'mh' / src.name; dst.parent.mkdir(exist_ok=True); shutil.copyfile(src, dst)
    for name in ('meta.json', 'scaler.json'):
        dst = work / 'knowledge' / name; dst.parent.mkdir(exist_ok=True); shutil.copyfile(ROOT / 'knowledge' / name, dst)
    for src in (ROOT / 'methods/shared-baselines/builds/federated-baselines-full/source/proj').glob('*.py'):
        shutil.copyfile(src, proj / src.name)
    for src in (BASE / 'source/proj').glob('*.py'): shutil.copyfile(src, proj / src.name)
    fixture = work / 'papers/pilot/fixture'; fixture.mkdir(parents=True, exist_ok=True)
    origin = ROOT / '.compat/papers/pilot/fixture'
    manifest = json.loads((origin / 'manifest.json').read_text())
    counts, files = {}, {}
    for k in range(a.clients):
        n = len(np.load(origin / f'train_{k % 20}_y.npy', mmap_mode='r'))
        indices = np.random.default_rng(k % 20).choice(n, 1024, replace=False)
        for name in ('X', 'y', 'g'):
            v = np.load(origin / f'train_{k % 20}_{name}.npy', mmap_mode='r')[indices].copy()
            if name == 'y': counts[f'train_{k}'] = np.bincount(v, minlength=16).tolist()
            dst = fixture / f'train_{k}_{name}.npy'; np.save(dst, v)
            files[dst.name] = hashlib.sha256(dst.read_bytes()).hexdigest()
    for hold in ('group', 'old'):
        for name in ('X', 'y'):
            dst = fixture / f'{hold}_{name}.npy'; np.save(dst, np.load(origin / dst.name, mmap_mode='r')[:256])
            files[dst.name] = hashlib.sha256(dst.read_bytes()).hexdigest()
    manifest.update(files=files, counts=counts, clients=list(range(a.clients)))
    (fixture / 'manifest.json').write_text(json.dumps(manifest))
    env = dict(os.environ, PYTHONPATH=f'{proj}:{work}', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', TORCHINDUCTOR_COMPILE_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    arm = 'fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl'
    def command(cmd, name, ok=True):
        with (work / f'{name}.log').open('w') as log:
            r = subprocess.run([sys.executable, *map(str, cmd)], env=env, stdout=log, stderr=subprocess.STDOUT)
        assert (r.returncode == 0) == ok, (name, (work / f'{name}.log').read_text()[-4000:])
    def train(name, extra=()):
        out = work / name
        command([proj / 'fd.py', '--arm', arm, '--seed', 42, '--smoke', '--final', '--backend', a.backend,
                 '--clients', a.clients, '--batch', 128, '--out', out, *extra], name)
        return out / f'{arm}_s42'
    inline = train('inline')
    if a.clients == 20:
        current = (proj / 'fd.py').read_bytes()
        try:
            shutil.copyfile(BASE / 'artifacts/input/fd-src-20261015-r23/fd.py', proj / 'fd.py')
            command([proj / 'fd.py', '--arm', arm, '--seed', 42, '--smoke', '--final', '--backend', a.backend,
                     '--batch', 128, '--out', work / 'before'], 'before')
        finally:
            (proj / 'fd.py').write_bytes(current)
        old = work / 'before' / f'{arm}_s42'
        for r in (1, 2):
            x, y = [torch.load(d / f'weights/round_{r:03d}.pt', weights_only=True) for d in (inline, old)]
            for key in ('private', 'proxy', 'ema_private'): assert same(x[key], y[key]), ('old source', key, r)
    deferred = train('deferred', ['--defer_eval', '--session_rounds', 1])
    resumed = train('resumed', ['--defer_eval', '--parent', deferred, '--parent_round', 1, '--workers', 2])
    for r, other in [(1, deferred), (2, resumed)]:
        x, y = [torch.load(d / f'weights/round_{r:03d}.pt', weights_only=True) for d in (inline, other)]
        for key in ('private', 'proxy', 'ema_private'): assert same(x[key], y[key]), (key, r)
        u, v = [torch.load(d / f'resume/round_{r:03d}.pt', weights_only=True) for d in (inline, other)]
        assert same(u['exact'], v['exact']) and same(u['ef'], v['ef'])
    for workers in (1, 2):
        out = work / f'eval{workers}'
        command([proj / 'fd_eval.py', '--run', resumed, '--out', out, '--round', 2, '--workers', workers], f'eval{workers}')
        actual = json.loads((out / 'eval/round_002.json').read_text())
        expected = json.loads((inline / 'complete/round_002.json').read_text())
        assert actual['eval'] == expected['eval'], ('evaluation metrics differ', workers)
        with np.load(out / 'predictions/round_002.npz') as x, np.load(inline / 'predictions/round_002.npz') as y:
            assert set(x.files) == set(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)
    fast = work / 'evalfast'   # fd_eval_fast.py (tolerance class): same outputs, predictions within fp16 rounding of inline
    command([proj / 'fd_eval_fast.py', '--run', inline, '--out', fast, '--rounds', '1,2', '--scratch', work / 'parts'], 'evalfast')
    for r in (1, 2):
        actual = json.loads((fast / f'eval/round_{r:03d}.json').read_text())
        expected = json.loads((inline / f'complete/round_{r:03d}.json').read_text())
        assert actual['fast_gate']['models'] == 2 * a.clients + 1 and actual['eval_source_sha'] == hashlib.sha256((proj / 'fd_eval_fast.py').read_bytes()).hexdigest()
        with np.load(fast / f'predictions/round_{r:03d}.npz') as x, np.load(inline / f'predictions/round_{r:03d}.npz') as y:
            assert set(x.files) == set(y.files) and np.array_equal(x['test_y'], y['test_y'])
            agree = min(float((x[k] == y[k]).mean()) for k in x.files)
        for role, holds in expected['eval'].items():
            for rule, ev in holds['test'].items():
                for key, value in ev['mean_metrics'].items(): assert abs(actual['eval'][role]['test'][rule]['mean_metrics'][key] - value) < .02, (role, rule, key)
        assert agree >= .99, ('fast evaluator agreement', r, agree)
        print('fast eval round', r, 'min prediction agreement', agree, 'gate', actual['fast_gate'], flush=True)
    assert not any((work / 'parts').iterdir()), 'parts left after merge'
    marker = resumed / 'complete/round_002.json'; original = marker.read_text()
    tampered = json.loads(original); tampered['weights_sha'] = '0' * 64; marker.write_text(json.dumps(tampered))
    command([proj / 'fd_eval.py', '--run', resumed, '--out', work / 'tampered', '--round', 2], 'tamper', ok=False)
    marker.write_text(original)
    assert 'weights sha' in (work / 'tamper.log').read_text()
    config = json.loads((deferred / 'config.json').read_text())['config']
    assert config['rounds'] == 50 and 'session_rounds' not in config
    cmd = [sys.executable, '-c', "from fd import fixture_path; from types import SimpleNamespace as S; assert str(fixture_path(S(final=True,smoke=False))).endswith('/papers/final/fixture')"]
    subprocess.run(cmd, env=env, check=True)
    result = json.loads((deferred / 'result.json').read_text())
    assert not result['complete'] and result['session_complete'] and result['evaluation_pending']
    print('PASS', a.clients, a.backend, 'inline == detached 1/2 workers; private/EMA/proxy predictions, 10 metrics; codec resume; SHA rejection', flush=True)


if __name__ == '__main__': main()
