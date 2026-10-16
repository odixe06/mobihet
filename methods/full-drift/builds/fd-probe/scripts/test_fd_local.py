"""Local code test (pilot fixture, 1024 rows/client, eager): fd_dml/fd_gaussian epoch == frozen full_run epoch bit-exact;
sub4 with one part == control; every arm runs a 2-epoch smoke. Run via scripts/run_local_guarded.py.

    python methods/full-drift/builds/fd-probe/scripts/test_fd_local.py --work <scratch> --part checks|smoke [--manifest m.json]
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
FULL = ROOT / 'methods/shared-baselines/builds/federated-baselines-full/source/proj'


def one_epoch(fd, arm, use_frozen, parts=None, batch=512):
    import torch
    from full_run import Run as FullRun
    a = SimpleNamespace(arm=arm, seed=42, rounds=50, batch=batch, lr=1e-3, lr_min=1e-5, clip=1., aux_batch=64, aux_weight=1.,
                        backend='eager', smoke=True, wandb_online=False)
    R = fd.Run(a)
    if parts: R.parts = parts
    m = fd.Method(R); m.prior(); step = 0
    for part in range(R.parts):
        step += 1; m.start_round()
        for k in range(R.K):
            if m.kd: m.prepare(k, step)
            if use_frozen: FullRun.private_epoch(R, m, k, 1)
            else: R.private_epoch(m, k, 1, part)
            if m.kd:
                packet, _ = m.proxy_epoch(k, step)
                if packet is not None: m.aggregate_sum.add(k, packet)
        m.aggregate(step)
    state = [v.clone() for model in R.priv for v in model.state_dict().values()]
    if m.kd: state += [v.clone() for v in m.global_state.values()]
    return state


def main():
    p = argparse.ArgumentParser(); p.add_argument('--work', required=True)
    p.add_argument('--manifest', help='data_manifest.json pulled from a Kaggle setup: exercises the production gate')
    # Owner RAM limit: run 'checks' and 'smoke' as separate invocations so only one CUDA process is alive at a time.
    p.add_argument('--part', choices=('checks', 'smoke'), required=True)
    a = p.parse_args()
    work = Path(a.work); proj = work / 'proj'
    if proj.exists(): shutil.rmtree(proj)
    proj.mkdir(parents=True)
    for f in ('full_run.py', 'full_method.py', 'full_driver.py'): shutil.copyfile(FULL / f, proj / f)
    shutil.copyfile(BASE / 'source/proj/fd.py', proj / 'fd.py')
    sys.path[:0] = [str(proj), str(ROOT)]
    if a.part == 'smoke':
        import ast
        tree = ast.parse((proj / 'fd.py').read_text())
        arms = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == 'ARMS')
        return smoke(arms, proj, work)
    import torch
    import fd
    if a.manifest:
        import json
        manifest = json.loads(Path(a.manifest).read_text()); n = [sum(manifest['counts'][f'train_{k}']) for k in range(20)]
        fd.check_full_manifest(manifest, n, 10761343)
        for bad in (n[:-1] + [n[-1] - 1], ):
            try: fd.check_full_manifest(manifest, bad, 10761343)
            except AssertionError: pass
            else: raise SystemExit('production gate accepted wrong row counts')
        print('production manifest gate PASS on', a.manifest, 'train rows', sum(n), flush=True)
    for arm in ('fd_dml', 'fd_gaussian'):
        ours, frozen = one_epoch(fd, arm, False), one_epoch(fd, arm, True)
        assert len(ours) == len(frozen) and all(torch.equal(x, y) for x, y in zip(ours, frozen)), arm
        print('BIT-EXACT vs frozen full_run epoch:', arm, len(ours), 'tensors', flush=True)
    # smoke has 1024 rows/client: batch128 gives 8 batches so every quarter is non-empty (full: ~4,000 batches)
    ctl, sub1 = one_epoch(fd, 'fd_dml', False, batch=128), one_epoch(fd, 'fd_dml_sub4', False, parts=1, batch=128)
    assert all(torch.equal(x, y) for x, y in zip(ctl, sub1)); print('sub4 with one part == control', flush=True)
    sub4 = one_epoch(fd, 'fd_dml_sub4', False, batch=128)
    assert any(not torch.equal(x, y) for x, y in zip(ctl, sub4)); print('sub4 four parts differs from control', flush=True)
    dense, q4 = one_epoch(fd, 'fd_dml_sub4', False, batch=128), one_epoch(fd, 'fd_dml_sub4_q4', False, batch=128)
    rel = max(float((x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-12)) for x, y in zip(q4, dense) if y.is_floating_point())
    assert 0 < rel < .5, rel; print('sub4_q4 differs from dense sub4 but stays close: max rel', round(rel, 4), flush=True)
    zs, zr, y = torch.randn(64, 16), torch.randn(64, 16), torch.randint(0, 16, (64,))
    kl = lambda a, b: float(torch.nn.functional.kl_div(fd.not_true(a, y), fd.not_true(b, y), log_target=True, reduction='batchmean'))
    moved = zr.clone(); moved[torch.arange(64), y] += 9.
    keep = ~torch.nn.functional.one_hot(y, 16).bool()
    ref = sum(torch.nn.functional.kl_div(zs[i][keep[i]].log_softmax(0), zr[i][keep[i]].log_softmax(0), log_target=True, reduction='sum')
              for i in range(64)) / 64
    assert abs(kl(zs, zr) - kl(zs, moved)) < 1e-6 and abs(kl(zs, zr) - float(ref)) < 1e-5
    print('ntd KL ignores the teacher true-class logit and equals the 15-class KL', flush=True)
    base = dict(seed=42, rounds=50, batch=128, lr=1e-3, lr_min=1e-5, clip=1., aux_batch=64, aux_weight=1., backend='eager',
                smoke=True, wandb_online=False)
    for arm, n in (('fd_dml_m4b', 13), ('fd_dml_swappos', 4), ('fd_dml_ntd', 0), ('fd_gaussian_m4b', 13), ('fd_dml_m4b_sub4_q4', 13)):
        R = fd.Run(SimpleNamespace(arm=arm, **base)); x = R.mask(R.X[0][:8].float())
        assert len(R.zero) == n and (x[:, R.zero] == 0).all() and R.swap == (arm == 'fd_dml_swappos') and R.ntd == (arm == 'fd_dml_ntd')
        del R
    print('round-3 hooks: m4b zeroes 13 receiver/position features, swappos swaps + zeroes 4, ntd masks none', flush=True)
    for arm in ('fd_dml_ntd', 'fd_dml_hkd'):
        state = one_epoch(fd, arm, False, batch=128)
        assert any(not torch.equal(x, y) for x, y in zip(ctl, state)); print(arm, 'differs from control', flush=True)
    for arm in ('fd_dml', 'fd_dml_avgm', 'fd_fedavg', 'fd_dml_sub16_q4', 'fd_fedavg_sub4_prox', 'fd_dml_m4b', 'fd_dml_swappos', 'fd_dml_ntd', 'fd_dml_q4', 'fd_dml_hkd', 'fd_dml_m4b_q4', 'fd_dml_m4b_sub4_q4'):
        state = one_epoch(fd, arm, False, batch=32 if 'sub16' in arm else 128)
        assert all(torch.isfinite(v.float()).all() for v in state); print('one epoch finite:', arm, flush=True)


def smoke(arms, proj, work):
    for arm in arms:
        cmd = [sys.executable, str(proj / 'fd.py'), '--arm', arm, '--seed', '43', '--smoke', '--backend', 'eager', '--batch', '32' if 'sub16' in arm else '128',
               '--out', str(work / 'runs')]
        out = work / 'runs' / f'{arm}_s43'
        if out.exists(): shutil.rmtree(out)
        r = subprocess.run(cmd, cwd=ROOT, env={**__import__('os').environ, 'PYTHONPATH': f'{proj}:{ROOT}'},
                           capture_output=True, text=True)
        print(arm, 'rc', r.returncode, r.stdout.strip().splitlines()[-3:], r.stderr[-1500:] if r.returncode else '', flush=True)
        assert r.returncode == 0, arm


if __name__ == '__main__':
    main()
