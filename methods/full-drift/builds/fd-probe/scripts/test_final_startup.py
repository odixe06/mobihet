"""Regression: final-only Gaussian context and notebook subprocess error propagation (CPU)."""
import contextlib
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch

from gen_fd import BASE, ROOT, FULL, RUN_POOL

sys.path[:0] = [str(BASE / 'source/proj'), str(FULL), str(ROOT / '.compat')]
import fd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', type=Path, help='small 20-client pilot fixture; enables non-smoke GPU initialization')
    args = parser.parse_args()
    original_root = fd.P.ROOT
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); fd.P.ROOT = root
        try:
            final = root / 'papers/final/fixture/manifest.json'
            final.parent.mkdir(parents=True); final.write_text('{"final": true}')
            run = SimpleNamespace(init=fd.P.make('dagsnet'), dev=torch.device('cpu'),
                                  a=SimpleNamespace(arm='n_full', seed=42, final=True, smoke=False))
            gaussian = fd.Gaussian(run, initialize=False)
            expected = hashlib.sha256(json.dumps(dict(arm='n_full', seed=42, fixture=fd.digest(final),
                omega=fd.G.OMEGA_SHA, epsilon=fd.G.EPS, schema=1), sort_keys=True).encode()).digest()[:16]
            assert gaussian.context == expected
            assert not (root / 'papers/pilot').exists(), 'regression needs final-only filesystem'
            # The same fixture passed to the frozen constructor must reproduce the observed crash.
            try: fd.G.Method(run, initialize=False)
            except FileNotFoundError: pass
            else: raise AssertionError('legacy constructor no longer reproduces the failure')
            pilot = root / 'papers/pilot/fixture/manifest.json'
            pilot.parent.mkdir(parents=True); pilot.write_text('{"pilot": true}')
            run.a.final = False
            old, new = fd.G.Method(run, initialize=False), fd.Gaussian(run, initialize=False)
            assert old.context == new.context and torch.equal(old.global_vector, new.global_vector)
            assert old.form == new.form and old.mode == new.mode and old.rank == new.rank
        finally:
            fd.P.ROOT = original_root

        # Execute the actual notebook cell, launching a real child process that exits with a known code.
        proj = root / 'proj'; proj.mkdir()
        for rc in (7, 0):
            out = root / f'output-{rc}'; out.mkdir()
            (proj / 'fd.py').write_text(
                "import pathlib,json,sys\n"
                f"p=pathlib.Path({str(out)!r})/'fd_dml_s42_e50';p.mkdir(exist_ok=True)\n"
                "(p/'result.json').write_text(json.dumps(dict(history=[dict(round=1)],session_complete=True)))\n"
                f"sys.exit({rc})\n")
            scope = dict(ARMS=['fd_dml:50'], SESSION_HOURS=2, SESSION_T0=time.monotonic(),
                time=SimpleNamespace(monotonic=time.monotonic, sleep=lambda _: None), OUT=out, PROJ=proj,
                WORK=root, SEED=42, WORKERS=2, CLIENTS=20, PARENT=None, PARENT_DIR=None,
                FAST=True, FINAL=True, DEFER_EVAL=True, SESSION_ROUNDS=4,
                os=os, sys=sys, subprocess=subprocess, json=json)
            failed = False
            with contextlib.redirect_stdout(io.StringIO()):
                try: exec(compile(RUN_POOL, '<RUN_POOL>', 'exec'), scope)
                except AssertionError as e:
                    assert 'training subprocess failed' in str(e); failed = True
            assert failed == (rc != 0), ('notebook failed to propagate child exit', rc)
            assert json.loads((out / 'fd_summary.json').read_text())['fd_dml:50']['rc'] == rc
    print('PASS final-only Gaussian; frozen constructor reproduces crash; pilot context exact; child rc7 fails notebook, rc0 succeeds')
    if args.fixture: production_initialization(args.fixture)


def production_initialization(origin):
    """Real non-smoke Run/Method; replace only production row-count gate with explicit small-fixture gate."""
    original_root, original_gate = fd.P.ROOT, fd.check_final_manifest
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); fixture = root / 'papers/final/fixture'; fixture.mkdir(parents=True)
        manifest = json.loads((origin / 'manifest.json').read_text())
        for k in range(100):
            for name in ('X', 'y'):
                target = fixture / f'train_{k}_{name}.npy'
                np.save(target, np.load(origin / f'train_{k % 20}_{name}.npy')[:1023])
        for name in ('X', 'y'):
            (fixture / f'test_{name}.npy').write_bytes((origin / f'old_{name}.npy').read_bytes())
        (fixture / 'manifest.json').write_text(json.dumps(manifest))
        fd.P.ROOT = root
        def fixture_gate(m, counts, test_rows):
            assert len(counts) in (20, 50, 100) and counts == [1023] * len(counts) and test_rows == 256
            assert m == manifest and not (root / 'papers/pilot').exists()
        fd.check_final_manifest = fixture_gate
        try:
            fd.deterministic(True)
            for clients in (20, 50, 100):
                for arm in ('fd_dml_m4b_gpool_frz_gkd', 'fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl'):
                    contexts = []
                    for rank in (0, 1):
                        config = dict(arm=arm, seed=42, final=True, smoke=False, fast=True, clients=clients,
                            backend='eager', batch=256 if clients == 100 else 512, aux_batch=256, aux_weight=1., wandb_online=False,
                            rounds=50, stop_after=50, lr=.001, lr_min=.00001, clip=1., workers=2, rank=rank)
                        run = fd.Run(SimpleNamespace(**config)); method = fd.Method(run); method.prior()
                        contexts.append(method.gaussian.context)
                        assert run.K == clients and run.a.smoke is False and method.gaussian.tables
                        # One actual production-size private/proxy step path, including tail and Gaussian KD.
                        method.start_round(); k = run.own[0]; method.prepare(k, 1)
                        row = run.private_epoch(method, k, 1); _, proxy = method.proxy_epoch(k, 1)
                        assert row['applied'] > 0 and proxy['applied'] > 0
                        del method, run
                        torch.cuda.empty_cache()
                    assert contexts[0] == contexts[1]
                    print('PASS non-smoke final-only Run/Method + private/proxy epoch', clients, arm, 'ranks 0/1', flush=True)
        finally:
            fd.P.ROOT, fd.check_final_manifest = original_root, original_gate


if __name__ == '__main__': main()
