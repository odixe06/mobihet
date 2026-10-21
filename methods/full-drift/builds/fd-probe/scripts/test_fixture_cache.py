"""Fixture cache path of the notebook setup cell (gen_fd.SETUP), run on a fake /kaggle/input (CPU, no GPU asserts).

Checks: an attached cache is SHA-checked and symlinked with the manifest bytes copied; a corrupted file stops the notebook;
no cache falls back to prepare_fd; the cache kernel cell and generated notebooks compile.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types

import numpy as np

import gen_fd
from gen_fixture import CELL


def block(inputs):
    text = gen_fd.SETUP
    start, end = text.index('t=time.monotonic()'), text.index("(SOURCE/'data_manifest.json')")
    return text[start:end].replace("Path('/kaggle/input')", f'Path({str(inputs)!r})')


def run(code, work, fixture, prepared):
    calls = []
    sys.modules['fd_prepare'] = types.SimpleNamespace(prepare_fd=lambda *a, **k: calls.append(1) or prepared)
    scope = dict(Path=Path, json=json, hashlib=hashlib, shutil=__import__('shutil'), time=__import__('time'),
                 WORK=work, FIXTURE=fixture, FINAL=True, CLIENTS=20)
    with contextlib.redirect_stdout(io.StringIO()): exec(compile(code, '<setup>', 'exec'), scope)
    return scope['data'], calls


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); cache = tmp / 'input/mobihet-fd-fixture-k20/cache/papers/final/fixture'; cache.mkdir(parents=True)
        files = {}
        for name in ('train_0_X.npy', 'train_0_y.npy', 'test_X.npy', 'test_y.npy'):
            np.save(cache / name, np.arange(64, dtype=np.float16)); files[name] = hashlib.sha256((cache / name).read_bytes()).hexdigest()
        manifest = dict(source_rows=43045415, counts=dict(train_0=[64]), test_rows=10761343, files=files)
        (cache / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        code = block(tmp / 'input')

        work = tmp / 'w1'; data, calls = run(code, work, manifest, None)
        fx = work / 'papers/final/fixture'
        assert data == manifest and not calls
        assert all((fx / n).is_symlink() and (fx / n).resolve() == (cache / n).resolve() for n in files)
        assert (fx / 'manifest.json').read_bytes() == (cache / 'manifest.json').read_bytes() and not (fx / 'manifest.json').is_symlink()

        data, calls = run(code, tmp / 'w2', None, {'prepared': True})   # non-final / no FIXTURE: always prepare
        assert calls and data == {'prepared': True}
        other = dict(manifest, test_rows=1)                               # cache for another fixture is ignored
        data, calls = run(code, tmp / 'w3', other, {'prepared': True}); assert calls

        np.save(cache / 'test_y.npy', np.zeros(64, dtype=np.float16))    # corrupted cache file must stop the notebook
        try: run(code, tmp / 'w4', manifest, None)
        except AssertionError as e: assert 'fixture cache' in str(e)
        else: raise AssertionError('corrupted cache accepted')
    compile('MH=[];INPUTS={};CLIENTS=20;EXPECTED={}\n' + CELL, 'fixture-cell', 'exec')
    for k in (20, 50, 100): assert gen_fd.fixture_manifest(k)['files'], k
    print('PASS fixture cache: SHA-checked symlinks + manifest bytes; corrupted file stops; absent/other cache prepares; cells compile')


if __name__ == '__main__': main()
