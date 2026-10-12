"""Actual generated controller: deadline terminates workers and identities survive."""
import ast
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def check(folder, mode):
    metadata = json.loads((folder / 'kernel-metadata.json').read_text())
    notebook = nbformat.read(folder / metadata['code_file'], as_version=4)
    code = ast.literal_eval(ast.parse(notebook.cells[2].source).body[0].value)
    manifest = json.loads((folder / 'source_manifest.json').read_text())
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        work, out, source, src = [base / name for name in ('work', 'runs', 'source', 'src')]
        for path in (work, source, src):
            path.mkdir()
        (src / 'manifest.json').write_bytes((folder / 'manifest.json').read_bytes())
        launched, terminated = [], []

        class Clock:
            calls = 0

            @classmethod
            def time(cls):
                cls.calls += 1
                return 0. if cls.calls <= (1 if mode == 'unstarted' else 5) else 4.6 * 3600

            @staticmethod
            def sleep(_):
                pass

        class Process:
            def __init__(self, command, **kwargs):
                self.arm = command[command.index('--arm')+1]
                assert '--wandb_online' in command
                launched.append(self.arm)
                self.returncode = None

            def poll(self):
                return self.returncode

            def terminate(self):
                terminated.append(self.arm)
                self.returncode = -15

            def kill(self):
                self.returncode = -9

        namespace = dict(sys=sys, os=SimpleNamespace(environ={'WANDB_API_KEY': 'TEST_ONLY'}),
            json=json, hashlib=hashlib, shutil=__import__('shutil'), Path=Path, time=Clock,
            subprocess=SimpleNamespace(Popen=Process, STDOUT=-2), OUT=out, WORK=work,
            SOURCE=source, SRC=src, CODE=code, manifest=manifest)
        try:
            exec(compile(notebook.cells[-1].source, 'actual V6 controller', 'exec'), namespace)
        except AssertionError:
            pass
        else:
            raise AssertionError('deadline failure accepted as complete')
        seed, methods = namespace['SEED'], namespace['METHODS']
        summary = json.loads((out / f'summary_s{seed}.json').read_text())
        assert set(summary) == set(methods) and all(not v['complete'] for v in summary.values())
        if mode == 'unstarted':
            assert not launched and not terminated
        else:
            assert launched == methods[:4] and terminated == methods[:4]
        for arm in methods[len(launched):]:
            identity = json.loads((out / f'{arm}_s{seed}/config.json').read_text())
            fingerprint = identity.pop('fingerprint')
            assert hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() == fingerprint
            assert summary[arm]['rc'] == -998
        assert not work.exists()
    print('PASS V6 deadline', folder.name, mode)


if __name__ == '__main__':
    for folder in sorted((ROOT / 'papers/v6/notebook').iterdir()):
        for mode in ('unstarted', 'terminate'):
            check(folder, mode)
