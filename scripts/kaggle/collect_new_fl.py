"""Pull completed/error S-a notebooks, audit frozen source, verify all retained artifacts."""
import argparse, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / '.agents/skills/kaggle-training-notebook/scripts'
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_new_fl import verify
from mh.new_fl import digest


def main():
    completed = 0
    for d in sorted((ROOT / 'papers/new-fl/notebook').iterdir()):
        if not (d / 'PUSHED').exists():
            continue
        meta = json.loads((d / 'kernel-metadata.json').read_text()); kid = meta['id']; owner = kid.split('/')[0]
        dest = ROOT / 'papers/new-fl/kaggle' / d.name
        marker = dest / 'VERIFIED.json'
        if marker.exists():
            stored = json.loads(marker.read_text())
            assert stored['kernel'] == kid and stored['version'] == 1
            assert stored['source'] == json.loads((d / 'source_manifest.json').read_text())
            completed += 1
            continue
        prefix = [sys.executable, str(H / 'kaggle_as.py'), owner, '--']
        status = subprocess.run(prefix + ['kaggle', 'kernels', 'status', kid], check=True, capture_output=True, text=True)
        print(status.stdout.strip(), flush=True)
        if not any(v in status.stdout for v in ('COMPLETE', 'ERROR')):
            continue
        subprocess.run(prefix + [sys.executable, str(H / 'pull_kernel_output.py'), kid, '-p', str(dest), '-j', '16'], check=True)
        subprocess.run(prefix + [sys.executable, str(H / 'download_executed_notebook.py'), kid,
            '--version', '1', '--output', str(d / (d.name + '.executed.ipynb'))], check=True)
        src = dest / 'source'; manifest = json.loads((src / 'source_manifest.json').read_text())
        assert manifest == json.loads((d / 'source_manifest.json').read_text())
        for name, sha in manifest.items():
            assert digest(src / name) == sha, (kid, name)
        assert digest(ROOT / 'knowledge/meta.json') == manifest['meta.json']
        for name in ('dagsnet.py', 'metrics.py'):
            assert digest(ROOT / 'mh' / name) == manifest[name], ('verifier code drift', name)
        seed = int(d.name.rsplit('s', 1)[1])
        summary = json.loads((dest / 'runs' / f'summary_s{seed}.json').read_text())
        assert set(summary) == {'local', 'diag', 'kme_gauss', 'kme', 'dm_gauss', 'dm'}
        assert all(v['rc'] == 0 and v['complete'] for v in summary.values()), summary
        log = ''.join(x.get('data', '') for x in json.loads((dest / (kid.split('/')[1] + '.log')).read_text()))
        assert "gpus ['Tesla T4', 'Tesla T4']" in log and 'verified source and fixture' in log
        for method in sorted(summary):
            run = dest / 'runs' / f'{method}_s{seed}'
            ident = json.loads((run / 'config.json').read_text())
            assert ident['config']['method'] == ident['config']['arm'] == method and ident['config']['seed'] == seed
            assert not ident['config']['smoke'] and ident['config']['rounds'] == 20
            assert all(manifest[name] == sha for name, sha in ident['source'].items())
            verify(run)
            target = ROOT / 'papers/new-fl/runs' / run.name; target.parent.mkdir(exist_ok=True)
            if target.is_symlink():
                assert target.resolve() == run.resolve()
            else:
                assert not target.exists()
                target.symlink_to(run.resolve(), target_is_directory=True)
        (dest / 'VERIFIED.json').write_text(json.dumps(dict(kernel=kid, version=1, seed=seed, runs=6, rounds=20, source=manifest), indent=2))
        print('VERIFIED KERNEL', kid, flush=True)
        completed += 1
    return completed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while main() < 2 and args.watch:
        time.sleep(55)
