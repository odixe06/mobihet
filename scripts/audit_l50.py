"""Audit lossless Core r50, actual CDF frames and teacher/CM/source checks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from methods.mobihet_lossless import decode, tensor_sha
from scripts.audit_screening import audit
from scripts.audit_r20 import extra


def main():
    p = argparse.ArgumentParser()
    for k in ("batch", "root", "out"): p.add_argument("--"+k, type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    common, detail = [], []
    for spec in json.loads(a.batch.read_text()):
        folder = a.root/spec['slug']; d = folder/spec['slug']
        codecs = []
        for rank in (0, 1):
            packet = (d/f'cdf_rank{rank}.bin').read_bytes()
            meta = json.loads((d/f'cdf_codec_rank{rank}.json').read_text())
            assert len(packet) == meta['packet_bytes']
            assert hashlib.sha256(packet).hexdigest() == meta['packet_sha256']
            cdf = torch.from_numpy(decode(packet).astype(np.int32)).double()/65535
            mass = torch.diff(cdf, dim=-1, prepend=torch.zeros_like(cdf[..., :1])).float()
            assert tensor_sha(cdf.float()) == meta['cdf_sha256'] and tensor_sha(mass) == meta['mass_sha256']
            assert meta['bitexact_cdf'] and meta['bitexact_mass'] and (mass>=0).all()
            assert meta['bootstrap_bytes'] == {'up':271031, 'down':729408-630784+len(packet)}
            codecs.append(meta)
        assert codecs[0] == codecs[1]
        bootstrap = sum(codecs[0]['bootstrap_bytes'].values())
        for kernel in spec.get('sessions', [spec['slug']]):
            log = (folder/(kernel+'.log')).read_text()
            assert 'npy cache verified at' in log and 'Tesla T4' in log
            assert all(f'compile gate rank{r}: PASS' in log for r in (0,1))
            assert 'Traceback (most recent call last)' not in log
        common.append(audit(a.root,spec,np.load(ROOT/'evidence/research-data/test_support.npy'),endpoint=50))
        detail.append(extra(a.root,spec,np.load(ROOT/'evidence/research-data/train_counts.npy'),
                            method='mobihet_lossless',bootstrap_total=bootstrap,endpoint=50) | {'cdf_codec':codecs[0]})
    for name, data in [('audit.json',common),('diagnostic-audit.json',detail)]:
        (a.out/name).write_text(json.dumps(data,indent=2)+'\n')
    print('L50_ARTIFACT_AUDIT_PASS',flush=True)


if __name__=='__main__':main()
