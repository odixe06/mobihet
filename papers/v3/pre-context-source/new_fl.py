"""Auditable unlimited-channel screen for the new methods; pilot-new-fl.md fixes the gate.

Checkpoint every round, reset optimizer each client/round, eval internal holdouts every five
rounds. No official test data. Finite-contact training is a separate gate, never inferred here.
"""
import argparse, csv, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from mh import pilot as P
from mh.new_methods import METHODS
from mh.metrics import METRIC_KEYS, metrics_from_confusion, per_class_from_confusion


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    if path.suffix == '.pt':
        torch.save(obj, tmp)
    else:
        tmp.write_text(json.dumps(obj, allow_nan=False) + '\n')
    os.replace(tmp, path)


class Run(P.Run):
    def __init__(self, a):
        super().__init__(a)
        if a.smoke:
            self.X = [x[:512] for x in self.X]; self.Y = [y[:512] for y in self.Y]
            self.H = {h: (x[:512], y[:512]) for h, (x, y) in self.H.items()}
            self.n = [len(y) for y in self.Y]
            self.counts = torch.stack([torch.bincount(y, minlength=P.C) for y in self.Y]).double()
            total = self.counts.sum(0)
            self.log_pg = torch.log((total + 1) / (total + 1).sum()).float()
            self.log_pi = torch.log((self.counts + 1) / (self.counts + 1).sum(1, keepdim=True)).float()

    def ship(self, r, k, d, kind, n):
        assert kind in P.WIRE | {'synthetic_codebook'}
        self.bytes.append((r, k, d, kind, int(n)))

    def train(self, m, k, r):
        mods = m.trainable(k)
        opt = torch.optim.AdamW([{'params': list(x.parameters())} for x in mods], lr=self.lr(r), weight_decay=0.)
        scaler = torch.amp.GradScaler()
        for x in mods:
            x.train()
        g = self.gen(r, k)
        torch.manual_seed(int(np.random.SeedSequence([self.a.seed, r, k, 7]).generate_state(1)[0]))
        X, Y, B = self.X[k], self.Y[k], self.a.batch
        perm = torch.randperm(len(Y), device=self.dev, generator=g)
        total = applied = skipped = 0
        for i in range(0, len(Y), B):
            idx = perm[i:i + B]
            with torch.autocast('cuda', dtype=torch.float16):
                loss = m.loss(k, X[idx].float(), Y[idx], g, r)
            assert torch.isfinite(loss), (k, r, 'nonfinite loss')
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            for x in mods:
                torch.nn.utils.clip_grad_norm_(x.parameters(), self.a.clip)
            before = scaler.get_scale()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if scaler.get_scale() < before:
                skipped += 1
            else:
                total += float(loss.detach()); applied += 1
        assert applied > 0 and skipped <= 16, (k, r, applied, skipped)
        for model in mods:
            assert all(torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point())
        return dict(client=k, rows=len(Y), applied=applied, skipped=skipped, loss=total / applied)

    @torch.no_grad()
    def evaluate_saved(self, out, r):
        result, arrays = {}, {}
        for hold, (X, y) in self.H.items():
            result[hold] = {}; arrays[hold + '_y'] = y.cpu().numpy().astype('uint8')
            cms = {'head': [], 'gp': []}; preds = {'head': [], 'gp': []}
            for k, model in enumerate(self.priv):
                z = self.logits(model, X)
                assert torch.isfinite(z).all(), (hold, k, 'nonfinite logits')
                for rule in cms:
                    bias = self.log_pg - self.log_pi[k] if rule == 'gp' else 0.
                    pred = (z + bias).argmax(1)
                    cm = torch.bincount(y * P.C + pred, minlength=P.C ** 2).view(P.C, P.C).cpu().numpy()
                    cms[rule].append(cm); preds[rule].append(pred.cpu().numpy().astype('uint8'))
            for rule, cc in cms.items():
                mets = [metrics_from_confusion(c) for c in cc]
                pc = np.mean([[p['f1'] for p in per_class_from_confusion(c, P.CLASSES)] for c in cc], 0)
                result[hold][rule] = dict(mean_metrics={key: float(np.mean([m[key] for m in mets])) for key in METRIC_KEYS},
                    f1_macro=float(np.mean([m['f1_macro'] for m in mets])), benign_f1=float(pc[0]),
                    f1_per_class=pc.tolist(), confusion_clients=np.stack(cc).tolist())
                arrays[f'{hold}_{rule}'] = np.stack(preds[rule])
        path = out / 'predictions' / f'round_{r:03d}.npz'; path.parent.mkdir(exist_ok=True)
        with path.with_suffix('.tmp').open('wb') as f:
            np.savez_compressed(f, **arrays)
        os.replace(path.with_suffix('.tmp'), path)
        return {'private': result}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', required=True); p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--rounds', type=int, default=20)
    p.add_argument('--private', default='dagsnet'); p.add_argument('--batch', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--lr_min', type=float, default=1e-5)
    p.add_argument('--clip', type=float, default=1.); p.add_argument('--eval_every', type=int, default=5)
    p.add_argument('--aux_batch', type=int, default=256); p.add_argument('--aux_weight', type=float, default=1.)
    p.add_argument('--synth_steps', type=int, default=200); p.add_argument('--ipc', type=int, default=4)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--resume', action='store_true')
    p.add_argument('--stop_after', type=int); p.add_argument('--out', default=str(P.ROOT / 'papers/new-fl/runs'))
    a = p.parse_args()
    if a.smoke:
        a.rounds, a.eval_every, a.synth_steps = 3, 1, 3
    assert a.private == 'dagsnet', 'current approved experiment uses DAGSNet'
    out = Path(a.out) / f'{a.arm}_s{a.seed}'; out.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(a).items() if k not in ('out', 'resume', 'stop_after')}
    source = {f.name: digest(f) for f in sorted((P.ROOT / 'mh').glob('*.py'))}
    fixture = P.ROOT / 'papers/pilot/fixture'
    manifest = json.loads((fixture / 'manifest.json').read_text())
    for name, sha in manifest['files'].items():
        assert digest(fixture / name) == sha, ('fixture mismatch', name)
    identity = dict(config=config, source=source, fixture=digest(fixture / 'manifest.json'))
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if (out / 'config.json').exists():
        assert a.resume and json.loads((out / 'config.json').read_text())['fingerprint'] == fingerprint
    else:
        assert not a.resume, 'resume requested without a run'
        save(out / 'config.json', dict(identity, fingerprint=fingerprint))
    t0 = time.time(); R = Run(a); m = METHODS[a.method](R)
    hist = []; previous = None
    if a.resume:
        saved = torch.load(out / 'knowledge.pt', weights_only=True, map_location=R.dev)
        for key, value in saved['state'].items():
            setattr(m, key, value)
        R.bytes = saved['bytes']
        for marker in sorted((out / 'complete').glob('round_*.json')):
            row = json.loads(marker.read_text()); r = row['round']
            assert r == len(hist) + 1 and row['prev_sha'] == previous and row['fingerprint'] == fingerprint
            weight = out / 'weights' / f'round_{r:03d}.pt'
            assert digest(weight) == row['weights_sha']
            assert row['knowledge_sha'] == digest(out / 'knowledge.pt')
            hist.append(row); previous = row['weights_sha']
        if hist:
            state = torch.load(out / 'weights' / f'round_{len(hist):03d}.pt', weights_only=True, map_location=R.dev)
            for model, sd in zip(R.priv, state['models']):
                model.load_state_dict(sd, strict=True)
    else:
        m.setup()
        state = {k: v for k, v in vars(m).items() if k != 'R'}
        save(out / 'knowledge.pt', dict(state=state, bytes=R.bytes))
        torch.load(out / 'knowledge.pt', weights_only=True, map_location='cpu')
    knowledge_sha = digest(out / 'knowledge.pt')
    print('setup', a.arm, f'{time.time() - t0:.1f}s', 'payload up/down',
          [sum(b[-1] for b in R.bytes if b[2] == direction) for direction in ('up', 'down')], flush=True)
    for r in range(len(hist) + 1, a.rounds + 1):
        t1 = time.time(); logs = []
        for k in range(R.K):
            logs.append(R.train(m, k, r))
        row = dict(round=r, lr=R.lr(r), clients=logs, loss=float(np.mean([l['loss'] for l in logs])),
                   fingerprint=fingerprint, prev_sha=previous, knowledge_sha=knowledge_sha)
        if r % a.eval_every == 0 or r == a.rounds:
            row['eval'] = R.evaluate_saved(out, r)
            row['predictions_sha'] = digest(out / 'predictions' / f'round_{r:03d}.npz')
        weights = dict(models=[{k: v.detach().cpu() for k, v in model.state_dict().items()} for model in R.priv],
                       architecture=dict(P.CFG, n_features=P.D, num_classes=P.C), fingerprint=fingerprint, round=r)
        path = out / 'weights' / f'round_{r:03d}.pt'; save(path, weights)
        row['weights_sha'] = digest(path); previous = row['weights_sha']; row['sec'] = time.time() - t1
        save(out / 'complete' / f'round_{r:03d}.json', row)  # marker last; static method needs no optimizer/RNG bundle
        hist.append(row)
        score = row.get('eval', {}).get('private', {}).get('group', {}).get('gp', {}).get('f1_macro')
        print(f'r{r} {a.arm} s{a.seed} group gp {score} loss {row["loss"]:.4f} {row["sec"]:.0f}s', flush=True)
        if a.stop_after is not None and r >= a.stop_after:
            break
    result = dict(config=vars(a), fingerprint=fingerprint, history=hist, bytes=R.bytes,
                  diagnostics=getattr(m, 'diag', {}), train_rows=R.n, minutes=(time.time() - t0) / 60,
                  complete=len(hist) == a.rounds)
    save(out / 'result.json', result)
    with (out / 'metrics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['round', 'holdout', 'rule', *METRIC_KEYS]); w.writeheader()
        for h in hist:
            for hold, rules in h.get('eval', {}).get('private', {}).items():
                for rule, e in rules.items():
                    w.writerow(dict(round=h['round'], holdout=hold, rule=rule, **e['mean_metrics']))
    print('done', a.arm, 'complete', result['complete'], flush=True)


if __name__ == '__main__':
    main()
