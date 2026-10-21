"""Join audited final training/evaluation artifacts; retain missing evaluations explicitly."""
import csv
import json
from pathlib import Path

from collect_fd import BASE, ROOT
from mh.metrics import METRIC_KEYS


def main():
    destination = ROOT / 'docs/methods/full-drift/results/final'
    destination.mkdir(parents=True, exist_ok=True)
    evaluations = {}
    for path in sorted((BASE / 'artifacts/kaggle').glob('*/AUDITED.json')):
        proof = json.loads(path.read_text())
        if 'weights_sha' in proof and ('round' in proof or 'rounds' in proof) and proof.get('complete'):
            rounds = proof.get('rounds', [proof.get('round')])   # fast-evaluator notebooks score several rounds
            shas = proof['weights_sha'] if isinstance(proof['weights_sha'], dict) else {str(rounds[0]): proof['weights_sha']}
            for r in rounds:
                key = (proof['fingerprint'], r, shas[str(r)])
                row = json.loads((path.parent / 'runs' / proof['run'] / 'eval' / f'round_{r:03d}.json').read_text())
                fast = proof.get('evaluator') == 'fd_eval_fast.py'   # owner 21/10: one evaluator for the campaign, fast wins
                if key in evaluations and evaluations[key][0] == fast:
                    assert evaluations[key][1]['eval'] == row['eval'], 'conflicting evaluation attempts'
                if key not in evaluations or fast: evaluations[key] = (fast, row)
    costs, metrics, inventory, scenarios = [], [], [], {}
    for path in sorted((BASE / 'artifacts/kaggle').glob('*/AUDITED.json')):
        proof = json.loads(path.read_text())
        for audit in proof.get('audits', []):
            run = path.parent / 'runs' / audit['run']; cfg = json.loads((run / 'config.json').read_text())
            config = cfg['config']
            if not config.get('final'): continue
            result = json.loads((run / 'result.json').read_text())
            base = dict(kernel=proof['kernel'], run=run.name, clients=config.get('clients', 20), seed=config['seed'], arm=config['arm'])
            ancestor = config.get('parent'); root_fingerprint = cfg['fingerprint']
            while ancestor:
                root_fingerprint = ancestor['fingerprint']; ancestor = ancestor.get('parent')
            key = (config['arm'], base['clients'], base['seed'], root_fingerprint)
            scenario = scenarios.setdefault(key, dict(arm=config['arm'], clients=base['clients'], seed=base['seed'],
                                                       root_fingerprint=root_fingerprint, weights={}, evaluated=[]))
            completed, scored = [], []
            for row in result['history']:
                r = row['round']; completed.append(r)
                if r in scenario['weights']: assert scenario['weights'][r] == row['weights_sha'], 'conflicting round lineage'
                scenario['weights'][r] = row['weights_sha']
                cost = dict(base, round=r, algorithm_seconds=row['algorithm_seconds'], committed_round_seconds=row['sec'], **row['cost'])
                by_client = [sum(v[4] for v in result['bytes'] if v[0] == r and v[1] == k) for k in range(base['clients'])]
                cost.update(per_client_round_bytes_min=min(by_client), per_client_round_bytes_max=max(by_client),
                            per_client_round_bytes_mean=sum(by_client) / len(by_client))
                cost['bootstrap_bytes_session'] = sum(v[4] for v in result['bytes'] if v[0] == 0)
                costs.append(cost)
                ev = evaluations.get((cfg['fingerprint'], r, row['weights_sha']), (None, None))[1]
                if ev:
                    scored.append(r)
                    scenario['evaluated'].append(r)
                    for role, holds in ev['eval'].items():
                        for rule, values in holds['test'].items():
                            metrics.append(dict(base, round=r, role=role, rule=rule, **values['mean_metrics']))
            inventory.append(dict(base, training_rounds=completed, evaluated_rounds=scored,
                                  missing_eval_rounds=sorted(set(completed) - set(scored))))
    for name, rows, columns in [('costs.csv', costs, None), ('metrics.csv', metrics, ['kernel', 'run', 'clients', 'seed', 'arm', 'round', 'role', 'rule', *METRIC_KEYS])]:
        columns = columns or sorted(set().union(*(set(row) for row in rows)))
        with (destination / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    (destination / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    status = []
    for scenario in scenarios.values():
        scenario['training_rounds'] = sorted(scenario.pop('weights'))
        scenario['evaluated_rounds'] = sorted(set(scenario.pop('evaluated')))
        scenario['training_complete'] = scenario['training_rounds'] == list(range(1, 51))
        scenario['evaluation_complete'] = scenario['evaluated_rounds'] == list(range(1, 51))
        status.append(scenario)
    (destination / 'completion.json').write_text(json.dumps(status, indent=2) + '\n')
    print('Audited final:', len(costs), 'trained rounds,', len(metrics), 'metric rows;', destination, flush=True)


if __name__ == '__main__': main()
