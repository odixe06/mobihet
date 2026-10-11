"""Communication-only probe. Sizes follow S-a; no training and no F1 claim.

One contact/client/round, histogram paid, coherent/resumable objects, server publications
after complete cohorts of >=5. Codebook delivery uses source/class objects. Packet/header
and frame/ACK costs are distinct. The sender tries its fixed queue; only the channel sees
the true budget. Availability is committed after all contacts, avoiding same-round cid bias.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from mh.channel import Transfer, budgets, contact


def probe(method, tier):
    K, rounds = 20, 20
    up_size = {'local': 0, 'diag': 8544, 'kme': 16736, 'dm': 8544}[method]
    down_size = {'diag': 4320, 'kme': 8416}
    prior_up = [Transfer(('count', k), 96) for k in range(K)]
    up = [Transfer(('knowledge', k), up_size) if up_size else None for k in range(K)]
    down = [None] * K; prior_down = [None] * K
    got_prior, priors_uploaded, uploaded = set(), set(), set()
    prior_version = version = 0
    prior_active = [0] * K; active = [0] * K
    got = [set() for _ in range(K)]
    first = [None] * K; history = []
    bt, _ = budgets(tier, rounds, K)
    for r in range(1, rounds + 1):
        inbox_count, inbox_up, used_round = [], [], 0.
        for k in range(K):
            remaining = float(bt[r, k])
            queue = []
            if k not in priors_uploaded:
                queue.append(prior_up[k])
            if prior_version > prior_active[k] and prior_down[k] is None:
                prior_down[k] = Transfer(('prior', prior_version), 96)
            if prior_down[k] is not None:
                queue.append(prior_down[k])
            if up[k] is not None and k not in uploaded:
                queue.append(up[k])
            for t in queue:
                done, used = contact([t], remaining); remaining -= used; used_round += used
                if done:
                    if t is prior_up[k]:
                        inbox_count.append(k)
                    elif t is prior_down[k]:
                        prior_active[k] = t.key[1]; got_prior.add(k); prior_down[k] = None
                    else:
                        inbox_up.append(k)
                if remaining <= 0:
                    break
            while remaining > 0 and method != 'local':
                if down[k] is None:
                    if method == 'dm':
                        # Round-robin source/class queue; packet contains its own ids.
                        order = [((k + 1 + i % (K - 1)) % K, (k + i) % 16) for i in range(16 * (K - 1))]
                        candidates = [(source, c) for source, c in order if source in uploaded and (source, c) not in got[k]]
                        if not candidates:
                            break
                        source, c = candidates[0]
                        down[k] = Transfer(('book', source, c), 32 + 4 + 4 * 66 * 2)
                    else:
                        if version <= active[k]:
                            break
                        down[k] = Transfer(('stats', version), down_size[method])
                t = down[k]; done, used = contact([t], remaining)
                remaining -= used; used_round += used
                if done:
                    if method == 'dm':
                        got[k].add((t.key[1], t.key[2]))
                    else:
                        active[k] = t.key[1]
                    first[k] = r if first[k] is None else first[k]
                    down[k] = None
                else:
                    break
        priors_uploaded.update(inbox_count); uploaded.update(inbox_up)
        prior_version = len(priors_uploaded) // 5
        version = len(uploaded) // 5
        assert used_round <= bt[r].sum() + 1e-6
        history.append(dict(round=r, bytes=used_round, budget=float(bt[r].sum()),
            uploads=len(uploaded), receivers=sum(x is not None for x in first), prior_receivers=len(got_prior)))
    return dict(method=method, tier=tier, first_knowledge_round=first, history=history,
                received_source_class_objects=[len(x) for x in got], active_stats_versions=active,
                received_classes=[len({c for _, c in x}) for x in got],
                received_sources=[len({s for s, _ in x}) for x in got])


results = [probe(method, tier) for tier in ('B0', 'B2') for method in ('local', 'diag', 'kme', 'dm')]
out = ROOT / 'papers/new-fl/contact-probe.json'; out.write_text(json.dumps(results, indent=2) + '\n')
print('| method | tier | uploads r20 | receivers r20 | median first round | mean bytes/contact |')
print('|---|---|---:|---:|---:|---:|')
for result in results:
    h = result['history']; first = [r for r in result['first_knowledge_round'] if r is not None]
    print(f'| {result["method"]} | {result["tier"]} | {h[-1]["uploads"]} | {h[-1]["receivers"]} | '
          f'{float(np.median(first)) if first else "N/A"} | {sum(x["bytes"] for x in h)/400:.0f} B |')
print('Communication-only; no training/F1 or secure-aggregation cryptographic overhead measured.')
