"""Post-hoc M20 mechanisms: paired cohorts, class confusion, Q contents, update exposure. No training."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.research_metrics import client_f1


def stats(x):
    x = np.asarray(x, dtype=float)
    return {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
            "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('runs/diagnosis-m20'))
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    root = Path('evidence/m20')
    batch = json.loads(Path('evidence/m20-summary/batch.json').read_text())
    audited = json.loads(Path('evidence/m20-summary/analysis.json').read_text())['arms']
    counts = np.load('evidence/research-data/train_counts.npy').astype(np.int64)
    support = np.load('evidence/research-data/test_support.npy').astype(np.int64)
    teachers = np.load('evidence/m20-summary/teacher-tables.npz')
    assert counts.shape == support.shape == (5343,16)
    assert counts.sum() == 37361273 and support.sum() == 9347679
    n = counts.sum(1); assert (n >= 64).all()
    steps = (n + 63) // 64; last = n - 64 * (steps - 1)
    last_synth = 64 / (64 + last)
    # BN momentum .1: weight of final batch within just this epoch's newly supplied moments.
    final_bn_weight = .1 / (1 - .9 ** steps)
    average_synthetic_bn = (.5 * (1 - .9 ** (steps - 1)) * .9 + .1 * last_synth) / (1 - .9 ** steps)
    cohorts = {'all': np.ones(len(n), dtype=bool), 'low256': n < 256,
               '256_1023': (n >= 256) & (n < 1024), '1024_4095': (n >= 1024) & (n < 4096),
               '4096_plus': n >= 4096}
    exposure = {}
    for name, mask in cohorts.items():
        exposure[name] = {'clients': int(mask.sum()), 'train_rows': int(n[mask].sum()), 'test_rows': int(support[mask].sum()),
            'steps_per_round': stats(steps[mask]), 'synthetic_per_class_20round': stats(steps[mask] * 4 * 20),
            'test_rows_per_client': stats(support[mask].sum(1)), 'last_batch_synthetic_fraction': stats(last_synth[mask]),
            'final_batch_share_new_bn_moments': stats(final_bn_weight[mask]),
            'synthetic_fraction_bn_moment_weights': stats(average_synthetic_bn[mask])}
    train_prior = counts / n[:, None]
    test_prior = support / support.sum(1, keepdims=True)
    prior_tv = np.abs(train_prior - test_prior).sum(1) / 2
    prior_shift = {key: stats(prior_tv[mask]) for key, mask in cohorts.items()}
    # Weights are inferred exactly from train counts; confidence/correctness of individual uploaders was not saved.
    aggregation = []
    for c in range(16):
        row = {'class': c, 'contributors': int((counts[:, c] > 0).sum())}
        for name, w in [('count', counts[:, c].astype(float)), ('sqrt', np.sqrt(counts[:, c]))]:
            w /= w.sum(); ordered = np.sort(w)[::-1]
            row[name] = {'effective_clients': float(1 / (w @ w)), 'top1pct_client_mass': float(ordered[:54].sum()),
                         'low256_mass': float(w[n < 256].sum())}
        aggregation.append(row)
    endpoints = {}; arm_detail = {}; probe_detail = {}
    for i, spec in enumerate(batch):
        arm = spec['arm']; d = root / spec['slug'] / spec['slug']
        with np.load(d/'cms_r020.npz') as z:
            cm = z['private'].astype(np.int64)
        assert np.array_equal(cm.sum(2), support)
        endpoints[arm] = cm
        pool = cm.sum(0); tp = pool.diagonal(); den = pool.sum(0) + pool.sum(1)
        f1 = np.divide(2 * tp, den, out=np.zeros(16), where=den > 0)
        np.testing.assert_allclose(f1.mean(), audited[arm]['final']['f1'], rtol=0, atol=1e-12)
        off = pool.copy(); np.fill_diagonal(off, 0)
        pairs = sorted(((int(off[y, p]), y, p) for y in range(16) for p in range(16) if y != p), reverse=True)
        class_support = {}
        for name, mask in cohorts.items():
            categories = {}
            for label, condition in [('absent', counts == 0), ('rare1_16', (counts > 0) & (counts <= 16)),
                                     ('seen_gt16', counts > 16)]:
                sel = condition & mask[:, None]
                denom = (support * sel).sum(0); correct = (cm.diagonal(axis1=1, axis2=2) * sel).sum(0)
                categories[label] = {'rows': denom.tolist(), 'correct': correct.tolist(),
                    'accuracy': np.divide(correct, denom, out=np.full(16, np.nan), where=denom > 0).tolist()}
            class_support[name] = categories
        train_cm = np.load(d/'train_cms_r020.npz')['private'].astype(np.int64)
        assert np.array_equal(train_cm.sum(2), counts)
        seen_classes = (support > 0) & (counts > 0)
        train_recall = np.divide(train_cm.diagonal(axis1=1, axis2=2), counts, out=np.zeros_like(counts, dtype=float), where=counts > 0)
        test_recall = np.divide(cm.diagonal(axis1=1, axis2=2), support, out=np.zeros_like(support, dtype=float), where=support > 0)
        reweighted = {}
        for key, mask in cohorts.items():
            matched = seen_classes & mask[:, None]
            matched_support = (support * matched).sum(0)
            expected_correct = (support * train_recall * matched).sum(0)
            actual_correct = (cm.diagonal(axis1=1, axis2=2) * matched).sum(0)
            reweighted[key] = {'matched_test_rows': matched_support.tolist(),
                'train_recall_reweighted_to_test_clients': np.divide(expected_correct, matched_support, out=np.full(16,np.nan), where=matched_support>0).tolist(),
                'actual_test_recall_same_clients': np.divide(actual_correct, matched_support, out=np.full(16,np.nan), where=matched_support>0).tolist()}
        arm_detail[arm] = {'class_f1': f1.tolist(), 'class_recall': (tp / pool.sum(1)).tolist(),
            'class_precision': (tp / pool.sum(0)).tolist(), 'error_total': int(off.sum()),
            'top_confusions': [{'true': y, 'pred': p, 'rows': v} for v, y, p in pairs[:8]],
            'benign_delay_error_fraction': float((off[0,13] + off[13,0]) / off.sum()),
            'class_support': class_support, 'same_client_class_recall': reweighted,
            'matched_seen_class_train_test_recall_gap': {key: stats((train_recall-test_recall)[seen_classes & mask[:,None]]) for key,mask in cohorts.items()}}
        probes = [json.loads(line) for rank in [0, 1] for line in (d/f'diagnostics_rank{rank}.jsonl').read_text().splitlines()]
        probe_detail[arm] = {}
        for r in [1, 2, 5, 10, 15, 20]:
            clients = [c for rec in probes if rec['round'] == r for c in rec['clients']]
            by_class = []
            for c in range(16):
                terms = [x for cl in clients for x in cl['classes'] if x['class'] == c and x['real_gate_keeps_class']]
                valid = sum(x['logit_cosine_count'] for x in terms)
                by_class.append({'class': c, 'valid_rows_kept': valid,
                    'negative_rows_kept': sum(x['logit_negative_count_before_gate'] for x in terms)})
            probe_detail[arm][r] = {'classes': by_class, 'cohorts': {}}
            for label, mask in [('all', lambda c: True), ('low256', lambda c: c['train_rows'] < 256)]:
                selected = [cl for cl in clients if mask(cl)]
                ratio = [cl['grad_synth_norm']/max(cl['grad_real_norm'],1e-12) for cl in selected]
                kdcos = [cl['cos_real_kd_real'] for cl in selected if cl['cos_real_kd_real'] is not None]
                probe_detail[arm][r]['cohorts'][label] = {'clients': len(selected), 'synth_to_real_norm': stats(ratio),
                    'negative_parameter_kd_cos_fraction': float(np.mean(np.asarray(kdcos) < 0)) if kdcos else None,
                    'shadow_clip_fraction': float(np.mean([cl['total_exceeds_clip'] for cl in selected]))}
    comparisons = [('soft_vs_aug', 2, 0), ('real_vs_noReal', 4, 2), ('fallback_vs_gate', 6, 4),
                   ('post_vs_gate', 8, 4), ('sqrt_vs_gate', 10, 4), ('uniform_vs_fallback', 12, 6),
                   ('local_vs_uniform', 14, 12)]
    paired = {}
    for name, left, right in comparisons:
        seed_deltas = [client_f1(endpoints[f'M{left+s}']) - client_f1(endpoints[f'M{right+s}']) for s in [0,1]]
        per_seed = []
        for s in [0,1]:
            delta = seed_deltas[s];df = np.asarray(arm_detail[f'M{left+s}']['class_f1'])-np.asarray(arm_detail[f'M{right+s}']['class_f1'])
            per_seed.append({'seed':42+s,'cohorts':{label:{**stats(delta[mask]),'improved_fraction':float((delta[mask]>1e-12).mean()),
                'harmed_gt_1pp_fraction':float((delta[mask]<-.01).mean())} for label,mask in cohorts.items()},
                'class_f1_delta':df.tolist(),'macro_delta':float(df.mean()),'macro_delta_excluding_delay':float(np.delete(df,13).mean())})
        paired[name]={'seeds':per_seed,'same_client_both_seeds':{label:{'improved_fraction':float(((seed_deltas[0]>1e-12)&(seed_deltas[1]>1e-12))[mask].mean()),
            'harmed_gt_1pp_fraction':float(((seed_deltas[0]<-.01)&(seed_deltas[1]<-.01))[mask].mean())} for label,mask in cohorts.items()}}
    seed_variability = {}
    for i in range(0,16,2):
        delta = client_f1(endpoints[f'M{i+1}']) - client_f1(endpoints[f'M{i}'])
        seed_variability[f'M{i}_M{i+1}'] = {key: {'abs_delta': stats(abs(delta[mask])),
            'abs_delta_gt_1pp_fraction': float((abs(delta[mask])>.01).mean())} for key,mask in cohorts.items()}
    qdetail = {}
    for arm in teachers.files:
        q = teachers[arm].astype(float); assert q.shape==(20,16,16)
        # Normalize only for entropy measures; retained teacher bytes remain untouched.
        q /= q.sum(2,keepdims=True)
        classes=[]
        for c in range(16):
            row = q[:,c];diag=row[:,c];other=row.copy();other[:,c]=0
            margin=diag-other.max(1);offmass=1-diag
            off=other/np.maximum(offmass[:,None],1e-12)
            hard=np.eye(16)[c];u=np.ones(16)/15;u[c]=0
            # FullQ vs same diagonal and uniform offdiag: target TVD, not classifier utility.
            uniform=diag[:,None]*hard+offmass[:,None]*u
            classes.append({'class':c,'q19_diagonal':float(diag[18]),'q19_margin':float(margin[18]),
                'max_offdiag_index_r19':int(other[18].argmax()),'max_offdiag_share_r19':float(off[18].max()),
                'argmax_flips_Q1_to_Q19':int(np.count_nonzero((row[1:19].argmax(1)==c)!=(row[:18].argmax(1)==c))),
                'q19_TVD_uniform_offdiag':float(np.abs(row[18]-uniform[18]).sum()/2),
                'q19_entropy':float(-(row[18]*np.log(np.maximum(row[18],1e-12))).sum())})
        qdetail[arm]={'classes':classes,'r19_mean_TVD_uniform_offdiag':float(np.mean([x['q19_TVD_uniform_offdiag'] for x in classes]))}
    bandwidth={'bootstrap_bytes':3311744,'Q_round_up_down_bytes':1088,'rounds':20,
               'bootstrap_fraction_total':3311744/(3311744+1088*20)}
    out={'scope':'Post-hoc descriptive M20; same clients across seeds, no independence/significance or causal claim. No protocol/training changes.',
        'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('evidence/m20-summary/batch.json'),Path('evidence/m20-summary/teacher-tables.npz'),Path('evidence/research-data/train_counts.npy'),Path('evidence/research-data/test_support.npy'),Path('evidence/m20.dvc'),Path('evidence/m20-summary.dvc'),Path('evidence/research-data.dvc')]},
        'exposure':exposure,'prior_TVD_per_client':prior_shift,'same_method_seed_variability':seed_variability,'aggregation':aggregation,'arms':arm_detail,'paired':paired,'probes':probe_detail,'teacher':qdetail,'bandwidth':bandwidth}
    def finite(v):
        if isinstance(v,dict):return {k:finite(x) for k,x in v.items()}
        if isinstance(v,list):return [finite(x) for x in v]
        if isinstance(v,float) and not np.isfinite(v):return None
        return v
    (a.out/'mechanisms.json').write_text(json.dumps(finite(out),indent=2,allow_nan=False)+'\n')
    plot(out, a.out / 'mechanisms.png')
    print('M20_MECHANISMS_ANALYSIS_PASS',len(batch),'arms;',len(comparisons),'paired contrasts;',int((n<256).sum()),'low-data clients')


def plot(x, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, key, title in [(axes[0,0], 'real_vs_noReal', 'Add real KD: class F1 change'),
                           (axes[0,1], 'sqrt_vs_gate', 'Server sqrt(n): class F1 change')]:
        values = np.asarray([s['class_f1_delta'] for s in x['paired'][key]['seeds']])
        mean = values.mean(0) * 100
        ax.bar(np.arange(16), mean, color=['#c7473c' if v < 0 else '#277baf' for v in mean])
        ax.scatter(np.arange(16), values[0]*100, s=12, c='black', marker='_', label='seed 42/43')
        ax.scatter(np.arange(16), values[1]*100, s=12, c='black', marker='_')
        ax.axhline(0, color='gray', lw=.6); ax.set_xticks(range(16)); ax.set_xlabel('Class ID (13 = timeDelay)')
        ax.set_title(title); ax.set_ylabel('F1 percentage points'); ax.grid(axis='y', alpha=.2)
    groups = ['absent', 'rare1_16', 'seen_gt16']
    for j, (base, label) in enumerate([(0,'augmentation'),(4,'real gate'),(10,'server sqrt(n)'),(14,'local mixture')]):
        acc = []
        for group in groups:
            v = [x['arms'][f'M{base+i}']['class_support']['low256'][group] for i in [0,1]]
            acc.append(np.mean([sum(c['correct'])/sum(c['rows']) for c in v]))
        axes[1,0].bar(np.arange(3)+(j-1.5)*.19, acc, width=.19, label=label)
    axes[1,0].set_xticks(range(3), ['0', '1–16', '>16']); axes[1,0].set_ylim(0,1)
    axes[1,0].set_title('Low-data clients: accuracy by local class support')
    axes[1,0].set_xlabel('Train rows of test sample’s true class');axes[1,0].set_ylabel('Accuracy')
    axes[1,0].legend(fontsize=8);axes[1,0].grid(axis='y',alpha=.2)
    names=['aug','soft/noReal','real gate','sqrt(n)','uniform','local']
    for j,base in enumerate([0,2,4,10,12,14]):
        v=[x['arms'][f'M{base+i}']['same_client_class_recall']['all'] for i in [0,1]]
        train=np.mean([c['train_recall_reweighted_to_test_clients'][13] for c in v])
        test=np.mean([c['actual_test_recall_same_clients'][13] for c in v])
        axes[1,1].bar(j-.18,train,.36,color='#277baf',label='Train, test-client reweighted' if j==0 else None)
        axes[1,1].bar(j+.18,test,.36,color='#d78b24',label='Own-test, same clients' if j==0 else None)
    axes[1,1].set_xticks(range(6),names,rotation=15);axes[1,1].set_ylim(0,1)
    axes[1,1].set_title('timeDelay: seen-client/class recall gap');axes[1,1].set_ylabel('Recall')
    axes[1,1].legend(fontsize=8);axes[1,1].grid(axis='y',alpha=.2)
    fig.suptitle('M20 endpoint 20; two-seed descriptive analysis, no significance or causal proof',fontsize=12)
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


if __name__ == '__main__':
    main()
