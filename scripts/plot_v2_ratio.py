"""Saved C gate/score/per-class diagnostics; association only, no causal selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def main():
    gates=json.loads((ROOT/'papers/v2/gates.json').read_text())['per_seed']
    scores=json.loads((ROOT/'papers/v4/ratio-score-probe.json').read_text())['checks']
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for seed,color in ((42,'#2563eb'),(43,'#d97706')):
        ess=gates[f'C_s{seed}']['ess_fraction']; sample=[v for v in scores if v['seed']==seed]
        def f1(arm):
            run=json.loads((ROOT/f'papers/v2/runs/{arm}_s{seed}/result.json').read_text())
            return np.array(run['history'][-1]['eval']['private']['group']['gp']['f1_per_class'])
        gain=f1('c_fl')-f1('diag')
        values=[ess,[v['floor_fraction'] for v in sample],
                [v['median_ratio_score_norm'] for v in sample],gain]
        for ax,y in zip(axes.flat,values): ax.plot(range(16),y,'.-',label=f'seed{seed}',color=color)
    titles=['Importance-weight ESS fraction, C r20','Ratio logit <= -10, 64 fresh q points/class',
            'Median ||score(z) + z||, FP32 clipped ratio','Private group/gp per-class F1: C_FL - diag']
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title,fontsize=10); ax.set_xticks(range(16)); ax.grid(alpha=.2); ax.legend()
    axes[1,1].axhline(0,color='black',lw=.7)
    axes[1,0].set_xlabel('class ID'); axes[1,1].set_xlabel('class ID')
    fig.suptitle('Verified C snapshots; score probe is historical diagnostic, no MALA outcome or causality')
    fig.tight_layout(); base=ROOT/'papers/v4'
    fig.savefig(base/'historical-c-diagnostics.png',dpi=180); fig.savefig(base/'historical-c-diagnostics.svg')
    plt.close(fig); print('Wrote historical C diagnostics PNG/SVG')


if __name__=='__main__': main()
