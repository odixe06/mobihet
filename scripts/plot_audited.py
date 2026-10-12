"""Publication-ready per-family curves, client-mean class F1 and pooled confusion illustration."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--build',choices=['v3','v4','v5','v6'],required=True); args=parser.parse_args()
    base=ROOT/f'papers/{args.build}'; families={}
    for marker in sorted((base/'kaggle').glob('*/AUDITED.json')):
        audit=json.loads(marker.read_text()); bucket=families.setdefault(audit['family'],{})
        for item in audit['audits']:
            path=marker.parent/'runs'/f'{item["arm"]}_s{audit["seed"]}'
            history=[json.loads(p.read_text()) for p in sorted((path/'complete').glob('round_*.json'))]
            points=[row for row in history if 'eval' in row]
            if points: bucket[(item['arm'],audit['seed'])]=(points,item['complete'])
    for family,runs in families.items():
        if not runs: continue
        fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True)
        for ax,seed in zip(axes,(42,43)):
            for (arm,s),(points,complete) in sorted(runs.items()):
                if s!=seed: continue
                role='proxy' if arm.startswith('b_') else 'private'; rule='head' if role=='proxy' else 'gp'
                ax.plot([p['round'] for p in points],[p['eval'][role]['group'][rule]['f1_macro'] for p in points],
                    '.-',label=arm+(' partial' if not complete else ''))
            ax.set_title(f'{args.build}/{family} seed{seed}'); ax.set_xlabel('round'); ax.grid(alpha=.2)
            if ax.lines: ax.legend(fontsize=7,loc='best')
        axes[0].set_ylabel('group macro-F1; private gp / teacher head')
        fig.tight_layout(); name=f'convergence-{family}'
        fig.savefig(base/f'{name}.png',dpi=180); fig.savefig(base/f'{name}.svg'); plt.close(fig)
        for seed in (42,43):
            chosen=[(arm,points[-1],complete) for (arm,s),(points,complete) in sorted(runs.items()) if s==seed]
            if not chosen: continue
            fig,axes=plt.subplots(len(chosen),2,figsize=(13,3.2*len(chosen)),squeeze=False)
            for row,(arm,last,complete) in enumerate(chosen):
                role='proxy' if arm.startswith('b_') else 'private'; rule='head' if role=='proxy' else 'gp'
                value=last['eval'][role]['group'][rule]
                cm=np.array(value['confusion_clients']).sum(0); normalized=cm/np.maximum(cm.sum(1,keepdims=True),1)
                axes[row,0].imshow(normalized,vmin=0,vmax=1,cmap='Blues',aspect='auto')
                axes[row,0].set_title(f'{arm} r{last["round"]}'+(' partial' if not complete else '')+' pooled CM illustration',fontsize=9)
                axes[row,0].set_xlabel('predicted class'); axes[row,0].set_ylabel('true class')
                axes[row,1].bar(range(16),value['f1_per_class']); axes[row,1].set_ylim(0,1)
                axes[row,1].set_title('mean per-client class F1; headline uses this aggregation',fontsize=9)
                axes[row,1].set_xlabel('class ID'); axes[row,1].set_ylabel('F1')
                for ax in axes[row]: ax.set_xticks(range(16))
            fig.tight_layout(); name=f'classes-{family}-s{seed}'
            fig.savefig(base/f'{name}.png',dpi=160); fig.savefig(base/f'{name}.svg'); plt.close(fig)
        print('Plotted',args.build,family,len(runs),'audited runs')
    lines=['# Figures from terminal audited artifacts','',
        'Pooled confusion is illustration; headline/per-class F1 are mean per-client metrics. Partial checkpoints retain round labels.','']
    for path in sorted(base.glob('convergence-*.png'))+sorted(base.glob('classes-*.png')):
        lines.extend([f'![{path.stem}]({path.name})',f'[SVG export]({path.with_suffix(".svg").name})',''])
    (base/'figures.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__': main()
