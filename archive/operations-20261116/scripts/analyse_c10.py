"""Read audited C10 own-test CMs; evaluate prelocked r10 gates and report bytes."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.metrics import TEN, ten
from scripts.research_metrics import client_f1
from scripts.analyse_m20 import conditional_accuracy


def main():
    out = ROOT / 'runs/c10-study'
    audit = json.loads((out/'audit.json').read_text())
    counts = np.load(ROOT/'evidence/research-data/train_counts.npy')
    small = counts.sum(1) < 256
    cms, curves, results = {}, {}, {}
    for a in audit:
        assert a['status'] == 'AUDIT_PASS'
        d = ROOT/'artifacts/c10'/a['slug']/a['slug']
        curves[a['arm']] = []
        for r in range(1, 11):
            cm = np.load(d/f'cms_r{r:03d}.npz')['private']
            f = client_f1(cm)
            pooled = ten(cm.sum(0))
            row = dict(round=r, **{k: pooled[k] for k in TEN},
                       p10=float(np.percentile(f,10)), low256=float(f[small].mean()),
                       mean=float(f.mean()), timeDelay=pooled['f1_per_class'][13],
                       absent=conditional_accuracy(cm[small], counts[small])['absent']['accuracy'],
                       rare=conditional_accuracy(cm[small], counts[small])['rare1_16']['accuracy'])
            curves[a['arm']].append(row)
        cms[a['arm']] = cm
        boot = a['bootstrap_codec']
        saving = boot['cdf_bytes']-boot['cdf_zlib_bytes']
        results[a['arm']] = dict(cell=a['cell'], endpoint=row, f1_per_class=pooled['f1_per_class'],
                                 conditional_all=conditional_accuracy(cm,counts),
                                 conditional_small=conditional_accuracy(cm[small],counts[small]),
                                 bytes=int(a['total_bytes_10']),
                                 dense_secagg_bytes=int(a['total_bytes_10']+(1655872-271031 if boot['bins']==256 else 0)),
                                 lossless_cdf_saving=saving, lossless_cdf_projected_bytes=int(a['total_bytes_10']-saving),
                                 seconds=sum(x['t_train']+x['t_eval'] for x in a['rounds']),
                                 skip_max=max(x['skipped_frac'] for x in a['rounds']))
    thresholds = dict(f1_macro=.001,p10=.006,low256=.003,timeDelay=.0035,absent=.01)
    base=results['C0']
    for arm,v in results.items():
        delta={k:v['endpoint'][k]-base['endpoint'][k] for k in thresholds}
        fdelta=client_f1(cms[arm])-client_f1(cms['C0'])
        v.update(delta=delta, gates={k:bool(delta[k]>=-thresholds[k]) for k in thresholds},
                 pass_screen=all(delta[k]>=-thresholds[k] for k in thresholds),
                 paired_clients=dict(down=int((fdelta<0).sum()),down_gt05=int((fdelta<-.05).sum()),
                                     up_gt05=int((fdelta>.05).sum()),min=float(fdelta.min()),
                                     p10=float(np.percentile(fdelta,10)),median=float(np.median(fdelta))))
    report=dict(arms=results,curves=curves,audit_sha256=hashlib.sha256((out/'audit.json').read_bytes()).hexdigest())
    (out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# C10 — nén Core, seed42, 10 round','',
           '04/11/2026: 4/4 kernel đủ10round; audit lại source/CM/teacher codec PASS. Cosine50,5343xe,DAGSNetW1,floor16.',
           'Thiết kế/gate khóa trước: [screening-c10](screening-c10.md). Artifact: `runs/c10-study/{audit,analysis}.json`.',
           '', '## Kết luận screening', '',
           'C1 bootstrap64 PASS: −22,9% tổng byte r10, pooled/p10/low256 tăng nhỏ. C2 Q8 và C3combo FAIL low256/absent gate; không đưa Q8 vào Core.',
           'Ưu tiên triển khai lossless CDF256 trước: Core dự kiến1011319→647914B/xe/r10 (−35,9%), không đổi tensor. Đây là measured codec + projected accounting, chưa chạy fullnetwork qua đường decode. C1 vẫn là ứng viên lossy, chưa thay mặc định50round.',
           'Không sụp không có nghĩa giữ mọi cohort: Q8 làm absentaccuracy xe<256 giảm4,18điểm phầntrăm; combo giảm3,15điểm.',
           '', '## Performance r10', '',
           '| ô | pooled F1 | p10 | low256 mean F1 | timeDelay F1 | absent accuracy xe<256 | gate |',
           '|---|---:|---:|---:|---:|---:|---|']
    for a,v in results.items():
        e=v['endpoint']
        lines.append(f"| {a} {v['cell']} | {e['f1_macro']:.6f} | {e['p10']:.6f} | {e['low256']:.6f} | {e['timeDelay']:.6f} | {e['absent']:.6f} | {'PASS' if v['pass_screen'] else 'FAIL'} |")
    lines+=['','## Đủ10metric pooled r10','', '| metric | C0 | C1 | C2 | C3 |','|---|---:|---:|---:|---:|']
    for k in TEN:
        lines.append('| '+k+' | '+' | '.join(f"{results[a]['endpoint'][k]:.6f}" for a in results)+' |')
    lines+=['','## Byte hai chiều/xe/r10, gồm bootstrap','', '| ô | thường | đường SecAgg dày | thêm lossless CDF (ước tính thường) |','|---|---:|---:|---:|']
    for a,v in results.items():
        lines.append(f"| {a} | {v['bytes']:,} | {v['dense_secagg_bytes']:,} | {v['lossless_cdf_projected_bytes']:,} |")
    lines+=['','CDF toàn mạng256bin:630784→267379B (zlib6);64bin:157696→75976B. Hai rank cùng hash/byte, roundtripPASS.',
            'Lossless đang là phép đo bổ sung trong kernel; tensor train không đi qua compressed CDF. Cột cuối là accounting triển khai dự kiến, chưa transport/header/ACK/crypto. Giữ mean/Cholesky fp16, prior và uplink hiện hành.',
            'Bootstrap64 tính uplink dày bảo thủ; chưa đo sparse64. SecAgg là accounting theo vector dày của proposal; chưa fixed-point/overflow/crypto end-to-end.',
            '', '## Delta và phân phối client', '', '| ô−Core | Δpooled | Δp10 | Δlow256 | ΔtimeDelay | Δabsent | xe giảm | giảm>.05 | tăng>.05 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a,v in results.items():
        d=v['delta'];p=v['paired_clients']
        lines.append(f"| {a} | "+' | '.join(f'{d[k]:+.6f}' for k in thresholds)+f" | {p['down']} | {p['down_gt05']} | {p['up_gt05']} |")
    lines+=['','Các count client là mô tả một seed; không dùng làm causalcount/significance. Rare=accuracy trên lớp có1–16trainrow, nhómxe<256; đầy đủ trong analysis.json.',
            '', '## Rare và thời gian', '', '| ô | rare accuracy xe<256 | train+eval10round (s) | max skip fraction |','|---|---:|---:|---:|']
    for a,v in results.items():
        lines.append(f"| {a} | {v['endpoint']['rare']:.6f} | {v['seconds']:.1f} | {v['skip_max']:.6f} |")
    lines+=['', 'Thời gian toàn2T4, không phải latency xe; chưa gồm setup/compile/artifactI/O. Không suy speedup từ một session.', '', '## F1 từng lớp r10','', '| lớp | C0 | C1 | C2 | C3 |','|---|---:|---:|---:|---:|']
    names=json.loads((ROOT/'knowledge/label_mapping.json').read_text())['id_to_label']
    for i in range(16):
        lines.append('| '+names[str(i)]+' | '+' | '.join(f"{results[a]['f1_per_class'][i]:.6f}" for a in results)+' |')
    lines+=['','## Giới hạn bằng chứng','',
            'Audit40CM→10metric/40snapshotnhỏ/40Q8packet (2ô×2rank×10round), source overlay/hash, teacherlag/chain và bootstrapmetadata. Snapshot nhỏ thiếu privateweights toàn mạng; chưaprediction→CM replay, chưa tái dựng streamuplink aggregate, chưa kiểm mọi tensor private/BN từ fullresume.',
            'Skip được kiểm từ counters theo round; finite snapshot không chứng minh toàn bộ privatebank. Startup đã kiểm2T4/compiled cả2rank. Seed42/r10/officialtest đã đọc nhiều lần: screening, không xác nhận tương đương r50 hoặc mọi xe giữ chất lượng.',
            'Không launch thêm, không kéo dài hoặc tinh chỉnh từ officialtest.']
    (ROOT/'docs/research/results-c10.md').write_text('\n'.join(lines)+'\n')
    for a,v in results.items(): print(a, v['endpoint'], v['delta'],v['gates'],v['bytes'],v['lossless_cdf_projected_bytes'])

if __name__=='__main__':main()
