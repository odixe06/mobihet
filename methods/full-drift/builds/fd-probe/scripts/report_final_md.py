"""Write docs/methods/full-drift/results/final/report.md from the audited CSVs of report_final.py (no hand-typed run numbers).
    python methods/full-drift/builds/fd-probe/scripts/report_final.py && python methods/full-drift/builds/fd-probe/scripts/report_final_md.py
"""
import csv
import json
import statistics
from collections import defaultdict

from collect_fd import ROOT
from mh.metrics import METRIC_KEYS

D = ROOT / 'docs/methods/full-drift/results/final'
METHOD, CONTROL = 'fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl', 'fd_dml_m4b_gpool_frz_gkd'
NAME = {METHOD: 'q3_zl la_kdpx', CONTROL: 'gkd'}
CHAIN = {(20, METHOD): 'za', (20, CONTROL): 'zc', (50, METHOD): 'ze', (50, CONTROL): 'zg', (100, METHOD): 'zi', (100, CONTROL): 'zk'}
KS = (20, 50, 100)
VIEWS = [('private', 'head'), ('private', 'gp'), ('private', 'gpe'), ('private_ema', 'head'), ('private_ema', 'gp'), ('proxy', 'head')]
SHORT = ['acc', 'P mac', 'P mic', 'P wtd', 'R mac', 'R mic', 'R wtd', 'F1 mac', 'F1 mic', 'F1 wtd']
# fl_dagsnet_veremi commit f9014e1, round 50 (F1 macro, accuracy) for 20 / 50 / 100 clients: report/<method>/report.md;
# TinyProto accuracy from tinyproto/papers/tinyproto-lee-2026/report/mean_over_clients_<K>client.csv
BASELINES = [('AFPHA', 'toàn cục', [(.7752, .7487), (.7162, .7116), (.6949, .6967)]),
             ('FD-IDS', 'toàn cục', [(.7655, .7327), (.6815, .6714), (.6606, .6552)]),
             ('LwFedNIDS', 'toàn cục', [(.7917, .7791), (.7526, .7360), (.7557, .7339)]),
             ('NILM-FL proxy w̄_r', 'toàn cục', [(.8228, .7902), (.7923, .7742), (.7568, .7510)]),
             ('NILM-FL w_s', 'TB client', [(.6963, .6805), (.6543, .6602), (.6111, .6289)]),
             ('PerFed-SKD ω_m', 'TB client', [(.6504, .6229), (.6901, .6619), (.6541, .6354)]),
             ('pFedES', 'TB client', [(.4856, .5276), (.3652, .4335), (.2900, .3692)]),
             ('TinyProto (luật proto)', 'TB client', [(.6212, .5639), (.4018, .4062), (.3435, .3416)])]


def f(x, n=4): return '—' if x is None else f'{x:.{n}f}'.replace('.', ',')
def sd(x): return '—' if x is None else f'{x:+.4f}'.replace('.', ',')
def vn(s): return s.replace(',', 'X').replace('.', ',').replace('X', '.')   # 1.234,56 as in the other docs
def mb(x): return vn(f'{x / 1e6:,.2f}')


def main():
    m = {}
    for r in csv.DictReader((D / 'metrics.csv').open()):
        m[(int(r['clients']), r['arm'], r['role'], r['rule'], int(r['round']))] = {k: float(r[k]) for k in (*METRIC_KEYS, 'f1_macro_client_min', 'benign_f1')}
    costs = defaultdict(list)
    for r in csv.DictReader((D / 'costs.csv').open()): costs[(int(r['clients']), r['arm'])].append(r)
    class_rows=list(csv.DictReader((D/'per-class-r50.csv').open()))
    done = {(s['clients'], s['arm']): s for s in json.loads((D / 'completion.json').read_text())}
    get = lambda k, arm, role, rule, rnd, key='f1_macro': m.get((k, arm, role, rule, rnd), {}).get(key)
    last = {(k, arm): max([rnd for (kk, a, ro, ru, rnd) in m if (kk, a, ro, ru) == (k, arm, 'private', 'head')], default=None)
            for k in KS for arm in (METHOD, CONTROL)}
    out = []
    w = out.append
    complete = all(done.get((k, a), {}).get('evaluation_complete') for k in KS for a in (METHOD, CONTROL))

    w('# Báo cáo run cuối FullDrift seed 42 — 20 / 50 / 100 client × 50 round\n')
    w('Sinh tự động bởi `methods/full-drift/builds/fd-probe/scripts/report_final_md.py` từ [`metrics.csv`](metrics.csv), [`costs.csv`](costs.csv),')
    w('[`completion.json`](completion.json) (do `report_final.py` ghép từ artifact đã audit: `AUDITED.json`, sha weights từng round, eval từng round).')
    w('Kernel/version, source SHA, audit SHA và round thực sự dùng: [`provenance.json`](provenance.json); '
      'từng đoạn train/chấm: [`inventory.json`](inventory.json).')
    w('Không số nào của run gõ tay; số repo đối chiếu trích nguồn ở §3.\n')
    status = ' · '.join(f"{CHAIN[(k, a)]} ({NAME[a]}, {k}c): train {len(done[(k, a)]['training_rounds'])}/50, chấm {len(done[(k, a)]['evaluated_rounds'])}/50"
                        for k in KS for a in (METHOD, CONTROL) if (k, a) in done)
    w(f"**Trạng thái:** {'đủ 6 chuỗi train + chấm r1–50 đã audit' if complete else 'CHƯA ĐỦ (ô thiếu ghi —)'}. {status}.\n")

    w('## 0. Tóm tắt — round 50 (headline theo protocol, không chọn round)\n')
    w('F1 macro = trung bình trên client của F1 macro từng model client, mỗi model chấm **toàn bộ** test20 (10.761.343 dòng); '
      'client tệ nhất = F1 macro thấp nhất trong K model. Proxy là một model chung (mọi client như nhau).\n')
    w('| K | chuỗi | arm | private head F1 mac | client tệ nhất | benign F1 | accuracy | private_ema head F1 mac | proxy F1 mac | byte/client/round (trung vị) |')
    w('|---:|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for k in KS:
        for a in (METHOD, CONTROL):
            r = last[(k, a)] if last[(k, a)] == 50 else None
            pc = [float(c['per_client_round_bytes_mean']) for c in costs[(k, a)]]
            w(f"| {k} | {CHAIN[(k, a)]} | {NAME[a]} | **{f(get(k, a, 'private', 'head', r))}** | {f(get(k, a, 'private', 'head', r, 'f1_macro_client_min'))} | "
              f"{f(get(k, a, 'private', 'head', r, 'benign_f1'))} | {f(get(k, a, 'private', 'head', r, 'accuracy'))} | {f(get(k, a, 'private_ema', 'head', r))} | "
              f"{f(get(k, a, 'proxy', 'head', r))} | {mb(statistics.median(pc)) + ' MB' if pc else '—'} |")
    w('')
    deltas = []
    for k in KS:
        d = [get(k, METHOD, ro, ru, 50) is not None and get(k, CONTROL, ro, ru, 50) is not None and get(k, METHOD, ro, ru, 50) - get(k, CONTROL, ro, ru, 50)
             for ro, ru in (('private', 'head'), ('private_ema', 'head'), ('proxy', 'head'))]
        worst = [get(k, a, 'private', 'head', 50, 'f1_macro_client_min') for a in (METHOD, CONTROL)]
        deltas.append(f"K{k}: private {sd(d[0]) if d[0] is not False else '—'}, private_ema {sd(d[1]) if d[1] is not False else '—'}, "
                      f"proxy {sd(d[2]) if d[2] is not False else '—'}, client tệ nhất {sd(worst[0] - worst[1]) if None not in worst else '—'}")
    w('- **Chênh q3_zl la_kdpx − gkd ở r50** (F1 macro): ' + '; '.join(deltas) + '.')
    ratio = [statistics.median(float(c['per_client_round_bytes_mean']) for c in costs[(k, METHOD)]) /
             statistics.median(float(c['per_client_round_bytes_mean']) for c in costs[(k, CONTROL)]) for k in KS if costs[(k, METHOD)] and costs[(k, CONTROL)]]
    total = lambda k, a: sum(float(c['per_client_round_bytes_mean']) for c in costs[(k, a)]) + max(float(c['bootstrap_bytes_session']) for c in costs[(k, a)]) / k
    w(f"- **Byte mỗi client mỗi round:** q3_zl la_kdpx = {', '.join(f'{x:.3f}'.replace('.', ',') + '×' for x in ratio)} gkd (dense fp32) ở K = 20 / 50 / 100; "
      f"tổng 50 round gồm bootstrap một lần: {', '.join(f'{mb(total(k, METHOD))} vs {mb(total(k, CONTROL))} MB' for k in KS if costs[(k, METHOD)] and costs[(k, CONTROL)])}.")
    w('- Một seed (42), chưa có khoảng tin cậy đa seed ở horizon50; mức lệch ~0,008 từng đo ở screening không phải ngưỡng kiểm định thống kê. '
      'Test20 chấm mọi round chỉ để báo cáo; không chọn siêu tham số/round bằng test. '
      'Contrast là recipe đầy đủ so gkd dense, không cô lập riêng tác dụng lượng tử hóa.\n')

    w('## 1. Thiết lập\n')
    w('- Dữ liệu VeReMi NextGen, train80 = 43.045.415 dòng (toàn bộ, không holdout), test20 = 10.761.343 dòng; partition Dirichlet α 0,5 cố định của '
      '`veremi-fl-20/50/100client` (cùng `fl_dagsnet_veremi`); batch 512 / 512 / 256; DAGSNet mọi model.')
    w('- 50 round × 1 epoch local trên toàn bộ dữ liệu client, 1 aggregation/round; LR schedule khóa theo tổng 50 round; seed 42.')
    w(f'- `{METHOD}` = phương pháp chốt (gkd + la + kdpx, proxy trao đổi 3-bit + LZMA2); `{CONTROL}` = đối chứng gkd (proxy fp32 dense). '
      '[Thiết kế](../../design.md), [kết quả screening](../../results.md).')
    w('- Model chấm mỗi round: private (K model, luật **head** = luật chính; gp/gpe phụ), private_ema (EMA của private), proxy (model chung).')
    w('- Train và chấm tách: notebook train ghi weights mọi round (sha nối chuỗi); notebook chấm `fd_eval_fast.py` (lớp tolerance owner duyệt, khớp '
      '99,96 % nhãn với bộ chấm gốc; gate tương đối ≤ 2 % max|logit|) — [pipeline](../../../../operations/final-pipeline.md).\n')
    w('- Caveat dữ liệu: scaler đã fit toàn train trước holdout screening; split thời gian, receiver/flow có thể trải nhiều client; '
      'benign lấy từ luồng không tấn công, Sybil có shortcut session, dòng nhập nhằng bị loại. '
      'Không suy các score thành hiệu năng triển khai xe độc lập — [dataset](../../../../../knowledge/dataset.md).\n')

    w('## 2. Round 50 — đủ 10 metric\n')
    w('Cột: ' + ', '.join(f'{s} = {k}' for s, k in zip(SHORT, METRIC_KEYS)) + '; "tệ nhất" = F1 macro client thấp nhất; benign = F1 lớp benign.\n')
    for k in KS:
        w(f'### {k} client\n')
        w('| arm | model / luật | ' + ' | '.join(SHORT) + ' | tệ nhất | benign |')
        w('|---|---|' + '---:|' * 12)
        for a in (METHOD, CONTROL):
            for ro, ru in VIEWS:
                v = m.get((k, a, ro, ru, 50))
                cells = [f(v[x]) if v else '—' for x in (*METRIC_KEYS, 'f1_macro_client_min', 'benign_f1')]
                if ro == 'proxy': cells[10] = '(1 model)'
                w(f'| {NAME[a]} | {ro} / {ru} | ' + ' | '.join(cells) + ' |')
        w('')

    w('### Theo lớp — round50, luật head\n')
    w('F1 từng lớp = trung bình F1 lớp đó trên K model; proxy là một model chung. '
      'Precision/recall/F1, support và mọi luật khác: [`per-class-r50.csv`](per-class-r50.csv).\n')
    lookup={(int(r['clients']),r['arm'],r['role'],r['rule'],int(r['class_id'])):float(r['f1']) for r in class_rows}
    class_names=json.loads((ROOT/'knowledge/meta.json').read_text())['class_names']
    for k in KS:
        w(f'**{k} client**\n')
        w('| lớp | private q3_zl la_kdpx | private gkd | Δ | proxy q3_zl la_kdpx | proxy gkd |')
        w('|---|---:|---:|---:|---:|---:|')
        for j,name in enumerate(class_names):
            a,b=lookup.get((k,METHOD,'private','head',j)),lookup.get((k,CONTROL,'private','head',j))
            cells=[f(a),f(b),sd(a-b) if None not in (a,b) else '—',
                   f(lookup.get((k,METHOD,'proxy','head',j))),f(lookup.get((k,CONTROL,'proxy','head',j)))]
            w(f'| {name} | '+' | '.join(cells)+' |')
        w('')

    w('## 3. So sánh với `fl_dagsnet_veremi` (round 50)\n')
    w('Repo [`odixe06/fl_dagsnet_veremi`](https://github.com/odixe06/fl_dagsnet_veremi) commit `f9014e1`, số đọc từ `report/<method>/report.md` '
      '(TinyProto accuracy từ `mean_over_clients_<K>client.csv`). Cùng dữ liệu/partition/test20, 50 round, chấm mọi round, headline r50, một seed. '
      '"TB client" = trung bình F1 macro các model client (cùng cách tính với private ở đây); "toàn cục" = một model.\n')
    w('| phương pháp | model | 20c F1 mac | 20c acc | 50c F1 mac | 50c acc | 100c F1 mac | 100c acc |')
    w('|---|---|---:|---:|---:|---:|---:|---:|')
    for a in (METHOD, CONTROL):
        for ro, label in (('private', 'TB client'), ('proxy', 'toàn cục')):
            w(f'| **MobiHet {NAME[a]}** {ro} | {label} | ' + ' | '.join(f"{f(get(k, a, ro, 'head', 50))} | {f(get(k, a, ro, 'head', 50, 'accuracy'))}" for k in KS) + ' |')
    for name, kind, vals in BASELINES:
        w(f'| {name} | {kind} | ' + ' | '.join(f'{f(x)} | {f(y)}' for x, y in vals) + ' |')
    w('\nByte mỗi client mỗi round của repo kia (tính từ giao thức, fp32): FedAvg/FedProx/NILM-FL 3,19 MB, pFedES 3,26 MB, PerFed-SKD ≈0,61 MB (20c), '
      'LwFedNIDS 0,30 MB, TinyProto 6,4 kB ([so sánh 20c](../../../../research/comparison-fl-dagsnet-veremi.md)). MobiHet đo trên dây (§5).\n')

    w('## 4. Diễn biến 50 round — F1 macro (test20)\n')
    w('Mô tả, không dùng chọn round. TB r41–50 = trung bình 10 round cuối.\n')
    w('![Quỹ đạo private và proxy trên toàn test20](learning-curves.svg)\n')
    w('| K | arm | private head r50 | TB r41–50 | private_ema head r50 | TB r41–50 | proxy r50 | TB r41–50 |')
    w('|---:|---|---:|---:|---:|---:|---:|---:|')
    for k in KS:
        for a in (METHOD, CONTROL):
            cells = []
            for ro in ('private', 'private_ema', 'proxy'):
                tail = [get(k, a, ro, 'head', r) for r in range(41, 51)]
                cells += [f(get(k, a, ro, 'head', 50)), f(statistics.mean(tail)) if None not in tail else '—']
            w(f'| {k} | {NAME[a]} | ' + ' | '.join(cells) + ' |')
    w('')
    w('| K | arm | private Δ r20→30 | private Δ r30→50 | proxy Δ r20→30 | proxy Δ r30→50 | proxy − private r50 |')
    w('|---:|---|---:|---:|---:|---:|---:|')
    for k in KS:
        for a in (METHOD, CONTROL):
            cells=[]
            for ro in ('private','proxy'):
                for left,right in ((20,30),(30,50)):
                    lo,hi=get(k,a,ro,'head',left),get(k,a,ro,'head',right)
                    cells.append(sd(hi-lo) if None not in (lo,hi) else '—')
            lo,hi=get(k,a,'private','head',50),get(k,a,'proxy','head',50)
            cells.append(sd(hi-lo) if None not in (lo,hi) else '—')
            w(f'| {k} | {NAME[a]} | '+' | '.join(cells)+' |')
    w('\nChênh tốc độ tăng và gap chỉ mô tả quỹ đạo; không xác định nguyên nhân, capacity, LR hoặc hiệu quả bộ sinh.\n')
    for k in KS:
        w(f'### {k} client — từng round\n')
        w(f'| round | private {NAME[METHOD]} | private {NAME[CONTROL]} | Δ | tệ nhất {NAME[METHOD]} | tệ nhất {NAME[CONTROL]} | '
          f'ema {NAME[METHOD]} | ema {NAME[CONTROL]} | proxy {NAME[METHOD]} | proxy {NAME[CONTROL]} |')
        w('|---:|' + '---:|' * 9)
        for r in range(1, 51):
            a, b = get(k, METHOD, 'private', 'head', r), get(k, CONTROL, 'private', 'head', r)
            w(f"| {r} | {f(a)} | {f(b)} | {sd(a - b) if None not in (a, b) else '—'} | {f(get(k, METHOD, 'private', 'head', r, 'f1_macro_client_min'))} | "
              f"{f(get(k, CONTROL, 'private', 'head', r, 'f1_macro_client_min'))} | {f(get(k, METHOD, 'private_ema', 'head', r))} | "
              f"{f(get(k, CONTROL, 'private_ema', 'head', r))} | {f(get(k, METHOD, 'proxy', 'head', r))} | {f(get(k, CONTROL, 'proxy', 'head', r))} |")
        w('')

    w('## 5. Chi phí train và truyền tin\n')
    w('Byte đo trên dây (gói thật, gồm header), lên + xuống, mỗi client. Bootstrap = gửi proxy ban đầu + moment Gaussian một lần (round 0). '
      'Giây train = round đã commit trên 2×T4 (không gồm chấm, chấm chạy notebook riêng).\n')
    w('| K | arm | byte/client/round trung vị (min–max round) | bootstrap/client | tổng byte/client 50 round | tổng byte K client | s/round trung vị | Σ giờ train | bước local/round | bước AMP bỏ qua (50 round) |')
    w('|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for k in KS:
        for a in (METHOD, CONTROL):
            c = sorted(costs[(k, a)], key=lambda x: int(x['round']))
            if not c: continue
            pc = [float(x['per_client_round_bytes_mean']) for x in c]
            boot = max(float(x['bootstrap_bytes_session']) for x in c) / k   # round 0 of the first session, once per chain
            total = sum(pc) + boot
            sec = [float(x['committed_round_seconds']) for x in c]
            steps, skipped = int(c[0]['local_steps']), sum(int(x['skipped_steps']) for x in c)
            w(f"| {k} | {NAME[a]} | {mb(statistics.median(pc))} MB ({mb(min(pc))}–{mb(max(pc))}) | {mb(boot)} MB | {mb(total)} MB | {mb(total * k)} MB | "
              f"{statistics.median(sec):.0f} | {f(sum(sec) / 3600, 1)} | {vn(f'{steps:,}')} | {vn(f'{skipped:,}')} |")
    w('')

    w('**Phạm vi R1:** owner22/10 khóa ≤231.000 B/contact, tính cả bootstrap. Bảng dưới là byte lên+xuống theo round, '
      'chưa phải lịch contact/deadline đã mô phỏng; tổng vượt cap nếu gộp vào một contact. Nén giảm byte so dense chưa chứng minh R1 đạt.\n')
    w('| K | arm | peak byte/client/round (max trên client và round) | bootstrap/client (trung bình) | cap/contact |')
    w('|---:|---|---:|---:|---:|')
    for k in KS:
        for a in (METHOD,CONTROL):
            c=costs[(k,a)]
            if not c:continue
            peak=max(int(x['per_client_round_bytes_max']) for x in c)
            boot=max(float(x['bootstrap_bytes_session']) for x in c)/k
            w(f'| {k} | {NAME[a]} | {vn(f"{peak:,}")} B | {vn(f"{boot:,.0f}")} B | 231.000 B |')
    w('\nR2 mixed architectures chưa thực nghiệm; mọi private/proxy trong báo cáo là DAGSNet. '
      'Các probe synthetic22/10 là nghiên cứu riêng, không ghép gain offline vào run cuối này.\n')

    titles = {('private', 'head'): 'A — private / head (luật chính)', ('private_ema', 'head'): 'B — private_ema / head', ('proxy', 'head'): 'C — proxy'}
    for (ro, ru), title in titles.items():
        w(f'## Phụ lục {title}: đủ 10 metric mỗi round\n')
        for k in KS:
            for a in (METHOD, CONTROL):
                w(f'<details><summary>{k} client — {CHAIN[(k, a)]} {NAME[a]}</summary>\n')
                w('| round | ' + ' | '.join(SHORT) + (' | tệ nhất | benign |' if ro != 'proxy' else ' | benign |'))
                w('|---:|' + '---:|' * (12 if ro != 'proxy' else 11))
                for r in range(1, 51):
                    v = m.get((k, a, ro, ru, r))
                    keys = (*METRIC_KEYS, 'f1_macro_client_min', 'benign_f1') if ro != 'proxy' else (*METRIC_KEYS, 'benign_f1')
                    w(f'| {r} | ' + ' | '.join(f(v[x]) if v else '—' for x in keys) + ' |')
                w('\n</details>\n')
    w('Mọi luật/model khác (private gp/gpe, private_ema gp) đủ 10 metric từng round: [`metrics.csv`](metrics.csv).')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),sharey=True)
    for ax,k in zip(axes,KS):
        for a,color in ((METHOD,'#1769aa'),(CONTROL,'#d56719')):
            for role,style in (('private','-'),('proxy','--')):
                pairs=[(r,get(k,a,role,'head',r)) for r in range(1,51) if get(k,a,role,'head',r) is not None]
                if pairs:ax.plot(*zip(*pairs),color=color,linestyle=style,linewidth=1.7,label=f'{NAME[a]} / {role}')
        ax.set_title(f'{k} clients');ax.set_xlabel('Round');ax.set_xlim(1,50);ax.grid(alpha=.2)
    axes[0].set_ylabel('Macro F1 on test20')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.14,1,1))
    for extension in ('svg','png'):fig.savefig(D/f'learning-curves.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    (D / 'report.md').write_text('\n'.join(out) + '\n')
    print('wrote', D / 'report.md', 'complete' if complete else 'INCOMPLETE')


if __name__ == '__main__': main()
