"""Rebuild M20 report, metric appendix and figure from retained audited evidence. Run at repo root."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core.metrics import TEN
base=Path('evidence/m20-summary');a=json.loads((base/'analysis.json').read_text());arms=a['arms'];audit=json.loads((base/'audit.json').read_text());inv=json.loads((base/'collection.json').read_text())
assert len(audit)==16 and all(len(x['rounds'])==20 for x in audit)
names=['augmentation','softQ/noReal','softQ/realgate','fallback','postepochQ','server sqrt(n)','uniform mixture','local support mixture']
lines=['# M20 — kết quả và chẩn đoán, 01/11/2026','',
'**16/16 run version1 hoàn tất và audit đủ20round.** Own-test từng xe là chính; chưa chốt phương pháp tổng.',
'Full5343xe, cosine50, seed42/43, DAGSNetW1; [thiết kế đãkhóa](screening-m20.md), [cơ chế](method.md).','',
'## Bằng chứng và phạm vi','',
f"Pull1360file/{sum(x['bytes'] for x in inv):,}B ({sum(x['bytes'] for x in inv)/2**30:.3f}GiB),56artifactđích/run. "
'Source/config/LR/IDs/support/2×T4/compile2rank,320testCM,96trainCM,3264shadowprobe và320snapshot PASS. '
'Qlag/fp16codec/synthetic-supportweight và externalpoolhash được đối chiếu từartifact. '
'Không fullprivatebank/predictionreplay/exactresume hoặc fullcontentfingerprintdataset.',
'Bản làm việc nhẹ tại `evidence/m20`, `evidence/m20-summary`, `evidence/research-data`; snapshotđầyđủ khôi phục theo [backup](../operations/backup.md).','',
'## Own-test tại round20','',
'| cell | F1 seed42/43 | p10 seed42/43 | meanF1 xe<256 seed42/43 |','|---|---|---|---|']
for j,name in enumerate(names):
 vals=[arms[f'M{j*2+k}']['final'] for k in [0,1]]
 pair=lambda key:'/'.join(f'{x[key]:.6f}' for x in vals)
 lines.append(f"| {name} | {pair('f1')} | {pair('p10')} | {pair('low256_mean')} |")
lines += ['', 'M0–M15 theo thứtự bảng, hai seed/cell. Full10metric mọi round và16lớp cuối ở [phụ lục](metrics-m20.md).',
'Accuracy=precision_micro=recall_micro=recall_weighted=f1_micro trong single-labelclassification; giữ đủ10cột.',
'', '## Ba khâu: điều đã đo được','',
'1. **Upload:** postepochQ so onlinegate ΔF1=−.000156/−.000104. Forwardcuốiepoch không tự tạo teacher hữuích hơn; '
'đây là can thiệp timing+BN/dropout gộp, không kếtluận riêng staleness. Có thêm localforwardtime.',
'2. **Server:** sqrt(n_kc) so n_kc ΔF1=+.001482/+.001158, nhưng meanΔxe<256=+.000276/−.000778. '
'Tăng pooled chưa giải quyết yêu cầu xe nhỏ; không gọi thêm đại diện là đủ.',
'3. **Client:** softQ/noReal vẫn hơnaugmentation+.001559/+.001624. Realgate đổi Δpooled=−.000046/−.000582 '
'nhưng tăngtail/meanxe. Localρ so uniformpooled=−.000692/+.000429, p10 vàmeanxe giảm cảhai; '
'riêng meanxe<256 tăngrất nhỏ+.000684/+.000461. Không chốt count-onlypersonalization thắng.',
'', '## Học tiếp và cơ chế đổi gate','',
'Gatecontrol tăngF1r15→20 +.003102/+.003058; noReal +.002834/+.002796. Fallback chỉ+.000179/−.000213. '
'Không bão hòa đồngloạt; mức tăng chậm vàranking phụthuộc endpoint. Gatexe<256 từkémaugmentationr10 sanghơn '
'khoảng+.00567/+.00555r20. Không chọn best-round.', '',
'![Own-test learning and low-data cohorts](figures/m20-curves.png)', '',
'Qargmaxđúng chưa nghĩa teacher mạnh: tạiQr13, timeDelaytrueprob≈.49 vàbenign≈.47. '
'Từ r14 bật lại realKD và đổi synthetic hard→soft cùnglúc; F1timeDelay giảm≈.07 ở cảhai seed. '
'Qr14 sai→r15hardfallback phục hồi; Qr15đúng→r16giảm lại. Bảng dùng đúngteacherQr−1, '
'không lấyQr đểgiải thích trainingr. Đây là bằngchứngtheothờigian, chưa tách nhân quả hai kênh.', '',
'| seed | round train | teacher round | Q13,13 | Q13,0 | gate giữ13 | timeDelay F1 |',
'|---|---:|---:|---:|---:|---|---:|']
z=np.load(base/'teacher-tables.npz')
for seed,k in [(42,'M6'),(43,'M7')]:
 for r in range(13,18):
  q=z[k][r-2];f=arms[k]['curve'][r-1]['timeDelay_f1']
  lines.append(f"| {seed} | {r} | {r-1} | {q[13,13]:.5f} | {q[13,0]:.5f} | {q[13].argmax()==13} | {f:.5f} |")
lines += ['', 'Fixedouterweight1 không cố định gradienttươngđối: ở một số xe probe r20, syntheticgradient lớn hơn realCE '
'nhiều lần khi CE đãnhỏ. Đây chỉshadow34xe, không actualAMPgradient/clippingfraction; chưa là nguyênnhân duy nhất.', '',
'## Non-IID, thiếu lớp và tổng quát hóa','',
'Nhóm848xe<256 có1555testrows absent-local và9776rows classcount1–16. Seed42, accuracytrên absentrows '
'augmentation=.5267, gate=.4778, localρ=.4656; gate cải thiện tốt hơn trên lớp có nhiều mẫu. '
'MeanF1xe tăng chưa bảo đảm nhận diện tốt lớp chưa gặp. Đây là accuracy điều kiện theo trueclass, không macroF1 mới.',
'Cross-client34xe chỉchẩn đoán phụ: augmentationmean≈.805 vs gate≈.785–.788, cùngpool/hashes/support. '
'Không dùng để chọnHP hoặc thay own-test; dữ liệu này không được cấp cho training/teacher/kênhFL.', '',
'## Byte và thời gian','',
'Q1088B/xe/r +3311744Bbootstrap =3333504B/xe/20r; augmentation3311744B. '
'Bootstrap chi phối; chưa end-to-endtransport/ACK/secureaggregation proof. Giây làwalljob2×T4, khônglatencyxe. '
'Chênh account không là causal speedup; postepochforward nằm trongt_train, không cộnghai lần.', '',
'| cell | train s/r seed42/43 (r2–20) | diagnostic tổng s seed42/43 |', '|---|---:|---:|']
for j,name in enumerate(names):
 vals=[arms[f'M{j*2+k}'] for k in [0,1]]
 lines.append('| '+name+' | '+'/'.join(f"{v['train_seconds_mean_r2_20']:.1f}" for v in vals)+' | '+'/'.join(f"{v['diagnostic_seconds_total']:.1f}" for v in vals)+' |')
lines += ['', '## Kết luận có giới hạn và bước tiếp','',
'Hai seed vànhiềubatchđãđọc officialtest; không significance hoặc r>20/mixedarchitecture/contactdrop proof. '
'Train+valđãfit, không independentvalidation. Không rawsharing nhưng chưa formalDP/secureaggregation.',
'Ưu tiên đối chứng tách chuyểnđổi realKD/synthetic, sauđó tách targetmixture khỏi realKDweight. '
'Kiểm nội dung offdiagonalQ cóutility hơn chỉlabel-smoothing trước khi tăngpacket hoặc đổigenerator. '
'Chi tiết [diagnosis](diagnosis.md); backup/cleanup xong mới khóa batch tiếp. Chưa launchthêm.']
Path('docs/research/results-m20.md').write_text('\n'.join(lines)+'\n')
appendix=['# M20 — đủ10metric mọi round và16lớp cuối','', 'Nguồn: audit.json đãkiểmCM; endpoint20, không best-round.']
for x in audit:
 appendix += ['',f"## {x['arm']} — seed{x['hp']['seed']}",'', '| round | '+' | '.join(TEN)+' |','|---:|'+'---:|'*len(TEN)]
 for r in x['rounds']:appendix.append('| '+str(r['round'])+' | '+' | '.join(f'{r[k]:.6f}' for k in TEN)+' |')
 f=x['rounds'][-1];appendix += ['', '| class ID | test support | F1r20 |','|---:|---:|---:|']
 for c,(sup,val) in enumerate(zip(f['support'],f['f1_per_class'])):appendix.append(f'| {c} | {sup} | {val:.6f} |')
Path('docs/research/metrics-m20.md').write_text('\n'.join(appendix)+'\n')
fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
for col,seed in enumerate([42,43]):
 for j,name in enumerate(names):
  v=arms[f'M{2*j+col}'];r=[p['round'] for p in v['curve']]
  axes[0,col].plot(r,[p['f1'] for p in v['curve']],label=name,lw=1.6)
  axes[1,col].plot(r,[p['low256_mean'] for p in v['curve']],label=name,lw=1.6)
 axes[0,col].set_title(f'Own-test pooled macro-F1, seed {seed}')
 axes[1,col].set_title(f'Mean client F1, train rows <256, seed {seed}')
 for ax in axes[:,col]:ax.grid(alpha=.25);ax.set_xlim(10,20);ax.set_xlabel('Round');ax.set_ylabel('F1')
 axes[0,col].set_ylim(.88,.898)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=4,fontsize=9)
fig.tight_layout(rect=(0,.10,1,1));out=Path('docs/research/figures');out.mkdir(exist_ok=True);fig.savefig(out/'m20-curves.png',dpi=160);plt.close(fig)
print('M20 report,320round metric tables,256class rows,figure generated')
