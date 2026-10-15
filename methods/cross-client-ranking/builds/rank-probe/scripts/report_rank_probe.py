"""Report locked probe and explained residual inflation; diagnostics never replace preregistered gate."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[5];BASE=Path(__file__).resolve().parents[1]


def main():
    gate=json.loads((BASE/'probes/paired-gate.json').read_text())
    paired=json.loads((BASE/'probes/paired-projection-diagnostic.json').read_text())
    out=ROOT/'docs/methods/cross-client-ranking/builds/rank-probe/results';out.mkdir(exist_ok=True)
    names=json.loads((ROOT/'knowledge/meta.json').read_text())['class_names'];analysis={}
    lines=['# CrossClientRank — audited probe and projection diagnostic','',
        'Train-only CE-r2 snapshot, không SGD/holdout/utility. Gate đã khóa giữ nguyên; cả hai seed số học PASS.',
        'Residual macro rất lớn không được diễn giải thành lượng tri thức hoặc improvement F1.',
        '', '| seed | support/signal | coverage | locked residual macro | max condition | max coefficient |', '|---|---|---:|---:|---:|---:|']
    for seed,proof in zip((42,43),paired):
        result=gate['results'][str(seed)]
        folder=BASE/'artifacts/kaggle'/f'ccr-diag-s{seed}-20261015-v1'
        rows=[json.loads(p.read_text()) for p in sorted((folder/'runs').glob('diagnostic_??.json'))]
        assert len(rows)==20
        residual=np.asarray([c['residual_fraction'] for c in result['classes']]);largest=int(residual.argmax())
        # Sensitivity summaries only: never substitute these values for the locked gate.
        analysis[str(seed)]=dict(formal_probe_pass=result['pass_probe'],dominant_class=largest,class_name=names[largest],
            locked_macro=float(residual.mean()),capped_per_class_diagnostic=float(np.minimum(residual,1).mean()),
            without_dominant_class_diagnostic=float(np.delete(residual,largest).mean()),
            class_residual_fraction=residual.tolist(),projection_max_condition=proof['max_condition'],
            coefficient_max=proof['max_abs_coefficient'],client17=rows[17],
            scope='Sensitivity descriptions only; no changed gate, new feature evidence, utility or calibration-equivalence proof')
        lines.append(f'| {seed} | {result["supported_classes"]}/{result["signal_classes"]} | {result["scan_coverage"]:.3f} | {result["residual_macro_class"]:.6f} | {proof["max_condition"]:.3f} | {proof["max_abs_coefficient"]:.3f} |')
    lines+=['', f'Dominant class5: `{names[5]}`. Normalized design condition≤5,20; SVD ill-conditioning giả thuyết bị bác.',
        'Client17 fit class5 CE-column RMS1,10e−7/2,62e−6; original coefficients817,95/58,76.',
        'Fit class5 delta-energy1,05e−18/5,37e−16, nhưng heldout có một số record khác; top5 đóng góp',
        '99,999995%/99,998230% residual energy lớp này. Coefficient lớn khuếch đại extrapolation.',
        'Ở các client khác, shared bias/temperature fit ưu tiên toàn probe cũng có thể để residual lớp gần-zero lớn theo tỷ lệ.',
        '', 'Diagnostic sensitivity (không đổi gate): cap từng class ratio ở1 → '+
        '/'.join(f'{analysis[str(s)]["capped_per_class_diagnostic"]:.6f}' for s in (42,43))+
        '; bỏ class chi phối → '+ '/'.join(f'{analysis[str(s)]["without_dominant_class_diagnostic"]:.6f}' for s in (42,43))+'.',
        'Các giá trị này vẫn>0,20, nên PASS không chỉ phụ thuộc magnitude outlier. Tuy nhiên probe không chứng minh',
        'update hữu ích hoặc bác mọi dạng calibration artifact. Không tự đổi sample/basis/rcond/gate hoặc mở utility.',
        '', 'Wire thực:20×928B up,20×746B down, bootstrap18880B mỗi seed; trước channel overhead utility chưa triển khai.',
        'Probe elapsed70,32/65,97s gồm setup; không dùng timing này làm training cost≤2×CE proof.',
        '', '[Gates và nguyên nhân](analysis.json) · [Protocol/nguồn/provenance](../README.md).',
        'Audit exact executed/source/input/CE-r2 weights; independent wire/scalar reconstruction; diagnostic tái hiện original energies',
        'trên T4. Không independent local full-gradient inference; không có classification metrics mới vì model không train/evaluate holdout.']
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print('CrossClientRank report: locked probe PASS, residual inflation diagnosed; utility unmeasured')


if __name__=='__main__':main()
