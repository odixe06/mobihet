# Captured hybrid-uplink regression

`uplink-fallback-r1.npz`: class6 packet rows from b14 local W1/200client smoke r1, 11/11/2026.
Source: `runs/sim-kaggle-b14/local-smoke/wire/rank0_r001.npz`.
Only model-output summaries/counts; no raw training features or weights.

- `counts`: nonempty sender counts, total18606; cells1–7 contribute80.
- `values`: fp16 centered-logprob payload of cells with≥8rows.
- `small`: uint8 probability payload of smaller cells.
- `stored_sum`: original GPU aggregate for class6, float64.

CPU replay mean-score discrepancy3.780150533965438e-10. Replaying only the fallback
log transform on GPU reproduces stored aggregate exactly. Old auditor incorrectly
applied pure-LG1e-10 envelope to the hybrid result. Regression tests also inject
corrupt aggregate values in classes with/without fallback and require rejection.
