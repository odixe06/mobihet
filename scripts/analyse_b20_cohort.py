"""Where (by client train size) do compute-reduction arms lose pooled macro-F1? Read-only on audited CMs."""
import sys
import numpy as np
ROOT = sys.argv[1]
A = ROOT + "/artifacts/"
sizes = np.load(ROOT + "/cache/train_sizes.npy")
EDGES = [0, 256, 1024, 4096, 16384, 10**9]
NAMES = ["<256", "256-1k", "1k-4k", "4k-16k", ">=16k"]


def macro(cm):
    tp = np.diag(cm).astype(float)
    p, r = cm.sum(0), cm.sum(1)
    f = np.where(p + r > 0, 2 * tp / np.maximum(p + r, 1), 0.0)
    return f.mean(), f


def load(path, r):
    z = np.load(f"{A}{path}/cms_r{r:03d}.npz")
    assert np.array_equal(z["ids"], np.arange(len(sizes)))
    return z["private"]


PAIRS = [
    ("SM42 r50", "b20-sm50/mh-k-b20-sm42-r50-c2/mh-k-b20-sm42p-r20", "codec-combo50/mh-k-cbb-s42-r50-c2/mh-k-cbb-s42-p20", 50),
    ("SM42 r20", "b20-sm50/mh-k-b20-sm42-r50-c2/mh-k-b20-sm42p-r20", "codec-combo50/mh-k-cbb-s42-r50-c2/mh-k-cbb-s42-p20", 20),
    ("SD42 r50", "b20-full/mh-k-b20-sd42-r50/mh-k-b20-sd42-r50", "codec-combo50/mh-k-cbb-s42-r50-c2/mh-k-cbb-s42-p20", 50),
    ("SD43 r50", "b20-full/mh-k-b20-sd43-r50/mh-k-b20-sd43-r50", "codec-combo50/mh-k-cbb-s43-r50-c2/mh-k-cbb-s43-p20", 50),
    ("SR48 r20", "g1-pilot20/mh-k-g1sr48-s42-p20/mh-k-g1sr48-s42-p20", "km50/mh-k-km16q8-s42-r50/mh-k-km16q8-s42-r50", 20),
    ("SR32 r20", "g1-pilot20/mh-k-g1sr32-s42-p20/mh-k-g1sr32-s42-p20", "km50/mh-k-km16q8-s42-r50/mh-k-km16q8-s42-r50", 20),
]
steps = np.ceil(sizes / 64)
syn_steps = np.maximum(steps, 16)
print("cohort   n_xe  share_real_rows  share_synth_steps  share_test_rows")
co = [(sizes >= lo) & (sizes < hi) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
cm0 = load(PAIRS[0][2], 50)
tst = cm0.sum((1, 2))
for n, m in zip(NAMES, co):
    print(f"{n:7s} {m.sum():5d}  {sizes[m].sum()/sizes.sum():.4f}  {syn_steps[m].sum()/syn_steps.sum():.4f}  {tst[m].sum()/tst.sum():.4f}")
for name, arm, ctl, r in PAIRS:
    a, c = load(arm, r), load(ctl, r)
    assert np.array_equal(a.sum(2), c.sum(2))
    m0, f0 = macro(c.sum(0))
    m1, f1 = macro(a.sum(0))
    print(f"\n== {name}: arm {m1:.4f} ctl {m0:.4f} d {m1-m0:+.4f} | benign {f1[0]-f0[0]:+.4f} TD {f1[13]-f0[13]:+.4f} c11 {f1[11]-f0[11]:+.4f} c7 {f1[7]-f0[7]:+.4f}")
    print("  cohort: d_macro if ONLY this cohort uses arm | cohort-own pooled macro arm-ctl")
    for n, m in zip(NAMES, co):
        h = c.copy(); h[m] = a[m]
        own = macro(a[m].sum(0))[0] - macro(c[m].sum(0))[0]
        print(f"  {n:7s} {macro(h.sum(0))[0]-m0:+.5f} | {own:+.5f}")
    print("  arm only on clients >= T: d_macro, share of synth steps affected")
    for T in [256, 1024, 4096, 16384]:
        m = sizes >= T
        h = c.copy(); h[m] = a[m]
        print(f"  T={T:6d} {macro(h.sum(0))[0]-m0:+.5f}  {syn_steps[m].sum()/syn_steps.sum():.4f}")

# FLOPs of size x phase synthetic-row schedules (x F per real row, network-wide), B20 convention: real pass 2.99 x F per row;
# a synthetic row costs the same, so synthetic = 2.99 * sum_k max(s_k, floor) * mean rows_k / 64 / sum_k s_k (s_k real steps;
# reproduces measured CBB 6.09 and SD 5.32). Late phase = rounds 26..50, half of the schedule.
print("\n== FLOPs of tiered schedules (rows/step early,late for tiers S <4096 | M 4096-16383 | L >=16384 train rows)")
tier = np.digitize(sizes, [4096, 16384])
SCH = {"CBB": [(64, 64)] * 3, "SD": [(64, 32)] * 3, "T1": [(64, 64), (64, 32), (64, 32)],
       "T3": [(64, 64), (64, 32), (48, 16)], "T4": [(64, 64), (48, 32), (32, 16)], "T5": [(64, 64), (32, 16), (16, 16)]}
for nm, sch, floor in [(k, v, 16) for k, v in SCH.items()] + [("T4F", SCH["T4"], 48)]:
    rows = np.array([(e + l) / 2 for e, l in sch])[tier]
    f = 2.99 * (1 + (np.maximum(steps, floor) * rows / 64).sum() / steps.sum())
    print(f"  {nm:4s} floor {floor} {sch}  -> {f:.2f} xF")
