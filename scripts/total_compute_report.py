"""Table of scripts/total_compute.py results (runs/total-compute-20261119/*.json) -> Markdown + JSON.

    python scripts/total_compute_report.py runs/total-compute-20261119

Per model, every method on the same panel / engine / counter: bootstrap (once), and round 2 (steady state) split into client
train, round_other (server + codecs + packets) and eval (reported apart, not in the total). Ops = mm FLOPs + other element
ops (scripts/op_counter.py); time = wall seconds with a CUDA synchronize per phase (compiled production path, round 2).
Total FL = bootstrap + 50 x (train + round_other) of round 2. x FedAvg = same model, same quantity.
"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
NAMES = {"fedavg": "FedAvg", "fedprox": "FedProx", "afpha": "AFPHA", "fd_ids": "FD-IDS", "lwfednids": "LwFedNIDS",
         "pfedes": "pFedES", "fedproto": "FedProto", "fedtgp": "FedTGP", "tinyproto": "TinyProto", "nilm": "NILM-FL",
         "perfedskd": "PerFed-SKD", "apfed": "APFed", "v2x_htfl": "V2X-HtFL", "canperfl": "CANPerFL",
         "mh-t3": "MobiHet T3", "mh-n4c": "FB N4c", "mh-n2c": "FB N2c", "mh-l4": "FB-L L4", "mh-p4": "FB-L P4", "mh-q4": "FB-L Q4",
         "mh-l2": "FB-L L2", "mh-p2": "FB-L P2", "mh-q2": "FB-L Q2"}


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def phases(c):
    out = {}
    for r in c["ops"]:
        ph = r["phase"].split(".")[-1] if r["phase"].startswith("r2.") or r["phase"] == "bootstrap" else None
        if r["phase"].startswith("r1."):
            continue
        k = out.setdefault(ph, {"mm": 0, "other": 0})
        k["mm"] += r["mm_flops"]
        k["other"] += r["other_ops"]
    return out


rows = []
for c_path in sorted(d.glob("*.count.json")):
    name = c_path.name.removesuffix(".count.json")
    model, key = name.split("-", 1)
    key = key.removeprefix("base-")
    c, t = load(c_path), load(d / f"{name}.time.json")
    if c.get("status") != "done":
        rows.append(dict(model=model, key=key, status=c.get("status")))
        continue
    ph = phases(c)
    z = {"mm": 0, "other": 0}
    boot, train, other, ev = (ph.get(k, z) for k in ("bootstrap", "train", "round_other", "eval"))
    real = c["rows"]["real_rows"]
    r2 = c["rows"].get("r2.train", {})
    tr2 = ((t or {}).get("rows") or {}).get("r2.train", {})
    scale = 1.0                                   # count run at a smaller group_rows (eager BERT OOM): rescale train ops
    if tr2.get("executed") and r2.get("executed") and tr2["executed"] != r2["executed"]:   # to the 4096 engine's rows
        scale = tr2["executed"] / r2["executed"]
        train = {"mm": train["mm"] * scale, "other": train["other"] * scale}
        r2 = tr2
    sec = (t or {}).get("seconds", {}) if (t or {}).get("status") == "done" else {}
    rows.append(dict(model=model, key=key, status="done", real_rows=real,
                     boot=boot["mm"] + boot["other"], boot_mm=boot["mm"],
                     train=train["mm"] + train["other"], train_mm=train["mm"], other=other["mm"] + other["other"],
                     eval=ev["mm"] + ev["other"], executed_rows=r2.get("executed"), useful_rows=r2.get("useful"),
                     t_boot=sec.get("bootstrap"), t_train=sec.get("r2.train"), t_other=sec.get("r2.round_other"),
                     t_eval=sec.get("r2.eval"), peak_gb=c["peak_cuda_bytes"] / 2 ** 30, train_scale=scale))

for r in rows:
    if r["status"] == "done":
        r["total50"] = r["boot"] + 50 * (r["train"] + r["other"])
        r["t_total50"] = (r["t_boot"] + 50 * (r["t_train"] + r["t_other"])) if r["t_train"] is not None else None
for model in ("dags", "bert"):
    fed = next((r for r in rows if r["model"] == model and r["key"] == "fedavg" and r["status"] == "done"), None)
    for r in rows:
        if r["model"] == model and r["status"] == "done" and fed:
            r["x_total"] = r["total50"] / fed["total50"]
            r["x_train"] = r["train"] / fed["train"]
            r["x_train_mm"] = r["train_mm"] / fed["train_mm"]
            r["x_time"] = r["t_total50"] / fed["t_total50"] if r["t_total50"] and fed["t_total50"] else None

G = 1e12
lines = []
for model, title in (("dags", "DAGSNet"), ("bert", "FeatureBERT")):
    lines += [f"### {title} — panel 200 xe, round 2, cùng engine/counter", "",
              "| method | bootstrap Top | train Top/round | ngoài train Top/round | eval Top/round | tổng FL 50r Top | × FedAvg tổng | "
              "× FedAvg train (mm) | GPU-s 50r | × FedAvg time |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    sel = sorted((r for r in rows if r["model"] == model), key=lambda r: r.get("x_total") or 9e9)
    for r in sel:
        n = NAMES.get(r["key"], r["key"])
        if r["status"] != "done":
            lines.append(f"| {n} | {r['status']} | | | | | | | | |")
            continue
        f = lambda v, p=3: "—" if v is None else f"{v:.{p}f}"
        lines.append(f"| {n} | {r['boot'] / G:.3f} | {r['train'] / G:.2f} | {r['other'] / G:.4f} | {r['eval'] / G:.2f} | "
                     f"{r['total50'] / G:.1f} | {f(r.get('x_total'))} | {f(r.get('x_train_mm'))} | {f(r['t_total50'], 0)} | {f(r.get('x_time'))} |")
    lines.append("")
(d / "report.md").write_text("\n".join(lines) + "\n")
(d / "report.json").write_text(json.dumps(rows, indent=1))
print("\n".join(lines))
