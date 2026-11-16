"""B21 Resume collection and audit runner for resumed kernels (T3P, T4FP, T142).

    python scripts/run_b21_resume.py collect
    python scripts/run_b21_resume.py audit
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_b20_full as B  # noqa: E402
from scripts import run_w20 as W  # noqa: E402
from scripts.watch_kaggle_batch import atomic_json, query  # noqa: E402

OUT, ART = ROOT / "runs/b21-resume", ROOT / "artifacts/b21-resume"
B.OUT, B.ART = OUT, ART
W.OUT, W.ART, W.TAG = OUT, ART, "B21RESUME"
W.COLLECT = r'(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz)$)'


def collect():
    OUT.mkdir(parents=True, exist_ok=True)
    batch = json.loads((OUT / "batch.json").read_text())
    states = [query(s) for s in batch]
    atomic_json(OUT / "status.json", states)
    for spec, st in zip(batch, states):
        print(spec["arm"], spec["slug"], st.get("status"), flush=True)
        if st.get("terminal") and not (ART / spec["slug"] / "pulled.ok").exists():
            print(f"Pulling {spec['slug']}...", flush=True)
            rc = W.collect(spec)
            print(f"Pull {spec['slug']} return code: {rc}", flush=True)
            if rc == 0:
                (ART / spec["slug"] / "pulled.ok").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    all_term = all(s.get("terminal") for s in states)
    print("ALL_TERMINAL" if all_term else "WAITING", flush=True)


def audit():
    B.audit()
    out = json.loads((OUT / "audit.json").read_text())
    ends = {s["arm"]: s["endpoint"] for s in json.loads((OUT / "batch.json").read_text())}
    for arm, res in out.items():
        if "checks" not in res:
            continue
        ok = res["delta_mean_last5"] >= -.0015 and res["delta_end"] >= -.003
        if ends[arm] == 50:
            ok = ok and res["endpoint"]["f1_macro"] > B.TOP3
        res["verdict"], res["rule"] = "PASS" if ok and res["checks"]["pass"] else "FAIL", "b21"
        print(arm, "B21", res["verdict"], "d_mean_last5 %+.4f d_end %+.4f macro %.4f" %
              (res["delta_mean_last5"], res["delta_end"], res["endpoint"]["f1_macro"]), flush=True)
    atomic_json(OUT / "audit.json", out)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_b21_resume.py [collect|audit]")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "collect":
        collect()
    elif cmd == "audit":
        audit()


if __name__ == "__main__":
    main()
