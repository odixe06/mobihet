"""Background watcher for B20 full-scale (SD42/SD43/SM42P) and SM r50 (SM42/SM43): collect every 20 min, audit when terminal.

    python scripts/watch_b20.py        # log runs/b20-watch/watch.log, state runs/b20-watch/state.json

Each tick runs the batch's own one-shot `collect` (status + pull terminal kernels); a batch whose kernels are all terminal is
audited once with its own `audit` (writes runs/<batch>/audit.json). Exits when both batches are audited, or early when any
kernel ends in a non-COMPLETE terminal state (ERROR/CANCEL), so the session is woken to inspect it. Max 40 ticks (~13 h).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/b20-watch"
BATCHES = {"b20-full": "scripts/run_b20_full.py", "b20-sm50": "scripts/run_b20_sm50.py"}
TICK, MAX_TICKS = 1200, 40


def run(script, cmd):
    p = subprocess.run([sys.executable, script, cmd], cwd=ROOT, capture_output=True, text=True, timeout=3000)
    return p.returncode, p.stdout + p.stderr


def log(msg):
    with (OUT / "watch.log").open("a") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S%z ") + msg.rstrip() + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    state = {"audited": {}, "ticks": 0}
    for tick in range(MAX_TICKS):
        state["ticks"] = tick + 1
        for name, script in BATCHES.items():
            if name in state["audited"]:
                continue
            rc, out = run(script, "collect")
            log(f"{name} collect rc={rc}\n{out}")
            sts = json.loads((ROOT / "runs" / name / "status.json").read_text())
            bad = [s["ref"] for s in sts if s.get("terminal") and s.get("status") != "COMPLETE"]
            if bad:
                log(f"STOP non-COMPLETE terminal: {bad}")
                state["stop"] = bad
                (OUT / "state.json").write_text(json.dumps(state, indent=1))
                print("B20W_NON_COMPLETE", bad, flush=True)
                return 2
            if rc == 0 and "ALL_TERMINAL" in out:
                rc, out = run(script, "audit")
                log(f"{name} audit rc={rc}\n{out}")
                state["audited"][name] = {"rc": rc, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "out": out[-2000:]}
        (OUT / "state.json").write_text(json.dumps(state, indent=1))
        if len(state["audited"]) == len(BATCHES):
            print("B20W_DONE", json.dumps(state["audited"]), flush=True)
            return 0
        time.sleep(TICK)
    print("B20W_TIMEOUT", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
