"""Keep Y50 going without an open agent session: restart dead wave workers, run the analysis at the end, then remove itself.

A worker owns all chaining (collect, r20 rule, -c2 when quota allows); this only makes sure one is alive per wave. Run from cron:

    python scripts/keep_y50.py --install      # every 10 minutes, also covers reboot
    python scripts/keep_y50.py --uninstall
    python scripts/keep_y50.py                # one pass; what cron runs

A wave that dies MAX_RESTARTS times with no change in progress.json is left stopped and reported in runs/y-study/keep.log.
Restarting cannot push twice: a push needs a fresh intent file and an unused slug, and a recorded push is adopted.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs/y-study"
TAG = "# mobihet-y50-keep"
MAX_RESTARTS = 6


def note(*text):
    print(time.strftime("%Y-%m-%dT%H:%M:%S%z"), *text, flush=True)


def alive(wave):
    with (STUDY / f"wave{wave}/worker.lock").open("a") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return True
    return False


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def crontab(lines):
    subprocess.run(["crontab", "-"], input="".join(x + "\n" for x in lines), text=True, check=True)


def current():
    p = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    return [x for x in p.stdout.splitlines() if TAG not in x] if p.returncode == 0 else []


def install():
    line = f"*/10 * * * * cd {ROOT} && {sys.executable} scripts/keep_y50.py >> runs/y-study/keep.log 2>&1 {TAG}"
    crontab([*current(), line])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--install", action="store_true"); ap.add_argument("--uninstall", action="store_true")
    a = ap.parse_args()
    if a.install: return install()
    if a.uninstall: return crontab(current())
    os.environ["PATH"] = f"{Path(sys.executable).parent}:/usr/local/bin:/usr/bin:/bin"
    guard = (STUDY / "keep.lock").open("a")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = STUDY / "keep-state.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    closed = 0
    for wave in (1, 2):
        out = STUDY / f"wave{wave}"
        if (out / "closure.json").exists(): closed += 1; continue
        if alive(wave): continue
        seen = digest(out / "progress.json")
        row = state.setdefault(str(wave), {"progress": seen, "restarts": 0})
        if row["progress"] != seen: row.update(progress=seen, restarts=0)
        if row["restarts"] >= MAX_RESTARTS:
            note("Y50_KEEP_ATTENTION wave", wave, "stopped after", MAX_RESTARTS, "restarts with no progress; read", f"runs/y-study/wave{wave}.log")
            continue
        row["restarts"] += 1
        with (STUDY / f"wave{wave}.log").open("a") as log:
            p = subprocess.Popen([sys.executable, "-u", "scripts/run_y50.py", "--wave", str(wave), "--collect-only"], cwd=ROOT,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        note("Y50_KEEP_RESTART wave", wave, "pid", p.pid, "restart", row["restarts"])
    path.write_text(json.dumps(state, indent=1) + "\n")
    if closed == 2:
        with (STUDY / "analysis.log").open("w") as log:
            rc = subprocess.run([sys.executable, "scripts/analyse_y50.py"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        note("Y50_KEEP_DONE analysis rc", rc)
        crontab(current())


if __name__ == "__main__": main()
