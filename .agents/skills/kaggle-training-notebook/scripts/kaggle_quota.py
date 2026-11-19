#!/usr/bin/env python
"""Bounded, resumable GPU-quota report for saved Kaggle accounts. Quota only: no kernel listing.

    python kaggle_quota.py --out quota-report.json [--users a b ...] [--jobs 4]
    python kaggle_quota.py --out quota-report.json --retry-failed     # re-query only UNVERIFIED rows

Each account runs in its own child process with finite requests timeouts (auth included) and a
hard wall-clock kill. Read-only reads retry with exponential backoff + jitter. A row is
  OK          -> server-confirmed numbers (gpu_left may legitimately be 0.0)
  UNVERIFIED  -> no answer; gpu_left is null. Never read it as 0 h or as a free slot.
OK rows already in --out and younger than --max-age are kept, so a timeout re-queries only
that account. Only exception class names are recorded; no token, header or child stderr.
"""
import argparse
import concurrent.futures
import datetime
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELDS = ("gpu_left", "gpu_total", "gpu_reserved", "tpu_left", "tpu_total", "refresh_at")


def now():
    return datetime.datetime.now().astimezone()


def child(user, connect, read):
    """Child mode: finite timeouts on every requests call, then one quota read. Prints one JSON line."""
    import requests
    try:  # optional project-scoped routing (e.g. MOBIHET_KAGGLE_API_VIA_WWW); absent elsewhere
        from kaggle_api_route import configure
        configure()
    except ImportError:
        pass
    orig_request, orig_send = requests.Session.request, requests.Session.send

    def request(self, method, url, **kw):
        kw["timeout"] = kw.get("timeout") or (connect, read)
        return orig_request(self, method, url, **kw)

    def send(self, req, **kw):
        kw["timeout"] = kw.get("timeout") or (connect, read)
        return orig_send(self, req, **kw)

    requests.Session.request, requests.Session.send = request, send
    sys.path.insert(0, str(HERE))
    import kaggle_account
    try:
        q = kaggle_account.account_quota(user)
        print(json.dumps({"ok": True, **{k: q[k] for k in ("user", *FIELDS)}}))
    except Exception as exc:  # message may echo request details; keep the class name only
        print(json.dumps({"ok": False, "error_type": type(exc).__name__}))


def query_once(user, args):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--_child", user,
           "--connect-timeout", str(args.connect_timeout), "--read-timeout", str(args.read_timeout)]
    env = dict(os.environ)
    scripts = HERE.parents[3] / "scripts"  # calling project's scripts/ (route helper), if any
    if scripts.is_dir():
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(scripts), env.get("PYTHONPATH")) if p)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error_type": "WallTimeout"}
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    try:
        return json.loads(lines[-1])
    except (IndexError, ValueError):
        return {"ok": False, "error_type": f"ChildExit{p.returncode}"}


def query(user, args, sleep=time.sleep, once=query_once):
    errors = []
    for attempt in range(1, args.attempts + 1):
        r = once(user, args)
        if r.get("ok"):
            if r.get("user") != user:
                return {"status": "UNVERIFIED", "gpu_left": None, "attempts": attempt, "errors": errors + ["IdentityMismatch"],
                        "checked_at": now().isoformat()}
            return {"status": "OK", **{k: r[k] for k in FIELDS}, "attempts": attempt, "checked_at": now().isoformat()}
        errors.append(r.get("error_type", "Unknown"))
        if attempt < args.attempts:
            sleep(min(args.backoff * 2 ** (attempt - 1), 60) * random.uniform(0.5, 1.5))
    return {"status": "UNVERIFIED", "gpu_left": None, "attempts": args.attempts, "errors": errors,
            "checked_at": now().isoformat()}


def fresh(row, max_age):
    if row.get("status") != "OK":
        return False
    age = now() - datetime.datetime.fromisoformat(row["checked_at"])
    return age.total_seconds() <= max_age


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)


def report(path, rows):
    out = {"at": now().isoformat(), "scope": "GPU/TPU quota only; no kernel/slot inventory",
           "verified": sorted(u for u, r in rows.items() if r["status"] == "OK"),
           "failed": sorted(u for u, r in rows.items() if r["status"] != "OK"),
           "accounts": dict(sorted(rows.items()))}
    atomic_json(path, out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--_child", help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--users", nargs="*", help="default: every saved account")
    ap.add_argument("--jobs", type=int, default=4, help="parallel accounts (keep <= 6)")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=5.0, help="base seconds; doubles, x[0.5,1.5] jitter")
    ap.add_argument("--timeout", type=float, default=90.0, help="hard wall seconds per attempt (auth + quota)")
    ap.add_argument("--connect-timeout", type=float, default=10.0)
    ap.add_argument("--read-timeout", type=float, default=30.0)
    ap.add_argument("--max-age", type=float, default=1800.0, help="reuse OK rows younger than this (s)")
    ap.add_argument("--retry-failed", action="store_true", help="query only rows not OK/fresh in --out")
    args = ap.parse_args(argv)
    if args._child:
        return child(args._child, args.connect_timeout, args.read_timeout) or 0
    if not args.out:
        ap.error("--out is required")
    sys.path.insert(0, str(HERE))
    import kaggle_account
    users = args.users or kaggle_account.known_users()
    old = json.loads(args.out.read_text()).get("accounts", {}) if args.out.exists() else {}
    keep = (lambda r: r.get("status") == "OK") if args.retry_failed else (lambda r: fresh(r, args.max_age))
    rows = {u: old[u] for u in users if u in old and keep(old[u])}
    todo = [u for u in users if u not in rows]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(args.jobs, 8))) as pool:
        futures = {pool.submit(query, u, args): u for u in todo}
        for fut in concurrent.futures.as_completed(futures):
            rows[futures[fut]] = fut.result()
            report(args.out, rows)  # persist each success at once; a crash keeps finished rows
    out = report(args.out, rows)
    for u, r in sorted(rows.items(), key=lambda kv: -(kv[1].get("gpu_left") or -1)):
        left = "?" if r["gpu_left"] is None else f"{r['gpu_left']:.2f}h"
        print(f"{u:<20} {r['status']:<10} {left:>8}  {r.get('refresh_at', '') or ','.join(r.get('errors', []))}")
    print(f"verified {len(out['verified'])}/{len(rows)}; failed: {' '.join(out['failed']) or '-'}; report {args.out}")
    return 0 if not out["failed"] else 3


if __name__ == "__main__":
    sys.exit(main())
