"""One-shot early collection/audit of a frozen terminal subset. Leave the main O watcher unchanged."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.metrics import TEN

OUT = ROOT / "runs/o-study/early-collect-20261101"
STAGE = ROOT / "artifacts/o20-early-20261101"
CANONICAL = ROOT / "artifacts/o20"
SKILL = ROOT / ".agents/skills/kaggle-training-notebook/scripts"
PATTERN = r'(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz|train_cms_r\d{3}\.npz|diagnostics_rank[01]\.jsonl|external_cms_rank[01]_r\d{3}\.npz)$|/models/r\d{3}\.pt$)'


def required(folder, slug):
    d = folder / slug
    return [folder / "provenance/source_manifest.json", folder / (slug + ".log"), d / "config.json", d / "rounds.jsonl"] + [
        d / f"cms_r{r:03}.npz" for r in range(1, 21)] + [d / "models" / f"r{r:03}.pt" for r in range(1, 21)] + [
        d / f"train_cms_r{r:03}.npz" for r in (1, 2, 5, 10, 15, 20)] + [
        d / f"diagnostics_rank{rank}.jsonl" for rank in (0, 1)] + [
        d / f"external_cms_rank{rank}_r{r:03}.npz" for rank in (0, 1) for r in (1, 10, 20)]


def link_immutable(source, dest):
    # The main pull helper rewrites its log. Give each view its own log inode.
    return shutil.copy2(source, dest) if str(source).endswith(".log") else os.link(source, dest)


def canonical_ready(folder, slug):
    receipt = ROOT / "runs/o-study/collection" / (slug + ".collect.log")
    return receipt.is_file() and "files ok, 0 missing," in receipt.read_text(errors="replace") and all(
        p.is_file() for p in required(folder, slug))


def publish(folder, slug):
    dest = CANONICAL / slug
    if dest.exists(): return "main_destination_exists; left untouched"
    temp = CANONICAL / (".early-" + slug + "-" + uuid.uuid4().hex)
    shutil.copytree(folder, temp, copy_function=link_immutable)
    # GNU mv uses no-replace semantics even if the main collector creates dest concurrently.
    subprocess.run(["mv", "--no-clobber", "-T", str(temp), str(dest)], check=True)
    if temp.exists():
        shutil.rmtree(temp)  # Only our newly created links, never the main collector's directory.
        return "main_destination_won_race; left untouched"
    return "published_complete_audited_files; main pull can skip matching sizes"


def write_summary(records):
    p = OUT / "progress.json"; temp = p.with_suffix(".tmp")
    temp.write_text(json.dumps({"audited_runs": records, "selected_runs": len(json.loads((OUT / "batch.json").read_text()))}, indent=2) + "\n")
    temp.replace(p)


def report(records):
    lines = ["# O20 — audit sớm một phần batch", "", "Chỉ các run trong bảng đã pull/audit đủ20round.",
             "Subset chọn từ exactversionstatus tại `runs/o-study/early-collect-20261101/status.json`;",
             "không khẳng định trạng thái hiện tại của các run còn lại. [Thiết kế khóa](screening-o20.md).", "",
             "| arm | seed | F1r20 | p10 | source/CM/20snapshot/replay audit |", "|---|---:|---:|---:|---|"]
    appendix = ["# O20 — 10metric của subset audit sớm", "", "Endpoint20, không best-round. [Kết quả](results-o20-early.md).",
                "Accuracy=precision_micro=recall_micro=recall_weighted=f1_micro trong single-labelclassification."]
    for rec in records:
        a = json.loads((OUT / "audits" / rec["arm"] / "audit.json").read_text())[0]
        row = a["rounds"][-1]
        lines.append(f'| {rec["arm"]} | {a["hp"]["seed"]} | {row["f1_macro"]:.6f} | {row["client_f1_macro_p10"]:.6f} | PASS |')
        appendix += ["", f'## {rec["arm"]} — seed{a["hp"]["seed"]}', "", "| round | " + " | ".join(TEN) + " |", "|---:|" + "---:|" * len(TEN)]
        for row in a["rounds"]:
            appendix.append("| " + str(row["round"]) + " | " + " | ".join(f"{row[k]:.6f}" for k in TEN) + " |")
        appendix += ["", "| class ID | test support | F1r20 |", "|---:|---:|---:|"]
        for c, (n, f) in enumerate(zip(row["support"], row["f1_per_class"])):
            appendix.append(f"| {c} | {n} | {f:.6f} |")
    lines += ["", "Source/config/LR/IDs/support/CM→10metric/finite/probe/externalhash/teacherlag/rawQ/target đã kiểm từ artifact.",
              "Không privatebank/predictionreplay/fullcontentfingerprintdataset hoặc formalprivacyproof.",
              "Chưa đủ4cell×2seed; chưa chọn winner. [Đủ10metric từng round](metrics-o20-early.md).", "",
              "Packet-equivalent3333504B/xe/20round; offlinepreload3402795B, không cộng lại cached downlink.",
              "Đây là teacherreplay chẩn đoán; chưa liveFL/contactcostproof. Snapshot/log/source giữ local, chưa DVCbackup."]
    (ROOT / "docs/research/results-o20-early.md").write_text("\n".join(lines) + "\n")
    (ROOT / "docs/research/metrics-o20-early.md").write_text("\n".join(appendix) + "\n")


def main():
    import fcntl
    lock = (OUT / "early.lock").open("a"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    batch = json.loads((OUT / "batch.json").read_text())
    statuses = {r["arm"]: r for r in json.loads((OUT / "status.json").read_text())["runs"]}
    assert batch and all(statuses[s["arm"]]["terminal"] for s in batch)
    STAGE.mkdir(exist_ok=True)
    (STAGE / ".gitignore").write_text("/*\n!/.gitignore\n")
    assert shutil.disk_usage(STAGE).free > 10 * 2**30
    records = []
    for spec in batch:
        slug = spec["slug"]; folder = STAGE / slug; canonical = CANONICAL / slug
        if not folder.exists() and canonical_ready(canonical, slug):
            shutil.copytree(canonical, folder, copy_function=link_immutable)
            mode = "reused_complete_main_files"
        else:
            folder.mkdir(exist_ok=True)
            cmd = [sys.executable, str(SKILL / "kaggle_as.py"), spec["account"], "--", sys.executable,
                   str(ROOT / "scripts/run_bounded_python.py"), str(SKILL / "pull_kernel_output.py"),
                   statuses[spec["arm"]]["ref"], "-p", str(folder), "-j", "4", "--pattern", PATTERN]
            log = OUT / (slug + ".collect.log")
            with log.open("w") as f: result = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            log.chmod(0o600); assert result.returncode == 0, (spec["arm"], result.returncode)
            mode = "early_pull_4_connections"
        assert all(p.is_file() for p in required(folder, slug)), spec["arm"]
        audit = OUT / "audits" / spec["arm"]; audit.mkdir(parents=True, exist_ok=True)
        single = audit / "batch.json"; single.write_text(json.dumps([spec], indent=2) + "\n")
        subprocess.run([sys.executable, str(ROOT / "scripts/audit_o20.py"), "--batch", str(single),
                        "--root", str(STAGE), "--out", str(audit)], cwd=ROOT, check=True)
        files = [p for p in folder.rglob("*") if p.is_file()]
        rec = {"arm": spec["arm"], "ref": statuses[spec["arm"]]["ref"], "mode": mode, "audit": "PASS20round",
               "files": len(files), "bytes": sum(p.stat().st_size for p in files), "publication": publish(folder, slug)}
        records.append(rec); write_summary(records); report(records)
        print("EARLY_AUDITED", rec, flush=True)
    (OUT / "complete.json").write_text(json.dumps({"runs": len(records), "records": records,
        "scope": "Frozen terminal subset only; main eight-run watcher/collector remains independent"}, indent=2) + "\n")
    print("O20_EARLY_PULL_AUDIT_DONE", flush=True)


if __name__ == "__main__": main()
