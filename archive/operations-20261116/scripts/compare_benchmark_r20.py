"""Read matching-round baseline CMs, keeping model-view/provenance differences explicit; no training."""
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from core import metrics

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT.parent / "fl_dagsnet_veremi_v2"
RUNS = {"fedavg": "fl2-fedavg", "fedprox": "fl2-fedprox", "scaffold": "fl2-scaffold", "apfed": "fl2-apfed-r2",
        "nilm": "fl2-nilm-r2", "perfedskd": "fl2-perfedskd-r2", "pfedes": "fl2-pfedes-r2",
        "tinyproto": "fl2-tinyproto-r4", "fedproto": "fl2-fedproto", "fedtgp": "fl2-fedtgp-r2",
        "v2x_htfl": "fl2-v2x-htfl-r2", "lwfednids": "fl2-lwfednids-r2", "fd_ids": "fl2-fd-ids-r2", "afpha": "fl2-afpha-r2"}


def main():
    counts = np.load(ROOT / "evidence/research-data/test_support.npy")
    records = []
    for name, slug in RUNS.items():
        folder = BENCH / "runs/pulls" / slug / slug
        path = folder / "cms_r020.npz"
        cfg = json.loads((folder / "config.json").read_text())
        assert cfg["clients"] == 5343
        if name in {"lwfednids", "v2x_htfl"}:
            # Existing methods prune/add a projection head; separate from identical-parameter reference.
            assert cfg["n_params"] == {"lwfednids": 39494, "v2x_htfl": 461456}[name]
            continue
        assert cfg["n_params"] == 395408
        assert cfg["hp"]["batch"] == 64 and cfg["hp"]["rounds"] == 50 and cfg["hp"]["seed"] == 42
        with np.load(path) as z:
            assert np.array_equal(z["ids"], np.arange(5343))
            for view in z.files:
                if view == "ids": continue
                cm = z[view]
                assert np.array_equal(cm.sum(2), counts) and (cm >= 0).all()
                summary = metrics.summary(cm)
                records.append({"method": name, "view": view, "round": 20, **{k: float(summary[k]) for k in metrics.TEN},
                    "client_mean": float(summary["client_f1_macro_mean"]), "p10": float(summary["client_f1_macro_p10"]),
                    "timeDelay_f1": float(summary["f1_per_class"][13]), "cm_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "path": str(path)})
    out = ROOT / "runs/p-study"; out.mkdir(exist_ok=True)
    (out / "benchmark-r20.json").write_text(json.dumps(records, indent=2) + "\n")
    fields = ["method", "view", "round", *metrics.TEN, "client_mean", "p10", "timeDelay_f1"]
    with (out / "benchmark-r20.csv").open("w") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(records)
    text = ["# Baseline r20 — tham chiếu cùng horizon, seed42", "",
        "Tính lại từ CM5343xe/9347679testrow của repo `../fl_dagsnet_veremi_v2`, SHA/config ở `runs/p-study/benchmark-r20.json`.",
        "Giữ view đúng phương pháp; globalmodel và modelriêng là hai cách triển khai. Đây là r20 diagnostic, không headline50round hoặc freshconfirmatorytest.",
        "Không chạy lại audit toàn source/privatebank; kế thừa giới hạn PASS_BOUNDED của repo benchmark. CANPerFL fine-tune sau50 không có r20 tương ứng; centralizedepoch khác nên không ghép.",
        "LwFedNIDS đãprune còn39494parameter, V2X-HtFL có461456parameter do head riêng; không ghép vào bảng đồng nhất395408parameter này.", "",
        "accuracy=precision_micro=recall_micro=recall_weighted=f1_micro trong singlelabelmulticlass.", "",
        "|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in records:
        text.append("|" + "|".join(f"{row[k]:.6f}" if isinstance(row[k], float) else str(row[k]) for k in fields) + "|")
    text += ["", "Các số này giúp định vị gap/chất lượng lớp; không quy chênh lệch giữa hai phương pháp thành causal effect riêng teacher.",
             "Private/teacher hiện tại phải đọc cùng own-test/perclass/weakclient và byte. Seed42 đơn lẻ chưa significance.", ""]
    (ROOT / "docs/research/benchmark-r20-reference.md").write_text("\n".join(text))
    print("BENCHMARK_R20_CM_VERIFIED", len(records), flush=True)
    for row in records:
        if row["method"] in {"fedavg", "apfed", "nilm"}: print(row["method"], row["view"], row["f1_macro"], row["timeDelay_f1"])


if __name__ == "__main__": main()
