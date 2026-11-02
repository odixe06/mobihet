"""Read-only owner dataset inventory after ambiguous create responses; no retry/create/version."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--owner", required=True); a = ap.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetMetadataRequest
    api = KaggleApi(); api.authenticate()
    owned = []
    for page in range(1, 101):
        rows = api.dataset_list(mine=True, page=page)
        if not rows: break
        owned.extend(row.ref for row in rows if row is not None)
    else: raise RuntimeError("Owner dataset pagination incomplete")
    request = ApiGetDatasetMetadataRequest(); request.owner_slug = a.owner
    request.dataset_slug = "mh-r20-frozen-teachers-20261102"
    record = {"owner": a.owner, "owned_refs": owned, "requested_ref": f"{a.owner}/{request.dataset_slug}"}
    try:
        with api.build_kaggle_client() as client:
            meta = client.datasets.dataset_api_client.get_dataset_metadata(request)
        record.update(metadata_error=meta.error_message, private=meta.info.is_private,
                      actual_owner=meta.info.owner_user, actual_slug=meta.info.dataset_slug)
    except Exception as exc:
        record.update(error_type=type(exc).__name__, http_status=getattr(getattr(exc, "response", None), "status_code", None))
    out = ROOT / "runs/r-study/push-recovery/input-inspection"; out.mkdir(exist_ok=True)
    (out / (a.owner + ".json")).write_text(json.dumps(record, indent=2) + "\n")
    print("R20_INPUT_INSPECTED", a.owner, record["requested_ref"] in owned,
          record.get("http_status", "metadata_available"), flush=True)


if __name__ == "__main__": main()
