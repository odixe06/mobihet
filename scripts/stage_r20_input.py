"""Create one owner-private immutable replay input and verify privacy, bytes and raw SHA before launch."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from requests.exceptions import HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def wait_ready(api, ref):
    # Private dataset visibility may lag a successful create. Retry the read only.
    for attempt in range(20):
        try:
            if api.dataset_status(ref).lower() == "ready": return
        except HTTPError as exc:
            if exc.response is None or exc.response.status_code not in {403, 404, 429, 500, 502, 503, 504}: raise
        if attempt < 19: time.sleep(15)
    raise RuntimeError("Private replay dataset readiness unknown after bounded reads; no recreate or GPU push")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--owner", required=True); args = parser.parse_args()
    from methods.mobihet_rstudy_data import FILE, SHA256, COMPRESSED_SHA256
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetMetadataRequest
    api = KaggleApi(); api.authenticate()
    slug = "mh-r20-frozen-teachers-20261102"; ref = f"{args.owner}/{slug}"
    folder = ROOT / "runs/r-study/input-datasets" / args.owner; folder.mkdir(parents=True, exist_ok=True)
    source = ROOT / "runs/r-study/teacher-input" / FILE
    assert hashlib.sha256(source.read_bytes()).hexdigest() == COMPRESSED_SHA256
    (folder / FILE).write_bytes(source.read_bytes())
    (folder / "dataset-metadata.json").write_text(json.dumps({"id": ref, "title": "MobiHet R20 frozen aggregate teachers",
        "licenses": [{"name": "CC0-1.0"}]}, indent=2) + "\n")
    intent = folder / "create-intent.json"
    assert not intent.exists(), "Existing dataset intent; inspect remote metadata/files before retry"
    # Keep receipt outside upload folder: input contains only teacher blob and dataset metadata.
    intent = folder.parent / (args.owner + ".create-intent.json")
    if not intent.exists():
        intent.write_text(json.dumps({"ref": ref, "private": True, "decoded_sha256": SHA256}) + "\n")
        response = api.dataset_create_new(str(folder), public=False, quiet=True, convert_to_csv=False, dir_mode="skip")
        result = {"ref": response.ref, "status": response.status, "error": response.error, "requested_ref": ref}
        (folder.parent / (args.owner + ".create-result.json")).write_text(json.dumps(result, indent=2) + "\n")
        assert not response.error and response.status.lower() != "error", "Private input create rejected; inspect create-result.json"
    # An existing intent permits inspection only, never another blind create/version request.
    wait_ready(api, ref)
    request = ApiGetDatasetMetadataRequest(); request.owner_slug = args.owner; request.dataset_slug = slug
    with api.build_kaggle_client() as client:
        metadata = client.datasets.dataset_api_client.get_dataset_metadata(request)
    assert not metadata.error_message and metadata.info.is_private, "Replay input must remain owner-private"
    listing = api.dataset_list_files(ref, page_size=100)
    assert len(listing.dataset_files) == 1 and not listing.next_page_token
    assert listing.dataset_files[0].name == FILE and listing.dataset_files[0].total_bytes == source.stat().st_size
    # Fetch the same input bytes again: size/listing alone cannot prove unchanged transport.
    verify = folder.parent / (args.owner + "-verify"); verify.mkdir(exist_ok=True)
    api.dataset_download_file(ref, FILE, path=str(verify), quiet=True, force=True)
    actual = verify / FILE
    if not actual.exists():
        import zipfile
        with zipfile.ZipFile(verify / (FILE + ".zip")) as archive: archive.extractall(verify)
    assert hashlib.sha256(actual.read_bytes()).hexdigest() == COMPRESSED_SHA256
    (folder.parent / (args.owner + ".ready.json")).write_text(json.dumps({"ref": ref, "private": True,
        "filename": FILE, "compressed_sha256": COMPRESSED_SHA256, "decoded_sha256": SHA256,
        "bytes": source.stat().st_size, "download_sha_verified": True}, indent=2) + "\n")
    print("R20_PRIVATE_INPUT_READY", ref, flush=True)


if __name__ == "__main__": main()
