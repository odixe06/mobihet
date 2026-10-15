"""Owner-scoped all-page sizes; the CPU import probe checks mounted SHA and state."""
import json
from pathlib import Path
import sys
from kaggle.api.kaggle_api_extended import KaggleApi
from requests.exceptions import HTTPError

folder = Path(sys.argv[1])
bridge = json.loads((folder/'checkpoint-dataset.json').read_text())
api = KaggleApi()
api.authenticate()
files, token = {}, None
while True:
    try:
        response = api.dataset_list_files(bridge['dataset'], page_token=token, page_size=200)
    except HTTPError as exc:
        if exc.response.status_code not in (403,404):raise
        print(bridge['dataset'],'not available for listing yet; inspect processing/access before push')
        sys.exit(3)
    files.update({f.name: f.total_bytes for f in response.dataset_files})
    token = response.next_page_token
    if not token:
        break
matched = {}
for name, value in bridge['files'].items():
    candidates = [p for p in [name, *([name.split('/', 1)[1]] if '/' in name else [])] if p in files]
    assert len(candidates) == 1, ('remote layout missing/ambiguous', name, candidates)
    mounted = candidates[0]
    assert files[mounted] == value['size'], ('remote size mismatch', name, files[mounted])
    matched[name] = mounted
bridge['uploaded'] = True
(folder/'checkpoint-dataset.json').write_text(json.dumps(bridge, indent=2)+'\n')
(folder/'dataset-files-verified.json').write_text(json.dumps(dict(
    dataset=bridge['dataset'], matched_files=len(bridge['files']), all_pages=True, paths=matched,
    evidence='names and sizes; SHA/state verified by CPU production import probe'), indent=2)+'\n')
print(bridge['dataset'], len(bridge['files']), 'remote names/sizes PASS')
