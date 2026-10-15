"""Verify every flat source/fixture file on the target owner's private dataset."""
import json
from pathlib import Path
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

stage, proof = map(Path, sys.argv[1:3])
dataset = json.loads((stage/'dataset-metadata.json').read_text())['id']
api = KaggleApi(); api.authenticate()
files, token = {}, None
while True:
    result = api.dataset_list_files(dataset, page_token=token, page_size=200)
    files.update({p.name:p.total_bytes for p in result.dataset_files})
    token = result.next_page_token
    if not token:break
expected = {p.name:p.stat().st_size for p in stage.iterdir() if p.is_file() and p.name!='dataset-metadata.json'}
assert all(files.get(n)==size for n,size in expected.items()), 'source/fixture upload size/layout mismatch'
proof.write_text(json.dumps(dict(dataset=dataset,files=len(expected),bytes=sum(expected.values()),all_pages=True,
    evidence='owner-scoped names/sizes; runtime verifies all source/fixture SHA before training'),indent=2)+'\n')
print(dataset,len(expected),'source/fixture files PASS')
