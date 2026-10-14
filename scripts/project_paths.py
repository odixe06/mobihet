"""Resolve historical physical paths using the current workspace relocation record."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MAPPING=json.loads((ROOT/'docs/operations/workspace-layout.json').read_text())['path_map']
for row in json.loads((ROOT/'docs/operations/compatibility-aliases.json').read_text())['aliases']:
    if row['alias'].startswith('.compat/scripts/'):
        MAPPING[row['alias'][len('.compat/'):]]=row['target']


def current_path(relative):
    value=str(relative)
    for old in sorted(MAPPING,key=len,reverse=True):
        if value==old or value.startswith(old+'/'):return MAPPING[old]+value[len(old):]
    return value
