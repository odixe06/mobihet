"""Import audited ClassTail warm-up state before any training-data decode."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import torch


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def import_run(mount,output,name,expected,manifest,verify):
    candidates=[p.parent for p in Path(mount).rglob('config.json') if p.parent.name==name
        and json.loads(p.read_text()).get('fingerprint')==expected['fingerprint']]
    assert len(candidates)==1,('require-resume missing/ambiguous',name,candidates)
    source=candidates[0];identity=json.loads((source/'config.json').read_text())
    assert identity['source']==expected['source'] and identity['fixture']==expected['fixture']
    verify(source,manifest,expected['fixture'],expected['last_round'],calibration=False)
    from mh.dagsnet import build_model,CFG
    model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True)
    weights=torch.load(source/'weights'/f'round_{expected["last_round"]:03d}.pt',map_location='cpu',weights_only=True)
    for sd in weights['private']+weights['proxy']:model.load_state_dict(sd,strict=True)
    destination=Path(output)/name
    files=[p for p in source.rglob('*') if p.is_file() and 'wandb' not in p.relative_to(source).parts and p.name!='log.txt']
    for marker_pass in (False,True):
        for path in files:
            relative=path.relative_to(source)
            if (relative.parts[0]=='complete')!=marker_pass:continue
            target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
            assert not target.exists(),('resume collision',target)
            temporary=target.with_name(target.name+'.import-part')
            shutil.copyfile(path,temporary);assert sha(temporary)==sha(path);os.replace(temporary,target)
    audit=verify(destination,manifest,expected['fixture'],expected['last_round'],calibration=False)
    assert audit['rounds']==expected['last_round']
    return dict(run=name,last_round=expected['last_round'],fingerprint=expected['fingerprint'],
                copied_files=len(files),markers_last=True,all_models_strict=True)
