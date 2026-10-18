"""CPU-only check of exact private parent mounts before spending GPU quota on continuation."""
import json
from pathlib import Path
import nbformat as nbf

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[2])
BASE=ROOT/'.compat/papers/full-validation'


def main():
    for source in sorted((BASE/'notebook').glob('*cont01*')):
        meta=json.loads((source/'kernel-metadata.json').read_text())
        original=nbf.read(source/meta['code_file'],as_version=4)
        # Execute the actual source/import gate, stopping before full data preparation.
        setup=original.cells[2].source.split('from full_prepare import prepare_full',1)[0]
        setup=setup.replace("assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) for i in range(2))",'assert torch.cuda.device_count()==0')
        setup=setup.replace("print('torch',torch.__version__,'gpus',[torch.cuda.get_device_name(i) for i in range(2)],flush=True)","print('torch',torch.__version__,'CPU resume probe',flush=True)")
        assert 'get_device_name' not in setup and 'get_device_capability' not in setup
        rid=source.name.replace('cont01','mount03')
        folder=BASE/'mount-probe'/rid;folder.mkdir(parents=True,exist_ok=True)
        assert not (folder/'PUSHED').exists()
        prefix='import os,sys,json,time\nSESSION_T0=time.monotonic()\n'
        suffix="\n(SOURCE/'MOUNT_VERIFIED.json').write_text(json.dumps({'imports':imports,'cpu_only':True,'trained_rounds':0},indent=2))\nprint('CPU REQUIRE RESUME MOUNT PASS',flush=True)\n"
        notebook=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('# CPU checkpoint mount/import probe\nNo training, no GPU, no data decode.'),nbf.v4.new_code_cell(prefix+setup+suffix)])
        notebook.metadata.update(kernelspec={'name':'python3','display_name':'Python 3','language':'python'},
            language_info={'name':'python','version':'3.12'},kaggle={'accelerator':'none','isGpuEnabled':False,'isInternetEnabled':True,'language':'python','sourceType':'notebook'})
        filename=rid+'.ipynb';owner=meta['id'].split('/')[0]
        metadata=dict(meta,id=f'{owner}/mobihet-{rid}',title=f'mobihet-{rid}',code_file=filename,enable_gpu=False,
            dataset_sources=[p for p in meta['dataset_sources'] if p.endswith('-src')])
        metadata.pop('machine_shape',None)
        nbf.validate(notebook);compile(notebook.cells[1].source,rid,'exec')
        nbf.write(notebook,folder/filename)
        (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
        print(metadata['id'],'CPU-only prepared')


if __name__=='__main__':main()
