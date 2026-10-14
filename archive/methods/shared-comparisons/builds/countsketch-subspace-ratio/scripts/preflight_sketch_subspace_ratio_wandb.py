"""Verify actual online login/init/log/finish; credentials remain outside source/outputs."""
import json
import netrc
from pathlib import Path
import wandb


def main():
    key=netrc.netrc().authenticators('api.wandb.ai')[2]
    assert key and wandb.login(key=key,anonymous='never',relogin=True,verify=True)
    run=wandb.init(project='mobihet-v2-preflight',name='online-auth-20261011',job_type='preflight',
                   mode='online',settings=wandb.Settings(save_code=False,disable_code=True))
    assert run.settings.mode=='online' and not run.settings._offline
    run.log({'preflight/online':1,'preflight/no_training':1}); url=run.url
    run.finish()
    result=dict(url=url,sdk=wandb.__version__,online=True,no_training=True)
    path=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])/'papers/v2/wandb-preflight.json'
    path.write_text(json.dumps(result,indent=2)); print('PASS online login/init/log/finish',url)


if __name__=='__main__': main()
