"""Prevent preflight from advancing restored W&B steps; model source stays frozen."""
import runpy
import sys


def install(wandb):
    run_class = wandb.sdk.wandb_run.Run
    original = run_class.log

    def log(self, data, *args, **kwargs):
        if set(data) == {'preflight/online'} and not args and 'step' not in kwargs:
            kwargs = dict(kwargs, commit=False)
        return original(self, data, *args, **kwargs)

    run_class.log = log


if __name__ == '__main__':
    import wandb
    install(wandb)
    driver = sys.argv.pop(1)
    sys.argv[0] = driver
    runpy.run_path(driver, run_name='__main__')
