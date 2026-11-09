"""Verified notebook checkpoint import, shared by same-owner and dataset handoffs."""


def kernel_import(run, round_, method, seed):
    return f'''parent_states = list(Path("/kaggle/input").glob("**/{run}/state.pt"))
assert len(parent_states) == 1, "Missing or ambiguous parent state"
parent_dir = parent_states[0].parent
for rank, name in enumerate(("state.pt", "state_rank1.pt")):
    assert (parent_dir/name).is_file(), name
    state = torch.load(parent_dir/name, map_location="cpu", weights_only=True, mmap=True)
    assert state["round"] == {round_} and "rng_cpu" in state and "rng_cuda" in state
    if rank == 0:
        assert state["method"] == {method!r} and state["hp"]["seed"] == {seed} and state["hp"]["rounds"] == 50
    del state
target = Path("/kaggle/working/{run}")
target.mkdir(exist_ok=True)
for item in parent_dir.iterdir():
    dest = target/item.name
    if item.is_dir(): shutil.copytree(item, dest, dirs_exist_ok=True, copy_function=shutil.copyfile)
    else: shutil.copyfile(item, dest)
print("resumed from verified two-rank checkpoint", str(parent_dir), "round", {round_})
'''


def dataset_import(run, round_, method, seed, digest, hp_sha256, cfg_sha256):
    verify = kernel_import(run,round_,method,seed)
    # Same rank/round/RNG checks, but the archive is first verified and extracted into working.
    checks = verify[verify.index('for rank, name'):verify.index('target = Path')]
    checks = checks.replace('    del state', f'''    if rank == 0:
        assert hashlib.sha256(json.dumps(state["hp"],sort_keys=True).encode()).hexdigest() == {hp_sha256!r}
        assert hashlib.sha256(json.dumps(state["cfg"],sort_keys=True).encode()).hexdigest() == {cfg_sha256!r}
    del state''')
    return f'''import tarfile
archives = list(Path("/kaggle/input").glob("**/resume.bin"))
assert len(archives) == 1, "Missing or ambiguous checkpoint archive"
with archives[0].open("rb") as fh: assert hashlib.file_digest(fh,"sha256").hexdigest() == {digest!r}
with tarfile.open(archives[0],"r") as tar: tar.extractall("/kaggle/working",filter="data")
parent_dir = Path("/kaggle/working/{run}")
{checks}
print("resumed from verified private dataset", "round", {round_})
'''
