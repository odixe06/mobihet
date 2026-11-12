"""Matched in-fit/out-of-fit class-Q source; auxiliary state remains client-local."""
import hashlib
import json
import os
from pathlib import Path
import time
import zlib

import numpy as np
import torch

from core import data as D, grouped, local
from methods.mobihet_refine import Method as CBB


def frame(codes, present, counts=None):
    mask = int(sum(1 << int(c) for c in np.flatnonzero(present))).to_bytes(2, "little")
    raw = codes[present].astype(np.uint8).tobytes()
    if counts is not None:
        raw = counts[present].astype("<i4").tobytes() + raw
    packed = zlib.compress(raw, 6)
    body = b"\x01" + packed if len(packed) < len(raw) else b"\x00" + raw
    return bytes(32) + mask + body


def store(path, **arrays):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def wire(frames):
    return dict(frames=np.frombuffer(b"".join(frames), np.uint8),
                offsets=np.r_[0, np.cumsum([len(f) for f in frames])])


@torch.no_grad()
def probe(method, data):
    """Eval raw logits, no BN update or dropout; same batching for E/H sizes."""
    env = method.env
    sums = torch.zeros(env.K, 16, 16, device=env.dev, dtype=torch.float64)
    counts = torch.zeros(env.K, 16, device=env.dev, dtype=torch.int64)
    cm = torch.zeros(env.K, 256, device=env.dev, dtype=torch.int64)
    # Fixed 512-row chunks, up to 16 models. Probe E/H have identical client sizes.
    for begin in range(0, env.K, 16):
        idx = torch.arange(begin, min(begin + 16, env.K), device=env.dev)
        n = data.off[idx + 1] - data.off[idx]
        P, B = env.lay.params(method.priv_p[idx]), env.lay.bufs(method.priv_b[idx])
        g = len(idx)
        for start in range(0, int(n.max()), 512):
            j = start + torch.arange(512, device=env.dev)[None]
            valid = j < n[:, None]
            rows = data.off[idx, None] + torch.where(valid, j, 0)
            # Empty clients use a safe global row, always masked from statistics.
            rows = torch.where(valid, rows, 0)
            with torch.autocast("cuda", dtype=torch.float16, enabled=env.dev.type == "cuda"):
                z, _ = grouped.EVAL(P, B, data.X[rows], env.cfg, train=False)
            p = z.float().softmax(-1)
            assert bool(torch.isfinite(p).all())
            y = data.Y[rows].long()
            gi = torch.arange(g, device=env.dev)[:, None].expand_as(y)
            s = torch.zeros(g, 16, 16, device=env.dev, dtype=torch.float64)
            c = torch.zeros(g, 16, device=env.dev, dtype=torch.int64)
            s.index_put_((gi, y), p.double() * valid[..., None], accumulate=True)
            c.index_put_((gi, y), valid.long(), accumulate=True)
            sums[idx] += s; counts[idx] += c
            code = gi * 256 + y * 16 + z.argmax(-1)
            cm[idx] += torch.bincount(code[valid], minlength=g * 256).view(g, 256)
    mean = sums / counts[..., None].clamp_min(1)
    codes = (mean * 255).round().byte()
    return codes.cpu().numpy(), counts.cpu().numpy(), cm.view(env.K, 16, 16).cpu().numpy(), mean.cpu().numpy()


class Method(CBB):
    def __init__(self, env):
        super().__init__(env)
        self.source_mode = env.hp.get("source_mode", "C")
        assert self.source_mode in {"C", "S", "E", "H"}
        self.source_root = Path(env.hp.get("source_root", "."))
        assert self.q_agg == "count" and self.refine_loss == "forward" and self.real_smoothing == 0
        self.dest = Path(os.environ.get("MOBIHET_RECOVERY_DIR", ".")).parent / "source"
        self.dest.mkdir(exist_ok=True, parents=True)
        if self.source_mode == "S":
            from scripts.b18_partition import read, subset
            part, _ = read(self.source_root)
            full = D.load_full_foldtrain(D.find_root(), "train", env.ids.tolist(), None, folds=["train"])
            self.probe_e, self.probe_h = (subset(full, part, bit).to(env.dev) for bit in (2, 4))
            assert np.array_equal(self.probe_e.sizes, self.probe_h.sizes)
            del full
        elif self.source_mode in {"E", "H"}:
            self.shadow_config = json.loads((self.source_root / "shadow/config.json").read_text())

    def payload_bytes(self):
        body = super().payload_bytes()
        if self.source_mode in {"E", "H"}:
            shadow = self.shadow_config["payload_bytes"]["bootstrap"]
            body = body | {"bootstrap_main": body["bootstrap"].copy(), "bootstrap_shadow": shadow,
                           "bootstrap": {k: body["bootstrap"][k] + shadow[k] for k in ("up", "down")}}
        return body

    def _round_private(self, r, lr):
        if self.source_mode == "C":
            return super()._round_private(r, lr)
        if self.source_mode == "S":
            info = super()._round_private(r, lr)
            before = self.priv_b.clone()
            rng = torch.get_rng_state()
            crng = torch.cuda.get_rng_state(self.env.dev) if self.env.dev.type == "cuda" else None
            started = time.monotonic()
            e, n, ce, me = probe(self, self.probe_e)
            h, nh, ch, mh = probe(self, self.probe_h)
            assert np.array_equal(n, nh) and torch.equal(before, self.priv_b)
            assert torch.equal(rng, torch.get_rng_state())
            if crng is not None:
                assert torch.equal(crng, torch.cuda.get_rng_state(self.env.dev))
            frames_e = [frame(q, c > 0, c) for q, c in zip(e, n)]
            frames_h = [frame(q, c > 0, c) for q, c in zip(h, n)]
            store(self.dest / f"rank{self.env.rank}_r{r:03d}.npz", ids=self.env.ids, counts=n,
                  E=e, H=h, mean_E=me, mean_H=mh, cm_E=ce, cm_H=ch,
                  E_frames=wire(frames_e)["frames"], E_offsets=wire(frames_e)["offsets"],
                  H_frames=wire(frames_h)["frames"], H_offsets=wire(frames_h)["offsets"])
            cost = torch.tensor([sum(map(len, frames_e)), sum(map(len, frames_h)), len(n), int(n.sum())],
                                device=self.env.dev, dtype=torch.float64)
            cost = self.env.allreduce(cost)
            info.update(source_probe_E_up=float(cost[0] / cost[2]), source_probe_H_up=float(cost[1] / cost[2]),
                        source_probe_rows_each=int(cost[3]), source_probe_seconds=time.monotonic() - started)
            return info
        env, original = self.env, local.local_epoch
        paths = sorted((self.source_root / "shadow/source").glob(f"rank*_r{r:03d}.npz"))
        assert len(paths) == env.world
        shards = [dict(np.load(p)) for p in paths]
        ids = np.concatenate([s["ids"] for s in shards])
        assert len(np.unique(ids)) == len(env.all_ids) and np.array_equal(np.sort(ids), env.all_ids)
        order = np.argsort(ids)
        pos = np.searchsorted(ids[order], env.ids)
        codes = np.concatenate([s[self.source_mode] for s in shards])[order][pos]
        nc = np.concatenate([s["counts"] for s in shards])[order][pos]
        source = torch.as_tensor(codes, device=env.dev)
        eligible = torch.as_tensor(nc >= 8, device=env.dev)
        accum = torch.zeros(16, 16, dtype=torch.float64, device=env.dev)
        records, frames = [], []

        def collect(lay, flat, bufs, data, idx, *args, **kwargs):
            st = original(lay, flat, bufs, data, idx, *args, **kwargs)
            G = len(idx)
            count = self.cnt[idx]
            online = ((self._csum[:G] / count[..., None].clamp_min(1)).clamp(0, 1) * 255).round().byte()
            mixed = torch.where(eligible[idx, :, None], source[idx], online)
            # Match Base's Q8 -> fp32 -> fp16 client-mean decoding exactly.
            decoded = (mixed.float() / 255).half().double()
            accum.add_((decoded * count[..., None].double()).sum(0))
            ix = idx.cpu().numpy()
            records.append((env.ids[ix], count.long().cpu().numpy(), mixed.cpu().numpy(), online.cpu().numpy()))
            for k, q, present in zip(ix, online.cpu().numpy(), (count > 0).cpu().numpy()):
                frames.append(frame(q, present & ~(nc[k] >= 8)))
            return st

        local.local_epoch = collect
        try:
            info = super()._round_private(r, lr)
        finally:
            local.local_epoch = original
        total, n = env.allreduce(accum), env.allreduce(self.cnt.sum(0).double())
        # Same fp16 server mean then renormalization, followed by Q8 downlink as CBB.
        q = (total / n[:, None].clamp_min(1)).float().half().float()
        q = q / q.sum(-1, keepdim=True).clamp_min(1e-6)
        down_codes = (q.clamp(0, 1) * 255).round().byte()
        self.Q = down_codes.float() / down_codes.float().sum(-1, keepdim=True).clamp_min(1)
        down = frame(down_codes.cpu().numpy(), np.ones(16, bool))
        cost = env.allreduce(torch.tensor([sum(map(len, frames)), env.K], device=env.dev, dtype=torch.float64))
        row = next(json.loads(s) for s in (self.source_root / "shadow/rounds.jsonl").read_text().splitlines()
                   if json.loads(s)["round"] == r)
        main_up = float(cost[0] / cost[1])
        info.update(source_main_up=main_up, source_main_down=len(down),
                    source_shadow_up=row["bytes_up"], source_shadow_down=row["bytes_down"],
                    source_measurement_up=row[f"source_probe_{self.source_mode}_up"],
                    bytes_up=main_up + row["bytes_up"] + row[f"source_probe_{self.source_mode}_up"],
                    bytes_down=len(down) + row["bytes_down"], source_mode=self.source_mode,
                    source_eligible_rows=int(env.allreduce((self.cnt.double() * eligible).sum())),
                    source_files_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])
        store(self.dest / f"rank{env.rank}_r{r:03d}.npz", ids=np.concatenate([x[0] for x in records]),
              counts=np.concatenate([x[1] for x in records]), mixed=np.concatenate([x[2] for x in records]),
              online=np.concatenate([x[3] for x in records]), **wire(frames), global_sum=total.cpu().numpy(),
              global_count=n.cpu().numpy(), down=np.frombuffer(down, np.uint8), Q=self.Q.cpu().numpy())
        return info
