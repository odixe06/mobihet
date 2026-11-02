"""P20 diagnostics: client reception, refreshed input teachers, server proxy, v4 missingness.

Only train aggregates or outputs on seeded synthetic inputs leave clients. Private weights/BN stay local.
Server proxy is independently trained from those aggregate outputs; client DAGSNet is unchanged.
"""
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from core import data, grouped, local
from core.optim import GroupAdamW
from methods.mobihet import Method as Base
from methods.mobihet_study_wire import roundtrip


MODES = ("real", "augment", "class", "gate", "margin", "strength", "bank_class", "bank_input",
         "proxy_bank", "proxy_stream", "proxy_hybrid", "bank_hard", "project_augment", "project_class",
         "proxy_weights_bank", "real_no_la")
BANK_MODES = {"bank_class", "bank_input", "proxy_bank", "proxy_stream", "proxy_hybrid", "bank_hard", "proxy_weights_bank"}
PROXY_MODES = {"proxy_bank", "proxy_stream", "proxy_hybrid", "proxy_weights_bank"}


def decode_probability(body):
    """fp16 body, fp32 normalized receiver values. Never aggregate unencoded source probabilities."""
    q = body.float()
    return q / q.sum(-1, keepdim=True).clamp_min(1e-12)


def teacher_signature(mode, q, bank_x, bank_target, teacher_p, teacher_b):
    values = (teacher_p, teacher_b) if mode == "proxy_stream" else (
        (bank_x, bank_target) if mode in BANK_MODES else (q,))
    if any(value is None for value in values): return None
    h = hashlib.sha256()
    for value in values:
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def require_shared(env, signature):
    """Synthetic row indices mean the same inputs/teacher on every simulated client and server rank."""
    if env.world > 1:
        signatures = [None] * env.world
        torch.distributed.all_gather_object(signatures, signature)
        assert len(set(signatures)) == 1, "Cross-rank synthetic/teacher identity mismatch; refuse aggregation"


def reception(q, counts, mode):
    diagonal = q.diag()
    other = q.masked_fill(torch.eye(16, dtype=torch.bool, device=q.device), -1).amax(1)
    margin = (diagonal - other).clamp_min(0)
    keep = (q.argmax(1) == torch.arange(16, device=q.device)).float()
    if mode == "margin":
        return margin[None].expand(len(counts), -1)
    # Same per-client count-weighted coefficient mass as margin, preserving the binary gate.
    scale = (counts * margin).sum(1) / (counts * keep).sum(1).clamp_min(1)
    return scale[:, None] * keep[None]


class Method(Base):
    def _bootstrap(self):
        if self.mode not in {"real", "real_no_la"}:
            return super()._bootstrap()
        env = self.env
        self.cnt = torch.zeros(env.K, 16, device=env.dev)
        client = torch.repeat_interleave(torch.arange(env.K, device=env.dev), env.train.off[1:] - env.train.off[:-1])
        self.cnt.index_put_((client, env.train.Y.long()), torch.ones(len(client), device=env.dev), accumulate=True)
        self.labels = torch.arange(self.aux, device=env.dev) % 16
        self.la_shift = torch.zeros_like(self.cnt)
        self.bootstrap_bytes = {"up": 0, "down": 0}
        if self.mode == "real":
            total = env.allreduce(self.cnt.sum(0).double())
            self.la_shift = ((self.cnt + 1) / (self.cnt.sum(1, keepdim=True) + 16)).log() - ((total + 1) / (total.sum() + 16)).log().float()
            self.bootstrap_bytes = {"up": 64, "down": 64}

    def __init__(self, env):
        self.mode = env.hp["study_mode"]
        assert self.mode in MODES
        self.project = self.mode.startswith("project_")
        self.bank_X = self.bank_y = self.bank_target = self.bank_body = None
        self.teacher_p = self.teacher_b = None
        self.aggregate_sums = self.aggregate_weights = None
        self.teacher_round = 0
        self.projection_pairs = []
        if self.project:
            scaler_path = Path(data.find_root()) / "scaler.json"
            scaler_raw = scaler_path.read_bytes()
            scaler = json.loads(scaler_raw)
            order = json.loads((Path(__file__).resolve().parents[1] / "knowledge/input_order.json").read_text())["inputs"]
            self.projection_pairs = [(order.index(name), order.index(flag)) for name, flag in scaler["nan_indicator"].items()]
            self.scaler_sha256 = hashlib.sha256(scaler_raw).hexdigest()
        super().__init__(env)
        assert self.arm == "none" and self.kd_weight == "fixed" and self.q_codec == "mean_fp16"
        assert self.anchor == "off" and self.kd_bins == 1 and self.q_collect == "online"
        self.bank_size = env.hp.get("teacher_rows", 1024)
        assert self.bank_size % 16 == 0
        self.server_p, self.server_b = self.g_p.clone(), self.g_b.clone()
        if self.mode == "real_no_la":
            self.la_shift = torch.zeros_like(self.la_shift)
        if self.mode in BANK_MODES and self.mode != "proxy_stream":
            self.bank_X, self.bank_y = self.make_bank(0)

    def _synth(self, ctx):
        if self.mode in BANK_MODES and self.mode != "proxy_stream" and self.bank_X is not None:
            # Class-balanced sequence; independent random support row within that class.
            labels = self.labels[None].expand(ctx.Gfix, -1)
            picks = torch.randint(self.bank_size // 16, labels.shape, device=self.env.dev, generator=self.gen_g)
            self.bank_index = picks * 16 + labels
            return self.bank_X[self.bank_index].to(ctx.x.dtype), labels
        x, y = super()._synth(ctx)
        if self.project:
            x = x.clone()
            for flag in sorted({f for _, f in self.projection_pairs}):
                x[..., flag] = (x[..., flag] >= 0.5).to(x.dtype)
            for feature, flag in self.projection_pairs:
                x[..., feature] = torch.where(x[..., flag] > 0, 0., x[..., feature])
        return x, y

    def make_bank(self, r):
        """Fresh copula support per round, identical across ranks/arms; not raw local rows."""
        old_gen, old_aux, old_labels = self.gen_g, self.aux, self.labels
        try:
            self.gen_g = torch.Generator(device=self.env.dev).manual_seed(self.env.hp["seed"] * 100003 + r)
            self.aux = self.env.hp.get("teacher_rows", 1024)
            self.labels = torch.arange(self.aux, device=self.env.dev) % 16
            ctx = SimpleNamespace(Gfix=1, x=self.env.train.X[:1][None], clients=torch.zeros(1, dtype=torch.long, device=self.env.dev))
            x, y = Base._synth(self, ctx)
            signature = hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            require_shared(self.env, signature)
            return x[0].contiguous(), y[0].contiguous()
        finally:
            self.gen_g, self.aux, self.labels = old_gen, old_aux, old_labels

    def _loss_private(self, ctx):
        if self.mode not in BANK_MODES and self.mode not in {"margin", "strength"}:
            return super()._loss_private(ctx)
        G, b = ctx.Gfix, ctx.x.shape[1]
        xg, labels = self._synth(ctx)
        xp = torch.cat([ctx.x, xg], 1)
        mask = torch.cat([ctx.mask, ctx.mask.any(1)[:, None].expand(G, self.aux)], 1)
        z, new_b = grouped.TRAIN(ctx.P, ctx.B, xp, self.env.cfg, mask=mask, train=True)
        ce = local.ce_loss(z[:, :b] + self._shift[:, None], ctx.y, ctx.mask)
        lsp = F.log_softmax(z, -1)
        if self.mode in {"margin", "strength"}:
            with torch.no_grad():
                prob = torch.where(ctx.mask[..., None], lsp[:, :b].detach().exp(), 0.)
                self._csum.index_put_((torch.arange(G, device=z.device)[:, None].expand_as(ctx.y), ctx.y), prob, accumulate=True)
            if self.Q is None:
                return ce - lsp[:, b:].gather(-1, labels[..., None]).squeeze(-1).mean(1), new_b
            lt = self.Q.clamp_min(1e-6).log()
            kr = (lt[ctx.y].exp() * (lt[ctx.y] - lsp[:, :b])).sum(-1)
            coeff = self.receive_weight[ctx.clients].gather(1, ctx.y)
            real = local.masked_mean(coeff * kr, ctx.mask)
            kg = (lt[labels].exp() * (lt[labels] - lsp[:, b:])).sum(-1).mean(1)
            return ce + real + kg, new_b
        if self.bank_target is None or self.mode == "bank_hard":
            return ce - lsp[:, b:].gather(-1, labels[..., None]).squeeze(-1).mean(1), new_b
        if self.mode == "bank_class":
            target = self.Q[labels]
        elif self.mode == "proxy_stream":
            with torch.no_grad():
                target = self.proxy_probability(xg)
        else:
            target = self.bank_target[self.bank_index]
        lt = target.clamp_min(1e-6).log()
        kg = (lt.exp() * (lt - lsp[:, b:])).sum(-1).mean(1)
        return ce + kg, new_b

    @torch.no_grad()
    def proxy_probability(self, x):
        # Flatten input groups: ONE shared teacher, fixed running BN, no private teacher replicas trained locally.
        shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        p, b = self.env.lay.params(self.teacher_p[None]), self.env.lay.bufs(self.teacher_b[None])
        result = []
        for start in range(0, len(x), 4096):
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.env.dev.type == "cuda"):
                z, _ = grouped.EVAL(p, b, x[start:start + 4096][None], self.env.cfg, train=False)
            result.append(z[0].float().softmax(-1))
        return torch.cat(result).reshape(*shape, 16)

    def fit_proxy(self, x, target):
        """Server-only DAGSNet; warm-start; no client parameters/BN sent to server."""
        env, lay = self.env, self.env.lay
        t0 = time.monotonic()
        if env.rank == 0:
            flat, bufs = self.server_p[None].clone(), self.server_b[None].clone()
            opt = GroupAdamW(flat, wd=env.hp["weight_decay"], clip=env.hp["clip"])
            leaves = {k: v.detach().requires_grad_(True) for k, v in lay.params(flat).items()}
            gen = torch.Generator(device=env.dev).manual_seed(env.hp["seed"] * 701 + self.version)
            cfg = dict(env.cfg, dropout=0.)
            objective = target
            if self.mode == "proxy_hybrid":
                objective = .5 * target + .5 * F.one_hot(self.next_y, 16)
            for _ in range(env.hp.get("proxy_steps", 200)):
                idx = torch.randint(len(x), (128,), device=env.dev, generator=gen)
                # fp32 server optimization; grouped reference architecture, train BN on synthetic only.
                z, nb = grouped.TRAIN(leaves, lay.bufs(bufs), x[idx][None].float(), cfg,
                                      mask=torch.ones(1, len(idx), dtype=torch.bool, device=env.dev), train=True)
                loss = -(objective[idx] * F.log_softmax(z[0], -1)).sum(-1).mean()
                loss.backward()
                grad = torch.cat([leaves[k].grad.reshape(1, -1) for k, _ in lay.p], 1)
                for v in leaves.values(): v.grad = None
                _, ok = opt.step(flat, grad, .001, torch.ones(1, dtype=torch.bool, device=env.dev))
                assert bool(ok.all()), "nonfinite server proxy update"
                bufs.copy_(lay.flat_bufs(nb))
            self.server_p, self.server_b = flat[0].detach(), bufs[0].detach()
        if env.world > 1:
            torch.distributed.broadcast(self.server_p, 0)
            torch.distributed.broadcast(self.server_b, 0)
        assert torch.isfinite(self.server_p).all() and torch.isfinite(self.server_b).all()
        # Actual lossy downlink replica. Both bank and streaming predictions use decoded values.
        if self.mode in {"proxy_stream", "proxy_weights_bank"}:
            replica, self.proxy_packet_bytes = roundtrip(torch.cat([self.server_p, self.server_b])[None, None], 3, self.version)
            self.teacher_p, self.teacher_b = replica[0, 0, :lay.P].float(), replica[0, 0, lay.P:].float()
        else:
            # Same fp16 teacher function even when server sends only its probabilities.
            self.teacher_p = self.server_p.half().float()
            self.teacher_b = self.server_b.half().float()
        pred = self.proxy_probability(x[None])[0]
        return pred, time.monotonic() - t0

    def round(self, r, lr):
        consumed = self.teacher_round or (r - 1 if r > 1 and not self.no_kd else 0)
        used_signature = None if self.no_kd else teacher_signature(self.mode, self.Q, self.bank_X,
            self.bank_target, self.teacher_p, self.teacher_b)
        if self.mode in {"margin", "strength"} and self.Q is not None:
            self.receive_weight = reception(self.Q, self.cnt, self.mode).detach()
        if self.mode not in BANK_MODES:
            info = super().round(r, lr)
        else:
            info = self.round_bank(r, lr)
        info["teacher_consumed_round"] = consumed if consumed else None
        info["teacher_produced_round"] = r if not self.no_kd else None
        info["teacher_consumed_sha256"] = used_signature
        info["teacher_produced_sha256"] = None if self.no_kd else teacher_signature(self.mode, self.Q, self.bank_X,
            self.bank_target, self.teacher_p, self.teacher_b)
        require_shared(self.env, info["teacher_produced_sha256"])
        if self.mode in {"margin", "strength"} and r > 1:
            info["reception_coefficient_mean"] = float(self.env.allreduce((self.receive_weight * self.cnt).sum()) /
                                                        self.env.allreduce(self.cnt.sum().clone()).clamp_min(1))
        return info

    def round_bank(self, r, lr):
        env, lay = self.env, self.env.lay
        losses = torch.zeros(env.K, device=env.dev)
        steps = torch.zeros(2, device=env.dev)
        xs, ys = self.make_bank(r)
        sums = torch.zeros(len(xs), 16, dtype=torch.float64, device=env.dev)
        weights = torch.zeros(len(xs), dtype=torch.float64, device=env.dev)
        started = time.monotonic()
        scoring_seconds = 0.
        for idx in env.groups_idx():
            Gf = local.group_size(env.train.off, idx, env.hp)
            self._shift = local.pad_rows(self.la_shift[idx], Gf)
            flat, bufs = self.priv_p[idx], self.priv_b[idx]
            st = local.local_epoch(lay, flat, bufs, env.train, idx, env.cfg, env.hp, lr, env.gen, loss_fn=self._loss_private)
            self.priv_p[idx], self.priv_b[idx] = flat, bufs
            losses[idx] = st["loss"]
            steps += torch.stack([st["applied"].sum(), st["skipped"].sum()])
            if self.mode != "bank_hard":
                t0 = time.monotonic()
                prob = self._probs(flat, bufs, xs, len(idx), chunk=128)
                packet, self.bank_upload_bytes = roundtrip(prob, 1, r)
                decoded = decode_probability(packet)
                w = self.cnt[idx][:, ys].double()
                sums += (decoded.double() * w[..., None]).sum(0)
                weights += w.sum(0)
                if env.dev.type == "cuda": torch.cuda.synchronize(env.dev)
                scoring_seconds += time.monotonic() - t0
        steps = env.allreduce(steps)
        n = env.allreduce(env.n.sum().clone())
        info = {"train_loss": float(env.allreduce((losses * env.n).sum()) / n),
                "skipped_frac": float(steps[1] / steps.sum().clamp_min(1)), "scoring_seconds_local": scoring_seconds}
        self.version = r
        if self.mode == "bank_hard":
            self.bank_X, self.bank_y = xs, ys
            info.update(bytes_up=0, bytes_down=0)
            return info
        sums, weights = env.allreduce(sums), env.allreduce(weights)
        assert bool((weights > 0).all())
        self.aggregate_sums, self.aggregate_weights = sums, weights
        # Every message refers to the exact same seeded row order.
        # Server retained aggregate is not yet a downlink. Encode the representation chosen below.
        self.bank_body = (sums / weights[:, None]).float().half()
        aggregate = decode_probability(self.bank_body)
        self.next_y = ys
        if self.mode in PROXY_MODES:
            prediction, seconds = self.fit_proxy(xs, aggregate)
            info["server_seconds"] = seconds
            info["proxy_fit_kl"] = float((aggregate * (aggregate.clamp_min(1e-6).log() - prediction.clamp_min(1e-6).log())).sum(1).mean())
            if self.mode in {"proxy_stream", "proxy_weights_bank"}:
                # Regenerated locally from decoded proxy, with the same cache precision as logits carrier.
                self.bank_target = decode_probability(prediction.half())
                down_bytes = self.proxy_packet_bytes
            else:
                packet, down_bytes = roundtrip(prediction[None], 2, r)
                self.bank_target = decode_probability(packet[0])
        else:
            packet, down_bytes = roundtrip(self.bank_body[None], 2, r)
            self.bank_target = decode_probability(packet[0])
        # bank_class receives a class mean of exactly the same aggregate input targets.
        class_table = torch.zeros(16, 16, device=env.dev).index_add_(0, ys, aggregate) / (len(ys) // 16)
        self.Q = decode_probability(class_table.half())
        if self.mode == "bank_class":
            packet, down_bytes = roundtrip(class_table[None], 2, r)
            self.Q = decode_probability(packet[0])
            self.bank_target = self.Q[ys]
        self.bank_X, self.bank_y = xs, ys
        self.teacher_round = r
        info.update(bytes_up=self.bank_upload_bytes, bytes_down=down_bytes,
                    teacher_aggregate_agreement=float((aggregate.argmax(1) == ys).float().mean()),
                    teacher_received_agreement=float((self.bank_target.argmax(1) == ys).float().mean()),
                    teacher_received_entropy=float(-(self.bank_target * self.bank_target.clamp_min(1e-6).log()).sum(1).mean()),
                    bank_seconds_local=time.monotonic() - started)
        return info

    def diagnostics(self, r, out):
        from methods.mobihet_study_diagnostics import run
        return run(self, r, out)

    def state(self):
        return super().state() | {k: None if getattr(self, k) is None else getattr(self, k).detach().cpu()
            for k in ("bank_X", "bank_y", "bank_target", "bank_body", "aggregate_sums", "aggregate_weights", "server_p", "server_b", "teacher_p", "teacher_b")} | {
                "teacher_round": self.teacher_round, "gen_g": self.gen_g.get_state().cpu()}

    def load_state(self, s):
        super().load_state(s)
        for k in ("bank_X", "bank_y", "bank_target", "bank_body", "aggregate_sums", "aggregate_weights", "server_p", "server_b", "teacher_p", "teacher_b"):
            setattr(self, k, None if s.get(k) is None else s[k].to(self.env.dev))
        self.teacher_round = s.get("teacher_round", 0)
        self.gen_g.set_state(s["gen_g"].cpu())
