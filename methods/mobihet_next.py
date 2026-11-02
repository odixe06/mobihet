"""Q20: preserve streaming classQ, test mixed support/input consensus/serverweight/reception."""
import hashlib
import time

import torch
import torch.nn.functional as F

from core import grouped, local
from methods.mobihet import Method as Base
from methods.mobihet_study import decode_probability, reception, require_shared
from methods.mobihet_study_wire import roundtrip
from scripts.run_p21_recovery import GuardedMethod

VARIANTS = ("stream", "query", "mixed_class", "mixed_input", "input_sqrt", "real", "margin", "strength")


def signature(q, x, target, variant):
    if q is None: return None
    values = (q,) if variant == "stream" else (q, x, target)
    digest = hashlib.sha256()
    for value in values:
        if value is None: return None
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class Method(GuardedMethod):
    def __init__(self, env):
        self.variant = env.hp["next_mode"]
        assert self.variant in VARIANTS and env.hp["study_mode"] == "class"
        super().__init__(env)
        assert self.aux == 64 and self.bank_size == 1024 and self.q_agg == "count"
        self.class_sums = self.class_totals = None
        if self.variant != "stream": self.bank_X, self.bank_y = self.make_bank(0)

    def _synth(self, ctx):
        if self.variant in {"stream", "query"}: return Base._synth(self, ctx)
        old_aux, old_labels = self.aux, self.labels
        try:
            self.aux, self.labels = 32, old_labels[:32]
            x, labels = Base._synth(self, ctx)
        finally:
            self.aux, self.labels = old_aux, old_labels
        picks = torch.randint(64, labels.shape, device=x.device, generator=self.gen_g)
        self.bank_index = picks * 16 + labels
        bank = self.bank_X[self.bank_index].to(ctx.x.dtype)
        return torch.cat([x, bank], 1), torch.cat([labels, labels], 1)

    def _loss_private(self, ctx):
        if self.variant in {"stream", "query"}: return Base._loss_private(self, ctx)
        G, b = ctx.Gfix, ctx.x.shape[1]
        x, labels = self._synth(ctx)
        mask = torch.cat([ctx.mask, ctx.mask.any(1)[:, None].expand(G, 64)], 1)
        z, new_b = grouped.TRAIN(ctx.P, ctx.B, torch.cat([ctx.x, x], 1), self.env.cfg, mask=mask, train=True)
        ce = local.ce_loss(z[:, :b] + self._shift[:, None], ctx.y, ctx.mask)
        lsp = F.log_softmax(z, -1)
        with torch.no_grad():
            probability = torch.where(ctx.mask[..., None], lsp[:, :b].detach().exp(), 0.)
            self._csum.index_put_((torch.arange(G, device=z.device)[:, None].expand_as(ctx.y), ctx.y), probability, accumulate=True)
        if self.Q is None:
            return ce + F.cross_entropy(z[:, b:].reshape(-1, 16), labels.reshape(-1), reduction="none").view(G, 64).mean(1), new_b
        target = self.Q[labels]
        if self.variant != "mixed_class":
            target = torch.cat([target[:, :32], self.bank_target[self.bank_index]], 1)
        lt = target.clamp_min(1e-6).log()
        synthetic = (lt.exp() * (lt - lsp[:, b:])).sum(-1).mean(1)
        real = torch.zeros_like(ce)
        if self.variant in {"real", "margin", "strength"}:
            qr = self.Q[ctx.y].clamp_min(1e-6).log()
            kr = (qr.exp() * (qr - lsp[:, :b])).sum(-1)
            coefficient = self.receive_weight[ctx.clients].gather(1, ctx.y)
            real = local.masked_mean(coefficient * kr, ctx.mask)
        return ce + synthetic + real, new_b

    def _round_mixed(self, r, lr):
        env, lay = self.env, self.env.lay
        losses, steps = torch.zeros(env.K, device=env.dev), torch.zeros(2, device=env.dev)
        sums = torch.zeros(16, 16, dtype=torch.float64, device=env.dev)
        if self.Q is not None and self.variant in {"real", "margin", "strength"}:
            if self.variant == "real":
                keep = (self.Q.argmax(1) == torch.arange(16, device=env.dev)).float()
                self.receive_weight = keep[None].expand(env.K, -1)
            else: self.receive_weight = reception(self.Q, self.cnt, self.variant)
        for idx in env.groups_idx():
            G, Gf = len(idx), local.group_size(env.train.off, idx, env.hp)
            self._shift = local.pad_rows(self.la_shift[idx], Gf)
            self._csum = torch.zeros(Gf, 16, 16, device=env.dev)
            flat, bufs = self.priv_p[idx], self.priv_b[idx]
            stat = local.local_epoch(lay, flat, bufs, env.train, idx, env.cfg, env.hp, lr, env.gen, loss_fn=self._loss_private)
            self.priv_p[idx], self.priv_b[idx] = flat, bufs
            losses[idx] = stat["loss"]
            steps += torch.stack([stat["applied"].sum(), stat["skipped"].sum()])
            means = self._csum[:G] / self.cnt[idx, :, None].clamp_min(1)
            decoded, self.class_up_bytes = roundtrip(means, 4, r)
            sums += (decoded.double() * self.cnt[idx, :, None].double()).sum(0)
        steps = env.allreduce(steps)
        self.class_sums = env.allreduce(sums)
        self.class_totals = env.allreduce(self.cnt.sum(0).double())
        body = (self.class_sums / self.class_totals[:, None].clamp_min(1)).float()
        decoded, down = roundtrip(body[None], 5, r)
        self.Q = decode_probability(decoded[0])
        self.version = r
        count = env.allreduce(env.n.sum().clone())
        return {"train_loss": float(env.allreduce((losses * env.n).sum()) / count),
                "skipped_frac": float(steps[1] / steps.sum().clamp_min(1)), "bytes_up": self.class_up_bytes, "bytes_down": down}

    def _query_bank(self, r):
        env = self.env
        xs, ys = self.make_bank(r)
        sums = torch.zeros(len(xs), 16, dtype=torch.float64, device=env.dev)
        weights = torch.zeros(len(xs), dtype=torch.float64, device=env.dev)
        started = time.monotonic()
        for idx in env.groups_idx():
            probability = self._probs(self.priv_p[idx], self.priv_b[idx], xs, len(idx), chunk=128)
            decoded, up = roundtrip(probability, 1, r)
            decoded = decode_probability(decoded)
            weight = self.cnt[idx][:, ys].double()
            if self.variant == "input_sqrt": weight = weight.sqrt()
            sums += (decoded.double() * weight[..., None]).sum(0)
            weights += weight.sum(0)
        self.aggregate_sums, self.aggregate_weights = env.allreduce(sums), env.allreduce(weights)
        assert bool((self.aggregate_weights > 0).all())
        self.bank_body = (self.aggregate_sums / self.aggregate_weights[:, None]).float().half()
        decoded, down = roundtrip(self.bank_body[None], 2, r)
        self.bank_X, self.bank_y, self.bank_target = xs, ys, decode_probability(decoded[0])
        if env.dev.type == "cuda": torch.cuda.synchronize(env.dev)
        return {"input_up_bytes": up, "input_down_bytes": down, "scoring_seconds_local": time.monotonic() - started,
                "teacher_received_agreement": float((self.bank_target.argmax(1) == ys).float().mean())}

    def round(self, r, lr):
        env = self.env
        if r == 1:
            count = int(env.allreduce(torch.tensor(int(grouped.TRAIN is not grouped.forward), device=env.dev)))
            self.record({"event": "compile_guard", "compiled_ranks": count, "world": env.world})
            assert count == env.world, "Q20 requires all compile gates PASS"
        previous = signature(self.Q, self.bank_X, self.bank_target, self.variant)
        self.guard_round, self.guard_calls = r, 0
        self.guard_active = self.variant != "stream"
        try:
            info = Base.round(self, r, lr) if self.variant in {"stream", "query"} else self._round_mixed(r, lr)
            if self.variant != "stream":
                query_info = self._query_bank(r)
                info["bytes_up"] += query_info["input_up_bytes"]
                info["bytes_down"] += query_info["input_down_bytes"]
                info.update(query_info)
        finally:
            self.guard_active = False
        assert self.guard_calls == (0 if self.variant == "stream" else len(env.groups))
        self.teacher_round = r
        current = signature(self.Q, self.bank_X, self.bank_target, self.variant)
        require_shared(env, current)
        info.update(teacher_consumed_round=None if r == 1 else r - 1, teacher_produced_round=r,
                    teacher_consumed_sha256=previous, teacher_produced_sha256=current)
        self.record({"event": "round_finite", "round": r, "groups": self.guard_calls, "clients": env.K})
        return info

    def diagnostics(self, r, out):
        old = getattr(self, "receive_weight", None)
        if self.variant in {"real", "margin", "strength"}:
            if self.variant == "real": self.receive_weight = (self.Q.argmax(1) == torch.arange(16, device=self.Q.device)).float()[None].expand(self.env.K, -1)
            else: self.receive_weight = reception(self.Q, self.cnt, self.variant)
        try: return super().diagnostics(r, out)
        finally:
            if old is None: self.__dict__.pop("receive_weight", None)
            else: self.receive_weight = old

    def state(self):
        return super().state() | {k: None if getattr(self, k) is None else getattr(self, k).detach().cpu()
                                 for k in ("class_sums", "class_totals")}

    def load_state(self, state):
        super().load_state(state)
        for k in ("class_sums", "class_totals"):
            setattr(self, k, None if state.get(k) is None else state[k].to(self.env.dev))
