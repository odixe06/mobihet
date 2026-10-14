"""Full-data V8 factorial with compiled training templates and the original V8 losses."""
import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
from parent_run import Run as FullRun
from mh.v2_methods import finite
from full_method import Method, ARMS
import full_driver as driver


class Run(FullRun):
    def compile_preflight(self):
        torch._dynamo.config.recompile_limit = 64
        auxiliary = 'gaussian' in self.a.arm or 'hybrid' in self.a.arm
        size = self.a.batch + (self.a.aux_batch if auxiliary else 0)
        for role, model in [('private', self.work_private), ('proxy', self.work_proxy), ('eval', self.eval_template)]:
            model.train(role != 'eval')
            batch = size if role != 'eval' else 16384
            x = self.X[0][:batch].float()
            if len(x) < batch:
                x = x.repeat((batch + len(x) - 1) // len(x), 1)[:batch]
            saved = {k: v.clone() for k, v in model.state_dict().items()}
            rng = torch.cuda.get_rng_state()
            dropout = [(m, m.p) for m in model.modules() if isinstance(m, torch.nn.Dropout)]
            for m, _ in dropout:
                m.p = 0.
            reference = copy.deepcopy(model)
            compiled = torch.compile(model, mode='reduce-overhead', dynamic=False)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                torch.compiler.cudagraph_mark_step_begin()
                expected = reference(x)[0].float()
                actual = compiled(x)[0].float().clone()
            error = float((actual - expected).abs().max())
            decisive = expected.topk(2, dim=1).values.diff(dim=1).abs().squeeze(1) > .02
            assert error <= .005 and torch.equal(actual.argmax(1)[decisive], expected.argmax(1)[decisive]), ('compile gate', role, error)
            model.load_state_dict(saved, strict=True)
            torch.cuda.set_rng_state(rng)
            for m, p in dropout:
                m.p = p
            self.compiled[id(model)] = (compiled, batch)
            self.compile_gates.append(dict(role=role, rows=batch, max_error=error, decisive=int(decisive.sum())))
            del reference, actual, expected, x
            torch.cuda.empty_cache()
        print('COMPILE PREFLIGHT PASS', self.compile_gates, flush=True)

    def private_epoch(self, method, k, r):
        started = time.time()
        original = self.priv[k]
        self.work_private.load_state_dict(original.state_dict(), strict=True)
        self.work_proxy.load_state_dict(method.proxy[k].state_dict(), strict=True)
        private, proxy = self.work_private, self.work_proxy
        private.train(); proxy.train()
        optimizer = torch.optim.AdamW(list(private.parameters()) + list(proxy.parameters()), lr=self.lr(r), weight_decay=0.)
        scaler = torch.amp.GradScaler()
        generator = self.gen(r, k, 0)
        torch.manual_seed(int(np.random.SeedSequence([self.a.seed, r, k, 7]).generate_state(1)[0]))
        order = torch.randperm(self.n[k], device=self.dev, generator=generator)
        total = torch.zeros(4, device=self.dev)
        applied = skipped = 0
        for start in range(0, len(order), self.a.batch):
            torch.compiler.cudagraph_mark_step_begin()
            index = order[start:start + self.a.batch]
            real_x, real_y = self.X[k][index].float(), self.Y[k][index]
            labels = torch.arange(self.a.aux_batch, device=self.dev) % 16
            synthetic = method.draw(labels, generator, k) if method.aux else torch.randn(len(labels), 66, device=self.dev, generator=generator)
            mixed = torch.cat([real_x, synthetic]) if method.aux else real_x
            with torch.autocast('cuda', dtype=torch.float16):
                zs = self.forward(private, mixed)[0].float()
                zr = self.forward(proxy, mixed)[0].float()
                ce_s = F.cross_entropy(zs[:len(index)], real_y)
                ce_r = F.cross_entropy(zr[:len(index)], real_y)
                kd_s = kd_r = ce_s.detach() * 0
                if method.kd:
                    ls, lr = F.log_softmax(zs[:len(index)], 1), F.log_softmax(zr[:len(index)], 1)
                    weight = 1 / (ce_s + ce_r).detach().clamp_min(1e-6)
                    kd_s = weight * F.kl_div(ls, lr.detach(), log_target=True, reduction='batchmean')
                    kd_r = weight * F.kl_div(lr, ls.detach(), log_target=True, reduction='batchmean')
                aux_s = aux_r = ce_s.detach() * 0
                if method.aux:
                    aux_s = F.cross_entropy(zs[len(index):], labels)
                    aux_r = F.cross_entropy(zr[len(index):], labels)
                else:
                    private.eval(); proxy.eval()
                    dummy = (private(synthetic)[0].float().sum() + proxy(synthetic)[0].float().sum()) * 0
                    private.train(); proxy.train()
                extra_s = kd_s + self.a.aux_weight * aux_s
                extra_r = kd_r + self.a.aux_weight * aux_r
                loss = ce_s + ce_r + extra_s + extra_r
                if not method.aux:
                    loss = loss + dummy
            assert torch.isfinite(loss), ('hybrid full loss', k, r)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            for model in (private, proxy):
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.a.clip)
            before = scaler.get_scale(); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() < before:
                skipped += 1
            else:
                applied += 1
                total += torch.stack([ce_s.detach(), extra_s.detach(), ce_r.detach(), extra_r.detach()])
            if self.a.wandb_online and (start // self.a.batch + 1) % 1024 == 0:
                import wandb
                wandb.run.log({'progress/round_in_flight': r, 'progress/client': k,
                    'progress/client_steps': start // self.a.batch + 1, 'progress/applied': applied,
                    'progress/skipped': skipped, 'progress/seconds': time.time() - started,
                    'progress/loss': float(loss.detach()), 'progress/grad_norm': float(norm),
                    'progress/max_logit': float(zs.detach().abs().max()),
                    'progress/softmax_entropy': float(-(F.softmax(zs.detach(), 1) * F.log_softmax(zs.detach(), 1)).sum(1).mean())})
        assert applied > 0 and skipped <= 16, (k, r, applied, skipped)
        finite(private); finite(proxy)
        original.load_state_dict(private.state_dict(), strict=True)
        method.proxy[k].load_state_dict(proxy.state_dict(), strict=True)
        ce_s, extra_s, ce_r, extra_r = (total / applied).tolist()
        common = dict(client=k, rows=self.n[k], applied=applied, skipped=skipped)
        method.logs[k] = dict(common, loss=ce_r + extra_r, positive_ce=ce_r, negative_ce=extra_r)
        return dict(common, loss=ce_s + extra_s, real_ce=ce_s, auxiliary=extra_s, seconds=time.time() - started)


if __name__ == '__main__':
    driver.ARMS, driver.Method, driver.Run = ARMS, Method, Run
    driver.main()
