"""P2 channel (R1): one vehicle-RSU contact per vehicle per round, from the archive contact simulator (mh/mobility.py).

Budget tiers (docs/research/de-xuat-huong-moi.md §3, scripts/budget_tiers.py; training happens outside the contact):
    bytes_true = rate_actual * rate_scale * max(0, tau_true * contact_scale - guard)   (capped at B0)
    bytes_hat  = the same from rate_hat / tau_hat: what the vehicle can plan with (it never sees the true deadline).
Byte accounting (archive rules, docs/legacy/kien-thuc-ke-thua.md §6): every fragment pays its payload + header + ACK;
the fragment the deadline cuts is charged up to the deadline and NOT committed; nothing is free. An object is used
only when every fragment of ONE version has arrived (coherent snapshot).
"""
import math
import numpy as np
from mh import mobility as M

SIM = {"rsu_radius_m": 800.0, "speed_min_mps": 5.0, "speed_max_mps": 20.0, "rate_min_Bps": 20000.0,
       "rate_max_Bps": 80000.0, "pos_noise_m": 8.0, "vel_noise_mps": 1.5, "t_ref_s": 10.0, "dev_min": 0.5,
       "dev_max": 2.0, "train_jitter": 0.1, "guard_s": 2.0, "sim_seed": 42}
TIERS = {'B0': (0.01, 0.2, 32768), 'B1': (0.1, 0.2, None), 'B2': (0.1, 1.0, None), 'B3': (1.0, 1.0, None)}
FRAG, HDR, ACK = 1024, 16, 8          # payload bytes per fragment; frame header (kind, object, version, offset, len, crc); ACK


def budgets(tier, rounds, K):
    """(true, hat) byte budgets, arrays [rounds + 1, K]; row 0 unused. Pure function of (tier, round, client)."""
    cs, rs, cap = TIERS[tier]
    guard = 0.01 if cs < 1 else 2.0
    true, hat = np.zeros((rounds + 1, K)), np.zeros((rounds + 1, K))
    for r in range(1, rounds + 1):
        for st in M.round_states(SIM, r, [1] * K):
            k = st['cid']
            true[r, k] = st['rate_actual_Bps'] * rs * max(0.0, st['tau_true_s'] * cs - guard)
            hat[r, k] = st['rate_hat_Bps'] * rs * max(0.0, st['tau_hat_s'] * cs - guard)
    if cap:
        true, hat = np.minimum(true, cap), np.minimum(hat, cap)
    return true, hat


def wire(n):
    """Wire bytes of an n-byte object sent whole."""
    return n + math.ceil(n / FRAG) * (HDR + ACK)


class Transfer:
    """One object version in flight; `sent` is the committed payload cursor."""
    def __init__(self, key, size, data=None):
        self.key, self.size, self.data, self.sent = key, int(size), data, 0

    @property
    def left(self):
        return self.size - self.sent


def contact(transfers, budget):
    """Send `transfers` in order within `budget` true bytes. Returns (completed transfers, bytes charged).
    Cursors advance in place; the caller decides whether an unfinished transfer is resumed next contact (cursor kept)
    or restarted (cursor reset)."""
    used, done = 0.0, []
    for t in transfers:
        while t.left > 0:
            c = min(FRAG, t.left) + HDR + ACK
            if used + c > budget:
                return done, budget            # the cut fragment is paid, not committed
            used += c; t.sent += min(FRAG, t.left)
        done.append(t)
    return done, used
