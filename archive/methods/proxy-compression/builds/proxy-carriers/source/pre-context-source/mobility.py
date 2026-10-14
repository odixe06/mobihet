"""Deterministic vehicle-RSU contact and compute simulator (MobiHet_IDS.md §5.2-§5.3, §7).

The VeReMi FL partitions carry features and labels but no receiver trajectory, so contact
time cannot be measured; it is SIMULATED, and every number it produces is a pure function
of (cfg, round, client, client size). Nothing here reads a clock.

Build 1 compared the real wall-clock seconds the GPU took to train a client (30-180 s on
a T4, and different for two workers sharing a GPU) with a simulated 30-60 s RSU contact:
from round 4 on almost every vehicle was "infeasible", no model trained again and the
metrics froze (docs/build1.md). Build 2 simulates the vehicle's training time as
well, so the schedule no longer depends on which GPU ran the round.

Per vehicle i (fixed for the run):   device factor  d_i ~ logU[dev_min, dev_max]
Per (vehicle, round):
    position  p ~ uniform over the RSU disk of radius R, heading ~ U(-pi, pi),
    speed v ~ U[v_min, v_max]              -> tau_true = time until p + v t leaves the disk
    noisy p', v' (sigma_pos, sigma_vel)    -> tau_hat  = the same exit time, estimated online
    rate r_hat ~ U[r_min, r_max], realised rate r = r_hat * U[0.75, 1.15]
    T_train_hat = t_ref * (n_i / median_n) * d_i          what the server predicts (§5.3)
    T_train     = T_train_hat * U[1 - jitter, 1 + jitter]  what the round actually costs
t_ref is the simulated epoch time of the scenario's MEDIAN client, so the three scenarios
(20/50/100 clients, same 43 M rows) stress the schedule equally; a client with more data
takes proportionally longer. The values of these knobs are calibrated by
scripts/calibrate_mobility.py and fixed in the notebook CFG.

Build 3/4 (docs/build3.md §2) add two knobs, both neutral at their defaults:
  compute_factor  T_train of a technique relative to build 2's step (measured ms/step ratio)
  time_downlink   the vehicle must RECEIVE the global knowledge before it trains:
                  T_down = bytes_down / rate is charged in the filter, the budget and the
                  delivery clock. Build 2 charged only T_train + T_upload (MobiHet_IDS.md §1.1).
Build 5 (docs/build5.md §8) adds N3's second-order part to every item and broadcast class
(cov_bytes). N1's record pass over the local hold-out is not charged separately: client_rows
stays the vehicle's whole span, i.e. T_train is priced as one epoch over all of it, while the
vehicle trains on 95 % and only forwards 5 % (~96.7 % of that time).
"""
import math
import numpy as np

SIM_KEYS = ("rsu_radius_m", "speed_min_mps", "speed_max_mps", "rate_min_Bps", "rate_max_Bps",
            "pos_noise_m", "vel_noise_mps", "t_ref_s", "dev_min", "dev_max",
            "train_jitter", "guard_s", "sim_seed")

PROTO_HEADER_BYTES = 32       # per upload: round, client id, class bitmap, lengths
PROTO_META_BYTES = 24         # per class item: class id, sample count, fp32 scale, checksum
HARD_META_BYTES = 4           # P4: the count of misclassified rows behind a hard prototype


def cov_bytes(latent_dim, cov_rank):
    """Build 5, N3 (docs/build5.md §8.2 item 5): the second-order part of a class record, the
    same size up and down -- cov_rank principal directions (scaled by sqrt eigenvalue) as fp16,
    plus 4 fp16 scalars (residual sigma, mean and std of |h|, and downlink |mu_g|: the vehicle
    rebuilds mu_g = |mu_g| p^g_c from the bank). 0 when cov_rank < 0 (N3 off)."""
    return cov_rank * latent_dim * 2 + 8 if cov_rank >= 0 else 0


def item_bytes(latent_dim, soft_classes=0, hard=False, cov_rank=-1):
    """One class item on the wire: the prototype's latent_dim fp32 values plus its metadata,
    plus (build 4, T5 / P2) a soft label of `soft_classes` fp16 values, plus (P4) a hard
    prototype of latent_dim fp32 values and its count, plus (build 5, N3) cov_bytes."""
    return (latent_dim * 4 + PROTO_META_BYTES + 2 * soft_classes
            + ((latent_dim * 4 + HARD_META_BYTES) if hard else 0)
            + cov_bytes(latent_dim, cov_rank))


def downlink_bytes(n_protos, n_soft, num_classes, latent_dim, n_hard=0, n_cov=0, cov_rank=-1):
    """What the server broadcasts at the start of a round (build 3/4): header, every
    prototype in the bank (k per class under T4), every class soft label (T5), every hard
    prototype (P4), the class prior pi_g as C fp32 counts, and (build 5, N3) the
    second-order part of every class that has one. Build 2 logged header + n_protos items only."""
    return (PROTO_HEADER_BYTES + (n_protos + n_hard) * item_bytes(latent_dim)
            + n_soft * 2 * num_classes + 4 * num_classes + n_cov * cov_bytes(latent_dim, cov_rank))


def t_down(st, cfg, down_bytes, actual):
    """Seconds to receive the broadcast; 0 unless cfg['time_downlink']."""
    if not cfg.get("time_downlink") or not down_bytes:
        return 0.0
    return down_bytes / st["rate_actual_Bps" if actual else "rate_hat_Bps"]


def _exit_time(p, v):
    """Largest t >= 0 with |p + v t| = R, for p inside the disk (R folded into p's units)."""
    a = float(v @ v)
    b = 2.0 * float(p @ v)
    c = float(p @ p) - 1.0
    if a <= 0.0:
        return math.inf
    disc = max(b * b - 4 * a * c, 0.0)
    return max((-b + math.sqrt(disc)) / (2 * a), 0.0)


def device_factors(cfg, n_clients):
    g = np.random.default_rng([int(cfg["sim_seed"]), 7919])
    lo, hi = math.log(cfg["dev_min"]), math.log(cfg["dev_max"])
    return np.exp(g.uniform(lo, hi, n_clients))


def round_states(cfg, rnd, client_rows):
    """Contact/compute state of every vehicle in round `rnd`. client_rows[i] = n_i."""
    n = np.asarray(client_rows, dtype=np.float64)
    med = float(np.median(n))
    dev = device_factors(cfg, len(n))
    R = float(cfg["rsu_radius_m"])
    out = []
    for i in range(len(n)):
        g = np.random.default_rng([int(cfg["sim_seed"]), int(rnd), i])
        rad = R * math.sqrt(g.uniform())                          # uniform over the disk
        ang, hed = g.uniform(-math.pi, math.pi, 2)
        spd = g.uniform(cfg["speed_min_mps"], cfg["speed_max_mps"])
        p = np.array([rad * math.cos(ang), rad * math.sin(ang)])
        v = spd * np.array([math.cos(hed), math.sin(hed)])
        tau_true = _exit_time(p / R, v / R)
        p_hat = p + g.normal(0.0, cfg["pos_noise_m"], 2)
        v_hat = v + g.normal(0.0, cfg["vel_noise_mps"], 2)
        # An estimate that lands outside the disk means "leaving now".
        tau_hat = _exit_time(p_hat / R, v_hat / R) if p_hat @ p_hat < R * R else 0.0
        r_hat = g.uniform(cfg["rate_min_Bps"], cfg["rate_max_Bps"])
        r_act = r_hat * g.uniform(0.75, 1.15)
        t_hat = cfg["t_ref_s"] * (n[i] / med) * dev[i] * float(cfg.get("compute_factor", 1.0))
        if cfg.get("record_mode") == "final":
            # Upper bound on the per-class reservoir; conservative admission, no free record.
            nr = min(n[i], cfg["record_samples"] * cfg["num_classes"])
            if cfg["exchange"] != "latent":
                nr += cfg["num_classes"] * cfg["anchor_send"]
            t_hat += (cfg["t_ref_s"] * dev[i] * nr / med
                      * cfg["record_forward_ratio"] / cfg["local_epochs"])
        t_act = t_hat * g.uniform(1.0 - cfg["train_jitter"], 1.0 + cfg["train_jitter"])
        out.append({"cid": i, "tau_hat_s": float(tau_hat), "tau_true_s": float(tau_true),
                    "rate_hat_Bps": float(r_hat), "rate_actual_Bps": float(r_act),
                    "train_hat_s": float(t_hat), "train_s": float(t_act),
                    "speed_mps": float(spd), "dist_m": float(rad)})
    return out


def feasible(st, cfg, latent_dim, down_bytes=0, soft_classes=0):
    """§5.3 filter, BEFORE training, on estimates only: the vehicle is selected iff its
    predicted download (build 3/4), training and the smallest useful upload (header + one
    class, with its build 5 second-order part) fit in the estimated residual contact minus
    the guard."""
    need = PROTO_HEADER_BYTES + (cfg.get("wire_item_bytes") or item_bytes(
        latent_dim, soft_classes, cov_rank=int(cfg.get("cov_rank", -1))))
    return (t_down(st, cfg, down_bytes, actual=False) + st["train_hat_s"]
            + need / st["rate_hat_Bps"] + cfg["guard_s"] <= st["tau_hat_s"])


def budget_bytes(st, cfg, down_bytes=0):
    """B_i,t = r_hat * max(0, tau_hat - T_down - T_train - T_guard), with the ACTUAL download
    and training times: the vehicle knows both when it schedules the upload."""
    return st["rate_hat_Bps"] * max(0.0, st["tau_hat_s"] - t_down(st, cfg, down_bytes, True)
                                    - st["train_s"] - cfg["guard_s"])


def schedule(items, budget):
    """Greedy Top-K by utility under the byte budget (§5.5). `items` are already sorted by
    decreasing utility; an item that does not fit is skipped, a smaller later one may."""
    used, chosen = PROTO_HEADER_BYTES, []
    if budget < PROTO_HEADER_BYTES:
        return chosen, 0
    for it in items:
        if used + it["bytes"] <= budget:
            chosen.append(it); used += it["bytes"]
    return chosen, used


def deliver(chosen, st, t_start=0.0):
    """What actually arrives before the TRUE deadline at the REALISED rate. The header goes
    first; items stream in utility order and the first one that would finish after the
    vehicle leaves coverage is lost together with everything behind it. t_start is the
    download time already spent in coverage (build 3/4; 0 in build 2)."""
    t = t_start + st["train_s"] + PROTO_HEADER_BYTES / st["rate_actual_Bps"]
    got, used = [], PROTO_HEADER_BYTES
    if t > st["tau_true_s"]:
        return got, 0
    for it in chosen:
        t += it["bytes"] / st["rate_actual_Bps"]
        if t > st["tau_true_s"]:
            break
        got.append(it); used += it["bytes"]
    return got, (used if got else 0)
