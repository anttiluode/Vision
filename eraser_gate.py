"""Eraser gate - is Martinotti-like delayed tuft inhibition an undo for top-down reads?

Vision (Claude, 7 Oct 2026; from VMNClaude Gates 5-6): a top-down query into an oscillating
apical tuft is a weak WRITE (the read signal a*sin(alpha) is also the lasting phase shift). A
second pulse of opposite sign undoes it to second order (VMNClaude Gate 6). Martinotti cells are
recruited by the pyramidal cell's own firing and inhibit the tuft after a delay, so they could be
that second pulse: query -> listen -> erase. Default: read-only. Disinhibition: write.

Timing matters. In the frame rotating with the rhythm, a pulse arriving d later than the query
lands at phase phi - omega*d. The undo needs -a e^{i phi}: an INHIBITORY copy of the input arriving a whole
number of cycles later (an excitatory copy half a cycle later is the same thing). At a
half-integer number of cycles, inhibition lands at the opposite phase and DOUBLES the damage.

Model (self-contained; same numbers as VMNClaude Gates 2/4/6):
  40 tuft oscillators (Stuart-Landau, mu = 0.5, beta = 0, noise sigma = 0.05), three spatial
  scales; their phases store a path-integrated position (200 steps). Each unit also turns at a
  common rhythm omega = 2 pi / T, T = 2 time units (amplitude relaxation time = 1).
  Query: goal q = x + D, |D| <= 0.35, kicked in as a * e^{i phi_k} with phi_k the goal's phase.
  Listen: 4 time units = 2 cycles; one-bank features (amplitude deviation and phase change since
  before the query, at t = 0.25, 0.5, 1, 2, 4). Reader: ridge or kNN, best on validation.
  Eraser fires at delay d after the query, inhibiting each tuft:
    input_locked   kick -g * a * e^{i(phi_k - omega d)}       (each tuft gets a copy of its own
                                                               input's strength; e.g. a reliable
                                                               spike drives its Martinotti cell)
    rate_driven    same, scaled by s_k = the tuft's own amplitude response during the first
                   cycle, normalised so its population mean is 1 (Martinotti recruited by firing
                   RATE; a rate tracks a*cos(alpha), not the phase kick a*sin(alpha))
  g = 1. Damage: unqueried twin sharing noise in each copy's own phase frame; RMS per-unit phase
  difference after a further 4 time units. a = 0.2.

Kill conditions, fixed before running:
  V1 (undo works)       input_locked at its best delay d >= 4 (after listening) leaves <= 1/3 of
                        the no-eraser damage, with answer error <= 1.2x the no-eraser error.
  V2 (timing rule)      input_locked at every half-integer delay >= 2 cycles (2.5, 3.5, 4.5) leaves
                        >= 1.5x the no-eraser damage: the delay must be a whole number of cycles.
  V3 (biological feed)  rate_driven at its best delay d >= 4 leaves <= 1/2 of the no-eraser damage.
"""
import json
import os
import time

import numpy as np

N, MU, SIGMA, DT, SUB, STEPS = 40, 0.5, 0.05, 0.25, 4, 200
SCALES = (2 * np.pi * 1.0, 2 * np.pi * 2.0, 2 * np.pi * 4.0)
T_CYC = 2.0
OMEGA = 2 * np.pi / T_CYC
LISTEN = 16                       # substeps (4 time units = 2 cycles)
SAMPLE = (1, 2, 4, 8, 16)
RELAX = 16
AMP = 0.2
DELAYS_CYC = (0.25, 0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0)
E = {"tr": 1000, "va": 300, "te": 500}
OFF = {"tr": 100, "va": 200, "te": 300}
SEEDS = (10, 11, 12)


def bank(seed):
    rng = np.random.default_rng(1000 + seed)
    ang = rng.uniform(0, 2 * np.pi, N)
    d = np.stack([np.cos(ang), np.sin(ang)], 1)
    b = np.array([SCALES[k % 3] for k in range(N)])
    return d, b


def trajectories(rng, n):
    x = np.zeros((STEPS + 1, n, 2)); v = np.zeros((STEPS, n, 2))
    x[0] = rng.uniform(0.1, 0.9, (n, 2))
    vel = rng.normal(size=(n, 2)) * 0.02
    for t in range(STEPS):
        vel = 0.9 * vel + 0.1 * rng.normal(size=(n, 2)) * 0.03
        nx = x[t] + vel
        for k in range(2):
            lo, hi = nx[:, k] < 0, nx[:, k] > 1
            nx[lo, k] = -nx[lo, k]; nx[hi, k] = 2 - nx[hi, k]
            vel[lo | hi, k] *= -1
        v[t] = nx - x[t]; x[t + 1] = nx
    return x, v


def rhs(z, lin):
    return lin * z - (z.real ** 2 + z.imag ** 2) * z


def rk4(z, lin):
    k1 = rhs(z, lin); k2 = rhs(z + 0.5 * DT * k1, lin)
    k3 = rhs(z + 0.5 * DT * k2, lin); k4 = rhs(z + DT * k3, lin)
    return z + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def noise(rng, shape):
    return SIGMA * np.sqrt(DT) * (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)


def shared_step(zp, zu, rng):
    """Advance both copies with one noise draw expressed in each copy's own phase frame."""
    e = noise(rng, zp.shape)
    up = zp / np.maximum(np.abs(zp), 1e-12); uu = zu / np.maximum(np.abs(zu), 1e-12)
    return rk4(zp, MU) + e * up, rk4(zu, MU) + e * uu


def integrate(seed, x, v, rng):
    d, b = bank(seed)
    z = np.sqrt(MU) * np.exp(1j * b[:, None] * (d @ x[0].T))
    for t in range(STEPS):
        lin = MU + 1j * (b[:, None] * (d @ v[t].T) / (SUB * DT))
        for _ in range(SUB):
            z = rk4(z, lin) + noise(rng, z.shape)
    return z


def query(seed, Z, q, eraser, delay_sub, rng):
    """Rotating frame of the common rhythm: a pulse delay_sub substeps after the query lands at
    phase - omega * delay. Returns one-bank features and the retained phase disturbance."""
    d, b = bank(seed)
    phi = b[:, None] * (d @ q.T)
    zp, zu = Z.copy(), Z.copy()
    th0 = np.angle(zp)
    zp = zp + AMP * np.sqrt(MU) * np.exp(1j * phi)
    feats, peak = [], np.zeros(zp.shape)
    total = max(LISTEN, delay_sub if delay_sub else 0)
    for s in range(1, total + 1):
        zp, zu = shared_step(zp, zu, rng)
        if s <= int(T_CYC / DT):
            peak = np.maximum(peak, np.abs(zp) - np.sqrt(MU))
        if s in SAMPLE:
            feats.append(np.abs(zp) - np.sqrt(MU)); feats.append(np.angle(zp * np.exp(-1j * th0)))
        if eraser != "none" and s == delay_sub:
            if eraser == "input_locked":
                s_k = 1.0
            else:
                s_k = peak / max(peak.mean(), 1e-12)
            zp = zp - s_k * AMP * np.sqrt(MU) * np.exp(1j * (phi - OMEGA * delay_sub * DT))
    for _ in range(RELAX):
        zp, zu = shared_step(zp, zu, rng)
    dphi = np.angle(zp * np.conj(zu))
    return np.concatenate(feats, 0).T, float(np.sqrt((dphi ** 2).mean()))


# ---------- readers (as VMNClaude Gate 4) ----------------------------------------------------
def standardize(Ftr, *others):
    m, s = Ftr.mean(0), Ftr.std(0) + 1e-9
    return [(F - m) / s for F in (Ftr,) + others]


def ridge(Ftr, Ytr, Fva, Yva, Fte):
    Ftr, Fva, Fte = standardize(Ftr, Fva, Fte)
    add = lambda F: np.concatenate([F, np.ones((len(F), 1))], 1)
    best = None
    for lam in (1e-2, 1e-1, 1.0, 10.0, 100.0):
        W = np.linalg.solve(add(Ftr).T @ add(Ftr) + lam * np.eye(Ftr.shape[1] + 1), add(Ftr).T @ Ytr)
        e = np.linalg.norm(add(Fva) @ W - Yva, axis=1).mean()
        if best is None or e < best[0]:
            best = (e, W)
    return add(Fte) @ best[1]


def knn(Ftr, Ytr, Fva, Yva, Fte):
    Ftr, Fva, Fte = standardize(Ftr, Fva, Fte)

    def pred(F, k):
        d2 = (F ** 2).sum(1)[:, None] - 2 * F @ Ftr.T + (Ftr ** 2).sum(1)[None]
        return Ytr[np.argpartition(d2, k, axis=1)[:, :k]].mean(1)
    k = min((5, 15, 40), key=lambda k: np.linalg.norm(pred(Fva, k) - Yva, axis=1).mean())
    return pred(Fte, k)


def answer_error(F, Y):
    return min(float(np.linalg.norm(f(F["tr"], Y["tr"], F["va"], Y["va"], F["te"]) - Y["te"], axis=1).mean())
               for f in (ridge, knn))


def run_seed(sd):
    Z, data = {}, {}
    for s in E:
        x, v = trajectories(np.random.default_rng(OFF[s] + sd), E[s])
        rng = np.random.default_rng(900 + OFF[s] + sd)
        ang = rng.uniform(0, 2 * np.pi, E[s]); rad = 0.35 * np.sqrt(rng.uniform(0, 1, E[s]))
        D = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
        data[s] = {"D": D, "q": x[-1] + D}
        Z[s] = integrate(sd, x, v, np.random.default_rng(31 + 7 * sd + {"tr": 1, "va": 2, "te": 3}[s]))
    Y = {s: data[s]["D"] for s in E}
    conds = [("none", 0)] + [(er, int(round(c * T_CYC / DT))) for er in ("input_locked", "rate_driven") for c in DELAYS_CYC]
    res = {"predict_zero_error": float(np.linalg.norm(Y["te"], axis=1).mean())}
    for er, ds in conds:
        F, dmg = {}, None
        for s in E:
            F[s], ph = query(sd, Z[s], data[s]["q"], er, ds, np.random.default_rng(5000 + 13 * sd + {"tr": 1, "va": 2, "te": 3}[s]))
            if s == "te":
                dmg = ph
        res[f"{er}|{ds * DT / T_CYC:g}"] = {"answer_error": answer_error(F, Y), "phase_damage": dmg}
    return res


def main():
    t0 = time.time()
    per = {sd: run_seed(sd) for sd in SEEDS}
    print(f"{time.time() - t0:.0f}s")
    keys = [k for k in per[SEEDS[0]] if "|" in k]
    S = {"predict_zero_error": float(np.mean([per[s]["predict_zero_error"] for s in SEEDS]))}
    for k in keys:
        S[k] = {m: [float(np.mean([per[s][k][m] for s in SEEDS])), float(np.std([per[s][k][m] for s in SEEDS]))]
                for m in ("answer_error", "phase_damage")}
    none_e, none_d = S["none|0"]["answer_error"][0], S["none|0"]["phase_damage"][0]
    print(f"none: answer {none_e:.4f}  damage {none_d:.4f}")
    for er in ("input_locked", "rate_driven"):
        for c in DELAYS_CYC:
            v = S[f"{er}|{c:g}"]
            print(f"{er:13s} delay {c:4g} cycles: answer {v['answer_error'][0]:.4f}  damage {v['phase_damage'][0]:.4f}")

    def best_after(er):
        cands = [(S[f"{er}|{c:g}"]["phase_damage"][0], c) for c in DELAYS_CYC if c * T_CYC >= LISTEN * DT]
        return min(cands)
    bl_d, bl_c = best_after("input_locked")
    br_d, br_c = best_after("rate_driven")
    half = [S[f"input_locked|{c:g}"]["phase_damage"][0] for c in DELAYS_CYC if c * T_CYC >= LISTEN * DT and (c * 2) % 2 == 1]
    V = {"V1_input_locked_undo": bool(bl_d <= none_d / 3 and S[f"input_locked|{bl_c:g}"]["answer_error"][0] <= 1.2 * none_e),
         "V1_values": {"best_delay_cycles": bl_c, "damage": bl_d, "none_damage": none_d,
                       "answer": S[f"input_locked|{bl_c:g}"]["answer_error"][0], "none_answer": none_e},
         "V2_half_cycle_doubles": bool(min(half) >= 1.5 * none_d),
         "V2_values": {"half_cycle_damages": half, "none_damage": none_d},
         "V3_rate_driven_halves": bool(br_d <= none_d / 2),
         "V3_values": {"best_delay_cycles": br_c, "damage": br_d, "none_damage": none_d}}
    out = {"setup": {"N": N, "T_cycle": T_CYC, "listen_time": LISTEN * DT, "amp": AMP, "delays_cycles": DELAYS_CYC,
                     "episodes": E, "seeds": SEEDS},
           "summary_mean_sd": S, "verdicts": V, "per_seed": {str(k): v for k, v in per.items()},
           "seconds": round(time.time() - t0, 1)}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/eraser_receipt.json", "w"), indent=2)
    print(json.dumps(V, indent=2))


if __name__ == "__main__":
    main()
