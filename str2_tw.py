
import os
import time

import numpy as np

from core_tw import (
    validate_tw,arrays_to_routes, routes_to_arrays, total_cost,
                     seq_feasible, seq_cost, nn_construct)

# bras du bandit : relatifs a la longueur moyenne des tournees
ARM_ABS = [5.0, 10.0, 18.0, 30.0]
ARM_MULT = [0.3, 0.6, 1.0, 1.5]
ARM_BLINK = [0.005, 0.01, 0.02, 0.03]
ARM_MIN, ARM_MAX = 4.0, 40.0


def bras_instance(n_clients, k_routes):
    Lbar = max(2.0, float(n_clients) / max(k_routes, 1))
    if int(os.environ.get("STR2_FIXED_SIZES", 0)):
        # ablation : tailles fixes (planchers), sans le terme mu_i * L
        return [(min(ARM_MAX, max(ARM_MIN, a)), b)
                for a, m, b in zip(ARM_ABS, ARM_MULT, ARM_BLINK)]
    return [(min(ARM_MAX, max(ARM_MIN, max(a, m * Lbar))), b)
            for a, m, b in zip(ARM_ABS, ARM_MULT, ARM_BLINK)]


def chaine_max(n_clients, k_routes):
    Lbar = float(n_clients) / max(k_routes, 1)
    return float(min(30.0, max(10.0, round(0.5 * Lbar))))


#  MINIMISATION DE FLOTTE (contrainte TW simulee integralement)
def try_drop_route_tw(routes, D, dem, Q, e, l, s, rng, n_tries=2):
    """Tente de vider la tournee ayant le MOINS de clients en les
    reinserant (faisabilite TW complete simulee position par position).
    Retourne (routes, True) si une tournee a disparu. Les clients les
    plus contraints (fenetre etroite, forte demande) sont places en
    premier. Plusieurs ordres aleatoires sont essayes."""
    if len(routes) <= 1:
        return routes, False
    seqs = [[c for c in r if c != 0] for r in routes]
    tgt = min(range(len(seqs)), key=lambda i: len(seqs[i]))
    victims = seqs[tgt]
    others = [seqs[i] for i in range(len(seqs)) if i != tgt]

    base_order = sorted(victims,
                        key=lambda u: (l[u] - e[u], -dem[u]))
    rand_first = int(os.environ.get("STR2_FLEET_RANDOM", 0))
    for t in range(n_tries):
        order = list(base_order)
        if t > 0 or rand_first:
            rng.shuffle(order)
        work = [list(sq) for sq in others]
        loads = [sum(dem[c] for c in sq) for sq in work]
        ok = True
        for u in order:
            bd, bi, bp = 1e18, -1, -1
            for i, sq in enumerate(work):
                if loads[i] + dem[u] > Q + 1e-9:
                    continue
                for p in range(len(sq) + 1):
                    cand = sq[:p] + [u] + sq[p:]
                    if not seq_feasible(cand, D, dem, Q, e, l, s):
                        continue
                    a = sq[p - 1] if p > 0 else 0
                    b = sq[p] if p < len(sq) else 0
                    d = D[a, u] + D[u, b] - D[a, b]
                    if d < bd:
                        bd, bi, bp = d, i, p
            if bi < 0:
                ok = False
                break
            work[bi].insert(bp, u)
            loads[bi] += dem[u]
        if ok:
            return [[0] + sq + [0] for sq in work if sq], True
    return routes, False


#  MOTEUR COMPILE — sisr_jit_tw
def sisr_jit_tw(routes0, D, dem, Q, e, l, s, knn, budget, seed,
                n_veh=None, verbose=False, chunk=None, on_best=None,
                diag_out=None, route_pool=None):
    """RR granulaire compile + fenetres dures.
    Retourne (routes, distance, n_vehicules, iterations).
    Si `route_pool` est une liste, les tournees du meilleur (a chaque
    bord de cycle, apres chaque reduction de flotte, et en fin de run)
    y sont ajoutees sous forme de tuples de clients, dedoublonnees et
    verifiees faisables : c'est le pool de la recombinaison
    (recombine_tw.py).
    Si `diag_out` est un dict, il est rempli en place avec les
    diagnostics du run (trajectoire, cycles, echantillon de deltas,
    compteurs noyau, flotte, bandit) — voir l'en-tete du module."""
    from kernel_tw import (rr_chunk_tw, recompute_all_times, MAXREM,
                           HAVE_NUMBA)
    n = len(dem)
    K = knn.shape[1]
    maxr = min(n + 2, max(len(routes0) + 40, 8))
    knn = np.ascontiguousarray(knn, dtype=np.int32)
    D = np.ascontiguousarray(D, dtype=np.float64)
    dem = np.ascontiguousarray(dem, dtype=np.float64)
    e = np.ascontiguousarray(e, dtype=np.float64)
    l = np.ascontiguousarray(l, dtype=np.float64)
    s = np.ascontiguousarray(s, dtype=np.float64)
    rng = np.random.default_rng(seed)
    vb = verbose or bool(int(os.environ.get("STR2_VERBOSE", 0)))

    nxt, prv, rte, first, load, nr = routes_to_arrays(routes0, n, dem, maxr)
    tstart = np.zeros(n, dtype=np.float64)
    lat = np.zeros(n, dtype=np.float64)
    recompute_all_times(D, e, l, s, nxt, prv, first, tstart, lat, nr)

    cur = total_cost(nxt, first, nr, D)
    best_c = cur
    best_nr = nr
    b_nxt, b_prv, b_rte = nxt.copy(), prv.copy(), rte.copy()
    b_first, b_load = first.copy(), load.copy()
    # OBJECTIF (STR2_OBJ) :
    #   "lex"  (defaut) : (vehicules, distance) — classement Homberger
    #   "dist"          : distance seule — convention DIMACS 2022
    # Le noyau n'optimise que la distance a flotte courante ; le
    # meilleur global est collecte selon la cle choisie.
    obj = os.environ.get("STR2_OBJ", "lex")
    lex_routes = arrays_to_routes(b_nxt, b_first, nr)
    lex_key = ((len(lex_routes), best_c) if obj == "lex"
               else (0, best_c))

    rem = np.zeros(MAXREM, np.int32)
    uc = np.zeros(MAXREM, np.int32)
    ua = np.zeros(MAXREM, np.int32)
    ur = np.full(MAXREM, -1, np.int32)
    pb = np.zeros(MAXREM, np.int32)
    pr = np.zeros(MAXREM, np.int32)
    tr = np.zeros(2 * MAXREM, np.int32)
    stats = np.zeros(8, np.int64)          # compteurs noyau (module doc)
    noscan = int(os.environ.get("STR2_NOSCAN", 0))
    _slack = int(os.environ.get("STR2_FLEET_SLACK", 0))
    k_floor = (int(np.ceil(float(dem[1:].sum()) / Q)) + _slack
               if _slack > 0 else 0)

    arms_i = bras_instance(n - 1, nr)
    max_chain = chaine_max(n - 1, nr)
    np.random.seed(seed % (2 ** 31))
    if chunk is None:
        chunk = int(os.environ.get("STR2_CHUNK", 20000))

    # calibrage T0 par sonde (mesure)
    n_probe = int(os.environ.get("STR2_TPROBE", 30000))
    dbuf = np.zeros(8192, np.float64)
    mode_T = os.environ.get("STR2_T0", "auto")
    p_acc = float(os.environ.get("STR2_TACC", 0.10))
    t_ratio = float(os.environ.get("STR2_TRATIO", 0.01))
    log_p = max(-np.log(max(p_acc, 1e-6)), 1e-9)

    def _sonder(cur, best_c, nr, settle):
        av0, bl0 = arms_i[1]
        if settle:
            cur, best_c, nr, _, _ = rr_chunk_tw(
                D, dem, Q, e, l, s, knn, nxt, prv, rte, first, load, nr,
                tstart, lat, b_nxt, b_prv, b_rte, b_first, b_load,
                cur, best_c, settle, float(av0), float(bl0),
                1e-9, 1e-11, 0.0, 0.0, rem, uc, ua, ur, pb, pr, tr,
                float(max_chain), dbuf, 0, stats, noscan)
        cur, best_c, nr, _, nd = rr_chunk_tw(
            D, dem, Q, e, l, s, knn, nxt, prv, rte, first, load, nr,
            tstart, lat, b_nxt, b_prv, b_rte, b_first, b_load,
            cur, best_c, n_probe, float(av0), float(bl0),
            1e-9, 1e-11, 0.0, 0.0, rem, uc, ua, ur, pb, pr, tr,
            float(max_chain), dbuf, 1, stats, noscan)
        m = float(np.median(dbuf[:nd])) if nd >= 200 else 0.0
        return cur, best_c, nr, m, nd

    def _temperature(med, cur):
        used_default = med <= 0.0
        t = med / log_p if med > 0.0 else cur * 1e-3
        tc = min(cur * 3e-2, max(cur * 1e-5, t))
        return tc, max(tc * t_ratio, 1e-11), tc != t, used_default

    diag = diag_out if isinstance(diag_out, dict) else None
    if diag is not None:
        diag.clear()
        diag.update(meta=dict(
            seed=int(seed), n=int(n - 1), budget=float(budget),
            obj=obj, chunk=int(chunk), n_probe=int(n_probe),
            p_acc=p_acc, t_ratio=t_ratio,
            sizes=[a for a, _ in arms_i],
            blinks=[b for _, b in arms_i],
            max_chain=float(max_chain), routes0=int(nr),
            cost0=float(cur),
            win_med=float(np.median(l[1:] - e[1:])),
            horizon=float(l[0] - e[0])),
            traj=[], cycles=[], delta_sample=[], fleet=[],
            bandit={}, totals={})
    t0_abs = os.environ.get("STR2_T0_ABS")
    t0 = time.time()
    med_d, ips, nd0 = 0.0, 0.0, 0
    clamped = defaulted = False
    if t0_abs is not None:
        # ablation : seuil constant absolu, aucune mesure
        T0 = max(float(t0_abs), 1e-9)
        Tend = max(T0 * t_ratio, 1e-11)
        recal = 0
    elif mode_T == "auto" and n_probe > 0:
        _tc = time.time()
        cur, best_c, nr, med_d, nd0 = _sonder(cur, best_c, nr, n_probe)
        T0, Tend, clamped, defaulted = _temperature(med_d, cur)
        ips = 2.0 * n_probe / max(time.time() - _tc, 1e-6)
        recal = int(os.environ.get("STR2_TRECAL", 1))
        if diag is not None:
            diag["delta_sample"] = [float(x) for x in dbuf[:nd0]]
    else:
        T0 = max(cur * float(mode_T if mode_T != "auto" else 1e-3), 1e-9)
        Tend = max(T0 * t_ratio, 1e-11)
        recal = int(os.environ.get("STR2_TRECAL", 1)) and mode_T == "auto"
    if diag is not None:
        diag["cycles"].append(dict(
            g=1, med=float(med_d), T0=float(T0), Tend=float(Tend),
            nd=int(nd0), clamped=bool(clamped), default=bool(defaulted),
            cost=float(cur)))

    # recuit cyclique
    cyc_env = os.environ.get("STR2_CYCLES", "auto")
    if cyc_env == "auto":
        cyc_target = float(os.environ.get("STR2_CYCLEN", 100.0))
        n_cycles = int(min(10, max(1, round(budget / max(cyc_target, 1.0)))))
    else:
        n_cycles = max(1, int(cyc_env))
    cyc_len = budget / n_cycles

    if vb:
        print(f"[CALIB-TW] seed {seed} — {nr} tournees, "
              f"{(n - 1) / max(nr, 1):.1f} clients/tournee, chaine max "
              f"{max_chain:.0f}, delta median {med_d:.3f} -> T0 {T0:.4f}, "
              f"Tend {Tend:.5f}, numba={HAVE_NUMBA}", flush=True)

    it = 2 * n_probe
    n_acc, n_seg, cyc_prev = 0, 0, 0
    it_c0, acc_c0 = it, 0                    # bornes du cycle courant
    dg_c0, dga_c0 = int(stats[4]), int(stats[5])
    # en mode "dist" (DIMACS), reduire la flotte n'est utile que si cela
    # baisse la distance ; la phase dediee est desactivee par defaut
    fleet_on = int(os.environ.get("STR2_FLEET",
                                  1 if obj == "lex" else 0))
    A = len(arms_i)
    arm = 1
    ucb_c = float(os.environ.get("STR2_UCB_C", 0.5))
    gamma = float(os.environ.get("STR2_UCB_GAMMA", 0.98))
    fixed_arm = int(os.environ.get("STR2_ARM", -1))
    if 0 <= fixed_arm < A:
        arm = fixed_arm
    cnt = np.zeros(A)
    tot_r = np.zeros(A)

    strict_best = int(os.environ.get("STR2_VALID_BEST", 0))

    def _lex_update():
        nonlocal lex_routes, lex_key
        rr = arrays_to_routes(b_nxt, b_first, nr)
        k = (len(rr), best_c) if obj == "lex" else (0, best_c)
        if k < lex_key:
            if strict_best:
                ok, _msg = validate_tw(rr, D, dem, Q, e, l, s, n)
                if not ok:
                    return
            lex_key, lex_routes = k, rr
            if on_best is not None:
                on_best(k, rr)

    _pool_seen = set()

    def _pool_add(routes_list):
        if route_pool is None:
            return
        for rt in routes_list:
            seq = tuple(int(c) for c in rt if c != 0)
            if not seq or seq in _pool_seen:
                continue
            _pool_seen.add(seq)
            if seq_feasible(list(seq), D, dem, Q, e, l, s):
                route_pool.append(seq)

    while True:
        el = time.time() - t0
        if el >= budget:
            break
        frac0 = min(1.0, (el % cyc_len) / cyc_len)
        span = min(1.0 - frac0, chunk / max(1.0, cyc_len * max(ips, 1.0)))
        avg_rm, blink = arms_i[arm]
        c_before = best_c
        t_seg = time.time()
        cur, best_c, nr, acc, _ = rr_chunk_tw(
            D, dem, Q, e, l, s, knn, nxt, prv, rte, first, load, nr,
            tstart, lat, b_nxt, b_prv, b_rte, b_first, b_load,
            cur, best_c, chunk, float(avg_rm), float(blink),
            T0, Tend, frac0, span, rem, uc, ua, ur, pb, pr, tr,
            float(max_chain), dbuf, 0, stats, noscan)
        it += chunk
        n_acc += acc
        n_seg += 1
        if diag is not None:
            diag["traj"].append([round(time.time() - t0, 3),
                                 round(float(best_c), 4), int(nr),
                                 int(arm), int(acc)])
        if n_seg % 100 == 0:                     # resynchronisation exacte
            cur = total_cost(nxt, first, nr, D)
        _lex_update()

        # bandit UCB1, oubli exponentiel
        dt = max(time.time() - t_seg, 1e-6)
        ips = chunk / dt if ips == 0.0 else 0.9 * ips + 0.1 * (chunk / dt)
        if gamma < 1.0:
            cnt *= gamma
            tot_r *= gamma
        cnt[arm] += 1.0
        tot_r[arm] += max(0.0, c_before - best_c) / dt
        tt = max(cnt.sum(), 1.0)
        mu_max = 0.0
        for a in range(A):
            if cnt[a] > 0 and tot_r[a] / cnt[a] > mu_max:
                mu_max = tot_r[a] / cnt[a]
        scale = ucb_c * max(mu_max, 1e-12)
        bestu = -1e18
        for a in range(A) if fixed_arm < 0 else ():
            if cnt[a] == 0:
                u = 1e17
            else:
                u = tot_r[a] / cnt[a] + scale * np.sqrt(
                    2 * np.log(tt) / cnt[a])
            if u > bestu:
                bestu, arm = u, a

        # minimisation de flotte (objectif primaire Homberger)
        if fleet_on and (n_veh is None or nr > 1) and nr > k_floor:
            routes_now = arrays_to_routes(b_nxt, b_first, nr)
            routes2, dropped = try_drop_route_tw(
                routes_now, D, dem, Q, e, l, s, rng)
            if dropped:
                nxt, prv, rte, first, load, nr = routes_to_arrays(
                    routes2, n, dem, maxr)
                recompute_all_times(D, e, l, s, nxt, prv, first,
                                    tstart, lat, nr)
                cur = total_cost(nxt, first, nr, D)
                best_c = cur
                b_nxt, b_prv, b_rte = nxt.copy(), prv.copy(), rte.copy()
                b_first, b_load = first.copy(), load.copy()
                arms_i = bras_instance(n - 1, nr)
                max_chain = chaine_max(n - 1, nr)
                _lex_update()
                _pool_add(arrays_to_routes(b_nxt, b_first, nr))
                if diag is not None:
                    diag["fleet"].append([round(time.time() - t0, 3),
                                          int(nr)])
                if vb:
                    print(f"[FLOTTE] tournee supprimee -> {nr} vehicules, "
                          f"distance {cur:.2f}", flush=True)

        # nouveau cycle : repartir du meilleur, chaud
        cyc_now = int((time.time() - t0) / cyc_len)
        if n_cycles > 1 and cyc_now != cyc_prev:
            cyc_prev = cyc_now
            _pool_add(arrays_to_routes(b_nxt, b_first, nr))
            nxt[:] = b_nxt
            prv[:] = b_prv
            rte[:] = b_rte
            first[:] = b_first
            load[:] = b_load
            recompute_all_times(D, e, l, s, nxt, prv, first,
                                tstart, lat, nr)
            cur = best_c
            acc_pct_prev = (100.0 * (n_acc - acc_c0)
                            / max(it - it_c0, 1))
            if diag is not None and diag["cycles"]:
                diag["cycles"][-1]["acc_pct"] = round(acc_pct_prev, 3)
                dg = int(stats[4]) - dg_c0
                dga = int(stats[5]) - dga_c0
                diag["cycles"][-1]["acc_degr_pct"] = round(
                    100.0 * dga / max(dg, 1), 3)
                diag["cycles"][-1]["stats"] = [int(x) for x in stats]
            m, ndg = 0.0, 0
            clamped = defaulted = False
            if recal:
                cur, best_c, nr, m, ndg = _sonder(cur, best_c, nr, 0)
                it += n_probe
                if m > 0.0:
                    med_d = m
                    T0, Tend, clamped, defaulted = _temperature(m, cur)
            it_c0, acc_c0 = it, n_acc
            dg_c0, dga_c0 = int(stats[4]), int(stats[5])
            if diag is not None:
                diag["cycles"].append(dict(
                    g=min(cyc_now + 1, n_cycles), med=float(m),
                    T0=float(T0), Tend=float(Tend), nd=int(ndg),
                    clamped=bool(clamped), default=bool(defaulted),
                    cost=float(cur),
                    cnt=[round(float(x), 2) for x in cnt]))
            if vb:
                print(f"[JIT-TW] cycle {min(cyc_now + 1, n_cycles)}"
                      f"/{n_cycles} — reprise a {best_c:.2f} "
                      f"({nr} veh.), T0 {T0:.4f}, acceptation "
                      f"{100 * n_acc / max(it, 1):.2f} %", flush=True)

    _lex_update()
    _pool_add(arrays_to_routes(b_nxt, b_first, nr))
    _pool_add(lex_routes)
    if diag is not None:
        if diag["cycles"]:
            diag["cycles"][-1]["acc_pct"] = round(
                100.0 * (n_acc - acc_c0) / max(it - it_c0, 1), 3)
            dg = int(stats[4]) - dg_c0
            dga = int(stats[5]) - dga_c0
            diag["cycles"][-1]["acc_degr_pct"] = round(
                100.0 * dga / max(dg, 1), 3)
            diag["cycles"][-1]["stats"] = [int(x) for x in stats]
        diag["bandit"] = dict(cnt=[round(float(x), 3) for x in cnt],
                              tot_r=[round(float(x), 5) for x in tot_r])
        diag["totals"] = dict(
            it=int(it), n_acc=int(n_acc),
            acc_pct=round(100.0 * n_acc / max(it, 1), 3),
            stats=[int(x) for x in stats],
            time=round(time.time() - t0, 2),
            best=float(lex_key[1]), vehicles=int(len(lex_routes)))
    if vb:
        print(f"[JIT-TW] seed {seed} — {it:,} iters, acceptation "
              f"{100 * n_acc / max(it, 1):.2f} %, meilleur "
              f"{len(lex_routes)} vehicules / {lex_key[1]:.2f}", flush=True)
    return lex_routes, lex_key[1], len(lex_routes), it


#  RESOLUTION D'UNE INSTANCE (une graine)
def solve_tw(inst, D, knn, budget, seed, verbose=False, diag_out=None,
             route_pool=None):
    dem, Q = inst["dem"], inst["Q"]
    e, l, s = inst["e"], inst["l"], inst["s"]
    rng = np.random.default_rng(seed)
    noise = 0.0 if seed % 4 == 0 else 0.1 * (seed % 4)
    routes0 = nn_construct(D, dem, Q, e, l, s, rng, noise=noise)
    out = sisr_jit_tw(routes0, D, dem, Q, e, l, s, knn, budget, seed,
                      n_veh=inst.get("n_veh"), verbose=verbose,
                      diag_out=diag_out, route_pool=route_pool)
    if isinstance(diag_out, dict):
        diag_out.setdefault("meta", {})["noise"] = noise
    return out
