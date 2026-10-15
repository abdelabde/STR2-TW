
import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:                                   # repli pur Python
    HAVE_NUMBA = False

    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]

        def deco(f):
            return f
        return deco


MAXREM = 256          # borne dure sur le nombre de clients retires
EPS = 1e-6            # tolerance numerique sur les temps


@njit(cache=True, fastmath=True)
def update_route_times(r, D, e, l, s, nxt, prv, first, tstart, lat):
    """Rafraichit tstart (avant) et lat (arriere) de la tournee r.
    O(longueur de la tournee). Suppose la tournee faisable."""
    e0 = e[0]
    l0 = l[0]
    # propagation AVANT : debuts de service courants
    c = first[r]
    p = 0
    dep = e0                       # depart du depot
    last = 0
    while c != 0:
        arr = dep + D[p, c]
        st = arr if arr > e[c] else e[c]
        tstart[c] = st
        dep = st + s[c]
        p = c
        last = c
        c = nxt[c]
    # propagation ARRIERE : debuts de service au plus tard
    c = last
    nx = 0
    while c != 0:
        if nx == 0:
            lt = l0 - D[c, 0] - s[c]
        else:
            lt = lat[nx] - D[c, nx] - s[c]
        if lt > l[c]:
            lt = l[c]
        lat[c] = lt
        nx = c
        c = prv[c]


@njit(cache=True, fastmath=True)
def recompute_all_times(D, e, l, s, nxt, prv, first, tstart, lat, nr):
    for r in range(nr):
        update_route_times(r, D, e, l, s, nxt, prv, first, tstart, lat)


@njit(cache=True, fastmath=True)
def route_feasible(r, D, e, l, s, nxt, first, Q, dem, load_r):
    """Verification complete d'une tournee (debug / validation)."""
    if load_r > Q + 1e-9:
        return False
    c = first[r]
    p = 0
    dep = e[0]
    while c != 0:
        arr = dep + D[p, c]
        st = arr if arr > e[c] else e[c]
        if st > l[c] + EPS:
            return False
        dep = st + s[c]
        p = c
        c = nxt[c]
    if dep + D[p, 0] > l[0] + EPS:
        return False
    return True


@njit(cache=True, fastmath=True)
def rr_chunk_tw(D, dem, Q, e, l, s, knn,
                nxt, prv, rte, first, load, nr,
                tstart, lat,
                b_nxt, b_prv, b_rte, b_first, b_load,
                cur, best_c, n_iters, avg_remove, blink, T0, Tend, frac0,
                frac_span, rem_buf, undo_c, undo_a, undo_r,
                place_buf, place_r, tr_buf,
                max_chain, dbuf, probe, stats, noscan):
    """Execute n_iters iterations de RR avec fenetres de temps DURES.
    Modifie les tableaux en place. `stats` et `noscan` : voir
    l'en-tete du module.
    Retourne (cur, best_c, nr, n_accept, n_deltas)."""
    n = dem.shape[0]
    K = knn.shape[1]
    n_acc = 0
    n_d = 0
    cap_d = dbuf.shape[0]
    e0 = e[0]
    l0 = l[0]

    for itr in range(n_iters):
        # RUIN : chaines autour d'un germe
        target = 1 + int(np.random.random() * 2.0 * avg_remove)
        if target > MAXREM:
            target = MAXREM
        seed_c = 1 + int(np.random.random() * (n - 1))
        if seed_c >= n:
            seed_c = n - 1
        n_rem = 0
        n_undo = 0
        delta = 0.0

        for jj in range(K + 1):
            if n_rem >= target:
                break
            v = seed_c if jj == 0 else knn[seed_c, jj - 1]
            r = rte[v]
            if r < 0:
                continue
            dup = False                    # une seule chaine par tournee
            for z in range(n_undo):
                if undo_r[z] == r:
                    dup = True
                    break
            if dup:
                continue
            L = 1 + int(np.random.random() * max_chain)
            if L > target - n_rem:
                L = target - n_rem
            c = v
            back = int(np.random.random() * L)
            for _ in range(back):
                if prv[c] == 0:
                    break
                c = prv[c]
            removed_here = 0
            for _ in range(L):
                if c == 0 or rte[c] < 0 or n_rem >= MAXREM:
                    break
                nx = nxt[c]
                a = prv[c]
                b = nxt[c]
                delta -= D[a, c] + D[c, b] - D[a, b]
                undo_c[n_undo] = c
                undo_a[n_undo] = a
                undo_r[n_undo] = rte[c]
                n_undo += 1
                rc = rte[c]                # retrait O(1)
                if a == 0:
                    first[rc] = b
                else:
                    nxt[a] = b
                if b != 0:
                    prv[b] = a
                load[rc] -= dem[c]
                rte[c] = -1
                nxt[c] = 0
                prv[c] = 0
                rem_buf[n_rem] = c
                n_rem += 1
                removed_here += 1
                c = nx
            if removed_here > 0:
                # la tournee a change : ses temps doivent etre exacts
                # avant toute evaluation d'insertion
                update_route_times(r, D, e, l, s, nxt, prv, first,
                                   tstart, lat)

        if n_rem == 0:
            continue

        # melange des retires (Fisher-Yates)
        for z in range(n_rem - 1, 0, -1):
            w = int(np.random.random() * (z + 1))
            tmp = rem_buf[z]
            rem_buf[z] = rem_buf[w]
            rem_buf[w] = tmp

        # RECREATE : insertion granulaire + TW
        ok = True
        n_pl = 0
        for z in range(n_rem):
            u = rem_buf[z]
            du = dem[u]
            eu = e[u]
            lu = l[u]
            su = s[u]
            bd = 1e18
            ba = -1
            br = -1
            for j in range(K):
                v = knn[u, j]
                rv = rte[v]
                if rv < 0:
                    continue
                if load[rv] + du > Q + 1e-9:
                    continue
                stats[2] += 1
                # position 1 : juste APRES v
                if np.random.random() < blink:
                    stats[3] += 1
                else:
                    w = nxt[v]
                    dep_a = tstart[v] + s[v]
                    arr_u = dep_a + D[v, u]
                    st_u = arr_u if arr_u > eu else eu
                    if st_u <= lu + EPS:
                        dep_u = st_u + su
                        if w == 0:
                            feas = dep_u + D[u, 0] <= l0 + EPS
                        else:
                            feas = dep_u + D[u, w] <= lat[w] + EPS
                        if feas:
                            d = D[v, u] + D[u, w] - D[v, w]
                            if d < bd:
                                bd = d
                                ba = v
                                br = rv
                # position 2 : juste AVANT v
                if np.random.random() < blink:
                    stats[3] += 1
                else:
                    p = prv[v]
                    dep_a = e0 if p == 0 else tstart[p] + s[p]
                    arr_u = dep_a + D[p, u]
                    st_u = arr_u if arr_u > eu else eu
                    if st_u <= lu + EPS:
                        dep_u = st_u + su
                        if dep_u + D[u, v] <= lat[v] + EPS:
                            d = D[p, u] + D[u, v] - D[p, v]
                            if d < bd:
                                bd = d
                                ba = p
                                br = rv
                # position 3 : tete de la tournee de v
                h = first[rv]
                arr_u = e0 + D[0, u]
                st_u = arr_u if arr_u > eu else eu
                if st_u <= lu + EPS:
                    dep_u = st_u + su
                    if h == 0:
                        feas = dep_u + D[u, 0] <= l0 + EPS
                        d = 2.0 * D[0, u]
                    else:
                        feas = dep_u + D[u, h] <= lat[h] + EPS
                        d = D[0, u] + D[u, h] - D[0, h]
                    if feas and d < bd:
                        bd = d
                        ba = 0
                        br = rv
            if br < 0 and noscan == 0:
                # repli : balayage complet des positions (rare, mais
                # indispensable avec TW — la seule position faisable
                # peut etre loin des kNN)
                stats[0] += 1
                arr_h = e0 + D[0, u]
                st_h = arr_h if arr_h > eu else eu
                for r in range(nr):
                    if load[r] + du > Q + 1e-9:
                        continue
                    h = first[r]
                    # tete (couvre aussi la tournee vide)
                    if st_h <= lu + EPS:
                        dep_u = st_h + su
                        if h == 0:
                            if dep_u + D[u, 0] <= l0 + EPS:
                                d = 2.0 * D[0, u]
                                if d < bd:
                                    bd = d
                                    ba = 0
                                    br = r
                        else:
                            if dep_u + D[u, h] <= lat[h] + EPS:
                                d = D[0, u] + D[u, h] - D[0, h]
                                if d < bd:
                                    bd = d
                                    ba = 0
                                    br = r
                    # toutes les positions internes
                    c = h
                    while c != 0:
                        w = nxt[c]
                        dep_a = tstart[c] + s[c]
                        arr_u = dep_a + D[c, u]
                        st_u = arr_u if arr_u > eu else eu
                        if st_u <= lu + EPS:
                            dep_u = st_u + su
                            if w == 0:
                                feas = dep_u + D[u, 0] <= l0 + EPS
                            else:
                                feas = dep_u + D[u, w] <= lat[w] + EPS
                            if feas:
                                d = D[c, u] + D[u, w] - D[c, w]
                                if d < bd:
                                    bd = d
                                    ba = c
                                    br = r
                        c = w
            if br < 0:
                ok = False
                stats[1] += 1
                break
            # insertion O(1)
            if ba == 0:
                b = first[br]
                first[br] = u
                prv[u] = 0
                nxt[u] = b
                if b != 0:
                    prv[b] = u
            else:
                b = nxt[ba]
                nxt[ba] = u
                prv[u] = ba
                nxt[u] = b
                if b != 0:
                    prv[b] = u
            rte[u] = br
            load[br] += du
            place_buf[n_pl] = u
            place_r[n_pl] = br
            n_pl += 1
            delta += bd
            # les temps de la tournee receptrice changent : rafraichir
            update_route_times(br, D, e, l, s, nxt, prv, first,
                               tstart, lat)

        # acceptation / annulation
        undo_it = not ok
        if ok:
            c_new = cur + delta
            acc = c_new < cur - 1e-9
            if probe == 1:
                if delta > 1e-9 and n_d < cap_d:
                    dbuf[n_d] = delta
                    n_d += 1
            elif not acc:
                f = frac0 + frac_span * (itr + 1) / n_iters
                if f > 1.0:
                    f = 1.0
                T = T0 * (Tend / T0) ** f
                acc = c_new < cur + T * (-np.log(np.random.random() + 1e-18))
            if acc:
                cur = c_new
                n_acc += 1
                if c_new < best_c - 1e-9:
                    best_c = c_new
                    for z in range(n):
                        b_nxt[z] = nxt[z]
                        b_prv[z] = prv[z]
                        b_rte[z] = rte[z]
                    for z in range(nr):
                        b_first[z] = first[z]
                        b_load[z] = load[z]
            else:
                undo_it = True

        if undo_it:
            # retirer ce qui a ete place
            for z in range(n_pl):
                u = place_buf[z]
                ru = rte[u]
                a = prv[u]
                b = nxt[u]
                if a == 0:
                    first[ru] = b
                else:
                    nxt[a] = b
                if b != 0:
                    prv[b] = a
                load[ru] -= dem[u]
                rte[u] = -1
                nxt[u] = 0
                prv[u] = 0
            # remettre ce qui avait ete retire (ordre inverse)
            for z in range(n_undo - 1, -1, -1):
                u = undo_c[z]
                a = undo_a[z]
                r = undo_r[z]
                if a == 0:
                    b = first[r]
                    first[r] = u
                    prv[u] = 0
                    nxt[u] = b
                    if b != 0:
                        prv[b] = u
                else:
                    b = nxt[a]
                    nxt[a] = u
                    prv[u] = a
                    nxt[u] = b
                    if b != 0:
                        prv[b] = u
                rte[u] = r
                load[r] += dem[u]
            # rafraichir les temps de TOUTES les tournees touchees
            # (dedup par petit balayage : au plus 2*MAXREM entrees)
            n_tr = 0
            for z in range(n_undo):
                r = undo_r[z]
                seen = False
                for y in range(n_tr):
                    if tr_buf[y] == r:
                        seen = True
                        break
                if not seen:
                    tr_buf[n_tr] = r
                    n_tr += 1
            for z in range(n_pl):
                r = place_r[z]
                seen = False
                for y in range(n_tr):
                    if tr_buf[y] == r:
                        seen = True
                        break
                if not seen:
                    tr_buf[n_tr] = r
                    n_tr += 1
            for y in range(n_tr):
                update_route_times(tr_buf[y], D, e, l, s, nxt, prv,
                                   first, tstart, lat)

    return cur, best_c, nr, n_acc, n_d
