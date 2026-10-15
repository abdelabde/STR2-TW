
import os
import re

import numpy as np


# instance
def parse_solomon(path):
    """Fichier Solomon / Homberger -> dict(name, n_veh, Q, coords,
    dem, e, l, s). Le noeud 0 est le depot."""
    with open(path) as f:
        lines = [ln.rstrip() for ln in f]
    name = lines[0].strip() if lines else os.path.basename(path)
    n_veh, Q = None, None
    rows = []
    mode = None
    for ln in lines[1:]:
        up = ln.upper()
        if "VEHICLE" in up:
            mode = "veh"
            continue
        if "CUSTOMER" in up:
            mode = "cust"
            continue
        nums = re.findall(r"-?\d+\.?\d*", ln)
        if mode == "veh" and n_veh is None and len(nums) >= 2:
            n_veh, Q = int(float(nums[0])), float(nums[1])
        elif mode == "cust" and len(nums) >= 7:
            rows.append([float(x) for x in nums[:7]])
    if Q is None or not rows:
        raise ValueError(f"format Solomon non reconnu : {path}")
    rows.sort(key=lambda r: r[0])
    arr = np.asarray(rows, dtype=np.float64)
    coords = arr[:, 1:3].copy()
    dem = arr[:, 3].copy()
    e = arr[:, 4].copy()
    l = arr[:, 5].copy()
    s = arr[:, 6].copy()
    dem[0] = 0.0
    s[0] = 0.0
    return dict(name=name, n_veh=int(n_veh or 10 ** 6), Q=float(Q),
                coords=coords, dem=dem, e=e, l=l, s=s)


def is_matrix_file(path):
    """True for the files written by export_matrices_txt.py."""
    opener = __import__("gzip").open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return f.readline().startswith("NAME")


def _parse_matrix_file(path):
    """Text format of export_matrices_txt.py -> (inst, T, Dkm).
    The two matrices are read by the C parser of pandas."""
    import pandas as pd
    opener = __import__("gzip").open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        name = f.readline().split()[1]
        n = int(f.readline().split()[1]) + 1
        parts = f.readline().split()
        n_veh, Q = int(parts[1]), float(parts[3])
    head = 5                                   # NAME .. column header
    nodes = pd.read_csv(path, sep=" ", header=None, skiprows=head,
                        nrows=n, engine="c").to_numpy(np.float64)
    skip_d = head + n + 1                      # + 'DISTANCE_KM'
    Dkm = pd.read_csv(path, sep=" ", header=None, skiprows=skip_d,
                      nrows=n, engine="c", dtype=np.float64).to_numpy()
    skip_t = skip_d + n + 1                    # + 'TIME_MIN'
    T = pd.read_csv(path, sep=" ", header=None, skiprows=skip_t,
                    nrows=n, engine="c", dtype=np.float64).to_numpy()
    dem, s = nodes[:, 1].copy(), nodes[:, 4].copy()
    dem[0] = 0.0
    s[0] = 0.0
    inst = dict(name=name, n_veh=n_veh, Q=Q, coords=None, dem=dem,
                e=nodes[:, 2].copy(), l=nodes[:, 3].copy(), s=s)
    return inst, T, Dkm


def load_instance(path):
    """Instance file -> (inst, T, Dkm).
    T   : travel-time matrix used by the search, for both the schedules
          and the cost (as in the paper);
    Dkm : distance matrix in kilometers, used only to report the final
          distance (None for the Solomon / Gehring-Homberger files).
    For a matrix file, the parsed arrays are cached in <path>.cache.npz,
    so that only the first run pays for the text parsing."""
    if not is_matrix_file(path):
        inst = parse_solomon(path)
        return inst, build_D(inst["coords"]), None
    cache = path + ".cache.npz"
    if os.path.exists(cache) and \
            os.path.getmtime(cache) >= os.path.getmtime(path):
        z = np.load(cache)
        inst = dict(name=str(z["name"]), n_veh=int(z["n_veh"]),
                    Q=float(z["Q"]), coords=None, dem=z["dem"],
                    e=z["e"], l=z["l"], s=z["s"])
        return inst, np.ascontiguousarray(z["T"]), \
            np.ascontiguousarray(z["Dkm"])
    inst, T, Dkm = _parse_matrix_file(path)
    T = np.ascontiguousarray(T)
    Dkm = np.ascontiguousarray(Dkm)
    try:
        np.savez(cache, name=inst["name"], n_veh=inst["n_veh"],
                 Q=inst["Q"], dem=inst["dem"], e=inst["e"], l=inst["l"],
                 s=inst["s"], T=T, Dkm=Dkm)
    except OSError:
        pass
    return inst, T, Dkm


def route_length(routes, M):
    """Sum of M over the arcs of the routes (depot = 0)."""
    tot = 0.0
    for r in routes:
        seq = [c for c in r if c != 0]
        p = 0
        for c in seq:
            tot += M[p, c]
            p = c
        tot += M[p, 0]
    return float(tot)


def build_D(coords):
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=2))
    if os.environ.get("STR2_TW_DIST", "float") == "trunc1":
        D = np.floor(D * 10.0) / 10.0        # ancienne convention Solomon
    np.fill_diagonal(D, 0.0)
    return np.ascontiguousarray(D, dtype=np.float64)


def build_knn(D, K):
    """k plus proches voisins de chaque client (hors depot, hors soi)."""
    n = D.shape[0]
    K = min(K, n - 2)
    idx = np.argsort(D[:, 1:], axis=1)[:, :K + 1] + 1
    out = np.empty((n, K), dtype=np.int32)
    for u in range(n):
        row = [int(v) for v in idx[u] if int(v) != u][:K]
        while len(row) < K:
            row.append(row[-1] if row else 1)
        out[u] = row
    return out


# tableaux <-> routes
def routes_to_arrays(routes, n, dem, max_routes):
    nxt = np.zeros(n, dtype=np.int32)
    prv = np.zeros(n, dtype=np.int32)
    rte = np.full(n, -1, dtype=np.int32)
    first = np.zeros(max_routes, dtype=np.int32)
    load = np.zeros(max_routes, dtype=np.float64)
    r = 0
    for route in routes:
        inner = [c for c in route if c != 0]
        if not inner:
            continue
        first[r] = inner[0]
        p = 0
        for c in inner:
            prv[c] = p
            if p != 0:
                nxt[p] = c
            rte[c] = r
            load[r] += dem[c]
            p = c
        nxt[p] = 0
        r += 1
    return nxt, prv, rte, first, load, r


def arrays_to_routes(nxt, first, nr):
    out = []
    for r in range(nr):
        c = int(first[r])
        if c == 0:
            continue
        route = [0]
        while c != 0:
            route.append(int(c))
            c = int(nxt[c])
        route.append(0)
        out.append(route)
    return out


def total_cost(nxt, first, nr, D):
    tot = 0.0
    for r in range(nr):
        c = int(first[r])
        if c == 0:
            continue
        p = 0
        while c != 0:
            tot += D[p, c]
            p = c
            c = int(nxt[c])
        tot += D[p, 0]
    return float(tot)


# faisabilite d'une sequence
def seq_feasible(seq, D, dem, Q, e, l, s, eps=1e-6):
    """`seq` = clients sans depot. Simulation avant complete."""
    if sum(dem[c] for c in seq) > Q + 1e-9:
        return False
    dep, p = e[0], 0
    for c in seq:
        st = max(e[c], dep + D[p, c])
        if st > l[c] + eps:
            return False
        dep = st + s[c]
        p = c
    return dep + D[p, 0] <= l[0] + eps


def seq_cost(seq, D):
    tot, p = 0.0, 0
    for c in seq:
        tot += D[p, c]
        p = c
    return tot + D[p, 0]


# construction initiale
def nn_construct(D, dem, Q, e, l, s, rng, noise=0.0):
    """Plus proche voisin sequentiel de Solomon : on remplit une tournee
    a la fois, en choisissant a chaque pas le client faisable qui
    minimise  distance + attente + urgence.  Toujours faisable-TW
    (chaque client seul dans une tournee est faisable sur ces jeux).
    `noise` > 0 bruite le critere pour diversifier les iles."""
    n = len(dem)
    unrouted = np.ones(n, dtype=bool)
    unrouted[0] = False
    idx = np.arange(n)
    routes = []
    while unrouted.any():
        seq, ld, p, dep = [], 0.0, 0, e[0]
        while True:
            cand = idx[unrouted]
            cand = cand[ld + dem[cand] <= Q + 1e-9]
            if cand.size == 0:
                break
            arr = dep + D[p, cand]
            st = np.maximum(arr, e[cand])
            okm = (st <= l[cand] + 1e-6) & \
                  (st + s[cand] + D[cand, 0] <= l[0] + 1e-6)
            cand, arr, st = cand[okm], arr[okm], st[okm]
            if cand.size == 0:
                break
            m = D[p, cand] + 0.2 * (st - arr) + 0.2 * (l[cand] - arr)
            if noise > 0.0:
                m = m * (1.0 + noise * rng.random(cand.size))
            u = int(cand[int(np.argmin(m))])
            seq.append(u)
            unrouted[u] = False
            ld += dem[u]
            dep = max(e[u], dep + D[p, u]) + s[u]
            p = u
        if seq:
            routes.append([0] + seq + [0])
        else:                                  # aucun client ne rentre :
            u = int(idx[unrouted][0])          # tournee singleton (surete)
            routes.append([0, u, 0])
            unrouted[u] = False
    return routes


# validation
def validate_tw(routes, D, dem, Q, e, l, s, n, n_veh=None, eps=1e-6):
    """Couverture exacte + capacite + fenetres DURES + retour depot."""
    served = [c for r in routes for c in r if c != 0]
    if len(served) != len(set(served)):
        return False, "clients dupliques"
    if sorted(served) != list(range(1, n)):
        return False, f"couverture {len(set(served))}/{n - 1}"
    for i, r in enumerate(routes, 1):
        if not r or r[0] != 0 or r[-1] != 0:
            return False, f"route {i} sans depot"
        seq = [c for c in r if c != 0]
        if sum(dem[c] for c in seq) > Q + 1e-9:
            return False, f"route {i} surchargee"
        dep, p = e[0], 0
        for c in seq:
            st = max(e[c], dep + D[p, c])
            if st > l[c] + eps:
                return False, (f"route {i} : fenetre violee au client {c} "
                               f"(debut {st:.2f} > due {l[c]:.2f})")
            dep = st + s[c]
            p = c
        if dep + D[p, 0] > l[0] + eps:
            return False, f"route {i} : retour depot apres l0"
    if n_veh is not None and len(routes) > n_veh:
        return False, f"{len(routes)} vehicules > flotte {n_veh}"
    return True, ""
