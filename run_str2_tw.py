
import glob
import json
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np

from core_tw import load_instance, build_knn, validate_tw, route_length
from str2_tw import solve_tw

_G = {}


def _worker(args):
    idx, budget, seed = args
    inst, D, knn = _G["inst"], _G["D"], _G["knn"]
    name = _G.get("name", "instance")
    verbose = idx == 0 or bool(int(os.environ.get("STR2_VERBOSE", 0)))
    diag = {} if int(os.environ.get("STR2_DIAG", 1)) else None
    routes, dist, nveh, it = solve_tw(inst, D, knn, budget, seed,
                                      verbose=verbose, diag_out=diag)
    if diag:
        ddir = os.environ.get("STR2_DIAGDIR", "diagnostics")
        try:
            os.makedirs(ddir, exist_ok=True)
            with open(os.path.join(
                    ddir, f"diagnostics_{name}_{seed}.json"), "w") as f:
                json.dump(diag, f)
        except Exception as exc:
            print(f"[DIAG] ecriture impossible ({exc})", flush=True)
    return nveh, dist, routes, it


def _init(inst, D, knn, name=""):
    _G.update(inst=inst, D=D, knn=knn, name=name)


def load_bks():
    """bks_tw.csv : 'instance,vehicles,distance' — vehicles peut etre
    vide (convention DIMACS : la flotte n'est pas minimisee), lignes
    '#' ignorees. Accepte aussi 'instance,distance'."""
    out = {}
    p = os.environ.get("STR2_BKS_TW",
                       os.path.join(os.path.dirname(
                           os.path.abspath(__file__)), "bks_tw.csv"))
    if os.path.exists(p):
        with open(p) as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = [x.strip() for x in ln.replace(";", ",").split(",")]
                try:
                    if len(parts) >= 3:
                        veh = int(float(parts[1])) if parts[1] else None
                        out[parts[0].upper()] = (veh, float(parts[2]))
                    elif len(parts) == 2:
                        out[parts[0].upper()] = (None, float(parts[1]))
                except ValueError:
                    continue
    return out


def bks_from_sol(path_txt, D):
    """Si <instance>.sol existe a cote du .txt, lit ses routes et
    recalcule le cout AVEC NOTRE matrice D — le BKS est ainsi exprime
    dans la MEME convention de distance que nos solutions (float ou
    trunc1), ce qui rend le gap exact. Retourne (n_veh, dist) ou None."""
    base = os.path.splitext(path_txt)[0]
    for ext in (".sol", ".SOL"):
        ps = base + ext
        if not os.path.exists(ps):
            continue
        routes = []
        try:
            with open(ps) as f:
                for ln in f:
                    if ":" not in ln:
                        continue
                    head, _, tail = ln.partition(":")
                    if "route" not in head.lower():
                        continue
                    seq = [int(x) for x in re.findall(r"\d+", tail)]
                    if seq:
                        routes.append(seq)
        except Exception:
            return None
        if not routes:
            return None
        dist = 0.0
        for seq in routes:
            p = 0
            for c in seq:
                dist += D[p, c]
                p = c
            dist += D[p, 0]
        return len(routes), float(dist)
    return None


def solve_file(path, budget, W):
    inst, D, Dkm = load_instance(path)
    n = len(inst["dem"])
    name = os.path.splitext(os.path.basename(path))[0]
    if budget == "auto":
        budget_s = max(60.0, 0.6 * (n - 1))
    else:
        budget_s = float(budget)
    K = int(os.environ.get("STR2_KNN", 30))
    knn = build_knn(D, K)

    print(f"\n=== {name} : {n - 1} clients, Q={inst['Q']:.0f}, flotte max "
          f"{inst['n_veh']}, budget {budget_s:.0f}s x {W} ile(s) ===",
          flush=True)
    t0 = time.time()
    jobs = [(i, budget_s, 1000 * i + 7) for i in range(W)]
    if W <= 1:
        _init(inst, D, knn, name)
        res = [_worker(jobs[0])]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(W, initializer=_init,
                      initargs=(inst, D, knn, name)) as pool:
            res = pool.map(_worker, jobs)
    res.sort(key=lambda z: (z[0], z[1]))          # lexicographique
    nveh, dist, routes, _ = res[0]
    dt = time.time() - t0
    tot_it = sum(z[3] for z in res)

    ok, why = validate_tw(routes, D, inst["dem"], inst["Q"],
                          inst["e"], inst["l"], inst["s"], n,
                          n_veh=None)
    # BKS : le .sol livre a cote de l'instance fait foi (cout recalcule
    # dans NOTRE convention de distance) ; la table CSV n'est qu'un repli.
    bks = bks_from_sol(path, D)
    src = ".sol"
    if bks is None:
        bks = load_bks().get(name.upper())
        src = "table"
    bks_v = bks[0] if bks else None
    bks_d = bks[1] if bks else None
    if bks_d:
        print(f"[BKS] {name} : {bks_d:.2f}"
              + (f" / {bks_v} veh." if bks_v else "")
              + f" ({src})", flush=True)
    print(f"{'=' * 64}")
    print(f"instance   : {name}")
    print(f"vehicules  : {nveh}" + (f"   (BKS {bks_v})" if bks_v else ""))
    dist_km = route_length(routes, Dkm) if Dkm is not None else None
    if dist_km is not None:
        print(f"distance km: {dist_km:.2f}")
    print(f"distance   : {dist:.2f}"
          + (f"   (BKS {bks_d:.2f}, gap "
             f"{100 * (dist - bks_d) / bks_d:+.2f} %)" if bks_d else ""))
    print(f"temps      : {dt:.0f} s, {tot_it:,} iterations cumulees, "
          f"dispersion des iles "
          f"{res[0][0]}-{res[-1][0]} veh. / "
          f"{res[0][1]:.1f}-{res[-1][1]:.1f}")
    print(f"VALIDITE   : {'SOLUTION VALIDE (TW dures respectees)' if ok else 'INVALIDE — ' + why}")
    print(f"{'=' * 64}", flush=True)

    sol_dir = os.environ.get("STR2_SOLDIR", "solutions")
    os.makedirs(sol_dir, exist_ok=True)
    out = os.path.join(sol_dir, f"solution_{name}.txt")
    with open(out, "w") as f:
        f.write(f"Instance : {name}\n")
        f.write(f"Vehicles : {nveh}\nDistance : {dist:.2f}\n")
        f.write(f"Valid    : {ok}\n")
        if bks_d:
            f.write(f"BKS      : {bks_d:.2f} "
                    f"(gap {100 * (dist - bks_d) / bks_d:+.2f} %)\n")
        for i, r in enumerate(routes, 1):
            seq = " ".join(str(c) for c in r if c != 0)
            f.write(f"Route {i} : {seq}\n")
    print(f"{out} ecrit", flush=True)
    return dict(instance=name, vehicles=nveh, distance=dist,
                distance_km=dist_km, valid=ok,
                why=why, time_s=dt, solution_file=out,
                bks_veh=bks_v, bks_dist=bks_d)


INSTANCE_DIR = os.environ.get(
    "STR2_TW_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "instances"))


def _pool_instances():
    """Tous les .txt/.TXT sous INSTANCE_DIR, SOUS-DOSSIERS COMPRIS
    (ex. instances/Holmberger/...)."""
    pats = [os.path.join(INSTANCE_DIR, "**", "*.txt"),
            os.path.join(INSTANCE_DIR, "**", "*.TXT")]
    out = set()
    for p in pats:
        out.update(glob.glob(p, recursive=True))
    return sorted(out)


def resolve_spec(spec):
    """<spec> -> liste de fichiers. Accepte : un chemin de fichier, un
    dossier, un motif glob, un NOM d'instance ('C1_2_1', avec ou sans
    extension, insensible a la casse — cherche recursivement dans
    ./instances), un PREFIXE de famille ('C1', 'C1_10', 'all')."""
    if os.path.isfile(spec):
        return [spec]
    if os.path.isdir(spec):
        out = set(glob.glob(os.path.join(spec, "**", "*.txt"),
                            recursive=True))
        out.update(glob.glob(os.path.join(spec, "**", "*.TXT"),
                             recursive=True))
        return sorted(out)
    if any(ch in spec for ch in "*?["):
        hits = sorted(glob.glob(spec, recursive=True))
        if hits:
            return hits
        # motif sans chemin ('*_10_*') : on l'applique aux NOMS des
        # instances du pool (insensible a la casse)
        import fnmatch
        pat = spec.upper()
        if not pat.endswith((".TXT",)):
            pat = pat + "*"
        return [p for p in _pool_instances()
                if fnmatch.fnmatch(os.path.basename(p).upper(), pat)]
    pool = _pool_instances()
    key = os.path.splitext(spec)[0].upper()
    if key in ("ALL", "TOUT"):
        return pool
    exact = [p for p in pool
             if os.path.splitext(os.path.basename(p))[0].upper() == key]
    if exact:
        return exact
    return [p for p in pool
            if os.path.basename(p).upper().startswith(key)]


def ecrire_resultats(rows, xlsx=None):
    """Ecrit le recapitulatif Excel (feuille 'resultats' + 'synthese')
    apres CHAQUE instance — une interruption ne perd rien. CSV en
    repli si pandas/openpyxl est absent. `xlsx` : nom de fichier
    explicite (sinon STR2_XLSX ou resultats_str2_tw.xlsx)."""
    recs = []
    for r in rows:
        d, bd = r.get("distance"), r.get("bks_dist")
        recs.append(dict(
            instance=r.get("instance"),
            vehicules=r.get("vehicles"),
            distance=round(d, 2) if d else None,
            distance_km=(round(r["distance_km"], 2)
                         if r.get("distance_km") is not None else None),
            BKS_veh=r.get("bks_veh"),
            BKS_dist=bd,
            gap_dist_pct=(round(100 * (d - bd) / bd, 2)
                          if d and bd else None),
            ecart_veh=(r.get("vehicles") - r.get("bks_veh")
                       if r.get("vehicles") and r.get("bks_veh") else None),
            valide=r.get("valid"),
            motif=r.get("why", ""),
            fichier_solution=r.get("solution_file", ""),
            temps_s=round(r.get("time_s", 0)),
            horodatage=time.strftime("%Y-%m-%d %H:%M:%S")))
    try:
        import pandas as pd
        df = pd.DataFrame(recs)
        okdf = df[df["valide"] == True]
        g = okdf["gap_dist_pct"].dropna() if len(okdf) else []
        syn = pd.DataFrame([dict(
            instances=len(df), valides=len(okdf),
            gap_moyen_pct=round(float(g.mean()), 2) if len(g) else None,
            gap_median_pct=round(float(g.median()), 2) if len(g) else None,
            gap_min_pct=round(float(g.min()), 2) if len(g) else None,
            gap_max_pct=round(float(g.max()), 2) if len(g) else None)])
        out = xlsx or os.environ.get("STR2_XLSX", "resultats_str2_tw.xlsx")
        with pd.ExcelWriter(out, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="resultats", index=False)
            syn.to_excel(w, sheet_name="synthese", index=False)
        return out
    except Exception:
        out = os.path.splitext(
            xlsx or os.environ.get("STR2_XLSX",
                                   "resultats_str2_tw.xlsx"))[0] + ".csv"
        with open(out, "w") as f:
            cols = list(recs[0].keys()) if recs else []
            f.write(",".join(cols) + "\n")
            for rc in recs:
                f.write(",".join("" if rc[c] is None else str(rc[c])
                                 for c in cols) + "\n")
        return out


def main():
    spec = sys.argv[1] if len(sys.argv) > 1 else "all"
    budget = sys.argv[2] if len(sys.argv) > 2 else "auto"
    if budget != "auto":
        budget = float(budget)
    W = int(sys.argv[3]) if len(sys.argv) > 3 else max(
        1, mp.cpu_count() - 1)

    files = resolve_spec(spec)
    if not files:
        print(f"aucune instance trouvee pour '{spec}' "
              f"(cherchee aussi dans {INSTANCE_DIR})")
        return 2
    print(f"{len(files)} instance(s) : "
          f"{', '.join(os.path.basename(p) for p in files[:12])}"
          f"{' ...' if len(files) > 12 else ''}", flush=True)

    rows = []
    for p in files:
        try:
            rows.append(solve_file(p, budget, W))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            rows.append(dict(instance=os.path.basename(p), valid=False,
                             why=f"{type(exc).__name__}: {exc}"))
        # recapitulatif mis a jour apres CHAQUE instance
        out = ecrire_resultats(rows)
        print(f"[EXCEL] {len(rows)}/{len(files)} instances -> {out}",
              flush=True)
    n_ok = sum(1 for r in rows if r.get("valid"))
    print(f"\n{n_ok}/{len(rows)} instances valides — resultats dans {out}")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
