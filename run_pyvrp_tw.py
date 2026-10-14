"""
  PyVRP sur les MEMES instances que STR2-TW — pour comparaison
    python run_pyvrp_tw.py <spec> [budget_s] [n_procs]

  Exemples (memes specs que run_str2_tw.py) :
    python run_pyvrp_tw.py d2020 1200 20
    STR2_TW_DIST=trunc1 python run_pyvrp_tw.py "*_10_*" 2400 20

Principe de la comparaison :
  * MEMES fichiers d'instances (resolution de noms identique, via
    run_str2_tw.resolve_spec) ;
  * MEME matrice de distances : on construit ProblemData directement
    avec la matrice D de core_tw (convention STR2_TW_DIST comprise),
    multipliee par SCALE=1000 puis arrondie (PyVRP travaille en
    entiers). Fenetres et services sont multiplies par le meme
    facteur : le probleme est identique au facteur d'echelle pres ;
  * MEME validation : les routes rendues par PyVRP sont revalidees et
    recoutees par NOTRE validate_tw / total_cost — les couts affiches
    sont donc directement comparables a ceux de STR2-TW, sans dependre
    de l'arrondi interne de PyVRP ;
  * MEME protocole d'iles : W processus PyVRP (mono-thread chacun)
    avec des graines differentes, meilleur retenu ;
  * MEME objectif : PYVRP_OBJ=lex (defaut) enchaine minimise_fleet
    puis solve a flotte fixee (vehicules d'abord, comme STR2-TW) ;
    PYVRP_OBJ=dist = distance seule (convention DIMACS).
  * PYVRP_FLEET_SLACK=k ajoute k vehicules a la cible rendue par
    minimise_fleet avant la phase de distance (0 par defaut) —
    utile quand la flotte minimale rend la journee infaisable.
  * PYVRP_FLEET_FIX=k : impose exactement k vehicules (la phase
    minimise_fleet est sautee) — pour comparer a flotte identique,
    par exemple a la flotte du plan STR2-TW sur une journee limite.
  * PYVRP_ALLOW_INFEAS=1 : si aucune solution faisable n'existe au
    bout du budget, retourner quand meme le meilleur plan interne
    (infaisable), qui sera marque INVALIDE par le validateur ; le
    retard maximal au depot est imprime. En pratique, un plan avec
    un retour tardif tolere vaut mieux qu'aucun plan.
  * PYVRP_STAGGER=s (0 par defaut) : l'ile i attend i x s secondes
    avant de construire ses donnees, et ce delai est retire de son
    budget (toutes les iles s'arretent au meme instant). Evite que
    les W iles atteignent ensemble leur pic memoire de demarrage
    (calcul des voisins) sur les tres grandes instances. Avec la
    valeur par defaut, le comportement est strictement inchange.

Sorties : solutions_pyvrp/solution_<instance>.txt et
resultats_pyvrp_tw.xlsx (memes colonnes que resultats_str2_tw.xlsx).

Diagnostics : chaque ile ecrit
  diagnostics_pyvrp/diagnostics_<instance>_<seed>.json
avec la trajectoire du meilleur cout faisable (temps, cout — extraite
de collect_stats), la duree et le resultat de la phase de flotte, et
le nombre de generations. Ces fichiers alimentent la figure duel
STR2-TW vs PyVRP de exploit_diagnostics.py. PYVRP_DIAG=0 desactive.
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

from core_tw import load_instance, validate_tw, route_length
from run_str2_tw import resolve_spec, bks_from_sol, load_bks

SCALE = int(os.environ.get("PYVRP_SCALE", 1000))

_G = {}


def build_problem_data(inst, D):
    """ProblemData PyVRP a partir de NOTRE matrice (entiers x SCALE).
    Compatible 0.11 (Client(x, y, ...)) ET >= 0.14 (Location)."""
    import pyvrp
    from pyvrp import Client, Depot, ProblemData, VehicleType

    n = len(inst["dem"])
    e, l, s, dem = inst["e"], inst["l"], inst["s"], inst["dem"]
    xy = inst["coords"] if inst["coords"] is not None \
        else np.zeros((len(inst["dem"]), 2))
    Di = np.round(D * SCALE).astype(np.int64)
    tw = lambda x: int(round(x * SCALE))

    vt = VehicleType(num_available=int(inst["n_veh"]),
                     capacity=[int(inst["Q"])],
                     tw_early=tw(e[0]), tw_late=tw(l[0]))
    if hasattr(pyvrp, "Location"):                     # API >= 0.14
        from pyvrp import Location
        locs = [Location(x=float(xy[c, 0]), y=float(xy[c, 1]))
                for c in range(n)]
        depot = Depot(location=0, tw_early=tw(e[0]), tw_late=tw(l[0]))
        clients = [Client(location=c, delivery=[int(dem[c])],
                          service_duration=tw(s[c]),
                          tw_early=tw(e[c]), tw_late=tw(l[c]))
                   for c in range(1, n)]
        return ProblemData(locations=locs, clients=clients,
                           depots=[depot], vehicle_types=[vt],
                           distance_matrices=[Di],
                           duration_matrices=[Di])
    # API 0.9 - 0.11 : coordonnees portees par Client/Depot
    depot = Depot(x=int(round(xy[0, 0] * SCALE)),
                  y=int(round(xy[0, 1] * SCALE)),
                  tw_early=tw(e[0]), tw_late=tw(l[0]))
    clients = [Client(x=int(round(xy[c, 0] * SCALE)),
                      y=int(round(xy[c, 1] * SCALE)),
                      delivery=[int(dem[c])],
                      service_duration=tw(s[c]),
                      tw_early=tw(e[c]), tw_late=tw(l[c]))
               for c in range(1, n)]
    return ProblemData(clients=clients, depots=[depot],
                       vehicle_types=[vt],
                       distance_matrices=[Di],
                       duration_matrices=[Di])


def _routes_of(sol):
    """0.11 : Route.visits() donne les indices de LOCATION (depots en
    tete, donc client = index direct dans notre numerotation).
    >= 0.14 : visits() a disparu, on lit schedule() dont idx est
    l'index CLIENT 0-based -> location = idx + 1."""
    out = []
    for r in sol.routes():
        if hasattr(r, "visits"):
            seq = [int(v) for v in r.visits()]
        else:
            seq = [int(a.idx) + 1 for a in r.schedule()
                   if a.is_client()]
        if seq:
            out.append([0] + seq + [0])
    return out


def _worker(args):
    idx, budget, seed = args
    from pyvrp import ProblemData, VehicleType, minimise_fleet, solve
    from pyvrp.stop import MaxRuntime

    # demarrages decales (PYVRP_STAGGER=s, 0 par defaut = inchange) :
    # l'ile idx attend idx x s secondes AVANT de construire ses donnees,
    # pour que les pics memoire de demarrage (construction des
    # matrices, calcul des voisins) ne se cumulent pas. Le delai est
    # retire du budget de recherche : toutes les iles s'arretent au
    # meme instant.
    delay = float(os.environ.get("PYVRP_STAGGER", 0)) * idx
    if delay > 0:
        time.sleep(delay)

    inst, D, obj = _G["inst"], _G["D"], _G["obj"]
    data = build_problem_data(inst, D)
    t0 = time.time()
    if obj == "lex":
        # phase 1 : minimiser la flotte (10 % du budget, borne 120 s)
        fleet_fix = int(os.environ.get("PYVRP_FLEET_FIX", 0))
        if fleet_fix > 0:
            vt = data.vehicle_types()[0].replace(num_available=fleet_fix)
            data = data.replace(vehicle_types=[vt])
        else:
            fb = min(0.10 * budget, 120.0)
            try:
                vt = minimise_fleet(data, stop=MaxRuntime(fb), seed=seed)
                slack = int(os.environ.get("PYVRP_FLEET_SLACK", 0))
                if slack > 0:
                    vt = vt.replace(
                        num_available=vt.num_available + slack)
                data = data.replace(vehicle_types=[vt])
            except Exception as exc:
                if idx == 0:
                    print(f"[PyVRP] minimise_fleet ignore : {exc}",
                          flush=True)
    # le delai de demarrage compte dans le temps ecoule : la
    # trajectoire reste alignee sur l'horloge commune des iles
    t_fleet = time.time() - t0 + delay
    k_fleet = data.vehicle_types()[0].num_available \
        if hasattr(data, "vehicle_types") else -1
    rest = max(5.0, budget - (time.time() - t0) - delay)
    want_diag = int(os.environ.get("PYVRP_DIAG", 1))
    res = solve(data, stop=MaxRuntime(rest), seed=seed, display=False,
                collect_stats=bool(want_diag))
    if want_diag:
        _write_diag(res, seed, t_fleet, k_fleet, budget)
    if not res.is_feasible():
        if not int(os.environ.get("PYVRP_ALLOW_INFEAS", 0)):
            return None
        try:
            routes = _routes_of(res.best)
        except Exception:
            return None
        if not routes:
            return None
        inst, D = _G["inst"], _G["D"]
        dist = sum(
            D[0, r[0]] + sum(D[a, b] for a, b in zip(r, r[1:]))
            + D[r[-1], 0]
            for r in routes if r)
        late = _max_lateness(routes, D, inst)
        try:
            tw = float(res.best.time_warp()) / SCALE
        except Exception:
            tw = float("nan")
        print(f"[PyVRP] seed {seed} : distance pure {dist:.1f}, "
              f"time warp interne {tw:.1f} min", flush=True)
        print(f"[PyVRP] seed {seed} : aucun plan faisable — meilleur "
              f"infaisable rendu ({len(routes)} veh., {dist:.1f}, "
              f"retard max au depot {late:.1f} min)", flush=True)
        return routes, dist, len(routes)
    routes = _routes_of(res.best)
    return routes, res.cost() / SCALE, len(routes)


def _max_lateness(routes, D, inst):
    """Retard maximal (minutes) sur les fins de fenetres et le
    retour depot, par re-simulation au plus tot."""
    e, l, s = inst["e"], inst["l"], inst["s"]
    worst = 0.0
    for r in routes:
        t, prev = e[0], 0
        for c in r:
            t = max(t + D[prev, c], e[c])
            worst = max(worst, t - l[c])
            t += s[c]
            prev = c
        t += D[prev, 0]
        worst = max(worst, t - l[0])
    return max(worst, 0.0)


def _write_diag(res, seed, t_fleet, k_fleet, budget):
    """Trajectoire (temps, meilleur cout faisable) depuis
    collect_stats, sous-echantillonnee aux ameliorations. Le temps est
    decale de la phase de flotte pour etre comparable au mur STR2-TW."""
    try:
        st = res.stats
        data_pts = getattr(st, "data", None)
        if data_pts is None:                     # API < 0.14
            data_pts = getattr(st, "feas_stats", None)
        rts = getattr(st, "runtimes", None)
        traj = []
        traj_kind = "feasible"
        if data_pts is not None and rts is not None:
            t = t_fleet
            last = None
            for d, r in zip(data_pts, rts):
                t += float(r)
                feas = getattr(d, "best_feas", True)
                bc = getattr(d, "best_cost", None)
                if not feas or bc is None:
                    continue
                bc = float(bc)
                if not np.isfinite(bc):          # pas encore de faisable
                    continue
                bc = bc / SCALE
                if last is None or bc < last - 1e-9:
                    traj.append([round(t, 3), round(bc, 4), 0, 0, 0])
                    last = bc
            if not traj:
                # journee sans solution faisable : repli sur le minimum
                # courant de l'objectif PENALISE (seul signal existant)
                traj_kind = "penalized"
                t, last = t_fleet, None
                for d, r in zip(data_pts, rts):
                    t += float(r)
                    cc = getattr(d, "current_cost", None)
                    if cc is None or not np.isfinite(float(cc)):
                        continue
                    cc = float(cc) / SCALE
                    if last is None or cc < last - 1e-9:
                        traj.append([round(t, 3), round(cc, 4), 0, 0, 0])
                        last = cc
        diag = dict(
            meta=dict(seed=int(seed), solver="pyvrp",
                      budget=float(budget), fleet_time=round(t_fleet, 2),
                      fleet_vehicles=int(k_fleet),
                      traj_kind=traj_kind),
            traj=traj,
            totals=dict(it=int(getattr(res, "num_iterations", 0)),
                        best=(float(res.cost()) / SCALE
                              if res.is_feasible() else None),
                        vehicles=(len(_routes_of(res.best))
                                  if res.is_feasible() else None),
                        time=round(float(getattr(res, "runtime", 0.0))
                                   + t_fleet, 2)),
            cycles=[], delta_sample=[], fleet=[], bandit={})
        ddir = os.environ.get("PYVRP_DIAGDIR", "diagnostics_pyvrp")
        os.makedirs(ddir, exist_ok=True)
        name = _G.get("name", "instance")
        with open(os.path.join(
                ddir, f"diagnostics_{name}_{seed}.json"), "w") as f:
            json.dump(diag, f)
    except Exception as exc:
        print(f"[DIAG-PyVRP] ecriture impossible ({exc})", flush=True)


def _init(inst, D, obj, name=""):
    _G.update(inst=inst, D=D, obj=obj, name=name)


def solve_file(path, budget, W):
    inst, D, Dkm = load_instance(path)
    n = len(inst["dem"])
    name = os.path.splitext(os.path.basename(path))[0]
    budget_s = max(60.0, 0.6 * (n - 1)) if budget == "auto" else float(budget)
    obj = os.environ.get("PYVRP_OBJ", "lex")

    print(f"\n=== [PyVRP] {name} : {n - 1} clients, Q={inst['Q']:.0f}, "
          f"flotte max {inst['n_veh']}, objectif {obj}, "
          f"budget {budget_s:.0f}s x {W} ile(s) ===", flush=True)
    stag = float(os.environ.get("PYVRP_STAGGER", 0))
    if stag > 0 and W > 1:
        print(f"[PyVRP] demarrages decales de {stag:g} s "
              f"(derniere ile : +{stag * (W - 1):g} s, retires de son "
              f"budget)", flush=True)
    t0 = time.time()
    jobs = [(i, budget_s, 1000 * i + 7) for i in range(W)]
    if W <= 1:
        _init(inst, D, obj, name)
        res = [_worker(jobs[0])]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(W, initializer=_init,
                      initargs=(inst, D, obj, name)) as pool:
            res = pool.map(_worker, jobs)
    res = [r for r in res if r is not None]
    if not res:
        print("PyVRP : aucune ile n'a trouve de solution faisable")
        return dict(instance=name, valid=False, why="infaisable",
                    time_s=time.time() - t0)

    # recout dans NOTRE convention, puis choix selon l'objectif
    from core_tw import routes_to_arrays, total_cost
    scored = []
    for routes, _, nveh in res:
        nxt, _, _, first, _, nr = routes_to_arrays(
            routes, n, inst["dem"], len(routes) + 2)
        c = total_cost(nxt, first, nr, D)
        scored.append(((nveh, c) if obj == "lex" else (0, c),
                       routes, c, nveh))
    scored.sort(key=lambda z: z[0])
    _, routes, dist, nveh = scored[0]
    dt = time.time() - t0

    ok, why = validate_tw(routes, D, inst["dem"], inst["Q"],
                          inst["e"], inst["l"], inst["s"], n, n_veh=None)
    bks = bks_from_sol(path, D) or load_bks().get(name.upper())
    bks_v = bks[0] if bks else None
    bks_d = bks[1] if bks else None
    print(f"{'=' * 64}")
    print(f"instance   : {name}  [PyVRP]")
    print(f"vehicules  : {nveh}" + (f"   (BKS {bks_v})" if bks_v else ""))
    dist_km = route_length(routes, Dkm) if Dkm is not None else None
    if dist_km is not None:
        print(f"distance km: {dist_km:.2f}")
    print(f"distance   : {dist:.2f}"
          + (f"   (BKS {bks_d:.2f}, gap "
             f"{100 * (dist - bks_d) / bks_d:+.2f} %)" if bks_d else ""))
    print(f"temps      : {dt:.0f} s, dispersion des iles "
          f"{scored[0][3]}-{scored[-1][3]} veh. / "
          f"{scored[0][2]:.1f}-{scored[-1][2]:.1f}")
    print(f"VALIDITE   : "
          f"{'SOLUTION VALIDE (TW dures respectees)' if ok else 'INVALIDE — ' + why}")
    print(f"{'=' * 64}", flush=True)

    sol_dir = os.environ.get("PYVRP_SOLDIR", "solutions_pyvrp")
    os.makedirs(sol_dir, exist_ok=True)
    out = os.path.join(sol_dir, f"solution_{name}.txt")
    with open(out, "w") as f:
        f.write(f"Instance : {name}\nSolver   : PyVRP\n")
        f.write(f"Vehicles : {nveh}\nDistance : {dist:.2f}\n")
        f.write(f"Valid    : {ok}\n")
        for i, r in enumerate(routes, 1):
            f.write(f"Route {i} : "
                    + " ".join(str(c) for c in r if c != 0) + "\n")
    return dict(instance=name, vehicles=nveh, distance=dist,
                distance_km=dist_km, valid=ok,
                why=why, time_s=dt, solution_file=out,
                bks_veh=bks_v, bks_dist=bks_d)


def main():
    try:
        import pyvrp                                   # noqa
    except ImportError:
        print("PyVRP absent : pip install pyvrp")
        return 2
    try:
        from importlib.metadata import version
        v = version("pyvrp")
        print(f"PyVRP {v}", flush=True)
        if tuple(int(x) for x in v.split(".")[:2]) < (0, 9):
            print("version trop ancienne (< 0.9) : "
                  "python -m pip install -U --user pyvrp")
            return 2
    except Exception:
        pass
    spec = sys.argv[1] if len(sys.argv) > 1 else "all"
    budget = sys.argv[2] if len(sys.argv) > 2 else "auto"
    if budget != "auto":
        budget = float(budget)
    W = int(sys.argv[3]) if len(sys.argv) > 3 else max(1, mp.cpu_count() - 1)

    files = resolve_spec(spec)
    if not files:
        print(f"aucune instance trouvee pour '{spec}'")
        return 2
    print(f"{len(files)} instance(s) : "
          f"{', '.join(os.path.basename(p) for p in files[:12])}"
          f"{' ...' if len(files) > 12 else ''}", flush=True)

    import run_str2_tw
    rows = []
    for p in files:
        try:
            rows.append(solve_file(p, budget, W))
        except Exception:
            import traceback
            traceback.print_exc()
            rows.append(dict(instance=os.path.basename(p), valid=False,
                             why="exception"))
        # fichier DEDIE a PyVRP : jamais le meme que celui de STR2-TW
        out = run_str2_tw.ecrire_resultats(
            rows, xlsx=os.environ.get("PYVRP_XLSX",
                                      "resultats_pyvrp_tw.xlsx"))
        print(f"[EXCEL] {len(rows)}/{len(files)} -> {out}", flush=True)
    n_ok = sum(1 for r in rows if r.get("valid"))
    print(f"\n{n_ok}/{len(rows)} instances valides — resultats dans {out}")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())