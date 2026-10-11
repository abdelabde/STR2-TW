# STR2-TW

**A self-tuning ruin-and-recreate heuristic for large-scale vehicle routing with time windows**

This repository contains the code accompanying the paper:

> *A Self-Tuning Ruin-and-Recreate Heuristic for Large-Scale Vehicle Routing with Time Windows.*
> <Authors>. Submitted to *INFORMS Journal on Computing*, 2026.

STR2-TW solves the vehicle routing problem with time windows (VRPTW) with hard
windows, at the size of industrial operations (several thousand customers) and
under practical time budgets (one hour or less). It builds on the
ruin-and-recreate framework and makes its main parameters self-tuned:

- **Acceptance threshold.** The simulated annealing threshold is not fixed in
  advance. It is calibrated on the instance being solved, from the median cost
  of the degrading moves of the search, so that a typical degrading move is
  accepted with a target probability `p`. The measurement is repeated at the
  start of every cycle.
- **Removal sizes.** The number of customers removed by a perturbation is set
  in proportion to the average route length of the instance, and a discounted
  upper-confidence-bound bandit selects among the candidate sizes during the
  run.
- **Hard feasibility at every moment.** Insertions are screened by a
  constant-time time-window test based on the forward time slack of
  Savelsbergh; removals never need to be tested. Every intermediate solution
  is feasible, so the solver can be stopped at any time with a valid plan.
- **Parallel islands.** `W` independent searches run in parallel and exchange
  their best solutions periodically.

---

## Repository structure

| File | Content |
|---|---|
| `core_tw.py` | Array representation of a solution, Solomon-format reader, neighbor lists, initial construction, and an independent validator that re-simulates every route from the depot. |
| `kernel_tw.py` | Compiled kernel (Numba): forward and backward schedules, constant-time insertion test, ruin and recreate operators, fallback scan, measurement mode. |
| `str2_tw.py` | Island loop: threshold calibration, cycles, bandit, fleet minimization. |
| `run_str2_tw.py` | Entry point: process management, shared matrices, objectives, result files and diagnostics. |
| `run_pyvrp_tw.py` | Runs PyVRP on the same instances, with the same matrices and the same validator, for the comparison of the paper. |
| `exploit_diagnostics.py` | Builds the tables and figures of the paper from the diagnostics files. |
| `instances/` | Instance files in Solomon format (see *Data* below). |

---

## Installation

Tested with Python 3.11 on Linux (AlmaLinux 9).

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
pip install numpy numba pandas openpyxl matplotlib
pip install pyvrp==0.11.3          # only for the comparison with PyVRP
```

The optional set-covering recombination stage requires <commercial MIP solver
and version>. It is disabled in all the experiments of the paper, and the
solver runs without it.

---

## Usage

### Solving instances

```bash
python run_str2_tw.py <spec> <budget_s> <n_islands>
```

- `<spec>`: an instance file, a folder, a glob pattern (in quotes), or a name
  prefix.
- `<budget_s>`: time budget in seconds.
- `<n_islands>`: number of parallel islands (one single-threaded process each).

Examples:

```bash
# one instance, one hour, 20 islands
python run_str2_tw.py instances/industrial/I1.txt 3600 20

# all instances of a folder
python run_str2_tw.py "instances/industrial/*.txt" 3600 20
```

### Objective

The default objective is **lexicographic**: the number of routes is minimized
first, then the total distance. This is the convention of the paper. The
distance-only convention of the DIMACS challenge is selected with:

```bash
NOVA_OBJ=dist NOVA_TW_DIST=trunc1 python run_str2_tw.py <spec> <budget_s> <n_islands>
```

### Main environment variables

| Variable | Effect |
|---|---|
| `NOVA_OBJ` | `lex` (default) or `dist`. |
| `NOVA_TW_DIST` | Distance convention; `trunc1` truncates distances to one decimal (DIMACS). |
| `NOVA_XLSX` | Name of the result spreadsheet (default `resultats_nova_tw.xlsx`). |
| `NOVA_DIAGDIR` | Folder of the diagnostics files (default `diagnostics/`). |

The full list is given in the header of `run_str2_tw.py`.

### Outputs

- **Result spreadsheet**: vehicles, distance, validity and time for each
  instance.
- **Solutions**: `solutions/solution_<instance>.txt`, one route per line.
- **Diagnostics**: `diagnostics/diagnostics_<instance>_<seed>.json`, one file
  per island, with the cost trajectory, the measured threshold of every
  cycle, the sample of degrading moves, the fleet reductions and the bandit
  allocation.

Every solution is checked by the independent validator of `core_tw.py`
before being reported.

---

## Comparison with PyVRP

```bash
python run_pyvrp_tw.py <spec> <budget_s> <n_islands>
```

PyVRP receives the same instance files and the same distance matrix (scaled to
integers), runs the same number of single-threaded processes, and its routes
are re-validated and re-costed by the validator of `core_tw.py`, so that the
two solvers are compared on exactly the same basis.

On very large instances, the start of the PyVRP processes can exceed the
memory of the machine. `PYVRP_STAGGER=<s>` starts island `i` after `i × s`
seconds and removes this delay from its budget, so that all islands stop at
the same time:

```bash
PYVRP_STAGGER=8 python run_pyvrp_tw.py instances/industrial/I8.txt 3600 20
```

The other options (`PYVRP_OBJ`, `PYVRP_FLEET_FIX`, `PYVRP_FLEET_SLACK`,
`PYVRP_ALLOW_INFEAS`, `PYVRP_DIAG`, …) are documented in the header of
`run_pyvrp_tw.py`.

---

## Reproducing the experiments of the paper

```bash
# STR2-TW on the eight industrial instances (lexicographic, 1 h, 20 islands)
python run_str2_tw.py "instances/industrial/*.txt" 3600 20

# PyVRP on the same instances
python run_pyvrp_tw.py "instances/industrial/*.txt" 3600 20

# tables and figures
python exploit_diagnostics.py diagnostics figures_paper
python exploit_diagnostics.py diagnostics_pyvrp figures_paper
```

All experiments of the paper were run on one Intel Core i7-12700 (12 cores,
20 hardware threads) with 64 GB of RAM. Each instance is solved by 20
single-threaded processes, and the best plan is retained. Results depend on
the machine, since the budget is a wall-clock time.

---

## Data

Instances use the Solomon format. Coordinates are planar, centered on the
depot, and expressed so that the Euclidean distance between two points is the
travel time in minutes at a constant speed of 50 km/h; kilometers are
obtained by dividing a cost by 1.2. Time windows and service durations are in
minutes from midnight.

| Instance | Customers | Description |
|---|---|---|
| I1–I5 | 2,315 to 2,919 | Montréal area, one morning of orders each |
| I6 | 2,175 | Montréal depot, deliveries gathered over the period |
| I7 | 4,198 | Toronto depot, deliveries gathered over the period |
| I8 | 6,424 | London (Ontario) depot, deliveries gathered over the period |

<State here whether the industrial instances are included, anonymized, or
available on request, according to the agreement with the industrial
partner.>

---

## Citation

If you use this code, please cite:

```bibtex
@article{str2tw2026,
  author  = {<Authors>},
  title   = {A Self-Tuning Ruin-and-Recreate Heuristic for Large-Scale
             Vehicle Routing with Time Windows},
  journal = {INFORMS Journal on Computing},
  year    = {2026},
  note    = {Submitted}
}
```

---

## License

<License, e.g. MIT.>
