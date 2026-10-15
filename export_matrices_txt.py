
import argparse
import gzip
import os

import numpy as np

from core_tw import parse_solomon, build_D

SPEED_KMH = 50.0
MIN_PER_KM = 60.0 / SPEED_KMH            # 1.2

# instance name in the paper -> instance file (adapt the paths if needed)
INSTANCES = {
    "I_1": "instances/2020/d2020_08_17.txt",
    "I_2": "instances/2020/d2020_08_18.txt",
    "I_3": "instances/2020/d2020_08_19.txt",
    "I_4": "instances/2020_fix/d2020_08_20.txt",
    "I_5": "instances/2020/d2020_08_21.txt",
    "I_6": "instances/stops/stops_L6.txt",
    "I_7": "instances/stops/stops_L2.txt",
    "I_8": "instances/stops/stops_L1101.txt",
}


def write_instance(name, path, out_dir, decimals, use_gzip):
    inst = parse_solomon(path)
    T = build_D(inst["coords"])                 # minutes
    Dkm = T / MIN_PER_KM                        # kilometers
    n = len(inst["dem"])
    fmt = f"%.{decimals}f"

    ext = ".txt.gz" if use_gzip else ".txt"
    out = os.path.join(out_dir, name + ext)
    opener = (lambda p: gzip.open(p, "wt")) if use_gzip else \
        (lambda p: open(p, "w"))
    with opener(out) as f:
        f.write(f"NAME {name}\n")
        f.write(f"CUSTOMERS {n - 1}\n")
        f.write(f"VEHICLES {inst['n_veh']} CAPACITY {inst['Q']:g}\n")
        f.write("NODES\n")
        f.write("id demand ready due service\n")
        for i in range(n):
            f.write(f"{i} {inst['dem'][i]:g} {inst['e'][i]:g} "
                    f"{inst['l'][i]:g} {inst['s'][i]:g}\n")
        f.write("DISTANCE_KM\n")
        np.savetxt(f, Dkm, fmt=fmt)
        f.write("TIME_MIN\n")
        np.savetxt(f, T, fmt=fmt)
        f.write("EOF\n")
    print(f"{name}: {n - 1} customers -> {out} "
          f"({os.path.getsize(out) / 1e6:.0f} MB)", flush=True)


def read_instance(path):
    """Reads a file written above. Returns (inst, Dkm, T), with inst in
    the format of core_tw.parse_solomon (coords set to zeros)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        name = f.readline().split()[1]
        n = int(f.readline().split()[1]) + 1
        parts = f.readline().split()
        n_veh, Q = int(parts[1]), float(parts[3])
        f.readline()                               # NODES
        f.readline()                               # header
        nodes = np.array([f.readline().split() for _ in range(n)],
                         dtype=float)
        f.readline()                               # DISTANCE_KM
        Dkm = np.array([f.readline().split() for _ in range(n)],
                       dtype=float)
        f.readline()                               # TIME_MIN
        T = np.array([f.readline().split() for _ in range(n)], dtype=float)
    inst = dict(name=name, n_veh=n_veh, Q=Q, coords=np.zeros((n, 2)),
                dem=nodes[:, 1], e=nodes[:, 2], l=nodes[:, 3],
                s=nodes[:, 4])
    return inst, Dkm, np.ascontiguousarray(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default="instances_matrices")
    ap.add_argument("--decimals", type=int, default=3)
    ap.add_argument("--gzip", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for name, path in INSTANCES.items():
        if not os.path.exists(path):
            print(f"{name}: file not found ({path}), skipped")
            continue
        write_instance(name, path, a.out_dir, a.decimals, a.gzip)


if __name__ == "__main__":
    main()
