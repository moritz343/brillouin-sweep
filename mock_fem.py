#!/usr/bin/env python3
"""Stand-in for the FEM solver, to test the pipeline before real runs.

Solves a scalar mass-in-mass spring lattice (host springs along x, y, z plus
optional farther-neighbour springs, and a local resonator) at every k-point
from kpoints.py and writes results in the format analyze.py expects.

Usage:
    python3 mock_fem.py k.csv > results.csv
"""
import csv
import math
import sys
from itertools import permutations, product

CASES = {
    # axial springs only: band edges at zone corners, path should be exact
    "axial": dict(k=[1, 0.5, 0.25], e=[0, 0, 0], kr=1.0, mu=1.0),
    # extra <110>, <111>, <200> springs: band-1 maximum strictly inside the zone
    "extra_springs": dict(k=[1, 0.41, 0.87], e=[0.23, 0.02, 0.38], kr=2.77, mu=0.51),
}


def orbit(v):
    seen = set()
    for p in permutations(v):
        for s in product((1, -1), repeat=3):
            w = tuple(a * b for a, b in zip(p, s))
            if next((x for x in w if x), 0) < 0:
                w = tuple(-x for x in w)
            seen.add(w)
    return sorted(seen)


SHELLS = [orbit(v) for v in ((1, 1, 0), (1, 1, 1), (2, 0, 0))]


def bands(kv, p):
    A = sum(2 * p["k"][i] * (1 - math.cos(kv[i])) for i in range(3))
    for s, sh in enumerate(SHELLS):
        if p["e"][s]:
            A += sum(p["e"][s] * 2 * (1 - math.cos(d[0] * kv[0] + d[1] * kv[1] + d[2] * kv[2])) for d in sh)
    kr, mu = p["kr"], p["mu"]
    a = A + kr
    b = a * mu + kr
    c = a * kr - kr * kr
    d = math.sqrt(max(b * b - 4 * mu * c, 0.0))
    return [math.sqrt(max((b - d) / (2 * mu), 0.0)), math.sqrt((b + d) / (2 * mu))]


def main():
    w = csv.writer(sys.stdout)
    w.writerow(["case", "id", "band", "freq"])
    with open(sys.argv[1]) as f:
        rows = list(csv.DictReader(f))
    for name, p in CASES.items():
        for r in rows:
            kv = [float(r[c]) * math.pi for c in ("ux", "uy", "uz")]
            for n, fr in enumerate(bands(kv, p), start=1):
                w.writerow([name, r["id"], n, f"{fr:.8f}"])


if __name__ == "__main__":
    main()
