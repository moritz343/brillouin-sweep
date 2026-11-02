#!/usr/bin/env python3
"""Polar k-point set for a cubic cell: rays from Γ along the three IBZ faces through Γ.

The cubic IBZ is the tetrahedron Γ X M R. Three of its faces contain Γ, so
each is a polar wedge (angle = direction, radius = |k|):

    A  k_z = 0     [100] -> [110]   0 ... 45°
    C  k_y = k_z   [100] -> [111]   0 ... 54.7°
    D  k_x = k_y   [110] -> [111]   0 ... 35.3°

Every ray runs from Γ to the outer face k_x = π/a, with radii spaced
quadratically (denser near Γ, where acoustic bands and gap edges change fast).

    python3 polar_kpoints.py runs/liu_polar          # writes k.csv + rays.csv

k.csv (id, ux, uy, uz in units of π/a) feeds voxel_fem.py; rays.csv maps
(face, direction index, radial step) to k-point ids for plotting.
"""
import argparse
import csv
import math
import os

SQ2 = math.sqrt(2)
FACES = {
    "A": (45.0, lambda t: (math.cos(t), math.sin(t), 0.0)),
    "C": (math.degrees(math.atan(SQ2)), lambda t: (math.cos(t), math.sin(t) / SQ2, math.sin(t) / SQ2)),
    "D": (math.degrees(math.atan(1 / SQ2)), lambda t: (math.cos(t) / SQ2, math.cos(t) / SQ2, math.sin(t))),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir")
    ap.add_argument("--step-deg", type=float, default=7.5, help="approximate angle between rays")
    ap.add_argument("--nr", type=int, default=8, help="radial points per ray (excluding Γ)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    key = lambda u: tuple(round(c, 6) + 0.0 for c in u)
    pts = {key((0, 0, 0)): 0}
    rows = []
    for f, (amax, direction) in FACES.items():
        nd = max(2, round(amax / a.step_deg) + 1)
        for i in range(nd):
            d = direction(math.radians(amax * i / (nd - 1)))
            reach = 1 / d[0]  # ray meets the outer face k_x = π/a
            for m in range(1, a.nr + 1):
                u = key([reach * (m / a.nr) ** 2 * c for c in d])
                pts.setdefault(u, len(pts))
                rows.append((f, i, round(amax * i / (nd - 1), 3), m, pts[u]))
    with open(os.path.join(a.outdir, "k.csv"), "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["id", "ux", "uy", "uz"])
        for u, i in sorted(pts.items(), key=lambda x: x[1]):
            w.writerow([i, *(f"{c:.6f}" for c in u)])
    with open(os.path.join(a.outdir, "rays.csv"), "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["face", "dir", "angle_deg", "step", "id"])
        w.writerows(rows)
    print(f"{len(pts)} unique k-points, {len(rows)} ray samples -> {a.outdir}")


if __name__ == "__main__":
    main()
