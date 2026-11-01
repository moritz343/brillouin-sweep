#!/usr/bin/env python3
"""Generate nested k-point sets over the irreducible Brillouin zone (IBZ).

One grid covers the whole IBZ volume. Every point is tagged with how many IBZ
boundary planes it lies on:
    level "edge"   -> on >= 2 planes  (the standard band-diagram path)
    level "face"   -> on exactly 1 plane
    level "inside" -> on none
So a single FEM sweep gives all three sampling levels, and the analysis can
compare path vs faces vs volume at the same k-spacing.

Supported lattices (box-shaped Brillouin zone, axis-aligned cells):
    cubic         Laue class m-3m   IBZ: 0 <= kz <= ky <= kx <= pi/a   (tetrahedron)
    tetragonal    Laue class 4/mmm  IBZ: 0 <= ky <= kx <= pi/a, 0 <= kz <= pi/c (prism)
    orthorhombic  Laue class mmm    IBZ: 0 <= ki <= pi/ai              (box)

Usage:
    python3 kpoints.py --lattice orthorhombic --n 12 --a 0.01 0.01 0.01 -o k.csv
    python3 kpoints.py --lattice cubic --n 20 --summary
"""
import argparse
import csv
import math
import sys


def boundary_planes(lattice, n):
    """Return functions of integer indices (i, j, l) that are True on each IBZ plane."""
    if lattice == "cubic":
        return [lambda i, j, l: l == 0, lambda i, j, l: j == l,
                lambda i, j, l: i == j, lambda i, j, l: i == n]
    if lattice == "tetragonal":
        return [lambda i, j, l: j == 0, lambda i, j, l: i == j, lambda i, j, l: i == n,
                lambda i, j, l: l == 0, lambda i, j, l: l == n]
    if lattice == "orthorhombic":
        return [lambda i, j, l: i == 0, lambda i, j, l: i == n, lambda i, j, l: j == 0,
                lambda i, j, l: j == n, lambda i, j, l: l == 0, lambda i, j, l: l == n]
    raise ValueError(lattice)


def inside_ibz(lattice, i, j, l):
    if lattice == "cubic":
        return i >= j >= l
    if lattice == "tetragonal":
        return i >= j
    return True


def generate(lattice, n, inside_stride=1):
    planes = boundary_planes(lattice, n)
    pts = []
    for i in range(n + 1):
        for j in range(n + 1):
            for l in range(n + 1):
                if not inside_ibz(lattice, i, j, l):
                    continue
                on = sum(1 for p in planes if p(i, j, l))
                level = "edge" if on >= 2 else "face" if on == 1 else "inside"
                if level == "inside" and (i % inside_stride or j % inside_stride or l % inside_stride):
                    continue
                pts.append((i, j, l, level))
    return pts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lattice", required=True, choices=["cubic", "tetragonal", "orthorhombic"])
    ap.add_argument("--n", type=int, default=12, help="grid steps per pi/a (k-spacing = pi/(n a))")
    ap.add_argument("--a", type=float, nargs=3, default=[1.0, 1.0, 1.0], metavar=("AX", "AY", "AZ"),
                    help="lattice constants in metres, used for absolute k in rad/m")
    ap.add_argument("--inside-stride", type=int, default=1,
                    help="keep only every s-th interior point per axis (coarser, cheaper volume check)")
    ap.add_argument("--levels", default="edge,face,inside",
                    help="comma list of levels to write (default: all)")
    ap.add_argument("-o", "--out", help="output CSV (default: stdout)")
    ap.add_argument("--summary", action="store_true", help="only print point counts")
    args = ap.parse_args(argv)

    pts = generate(args.lattice, args.n, args.inside_stride)
    counts = {lv: sum(1 for p in pts if p[3] == lv) for lv in ("edge", "face", "inside")}
    msg = (f"{args.lattice}, n={args.n}: edge {counts['edge']}, face {counts['face']}, "
           f"inside {counts['inside']}; path={counts['edge']}, "
           f"faces incl. path={counts['edge'] + counts['face']}, volume={len(pts)}")
    print(msg, file=sys.stderr)
    if args.summary:
        return

    keep = set(args.levels.split(","))
    out = open(args.out, "w", newline="") if args.out else sys.stdout
    w = csv.writer(out)
    w.writerow(["id", "i", "j", "l", "n", "level", "ux", "uy", "uz", "kx", "ky", "kz"])
    for pid, (i, j, l, level) in enumerate(pts):
        if level not in keep:
            continue
        u = (i / args.n, j / args.n, l / args.n)          # fractions of pi/a_i
        k = [u[c] * math.pi / args.a[c] for c in range(3)]  # rad/m
        w.writerow([pid, i, j, l, args.n, level, *(f"{v:.6f}" for v in u), *(f"{v:.8g}" for v in k)])
    if args.out:
        out.close()


if __name__ == "__main__":
    main()
