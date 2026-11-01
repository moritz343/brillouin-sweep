#!/usr/bin/env python3
"""Floquet-Bloch band solver for 3D elastic unit cells on a voxel grid.

The cell is split into nx*ny*nz equal voxels, each with its own material.
Elements are 8-node trilinear hexahedra (2x2x2 Gauss), mass is lumped.
Nodes on opposite faces are the same node; when an element wraps across the
cell boundary its DOFs pick up the Bloch phase exp(i k.R). This gives the
Hermitian problem K(k) x = w^2 M x, solved for the lowest modes by
shift-invert Lanczos (scipy eigsh).

Usage:
    python kpoints.py --lattice orthorhombic --n 8 -o k.csv
    .venv/bin/python voxel_fem.py cells/coated_core.json k.csv -o results.csv --jobs 4

Cell file (JSON): see cells/*.json. Geometry is a list of ellipsoids painted
in order (later ones overwrite earlier ones) over a background material.
"""
import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh, splu

MATERIALS = {  # density kg/m3, Young's modulus Pa, Poisson ratio
    "epoxy": (1180.0, 4.35e9, 0.368),
    "steel": (7800.0, 210e9, 0.30),
    "lead": (11600.0, 40.8e9, 0.37),
    "aluminium": (2700.0, 70e9, 0.33),
    "silicone": (1300.0, 1.175e5, 0.469),
    "void": (1e-3, 1e1, 0.30),  # near-empty filler for porous cells
}


def hex8_matrices(h):
    """Unit element matrices for a box with edge lengths h: K = lam*KL + mu*KM, M = rho*M0 (lumped)."""
    g = 1 / math.sqrt(3)
    corners = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float)
    KL = np.zeros((24, 24))
    KM = np.zeros((24, 24))
    vol = h[0] * h[1] * h[2]
    for gx in (-g, g):
        for gy in (-g, g):
            for gz in (-g, g):
                xi = np.array([gx, gy, gz])
                s = 2 * corners - 1  # node signs
                # shape-function derivatives in physical coordinates
                dN = np.empty((8, 3))
                for a in range(8):
                    for c in range(3):
                        o = [(1 + s[a, d] * xi[d]) / 2 for d in range(3)]
                        o[c] = s[a, c] / 2
                        dN[a, c] = o[0] * o[1] * o[2] * 2 / h[c]
                B = np.zeros((6, 24))
                for a in range(8):
                    B[0, 3 * a] = dN[a, 0]
                    B[1, 3 * a + 1] = dN[a, 1]
                    B[2, 3 * a + 2] = dN[a, 2]
                    B[3, 3 * a + 1], B[3, 3 * a + 2] = dN[a, 2], dN[a, 1]
                    B[4, 3 * a], B[4, 3 * a + 2] = dN[a, 2], dN[a, 0]
                    B[5, 3 * a], B[5, 3 * a + 1] = dN[a, 1], dN[a, 0]
                DL = np.zeros((6, 6)); DL[:3, :3] = 1
                DM = np.diag([2, 2, 2, 1, 1, 1.0])
                w = vol / 8
                KL += w * B.T @ DL @ B
                KM += w * B.T @ DM @ B
    M0 = np.eye(24) * vol / 8
    return KL, KM, M0, corners.astype(int)


class Cell:
    def __init__(self, cfg):
        self.a = np.array(cfg["a"], float)
        self.n = np.array(cfg["grid"], int)
        h = self.a / self.n
        mats = [cfg["background"]] + [s["material"] for s in cfg.get("shapes", [])]
        names = sorted(set(mats))
        props = np.array([MATERIALS[m] for m in names])
        rho, E, nu = props.T
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        nx, ny, nz = self.n
        # voxel centres in cell coordinates, origin at the cell centre
        cx = (np.arange(nx) + 0.5) * h[0] - self.a[0] / 2
        cy = (np.arange(ny) + 0.5) * h[1] - self.a[1] / 2
        cz = (np.arange(nz) + 0.5) * h[2] - self.a[2] / 2
        X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
        mat = np.full(X.shape, names.index(cfg["background"]))
        for s in cfg.get("shapes", []):
            r = s["semi_axes"]
            c = s.get("centre", [0, 0, 0])
            inside = ((X - c[0]) / r[0]) ** 2 + ((Y - c[1]) / r[1]) ** 2 + ((Z - c[2]) / r[2]) ** 2 <= 1
            mat[inside] = names.index(s["material"])
        self.material_names, self.voxel_material = names, mat
        KL, KM, M0, corners = hex8_matrices(h)

        nn = nx * ny * nz
        self.ndof = 3 * nn
        I, J, L = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        I, J, L, m = I.ravel(), J.ravel(), L.ravel(), mat.ravel()
        ne = I.size
        node = np.empty((ne, 8), int)
        wrap = np.empty((ne, 8, 3), int)
        for a, (dx, dy, dz) in enumerate(corners):
            ii, jj, ll = I + dx, J + dy, L + dz
            wrap[:, a] = np.stack([ii // nx, jj // ny, ll // nz], 1)
            node[:, a] = (ii % nx) * ny * nz + (jj % ny) * nz + (ll % nz)
        dof = (3 * node[:, :, None] + np.arange(3)).reshape(ne, 24)
        wdof = np.repeat(wrap, 3, axis=1)  # (ne, 24, 3)
        rows = np.repeat(dof, 24, axis=1).ravel()
        cols = np.tile(dof, (1, 24)).ravel()
        vals = (lam[m][:, None, None] * KL + mu[m][:, None, None] * KM).reshape(ne, -1).ravel()
        # Bloch phase for entry (a, b) is exp(i k.(w_b - w_a) a)
        dw = (np.tile(wdof, (1, 24, 1)) - np.repeat(wdof, 24, axis=1)).reshape(-1, 3)
        keep = vals != 0
        self.rows, self.cols, self.vals = rows[keep], cols[keep], vals[keep]
        self.shift = (dw[keep] * self.a).astype(float)  # metres
        md = np.zeros(self.ndof)
        np.add.at(md, dof.ravel(), np.repeat(rho[m], 24) * M0[0, 0])
        self.mdiag = md
        self.minv_sqrt = sp.diags(1 / np.sqrt(md))
        # shift just below zero, scaled by the softest material present, so the
        # lowest modes are well separated after shift-invert
        used = np.unique(m)
        self.sigma = -0.1 * float(np.min(mu[used] / rho[used])) / float(np.min(self.a)) ** 2

    def solve(self, k, nev, backend="auto"):
        phase = np.exp(1j * (self.shift @ np.asarray(k, float)))
        K = sp.csr_matrix((self.vals * phase, (self.rows, self.cols)), shape=(self.ndof, self.ndof))
        H = (self.minv_sqrt @ K @ self.minv_sqrt).tocsc()  # Hermitian, standard form (lumped mass)
        if backend == "auto":
            backend = "pardiso" if _has_pardiso() else "splu"
        if backend == "splu":
            lu = splu((H - self.sigma * sp.identity(self.ndof, format="csc")).tocsc(), permc_spec="MMD_AT_PLUS_A")
            op = LinearOperator(H.shape, matvec=lu.solve, dtype=complex)
            w2, Y = eigsh(H, k=nev, sigma=self.sigma, OPinv=op, which="LM")
        else:
            # real-equivalent form [[Re, -Im], [Im, Re]]: every eigenvalue appears twice
            import pypardiso
            R = sp.bmat([[H.real, -H.imag], [H.imag, H.real]]).tocsr()
            S = (R - self.sigma * sp.identity(R.shape[0], format="csr")).tocsr()
            solver = pypardiso.PyPardisoSolver()
            solver.factorize(S)
            solver.set_phase(33)
            op = LinearOperator(R.shape, dtype=float,
                                matvec=lambda x: solver._call_pardiso(S, np.ascontiguousarray(x, dtype=float)).ravel())
            w2, Z = eigsh(R, k=2 * nev, sigma=self.sigma, OPinv=op, which="LM")
            o = np.argsort(w2)[::2]
            w2, Z = w2[o], Z[:, o]
            Y = Z[: self.ndof] + 1j * Z[self.ndof:]
        order = np.argsort(w2)
        w2, Y = w2[order], Y[:, order]
        f = np.sqrt(np.clip(w2, 0, None)) / (2 * math.pi)
        e = np.abs(Y) ** 2  # mass-weighted coordinates: |y|^2 is kinetic-energy share
        p = np.stack([e[c::3].sum(0) for c in range(3)], 1)
        return f, p / p.sum(1, keepdims=True)


def _has_pardiso():
    try:
        import pypardiso  # noqa: F401
        return True
    except ImportError:
        return False


_cell = None


def _init(cfg):
    global _cell
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _cell = Cell(cfg)


def _run(args):
    pid, k, nev = args
    f, p = _cell.solve(k, nev)
    return pid, f, p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell", help="cell JSON file")
    ap.add_argument("kpoints", help="CSV from kpoints.py")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--nev", type=int, default=12, help="modes per k-point")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--case", help="case name (default: cell file name)")
    args = ap.parse_args(argv)

    cfg = json.load(open(args.cell))
    case = args.case or os.path.splitext(os.path.basename(args.cell))[0]
    a = cfg["a"]
    tasks = []
    with open(args.kpoints) as f:
        for r in csv.DictReader(f):
            k = [float(r[c]) * math.pi / a[i] for i, c in enumerate(("ux", "uy", "uz"))]
            tasks.append((int(r["id"]), k, args.nev))
    probe = Cell(cfg)
    print(f"{case}: {probe.ndof} DOF, materials {probe.material_names}, {len(tasks)} k-points", file=sys.stderr)
    del probe
    with open(args.out, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["case", "id", "band", "freq", "px", "py", "pz"])
        with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(cfg,)) as ex:
            for done, (pid, fr, p) in enumerate(ex.map(_run, tasks, chunksize=4), 1):
                for b in range(len(fr)):
                    w.writerow([case, pid, b + 1, f"{fr[b]:.6f}", *(f"{v:.4f}" for v in p[b])])
                if done % 50 == 0 or done == len(tasks):
                    print(f"  {done}/{len(tasks)}", file=sys.stderr)


if __name__ == "__main__":
    main()
