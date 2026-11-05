"""Symmetric (both xy diagonals) vs symmetry-broken (only (1,1,0)) mass-in-mass cell.

Finds the gap between bands 3 and 4 over the path (IBZ box edges), the faces and
the volume, with local refinement. Reproduces the example on the page
("When the faces matter"):

    cd ~/brillouin-sweep && .venv/bin/python study/symmetry_pair.py
"""
import numpy as np
from scipy.optimize import minimize
P = np.pi


def bands(k, p, sym):
    k = np.atleast_2d(np.asarray(k, float)); c = 1 - np.cos(k); D = np.zeros((len(k), 3, 3))
    for i in range(3):
        for q in range(3):
            D[:, q, q] += 2 * (p["k"][i] if i == q else p["s"] * p["k"][i]) * c[:, i]
    for d in ([(1, 1, 0), (1, -1, 0)] if sym else [(1, 1, 0)]):
        n = np.array(d) / np.sqrt(2)
        D += 2 * p["g"] * (1 - np.cos(k @ np.array(d, float)))[:, None, None] * np.outer(n, n)
    r = np.array(p["r"], float); mu = p["mu"]; K = np.zeros((len(k), 6, 6)); K[:, :3, :3] = D
    for a in range(3):
        K[:, a, a] += r[a]; K[:, a, a + 3] = K[:, a + 3, a] = -r[a] / np.sqrt(mu); K[:, a + 3, a + 3] = r[a] / mu
    return np.sqrt(np.clip(np.linalg.eigvalsh(K), 0, None))


def box(sym):
    """IBZ: [0,π]³ with both diagonals (mirror symmetric); kx doubles to [-π,π] without the partner."""
    return (np.array([0, 0, 0.]) if sym else np.array([-P, 0, 0.])), np.array([P, P, P])


def grids(sym, n=16):
    lo, hi = box(sym)
    ax = [np.linspace(lo[i], hi[i], (2 * n if (i == 0 and not sym) else n) + 1) for i in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    on = sum(np.isclose(G[:, i], lo[i]) | np.isclose(G[:, i], hi[i]) for i in range(3))
    return {"path": G[on >= 2], "faces": G[on >= 1], "vol": G}


def ext(p, sym, b, sgn, pts, mode):
    """Extremum (sgn=+1 max, -1 min) of band b over pts, refined within the edge/face/volume."""
    lo, hi = box(sym)
    w = bands(pts, p, sym)[:, b]; j = np.argmax(sgn * w); k0 = pts[j].copy(); best = w[j]
    bnd = [i for i in range(3) if np.isclose(k0[i], lo[i]) or np.isclose(k0[i], hi[i])]
    if mode == "path":
        free = [i for i in range(3) if i not in bnd] if len(bnd) == 2 else []
    elif mode == "faces":
        free = [i for i in range(3) if i != bnd[0]] if bnd else []
    else:
        free = [0, 1, 2]
    if free:
        def f(x):
            k = k0.copy(); k[free] = x; return -sgn * bands(k, p, sym)[0, b]
        r = minimize(f, k0[free], method="L-BFGS-B", bounds=[(lo[i], hi[i]) for i in free])
        if -r.fun > sgn * best:
            best = sgn * -r.fun; k0[free] = r.x
    return best, k0


def where(k, sym):
    lo, hi = box(sym)
    m = sum(min(abs(k[i] - lo[i]), abs(k[i] - hi[i])) < 1e-4 for i in range(3))
    return {3: "corner", 2: "edge", 1: "face", 0: "interior"}[m]


def gap(p, sym, b=2):
    """{path|faces|vol: (gap, (max of band b, k), (min of band b+1, k))}"""
    G = grids(sym); out = {}
    for d in ("path", "faces", "vol"):
        lo = ext(p, sym, b, 1, G[d], d); hi = ext(p, sym, b + 1, -1, G[d], d)
        out[d] = (hi[0] - lo[0], lo, hi)
    for a, bb in (("faces", "path"), ("vol", "faces")):  # supersets
        if out[bb][0] < out[a][0]:
            out[a] = out[bb]
    return out


if __name__ == "__main__":
    ex = dict(k=[0.8, 1.3, 0.6], r=[10, 2.5, 8], mu=0.6, s=0.6, g=1.6)
    for sym in (True, False):
        R = gap(ex, sym)
        print(f"{'both diagonals' if sym else 'only (1,1,0)':15s}",
              "  ".join(f"gap {d}: {R[d][0]:.4f}" for d in ("path", "faces", "vol")),
              "| band-3 max:", where(R["vol"][1][1], sym))
