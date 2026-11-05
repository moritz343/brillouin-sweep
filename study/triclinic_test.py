"""Triclinic test: single diagonals (1,1,0), (0,1,1), (1,0,1), no mirror partners, unequal springs.
Only inversion (time reversal) is left, so no IBZ face is a mirror plane.
IBZ: half the zone, kx, ky in [-π, π], kz in [0, π]. Where does the band-3 maximum sit, and do faces suffice?"""
import collections
import numpy as np
from scipy.optimize import minimize
P = np.pi
DIRS = [(1, 1, 0), (0, 1, 1), (1, 0, 1)]
LO, HI = np.array([-P, -P, 0.]), np.array([P, P, P])


def bands(k, p):
    k = np.atleast_2d(np.asarray(k, float)); c = 1 - np.cos(k); D = np.zeros((len(k), 3, 3))
    for i in range(3):
        for q in range(3):
            D[:, q, q] += 2 * (p["k"][i] if i == q else p["s"] * p["k"][i]) * c[:, i]
    for d, g in zip(DIRS, p["g"]):
        n = np.array(d) / np.sqrt(2)
        D += 2 * g * (1 - np.cos(k @ np.array(d, float)))[:, None, None] * np.outer(n, n)
    r = np.array(p["r"], float); mu = p["mu"]; K = np.zeros((len(k), 6, 6)); K[:, :3, :3] = D
    for a in range(3):
        K[:, a, a] += r[a]; K[:, a, a + 3] = K[:, a + 3, a] = -r[a] / np.sqrt(mu); K[:, a + 3, a + 3] = r[a] / mu
    return np.sqrt(np.clip(np.linalg.eigvalsh(K), 0, None))


ax = [np.linspace(LO[i], HI[i], (16 if i < 2 else 8) + 1) for i in range(3)]
G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
bnd = np.isclose(G, LO) | np.isclose(G, HI)
nb = bnd.sum(1)


def ext(p, b, sgn, mask, mode):
    idx = np.where(mask)[0]; w = bands(G[idx], p)[:, b]; j = idx[np.argmax(sgn * w)]; k0 = G[j].copy(); best = sgn * np.max(sgn * w)
    fixed = [i for i in range(3) if bnd[j, i]]
    free = {"path": [i for i in range(3) if i not in fixed] if len(fixed) == 2 else [],
            "faces": [i for i in range(3) if i != fixed[0]] if fixed else [], "vol": [0, 1, 2]}[mode]
    if free:
        def f(x):
            k = k0.copy(); k[free] = x; return -sgn * bands(k, p)[0, b]
        r = minimize(f, k0[free], method="L-BFGS-B", bounds=[(LO[i], HI[i]) for i in free])
        if -r.fun > sgn * best:
            best = sgn * -r.fun; k0[free] = r.x
    return best, k0


rng = np.random.default_rng(8); where = collections.Counter(); errs = []
for it in range(250):
    p = dict(k=list(rng.uniform(0.05, 2, 3)), r=list(rng.uniform(0.5, 10, 3)), mu=float(rng.uniform(0.1, 4)),
             s=float(rng.uniform(0, 1)), g=list(rng.uniform(0, 2.5, 3)))
    R = {}
    for mode, mask in (("path", nb >= 2), ("faces", nb >= 1), ("vol", nb >= 0)):
        R[mode] = (ext(p, 2, 1, mask, mode), ext(p, 3, -1, mask, mode))
    for a, b in (("faces", "path"), ("vol", "faces")):
        lo = max(R[a][0], R[b][0], key=lambda t: t[0]); hi = min(R[a][1], R[b][1], key=lambda t: t[0]); R[a] = (lo, hi)
    gv = R["vol"][1][0] - R["vol"][0][0]
    if gv <= 1e-3:
        continue
    k = R["vol"][0][1]; on = sum(min(abs(k[i] - LO[i]), abs(k[i] - HI[i])) < 1e-4 for i in range(3))
    where[{0: "volume interior", 1: "face", 2: "edge", 3: "corner"}[on]] += 1
    errs.append(((R["faces"][1][0] - R["faces"][0][0]) - gv) / gv)
e = np.array(errs)
print(dict(where))
print(f"{len(e)} cells with a gap; faces overestimate the gap by >0.1 %: {np.sum(e > 1e-3)}, >1 %: {np.sum(e > 1e-2)}, max {100*e.max():.1f} %")
