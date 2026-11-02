"""Symmetry-broken cell (only (1,1,0)): point group 2/m with unique axis z.
Mirror faces of the IBZ box [-π,π]x[0,π]x[0,π]: k_z = 0 and k_z = π. Non-mirror faces: k_y = 0, π and k_x = ±π.
Where do face-interior and volume-interior maxima of band 3 sit?"""
import sys, collections
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
import symmetry_pair as Q

rng = np.random.default_rng(21)
where = collections.Counter(); face_err = []
for it in range(400):
    p = dict(k=list(rng.uniform(0.05, 2, 3)), r=list(rng.uniform(0.02, 10, 3)), mu=float(rng.uniform(0.1, 5)),
             s=float(rng.uniform(0, 1)), g=float(rng.uniform(0, 2.5)))
    A = Q.gap(p, False); gv = A["vol"][0]
    if gv <= 1e-3:
        continue
    k = A["vol"][1][1]; lo, hi = Q.box(False)
    on = [i for i in range(3) if min(abs(k[i] - lo[i]), abs(k[i] - hi[i])) < 1e-4]
    tag = {0: "volume interior", 1: None, 2: "edge", 3: "corner"}[len(on)]
    if len(on) == 1:
        tag = "mirror face k_z = 0/π" if on[0] == 2 else "non-mirror face (k_" + "xyz"[on[0]] + ")"
    where[tag] += 1
    face_err.append((A["faces"][0] - gv) / gv)
print(dict(where))
fe = np.array(face_err)
print(f"{len(fe)} cells with a gap; faces overestimate the gap by >0.1 %: {np.sum(fe > 1e-3)}, >1 %: {np.sum(fe > 1e-2)}, max {100*fe.max():.2f} %")
