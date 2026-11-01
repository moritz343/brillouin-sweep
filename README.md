# FEM sweep plan: path vs faces vs volume in 3D unit cells

## Research question

For 3D phononic and locally resonant unit cells, how often and by how much does
the standard band diagram (k on the edges of the irreducible Brillouin zone,
IBZ) misjudge complete band gaps? How much of that error do the IBZ faces
recover, and how often is a full volume search needed?

This is the 3D counterpart of Maurin, Claeys, Deckers & Desmet,
*Int. J. Solids Struct.* 135 (2018), which did it for the 17 plane groups in 2D.
Read their paper first and reuse their metrics where possible, so the results
compare directly.

## 1. Lattices (phase 1: box-shaped zones only)

The IBZ follows from the point-group symmetry of the **whole unit cell**
(matrix + inclusion), plus time reversal, which adds inversion (the Laue class).
Pick the IBZ from the actual cell, not from the lattice alone.

| Laue class | Example cell | IBZ | Edges = path |
|---|---|---|---|
| m-3m (cubic) | sphere in a cube | tetrahedron Γ X M R | Γ–X–M–Γ–R–X\|M–R |
| 4/mmm (tetragonal) | spheroid along z, or a = b ≠ c | prism Γ X M Z R A | Γ–X–M–Γ–Z–R–A–Z\|X–R\|M–A |
| mmm (orthorhombic) | triaxial ellipsoid along the axes | box | Γ–X–S–Y–Γ–Z–U–R–T–Z\|Y–T\|U–X\|S–R |

Phase 2 (optional): FCC/BCC cells and lower symmetry (tilted ellipsoids:
monoclinic, triclinic). Those need non-box zones, which kpoints.py does not do yet.

## 2. Geometry families

Sample the parameters randomly (Latin hypercube) within manufacturable limits.

**A. Inclusion / porous phononic crystal.** Matrix with one ellipsoidal
inclusion or void. Parameters: semi-axes (filling fraction 5–50 %), cell aspect
ratios, material pair (e.g. steel in epoxy, voids in aluminium). These are
Bragg-type gaps.

**B. Locally resonant, coated core.** Heavy core (lead or steel), soft coating
(silicone), stiff matrix (epoxy). Make the core or the coating ellipsoidal so
the three translational resonances differ (as in Ma et al.,
*Nat. Commun.* 2016). Parameters: core semi-axes, coating thickness per axis,
cell size. These are the gaps this project's page models with springs.

Suggested size: 2 families × 3 Laue classes × 50–100 geometries.

## 3. FEM model

### Open-source route (no COMSOL): `voxel_fem.py`

The cell is a regular grid of voxels, each assigned a material from simple
geometry (ellipsoids painted over a background, see `cells/*.json`). Opposite
faces share nodes, and the Bloch phase is applied where elements wrap across
the cell boundary, so the cell is periodic by construction and needs no special
meshing. Elements are 8-node hexahedra with lumped mass. The lowest modes come
from shift-invert Lanczos, using PARDISO (MKL, via `pypardiso`) when installed
and SciPy's sparse LU otherwise. Output is the results CSV `analyze.py` reads,
including the x/y/z motion shares of each mode.

Set-up, once:

    cd ~/brillouin-sweep
    python3 -m venv .venv
    .venv/bin/pip install numpy scipy pypardiso

Run one geometry:

    python3 kpoints.py --lattice orthorhombic --n 8 --inside-stride 2 -o k.csv
    MKL_NUM_THREADS=2 .venv/bin/python voxel_fem.py cells/coated_core_ortho.json k.csv -o results.csv --nev 12 --jobs 4
    python3 analyze.py k.csv results.csv --min-rel 0.01

Checks done so far:
- Uniform epoxy cell: the lowest shear and longitudinal waves match the exact
  speeds within 0.07 % (6³ voxels) and 0.01–0.09 % (14³), converging with
  refinement. Motion shares come out right (a longitudinal wave along x is
  100 % x-motion).
- Liu-type cell (`cells/liu_cubic.json`, 16³ voxels): a flat, triply
  degenerate resonance band at 344.5 Hz at Γ, X and R. Compare with the
  published band structure before relying on it.
- The two eigen-solver back-ends agree to all printed digits.

Known limits:
- Curved boundaries are staircased. **Thin layers need at least 2–3 voxels
  across.** At 10³ voxels, the 1.6 mm coating in `coated_core_ortho.json` is
  only one voxel thick, the core touches the epoxy, and the low resonance
  disappears. Use ≥ 16³ for coated cores, and confirm with a convergence run.
- 8-node hexahedra are slightly too stiff (frequencies converge from below
  here, error ∝ h²). Nearly incompressible materials (silicone, ν ≈ 0.47) are
  the main risk. Check convergence on the bands you report.
- Speed on this laptop (8 cores, ~2 GB free RAM): about 1.5 s per k-point at
  3,000 unknowns (10³ voxels, 4 workers), and 30–70 s per k-point at 12,000
  unknowns (16³, PARDISO). A 16³ cell with ~400 k-points is about 5 hours here.

### Commercial alternative (COMSOL, Abaqus, Ansys)

1. Solid Mechanics, 3D unit cell, linear elastic isotropic materials.
2. **Periodic Condition → Floquet periodicity** on each pair of opposite faces,
   with the wavevector as parameters `kx, ky, kz` (rad/m).
3. **Mesh: Copy Face** from each source face to its destination face, so the
   periodic meshes are identical. Without this the Floquet condition is
   inaccurate.
4. **Eigenfrequency study:** enough modes to cover the bands of interest
   (e.g. 20–30), searching around a small positive frequency so the three
   zero-frequency modes at Γ don't cause trouble.
5. **Parametric sweep** over `kx, ky, kz` using the list from kpoints.py
   ("specified combinations", loaded from file). Check the exact menu names in
   your COMSOL version. To script it from Python instead, the `mph` package
   drives COMSOL.
6. **Per mode, export:** eigenfrequency, plus the polarisation shares
   `px = ∫ρ|u|² dV / ∫ρ(|u|²+|v|²+|w|²) dV` and likewise `py`, `pz`
   (u, v, w = displacement components; use abs() for the complex fields).
7. Write the results as CSV with columns `case,id,band,freq,px,py,pz`.
   `id` is the kpoints.py row id, so keep it with each sweep step. Sort bands
   by frequency at each k, starting at 1.

**Checks before any production run**
- Reproduce a published 3D band structure (for example the locally resonant
  sonic crystal of Liu et al., *Science* 2000) to validate the model.
- Mesh convergence: the frequencies of the bands of interest change < 0.5 %
  on refinement, at Γ and at one generic k.
- Time one solve. This sets the whole budget below.

## 4. k-point sampling

`kpoints.py` makes one nested grid. Every point is tagged `edge` (on the
standard path), `face` or `inside`, so a single sweep gives all three levels
at the same k-spacing. `--inside-stride 2` keeps every second interior point
per axis. In the test case it still caught an interior band edge (path error
3.1 % vs 3.2 % with the full grid); stride 4 did not (2.5 %).

Solves per geometry (n = grid steps per π/a):

| Lattice | n | Path | Faces (incl. path) | Volume, full | Volume, stride 2 |
|---|---|---|---|---|---|
| cubic | 8 | 46 | 130 | 165 | 131 |
| cubic | 12 | 70 | 290 | 455 | 300 |
| cubic | 16 | 94 | 514 | 969 | 549 |
| tetragonal | 8 | 69 | 258 | 405 | 267 |
| tetragonal | 12 | 105 | 578 | 1183 | 628 |
| orthorhombic | 8 | 92 | 386 | 729 | 413 |
| orthorhombic | 12 | 140 | 866 | 2197 | 991 |
| orthorhombic | 16 | 188 | 1538 | 4913 | 1881 |

**Budget example** with `voxel_fem.py` at 16³ voxels (~45 s per k-point,
measured on this laptop): 300 geometries × ~400 k-points × 45 s ≈ 1,500
CPU-hours. That is about 2 months on this laptop, so use it for the pilot only.
For production, rent a many-core cloud machine or use a university cluster for
a few days. With open-source code there are no licence limits on parallel runs.
The pilot decides n; if n = 6 is accurate enough, the cost roughly halves.

## 5. Analysis

`analyze.py` reports, for every case and every neighbouring band pair:

- gap on the path, on the faces and in the volume, and the relative errors;
- where the true band edges sit (edge / face / inside);
- a verdict: `path exact`, `faces needed`, `volume needed`, or
  `false gap on path` (a gap on the path that is not complete).

Report these as proportions per Laue class and family, with confidence
intervals (e.g. Wilson intervals), plus the distribution of the width error,
as Maurin et al. did in 2D. Use `--min-rel` to ignore negligible gaps.

## 6. Phases

1. **Set-up (1–2 weeks):** build the model, validate it against literature,
   check mesh convergence, time a solve.
2. **Pilot (2–4 weeks):** about 5 geometries per class on the full volume grid
   at n = 16, as ground truth. Decide the production n and stride (target:
   gap error < 0.5 % against the pilot truth).
3. **Production:** random geometries, the chosen grid, run unattended.
4. **Analysis and figures:** the statistics, plus face-net surface plots of
   the interesting cases (the web page can be adapted to read results CSV).

## Files

- `kpoints.py`: nested k-point grid with edge/face/inside tags.
- `analyze.py`: path vs faces vs volume gap statistics.
- `voxel_fem.py`: open-source Floquet–Bloch FEM solver for voxel unit cells.
- `cells/`: example cells (`liu_cubic.json`, `coated_core_ortho.json`).
- `mock_fem.py`: spring-lattice stand-in for the FEM solver, to test the analysis quickly.

Quick test:

    python3 kpoints.py --lattice orthorhombic --n 16 -o k.csv
    python3 mock_fem.py k.csv > results.csv
    python3 analyze.py k.csv results.csv

Expected: the `axial` case is `path exact`. The `extra_springs` case is
`volume needed`, with the path off by +3.2 % and the faces by +0.9 %.
