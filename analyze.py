#!/usr/bin/env python3
"""Compare band gaps found on the IBZ edges (path), faces and full volume.

Inputs
    kpoints CSV  from kpoints.py (columns id, level, ux, uy, uz, ...)
    results CSV  from the FEM sweep, long format, one row per mode:
                 case, id, band, freq   [, px, py, pz]
                 (band counts from 1 upwards, sorted by frequency at each k;
                  px/py/pz = share of kinetic energy along x/y/z, optional)

For every case and every pair of neighbouring bands (n, n+1) it reports
    gap_path    min f_{n+1} - max f_n over edge points
    gap_faces   same over edge + face points
    gap_volume  same over all points
and where the true band edges sit (edge / face / inside).

Usage:
    python3 analyze.py k.csv results.csv                 # per-gap table
    python3 analyze.py k.csv results.csv --stats         # summary over all cases
    python3 analyze.py k.csv results.csv --min-rel 0.01  # ignore gaps < 1% of mid-gap
"""
import argparse
import csv
from collections import defaultdict

LEVEL_RANK = {"edge": 0, "face": 1, "inside": 2}


def load(kfile, rfile):
    kp = {}
    with open(kfile) as f:
        for r in csv.DictReader(f):
            kp[int(r["id"])] = (r["level"], (float(r["ux"]), float(r["uy"]), float(r["uz"])))
    data = defaultdict(lambda: defaultdict(dict))  # case -> id -> band -> freq
    with open(rfile) as f:
        for r in csv.DictReader(f):
            data[r["case"]][int(r["id"])][int(r["band"])] = float(r["freq"])
    return kp, data


def gaps_for_case(kp, modes, min_rel):
    ids = [i for i in modes if i in kp]
    nb = min(len(modes[i]) for i in ids)
    rows = []
    for n in range(1, nb):
        res = {}
        for name, maxrank in (("path", 0), ("faces", 1), ("volume", 2)):
            sel = [i for i in ids if LEVEL_RANK[kp[i][0]] <= maxrank]
            top = max(sel, key=lambda i: modes[i][n])
            bot = min(sel, key=lambda i: modes[i][n + 1])
            res[name] = (modes[bot][n + 1] - modes[top][n], top, bot)
        gv = res["volume"][0]
        gp = res["path"][0]
        if max(gp, gv) <= 0:
            continue  # no gap at any level
        mid = 0.5 * (modes[res["volume"][1]][n] + modes[res["volume"][2]][n + 1])
        if max(gp, gv) < min_rel * mid:
            continue
        rows.append({
            "pair": f"{n}/{n + 1}",
            "gap_path": gp, "gap_faces": res["faces"][0], "gap_volume": gv,
            "top_at": kp[res["volume"][1]][0], "bottom_at": kp[res["volume"][2]][0],
            "top_k": kp[res["volume"][1]][1], "bottom_k": kp[res["volume"][2]][1],
        })
    return rows


def classify(r):
    gp, gf, gv = r["gap_path"], r["gap_faces"], r["gap_volume"]
    if gv <= 0 and gp > 0:
        return "false gap on path"
    if abs(gp - gv) <= 1e-9 * max(1.0, abs(gv)):
        return "path exact"
    if abs(gf - gv) <= 1e-9 * max(1.0, abs(gv)):
        return "faces needed"
    return "volume needed"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kpoints")
    ap.add_argument("results")
    ap.add_argument("--min-rel", type=float, default=0.0, help="ignore gaps narrower than this fraction of mid-gap")
    ap.add_argument("--stats", action="store_true", help="print only the summary over all cases")
    args = ap.parse_args(argv)

    kp, data = load(args.kpoints, args.results)
    allrows = []
    for case in sorted(data):
        for r in gaps_for_case(kp, data[case], args.min_rel):
            r["case"] = case
            r["class"] = classify(r)
            allrows.append(r)

    if not args.stats:
        fmt = "{:<14} {:>6} {:>11} {:>11} {:>11} {:>8} {:>8}  {:<8} {:<8} {}"
        print(fmt.format("case", "bands", "gap_path", "gap_faces", "gap_volume", "err_path", "err_face", "top_at", "bot_at", "verdict"))
        for r in allrows:
            gv = r["gap_volume"]
            ep = f"{(r['gap_path'] - gv) / abs(gv) * 100:+.1f}%" if gv > 0 else "-"
            ef = f"{(r['gap_faces'] - gv) / abs(gv) * 100:+.1f}%" if gv > 0 else "-"
            print(fmt.format(r["case"][:14], r["pair"], f"{r['gap_path']:.5g}", f"{r['gap_faces']:.5g}",
                             f"{gv:.5g}", ep, ef, r["top_at"], r["bottom_at"], r["class"]))

    if not allrows:
        print("no gaps found")
        return
    tot = len(allrows)
    by = defaultdict(int)
    for r in allrows:
        by[r["class"]] += 1
    real = [r for r in allrows if r["gap_volume"] > 0]
    errs = sorted((r["gap_path"] - r["gap_volume"]) / r["gap_volume"] for r in real)
    print(f"\n{tot} gaps over {len(data)} cases")
    for c in ("path exact", "faces needed", "volume needed", "false gap on path"):
        print(f"  {c:<18} {by[c]:>5}  ({by[c] / tot * 100:.1f}%)")
    if errs:
        print(f"  path width error: median {errs[len(errs) // 2] * 100:+.2f}%, max {errs[-1] * 100:+.2f}%")


if __name__ == "__main__":
    main()
