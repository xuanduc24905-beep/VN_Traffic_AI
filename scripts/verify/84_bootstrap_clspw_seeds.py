#!/usr/bin/env python3
"""Bootstrap cho doi chung cls_pw day du: trung binh mAP@0,5 cua 3 hat giong moi dieu kien.

Moi lan lay mau lai anh kiem thu, tinh mAP@0,5 cua ca 6 luot tren cung mau, lay trung binh theo dieu kien,
roi lay hieu so cw05 - cw0. Khoang tin cay chi phan anh viec chon anh kiem thu; dao dong giua hat giong
duoc bao cao rieng bang do lech chuan cua 3 lan chay.
Dung lai du doan da luu boi 80_unified_coco_eval.py.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/xuand/vn_traffic_ai")
D = ROOT / "outputs/verify/unified_eval"
spec = importlib.util.spec_from_file_location("bs", ROOT / "scripts/verify/81_bootstrap_ci.py")
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)

B = 1000
out = {}
rng = np.random.default_rng(0)
for sname in ("v5plus_test", "aerial_test"):
    gt = bs.load_gt(sname)
    n = len(gt)
    runs = {c: [f"{c}_s{s}" for s in range(3)] for c in ("cw0", "cw05")}
    extra = ["r_clspw_v2", "exp_baseline", "frcnn_300"] + (["frcnn_v5plus_300"] if (D / f"pred_{sname}_frcnn_v5plus_300.npz").exists() else [])
    names = runs["cw0"] + runs["cw05"] + extra
    S = {m: bs.match(gt, sname, m) for m in names}
    full = {m: float(bs.mAP(S[m], gt, range(n))[0]) for m in names}
    idx = [rng.integers(0, n, n) for _ in range(B)]
    boot = {m: np.array([bs.mAP(S[m], gt, ii)[0] for ii in idx]) for m in names}
    mean = {c: np.mean([boot[m] for m in runs[c]], axis=0) for c in runs}
    pt = {c: float(np.mean([full[m] for m in runs[c]])) for c in runs}
    sd = {c: float(np.std([full[m] for m in runs[c]], ddof=1)) for c in runs}

    res = {"per_run": full, "mean": pt, "sd_seeds": sd, "diffs": {}}
    d = mean["cw05"] - mean["cw0"]
    res["diffs"]["cw05_mean - cw0_mean"] = {"point": pt["cw05"] - pt["cw0"],
                                           "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    d = mean["cw0"] - boot["exp_baseline"]
    res["diffs"]["cw0_mean - exp_baseline"] = {"point": pt["cw0"] - full["exp_baseline"],
                                               "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    for fr in [x for x in extra if x.startswith("frcnn")]:
        d = mean["cw05"] - boot[fr]
        res["diffs"][f"cw05_mean - {fr}"] = {"point": pt["cw05"] - full[fr],
                                            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    out[sname] = res
    print(sname, json.dumps(res, indent=1), flush=True)
(D / "bootstrap_clspw_seeds.json").write_text(json.dumps(out, indent=1))
