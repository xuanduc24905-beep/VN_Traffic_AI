#!/usr/bin/env python3
"""Inventory: list checkpoints, dataset YAMLs, runs/ folders with size+mtime+sha256."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/xuand/vn_traffic_ai")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finfo(p: Path, do_hash: bool = True):
    st = p.stat()
    return {
        "path": str(p.relative_to(ROOT)),
        "size": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        "sha256": sha256(p) if do_hash else None,
    }


def main():
    out = {"checkpoints": [], "yamls": [], "runs_detect": [], "run_args": [], "run_results": []}

    # Checkpoints
    cp_dirs = [ROOT / "weights", ROOT, ROOT / "weights/trained", ROOT / "weights/pretrained", ROOT / "weights/trained/frcnn"]
    seen = set()
    for d in cp_dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.pt")):
            if p in seen:
                continue
            seen.add(p)
            out["checkpoints"].append(finfo(p))
        for p in sorted(d.glob("*.pth")):
            if p in seen:
                continue
            seen.add(p)
            out["checkpoints"].append(finfo(p))

    # Also recursive under runs/detect for best.pt / last.pt
    for p in sorted((ROOT / "runs/detect").rglob("best.pt")):
        if p not in seen:
            seen.add(p)
            out["checkpoints"].append(finfo(p, do_hash=True))
    for p in sorted((ROOT / "results").rglob("best.pt")):
        if p not in seen:
            seen.add(p)
            out["checkpoints"].append(finfo(p, do_hash=True))

    # YAMLs under data/
    for p in sorted((ROOT / "data").glob("*.yaml")):
        out["yamls"].append(finfo(p, do_hash=False))

    # Run dirs: just the leaf names
    rd = ROOT / "runs/detect"
    if rd.exists():
        for d in sorted(rd.iterdir()):
            if d.is_dir():
                entry = {"name": d.name, "has_args": (d / "args.yaml").exists(),
                         "has_results": (d / "results.csv").exists(),
                         "has_best": (d / "weights/best.pt").exists()}
                out["runs_detect"].append(entry)

    # Snapshot args.yaml filenames + results.csv filenames for runs of interest
    of_interest = [
        "r_clspw", "r_p2", "r_progressive_s1", "r_progressive",
        "p4_scale", "p4_imgsz1280_from_baseline", "train_v8s_v5_honest",
        "train_v8s_v5plus_960", "train_v8s_v5plus_warm",
    ]
    for d in (ROOT / "runs/detect").iterdir() if (ROOT / "runs/detect").exists() else []:
        if not d.is_dir():
            continue
        # any args.yaml below? check nested
        for ap in d.rglob("args.yaml"):
            out["run_args"].append(str(ap.relative_to(ROOT)))
        for rp in d.rglob("results.csv"):
            out["run_results"].append(str(rp.relative_to(ROOT)))
    # Same for results/ where overnight runs may live
    for ap in (ROOT / "results").rglob("args.yaml"):
        out["run_args"].append(str(ap.relative_to(ROOT)))
    for rp in (ROOT / "results").rglob("results.csv"):
        out["run_results"].append(str(rp.relative_to(ROOT)))

    Path("outputs/verify/inventory.json").write_text(json.dumps(out, indent=2))
    print(f"CKPTs: {len(out['checkpoints'])}  YAMLs: {len(out['yamls'])}  runs: {len(out['runs_detect'])}  args: {len(out['run_args'])}  results.csv: {len(out['run_results'])}")


if __name__ == "__main__":
    main()
