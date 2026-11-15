#!/usr/bin/env python3
"""BƯỚC 2 — Training config & results.

For each run of interest, read args.yaml and results.csv. Produce a JSON that
maps each run to:
- actual args (imgsz, batch, epochs, lr0, lrf, cls_pw, cos_lr, mosaic, mixup,
  scale, translate, close_mosaic, patience, model, data)
- training dynamics (n_epochs logged, best epoch by fitness, best mAP50 /
  mAP50-95, total train time inferred from results.csv 'time' column, final
  loss values).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

ROOT = Path("/home/xuand/vn_traffic_ai")

RUNS = {
    "V5_honest":     "runs/detect/train_v8s_v5_honest",
    "V5+_baseline":  "runs/detect/train_v8s_v5plus_960",
    "p4_scale":      "runs/detect/p4_scale_from_baseline",   # warm-start = V5+ best → map to report "p4_scale"
    "p4_scale_cold": "runs/detect/p4_scale",
    "p4_imgsz1280":  "runs/detect/p4_imgsz1280_from_baseline",
    "r_p2":          "runs/detect/r_p2",
    "r_clspw":       "runs/detect/r_clspw",
    "r_progressive_s1": "runs/detect/r_progressive_s1",
    "r_progressive_s2": "runs/detect/r_progressive",
}

KEYS = [
    "model", "data", "imgsz", "batch", "epochs", "patience", "optimizer",
    "lr0", "lrf", "cos_lr", "momentum", "weight_decay",
    "warmup_epochs", "box", "cls", "dfl",
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste",
    "close_mosaic", "amp",
    # YOLOv8 specific class positive weight
    "cls_pw", "obj_pw", "label_smoothing",
]


def read_results_csv(path: Path) -> dict:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v.strip() for k, v in row.items()})
    if not rows:
        return {"n_epochs": 0}
    # heuristics
    def nf(x):
        try:
            return float(x)
        except Exception:
            return None
    # The "fitness" in Ultralytics YOLOv8 = 0.1*mAP50 + 0.9*mAP50-95
    map50_key = next((k for k in rows[0] if "mAP50" in k and "95" not in k and ("B" in k or "val" in k.lower())), None)
    map5095_key = next((k for k in rows[0] if "mAP50-95" in k), None)
    if not map50_key:
        # try any
        for k in rows[0]:
            if "mAP_0.5" in k or "mAP50" in k:
                map50_key = k
                break
    time_key = next((k for k in rows[0] if k.lower() == "time"), None)

    best_idx = 0
    best_fit = -1
    for i, r in enumerate(rows):
        m50 = nf(r.get(map50_key or "", ""))
        m95 = nf(r.get(map5095_key or "", ""))
        if m50 is None:
            continue
        fit = 0.1 * m50 + 0.9 * (m95 if m95 is not None else 0.0)
        if fit > best_fit:
            best_fit = fit
            best_idx = i
    last = rows[-1]
    best = rows[best_idx]

    total_time_sec = None
    if time_key:
        try:
            total_time_sec = float(last[time_key]) - float(rows[0][time_key])
            # Ultralytics "time" column is cumulative seconds since start of training.
            total_time_sec = float(last[time_key])  # cumulative since epoch 0
        except Exception:
            pass

    return {
        "n_epochs_logged": len(rows),
        "columns": list(rows[0].keys()),
        "best_epoch_idx": best_idx,
        "best_epoch_row": best,
        "last_row": last,
        "map50_key": map50_key,
        "map5095_key": map5095_key,
        "time_key": time_key,
        "total_time_sec": total_time_sec,
        "best_fitness": best_fit,
    }


def main():
    out = {}
    for label, rel in RUNS.items():
        d = ROOT / rel
        entry = {"path": rel, "exists": d.exists()}
        args_p = d / "args.yaml"
        res_p = d / "results.csv"
        if args_p.exists():
            a = yaml.safe_load(args_p.read_text())
            entry["args"] = {k: a.get(k) for k in KEYS if k in a}
            entry["args_all_keys"] = sorted(a.keys())
        else:
            entry["args"] = None
        if res_p.exists():
            entry["results"] = read_results_csv(res_p)
        else:
            entry["results"] = None
        out[label] = entry

    # FRCNN results (search in results/ or similar)
    frcnn_logs = list((ROOT / "results").rglob("frcnn*.json")) + list((ROOT / "logs").rglob("frcnn*.json"))
    out["_frcnn_log_candidates"] = [str(p.relative_to(ROOT)) for p in frcnn_logs]

    (ROOT / "outputs/verify/training_config.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote outputs/verify/training_config.json")

    # Human summary
    lines = []
    def e(s=""):
        lines.append(s)
        print(s)
    for label, entry in out.items():
        if label.startswith("_"):
            continue
        e(f"\n--- {label} ({entry['path']}) ---")
        if not entry["exists"]:
            e("  NOT FOUND")
            continue
        a = entry["args"] or {}
        e(f"  model={a.get('model')} data={a.get('data')} imgsz={a.get('imgsz')} batch={a.get('batch')}")
        e(f"  epochs(max)={a.get('epochs')} patience={a.get('patience')} optimizer={a.get('optimizer')}")
        e(f"  lr0={a.get('lr0')} lrf={a.get('lrf')} cos_lr={a.get('cos_lr')} close_mosaic={a.get('close_mosaic')}")
        e(f"  scale={a.get('scale')} translate={a.get('translate')} mosaic={a.get('mosaic')} mixup={a.get('mixup')} copy_paste={a.get('copy_paste')}")
        e(f"  cls_pw={a.get('cls_pw')} obj_pw={a.get('obj_pw')}  box={a.get('box')} cls={a.get('cls')} dfl={a.get('dfl')}")
        r = entry["results"]
        if r:
            total_sec = r.get("total_time_sec") or 0
            m = int(total_sec // 60)
            s = int(total_sec % 60)
            e(f"  results.csv: {r['n_epochs_logged']} rows, best@epoch_idx={r['best_epoch_idx']} fitness={r['best_fitness']:.4f}  total_time≈{m}m{s}s")
            e(f"    best row keys: mAP50={r['map50_key']} mAP50-95={r['map5095_key']}")
            best = r["best_epoch_row"]
            if r['map50_key']:
                e(f"    best mAP50={best.get(r['map50_key'])}  best mAP50-95={best.get(r['map5095_key'])}")

    (ROOT / "outputs/verify/training_config_summary.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
