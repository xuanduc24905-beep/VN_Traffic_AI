#!/usr/bin/env python3
"""Ve lai Hinh 3 (ma tran nham lan YOLOv8s-CW tren V5+ test) voi nhan tieng Viet.
So lieu doc truc tiep tu outputs/verify/error_analysis_r_clspw_v2.json (31_error_analysis.py)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/xuand/vn_traffic_ai")
E = json.loads((ROOT / "outputs/verify/error_analysis_r_clspw_v2.json").read_text())
cm = E["confusion_matrix"]
assert cm["rows_pred"] == ["motor", "car", "bus", "truck", "bg"] and cm["cols_gt"] == cm["rows_pred"]
M = np.array(cm["matrix"])
lab = ["Xe máy", "Ô tô", "Xe buýt", "Xe tải", "Nền"]

fig, ax = plt.subplots(figsize=(7.65, 6.7), dpi=200, layout="constrained")
L = np.log1p(M)
im = ax.imshow(L, cmap="Greys", vmin=0, vmax=L.max() * 1.15)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{M[i, j]:,}".replace(",", "."), ha="center", va="center", fontsize=12,
                color="white" if L[i, j] > L.max() * 0.62 else "black")
ax.set_xticks(range(5), lab, fontsize=12)
ax.set_yticks(range(5), lab, fontsize=12)
ax.set_xlabel("Lớp thực tế", fontsize=13)
ax.set_ylabel("Lớp dự đoán", fontsize=13)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("log(1 + số trường hợp)", fontsize=11)

out = ROOT / "outputs/verify/fig_confusion_vi.png"
fig.savefig(out, dpi=200)
print("saved", out)
