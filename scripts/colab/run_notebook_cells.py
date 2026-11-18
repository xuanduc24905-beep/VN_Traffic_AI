#!/usr/bin/env python3
"""Chay cac o code cua notebook Colab tren may local (bo o 1 cai pip). Dung cho resume_frcnn_local.sh."""
import json
import sys

nb = json.load(open(sys.argv[1]))
g = {"__name__": "__main__"}
for i, src in enumerate([c["source"] for c in nb["cells"] if c["cell_type"] == "code"][1:], 2):
    print(f"===== O {i}", flush=True)
    exec(compile(src, f"cell{i}", "exec"), g)
