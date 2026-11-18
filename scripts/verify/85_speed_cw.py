#!/usr/bin/env python3
"""Do lai toc do bang dung diem luu YOLOv8s-CW (r_clspw_v2), cung quy trinh 50_speed_benchmark.py
(batch 1, 20 anh khoi dong, 200 anh do, anh ngau nhien hat giong 42 tu V5+ test).
Faster R-CNN do lai trong cung phien de so sanh; quy trinh pipeline giong 50_speed_benchmark.py nhung dung r_clspw_v2."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/xuand/vn_traffic_ai")
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("sb", ROOT / "scripts/verify/50_speed_benchmark.py")
sb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sb)

CW = "runs/detect/r_clspw_v2/weights/best.pt"
gpu_busy = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
res = {"gpu_processes_before": gpu_busy,
       "cw_fp16_960": sb.bench_yolo_model_only(CW, 960, True),
       "cw_fp32_960": sb.bench_yolo_model_only(CW, 960, False),
       "frcnn_fp32": sb.bench_frcnn_model_only()}
# pipeline: thay diem luu trong ham goc bang r_clspw_v2
src = Path(spec.origin).read_text()
assert 'weights = "runs/detect/r_clspw/weights/best.pt"' in src
ns = {}
code = src.split("def bench_pipeline_end_to_end")[1].split("\ndef main")[0]
exec("import json, time\nfrom pathlib import Path\nROOT = Path('/home/xuand/vn_traffic_ai')\ndef bench_pipeline_end_to_end"
     + code.replace('runs/detect/r_clspw/weights/best.pt', CW), ns)
ns["bench_pipeline_end_to_end"](write_video=False)  # khoi dong
res["pipeline_no_write"] = ns["bench_pipeline_end_to_end"](write_video=False)
res["pipeline_with_write"] = ns["bench_pipeline_end_to_end"](write_video=True)
res["frcnn_over_cw_fp32"] = res["frcnn_fp32"]["mean_ms"] / res["cw_fp32_960"]["mean_ms"]
out = ROOT / "outputs/verify/speed_cw.json"
out.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
