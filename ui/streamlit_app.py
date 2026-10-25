"""Streamlit UI cho hệ thống đếm phương tiện.

Chạy:
    cd /home/xuand/vn_traffic_ai
    conda activate yolov8_ft
    streamlit run ui/streamlit_app.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st


def transcode_to_h264(src: str, dst: str) -> bool:
    """Convert video sang H.264 để browser (st.video) play inline được.

    Trả về True nếu thành công. Cần ffmpeg trong PATH.
    """
    if shutil.which("ffmpeg") is None:
        return False
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", src, "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "23", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart",
             "-an", dst],
            check=True, capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.run import run_pipeline
from src.tracking.counter import Line
from src.stats.aggregator import (
    events_to_df, summarize, counts_per_bucket, flow_rate, peak_period,
    cumulative_counts, format_bucket_label, counts_by_direction,
)
from src.evaluation.metrics import report as eval_report


def run_baseline_pipeline(video: str, weights: str, line_y: int,
                          out_video: str, out_csv: str,
                          imgsz: int = 640, half: bool = True,
                          progress_cb=None) -> tuple:
    """Wrap baseline/10_traffic_counting.py cho Streamlit.

    Baseline: 1 line ngang, chỉ đếm chiều xuống (last_cy < line_y <= cy),
    không direction, không log per-event.

    Trả về (events_list, totals_dict, class_names_list) khớp schema run_pipeline.
    """
    import cv2
    import csv as _csv
    from collections import defaultdict
    from ultralytics import YOLO

    VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle"}
    model = YOLO(weights)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    counted_ids = set()
    class_counts = defaultdict(int)
    track_history = {}
    events = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, persist=True, conf=0.4, verbose=False,
                              imgsz=imgsz, half=half, device=0)[0]
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)

        if results.boxes.id is not None:
            for box, track_id in zip(results.boxes, results.boxes.id):
                cls_name = model.names[int(box.cls[0])]
                if cls_name not in VEHICLE_CLASSES:
                    continue
                tid = int(track_id)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cy = (y1 + y2) // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls_name} #{tid}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                last_cy = track_history.get(tid)
                if (last_cy is not None
                        and last_cy < line_y <= cy
                        and tid not in counted_ids):
                    counted_ids.add(tid)
                    class_counts[cls_name] += 1
                    events.append({
                        "frame": frame_idx,
                        "time_sec": round(frame_idx / fps, 3),
                        "line": "baseline_line",
                        "direction": "down",
                        "track_id": tid,
                        "class_name": cls_name,
                    })
                track_history[tid] = cy

        y_off = 30
        cv2.putText(frame, f"Tong: {sum(class_counts.values())}",
                    (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        for cn, ct in class_counts.items():
            y_off += 25
            cv2.putText(frame, f"{cn}: {ct}", (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        writer.write(frame)
        frame_idx += 1
        if progress_cb is not None:
            progress_cb(frame_idx, total_frames)

    cap.release(); writer.release()

    if events:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            w_csv = _csv.DictWriter(f, fieldnames=["frame", "time_sec", "line",
                                                    "direction", "track_id", "class_name"])
            w_csv.writeheader(); w_csv.writerows(events)

    # Chuyển dict class_counts -> totals schema {line: {direction: {class_id: count}}}
    class_names_list = list(model.names.values())
    name_to_id = {v: k for k, v in model.names.items()}
    totals = {"baseline_line": {"down": {}}}
    for cn, ct in class_counts.items():
        if cn in name_to_id:
            totals["baseline_line"]["down"][name_to_id[cn]] = ct
    return events, totals, class_names_list


st.set_page_config(page_title="VN Traffic AI", page_icon="", layout="wide")

# ---------- Sidebar ----------
st.sidebar.title(" Cấu hình")

WEIGHTS_DIR = ROOT / "weights"
available_weights = sorted([str(p.relative_to(ROOT)) for p in WEIGHTS_DIR.rglob("*.pt")])
runs_weights = sorted([str(p.relative_to(ROOT))
                       for p in (ROOT / "runs").rglob("best.pt")]) if (ROOT / "runs").exists() else []
all_weights = available_weights + runs_weights

if not all_weights:
    st.sidebar.error("Không tìm thấy file .pt nào trong weights/ hoặc runs/")
    st.stop()

pipeline_choice = st.sidebar.radio(
    "Pipeline",
    ["Improved (nhóm) — nhiều line + direction + log event",
     "Baseline (giáo viên) — 1 line ngang, không direction"],
    index=0,
    help="Chọn logic pipeline áp dụng ở chế độ 'Chạy 1 pipeline'. "
         "Ở chế độ 'So sánh 2 pipeline' thì mỗi bên chọn riêng."
)
pipeline_kind = "improved_pipeline" if pipeline_choice.startswith("Improved") else "baseline_pipeline"

_preferred = ["weights/v3_vn_best.pt", "weights/trained/v3_vn_best.pt"]
_default_idx = next((i for i, w in enumerate(all_weights) if w in _preferred), 0)
weights_path = st.sidebar.selectbox("Model weights", all_weights, index=_default_idx,
                                    help="v3_vn_best.pt là model tốt nhất hiện tại (mAP50=0.951 val VN).")
tracker = st.sidebar.radio("Tracker", ["bytetrack.yaml", "botsort.yaml"],
                           help="Chỉ áp dụng cho Improved pipeline. Baseline pipeline dùng ByteTrack mặc định.")
conf = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.3, 0.05)
iou = st.sidebar.slider("IoU threshold (NMS)", 0.1, 0.9, 0.5, 0.05)
imgsz = st.sidebar.selectbox("Image size", [320, 416, 512, 640, 768], index=3,
                             help="Nhỏ hơn = nhanh hơn, nhưng miss xe nhỏ. 416 thường đủ cho video 1080p.")
use_half = st.sidebar.checkbox("FP16 inference (nhanh 1.3-1.5×)", value=True,
                               help="GPU Ada/Ampere/Turing hỗ trợ tốt. Tắt nếu chạy CPU hoặc GPU cũ.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Line đếm**")
line_y_pct = st.sidebar.slider("Vị trí line (% chiều cao)", 10, 90, 50, 5)
line_direction = st.sidebar.radio("Hướng line", ["horizontal", "vertical"])
count_direction = st.sidebar.radio(
    "Chiều đếm", ["both", "ltr", "rtl"],
    help="ltr = tráiphải theo chiều vector line (p1p2); rtl = ngược lại. "
         "'both' đếm cả hai chiều nhưng vẫn phân biệt trong output.")

# ---------- Main ----------
st.title(" VN Traffic AI — Đếm và phân loại phương tiện")
st.caption("Detection + Tracking (ByteTrack/BoT-SORT) + Counting theo line")

tab_run, tab_stats, tab_eval = st.tabs(["▶ Chạy pipeline", " Thống kê", " Đánh giá"])

# ---------- Tab 1: Run ----------
with tab_run:
    uploaded = st.file_uploader("Upload video giao thông (mp4/avi/mov)",
                                type=["mp4", "avi", "mov"])

    col_a, col_b = st.columns([1, 1])
    with col_a:
        preview_video = st.empty()
    with col_b:
        info_placeholder = st.empty()

    if uploaded:
        # Save uploaded to temp
        tmp_dir = Path(tempfile.gettempdir()) / "vn_traffic_ai_uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_video = tmp_dir / uploaded.name
        tmp_video.write_bytes(uploaded.read())
        st.session_state["tmp_video"] = str(tmp_video)
        preview_video.video(str(tmp_video))

        # get video dimensions
        import cv2
        cap = cv2.VideoCapture(str(tmp_video))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        info_placeholder.info(
            f"**{w}×{h}** @ **{fps:.1f}** fps · **{n_frames}** frames · "
            f"**{n_frames/fps:.1f}s**"
        )
        st.session_state["video_meta"] = dict(w=w, h=h, fps=fps, n_frames=n_frames)

    st.markdown("---")
    mode = st.radio(
        "Chế độ",
        ["▶ Chạy 1 pipeline", "⚖ So sánh 2 pipeline (Baseline vs Improved)"],
        horizontal=True,
    )

    if mode.startswith("⚖"):
        # ---------- Compare mode ----------
        st.markdown("### Chọn 2 cấu hình để so sánh (pipeline + weights)")
        _bl_default = next((i for i, w in enumerate(all_weights)
                            if w.endswith("yolov8s.pt")
                            or w.endswith("pretrained/yolov8s.pt")), 0)
        _im_default = next((i for i, w in enumerate(all_weights)
                            if "v3_vn" in w and w.endswith(".pt")),
                           next((i for i, w in enumerate(all_weights)
                                 if "v2_fix" in w), 0))
        PIPELINE_OPTIONS = [
            "Baseline (giáo viên) — 1 line, không direction, không log event",
            "Improved (nhóm) — nhiều line + direction + log event + stats",
        ]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Side A — Baseline**")
            bl_pipeline = st.selectbox("Pipeline A", PIPELINE_OPTIONS, index=0, key="pip_a")
            bl_weights = st.selectbox("Weights A", all_weights, index=_bl_default, key="w_a",
                                      help="Mặc định yolov8s.pt COCO gốc.")
        with c2:
            st.markdown("**Side B — Improved**")
            im_pipeline = st.selectbox("Pipeline B", PIPELINE_OPTIONS, index=1, key="pip_b")
            im_weights = st.selectbox("Weights B", all_weights, index=_im_default, key="w_b",
                                      help="Mặc định best.pt fine-tune VN.")

        _kind_bl = "baseline_pipeline" if bl_pipeline.startswith("Baseline") else "improved_pipeline"
        _kind_im = "baseline_pipeline" if im_pipeline.startswith("Baseline") else "improved_pipeline"
        st.caption(f"➡ Side A: **{_kind_bl}** + `{bl_weights}`  |  "
                   f"Side B: **{_kind_im}** + `{im_weights}`")

        run_compare = st.button("⚖ Chạy so sánh 2 pipeline",
                                type="primary",
                                disabled="tmp_video" not in st.session_state)

        if run_compare:
            import time as _time
            meta = st.session_state["video_meta"]
            w, h = meta["w"], meta["h"]

            if line_direction == "horizontal":
                y = int(h * line_y_pct / 100)
                line = Line(name="line_1", p1=(50, y), p2=(w - 50, y),
                            count_direction=count_direction)
            else:
                xln = int(w * line_y_pct / 100)
                line = Line(name="line_1", p1=(xln, 50), p2=(xln, h - 50),
                            count_direction=count_direction)

            out_dir = ROOT / "results"
            stem = Path(st.session_state["tmp_video"]).stem
            paths = {}
            for tag, wpath in [("baseline", bl_weights), ("improved", im_weights)]:
                paths[tag] = {
                    "video": out_dir / "videos" / f"cmp_{tag}_{stem}.mp4",
                    "csv":   out_dir / "tables" / f"cmp_{tag}_{stem}.csv",
                }

            results = {}
            side_configs = [
                ("baseline", bl_weights, _kind_bl, bl_pipeline),
                ("improved", im_weights, _kind_im, im_pipeline),
            ]
            for tag, wpath, kind, label in side_configs:
                st.info(f"Đang chạy **{label.split('—')[0].strip()}** + `{wpath}`...")
                progress = st.progress(0, text=f"{tag} — frame 0")

                def _cb(cur, total, _tag=tag, _p=progress):
                    if total > 0:
                        _p.progress(min(cur/total, 1.0),
                                    text=f"{_tag} — frame {cur}/{total}")

                t0 = _time.time()
                if kind == "baseline_pipeline":
                    # Convert line_y_pct → absolute pixel Y
                    if line_direction == "horizontal":
                        line_y_abs = int(h * line_y_pct / 100)
                    else:
                        line_y_abs = int(h * 0.5)  # baseline chỉ hỗ trợ line ngang
                    events, totals, class_names = run_baseline_pipeline(
                        video=st.session_state["tmp_video"],
                        weights=str(ROOT / wpath),
                        line_y=line_y_abs,
                        out_video=str(paths[tag]["video"]),
                        out_csv=str(paths[tag]["csv"]),
                        imgsz=imgsz, half=use_half,
                        progress_cb=_cb,
                    )
                else:
                    events, totals, class_names = run_pipeline(
                        video=st.session_state["tmp_video"],
                        weights=str(ROOT / wpath),
                        lines=[line],
                        tracker=tracker,
                        conf=conf, iou=iou, imgsz=imgsz, half=use_half,
                        out_csv=str(paths[tag]["csv"]),
                        out_video=str(paths[tag]["video"]),
                        progress_cb=_cb,
                    )
                dt = _time.time() - t0
                progress.empty()

                h264_p = paths[tag]["video"].with_name(paths[tag]["video"].stem + "_h264.mp4")
                if transcode_to_h264(str(paths[tag]["video"]), str(h264_p)):
                    play_path = str(h264_p)
                else:
                    play_path = str(paths[tag]["video"])

                results[tag] = {
                    "weights": wpath,
                    "pipeline_kind": kind,
                    "pipeline_label": label.split("—")[0].strip(),
                    "events": events,
                    "totals": totals,
                    "class_names": class_names,
                    "runtime": dt,
                    "fps": meta["n_frames"] / dt if dt > 0 else 0,
                    "play_path": play_path,
                    "raw_video": str(paths[tag]["video"]),
                    "csv": str(paths[tag]["csv"]),
                }

            st.session_state["compare_results"] = results
            # Đưa Improved làm nguồn cho Tab Stats + Eval (mặc định)
            st.session_state["events_csv"] = results["improved"]["csv"]
            st.session_state["totals"] = results["improved"]["totals"]
            st.session_state["class_names"] = results["improved"]["class_names"]
            st.session_state["stats_source_tag"] = "improved"
            st.success("Xong! Kéo xuống xem so sánh. "
                       "Tab Thống kê & Đánh giá đang lấy dữ liệu từ Improved.")

        # ---------- Display compare results ----------
        if "compare_results" in st.session_state:
            results = st.session_state["compare_results"]
            bl_r, im_r = results["baseline"], results["improved"]

            st.markdown("## Kết quả so sánh side-by-side")
            c1, c2 = st.columns(2)
            for col, tag, r in [(c1, "Side A", bl_r), (c2, "Side B", im_r)]:
                with col:
                    st.markdown(f"### ▶ {tag}: {r['pipeline_label']}")
                    st.caption(f"Pipeline: `{r['pipeline_kind']}`  |  Weights: `{r['weights']}`")
                    st.video(r["play_path"])

                    def _totals_to_df(totals, class_names):
                        # totals: {line: {direction: {class_id: count}}}
                        rows = []
                        for ln, dirs in totals.items():
                            for direction, per_cls in dirs.items():
                                for cid, n in per_cls.items():
                                    rows.append({
                                        "line": ln, "direction": direction,
                                        "class": class_names[int(cid)] if class_names else str(cid),
                                        "count": n,
                                    })
                        return pd.DataFrame(rows)

                    df_tot = _totals_to_df(r["totals"], r["class_names"])
                    if df_tot.empty:
                        st.warning("Không đếm được lượt nào qua line.")
                    else:
                        st.dataframe(df_tot.groupby("class")["count"].sum().reset_index(),
                                     use_container_width=True, hide_index=True)

                    st.metric("Tổng đếm", int(sum(df_tot["count"]) if not df_tot.empty else 0))
                    st.metric("Runtime", f"{r['runtime']:.2f}s")
                    st.metric("FPS", f"{r['fps']:.1f}")

            # ---------- Summary + chart ----------
            st.markdown("## So sánh chỉ số chính")
            total_bl = sum(sum(c.values()) for d in bl_r["totals"].values() for c in d.values())
            total_im = sum(sum(c.values()) for d in im_r["totals"].values() for c in d.values())
            speedup = im_r["fps"] / bl_r["fps"] if bl_r["fps"] > 0 else 0
            diff_count = total_im - total_bl
            df_sum = pd.DataFrame([
                {"Chỉ số": "Tổng đếm", "Baseline": total_bl, "Improved": total_im,
                 "Chênh lệch": f"{diff_count:+d}"},
                {"Chỉ số": "Runtime (s)", "Baseline": round(bl_r["runtime"], 2),
                 "Improved": round(im_r["runtime"], 2),
                 "Chênh lệch": f"{im_r['runtime']-bl_r['runtime']:+.2f}"},
                {"Chỉ số": "FPS", "Baseline": round(bl_r["fps"], 1),
                 "Improved": round(im_r["fps"], 1),
                 "Chênh lệch": f"{speedup:.2f}x nhanh hơn" if speedup >= 1 else f"{1/speedup:.2f}x chậm hơn"},
            ])
            st.dataframe(df_sum, use_container_width=True, hide_index=True)

            # Per-class chart
            def _class_dict(r):
                d = {}
                for _, dirs in r["totals"].items():
                    for _, per_cls in dirs.items():
                        for cid, n in per_cls.items():
                            name = r["class_names"][int(cid)] if r["class_names"] else str(cid)
                            d[name] = d.get(name, 0) + n
                return d

            bl_cls = _class_dict(bl_r)
            im_cls = _class_dict(im_r)
            all_cls = sorted(set(bl_cls) | set(im_cls))
            if all_cls:
                import matplotlib.pyplot as _plt
                import numpy as _np
                fig, ax = _plt.subplots(figsize=(8, 4))
                x = _np.arange(len(all_cls)); wbar = 0.38
                bl_v = [bl_cls.get(c, 0) for c in all_cls]
                im_v = [im_cls.get(c, 0) for c in all_cls]
                ax.bar(x - wbar/2, bl_v, wbar, label="Baseline", color="#7f7f7f", edgecolor="black")
                ax.bar(x + wbar/2, im_v, wbar, label="Improved", color="#d62728", edgecolor="black")
                for xi, v in zip(x - wbar/2, bl_v):
                    if v > 0: ax.text(xi, v+0.3, str(v), ha="center", fontsize=9)
                for xi, v in zip(x + wbar/2, im_v):
                    if v > 0: ax.text(xi, v+0.3, str(v), ha="center", fontsize=9)
                ax.set_xticks(x); ax.set_xticklabels(all_cls, rotation=15)
                ax.set_ylabel("Số phương tiện đếm được")
                ax.set_title("Counting per class — Baseline vs Improved")
                ax.legend(); ax.grid(axis="y", alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)

            st.info(
                f"**Đọc kết quả:** Improved đếm {total_im} vs Baseline {total_bl} "
                f"(chênh lệch {diff_count:+d}). Improved chạy nhanh hơn Baseline "
                f"{speedup:.2f}× ({im_r['fps']:.1f} vs {bl_r['fps']:.1f} FPS)."
                if speedup >= 1 else
                f"**Đọc kết quả:** Improved đếm {total_im} vs Baseline {total_bl}. "
                f"Improved chậm hơn Baseline ({im_r['fps']:.1f} vs {bl_r['fps']:.1f} FPS)."
            )

    elif st.button(" Chạy pipeline", type="primary", disabled="tmp_video" not in st.session_state):
        meta = st.session_state["video_meta"]
        w, h = meta["w"], meta["h"]

        # Build line
        if line_direction == "horizontal":
            y = int(h * line_y_pct / 100)
            line = Line(name="line_1", p1=(50, y), p2=(w - 50, y),
                        count_direction=count_direction)
        else:
            x = int(w * line_y_pct / 100)
            line = Line(name="line_1", p1=(x, 50), p2=(x, h - 50),
                        count_direction=count_direction)

        out_dir = ROOT / "results"
        out_video = out_dir / "videos" / f"streamlit_{Path(st.session_state['tmp_video']).stem}_out.mp4"
        out_csv = out_dir / "tables" / f"streamlit_{Path(st.session_state['tmp_video']).stem}_events.csv"

        progress = st.progress(0, text="Chạy pipeline...")

        def _cb(cur, total):
            if total > 0:
                progress.progress(min(cur / total, 1.0),
                                  text=f"Frame {cur}/{total}")

        _pipe_label = pipeline_choice.split("—")[0].strip()
        with st.spinner(f"[{_pipe_label}] Detecting + tracking + counting..."):
            if pipeline_kind == "baseline_pipeline":
                line_y_abs = int(h * line_y_pct / 100) if line_direction == "horizontal" else int(h * 0.5)
                events, totals, class_names = run_baseline_pipeline(
                    video=st.session_state["tmp_video"],
                    weights=str(ROOT / weights_path),
                    line_y=line_y_abs,
                    out_video=str(out_video),
                    out_csv=str(out_csv),
                    imgsz=imgsz, half=use_half,
                    progress_cb=_cb,
                )
            else:
                events, totals, class_names = run_pipeline(
                    video=st.session_state["tmp_video"],
                    weights=str(ROOT / weights_path),
                    lines=[line],
                    tracker=tracker,
                    conf=conf, iou=iou, imgsz=imgsz, half=use_half,
                    out_csv=str(out_csv),
                    out_video=str(out_video),
                    progress_cb=_cb,
                )

        progress.empty()
        st.success(f"[{_pipe_label}] Xong! {len(events)} lượt qua line.")
        st.session_state["events_csv"] = str(out_csv)
        st.session_state["out_video"] = str(out_video)
        st.session_state["totals"] = totals
        st.session_state["class_names"] = class_names

        # Transcode sang H.264 để play inline trong browser (mp4v không play được)
        h264_video = out_video.with_name(out_video.stem + "_h264.mp4")
        with st.spinner("Chuyển codec H.264 để xem trực tiếp trong trình duyệt..."):
            if transcode_to_h264(str(out_video), str(h264_video)):
                st.session_state["out_video_h264"] = str(h264_video)
            else:
                st.session_state["out_video_h264"] = None
                st.warning("Không tìm thấy ffmpeg hoặc transcode thất bại. "
                           "Video tải về sẽ vẫn xem được, nhưng preview inline có thể lỗi.")

    if not mode.startswith("⚖") and "out_video" in st.session_state:
        st.markdown("### Video output")
        # Ưu tiên bản H.264 (browser play được); fallback về mp4v gốc
        play_path = st.session_state.get("out_video_h264") or st.session_state["out_video"]
        st.video(play_path)
        c1, c2 = st.columns(2)
        with c1:
            with open(st.session_state["out_video"], "rb") as f:
                st.download_button("Tải video (mp4v gốc)", f,
                                   file_name=Path(st.session_state["out_video"]).name)
        with c2:
            if st.session_state.get("out_video_h264"):
                with open(st.session_state["out_video_h264"], "rb") as f:
                    st.download_button("Tải video (H.264, mở mọi player)", f,
                                       file_name=Path(st.session_state["out_video_h264"]).name)

# ---------- Tab 2: Stats ----------
with tab_stats:
    if "events_csv" not in st.session_state:
        st.info("Chạy pipeline ở tab đầu tiên để có dữ liệu thống kê.")
    else:
        # Nếu ở compare mode, cho phép chọn pipeline nguồn
        if "compare_results" in st.session_state:
            _cr = st.session_state["compare_results"]
            _sel = st.radio("Xem thống kê của pipeline nào?",
                            ["Improved (fine-tune)", "Baseline (COCO)"],
                            horizontal=True,
                            index=0 if st.session_state.get("stats_source_tag") == "improved" else 1,
                            key="stats_source_radio")
            _tag = "improved" if _sel.startswith("Improved") else "baseline"
            st.session_state["events_csv"] = _cr[_tag]["csv"]
            st.session_state["totals"] = _cr[_tag]["totals"]
            st.session_state["class_names"] = _cr[_tag]["class_names"]
            st.session_state["stats_source_tag"] = _tag
            st.caption(f"Đang thống kê **{_sel}** — weights: `{_cr[_tag]['weights']}`")

        df = pd.read_csv(st.session_state["events_csv"])
        if df.empty:
            st.warning("Không có event nào (0 xe qua line). Thử giảm confidence hoặc đổi vị trí line.")
        else:
            df_enriched = events_to_df(df.to_dict("records"))
            summ = summarize(df_enriched)

            # --- Bucket selector ---
            BUCKETS = {"30 giây": 30, "1 phút": 60, "5 phút": 300,
                       "15 phút": 900, "30 phút": 1800, "1 giờ": 3600}
            bucket_choice = st.selectbox("Chia thời gian theo",
                                         list(BUCKETS.keys()),
                                         index=1)
            bucket_sec = BUCKETS[bucket_choice]
            pivot = counts_per_bucket(df_enriched, bucket_sec)

            # --- Metrics tổng thể ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tổng lượt", summ["total"])
            c2.metric("Số loại xe", len(summ["per_class"]))
            c3.metric("Loại nhiều nhất",
                      max(summ["per_class"], key=summ["per_class"].get))
            c4.metric("Thời lượng (s)", f"{summ['duration_sec']:.1f}")

            # --- Flow rate ---
            st.markdown("#### Tốc độ lưu lượng")
            fr_min = flow_rate(df_enriched, unit="minute")
            fr_hour = flow_rate(df_enriched, unit="hour")
            fc1, fc2 = st.columns(2)
            fc1.metric("Tổng xe/phút", fr_min.get("total", 0))
            fc2.metric("Tổng xe/giờ", fr_hour.get("total", 0))

            if fr_hour.get("per_class"):
                fr_df = pd.DataFrame([
                    {"class": k, "xe/phút": fr_min["per_class"].get(k, 0),
                     "xe/giờ": fr_hour["per_class"].get(k, 0)}
                    for k in fr_hour["per_class"]
                ])
                st.dataframe(fr_df, use_container_width=True)

            # --- Peak period ---
            peak = peak_period(df_enriched, bucket_sec)
            if peak:
                st.markdown("#### Khung giờ cao điểm")
                pc1, pc2, pc3 = st.columns(3)
                pc1.metric(f"Bucket cao điểm ({bucket_choice})",
                           f"#{peak['peak_bucket']}")
                pc2.metric("Từ giây",
                           f"{peak['peak_start_sec']}–{peak['peak_end_sec']}")
                pc3.metric("Số lượt trong bucket", peak["peak_total"])

            # --- Chart theo bucket ---
            st.markdown(f"#### Lưu lượng theo {bucket_choice}")
            if pivot.empty:
                st.caption(f"Video quá ngắn để chia theo {bucket_choice}.")
            else:
                chart_kind = st.radio("Kiểu biểu đồ",
                                      ["Line", "Bar chồng"], horizontal=True)
                if chart_kind == "Line":
                    st.line_chart(pivot)
                else:
                    st.bar_chart(pivot)

            # --- Heatmap ---
            st.markdown("#### Heatmap thời gian × loại xe")
            if not pivot.empty:
                # Streamlit dùng dataframe styling để hiển thị màu
                st.dataframe(
                    pivot.style.background_gradient(cmap="YlOrRd", axis=None),
                    use_container_width=True,
                )

            # --- Cumulative ---
            st.markdown("#### Đếm tích luỹ theo thời gian")
            cum = cumulative_counts(df_enriched)
            if not cum.empty:
                st.line_chart(cum)

            # --- Per class ---
            st.markdown("#### Số lượng theo loại xe (tổng)")
            counts_df = pd.DataFrame(
                {"class_name": list(summ["per_class"].keys()),
                 "count": list(summ["per_class"].values())}
            ).sort_values("count", ascending=False)
            st.bar_chart(counts_df.set_index("class_name"))

            # --- Per direction ---
            dir_df = counts_by_direction(df_enriched)
            if not dir_df.empty:
                st.markdown("#### Lưu lượng theo chiều đi (line × direction)")
                st.dataframe(dir_df, use_container_width=True)
                st.bar_chart(dir_df)

            # --- Per line ---
            if len(summ["per_line"]) > 1:
                st.markdown("#### So sánh lưu lượng giữa các line")
                line_df = pd.DataFrame(
                    {"line": list(summ["per_line"].keys()),
                     "count": list(summ["per_line"].values())}
                )
                st.bar_chart(line_df.set_index("line"))

            # --- Event log ---
            st.markdown("#### Event log")
            st.dataframe(df_enriched, use_container_width=True, height=250)

            # --- Downloads ---
            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                st.download_button("Events CSV",
                                   df.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="events.csv")
            with dc2:
                st.download_button("Summary CSV",
                                   counts_df.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="summary.csv")
            with dc3:
                if not pivot.empty:
                    st.download_button(
                        f"Bucket {bucket_choice} CSV",
                        pivot.to_csv().encode("utf-8-sig"),
                        file_name=f"bucket_{bucket_sec}s.csv")

# ---------- Tab 3: Eval ----------
with tab_eval:
    if "events_csv" not in st.session_state:
        st.info("Chạy pipeline trước để có kết quả dự đoán.")
    else:
        # Nếu ở compare mode, cho phép chọn pipeline nguồn
        if "compare_results" in st.session_state:
            _cr = st.session_state["compare_results"]
            _sel = st.radio("Đánh giá pipeline nào?",
                            ["Improved (fine-tune)", "Baseline (COCO)"],
                            horizontal=True,
                            index=0 if st.session_state.get("stats_source_tag") == "improved" else 1,
                            key="eval_source_radio")
            _tag = "improved" if _sel.startswith("Improved") else "baseline"
            st.session_state["events_csv"] = _cr[_tag]["csv"]
            st.session_state["totals"] = _cr[_tag]["totals"]
            st.session_state["class_names"] = _cr[_tag]["class_names"]
            st.session_state["stats_source_tag"] = _tag
            st.caption(f"Đang đánh giá **{_sel}** — weights: `{_cr[_tag]['weights']}`")

        st.markdown("Upload JSON ground truth (đếm tay). "
                    "Cấu trúc: `{'counts_by_class': {'car': N, 'bus': N, ...}}`")

        gt_file = st.file_uploader("Ground truth JSON", type=["json"], key="gt")

        col_x, col_y = st.columns(2)
        with col_x:
            st.markdown("#### Hoặc điền tay bên dưới")
            st.caption("⚠ Nhập số xe **thật** bạn đếm được trong video (không phải số model dự đoán). "
                       "Nếu để mặc định = 0 sẽ không đánh giá đúng.")
            df = pd.read_csv(st.session_state["events_csv"])
            preds = df["class_name"].value_counts().to_dict() if not df.empty else {}
            gt_manual = {}
            for cls in ["motorcycle", "car", "bus", "truck", "van", "others", "bicycle"]:
                pred_val = preds.get(cls, 0)
                gt_manual[cls] = st.number_input(
                    f"GT — {cls}  (model dự đoán: {pred_val})",
                    min_value=0, value=0, step=1,
                    key=f"gt_input_{cls}",
                )

        with col_y:
            if gt_file is not None:
                gt_json = json.loads(gt_file.read())
                gt_counts = gt_json.get("counts_by_class", {})
            else:
                gt_counts = {k: v for k, v in gt_manual.items() if v > 0}

            pred_counts = df["class_name"].value_counts().to_dict() if not df.empty else {}
            if not gt_counts:
                st.warning("Chưa có GT.")
            else:
                rep = eval_report(pred_counts, gt_counts)
                st.markdown("#### Kết quả đánh giá")

                rows = []
                for cls, r in rep["per_class"].items():
                    rows.append({"Class": cls, "Pred": r["pred"], "GT": r["gt"],
                                 "|Err|": r["abs_err"], "Accuracy": r["accuracy"]})
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Overall Accuracy", f"{rep['overall_accuracy']:.3f}")
                c2.metric("MAE", f"{rep['MAE']:.2f}")
                c3.metric("MAPE (%)", f"{rep['MAPE_percent']:.2f}")

                st.download_button("Download eval report (JSON)",
                                   json.dumps(rep, indent=2, ensure_ascii=False),
                                   file_name="eval_report.json")

st.markdown("---")
st.caption("VN Traffic AI · Chủ đề 10 — Detection + Tracking + Counting")
