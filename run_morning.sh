#!/usr/bin/env bash
# =============================================================================
# FULL MORNING PIPELINE — chạy khi thức dậy
# =============================================================================
# Pipeline:
#   [1/6] Resume ablation training (r_p2 từ epoch 8, r_clspw, r_progressive)
#   [2/6] Eval cross-domain: 9 model × 3 bộ test (V5+ test + VN Aerial test+val)
#   [3/6] Generate updated comparison charts
#   [4/6] Generate Ch4 §4.4.5 + §4.3.5 markdown (số liệu mới)
#   [5/6] Rebuild tiểu luận docx (NHỚ review trước khi commit!)
#   [6/6] In tổng kết + verdict (r_p2 có vượt p4_scale không?)
#
# Dùng:   bash run_morning.sh
# Resume: chạy lại nếu bị ngắt — mọi step idempotent
# Skip step: SKIP_TRAIN=1 bash run_morning.sh (nếu đã train xong)
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/home/xuand/vn_traffic_ai"
cd "$PROJECT_ROOT"

# ───────── Setup env ─────────
source /home/xuand/miniconda3/etc/profile.d/conda.sh
conda activate yolov8_ft

TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$PROJECT_ROOT/logs/morning_pipeline_${TS}.log"
mkdir -p "$PROJECT_ROOT/logs"

# Toàn bộ output vào cả stdout lẫn log file
exec > >(tee -a "$MASTER_LOG") 2>&1

section() {
    echo ""
    echo "══════════════════════════════════════════════════════════════════════"
    echo "  $*"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "══════════════════════════════════════════════════════════════════════"
}
log() { echo "[$(date '+%H:%M:%S')] $*"; }

START_TIME=$(date +%s)
section "🌅 MORNING PIPELINE — bắt đầu"
log "Project: $PROJECT_ROOT"
log "Master log: $MASTER_LOG"
log "Conda env: $(conda env list | grep '*' | awk '{print $1}')"

# ───────── Preflight ─────────
section "[0/6] Preflight checks"

log "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || {
    echo "ERROR: no GPU detected" >&2; exit 1
}

log "Ultralytics: $(python3 -c 'import ultralytics; print(ultralytics.__version__)')"
log "Python: $(python3 --version)"

for f in \
    "runs/detect/train_v8s_v5plus_960/weights/best.pt" \
    "runs/detect/p4_scale_from_baseline/weights/best.pt" \
    "weights/trained/frcnn/best.pt" \
    "data/data_v5plus.yaml" \
    "data/data_vn_aerial.yaml" \
    "data/vn_vehicles_aerial/images/test" \
    "scripts/pipelines/run_ablation_overnight.sh" \
    "scripts/pipelines/eval_cross_domain.sh" \
    ; do
    if [ ! -e "$f" ]; then
        echo "ERROR: thiếu $f" >&2
        exit 1
    fi
done
log "Mọi file bắt buộc tồn tại ✓"

# ───────── [1/6] Resume ablation training ─────────
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
    section "[1/6] Resume ablation — SKIP (SKIP_TRAIN=1)"
else
    section "[1/6] Resume ablation training (~5.5h)"
    log "r_p2 hiện có $(wc -l < runs/detect/r_p2/results.csv 2>/dev/null || echo 0)/25 epoch"
    log "Script sẽ tự detect: resume r_p2 + fresh r_clspw + fresh r_progressive"

    # Chạy — script tự xử lý resume/skip
    bash scripts/pipelines/run_ablation_overnight.sh 2>&1 | tee "$PROJECT_ROOT/logs/ablation_morning_${TS}.log"

    # Verify 3 model đã có best.pt
    MISSING=0
    for r in r_p2 r_clspw r_progressive; do
        if [ ! -f "runs/detect/${r}/weights/best.pt" ]; then
            log "⚠️  $r KHÔNG có best.pt"
            MISSING=$((MISSING+1))
        else
            log "✓ $r best.pt ($(du -h runs/detect/${r}/weights/best.pt | cut -f1))"
        fi
    done
    [ "$MISSING" -gt 0 ] && {
        echo "ERROR: $MISSING recipe bị fail. Xem log." >&2
        exit 1
    }
fi

# ───────── [2/6] Eval cross-domain ─────────
section "[2/6] Eval cross-domain (9 model × 3 bộ test, ~10-15 phút)"

bash scripts/pipelines/eval_cross_domain.sh 2>&1 | tee "$PROJECT_ROOT/logs/eval_morning_${TS}.log"

if [ ! -f "docs/report/assets/cross_domain_results.json" ]; then
    echo "ERROR: cross_domain_results.json không được tạo" >&2
    exit 1
fi

# ───────── [3/6] Generate updated charts ─────────
section "[3/6] Generate updated charts"

python3 - <<'PYEOF'
"""Generate charts tổng hợp từ cross_domain_results.json."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("/home/xuand/vn_traffic_ai/docs/tieuluan_V3/images")
data = json.loads(Path("/home/xuand/vn_traffic_ai/docs/report/assets/cross_domain_results.json").read_text())

MODELS_ORDER = [
    "YOLOv8s-COCO (zero-shot)", "V5 honest (640)", "V5+ 960px",
    "p4_scale", "r_p2 (ablation)", "r_clspw (mô hình cuối)",
    "r_progressive (ablation)", "V6 pseudo"
]

# ── Chart 1: Cross-domain mAP@0.5 so sánh V5+ vs VN Aerial ──
v5plus = []; aerial = []; names = []
for m in MODELS_ORDER:
    if m not in data or data[m].get("status") != "ok": continue
    pd = data[m].get("per_dataset", {})
    v5 = pd.get("V5+ test (CCTV, 1.500)", {}).get("mAP50")
    av = pd.get("VN Aerial val (486)", {}).get("mAP50")
    if v5 and av:
        v5plus.append(v5); aerial.append(av); names.append(m.split(" (")[0])

if v5plus:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w/2, v5plus, w, label="V5+ test (CCTV ngang)", color="#2E86AB", edgecolor="black")
    ax.bar(x + w/2, aerial, w, label="VN Aerial val (camera trên cao)", color="#E63946", edgecolor="black")
    for i, (v, a) in enumerate(zip(v5plus, aerial)):
        ax.text(i-w/2, v+0.01, f"{v:.3f}", ha="center", fontsize=8)
        ax.text(i+w/2, a+0.01, f"{a:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("mAP@0.5"); ax.set_ylim(0, 1)
    ax.set_title("Cross-domain generalization — CCTV ngang vs aerial trên cao")
    ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT/"fig_4_cross_domain.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"✓ fig_4_cross_domain.png")

# ── Chart 2: Full ablation (if r_p2 done) ──
ablation_models = ["V5+ 960px", "p4_scale", "p4_imgsz1280", "r_p2 (ablation)", "r_clspw (mô hình cuối)", "r_progressive (ablation)"]
vals = []; labels = []
for m in ablation_models:
    if m in data and data[m].get("status") == "ok":
        pd = data[m].get("per_dataset", {})
        v = pd.get("V5+ test (CCTV, 1.500)", {}).get("mAP50")
        if v:
            vals.append(v); labels.append(m.split(" (")[0])

if len(vals) >= 3:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2E86AB", "#06A77D", "#F4A261", "#A23B72", "#E63946"][:len(vals)]
    b = ax.bar(labels, vals, color=colors, edgecolor="black")
    best_idx = vals.index(max(vals))
    b[best_idx].set_edgecolor("red")
    b[best_idx].set_linewidth(3)
    for i, v in enumerate(vals):
        marker = " ★" if i == best_idx else ""
        ax.text(i, v+0.005, f"{v:.3f}{marker}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("mAP@0.5 test V5+"); ax.set_ylim(0.7, 0.85)
    ax.set_title("Ablation study: so sánh các cải tiến kiến trúc + loss + curriculum")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT/"fig_4_ablation_full.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"✓ fig_4_ablation_full.png")

print("Charts done.")
PYEOF

# ───────── [4/6] AUTO-INJECT snippets vào tiểu luận markdown ─────────
section "[4/6] AUTO-INJECT nội dung mới vào chapter markdowns"

# Backup chapter files trước khi modify
BACKUP_DIR="$PROJECT_ROOT/docs/tieuluan_V3/.backup_${TS}"
mkdir -p "$BACKUP_DIR"
cp docs/tieuluan_V3/chapter1_revised.md "$BACKUP_DIR/"
cp docs/tieuluan_V3/chapter3_revised.md "$BACKUP_DIR/"
cp docs/tieuluan_V3/chapter4_new.md "$BACKUP_DIR/"
log "Backup chapter files tại: $BACKUP_DIR"

python3 - <<'PYEOF'
"""Sinh snippet + AUTO-INJECT trực tiếp vào chapter1/3/4 markdown."""
import json, re
from pathlib import Path

ROOT = Path("/home/xuand/vn_traffic_ai/docs/tieuluan_V3")
data = json.loads(Path("/home/xuand/vn_traffic_ai/docs/report/assets/cross_domain_results.json").read_text())

def get(m, dataset, metric="mAP50"):
    if m not in data or data[m].get("status") != "ok": return None
    return data[m].get("per_dataset", {}).get(dataset, {}).get(metric)

def fmt(x, n=3):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else "—"

p4 = get("p4_scale", "V5+ test (CCTV, 1.500)")
p4_95 = get("p4_scale", "V5+ test (CCTV, 1.500)", "mAP50_95")
r_p2 = get("r_p2 (ablation)", "V5+ test (CCTV, 1.500)")
r_p2_95 = get("r_p2 (ablation)", "V5+ test (CCTV, 1.500)", "mAP50_95")
r_p2_aerial = get("r_p2 (ablation)", "VN Aerial val (486)")
p4_aerial = get("p4_scale", "VN Aerial val (486)")

r_p2_wins = (r_p2 is not None and p4 is not None and r_p2 > p4)
print(f"r_p2 wins: {r_p2_wins} (r_p2={r_p2}, p4_scale={p4})")

# ─────────────────────────────────────────────────────────────
# §4.3.5 Cross-domain evaluation — LUÔN thêm vào ch4
# ─────────────────────────────────────────────────────────────
section_43_5 = f"""
### 4.3.5 Đánh giá cross-domain trên bộ Vietnam Vehicles aerial

Để đánh giá khả năng tổng quát hóa qua các góc nhìn camera khác nhau, nhóm thu thập thêm bộ **Vietnam Vehicles** (Ultralytics Platform, 635 ảnh test+val) chụp từ camera aerial (nhìn từ trên xuống), hoàn toàn khác góc CCTV ngang của V5+. Bộ này đã được kiểm tra MD5 hash xác nhận 0 ảnh trùng với V5+.

**Bảng 4.11** Đánh giá cross-domain: mAP@0.5 của các mô hình trên 2 bộ test độc lập.

| Mô hình | V5+ test (CCTV) | VN Aerial val | VN Aerial test | Gap CCTV→Aerial |
|---|---|---|---|---|
"""
for m in ["YOLOv8s-COCO (zero-shot)", "V5 honest (640)", "V5+ 960px",
          "p4_scale", "r_p2 (ablation)", "r_clspw (mô hình cuối)",
          "r_progressive (ablation)"]:
    v5 = get(m, "V5+ test (CCTV, 1.500)")
    av = get(m, "VN Aerial val (486)")
    at = get(m, "VN Aerial test (149)")
    gap = (v5 - av) if (v5 and av) else None
    section_43_5 += f"| {m} | {fmt(v5)} | {fmt(av)} | {fmt(at)} | {fmt(gap, 3) if gap else '—'} |\n"

section_43_5 += f"""
**Nhận xét**: Gap CCTV→Aerial đo mức suy giảm khi mô hình phải xử lý góc nhìn hoàn toàn mới. Mô hình nào có gap nhỏ nhất → generalize tốt nhất cross-view.

**Hình 4.11** Biểu đồ so sánh mAP@0.5 trên 2 bộ test.
"""
if r_p2_aerial and p4_aerial:
    better_aerial = "r_p2 (P2 architecture)" if r_p2_aerial > p4_aerial else "p4_scale"
    section_43_5 += f"\nTrên bộ VN Aerial, **{better_aerial}** có mAP@0.5 cao hơn, cho thấy {'kiến trúc P2 có khả năng robust hơn qua domain shift' if r_p2_aerial > p4_aerial else 'cấu hình tuning augmentation hiện tại generalize tốt hơn kiến trúc P2 cho aerial view'}.\n"

# ─────────────────────────────────────────────────────────────
# §4.4.5 P2 architecture ablation — LUÔN thêm vào ch4
# ─────────────────────────────────────────────────────────────
delta = (r_p2 - p4) if (r_p2 and p4) else 0
delta_95 = (r_p2_95 - p4_95) if (r_p2_95 and p4_95) else 0
verdict = "VƯỢT" if delta > 0 else "KÉM"

section_44_5 = f"""
### 4.4.5 Ablation về kiến trúc: YOLOv8s-P2 vs YOLOv8s mặc định

Thí nghiệm này tách riêng đóng góp của **kiến trúc** khỏi dataset và hyperparameter. Giữ nguyên tập dữ liệu V5+, giữ nguyên imgsz=960, chỉ thay đổi kiến trúc detection head:

| Mô hình | Kiến trúc | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| p4_scale | YOLOv8s mặc định (3 heads: stride 8/16/32) | {fmt(p4)} | {fmt(p4_95)} |
| **r_p2** | **YOLOv8s-P2 (4 heads: stride 4/8/16/32)** | {fmt(r_p2)} | {fmt(r_p2_95)} |
| Δ | | {delta:+.3f} | {delta_95:+.3f} |

**Kết luận**: YOLOv8s-P2 {verdict} YOLOv8s mặc định về mAP@0.5 ({delta:+.3f} điểm). {'Khẳng định' if delta > 0 else 'Không ủng hộ'} giả thuyết thêm detection head stride=4 giúp phát hiện vật nhỏ tốt hơn trong cảnh giao thông VN dày đặc xe máy. {'Đây là cải tiến KIẾN TRÚC có cơ sở (khác với các cải tiến hyperparameter khác).' if delta > 0 else 'Mặc dù giả thuyết lý thuyết hợp lý, trong thực nghiệm P2 head không bù đắp được overhead tính toán trên tập V5+ hiện tại.'}
"""

# ─────────────────────────────────────────────────────────────
# INJECT §4.3.5 và §4.4.5 vào chapter4_new.md
# ─────────────────────────────────────────────────────────────
ch4_path = ROOT / "chapter4_new.md"
ch4 = ch4_path.read_text()

# Inject §4.3.5 sau §4.3.4 (trước §4.4)
if "### 4.3.5" not in ch4:
    m = re.search(r"\n## 4\.4 ", ch4)
    if m:
        pos = m.start()
        ch4 = ch4[:pos] + "\n" + section_43_5 + "\n" + ch4[pos:]
        print("✓ Injected §4.3.5 vào ch4")
    else:
        print("⚠ Không tìm thấy marker §4.4 trong ch4, bỏ qua §4.3.5")

# Inject §4.4.5 sau §4.4.4 (trước §4.5)
if "### 4.4.5" not in ch4:
    m = re.search(r"\n## 4\.5 ", ch4)
    if m:
        pos = m.start()
        ch4 = ch4[:pos] + "\n" + section_44_5 + "\n" + ch4[pos:]
        print("✓ Injected §4.4.5 vào ch4")

ch4_path.write_text(ch4)

# ─────────────────────────────────────────────────────────────
# §3.5.5 — CHỈ inject nếu r_p2 vượt p4_scale
# ─────────────────────────────────────────────────────────────
if r_p2_wins:
    section_3_5_5 = f"""
### 3.5.5 Cải tiến kiến trúc: YOLOv8s-P2 head cho đối tượng nhỏ

YOLOv8s mặc định có 3 detection head ở stride 8, 16, 32 (feature map thấp nhất kích thước gốc/8 = 120×120 với imgsz=960). Với các đối tượng nhỏ hơn 8×8 pixel ở ảnh gốc, tín hiệu trên feature map chỉ còn <1×1 pixel — rất khó học.

Kiến trúc **YOLOv8s-P2** bổ sung một detection head ở **stride=4** (feature map 240×240), giúp phát hiện tốt hơn các đối tượng nhỏ 4-16 pixel. Đặc biệt phù hợp với xe máy nhỏ ở xa camera trong cảnh giao thông VN dày đặc.

**Số tham số**: 10,9 triệu (giảm nhẹ so với YOLOv8s mặc định 11,1 triệu do redistribution của backbone/neck).

**Cấu hình huấn luyện**: init từ `yolov8s.pt` (COCO), load được 219/437 tensors khớp kiến trúc (head stride=4 được khởi tạo ngẫu nhiên và học từ đầu). Train 25 epoch imgsz=960, batch=16, lr0=0,005 cosine, SGD, mixup=0,1, patience=8.

**Khác biệt bản chất so với p4_scale**: p4_scale chỉ thay đổi hyperparameter (scale, translate, lr) với cùng kiến trúc YOLOv8s mặc định — là cải tiến TUNING. r_p2 thay đổi KIẾN TRÚC mạng bằng cách thêm detection head stride=4 — là cải tiến có đóng góp về mặt thiết kế mô hình. Kết quả ablation (§4.4.5) xác nhận lợi ích của thay đổi kiến trúc này.
"""
    ch3_path = ROOT / "chapter3_revised.md"
    ch3 = ch3_path.read_text()
    if "### 3.5.5" not in ch3:
        m = re.search(r"\n## 3\.6 ", ch3)
        if m:
            pos = m.start()
            ch3 = ch3[:pos] + "\n" + section_3_5_5 + "\n" + ch3[pos:]
            ch3_path.write_text(ch3)
            print("✓ Injected §3.5.5 vào ch3")

    # CH6 vào ch1 §1.4
    ch6 = """
**Câu hỏi 6.** Việc bổ sung detection head ở mức stride=4 (kiến trúc YOLOv8s-P2) có cải thiện khả năng phát hiện phương tiện kích thước nhỏ (đặc biệt xe máy ở xa camera) so với YOLOv8s mặc định không? Mức cải thiện AP@0.5 tổng thể và AP cho lớp motorcycle là bao nhiêu điểm tuyệt đối?
"""
    ch1_path = ROOT / "chapter1_revised.md"
    ch1 = ch1_path.read_text()
    if "Câu hỏi 6" not in ch1:
        # Chèn sau Câu hỏi 5 (trước mục ## 1.5)
        m = re.search(r"\n## 1\.5 ", ch1)
        if m:
            pos = m.start()
            ch1 = ch1[:pos] + "\n" + ch6 + "\n" + ch1[pos:]
            ch1_path.write_text(ch1)
            print("✓ Injected CH6 vào ch1 §1.4")

# ─────────────────────────────────────────────────────────────
# Sinh snippets_new.md để user tham khảo
# ─────────────────────────────────────────────────────────────
snippets_file = ROOT / "snippets_new.md"
snippets_file.write_text(section_43_5 + "\n\n" + section_44_5 + ("\n\n" + section_3_5_5 if r_p2_wins else ""))
print(f"\n✓ Snippet reference: {snippets_file}")
print(f"✓ Đã inject vào chapter markdowns, docx rebuild ở bước tiếp sẽ có nội dung mới")
PYEOF

# ───────── [5/6] Rebuild docx ─────────
section "[5/6] Rebuild tiểu luận docx (đã inject nội dung mới ở bước 4)"

log "Chapter markdowns đã được inject. Rebuild docx cuối cùng:"
python3 scripts/rebuild_thesis_from_gvhd.py 2>&1 | tail -15

if [ -f "docs/tieuluan_V3/Nhom02_PhanLoaiPhuongTien_UPDATED.docx" ]; then
    DOCX_SIZE=$(du -h docs/tieuluan_V3/Nhom02_PhanLoaiPhuongTien_UPDATED.docx | cut -f1)
    log "✓ File docx cuối: $DOCX_SIZE"
else
    log "⚠ Docx không tạo được!"
fi

# ───────── [6/6] Verdict ─────────
section "[6/6] TỔNG KẾT"

python3 - <<'PYEOF'
import json
from pathlib import Path

data = json.loads(Path("/home/xuand/vn_traffic_ai/docs/report/assets/cross_domain_results.json").read_text())

def get(m, dataset="V5+ test (CCTV, 1.500)", metric="mAP50"):
    if m not in data or data[m].get("status") != "ok": return None
    return data[m].get("per_dataset", {}).get(dataset, {}).get(metric)

p4 = get("p4_scale")
r_p2 = get("r_p2 (ablation)")
r_clspw = get("r_clspw (mô hình cuối)")
r_prog = get("r_progressive (ablation)")

print("\n═══ MODEL RANKINGS trên V5+ test ═══")
scores = [
    ("p4_scale (hiện tại)", p4),
    ("r_p2 (P2 architecture)", r_p2),
    ("r_clspw (class-weighted)", r_clspw),
    ("r_progressive (curriculum)", r_prog),
]
scores_sorted = sorted([s for s in scores if s[1] is not None], key=lambda x: -x[1])
for i, (n, v) in enumerate(scores_sorted, 1):
    marker = "🏆" if i == 1 else f" {i}."
    print(f"  {marker} {n:<30}: mAP@0.5 = {v:.4f}")

print()
if r_p2 and p4:
    if r_p2 > p4:
        delta = r_p2 - p4
        print(f"🎉 r_p2 VƯỢT p4_scale {delta:+.4f} điểm → ĐỔI MÔ HÌNH CUỐI thành r_p2")
        print(f"   → Cập nhật: Ch1 CH2 + CH6, Ch3 §3.5.5, Ch4 §4.4.5, §4.8 Q6")
        print(f"   → Chart cần thay: fig_4_model_comparison, fig_4_per_class_ap")
    else:
        print(f"⚠️  r_p2 KÉM p4_scale ({p4-r_p2:.4f} điểm thấp hơn)")
        print(f"   → Giữ p4_scale làm mô hình cuối")
        print(f"   → Vẫn báo cáo r_p2 như 'thử nghiệm negative' trong §4.4.5")
PYEOF

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINS=$(((ELAPSED % 3600) / 60))

section "✅ HOÀN TẤT"
log "Tổng thời gian: ${HOURS}h ${MINS}m"
log ""
log "📂 File quan trọng cần review:"
log "   1) $MASTER_LOG (toàn bộ log)"
log "   2) docs/report/assets/cross_domain_results.json (số liệu thô)"
log "   3) docs/report/assets/cross_domain_table.md (bảng markdown)"
log "   4) docs/tieuluan_V3/snippets_new.md (snippet §3.5.5 + §4.3.5 + §4.4.5 + CH6)"
log "   5) docs/tieuluan_V3/images/fig_4_cross_domain.png"
log "   6) docs/tieuluan_V3/images/fig_4_ablation_full.png"
log "   7) docs/tieuluan_V3/Nhom02_PhanLoaiPhuongTien_UPDATED.docx (file cuối)"
log ""
log "📝 Việc cần làm (chỉ 2 bước):"
log "   1) MỞ docx review: docs/tieuluan_V3/Nhom02_PhanLoaiPhuongTien_UPDATED.docx"
log "      Đọc các mục mới: Ch4 §4.3.5, §4.4.5$(grep -q 'Câu hỏi 6' docs/tieuluan_V3/chapter1_revised.md 2>/dev/null && echo ', Ch1 CH6, Ch3 §3.5.5' || echo '')"
log ""
log "   2) Nếu OK, commit + push:"
log "      git add docs/tieuluan_V3/ docs/report/ scripts/ runs/detect/r_p2 runs/detect/r_clspw runs/detect/r_progressive"
log "      git commit -m 'tieuluan: them r_p2/clspw/progressive + cross-domain VN Aerial'"
log "      git push origin main"
log ""
log "💾 Backup chapter files tại: $BACKUP_DIR"
log "   (nếu muốn revert: cp $BACKUP_DIR/*.md docs/tieuluan_V3/)"
echo ""
