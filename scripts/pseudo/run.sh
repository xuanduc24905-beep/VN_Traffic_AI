#!/bin/bash
# Pipeline pseudo-labeling V6 — chạy từng bước dễ dàng
#
# Usage:
#   ./scripts/pseudo/run.sh              # mở menu
#   ./scripts/pseudo/run.sh scrape       # chạy bước 1: scrape URL camera
#   ./scripts/pseudo/run.sh collect      # chạy bước 2: thu snapshot
#   ./scripts/pseudo/run.sh dedup        # chạy bước 3: loại ảnh trùng
#   ./scripts/pseudo/run.sh status       # xem tiến độ hiện tại

set -e

PY=/home/xuand/miniconda3/envs/yolov8_ft/bin/python3
ROOT=/home/xuand/vn_traffic_ai
SCRIPTS=$ROOT/scripts/pseudo
LOGS=$ROOT/logs
YAML=$ROOT/data/raw/hcm_cameras.yaml
FRAMES=$ROOT/data/raw/frames/hcm

cd "$ROOT"
mkdir -p "$LOGS" "$FRAMES"

# ─── Colors ────────────────────────────────────────────────
G='\033[1;32m'; Y='\033[1;33m'; R='\033[1;31m'; B='\033[1;34m'; N='\033[0m'

msg()  { echo -e "${B}>>${N} $1"; }
ok()   { echo -e "${G}✓${N} $1"; }
warn() { echo -e "${Y}!${N} $1"; }
err()  { echo -e "${R}✗${N} $1"; }

check_env() {
  if ! [ -x "$PY" ]; then
    err "Không tìm thấy python env: $PY"
    err "Sửa biến PY ở đầu script, hoặc chạy: conda env list"
    exit 1
  fi
  ok "Python env: $PY"
}

# ─── Commands ──────────────────────────────────────────────
cmd_scrape() {
  msg "Bước 1: Scrape danh sách camera từ cameragiaothong.net"
  if [ -f "$YAML" ]; then
    n=$(grep -c 'image_url: https' "$YAML" 2>/dev/null || echo 0)
    warn "Yaml đã có $n camera. Script sẽ resume từ vị trí dừng."
    read -p "Tiếp tục? [y/N] " yn
    [ "$yn" != "y" ] && return
  fi
  LOG=$LOGS/scrape_$(date +%Y%m%d_%H%M%S).log
  msg "Log: $LOG"
  $PY "$SCRIPTS/01_scrape_camera_urls.py" 2>&1 | tee "$LOG"
  ok "Scrape xong. Yaml: $YAML"
}

cmd_collect() {
  msg "Bước 2: Thu snapshot từ camera"
  if ! [ -f "$YAML" ]; then
    err "Chưa có $YAML — chạy 'scrape' trước"
    exit 1
  fi

  # Hỏi tham số
  read -p "Thời lượng (phút) [30]: " dur; dur=${dur:-30}
  read -p "Số camera (random sample) [30]: " ncam; ncam=${ncam:-30}

  LOG=$LOGS/collect_$(date +%Y%m%d_%H%M%S).log
  msg "Chạy: $dur phút, $ncam camera. Log: $LOG"
  $PY "$SCRIPTS/02_collect_snapshots.py" --duration "$dur" --max-cameras "$ncam" 2>&1 | tee "$LOG"
  ok "Collect xong. Frames: $FRAMES"
}

cmd_infer() {
  msg "Bước 4: Inference V5 model -> pseudo-label"
  if ! [ -d "$FRAMES" ] || [ -z "$(ls -A "$FRAMES" 2>/dev/null)" ]; then
    err "Chưa có frames — chạy 'collect' trước"
    exit 1
  fi
  LOG=$LOGS/infer_$(date +%Y%m%d_%H%M%S).log
  msg "Log: $LOG"
  $PY "$SCRIPTS/04_inference.py" --batch 1 2>&1 | tee "$LOG"
  ok "Inference xong. Pseudo-labels: data/pseudo_v6/round1/predict/labels/"
}

cmd_review() {
  msg "Bước 6: Visualize pseudo-label để review tay"
  if ! [ -d "data/pseudo_v6/accepted/images" ]; then
    err "Chưa có accepted/ — chạy 'filter' trước"
    exit 1
  fi
  read -p "Số ảnh review [500]: " n; n=${n:-500}
  $PY "$SCRIPTS/06_visualize_review.py" --n "$n"
  ok "Review ảnh ở: data/pseudo_v6/review/"
  echo ""
  msg "Mở folder bằng:"
  echo "  code data/pseudo_v6/review/"
}

cmd_filter() {
  msg "Bước 5: Filter pseudo-label (per-class threshold + size)"
  if ! [ -d "data/pseudo_v6/round1/predict/labels" ]; then
    err "Chưa có pseudo-label — chạy 'infer' trước"
    exit 1
  fi
  echo ""
  msg "Dry-run trước:"
  $PY "$SCRIPTS/05_filter.py"
  echo ""
  read -p "Apply filter (copy ảnh + label qua accepted/)? [y/N] " yn
  if [ "$yn" = "y" ]; then
    $PY "$SCRIPTS/05_filter.py" --apply
    ok "Filter xong. Accepted: data/pseudo_v6/accepted/"
  else
    warn "Đã huỷ."
  fi
}

cmd_dedup() {
  msg "Bước 3: Dedup (loại ảnh trùng)"
  if ! [ -d "$FRAMES" ] || [ -z "$(ls -A "$FRAMES" 2>/dev/null)" ]; then
    err "Chưa có frames — chạy 'collect' trước"
    exit 1
  fi

  echo ""
  msg "Dry-run trước (xem bao nhiêu ảnh sẽ bị xoá):"
  $PY "$SCRIPTS/03_dedup.py"
  echo ""
  read -p "Thực sự xoá? [y/N] " yn
  if [ "$yn" = "y" ]; then
    $PY "$SCRIPTS/03_dedup.py" --apply
    ok "Dedup xong."
  else
    warn "Đã huỷ, không xoá."
  fi
}

cmd_status() {
  msg "═══════ Trạng thái pipeline V6 ═══════"
  echo ""

  # Scrape status
  if [ -f "$YAML" ]; then
    total=$(grep -c "^  '" "$YAML" 2>/dev/null || echo 0)
    ok_url=$(grep -c 'image_url: https' "$YAML" 2>/dev/null || echo 0)
    ok "Scrape: $ok_url/$total camera có URL"
  else
    warn "Scrape: chưa chạy"
  fi

  # Collect status
  if [ -d "$FRAMES" ]; then
    n_frames=$(find "$FRAMES" -name "*.jpg" 2>/dev/null | wc -l)
    n_wards=$(find "$FRAMES" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    if [ "$n_frames" -gt 0 ]; then
      ok "Collect: $n_frames ảnh ở $n_wards phường"
      echo "   Top phường (số ảnh):"
      find "$FRAMES" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | while read d; do
        n=$(ls "$d"/*.jpg 2>/dev/null | wc -l)
        echo "   - $n  $(basename "$d")"
      done | sort -rn | head -5
    else
      warn "Collect: chưa có ảnh"
    fi
  else
    warn "Collect: chưa chạy"
  fi

  # Background processes
  echo ""
  n_proc=$(ps -ef | grep -E "scripts/pseudo/0[123]" | grep -v grep | wc -l)
  if [ "$n_proc" -gt 0 ]; then
    warn "Đang có $n_proc process pipeline chạy nền:"
    ps -ef | grep -E "scripts/pseudo/0[123]" | grep -v grep | awk '{print "   PID " $2 ":", $NF}'
  fi
}

cmd_menu() {
  echo ""
  echo -e "${B}═══════════ Pipeline Pseudo-label V6 ═══════════${N}"
  echo ""
  echo "  1) scrape   — Thu danh sách camera (1 lần, ~15 phút)"
  echo "  2) collect  — Thu snapshot (session 30 phút mặc định)"
  echo "  3) dedup    — Loại ảnh trùng sau collect"
  echo "  4) infer    — Chạy V5 inference tạo pseudo-label"
  echo "  5) filter   — Lọc pseudo-label (confidence + size)"
  echo "  6) review   — Visualize bbox để duyệt tay 500 ảnh"
  echo "  7) status   — Xem tiến độ hiện tại"
  echo "  8) exit"
  echo ""
  read -p "Chọn [1-8]: " choice
  case "$choice" in
    1) cmd_scrape ;;
    2) cmd_collect ;;
    3) cmd_dedup ;;
    4) cmd_infer ;;
    5) cmd_filter ;;
    6) cmd_review ;;
    7) cmd_status ;;
    8|"") exit 0 ;;
    *) err "Lựa chọn không hợp lệ" ;;
  esac
}

# ─── Entry point ───────────────────────────────────────────
check_env

case "${1:-menu}" in
  scrape)  cmd_scrape ;;
  collect) cmd_collect ;;
  dedup)   cmd_dedup ;;
  infer)   cmd_infer ;;
  filter)  cmd_filter ;;
  review)  cmd_review ;;
  status)  cmd_status ;;
  menu|"") cmd_menu ;;
  -h|--help|help)
    echo "Usage: $0 [scrape|collect|dedup|infer|filter|review|status|menu]"
    ;;
  *)
    err "Lệnh không hợp lệ: $1"
    echo "Usage: $0 [scrape|collect|dedup|infer|filter|review|status|menu]"
    exit 1
    ;;
esac
