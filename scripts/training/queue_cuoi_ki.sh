#!/usr/bin/env bash
# Hàng đợi: chờ chuỗi cuoi_ki hiện tại (khoá logs/cuoi_ki.lock) rảnh rồi chạy lần lượt các bước.
#   bash scripts/training/queue_cuoi_ki.sh t2_ghep t2_pseudo
main() {
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
for step in "$@"; do
    until flock -n logs/cuoi_ki.lock true; do sleep 60; done
    bash scripts/training/run_cuoi_ki.sh $step
done
}
main "$@"; exit
