#!/usr/bin/env bash
# Dong goi V5+ (anh .jpg + nhan .txt, KHONG kem cache .npy) de upload len Google Drive cho Colab.
# Tao: data/colab/v5plus_frcnn.tar + data/colab/v5plus_frcnn.manifest.json (so anh, so nhan, sha256 cua tar).
# Chay:  bash scripts/colab/pack_v5plus_for_colab.sh
set -euo pipefail
cd /home/xuand/vn_traffic_ai
OUT=data/colab
mkdir -p "$OUT"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
for s in train val test; do
  mkdir -p "$TMP/v5plus/images/$s" "$TMP/v5plus/labels/$s"
  # -L: theo symlink de chep file that
  find -L data/merged_v5plus/images/$s -maxdepth 1 -name '*.jpg' -exec cp -L -t "$TMP/v5plus/images/$s/" {} +
  find -L data/merged_v5plus/labels/$s -maxdepth 1 -name '*.txt' -exec cp -L -t "$TMP/v5plus/labels/$s/" {} +
done
tar -C "$TMP" -cf "$OUT/v5plus_frcnn.tar" v5plus
python3 - "$TMP" "$OUT" <<'EOF'
import hashlib, json, sys
from pathlib import Path
tmp, out = Path(sys.argv[1]) / "v5plus", Path(sys.argv[2])
m = {s: {"images": len(list((tmp / "images" / s).glob("*.jpg"))),
         "labels": len(list((tmp / "labels" / s).glob("*.txt")))} for s in ("train", "val", "test")}
h = hashlib.sha256()
with open(out / "v5plus_frcnn.tar", "rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""):
        h.update(b)
m["tar_sha256"] = h.hexdigest()
(out / "v5plus_frcnn.manifest.json").write_text(json.dumps(m, indent=1))
print(json.dumps(m, indent=1))
EOF
ls -lh "$OUT"
