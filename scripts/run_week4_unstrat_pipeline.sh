#!/usr/bin/env bash
# Wait for unstratified bbox-grid, then rebuild week4 GT → EDA → train → 10/100m → briefing.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="${ROOT}/.venv/bin/python"
LOG="${ROOT}/outputs/week4_unstrat_pipeline.log"
mkdir -p "${ROOT}/outputs"
exec > >(tee -a "$LOG") 2>&1

echo "=== week4 unstrat pipeline start $(date -Is) ==="

wait_bbox() {
  local manifest="${ROOT}/datasets/building_bbox_grid/manifest.json"
  local buildings="${ROOT}/datasets/building_bbox_grid/buildings.parquet"
  echo "Waiting for bbox-grid finish (fresh buildings + manifest) ..."
  while true; do
    if [[ -f "$manifest" && -f "$buildings" ]]; then
      # must be newer than labels_only (unstrat rebuild marker)
      if [[ "$buildings" -nt "${ROOT}/outputs/building_type_dataset/labels_only.parquet" ]]; then
        local n
        n="$($PY - <<'PY'
import pandas as pd
from pathlib import Path
b=pd.read_parquet(Path("datasets/building_bbox_grid/buildings.parquet"), columns=["y_res"])
print(f"{len(b)} {b.y_res.mean():.3f}")
PY
)"
        local nb pct
        nb=$(echo "$n" | awk '{print $1}')
        pct=$(echo "$n" | awk '{print $2}')
        # expect ~490k and ~0.90 RES
        if [[ "$nb" -gt 100000 ]]; then
          echo "bbox-grid ready: n=$nb pct_res=$pct"
          return 0
        fi
      fi
    fi
    if ! pgrep -f 'build_building_bbox_grid_dataset.py --rebuild' >/dev/null 2>&1; then
      echo "ERROR: bbox rebuild process not running and outputs not ready" >&2
      exit 1
    fi
    sleep 30
  done
}

wait_bbox

echo "=== 1. week4 pixel GT ==="
$PY -u scripts/build_week4_pixel_gt.py

echo "=== 2. week4 pixel GT balanced (undersample majority/RES) ==="
$PY -u scripts/build_week4_pixel_gt_balanced.py

echo "=== 3. RES/NRES EDA ==="
$PY -u scripts/build_week4_res_nres_eda.py

echo "=== 4. train balanced LORO + score full GT OOF ==="
$PY -u scripts/train_week4_pixel_balanced_loro.py --also-score-full-gt

echo "=== 5. GHSL at AE centers (full GT) ==="
$PY -u scripts/eval_week4_ghsl_pixel.py

echo "=== 6. 100 m tiles + 10 vs 100 m fraction eval ==="
$PY -u scripts/build_week4_100m_tiles.py

echo "=== 7. 100 m tile-balanced + paired F1 ==="
$PY -u scripts/build_week4_100m_tiles_balanced.py

echo "=== 8. model comparison plots ==="
$PY -u scripts/plot_week4_model_comparison.py || true

echo "=== 9. update week4 briefing notebook ==="
$PY -u scripts/update_week4_briefing.py

echo "=== week4 unstrat pipeline DONE $(date -Is) ==="
