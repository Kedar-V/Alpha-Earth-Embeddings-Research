#!/usr/bin/env python3
"""Build interactive Leaflet map overlay for Overture + AlphaEarth + CNN + GHSL.

Writes outputs/week2_cnn_bbox/map_overlay/:
  index.html                 interactive map (toggles, lat/lon, address search)
  regions.json               metro bounds + counts
  buildings_<region>.geojson Overture polygons + CNN false-color + labels
  cells_<region>.geojson     10 m AE grid cells (rectangles) + AE false-color
  ghsl_<region>.png          GHS-BUILT-C FUN image overlay (WGS84, nearest)

Serve with:
  python scripts/serve_map_overlay.py
  then open http://127.0.0.1:8765/

Fast path (GHSL + HTML only, keep existing geojson):
  python scripts/build_map_overlay.py --ghsl-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import reproject
from shapely import wkt as shapely_wkt
from shapely.geometry import box
from shapely.ops import transform as shapely_transform
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BBOX = ROOT / "datasets" / "building_bbox_grid"
EMB = ROOT / "outputs" / "week2_cnn_bbox" / "cnn_embeddings_oof.parquet"
PRED = ROOT / "outputs" / "week2_cnn_bbox" / "cnn_oof_predictions.parquet"
GHSL_BUILDING = ROOT / "outputs" / "validation" / "per_building_predictions.parquet"
OUT = ROOT / "outputs" / "week2_cnn_bbox" / "map_overlay"

# GHS-BUILT-C FUN display colors (RGBA)
GHSL_RGBA = {
    0: (220, 225, 230, 55),  # non-built
    1: (39, 174, 96, 210),  # RES
    2: (230, 126, 34, 210),  # NRES
    255: (0, 0, 0, 0),  # nodata
}
GHSL_HEX = {"RES": "#27ae60", "NRES": "#e67e22", "NONBUILT": "#bdc3c7", "NODATA": "#7f8c8d"}
GHSL_MAX_DIM = 2048
GHSL_PAD_DEG = 0.01

BANDS = [f"A{i:02d}" for i in range(64)]
ECOLS = [f"E{i:03d}" for i in range(256)]
# AlphaEarth Foundations COG pixel size (meters in UTM). Centers are mid-pixel.
PIXEL_M = 10.0
HALF_M = PIXEL_M / 2.0
# Default continuous colormap; UI can switch among these.
CMAP_NAME = "cividis"
CMAP_CHOICES = ("cividis", "turbo", "inferno")
# Continuous embedding color: UMAP 1D → colormap (UI switches cividis/turbo/inferno).
MANIFOLD_METHODS = ("umap",)
MANIFOLD_DEFAULT = "umap"
MANIFOLD_PCA_DIM = 50  # PCA warmup before UMAP
MANIFOLD_N_COMPONENTS = 1  # scalar for colormap (not RGB packing)
PCA_N = 10  # legacy
PCA_RGB_N = 3


def robust_unit(x: np.ndarray) -> np.ndarray:
    """Map values to [0, 1] via 2–98 percentiles (NaNs → 0.5)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, 0.5, dtype=np.float64)
    m = np.isfinite(x)
    if not m.any():
        return out
    lo, hi = np.percentile(x[m], [2, 98])
    if hi <= lo:
        return out
    out[m] = np.clip((x[m] - lo) / (hi - lo), 0, 1)
    return out


def cmap_to_hex(values_01: np.ndarray, cmap_name: str = CMAP_NAME) -> list[str]:
    """Encode [0,1] scalars with a perceptually uniform matplotlib colormap."""
    cmap = colormaps[cmap_name]
    rgba = cmap(np.clip(values_01, 0, 1))
    out = []
    for r, g, b, _a in rgba:
        out.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return out


def _scores_to_rgb_hex(Z: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Robust-scale each column of Z and pack first 3 as RGB hex."""
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = Z.shape[0]
    channels = [robust_unit(Z[:, i]) for i in range(min(3, Z.shape[1]))]
    while len(channels) < 3:
        channels.append(np.full(n, 0.5, dtype=np.float64))
    r, g, b = channels[0], channels[1], channels[2]
    hexes = [
        f"#{int(rr * 255):02x}{int(gg * 255):02x}{int(bb * 255):02x}"
        for rr, gg, bb in zip(r, g, b)
    ]
    return hexes, np.column_stack(channels)


def manifold_embed(
    X: np.ndarray,
    method: str = MANIFOLD_DEFAULT,
    pca_dim: int = MANIFOLD_PCA_DIM,
    n_components: int = MANIFOLD_N_COMPONENTS,
    random_state: int = 42,
) -> np.ndarray:
    """Embed rows of X with UMAP or t-SNE (PCA warmup). Default: 1D for colormap."""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    n_comp = int(max(1, n_components))
    if n == 0:
        return np.zeros((0, n_comp), dtype=np.float64)
    if n == 1:
        return np.zeros((1, n_comp), dtype=np.float64)

    # PCA warmup for speed / noise reduction
    n_pca = int(min(pca_dim, d, n - 1))
    n_pca = max(n_pca, 2)
    Xp = PCA(n_components=n_pca, random_state=random_state).fit_transform(X)
    method = (method or MANIFOLD_DEFAULT).lower().strip()
    n_comp = int(min(n_comp, n - 1, Xp.shape[1]))
    n_comp = max(n_comp, 1)

    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = float(min(30, max(5, (n - 1) // 3)))
        Z = TSNE(
            n_components=n_comp,
            perplexity=perplexity,
            metric="cosine",
            init="pca",
            learning_rate="auto",
            random_state=random_state,
            max_iter=750,
        ).fit_transform(Xp)
        return np.asarray(Z, dtype=np.float64).reshape(n, n_comp)

    # default: umap
    import umap

    n_neighbors = int(min(15, max(2, n - 1)))
    reducer = umap.UMAP(
        n_components=n_comp,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
    )
    Z = reducer.fit_transform(Xp)
    return np.asarray(Z, dtype=np.float64).reshape(n, n_comp)


def manifold_3d(
    X: np.ndarray,
    method: str = MANIFOLD_DEFAULT,
    pca_dim: int = MANIFOLD_PCA_DIM,
    random_state: int = 42,
) -> np.ndarray:
    """Legacy 3D manifold embed (RGB packing). Prefer ``manifold_embed`` + colormap."""
    return manifold_embed(X, method=method, pca_dim=pca_dim, n_components=3, random_state=random_state)


def unit_hex_from_manifold(
    X: np.ndarray,
    method: str = MANIFOLD_DEFAULT,
    pca_dim: int = MANIFOLD_PCA_DIM,
    cmap_name: str = CMAP_NAME,
    random_state: int = 42,
) -> tuple[list[str], np.ndarray]:
    """UMAP/t-SNE → 1D → [0,1] units + default-colormap hex (UI remaps via LUT)."""
    Z = manifold_embed(X, method=method, pca_dim=pca_dim, n_components=1, random_state=random_state)
    u = robust_unit(Z[:, 0])
    return cmap_to_hex(u, cmap_name), u.reshape(-1, 1)


def rgb_hex_from_manifold(
    X: np.ndarray,
    method: str = MANIFOLD_DEFAULT,
    pca_dim: int = MANIFOLD_PCA_DIM,
    random_state: int = 42,
) -> tuple[list[str], np.ndarray]:
    """UMAP/t-SNE → colormap hex from 1st axis (returns units as (n,1) for ``[:,0]`` callers)."""
    return unit_hex_from_manifold(X, method=method, pca_dim=pca_dim, random_state=random_state)


def pca_scores(X: np.ndarray, n_components: int = PCA_N) -> np.ndarray:
    """Return (n, n_components) PCA scores (deterministic)."""
    X = np.asarray(X, dtype=np.float64)
    n_comp = int(min(n_components, X.shape[0], X.shape[1]))
    return PCA(n_components=n_comp, random_state=42).fit_transform(X)


def rgb_hex_from_pca(X: np.ndarray, n_components: int = PCA_N, n_rgb: int = PCA_RGB_N) -> tuple[list[str], np.ndarray]:
    """Legacy: fit ``n_components`` PCs; map first ``n_rgb`` → RGB hex."""
    Z = pca_scores(X, n_components=n_components)
    return _scores_to_rgb_hex(Z[:, :n_rgb])


def pc1_unit(X: np.ndarray) -> np.ndarray:
    """First PCA component → [0, 1] via robust percentiles (legacy helper)."""
    z = pca_scores(X, n_components=1).ravel()
    return robust_unit(z)


def pc1_to_hex(X: np.ndarray, cmap_name: str = CMAP_NAME) -> list[str]:
    """First PCA component → colormap hex (default cividis)."""
    return cmap_to_hex(pc1_unit(X), cmap_name)


def write_colormaps_json(out_dir: Path | None = None, n: int = 256) -> Path:
    """Write 256-stop LUTs for UI colormap switching."""
    out_dir = out_dir or OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    luts: dict[str, list[str]] = {}
    xs = np.linspace(0.0, 1.0, n)
    for name in CMAP_CHOICES:
        luts[name] = cmap_to_hex(xs, name)
    path = out_dir / "colormaps.json"
    path.write_text(json.dumps(luts))
    return path


def cell_props(r) -> dict:
    return {
        "id": r["id"],
        "ae_hex": r.get("ae_hex") or r.get("ae_hex_umap") or "#888888",
        "ae_hex_umap": r.get("ae_hex_umap") or r.get("ae_hex") or "#888888",
        "sim_hex": r.get("sim_hex") or "#888888",
        "cnn_hex": r.get("cnn_hex") or r.get("cnn_hex_umap") or "#888888",
        "cnn_hex_umap": r.get("cnn_hex_umap") or r.get("cnn_hex") or "#888888",
        "ae_u": float(r["ae_u"]) if pd.notna(r.get("ae_u")) else None,
        "sim_u": float(r["sim_u"]) if pd.notna(r.get("sim_u")) else None,
        "cnn_u": float(r["cnn_u"]) if pd.notna(r.get("cnn_u")) else None,
        "color_mode": r.get("color_mode") or "umap_cmap",
        "sim": float(r["sim"]) if pd.notna(r.get("sim")) else None,
        "i": int(r["i"]),
        "j": int(r["j"]),
        "y_res": int(r["y_res"]) if pd.notna(r.get("y_res")) else None,
        "label": r.get("label"),
        "utm_x": float(r["utm_x"]),
        "utm_y": float(r["utm_y"]),
        "crs": str(r["crs"]),
        "inside_polygon": bool(r["inside_polygon"]) if pd.notna(r.get("inside_polygon")) else None,
        "misclassified_multi": bool(r["misclassified_multi"])
        if pd.notna(r.get("misclassified_multi"))
        else False,
        "misclassified_res": bool(r["misclassified_res"])
        if pd.notna(r.get("misclassified_res"))
        else False,
        "error_res": (r.get("error_res") if pd.notna(r.get("error_res")) else "correct") or "correct",
        "pred_label": r.get("pred_label") if pd.notna(r.get("pred_label")) else None,
    }


def render_ghsl_png(
    west: float,
    south: float,
    east: float,
    north: float,
    out_png: Path,
    mosaic,
    mask_geoms: list | None = None,
    max_dim: int = GHSL_MAX_DIM,
) -> dict:
    """Nearest-neighbor warp of GHS-BUILT-C FUN → WGS84 RGBA PNG for Leaflet.

    If ``mask_geoms`` is provided (typically enclosing bboxes), pixels outside
    those polygons are fully transparent so GHSL only shows inside the boxes.
    """
    from rasterio import features as rio_features

    to_m = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
    corners = [(west, south), (east, south), (east, north), (west, north)]
    xy = [to_m.transform(x, y) for x, y in corners]
    minx, maxx = min(x for x, _ in xy), max(x for x, _ in xy)
    miny, maxy = min(y for _, y in xy), max(y for _, y in xy)
    data, _window, src_transform, _ds = mosaic.read_window(minx, miny, maxx, maxy, pad_px=4)

    aspect = (east - west) / max(north - south, 1e-12)
    if aspect >= 1:
        out_w = max_dim
        out_h = max(2, int(round(max_dim / aspect)))
    else:
        out_h = max_dim
        out_w = max(2, int(round(max_dim * aspect)))

    dst = np.full((out_h, out_w), 255, dtype=np.uint8)
    dst_transform = transform_from_bounds(west, south, east, north, out_w, out_h)
    reproject(
        source=data,
        destination=dst,
        src_transform=src_transform,
        src_crs="ESRI:54009",
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        src_nodata=255,
        dst_nodata=255,
        resampling=Resampling.nearest,
    )

    rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    for code, color in GHSL_RGBA.items():
        rgba[dst == code] = color
    known = (dst == 0) | (dst == 1) | (dst == 2) | (dst == 255)
    rgba[~known] = (0, 0, 0, 0)

    n_mask_geoms = 0
    if mask_geoms:
        n_mask_geoms = len(mask_geoms)
        mask = rio_features.rasterize(
            [(g, 1) for g in mask_geoms if g is not None and not g.is_empty],
            out_shape=(out_h, out_w),
            transform=dst_transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )
        rgba[mask == 0, 3] = 0

    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(out_png, optimize=True)
    visible = dst if not mask_geoms else dst[mask == 1]
    u, c = np.unique(visible, return_counts=True)
    return {
        "png": out_png.name,
        "west": west,
        "south": south,
        "east": east,
        "north": north,
        "width": out_w,
        "height": out_h,
        "n_mask_geoms": n_mask_geoms,
        "value_counts": {str(int(k)): int(v) for k, v in zip(u.tolist(), c.tolist())},
    }


def load_enclosing_bboxes(bbpath: Path) -> tuple[list, tuple[float, float, float, float] | None]:
    """Load axis-aligned enclosing bbox polygons + overall bounds (west,south,east,north)."""
    from shapely.geometry import shape

    if not bbpath.exists():
        return [], None
    fc = json.loads(bbpath.read_text())
    geoms = []
    for feat in fc.get("features", []):
        try:
            g = shape(feat["geometry"])
            if not g.is_empty:
                geoms.append(g)
        except Exception:
            continue
    if not geoms:
        return [], None
    minx = min(g.bounds[0] for g in geoms)
    miny = min(g.bounds[1] for g in geoms)
    maxx = max(g.bounds[2] for g in geoms)
    maxy = max(g.bounds[3] for g in geoms)
    return geoms, (minx, miny, maxx, maxy)


def load_ghsl_building_lookup() -> pd.DataFrame | None:
    if not GHSL_BUILDING.exists():
        return None
    pb = pd.read_parquet(GHSL_BUILDING)
    pb["building_id"] = pb["building_id"].astype(str)
    keep = [
        "building_id",
        "ghsl_prediction",
        "ghsl_covered",
        "frac_res",
        "frac_nres",
        "frac_nonbuilt",
        "n_pixels",
    ]
    keep = [c for c in keep if c in pb.columns]
    return pb[keep].drop_duplicates("building_id")


def enrich_buildings_geojson_with_ghsl(bpath: Path, lookup: pd.DataFrame) -> tuple[int, int]:
    """Patch buildings_*.geojson with GHSL labels + CNN-vs-GHSL disagreement flag.

    Returns (n_enriched, n_disagree).
    """
    fc = json.loads(bpath.read_text())
    by_id = lookup.set_index("building_id").to_dict(orient="index")
    n = 0
    n_disagree = 0
    for feat in fc.get("features", []):
        props = feat.setdefault("properties", {})
        bid = str(props.get("id", ""))
        row = by_id.get(bid)
        if not row:
            props.setdefault("ghsl_prediction", None)
            props.setdefault("ghsl_covered", False)
            props["ghsl_cnn_disagree"] = False
            continue
        ghsl = row.get("ghsl_prediction")
        covered = bool(row.get("ghsl_covered"))
        props["ghsl_prediction"] = ghsl
        props["ghsl_covered"] = covered
        props["ghsl_frac_res"] = float(row["frac_res"]) if row.get("frac_res") is not None else None
        props["ghsl_frac_nres"] = float(row["frac_nres"]) if row.get("frac_nres") is not None else None
        props["ghsl_n_pixels"] = float(row["n_pixels"]) if row.get("n_pixels") is not None else None

        pred_res = props.get("pred_res")
        cnn = None
        try:
            if pred_res is not None and pred_res == pred_res:
                cnn = "RES" if int(pred_res) == 1 else "NRES"
        except (TypeError, ValueError):
            cnn = None
        disagree = bool(covered and cnn is not None and ghsl in ("RES", "NRES") and ghsl != cnn)
        props["ghsl_cnn_disagree"] = disagree
        props["cnn_res_label"] = cnn
        if disagree:
            n_disagree += 1
        n += 1
    bpath.write_text(json.dumps(fc))
    return n, n_disagree


def enrich_bboxes_geojson_with_ghsl(bbpath: Path, building_props_by_id: dict[str, dict]) -> int:
    """Copy GHSL / disagreement flags onto enclosing-bbox features."""
    if not bbpath.exists():
        return 0
    fc = json.loads(bbpath.read_text())
    n = 0
    for feat in fc.get("features", []):
        props = feat.setdefault("properties", {})
        bid = str(props.get("id", ""))
        src = building_props_by_id.get(bid)
        if not src:
            props["ghsl_cnn_disagree"] = False
            continue
        for k in (
            "ghsl_prediction",
            "ghsl_covered",
            "ghsl_cnn_disagree",
            "cnn_res_label",
            "pred_res",
            "y_res",
        ):
            if k in src:
                props[k] = src[k]
        n += 1
    bbpath.write_text(json.dumps(fc))
    return n


def building_props_index(bpath: Path) -> dict[str, dict]:
    fc = json.loads(bpath.read_text())
    out: dict[str, dict] = {}
    for feat in fc.get("features", []):
        p = feat.get("properties") or {}
        bid = str(p.get("id", ""))
        if bid:
            out[bid] = p
    return out


def write_ghsl_layers(out_dir: Path | None = None) -> list[dict]:
    """Render per-region GHSL PNGs, enrich buildings geojson, update regions.json + HTML."""
    from validation.ghsl import open_ghsl
    from validation.io import load_config

    out_dir = out_dir or OUT
    regions_path = out_dir / "regions.json"
    if not regions_path.exists():
        raise FileNotFoundError(f"{regions_path} missing — run a full build first")

    regions = json.loads(regions_path.read_text())
    cfg = load_config()
    mosaic = open_ghsl(cfg)
    lookup = load_ghsl_building_lookup()
    if lookup is None:
        print("WARNING: no per_building_predictions.parquet — building GHSL popups limited", flush=True)

    try:
        for meta in regions:
            region = meta["region"]
            bb_name = meta.get("bboxes") or f"bboxes_{region}.geojson"
            bpath = out_dir / meta["buildings"]

            # Enrich buildings/bboxes first so we know which boxes disagree
            n_dis = 0
            if lookup is not None and bpath.exists():
                n, n_dis = enrich_buildings_geojson_with_ghsl(bpath, lookup)
                meta["n_ghsl_cnn_disagree"] = n_dis
                n_bb = enrich_bboxes_geojson_with_ghsl(
                    out_dir / bb_name, building_props_index(bpath)
                )
                print(
                    f"  enriched {n}/{meta['n_buildings']} buildings "
                    f"({n_dis} GHSL↔CNN disagree) · {n_bb} bboxes",
                    flush=True,
                )
            else:
                meta["n_ghsl_cnn_disagree"] = 0

            mask_geoms, bbox_bounds = load_enclosing_bboxes(out_dir / bb_name)
            if bbox_bounds is not None:
                west, south, east, north = bbox_bounds
                pad = 1e-5
                west, south, east, north = west - pad, south - pad, east + pad, north + pad
            else:
                west = float(meta["west"]) - GHSL_PAD_DEG
                south = float(meta["south"]) - GHSL_PAD_DEG
                east = float(meta["east"]) + GHSL_PAD_DEG
                north = float(meta["north"]) + GHSL_PAD_DEG
                print(f"  WARNING: no enclosing bboxes for {region}; using metro bounds", flush=True)

            png_path = out_dir / f"ghsl_{region}.png"
            print(
                f"GHSL overlay {region} → {png_path.name} "
                f"(masked to {len(mask_geoms)} enclosing bboxes)",
                flush=True,
            )
            info = render_ghsl_png(
                west, south, east, north, png_path, mosaic, mask_geoms=mask_geoms or None
            )
            meta["ghsl"] = info["png"]
            meta["ghsl_west"] = info["west"]
            meta["ghsl_south"] = info["south"]
            meta["ghsl_east"] = info["east"]
            meta["ghsl_north"] = info["north"]
            meta["ghsl_value_counts"] = info["value_counts"]
            meta["ghsl_masked_to_bboxes"] = bool(mask_geoms)
            meta["ghsl_n_bboxes"] = int(info.get("n_mask_geoms") or 0)

            # Second overlay: only enclosing bboxes where GHSL ≠ CNN
            disagree_geoms = []
            if (out_dir / bb_name).exists():
                from shapely.geometry import shape

                bb_fc = json.loads((out_dir / bb_name).read_text())
                for feat in bb_fc.get("features", []):
                    if not (feat.get("properties") or {}).get("ghsl_cnn_disagree"):
                        continue
                    try:
                        g = shape(feat["geometry"])
                        if not g.is_empty:
                            disagree_geoms.append(g)
                    except Exception:
                        continue
            dis_png = out_dir / f"ghsl_disagree_{region}.png"
            if disagree_geoms:
                print(
                    f"  disagree overlay → {dis_png.name} ({len(disagree_geoms)} bboxes)",
                    flush=True,
                )
                render_ghsl_png(
                    west, south, east, north, dis_png, mosaic, mask_geoms=disagree_geoms
                )
                meta["ghsl_disagree"] = dis_png.name
            else:
                # empty transparent placeholder keeps UI simple
                Image.new("RGBA", (8, 8), (0, 0, 0, 0)).save(dis_png)
                meta["ghsl_disagree"] = dis_png.name
    finally:
        mosaic.close()

    regions_path.write_text(json.dumps(regions, indent=2))
    write_colormaps_json(out_dir)
    write_html(out_dir / "index.html")
    print("Updated GHSL layers +", out_dir / "index.html", flush=True)
    return regions


def _jsonable(v):
    if v is None:
        return None
    if isinstance(v, (list, dict, str, bool, int, float)):
        return v
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    try:
        if pd.isna(v):
            return None
    except (ValueError, TypeError):
        pass
    # numpy / pandas containers
    if hasattr(v, "tolist"):
        try:
            return v.tolist()
        except Exception:
            pass
    return str(v)


def building_feature_props(r, region: str | None = None) -> dict:
    """Overture + GT + model fields for map popups."""
    y_res = int(r["y_res"]) if pd.notna(r.get("y_res")) else None
    props = {
        "id": str(r["id"]),
        "region": region if region is not None else (r.get("region") if pd.notna(r.get("region")) else None),
        # Ground truth (resi_def_v1)
        "y_res": y_res,
        "gt_res_label": None if y_res is None else ("RES" if y_res == 1 else "NRES"),
        "label": r.get("label") if pd.notna(r.get("label")) else None,
        "brief_category": r.get("brief_category") if pd.notna(r.get("brief_category")) else None,
        "label_source": r.get("label_source") if pd.notna(r.get("label_source")) else None,
        # Overture Buildings metadata
        "overture_class": r.get("class") if pd.notna(r.get("class")) else None,
        "overture_subtype": r.get("subtype") if pd.notna(r.get("subtype")) else None,
        "height_m": float(r["height"]) if pd.notna(r.get("height")) else None,
        "num_floors": int(r["num_floors"]) if pd.notna(r.get("num_floors")) else None,
        "area_m2": float(r["area_m2_approx"]) if pd.notna(r.get("area_m2_approx")) else None,
        "source_primary": _jsonable(r.get("source_primary")),
        # Geometry / sample
        "lon": float(r["lon"]) if pd.notna(r.get("lon")) else None,
        "lat": float(r["lat"]) if pd.notna(r.get("lat")) else None,
        "n_cells": int(r["n_cells"]) if pd.notna(r.get("n_cells")) else 0,
        "embed_year": int(r["embed_year"]) if pd.notna(r.get("embed_year")) else None,
        "sample_method": r.get("sample_method") if pd.notna(r.get("sample_method")) else None,
    }
    if hasattr(r, "get") and pd.notna(r.get("cnn_u")):
        props["cnn_u"] = float(r.get("cnn_u"))
    elif "cnn_u" in getattr(r, "index", []) and pd.notna(r["cnn_u"]):
        props["cnn_u"] = float(r["cnn_u"])
    else:
        props["cnn_u"] = None
    if "cnn_hex" in getattr(r, "index", []) or (hasattr(r, "get") and r.get("cnn_hex") is not None):
        props["cnn_hex"] = (r.get("cnn_hex") if hasattr(r, "get") else r["cnn_hex"]) or "#888888"
    for key, cast in (
        ("pred_label", str),
        ("pred_prob", float),
        ("pred_res", int),
        ("misclassified_multi", bool),
        ("misclassified_res", bool),
        ("error_res", str),
        ("error_multi", str),
    ):
        if key not in getattr(r, "index", []) and not (hasattr(r, "get") and key in r):
            continue
        val = r.get(key) if hasattr(r, "get") else r[key]
        if pd.isna(val):
            props[key] = False if cast is bool else None
        elif cast is bool:
            props[key] = bool(val)
        elif cast is str:
            props[key] = str(val) if val is not None else None
        else:
            props[key] = cast(val)
    props.setdefault("error_res", "correct")
    props.setdefault("error_multi", "correct")
    props.setdefault("misclassified_multi", False)
    props.setdefault("misclassified_res", False)
    props.setdefault("cnn_hex", "#888888")
    return props


def enrich_buildings_overture_metadata(out_dir: Path | None = None) -> None:
    """Patch existing buildings_*.geojson with Overture / GT fields from parquet."""
    out_dir = out_dir or OUT
    src = pd.read_parquet(BBOX / "buildings.parquet")
    src["id"] = src["id"].astype(str)
    by_id = {str(r.id): r for _, r in src.iterrows()}
    for bpath in sorted(out_dir.glob("buildings_*.geojson")):
        fc = json.loads(bpath.read_text())
        n = 0
        for feat in fc.get("features", []):
            props = feat.setdefault("properties", {})
            row = by_id.get(str(props.get("id", "")))
            if row is None:
                continue
            meta = building_feature_props(row, region=props.get("region") or row.get("region"))
            # keep existing model/GHSL fields; fill Overture/GT
            for k, v in meta.items():
                if k in (
                    "cnn_hex",
                    "cnn_u",
                    "pred_label",
                    "pred_prob",
                    "pred_res",
                    "misclassified_multi",
                    "misclassified_res",
                    "error_res",
                    "error_multi",
                ) and k in props and props[k] is not None:
                    continue
                props[k] = v
            n += 1
        bpath.write_text(json.dumps(fc))
        print(f"  Overture/GT metadata → {bpath.name} ({n})", flush=True)


def _ae_region_manifold_job(payload: tuple) -> tuple[str, dict]:
    """Worker: (region, X, methods) → {method: (hexes, u0)}."""
    region, X, methods = payload
    X = np.asarray(X, dtype=np.float64)
    out: dict = {}
    for method in methods:
        hexes, Zu = rgb_hex_from_manifold(X, method=method)
        out[method] = (hexes, Zu[:, 0].astype(np.float64))
    return region, out


def patch_color_units(
    out_dir: Path | None = None,
    methods: tuple[str, ...] = MANIFOLD_METHODS,
) -> None:
    """Recompute UMAP→colormap for AE cells (per region) + CNN buildings; patch geojson."""
    out_dir = out_dir or OUT
    methods = tuple(m for m in methods if m in MANIFOLD_METHODS) or (MANIFOLD_DEFAULT,)
    cells = pd.read_parquet(BBOX / "building_cells.parquet")
    emb = pd.read_parquet(EMB)
    cells["id"] = cells["id"].astype(str)
    emb["id"] = emb["id"].astype(str)
    if "region" not in cells.columns:
        bmeta = pd.read_parquet(BBOX / "buildings.parquet", columns=["id", "region"])
        bmeta["id"] = bmeta["id"].astype(str)
        cells = cells.merge(bmeta, on="id", how="left")

    print(f"computing CNN manifold colormap methods={methods} (n={len(emb)})…", flush=True)
    emb = emb.copy()
    Xp = emb[ECOLS].to_numpy(dtype=np.float64)
    for method in methods:
        print(f"  CNN {method}…", flush=True)
        hexes, Zu = rgb_hex_from_manifold(Xp, method=method)
        emb[f"cnn_hex_{method}"] = hexes
        if method == MANIFOLD_DEFAULT:
            emb["cnn_hex"] = hexes
            emb["cnn_u"] = Zu[:, 0]
    if "cnn_hex" not in emb.columns:
        emb["cnn_hex"] = emb[f"cnn_hex_{methods[0]}"]
        emb["cnn_u"] = 0.5
    keep_cnn = ["cnn_u", "cnn_hex"] + [f"cnn_hex_{m}" for m in methods]
    cnn_by_id = emb.set_index("id")[keep_cnn].to_dict(orient="index")

    print("computing per-building cosine similarity…", flush=True)
    cells = cells.copy()
    Xc_all = cells[BANDS].to_numpy(dtype=np.float64)
    means = cells.groupby("id", sort=False)[BANDS].transform("mean").to_numpy(dtype=np.float64)
    num = (Xc_all * means).sum(axis=1)
    den = np.linalg.norm(Xc_all, axis=1) * np.linalg.norm(means, axis=1) + 1e-12
    cells["sim"] = num / den
    cells["sim_u"] = robust_unit(cells["sim"].to_numpy(dtype=np.float64))
    cells["sim_hex"] = cmap_to_hex(cells["sim_u"].to_numpy(dtype=np.float64))

    cells["ae_hex_umap"] = "#888888"
    cells["ae_hex"] = "#888888"
    cells["ae_u"] = 0.5

    # Serial per-region fits: ProcessPool after OpenMP/UMAP has loaded is unsafe
    # ("fork() called from a process already using GNU OpenMP").
    for region, csub in cells.groupby("region", sort=False):
        idx = csub.index.to_numpy()
        X = csub[BANDS].to_numpy(dtype=np.float64)
        print(f"  AE {region}: n={len(csub)} methods={methods}", flush=True)
        for method in methods:
            hexes, Zu = rgb_hex_from_manifold(X, method=method)
            cells.loc[idx, f"ae_hex_{method}"] = hexes
            if method == MANIFOLD_DEFAULT:
                cells.loc[idx, "ae_hex"] = hexes
                cells.loc[idx, "ae_u"] = Zu[:, 0]

    cells["cnn_u"] = cells["id"].map(lambda i: cnn_by_id.get(i, {}).get("cnn_u"))
    cells["cnn_hex"] = cells["id"].map(lambda i: cnn_by_id.get(i, {}).get("cnn_hex") or "#888888")
    for method in methods:
        cells[f"cnn_hex_{method}"] = cells["id"].map(
            lambda i, m=method: cnn_by_id.get(i, {}).get(f"cnn_hex_{m}") or "#888888"
        )

    cols = [
        "id", "i", "j", "ae_u", "ae_hex", "sim", "sim_u", "sim_hex", "cnn_u", "cnn_hex",
        "ae_hex_umap", "cnn_hex_umap",
    ]
    cell_lookup = {
        (str(r.id), int(r.i), int(r.j)): r
        for _, r in cells[[c for c in cols if c in cells.columns]].iterrows()
    }

    for bpath in sorted(out_dir.glob("buildings_*.geojson")):
        fc = json.loads(bpath.read_text())
        n = 0
        for feat in fc.get("features", []):
            props = feat.setdefault("properties", {})
            row = cnn_by_id.get(str(props.get("id", "")))
            if not row:
                continue
            props["cnn_u"] = float(row["cnn_u"]) if row.get("cnn_u") is not None else None
            props["cnn_hex"] = row.get("cnn_hex") or props.get("cnn_hex") or "#888888"
            for method in methods:
                props[f"cnn_hex_{method}"] = row.get(f"cnn_hex_{method}") or props["cnn_hex"]
            props["color_mode"] = "umap_cmap"
            n += 1
        bpath.write_text(json.dumps(fc))
        print(f"  UMAP RGB → {bpath.name} ({n})", flush=True)

    for cpath in sorted(out_dir.glob("cells_*.geojson")):
        fc = json.loads(cpath.read_text())
        n = 0
        for feat in fc.get("features", []):
            props = feat.setdefault("properties", {})
            key = (str(props.get("id", "")), int(props.get("i", -1)), int(props.get("j", -1)))
            row = cell_lookup.get(key)
            if row is None:
                continue
            props["ae_u"] = float(row.ae_u) if pd.notna(row.ae_u) else None
            props["sim_u"] = float(row.sim_u) if pd.notna(row.sim_u) else None
            props["cnn_u"] = float(row.cnn_u) if pd.notna(row.cnn_u) else None
            props["sim"] = float(row.sim) if pd.notna(row.sim) else None
            props["ae_hex"] = row.ae_hex or "#888888"
            props["sim_hex"] = row.sim_hex or "#888888"
            props["cnn_hex"] = row.cnn_hex or "#888888"
            props["ae_hex_umap"] = getattr(row, "ae_hex_umap", None) or props["ae_hex"]
            props["cnn_hex_umap"] = getattr(row, "cnn_hex_umap", None) or props["cnn_hex"]
            props["color_mode"] = "umap_cmap"
            n += 1
        cpath.write_text(json.dumps(fc))
        print(f"  manifold colors → {cpath.name} ({n})", flush=True)

    write_colormaps_json(out_dir)
    write_html(out_dir / "index.html")
    print("Updated UMAP→colormap colors + HTML", flush=True)


def rebake_hex_from_units(out_dir: Path | None = None, cmap_name: str = CMAP_NAME) -> None:
    """Remap existing ae_u/cnn_u/sim_u → default-colormap hex (no UMAP recompute) + refresh HTML."""
    out_dir = out_dir or OUT
    print(f"Rebaking hex from units with {cmap_name}…", flush=True)

    def _hex_from_u(u) -> str | None:
        if u is None:
            return None
        try:
            v = float(u)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(v):
            return None
        return cmap_to_hex(np.array([v], dtype=np.float64), cmap_name)[0]

    for bpath in sorted(out_dir.glob("buildings_*.geojson")):
        fc = json.loads(bpath.read_text())
        n = 0
        for feat in fc.get("features", []):
            props = feat.setdefault("properties", {})
            hx = _hex_from_u(props.get("cnn_u"))
            if hx:
                props["cnn_hex"] = hx
                props["cnn_hex_umap"] = hx
                props["color_mode"] = "umap_cmap"
                n += 1
        bpath.write_text(json.dumps(fc))
        print(f"  buildings hex → {bpath.name} ({n})", flush=True)

    for cpath in sorted(out_dir.glob("cells_*.geojson")):
        fc = json.loads(cpath.read_text())
        n = 0
        for feat in fc.get("features", []):
            props = feat.setdefault("properties", {})
            ae = _hex_from_u(props.get("ae_u"))
            if ae:
                props["ae_hex"] = ae
                props["ae_hex_umap"] = ae
            cnn = _hex_from_u(props.get("cnn_u"))
            if cnn:
                props["cnn_hex"] = cnn
                props["cnn_hex_umap"] = cnn
            sim = _hex_from_u(props.get("sim_u"))
            if sim:
                props["sim_hex"] = sim
            props["color_mode"] = "umap_cmap"
            n += 1
        cpath.write_text(json.dumps(fc))
        print(f"  cells hex → {cpath.name} ({n})", flush=True)

    write_colormaps_json(out_dir)
    write_html(out_dir / "index.html")
    print("Rebaked colormap hex + HTML", flush=True)


def shapely_to_coords(geom) -> dict:
    g = geom
    if g.geom_type == "Polygon":
        rings = [list(g.exterior.coords)]
        rings += [list(r.coords) for r in g.interiors]
        return {"type": "Polygon", "coordinates": [[[float(x), float(y)] for x, y in ring] for ring in rings]}
    if g.geom_type == "MultiPolygon":
        polys = []
        for poly in g.geoms:
            rings = [list(poly.exterior.coords)]
            rings += [list(r.coords) for r in poly.interiors]
            polys.append([[[float(x), float(y)] for x, y in ring] for ring in rings])
        return {"type": "MultiPolygon", "coordinates": polys}
    # fallback centroid point
    return {"type": "Point", "coordinates": [float(g.centroid.x), float(g.centroid.y)]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    buildings = pd.read_parquet(BBOX / "buildings.parquet")
    cells = pd.read_parquet(BBOX / "building_cells.parquet")
    emb = pd.read_parquet(EMB)
    buildings["id"] = buildings["id"].astype(str)
    cells["id"] = cells["id"].astype(str)
    emb["id"] = emb["id"].astype(str)

    # CNN building fill: UMAP/t-SNE → RGB (default UMAP)
    print(f"CNN {MANIFOLD_DEFAULT}→colormap…", flush=True)
    Xp = emb[ECOLS].to_numpy(dtype=np.float64)
    emb = emb.copy()
    cnn_hex, Zcnn = rgb_hex_from_manifold(Xp, method=MANIFOLD_DEFAULT)
    emb["cnn_hex"] = cnn_hex
    emb["cnn_hex_umap"] = cnn_hex
    emb["cnn_u"] = Zcnn[:, 0]

    buildings = buildings.merge(
        emb[["id", "cnn_hex", "cnn_u", "cnn_hex_umap", "y_res"]],
        on="id",
        how="left",
        suffixes=("", "_e"),
    )
    if "y_res_e" in buildings.columns:
        buildings["y_res"] = buildings["y_res"].fillna(buildings["y_res_e"])

    if PRED.exists():
        preds = pd.read_parquet(PRED)
        preds["id"] = preds["id"].astype(str)
        keep = [
            "id",
            "pred_label",
            "pred_prob",
            "pred_res",
            "misclassified_multi",
            "misclassified_res",
            "error_res",
            "error_multi",
        ]
        keep = [c for c in keep if c in preds.columns]
        buildings = buildings.merge(preds[keep], on="id", how="left")
    else:
        print("WARNING: missing cnn_oof_predictions.parquet — misclassified toggle disabled", flush=True)
        buildings["pred_label"] = None
        buildings["pred_prob"] = np.nan
        buildings["pred_res"] = np.nan
        buildings["misclassified_multi"] = False
        buildings["misclassified_res"] = False
        buildings["error_res"] = "correct"
        buildings["error_multi"] = "correct"

    # Cell colors: per-region UMAP → RGB + similarity colormap units
    print(f"AE per-region {MANIFOLD_DEFAULT}→colormap…", flush=True)
    cells = cells.copy()
    cells["ae_hex"] = "#888888"
    cells["ae_hex_umap"] = "#888888"
    cells["ae_u"] = 0.5
    Xc = cells[BANDS].to_numpy(dtype=np.float64)
    for region, csub in cells.groupby("region", sort=False):
        idx = csub.index.to_numpy()
        X = csub[BANDS].to_numpy(dtype=np.float64)
        print(f"  AE {region} n={len(csub)}", flush=True)
        hexes, Zu = rgb_hex_from_manifold(X, method=MANIFOLD_DEFAULT)
        cells.loc[idx, "ae_hex_umap"] = hexes
        cells.loc[idx, "ae_hex"] = hexes
        cells.loc[idx, "ae_u"] = Zu[:, 0]

    # Cell color B: cosine similarity to building mean → unit + hex
    print("per-building cosine-to-mean…", flush=True)
    means = cells.groupby("id", sort=False)[BANDS].transform("mean").to_numpy(dtype=np.float64)
    num = (Xc * means).sum(axis=1)
    den = np.linalg.norm(Xc, axis=1) * np.linalg.norm(means, axis=1) + 1e-12
    cells["sim"] = num / den
    cells["sim_u"] = robust_unit(cells["sim"].to_numpy(dtype=np.float64))
    cells["sim_hex"] = cmap_to_hex(cells["sim_u"].to_numpy(dtype=np.float64))

    cells = cells.merge(
        buildings[
            [
                "id",
                "cnn_hex",
                "cnn_u",
                "cnn_hex_umap",
                "label",
                "y_res",
                "region",
                "misclassified_multi",
                "misclassified_res",
                "error_res",
                "pred_label",
            ]
        ],
        on="id",
        how="left",
    )
    region_meta = []
    for region, bsub in buildings.groupby("region"):
        print(f"export {region} · buildings={len(bsub)}", flush=True)
        feats = []
        for _, r in bsub.iterrows():
            try:
                g = shapely_wkt.loads(r["wkt"])
                if not g.is_valid:
                    g = g.buffer(0)
                geom = shapely_to_coords(g)
            except Exception:
                continue
            feats.append(
                {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": building_feature_props(r, region=region),
                }
            )
        bpath = OUT / f"buildings_{region}.geojson"
        bpath.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))

        # Axis-aligned enclosing rectangles (same WGS84 bounds used for cell collection)
        bboxes = []
        for _, r in bsub.iterrows():
            if not all(pd.notna(r.get(k)) for k in ("bbox_west", "bbox_south", "bbox_east", "bbox_north")):
                continue
            w, s, e, n = map(float, (r["bbox_west"], r["bbox_south"], r["bbox_east"], r["bbox_north"]))
            bboxes.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
                    },
                    "properties": {
                        "id": r["id"],
                        "label": r.get("label"),
                        "misclassified_multi": bool(r["misclassified_multi"])
                        if pd.notna(r.get("misclassified_multi"))
                        else False,
                        "misclassified_res": bool(r["misclassified_res"])
                        if pd.notna(r.get("misclassified_res"))
                        else False,
                        "error_res": (r.get("error_res") if pd.notna(r.get("error_res")) else "correct")
                        or "correct",
                    },
                }
            )
        bbpath = OUT / f"bboxes_{region}.geojson"
        bbpath.write_text(json.dumps({"type": "FeatureCollection", "features": bboxes}))

        n_err_res = int(bsub["misclassified_res"].fillna(False).sum())
        n_err_multi = int(bsub["misclassified_multi"].fillna(False).sum())
        csub = cells[cells["region"] == region]
        print(f"  cells={len(csub)}", flush=True)
        cfeats = []
        use_wkt = "pixel_wkt" in csub.columns and csub["pixel_wkt"].notna().any()
        if use_wkt:
            for _, r in csub.iterrows():
                try:
                    geom = shapely_to_coords(shapely_wkt.loads(r["pixel_wkt"]))
                except Exception:
                    continue
                cfeats.append(
                    {
                        "type": "Feature",
                        "geometry": geom,
                        "properties": cell_props(r),
                    }
                )
        else:
            for crs, ccrs in csub.groupby("crs", sort=False):
                to_wgs = Transformer.from_crs(str(crs), "EPSG:4326", always_xy=True)
                for _, r in ccrs.iterrows():
                    ux, uy = float(r["utm_x"]), float(r["utm_y"])
                    geom_utm = box(ux - HALF_M, uy - HALF_M, ux + HALF_M, uy + HALF_M)
                    geom = shapely_transform(to_wgs.transform, geom_utm)
                    props = cell_props(r)
                    props["utm_x"] = ux
                    props["utm_y"] = uy
                    props["crs"] = str(crs)
                    cfeats.append(
                        {
                            "type": "Feature",
                            "geometry": shapely_to_coords(geom),
                            "properties": props,
                        }
                    )
        cpath = OUT / f"cells_{region}.geojson"
        cpath.write_text(json.dumps({"type": "FeatureCollection", "features": cfeats}))

        region_meta.append(
            {
                "region": region,
                "n_buildings": len(feats),
                "n_cells": len(cfeats),
                "n_err_res": n_err_res,
                "n_err_multi": n_err_multi,
                "west": float(bsub["lon"].min()),
                "south": float(bsub["lat"].min()),
                "east": float(bsub["lon"].max()),
                "north": float(bsub["lat"].max()),
                "lon": float(bsub["lon"].mean()),
                "lat": float(bsub["lat"].mean()),
                "buildings": bpath.name,
                "bboxes": bbpath.name,
                "cells": cpath.name,
            }
        )

    (OUT / "regions.json").write_text(json.dumps(region_meta, indent=2))
    write_colormaps_json(OUT)
    write_html(OUT / "index.html")
    print("Wrote", OUT, flush=True)
    write_ghsl_layers(OUT)


def write_html(path: Path) -> None:
    path.write_text(
        """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Overture × AlphaEarth × CNN × GHSL overlay</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet-control-geocoder@2.4.0/dist/Control.Geocoder.css"/>
<style>
  html, body, #map { height: 100%; margin: 0; }
  body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  .panel {
    position: absolute; z-index: 1000; top: 12px; left: 12px;
    background: rgba(255,255,255,0.96); border: 1px solid #ddd; border-radius: 8px;
    padding: 12px 14px; width: 320px; max-height: calc(100% - 24px); overflow: auto;
    box-shadow: 0 1px 4px rgba(0,0,0,0.12);
  }
  .panel h1 { font-size: 15px; margin: 0 0 8px; font-weight: 650; }
  .panel h2 { font-size: 12px; margin: 12px 0 6px; color: #444; text-transform: uppercase; letter-spacing: 0.04em; }
  .row { display: flex; gap: 8px; align-items: center; margin: 6px 0; font-size: 13px; }
  .row label { flex: 1; }
  input[type="text"], select {
    width: 100%; box-sizing: border-box; padding: 7px 8px; border: 1px solid #ccc; border-radius: 6px;
    font-size: 13px;
  }
  button {
    border: 1px solid #bbb; background: #f7f7f7; border-radius: 6px; padding: 7px 10px;
    cursor: pointer; font-size: 13px;
  }
  button.primary { background: #1f4e79; color: #fff; border-color: #1f4e79; }
  button:hover { filter: brightness(0.97); }
  .hint { font-size: 11px; color: #666; line-height: 1.35; margin-top: 4px; }
  .status { font-size: 12px; color: #333; margin-top: 8px; min-height: 1.2em; }
  .swatch { width: 14px; height: 14px; border-radius: 3px; border: 1px solid #999; display: inline-block; }
  .leaflet-container { background: #e8eef3; }
</style>
</head>
<body>
<div id="map"></div>
<div class="panel">
  <h1>Overture · AlphaEarth · CNN · GHSL</h1>
  <div class="hint">Toggle layers. Search an address (Nominatim) or jump to lat,lon. Pick a metro to load its footprints + 10&nbsp;m grids + GHS-BUILT-C FUN.</div>

  <h2>Region</h2>
  <select id="region"></select>
  <div class="row" style="margin-top:8px">
    <button class="primary" id="loadRegion" style="flex:1">Load region</button>
  </div>

  <h2>Search</h2>
  <input id="address" type="text" placeholder="Address (e.g. Fenway Park, Boston)"/>
  <div class="row">
    <button id="searchAddress" class="primary" style="flex:1">Search address</button>
  </div>
  <input id="latlon" type="text" placeholder="lat, lon  (e.g. 42.3467, -71.0972)" style="margin-top:6px"/>
  <div class="row">
    <button id="goLatLon" style="flex:1">Go to lat/lon</button>
  </div>
  <select id="contrastPreset" style="margin-top:8px">
    <option value="">RES vs NRES UMAP contrast (same metro)…</option>
  </select>
  <div class="hint" style="margin:4px 0 0">
    Within one metro: jump between nearby <b>residential</b> and <b>non-residential</b> clusters where CNN UMAP colors diverge (uses <b>inferno</b>).
  </div>

  <h2>Layers</h2>
  <div class="row"><input type="checkbox" id="lyrOsm" checked/><label for="lyrOsm">Basemap (OSM)</label></div>
  <div class="row"><input type="checkbox" id="lyrGhsl" checked/><label for="lyrGhsl">GHSL inside enclosing bboxes</label></div>
  <div class="row" style="gap:10px; align-items:center">
    <label for="ghslOpacity" style="flex:0 0 auto; font-size:12px; color:#555">GHSL opacity</label>
    <input id="ghslOpacity" type="range" min="0" max="100" value="65" style="flex:1"/>
    <span id="ghslOpacityVal" style="font-size:12px; width:2.5em; text-align:right">65%</span>
  </div>
  <div class="hint" style="margin:2px 0 8px">
    <span class="swatch" style="background:#27ae60"></span> RES (1)
    &nbsp;<span class="swatch" style="background:#e67e22"></span> NRES (2)
    &nbsp;<span class="swatch" style="background:#dce1e6"></span> non-built (0)
  </div>
  <div class="row"><input type="checkbox" id="lyrPoly" checked/><label for="lyrPoly">Overture building polygons</label></div>
  <div class="row"><input type="checkbox" id="lyrBbox"/><label for="lyrBbox">Enclosing bbox (axis-aligned)</label></div>
  <div class="row"><input type="checkbox" id="lyrCnn" checked/><label for="lyrCnn">CNN embedding color (UMAP + colormap)</label></div>
  <div class="row"><input type="checkbox" id="lyrGhslFill"/><label for="lyrGhslFill">Color buildings by GHSL majority</label></div>
  <div class="row"><input type="checkbox" id="lyrGhslCnnDisagree"/><label for="lyrGhslCnnDisagree">GHSL ↔ CNN disagree only</label></div>
  <div class="hint" style="margin:0 0 8px">
    Filters <b>buildings + enclosing bboxes</b> to GHSL≠CNN cases and clips the GHSL raster to those boxes.
    AlphaEarth cells / basemap stay on. Buildings draw <b>above</b> GHSL so CNN colors remain visible.
  </div>
  <div class="row"><input type="checkbox" id="lyrCells" checked/><label for="lyrCells">AlphaEarth 10 m grid cells</label></div>
  <select id="cellColor" style="margin:4px 0 6px">
    <option value="pca" selected>Cell color: AE UMAP + colormap</option>
    <option value="sim">Cell color: similarity to building mean</option>
    <option value="cnn">Cell color: building CNN UMAP + colormap</option>
  </select>
  <div class="hint" style="margin:0 0 6px">AE / CNN / similarity fills all use the colormap picker (UMAP 1st axis → [0,1] → LUT). Per-metro UMAP for AE.</div>
  <div class="row" style="gap:8px; align-items:center; margin:4px 0 6px">
    <label for="cmap" style="flex:0 0 auto; font-size:12px; color:#555">Colormap</label>
    <select id="cmap" style="flex:1; margin:0">
      <option value="cividis" selected>cividis</option>
      <option value="turbo">turbo</option>
      <option value="inferno">inferno</option>
    </select>
  </div>
  <div id="cmapRamp" style="height:12px; border-radius:3px; border:1px solid #ccc; margin:0 0 8px; background:linear-gradient(90deg,#00224e,#7d7c78,#fee838)"></div>
  <div class="row"><input type="checkbox" id="lyrResi"/><label for="lyrResi">Residential outline only (blue/orange)</label></div>

  <h2>Errors (LORO CNN)</h2>
  <select id="errTask">
    <option value="res" selected>Residential (sf+mf collapse)</option>
    <option value="multi">Building type (10-class)</option>
  </select>
  <div class="row"><input type="checkbox" id="lyrErrOnly"/><label for="lyrErrOnly">Show misclassified only</label></div>
  <div class="row"><input type="checkbox" id="lyrErrColor" checked/><label for="lyrErrColor">Color by error type</label></div>
  <div class="hint">
    FP = predicted residential, GT non-res (vermillion). FN = missed residential (blue).
    Multiclass = wrong type (purple). Click a polygon for true → pred.
    Continuous fills use the selected colormap (cividis / turbo / inferno).
  </div>

  <div class="status" id="status">Pick a region and click Load — or search an address (auto-loads nearest metro).</div>
  <div class="hint" style="margin-top:10px">
    Tip: zoom to ~building scale (level 16+) to see footprints. Search auto-loads the nearest metro AOI.
    CNN / AE continuous fills = UMAP 1st axis → selected colormap (cividis / turbo / inferno).
    Similarity mode = cosine to that building’s mean (same colormap picker).
    Bbox = axis-aligned WGS84 rectangle used to collect those cells (may extend past the footprint).
    GHSL = GHS-BUILT-C FUN 2018 @ 10&nbsp;m (nearest-neighbor warp to WGS84),
    clipped to each building’s enclosing bbox only. Building GHSL color = majority of RES/NRES pixels under the footprint.
  </div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet-control-geocoder@2.4.0/dist/Control.Geocoder.js"></script>
<script>
const map = L.map('map', { zoomControl: true }).setView([39.8, -98.5], 4);
// GHSL under vectors so CNN / Overture / AE cells stay visible
map.createPane('ghslPane');
map.getPane('ghslPane').style.zIndex = 350;
map.getPane('ghslPane').style.pointerEvents = 'none';
// Buildings above AE cells so polygon clicks reach Overture popups
map.createPane('buildingsPane');
map.getPane('buildingsPane').style.zIndex = 450;
const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 20,
  attribution: '&copy; OpenStreetMap'
}).addTo(map);

let regions = [];
let buildingsLayer = null;
let cellsLayer = null;
let bboxLayer = null;
let ghslLayer = null;
let ghslLayerUrl = null;
let disagreeIds = new Set();
let currentRegion = null;
let focusMarker = null;
let contrastMarkers = null;
let COLORMAPS = null;

// Within-metro RES vs NRES spots where CNN UMAP scalars diverge (same AOI).
// Colors: lower cnn_u → darker/blue on inferno; higher → yellow/red.
const CONTRAST_PRESETS = [
  {
    id: 'chicago_res_nres',
    region: 'Chicago_MW',
    label: 'Chicago · Bronzeville — RES vs NRES (~150 m)',
    note: 'Single-family (cooler UMAP) next to hotel/office/civic (warmer)',
    res: { lat: 41.83055, lon: -87.62214, title: 'RES · single-family', u: 0.38 },
    nres: { lat: 41.83000, lon: -87.62104, title: 'NRES · hotel/office/civic', u: 0.54 },
  },
  {
    id: 'atlanta_res_nres',
    region: 'Atlanta_SE',
    label: 'Atlanta · Inman Park edge — RES vs NRES (~180 m)',
    note: 'Housing (cooler) next to hotel/office/medical (warmer / yellower)',
    res: { lat: 33.76402, lon: -84.36572, title: 'RES · sf / mf', u: 0.82 },
    nres: { lat: 33.76314, lon: -84.36709, title: 'NRES · hotel/office/medical', u: 0.95 },
  },
  {
    id: 'nyc_res_nres',
    region: 'NYC_NE',
    label: 'NYC · Fort Greene — RES vs NRES (~450 m)',
    note: 'Housing (warmer UMAP here) vs industrial/education (cooler)',
    res: { lat: 40.68912, lon: -73.96581, title: 'RES · sf / mf', u: 0.74 },
    nres: { lat: 40.69052, lon: -73.96190, title: 'NRES · industrial/education', u: 0.61 },
  },
  {
    id: 'houston_res_nres',
    region: 'Houston_S',
    label: 'Houston · Near Northside — RES vs NRES (~550 m)',
    note: 'Single-family (cooler) vs civic/education/religious (warmer)',
    res: { lat: 29.80180, lon: -95.38024, title: 'RES · single-family', u: 0.22 },
    nres: { lat: 29.80020, lon: -95.37545, title: 'NRES · civic/education', u: 0.37 },
  },
  {
    id: 'sf_res_nres',
    region: 'SF_W',
    label: 'SF · Mission — RES vs NRES (~640 m)',
    note: 'Single-family (cooler) vs office/medical/retail (warmer)',
    res: { lat: 37.74376, lon: -122.42006, title: 'RES · single-family', u: 0.54 },
    nres: { lat: 37.74853, lon: -122.41687, title: 'NRES · office/medical/retail', u: 0.65 },
  },
];

const GHSL_FILL = { RES: '#27ae60', NRES: '#e67e22', NONBUILT: '#bdc3c7', NODATA: '#7f8c8d' };

const statusEl = document.getElementById('status');
function setStatus(msg) { statusEl.textContent = msg; }

function selectedCmap() {
  return document.getElementById('cmap').value || 'cividis';
}

function manifoldHex(p, kind) {
  // UMAP 1st axis → selected colormap (baked hex is cividis fallback)
  if (kind === 'ae') return colorFromUnit(p.ae_u, p.ae_hex_umap || p.ae_hex || '#888');
  return colorFromUnit(p.cnn_u, p.cnn_hex_umap || p.cnn_hex || '#888');
}

function colorFromUnit(u, fallbackHex) {
  if (u == null || !isFinite(Number(u)) || !COLORMAPS) return fallbackHex || '#888888';
  const lut = COLORMAPS[selectedCmap()] || COLORMAPS.cividis;
  if (!lut || !lut.length) return fallbackHex || '#888888';
  const i = Math.max(0, Math.min(lut.length - 1, Math.round(Number(u) * (lut.length - 1))));
  return lut[i];
}

function updateCmapRamp() {
  if (!COLORMAPS) return;
  const lut = COLORMAPS[selectedCmap()] || COLORMAPS.cividis;
  if (!lut || lut.length < 2) return;
  const stops = [0, 0.25, 0.5, 0.75, 1].map(t => {
    const i = Math.round(t * (lut.length - 1));
    return lut[i];
  });
  document.getElementById('cmapRamp').style.background =
    `linear-gradient(90deg, ${stops.join(', ')})`;
}

async function loadColormaps() {
  try {
    COLORMAPS = await (await fetch('colormaps.json')).json();
    updateCmapRamp();
  } catch (e) {
    COLORMAPS = null;
    setStatus('Colormap LUT missing — using baked hex colors. Run build_map_overlay.py --colors-only');
  }
}

function nearestRegion(lat, lon) {
  let best = null, bestD = Infinity;
  for (const r of regions) {
    const d = Math.hypot(r.lat - lat, r.lon - lon);
    if (d < bestD) { bestD = d; best = r; }
  }
  return { region: best, distDeg: bestD };
}

function markFocus(lat, lon, popupHtml) {
  if (focusMarker) map.removeLayer(focusMarker);
  focusMarker = L.marker([lat, lon]).addTo(map);
  if (popupHtml) focusMarker.bindPopup(popupHtml).openPopup();
}

function clearContrastMarkers() {
  if (contrastMarkers) {
    map.removeLayer(contrastMarkers);
    contrastMarkers = null;
  }
}

function fillContrastPresets() {
  const sel = document.getElementById('contrastPreset');
  // keep first placeholder
  while (sel.options.length > 1) sel.remove(1);
  for (const p of CONTRAST_PRESETS) {
    const opt = document.createElement('option');
    opt.value = p.id;
    opt.textContent = p.label;
    sel.appendChild(opt);
  }
}

async function goContrastPreset() {
  const id = document.getElementById('contrastPreset').value;
  if (!id) return;
  const p = CONTRAST_PRESETS.find(x => x.id === id);
  if (!p) return;

  // Show CNN UMAP with inferno so cooler/warmer is obvious
  document.getElementById('cmap').value = 'inferno';
  document.getElementById('lyrCnn').checked = true;
  document.getElementById('lyrPoly').checked = true;
  document.getElementById('lyrResi').checked = false;
  document.getElementById('lyrGhslFill').checked = false;
  document.getElementById('lyrErrOnly').checked = false;
  document.getElementById('lyrGhslCnnDisagree').checked = false;
  updateCmapRamp();

  document.getElementById('region').value = p.region;
  const midLat = (p.res.lat + p.nres.lat) / 2;
  const midLon = (p.res.lon + p.nres.lon) / 2;
  document.getElementById('latlon').value = `${midLat.toFixed(6)}, ${midLon.toFixed(6)}`;

  await loadRegion({ focus: [midLat, midLon], zoom: 17 });

  clearContrastMarkers();
  if (focusMarker) { map.removeLayer(focusMarker); focusMarker = null; }
  contrastMarkers = L.layerGroup().addTo(map);
  const resCooler = p.res.u <= p.nres.u;
  const resHint = resCooler ? 'cooler / darker on inferno' : 'warmer / yellower on inferno';
  const nresHint = resCooler ? 'warmer / yellower on inferno' : 'cooler / darker on inferno';
  const resM = L.marker([p.res.lat, p.res.lon], { title: p.res.title });
  resM.bindPopup(
    `<b>${p.res.title}</b><br/>GT residential<br/>CNN UMAP ≈ ${p.res.u.toFixed(2)}` +
    `<br/><span style="color:#666">${resHint}</span>`
  );
  const nresM = L.marker([p.nres.lat, p.nres.lon], { title: p.nres.title });
  nresM.bindPopup(
    `<b>${p.nres.title}</b><br/>GT non-residential<br/>CNN UMAP ≈ ${p.nres.u.toFixed(2)}` +
    `<br/><span style="color:#666">${nresHint}</span>`
  );
  contrastMarkers.addLayer(resM);
  contrastMarkers.addLayer(nresM);
  const bounds = L.latLngBounds([p.res.lat, p.res.lon], [p.nres.lat, p.nres.lon]).pad(0.6);
  map.fitBounds(bounds, { maxZoom: 18 });
  resM.openPopup();
  setStatus(`${p.label} · ${p.note}`);
}

async function loadRegions() {
  regions = await (await fetch('regions.json')).json();
  const sel = document.getElementById('region');
  sel.innerHTML = '';
  for (const r of regions) {
    const opt = document.createElement('option');
    opt.value = r.region;
    opt.textContent = `${r.region} · ${r.n_buildings} bldgs · ${r.n_cells} cells`;
    sel.appendChild(opt);
  }
}

function clearDataLayers() {
  if (buildingsLayer) { map.removeLayer(buildingsLayer); buildingsLayer = null; }
  if (cellsLayer) { map.removeLayer(cellsLayer); cellsLayer = null; }
  if (bboxLayer) { map.removeLayer(bboxLayer); bboxLayer = null; }
  if (ghslLayer) { map.removeLayer(ghslLayer); ghslLayer = null; }
  clearContrastMarkers();
  ghslLayerUrl = null;
  disagreeIds = new Set();
}

function isGhslCnnDisagree(p) {
  if (p.ghsl_cnn_disagree != null) return !!p.ghsl_cnn_disagree;
  return disagreeIds.has(String(p.id));
}

function passesFilters(p, { forCells = false } = {}) {
  const errOnly = document.getElementById('lyrErrOnly').checked;
  const disOnly = document.getElementById('lyrGhslCnnDisagree').checked;
  if (errOnly && !isMisclassified(p)) return false;
  // Disagree filter applies to buildings/bboxes only — keep AE cells so other layers stay visible
  if (!forCells && disOnly && !isGhslCnnDisagree(p)) return false;
  return true;
}

function bboxStyle(f) {
  const p = f.properties;
  if (!passesFilters(p)) {
    return { opacity: 0, fillOpacity: 0, weight: 0 };
  }
  const disOnly = document.getElementById('lyrGhslCnnDisagree').checked;
  return {
    color: disOnly ? '#c0392b' : '#111',
    weight: disOnly ? 2.2 : 1.4,
    opacity: 0.95,
    fillColor: '#111', fillOpacity: 0,
    dashArray: disOnly ? null : '4 3',
  };
}

function isMisclassified(p) {
  const task = document.getElementById('errTask').value;
  return task === 'multi' ? !!p.misclassified_multi : !!p.misclassified_res;
}

function errorFill(p) {
  // Okabe–Ito colorblind-safe categorical palette
  const task = document.getElementById('errTask').value;
  if (task === 'multi') return '#CC79A7'; // reddish purple
  if (p.error_res === 'fp') return '#D55E00'; // vermillion
  if (p.error_res === 'fn') return '#0072B2'; // blue
  return '#CC79A7';
}

function buildingStyle(f) {
  const p = f.properties;
  const errOnly = document.getElementById('lyrErrOnly').checked;
  const errColor = document.getElementById('lyrErrColor').checked;
  const disOnly = document.getElementById('lyrGhslCnnDisagree').checked;
  const bad = isMisclassified(p);
  if (!passesFilters(p)) {
    return { opacity: 0, fillOpacity: 0, weight: 0, interactive: false };
  }
  const useCnn = document.getElementById('lyrCnn').checked;
  const useResi = document.getElementById('lyrResi').checked;
  const useGhslFill = document.getElementById('lyrGhslFill').checked;
  let fill = '#6b7c8a';
  if (errOnly && errColor && bad) fill = errorFill(p);
  else if (useGhslFill) fill = GHSL_FILL[p.ghsl_prediction] || '#7f8c8d';
  else if (useResi) fill = p.y_res === 1 ? '#0072B2' : '#E69F00';
  else if (useCnn) fill = manifoldHex(p, 'cnn');
  const show = document.getElementById('lyrPoly').checked || useCnn || useResi || useGhslFill || errOnly || disOnly;
  return {
    color: disOnly ? '#111' : ((errOnly && bad) ? fill : (useResi || useGhslFill ? fill : '#222')),
    weight: disOnly ? 1.8 : ((errOnly && bad) ? 1.6 : (useResi || useGhslFill ? 1.2 : 0.8)),
    opacity: 0.95,
    fillColor: fill,
    fillOpacity: show ? (disOnly ? 0.85 : 0.75) : 0,
    interactive: !!show,
  };
}

function cellStyle(f) {
  const p = f.properties;
  if (!passesFilters(p, { forCells: true })) {
    return { opacity: 0, fillOpacity: 0, weight: 0 };
  }
  const mode = document.getElementById('cellColor').value;
  const errColor = document.getElementById('lyrErrColor').checked;
  const errOnly = document.getElementById('lyrErrOnly').checked;
  let fill = manifoldHex(p, 'ae');
  if (mode === 'cnn') fill = manifoldHex(p, 'cnn');
  else if (mode === 'sim') fill = colorFromUnit(p.sim_u, p.sim_hex || '#888');
  else fill = manifoldHex(p, 'ae');
  if (errOnly && errColor && isMisclassified(p)) fill = errorFill(p);
  return {
    color: fill, weight: 0, opacity: 1,
    fillColor: fill, fillOpacity: 0.85,
  };
}

function applyFeatureStyles(layerGroup, styleFn) {
  // Leaflet Path.setStyle expects an options object. Passing a function via
  // layerGroup.setStyle(fn) does NOT re-evaluate per feature (only the GeoJSON
  // constructor / resetStyle path does). So we must apply explicitly.
  if (!layerGroup) return;
  layerGroup.eachLayer(layer => {
    if (layer.feature && layer.setStyle) layer.setStyle(styleFn(layer.feature));
  });
}

function refreshStyles() {
  updateCmapRamp();
  // When focusing on disagreements, turn on bbox outlines so boxes are obvious
  const disOnly = document.getElementById('lyrGhslCnnDisagree').checked;
  if (disOnly && !document.getElementById('lyrBbox').checked) {
    document.getElementById('lyrBbox').checked = true;
  }
  applyFeatureStyles(buildingsLayer, buildingStyle);
  if (bboxLayer) {
    const on = document.getElementById('lyrBbox').checked;
    if (on && !map.hasLayer(bboxLayer)) bboxLayer.addTo(map);
    if (!on && map.hasLayer(bboxLayer)) map.removeLayer(bboxLayer);
    if (on) applyFeatureStyles(bboxLayer, bboxStyle);
  }
  if (cellsLayer) {
    const on = document.getElementById('lyrCells').checked;
    if (on && !map.hasLayer(cellsLayer)) cellsLayer.addTo(map);
    if (!on && map.hasLayer(cellsLayer)) map.removeLayer(cellsLayer);
    if (on) applyFeatureStyles(cellsLayer, cellStyle);
  }
  // Keep buildings above cells after layer toggles
  if (buildingsLayer && map.hasLayer(buildingsLayer)) buildingsLayer.bringToFront();
  if (ghslLayer) {
    const on = document.getElementById('lyrGhsl').checked;
    const op = parseInt(document.getElementById('ghslOpacity').value, 10) / 100;
    document.getElementById('ghslOpacityVal').textContent = Math.round(op * 100) + '%';
    const wantUrl = (disOnly && currentRegion && currentRegion.ghsl_disagree)
      ? currentRegion.ghsl_disagree
      : (currentRegion && currentRegion.ghsl);
    if (wantUrl && ghslLayerUrl !== wantUrl) {
      const bounds = ghslLayer.getBounds();
      const wasOn = map.hasLayer(ghslLayer);
      map.removeLayer(ghslLayer);
      ghslLayer = L.imageOverlay(wantUrl, bounds, {
        opacity: op, interactive: false, pane: 'ghslPane'
      });
      ghslLayerUrl = wantUrl;
      if (wasOn || on) ghslLayer.addTo(map);
    } else {
      ghslLayer.setOpacity(op);
    }
    if (on && !map.hasLayer(ghslLayer)) ghslLayer.addTo(map);
    if (!on && map.hasLayer(ghslLayer)) map.removeLayer(ghslLayer);
  }
  if (document.getElementById('lyrOsm').checked) {
    if (!map.hasLayer(osm)) osm.addTo(map);
  } else if (map.hasLayer(osm)) {
    map.removeLayer(osm);
  }
  updateStatusCounts();
}

function updateStatusCounts() {
  if (!currentRegion || !buildingsLayer) return;
  const errOnly = document.getElementById('lyrErrOnly').checked;
  const disOnly = document.getElementById('lyrGhslCnnDisagree').checked;
  const task = document.getElementById('errTask').value;
  const nErr = task === 'multi' ? (currentRegion.n_err_multi || 0) : (currentRegion.n_err_res || 0);
  const nDis = currentRegion.n_ghsl_cnn_disagree || 0;
  let msg = `Loaded ${currentRegion.region}: ${currentRegion.n_buildings} buildings, ${currentRegion.n_cells} cells`;
  if (disOnly) {
    msg += ` · showing ${nDis} GHSL↔CNN disagreements`;
  } else if (errOnly) {
    msg += ` · showing ${nErr} misclassified (${task === 'multi' ? '10-class' : 'residential'})`;
  } else {
    msg += ` · errors: ${currentRegion.n_err_res || 0} res / ${currentRegion.n_err_multi || 0} type · GHSL≠CNN: ${nDis}`;
  }
  if (map.getZoom() < 15) msg += ' · tip: zoom in (16+)';
  setStatus(msg);
}

async function loadRegion(opts = {}) {
  const name = document.getElementById('region').value;
  const meta = regions.find(r => r.region === name);
  if (!meta) return;
  const focus = opts.focus || null; // [lat, lon]
  const zoom = opts.zoom || 17;

  setStatus(`Loading ${name}…`);
  clearDataLayers();
  currentRegion = meta;

  if (meta.ghsl) {
    const op = parseInt(document.getElementById('ghslOpacity').value, 10) / 100;
    const disOnly = document.getElementById('lyrGhslCnnDisagree').checked;
    const url = (disOnly && meta.ghsl_disagree) ? meta.ghsl_disagree : meta.ghsl;
    ghslLayer = L.imageOverlay(
      url,
      [[meta.ghsl_south, meta.ghsl_west], [meta.ghsl_north, meta.ghsl_east]],
      { opacity: op, interactive: false, pane: 'ghslPane' }
    );
    ghslLayerUrl = url;
    if (document.getElementById('lyrGhsl').checked) ghslLayer.addTo(map);
  }

  const bfc = await (await fetch(meta.buildings)).json();
  disagreeIds = new Set(
    (bfc.features || [])
      .filter(f => f.properties && f.properties.ghsl_cnn_disagree)
      .map(f => String(f.properties.id))
  );
  buildingsLayer = L.geoJSON(bfc, {
    pane: 'buildingsPane',
    style: buildingStyle,
    onEachFeature: (f, layer) => {
      const p = f.properties;
      const esc = (v) => {
        if (v == null || v === '') return '—';
        return String(v).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      };
      const pred = p.pred_label || '?';
      const conf = (p.pred_prob != null) ? ` (${(100 * p.pred_prob).toFixed(0)}%)` : '';
      const resErr = p.error_res && p.error_res !== 'correct' ? p.error_res.toUpperCase() : 'ok';
      const gtRes = p.gt_res_label || (p.y_res === 1 ? 'RES' : (p.y_res === 0 ? 'NRES' : '—'));
      const cnnLab = p.cnn_res_label || (p.pred_res === 1 ? 'RES' : (p.pred_res === 0 ? 'NRES' : '—'));
      const ghsl = p.ghsl_prediction || '—';
      const ghslPx = (p.ghsl_n_pixels != null) ? ` · ${Number(p.ghsl_n_pixels).toFixed(0)} px` : '';
      const disagree = p.ghsl_cnn_disagree ? ' <b style="color:#c0392b">DISAGREE</b>' : '';
      const area = (p.area_m2 != null) ? `${Number(p.area_m2).toFixed(0)} m²` : '—';
      const height = (p.height_m != null) ? `${Number(p.height_m).toFixed(1)} m` : '—';
      const floors = (p.num_floors != null) ? String(p.num_floors) : '—';
      let sourceBits = '—';
      if (Array.isArray(p.source_primary) && p.source_primary.length) {
        sourceBits = p.source_primary.map(s => esc(s.dataset || s.property || JSON.stringify(s))).join(', ');
      } else if (p.source_primary) {
        sourceBits = esc(typeof p.source_primary === 'string' ? p.source_primary : JSON.stringify(p.source_primary));
      }
      layer.bindPopup(
        `<div style="min-width:220px;font-size:12px;line-height:1.35">` +
        `<div style="font-size:13px;font-weight:650;margin-bottom:6px">${esc(p.label || 'building')}</div>` +
        `<div style="color:#666;margin-bottom:8px"><code style="font-size:11px">${esc(p.id)}</code>` +
        (p.region ? ` · ${esc(p.region)}` : '') + `</div>` +

        `<div style="font-weight:650;color:#1f4e79;margin:6px 0 2px">Ground truth</div>` +
        `<div>Residential (<code>resi_def_v1</code>): <b>${esc(gtRes)}</b> (y_res=${p.y_res ?? '—'})</div>` +
        `<div>Study label: <b>${esc(p.label)}</b></div>` +
        `<div>Brief category: ${esc(p.brief_category)}</div>` +
        `<div>Label source: ${esc(p.label_source)}</div>` +

        `<div style="font-weight:650;color:#1f4e79;margin:8px 0 2px">Overture</div>` +
        `<div>class: <b>${esc(p.overture_class)}</b></div>` +
        `<div>subtype: <b>${esc(p.overture_subtype)}</b></div>` +
        `<div>area ≈ ${area} · height ${height} · floors ${floors}</div>` +
        `<div>source: ${sourceBits}</div>` +

        `<div style="font-weight:650;color:#1f4e79;margin:8px 0 2px">CNN / GHSL</div>` +
        `<div>CNN type: <b>${esc(pred)}</b>${conf} · res <b>${esc(cnnLab)}</b> (${resErr})</div>` +
        `<div>GHSL majority: <b>${esc(ghsl)}</b>${disagree}${ghslPx}</div>` +
        `<div>type error: ${p.misclassified_multi ? 'yes' : 'no'} · res error: ${p.misclassified_res ? 'yes' : 'no'}</div>` +
        `<div>cells: ${p.n_cells ?? '—'}` +
        (p.lon != null && p.lat != null ? ` · ${Number(p.lon).toFixed(5)}, ${Number(p.lat).toFixed(5)}` : '') +
        `</div>` +
        `</div>`,
        { maxWidth: 340 }
      );
    }
  }).addTo(map);

  if (meta.bboxes) {
    const bbfc = await (await fetch(meta.bboxes)).json();
    bboxLayer = L.geoJSON(bbfc, { style: bboxStyle, interactive: false });
    if (document.getElementById('lyrBbox').checked) bboxLayer.addTo(map);
  }

  setStatus(`Loading ${name} grid cells…`);
  const cfc = await (await fetch(meta.cells)).json();
  // Non-interactive so AE cells don't steal clicks from building popups
  cellsLayer = L.geoJSON(cfc, { style: cellStyle, interactive: false });
  if (document.getElementById('lyrCells').checked) {
    cellsLayer.addTo(map);
  }
  if (buildingsLayer) buildingsLayer.bringToFront();

  if (focus) {
    map.setView(focus, zoom);
    markFocus(focus[0], focus[1], opts.popup || `${focus[0].toFixed(5)}, ${focus[1].toFixed(5)}`);
  } else {
    map.fitBounds([
      [meta.south, meta.west],
      [meta.north, meta.east],
    ], { padding: [30, 30] });
  }
  refreshStyles();
}

async function ensureRegionFor(lat, lon, popup) {
  const { region, distDeg } = nearestRegion(lat, lon);
  if (!region) return;
  document.getElementById('region').value = region.region;
  // ~3 deg ≈ metro-scale; still load nearest even if farther, but warn
  if (distDeg > 5) {
    setStatus(`Nearest labeled AOI is ${region.region} (${distDeg.toFixed(1)}° away) — overlays only cover that downtown box.`);
  }
  await loadRegion({ focus: [lat, lon], zoom: 17, popup });
}

async function goLatLon() {
  const raw = document.getElementById('latlon').value.trim();
  const m = raw.match(/(-?\\d+\\.?\\d*)\\s*,\\s*(-?\\d+\\.?\\d*)/);
  if (!m) { setStatus('Enter coordinates as: lat, lon'); return; }
  const lat = parseFloat(m[1]), lon = parseFloat(m[2]);
  if (!isFinite(lat) || !isFinite(lon) || Math.abs(lat) > 90 || Math.abs(lon) > 180) {
    setStatus('Invalid lat/lon'); return;
  }
  await ensureRegionFor(lat, lon, `lat ${lat.toFixed(6)}, lon ${lon.toFixed(6)}`);
}

async function searchAddress() {
  const q = document.getElementById('address').value.trim();
  if (!q) { setStatus('Enter an address'); return; }
  setStatus('Geocoding…');
  const url = 'https://nominatim.openstreetmap.org/search?format=json&limit=1&q=' + encodeURIComponent(q);
  try {
    const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
    const data = await res.json();
    if (!data.length) { setStatus('No results for that address'); return; }
    const hit = data[0];
    const lat = parseFloat(hit.lat), lon = parseFloat(hit.lon);
    document.getElementById('latlon').value = `${lat.toFixed(6)}, ${lon.toFixed(6)}`;
    await ensureRegionFor(lat, lon, hit.display_name);
    setStatus(`${hit.display_name} · loaded nearest metro ${document.getElementById('region').value}`);
  } catch (e) {
    setStatus('Geocoding failed (network / Nominatim). Try lat,lon instead.');
  }
}

document.getElementById('loadRegion').onclick = () => loadRegion();
document.getElementById('goLatLon').onclick = goLatLon;
document.getElementById('searchAddress').onclick = searchAddress;
document.getElementById('contrastPreset').addEventListener('change', goContrastPreset);
document.getElementById('address').addEventListener('keydown', e => { if (e.key === 'Enter') searchAddress(); });
document.getElementById('latlon').addEventListener('keydown', e => { if (e.key === 'Enter') goLatLon(); });
['lyrOsm','lyrGhsl','lyrPoly','lyrBbox','lyrCnn','lyrGhslFill','lyrGhslCnnDisagree','lyrCells','lyrResi','lyrErrOnly','lyrErrColor','errTask','cellColor','cmap'].forEach(id => {
  document.getElementById(id).addEventListener('change', refreshStyles);
});
document.getElementById('ghslOpacity').addEventListener('input', refreshStyles);

L.Control.geocoder({
  defaultMarkGeocode: false,
  placeholder: 'Search address…',
  geocoder: L.Control.Geocoder.nominatim()
}).on('markgeocode', async function(e) {
  const { center, name } = e.geocode;
  document.getElementById('latlon').value = `${center.lat.toFixed(6)}, ${center.lng.toFixed(6)}`;
  document.getElementById('address').value = name;
  await ensureRegionFor(center.lat, center.lng, name);
}).addTo(map);

loadColormaps().then(() => loadRegions()).then(() => {
  fillContrastPresets();
  setStatus('Ready — try a RES vs NRES contrast preset, search an address, or Load region.');
});
</script>
</body>
</html>
"""
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ghsl-only",
        action="store_true",
        help="Only rebuild GHSL PNG overlays + HTML (keep existing geojson)",
    )
    ap.add_argument(
        "--metadata-only",
        action="store_true",
        help="Patch Overture/GT fields into buildings geojson + refresh HTML",
    )
    ap.add_argument(
        "--colors-only",
        action="store_true",
        help="Recompute UMAP→colormap units into geojson, write colormaps.json, refresh HTML",
    )
    ap.add_argument(
        "--rebake-cmap",
        action="store_true",
        help="Remap existing ae_u/cnn_u via default colormap (no UMAP) + refresh HTML",
    )
    ap.add_argument(
        "--manifold",
        default="umap",
        choices=["umap", "tsne", "both"],
        help="Manifold method for --colors-only (default: umap; tsne parked)",
    )
    args = ap.parse_args()
    if args.rebake_cmap:
        rebake_hex_from_units(OUT)
    elif args.colors_only:
        if args.manifold == "both":
            methods = ("umap",)  # t-SNE parked
            print("note: t-SNE parked — computing UMAP only", flush=True)
        elif args.manifold == "tsne":
            print("note: t-SNE parked — falling back to UMAP", flush=True)
            methods = ("umap",)
        else:
            methods = (args.manifold,)
        patch_color_units(OUT, methods=methods)
    elif args.metadata_only:
        enrich_buildings_overture_metadata(OUT)
        write_colormaps_json(OUT)
        write_html(OUT / "index.html")
    elif args.ghsl_only:
        write_ghsl_layers(OUT)
    else:
        main()
