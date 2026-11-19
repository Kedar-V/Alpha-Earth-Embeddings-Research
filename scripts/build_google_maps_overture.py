#!/usr/bin/env python3
"""Build Google Maps + Overture footprint overlay for Track-A US metros.

Writes outputs/google_maps_overture/:
  index.html, regions.json, manifest.json, README.md, geojson/<region>.geojson

Serve with:
  python scripts/serve_google_maps_overture.py
  → http://127.0.0.1:8770/
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.geometry import mapping

ROOT = Path(__file__).resolve().parents[1]
BUILDINGS = ROOT / "datasets" / "building_bbox_grid" / "buildings.parquet"
OUT = ROOT / "outputs" / "google_maps_overture"

REGION_ORDER = [
    "Boston_NE",
    "NYC_NE",
    "Atlanta_SE",
    "Houston_S",
    "Chicago_MW",
    "Phoenix_SW",
    "Denver_W",
    "LA_W",
    "SF_W",
    "Seattle_NW",
]

PRETTY = {
    "Boston_NE": "Boston",
    "NYC_NE": "New York City",
    "Atlanta_SE": "Atlanta",
    "Houston_S": "Houston",
    "Chicago_MW": "Chicago",
    "Phoenix_SW": "Phoenix",
    "Denver_W": "Denver",
    "LA_W": "Los Angeles",
    "SF_W": "San Francisco",
    "Seattle_NW": "Seattle",
}

LABEL_ORDER = [
    "single_family",
    "multi_family",
    "education",
    "medical",
    "retail",
    "office",
    "hotel",
    "industrial",
    "religious",
    "civic",
]

LABEL_COLORS = {
    "single_family": "#4C78A8",
    "multi_family": "#72B7B2",
    "education": "#54A24B",
    "medical": "#EECA3B",
    "retail": "#F58518",
    "office": "#B279A2",
    "hotel": "#FF9DA6",
    "industrial": "#9D755D",
    "religious": "#BAB0AC",
    "civic": "#E45756",
}

RES_COLOR = "#1F6FA8"
NRES_COLOR = "#C2691A"
# GHSL functional domain colors (user: green = RES)
GHSL_RES_COLOR = "#2E8B57"
GHSL_NRES_COLOR = "#C2185B"
SAMPLE_N = 3000
SEED = 42
WEEK6_GHSL = (
    ROOT
    / "outputs"
    / "week6_overture_polygon_mae"
    / "20261110_041557"
    / "artifacts"
    / "per_building_with_pls.parquet"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """1:1 RES/NRES up to n (or all if smaller)."""
    res = df[df["y_res"] == 1]
    nres = df[df["y_res"] == 0]
    if len(df) <= n:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    half = n // 2
    n_res = min(len(res), half)
    n_nres = min(len(nres), half)
    # fill remainder from the larger class
    taken = n_res + n_nres
    rem = n - taken
    parts = []
    if n_res:
        parts.append(res.sample(n=n_res, random_state=seed))
    if n_nres:
        parts.append(nres.sample(n=n_nres, random_state=seed + 1))
    if rem > 0:
        used = set()
        if parts:
            used = set(pd.concat(parts)["id"].astype(str))
        pool = df[~df["id"].astype(str).isin(used)]
        if len(pool):
            parts.append(pool.sample(n=min(rem, len(pool)), random_state=seed + 2))
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)


def attach_ghsl(df: pd.DataFrame) -> pd.DataFrame:
    """Join week-6 polygon-zonal GHSL soft fractions; majority channel → ghsl_y."""
    out = df.copy()
    out["ghsl_frac_res"] = np.nan
    out["ghsl_frac_nres"] = np.nan
    out["ghsl_covered"] = False
    out["ghsl_y"] = -1  # -1 unknown, 1 RES, 0 NRES
    if not WEEK6_GHSL.exists():
        print(f"[{utc_now()}] WARN missing GHSL join file: {WEEK6_GHSL}")
        return out
    g = pd.read_parquet(
        WEEK6_GHSL,
        columns=["id", "ghsl_frac_res", "ghsl_frac_nres", "ghsl_covered"],
    )
    g["id"] = g["id"].astype(str)
    g = g.drop_duplicates("id")
    out = out.drop(columns=["ghsl_frac_res", "ghsl_frac_nres", "ghsl_covered"], errors="ignore")
    out = out.merge(g, on="id", how="left")
    out["ghsl_covered"] = out["ghsl_covered"].fillna(False).astype(bool)
    bu = out["ghsl_frac_res"].fillna(0.0) + out["ghsl_frac_nres"].fillna(0.0)
    share = np.where(bu > 1e-9, out["ghsl_frac_nres"].fillna(0.0) / bu, np.nan)
    # majority RES when NRES share of built < 0.5
    out["ghsl_y"] = np.where(
        ~out["ghsl_covered"] | ~np.isfinite(share),
        -1,
        np.where(share < 0.5, 1, 0),
    ).astype(int)
    return out


def row_to_feature(row: pd.Series) -> dict | None:
    try:
        geom = shapely_wkt.loads(row["wkt"])
    except Exception:
        return None
    if geom is None or geom.is_empty:
        return None
    props = {
        "id": str(row["id"]),
        "region": str(row["region"]),
        "y_res": int(row["y_res"]),
        "label": str(row["label"]) if pd.notna(row["label"]) else "",
        "class": str(row["class"]) if pd.notna(row["class"]) else "",
        "lon": float(row["lon"]),
        "lat": float(row["lat"]),
        "ghsl_covered": bool(row.get("ghsl_covered", False)),
        "ghsl_y": int(row["ghsl_y"]) if pd.notna(row.get("ghsl_y")) else -1,
    }
    if pd.notna(row.get("ghsl_frac_res")):
        props["ghsl_frac_res"] = round(float(row["ghsl_frac_res"]), 4)
    if pd.notna(row.get("ghsl_frac_nres")):
        props["ghsl_frac_nres"] = round(float(row["ghsl_frac_nres"]), 4)
    return {"type": "Feature", "geometry": mapping(geom), "properties": props}


def write_geojson(df: pd.DataFrame, path: Path) -> int:
    feats = []
    for _, row in df.iterrows():
        f = row_to_feature(row)
        if f is not None:
            feats.append(f)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return len(feats)


def build_index_html() -> str:
    label_colors_js = json.dumps(LABEL_COLORS)
    label_order_js = json.dumps(LABEL_ORDER)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Google Maps · Overture + GHSL</title>
<style>
  :root {{
    --bg: #0f1419;
    --panel: #1a222c;
    --text: #e8eef4;
    --muted: #9aa7b5;
    --accent: #1F6FA8;
    --nres: #C2691A;
    --border: #2c3846;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; height: 100%; font-family: "IBM Plex Sans", system-ui, sans-serif; background: var(--bg); color: var(--text); }}
  #map {{ position: absolute; inset: 0; }}
  #ui {{
    position: absolute; top: 16px; left: 16px; z-index: 5;
    width: min(360px, calc(100vw - 32px));
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 14px 16px; box-shadow: 0 8px 28px rgba(0,0,0,.35);
    max-height: calc(100vh - 32px); overflow: auto;
  }}
  #ui h1 {{ margin: 0 0 4px; font-size: 16px; font-weight: 600; }}
  #ui p.sub {{ margin: 0 0 12px; font-size: 12px; color: var(--muted); line-height: 1.35; }}
  label.field {{ display: block; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin: 10px 0 4px; }}
  select, button {{
    width: 100%; padding: 8px 10px; border-radius: 8px; border: 1px solid var(--border);
    background: #121820; color: var(--text); font-size: 14px;
  }}
  button {{ cursor: pointer; margin-top: 8px; background: var(--accent); border-color: transparent; font-weight: 600; }}
  button.secondary {{ background: #243040; }}
  .row {{ display: flex; gap: 8px; align-items: center; margin-top: 10px; font-size: 13px; }}
  .row input {{ width: auto; }}
  #legend {{ margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--border); font-size: 12px; }}
  #legend .item {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
  #legend .swatch {{ width: 12px; height: 12px; border-radius: 2px; flex: 0 0 auto; }}
  #status {{ margin-top: 8px; font-size: 11px; color: var(--muted); line-height: 1.4; }}
  #err {{ color: #ff8a7a; font-size: 12px; margin-top: 8px; display: none; }}
  .gm-style .gm-style-iw-c, .gm-style .gm-style-iw-d {{ color: #14231F !important; }}
  .gm-style .gm-style-iw-d {{ overflow: auto !important; }}
</style>
</head>
<body>
<div id="map"></div>
<div id="ui">
  <h1>Overture + GHSL on Google Maps</h1>
  <p class="sub">Track-A footprints · GHSL colors = polygon-zonal BUILT-S majority (green = RES)</p>
  <label class="field" for="city">Metro</label>
  <select id="city"></select>
  <div class="row"><input type="checkbox" id="lyrOverture" checked/><label for="lyrOverture">Footprints</label></div>
  <div class="row"><input type="checkbox" id="sat"/><label for="sat">Satellite basemap</label></div>
  <div class="row"><input type="checkbox" id="hotelsOnly"/><label for="hotelsOnly">Hotels only</label></div>
  <div class="row"><input type="checkbox" id="hotelsGhslRes"/><label for="hotelsGhslRes">Hotels where GHSL = RES (green)</label></div>
  <label class="field" for="colorMode">Color mode</label>
  <select id="colorMode">
    <option value="ghsl">GHSL RES / NRES</option>
    <option value="yn">Overture RES / NRES</option>
    <option value="label">Study label</option>
  </select>
  <div id="legend"></div>
  <div id="status">Loading…</div>
  <div id="err"></div>
</div>
<script src="/config.js"></script>
<script>
const RES_COLOR = "{RES_COLOR}";
const NRES_COLOR = "{NRES_COLOR}";
const GHSL_RES_COLOR = "{GHSL_RES_COLOR}";
const GHSL_NRES_COLOR = "{GHSL_NRES_COLOR}";
const LABEL_COLORS = {label_colors_js};
const LABEL_ORDER = {label_order_js};
let map, dataLayer, regions = [], currentRegion = null, colorMode = "ghsl";
let hotelsOnly = false, hotelsGhslRes = false, lastGeo = null;
let hotelMarkers = [];
let infoWindow = null;

function showErr(msg) {{
  const el = document.getElementById("err");
  el.style.display = "block";
  el.textContent = msg;
}}

function ghslYOf(obj) {{
  // Data layer Feature or plain geojson props
  const raw = obj.getProperty ? obj.getProperty("ghsl_y") : obj.ghsl_y;
  return Number(raw);
}}

function fillLegend() {{
  const box = document.getElementById("legend");
  let html = "";
  if (colorMode === "ghsl") {{
    html += `<div class="item"><span class="swatch" style="background:${{GHSL_RES_COLOR}}"></span>GHSL RES (majority)</div>`;
    html += `<div class="item"><span class="swatch" style="background:${{GHSL_NRES_COLOR}}"></span>GHSL NRES (majority)</div>`;
    html += `<div class="item"><span class="swatch" style="background:#666"></span>No GHSL coverage</div>`;
  }} else if (colorMode === "yn") {{
    html += `<div class="item"><span class="swatch" style="background:${{RES_COLOR}}"></span>Overture RES (y_res=1)</div>`;
    html += `<div class="item"><span class="swatch" style="background:${{NRES_COLOR}}"></span>Overture NRES (y_res=0)</div>`;
  }} else {{
    for (const lab of LABEL_ORDER) {{
      html += `<div class="item"><span class="swatch" style="background:${{LABEL_COLORS[lab]}}"></span>${{lab}}</div>`;
    }}
  }}
  if (hotelsOnly || hotelsGhslRes) {{
    html += `<div class="item" style="margin-top:6px;color:var(--muted)">Dots = hotel centroids (visible at metro zoom)</div>`;
  }}
  box.innerHTML = html;
}}

function passesFilterProps(p) {{
  const lab = p.label;
  const gy = Number(p.ghsl_y);
  if (hotelsGhslRes) return lab === "hotel" && gy === 1;
  if (hotelsOnly) return lab === "hotel";
  return true;
}}

function passesFilter(feature) {{
  return passesFilterProps({{
    label: feature.getProperty("label"),
    ghsl_y: feature.getProperty("ghsl_y"),
  }});
}}

function colorForProps(p) {{
  if (colorMode === "ghsl") {{
    const gy = Number(p.ghsl_y);
    if (gy === 1) return GHSL_RES_COLOR;
    if (gy === 0) return GHSL_NRES_COLOR;
    return "#666";
  }}
  if (colorMode === "yn") {{
    return Number(p.y_res) === 1 ? RES_COLOR : NRES_COLOR;
  }}
  return LABEL_COLORS[p.label] || "#888";
}}

function styleFeature(feature) {{
  if (!passesFilter(feature)) {{
    return {{ visible: false }};
  }}
  const isHotel = feature.getProperty("label") === "hotel";
  const color = colorForProps({{
    label: feature.getProperty("label"),
    ghsl_y: feature.getProperty("ghsl_y"),
    y_res: feature.getProperty("y_res"),
  }});
  return {{
    visible: true,
    fillColor: color,
    fillOpacity: isHotel ? 0.75 : 0.45,
    strokeColor: isHotel ? "#ffffff" : color,
    strokeWeight: isHotel ? 2.5 : 1,
    strokeOpacity: 0.95,
    zIndex: isHotel ? 10 : 1,
  }};
}}

function clearHotelMarkers() {{
  for (const m of hotelMarkers) m.setMap(null);
  hotelMarkers = [];
}}

function popupHtml(p) {{
  const gy = Number(p.ghsl_y);
  const ghslTxt = gy === 1 ? "RES" : (gy === 0 ? "NRES" : "none");
  return `<div style="font:13px/1.4 'IBM Plex Sans',system-ui,sans-serif;max-width:280px;color:#14231F;background:#fff">
    <div style="color:#14231F;font-weight:600"><b style="color:#14231F">${{p.label}}</b> · ${{p.class || ""}}</div>
    <div style="color:#14231F;margin-top:2px">Overture y_res=${{p.y_res}} (${{Number(p.y_res)===1?"RES":"NRES"}})</div>
    <div style="color:#14231F;margin-top:2px">GHSL majority=<b>${{ghslTxt}}</b> · frac res=${{p.ghsl_frac_res ?? "—"}} nres=${{p.ghsl_frac_nres ?? "—"}}</div>
    <div style="color:#5a6570;font-size:11px;margin-top:4px">${{p.id}}</div>
  </div>`;
}}

function rebuildHotelMarkers() {{
  clearHotelMarkers();
  if (!map || !lastGeo || !lastGeo.features) return;
  if (!(hotelsOnly || hotelsGhslRes)) return;
  const bounds = new google.maps.LatLngBounds();
  let n = 0;
  for (const f of lastGeo.features) {{
    const p = f.properties;
    if (!passesFilterProps(p)) continue;
    if (p.lat == null || p.lon == null) continue;
    const color = colorForProps(p);
    const pos = {{ lat: Number(p.lat), lng: Number(p.lon) }};
    const marker = new google.maps.Marker({{
      map,
      position: pos,
      title: `${{p.label}} · GHSL ${{Number(p.ghsl_y)===1?"RES":"NRES"}}`,
      zIndex: 50,
      icon: {{
        path: google.maps.SymbolPath.CIRCLE,
        scale: 7,
        fillColor: color,
        fillOpacity: 0.95,
        strokeColor: "#ffffff",
        strokeWeight: 2,
      }},
    }});
    marker.addListener("click", () => {{
      if (!infoWindow) infoWindow = new google.maps.InfoWindow();
      infoWindow.setContent(popupHtml(p));
      infoWindow.setPosition(pos);
      infoWindow.open({{ map }});
    }});
    hotelMarkers.push(marker);
    bounds.extend(pos);
    n += 1;
  }}
  if (n > 0) {{
    map.fitBounds(bounds, 48);
    // avoid over-zooming a single hotel
    google.maps.event.addListenerOnce(map, "bounds_changed", () => {{
      if (map.getZoom() > 16) map.setZoom(16);
    }});
  }}
}}

function applyStyles() {{
  if (!dataLayer) return;
  dataLayer.setStyle(styleFeature);
  fillLegend();
  updateStatus();
  rebuildHotelMarkers();
}}

function updateStatus() {{
  if (!lastGeo || !lastGeo.features) return;
  const feats = lastGeo.features;
  const hotels = feats.filter(f => f.properties.label === "hotel");
  const hRes = hotels.filter(f => Number(f.properties.ghsl_y) === 1).length;
  const hNres = hotels.filter(f => Number(f.properties.ghsl_y) === 0).length;
  const hMiss = hotels.length - hRes - hNres;
  const meta = regions.find(r => r.region === currentRegion);
  let filterNote = "";
  if (hotelsGhslRes) filterNote = " · filter: hotels GHSL=RES";
  else if (hotelsOnly) filterNote = " · filter: hotels";
  const shown = feats.filter(f => passesFilterProps(f.properties)).length;
  document.getElementById("status").textContent =
    `${{meta ? meta.pretty : currentRegion}} · showing ${{shown.toLocaleString()}} / ${{feats.length.toLocaleString()}}` +
    ` · hotels ${{hotels.length}} (GHSL RES ${{hRes}} / NRES ${{hNres}} / none ${{hMiss}})` +
    filterNote;
}}

async function loadRegion(region) {{
  currentRegion = region;
  document.getElementById("status").textContent = `Loading ${{region}}…`;
  clearHotelMarkers();
  dataLayer.forEach(f => dataLayer.remove(f));
  const url = `geojson/${{region}}.geojson`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to load ${{url}} (${{resp.status}})`);
  const geo = await resp.json();
  lastGeo = geo;
  dataLayer.addGeoJson(geo);
  applyStyles();
  if (!(hotelsOnly || hotelsGhslRes)) {{
    const meta = regions.find(r => r.region === region);
    if (meta && meta.bounds) {{
      const b = meta.bounds;
      map.fitBounds({{ north: b.north, south: b.south, east: b.east, west: b.west }});
    }}
  }}
}}

async function initMap() {{
  if (!window.GOOGLE_MAPS_API_KEY) {{
    showErr("Missing GOOGLE_MAPS_API_KEY — serve via scripts/serve_google_maps_overture.py");
    return;
  }}
  const regResp = await fetch("regions.json");
  regions = await regResp.json();
  const sel = document.getElementById("city");
  for (const r of regions) {{
    const opt = document.createElement("option");
    opt.value = r.region;
    opt.textContent = `${{r.pretty}} (${{r.n_sample.toLocaleString()}})`;
    sel.appendChild(opt);
  }}
  // Prefer LA — densest hotel GHSL=RES set
  const start = regions.find(r => r.region === "LA_W") || regions[0];
  sel.value = start.region;
  map = new google.maps.Map(document.getElementById("map"), {{
    center: {{ lat: start.center.lat, lng: start.center.lon }},
    zoom: 12,
    mapTypeId: "roadmap",
    mapTypeControl: false,
    streetViewControl: false,
    fullscreenControl: true,
  }});
  infoWindow = new google.maps.InfoWindow();
  dataLayer = new google.maps.Data({{ map: map }});
  dataLayer.addListener("click", (e) => {{
    const p = {{
      id: e.feature.getProperty("id"),
      label: e.feature.getProperty("label"),
      class: e.feature.getProperty("class"),
      y_res: e.feature.getProperty("y_res"),
      ghsl_y: e.feature.getProperty("ghsl_y"),
      ghsl_frac_res: e.feature.getProperty("ghsl_frac_res"),
      ghsl_frac_nres: e.feature.getProperty("ghsl_frac_nres"),
    }};
    infoWindow.setContent(popupHtml(p));
    infoWindow.setPosition(e.latLng);
    infoWindow.open({{ map }});
  }});
  sel.addEventListener("change", () => loadRegion(sel.value).catch(err => showErr(String(err))));
  document.getElementById("lyrOverture").addEventListener("change", (ev) => {{
    dataLayer.setMap(ev.target.checked ? map : null);
    for (const m of hotelMarkers) m.setMap(ev.target.checked ? map : null);
  }});
  document.getElementById("sat").addEventListener("change", (ev) => {{
    map.setMapTypeId(ev.target.checked ? "hybrid" : "roadmap");
  }});
  document.getElementById("hotelsOnly").addEventListener("change", (ev) => {{
    hotelsOnly = ev.target.checked;
    if (hotelsOnly) {{
      document.getElementById("hotelsGhslRes").checked = false;
      hotelsGhslRes = false;
    }}
    applyStyles();
  }});
  document.getElementById("hotelsGhslRes").addEventListener("change", (ev) => {{
    hotelsGhslRes = ev.target.checked;
    if (hotelsGhslRes) {{
      document.getElementById("hotelsOnly").checked = false;
      hotelsOnly = false;
    }}
    applyStyles();
  }});
  document.getElementById("colorMode").addEventListener("change", (ev) => {{
    colorMode = ev.target.value;
    applyStyles();
  }});
  fillLegend();
  await loadRegion(start.region);
}}

function loadGoogleMaps() {{
  const s = document.createElement("script");
  s.src = `https://maps.googleapis.com/maps/api/js?key=${{encodeURIComponent(window.GOOGLE_MAPS_API_KEY)}}&callback=initMap`;
  s.async = true;
  s.defer = true;
  s.onerror = () => showErr("Failed to load Google Maps JS — check API key / Maps JavaScript API enablement");
  document.head.appendChild(s);
}}
window.initMap = initMap;
loadGoogleMaps();
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=SAMPLE_N)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    t0 = time.time()
    out = args.out
    geo_dir = out / "geojson"
    geo_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{utc_now()}] loading {BUILDINGS}")
    cols = ["id", "region", "lon", "lat", "wkt", "y_res", "label", "class"]
    df = pd.read_parquet(BUILDINGS, columns=cols)
    df["id"] = df["id"].astype(str)
    print(f"[{utc_now()}] joining week-6 GHSL fractions")
    df = attach_ghsl(df)

    regions_meta = []
    manifest_regions = {}
    for region in REGION_ORDER:
        sub = df[df["region"] == region]
        if sub.empty:
            print(f"  skip {region}: empty")
            continue
        sample = stratified_sample(sub, args.n, args.seed)
        # Force-include every hotel so GHSL hotel clash is inspectable
        hotels = sub[sub["label"].astype(str) == "hotel"]
        if len(hotels):
            sample = (
                pd.concat([sample, hotels], ignore_index=True)
                .drop_duplicates(subset=["id"])
                .reset_index(drop=True)
            )
        n_feat = write_geojson(sample, geo_dir / f"{region}.geojson")
        west = float(sample["lon"].min())
        east = float(sample["lon"].max())
        south = float(sample["lat"].min())
        north = float(sample["lat"].max())
        # pad bounds slightly
        pad_lon = max(0.01, (east - west) * 0.05)
        pad_lat = max(0.01, (north - south) * 0.05)
        n_hotel = int((sample["label"].astype(str) == "hotel").sum())
        n_hotel_ghsl_res = int(
            ((sample["label"].astype(str) == "hotel") & (sample["ghsl_y"] == 1)).sum()
        )
        n_hotel_ghsl_nres = int(
            ((sample["label"].astype(str) == "hotel") & (sample["ghsl_y"] == 0)).sum()
        )
        meta = {
            "region": region,
            "pretty": PRETTY.get(region, region),
            "n_sample": int(n_feat),
            "n_res": int((sample["y_res"] == 1).sum()),
            "n_nres": int((sample["y_res"] == 0).sum()),
            "n_hotels": n_hotel,
            "n_hotels_ghsl_res": n_hotel_ghsl_res,
            "n_hotels_ghsl_nres": n_hotel_ghsl_nres,
            "n_available": int(len(sub)),
            "center": {"lon": float(sample["lon"].mean()), "lat": float(sample["lat"].mean())},
            "bounds": {
                "west": west - pad_lon,
                "east": east + pad_lon,
                "south": south - pad_lat,
                "north": north + pad_lat,
            },
            "geojson": f"geojson/{region}.geojson",
        }
        regions_meta.append(meta)
        manifest_regions[region] = meta
        print(
            f"  {region}: sample={n_feat:,} RES={meta['n_res']:,} NRES={meta['n_nres']:,} "
            f"hotels={n_hotel} (GHSL RES {n_hotel_ghsl_res} / NRES {n_hotel_ghsl_nres}) "
            f"(of {len(sub):,})"
        )

    (out / "regions.json").write_text(json.dumps(regions_meta, indent=2))
    (out / "index.html").write_text(build_index_html())
    manifest = {
        "built_at": utc_now(),
        "sample_n": args.n,
        "seed": args.seed,
        "source": str(BUILDINGS),
        "ghsl_source": str(WEEK6_GHSL),
        "res_color": RES_COLOR,
        "nres_color": NRES_COLOR,
        "ghsl_res_color": GHSL_RES_COLOR,
        "ghsl_nres_color": GHSL_NRES_COLOR,
        "label_colors": LABEL_COLORS,
        "regions": manifest_regions,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "README.md").write_text(
        """# Google Maps · Overture overlay

Stratified Track-A building samples on a Google Maps basemap.

```bash
# rebuild samples + HTML
python scripts/build_google_maps_overture.py

# serve (injects GOOGLE_MAPS_API_KEY from .env)
python scripts/serve_google_maps_overture.py
# → http://127.0.0.1:8766/
```

Requires Maps JavaScript API enabled for the key in `.env`.
"""
    )
    print(f"[{utc_now()}] done → {out} ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
