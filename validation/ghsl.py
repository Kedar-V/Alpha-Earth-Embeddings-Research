"""GHSL GHS-BUILT-C FUN tile IO (no resampling of categorical values)."""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlretrieve
from xml.sax.saxutils import escape

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from rasterio.windows import Window, from_bounds

from .schemas import GHSL_TO_LABEL


def map_ghsl_value(value: int | float, cfg: dict[str, Any]) -> str:
    g = cfg["ghsl"]
    v = int(value)
    if v == int(g["residential_value"]):
        return "RES"
    if v == int(g["non_residential_value"]):
        return "NRES"
    if v == int(g["nonbuilt_value"]):
        return "NONBUILT"
    if v == int(g["nodata_value"]):
        return "NODATA"
    return GHSL_TO_LABEL.get(v, "NODATA")


def tile_ids_for_bounds(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    cfg: dict[str, Any],
) -> list[tuple[int, int]]:
    g = cfg["ghsl"]
    x0 = float(g["tile_x0"])
    ymax = float(g["tile_ymax"])
    size = float(g["tile_size_m"])
    c0 = int((minx - x0) // size) + 1
    c1 = int((maxx - x0) // size) + 1
    r0 = int((ymax - maxy) // size) + 1
    r1 = int((ymax - miny) // size) + 1
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def ensure_tiles(tile_ids: Iterable[tuple[int, int]], cfg: dict[str, Any]) -> list[Path]:
    tiles_dir = Path(cfg["paths"]["ghsl_tiles_dir"])
    tiles_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for row, col in tile_ids:
        name = f"GHS_BUILT_C_FUN_E2018_GLOBE_R2023A_54009_10_V1_0_R{row}_C{col}.tif"
        tif = tiles_dir / name
        if not tif.exists():
            url = cfg["ghsl"]["tile_url_template"].format(row=row, col=col)
            zpath = tiles_dir / f"{name}.zip"
            print(f"Downloading GHSL tile R{row}_C{col} …", flush=True)
            urlretrieve(url, zpath)
            with zipfile.ZipFile(zpath) as zf:
                members = [m for m in zf.namelist() if m.endswith(".tif") and ".ovr" not in m]
                if not members:
                    raise FileNotFoundError(f"No GeoTIFF in {zpath}")
                zf.extract(members[0], path=tiles_dir)
                extracted = tiles_dir / members[0]
                if extracted.resolve() != tif.resolve():
                    extracted.replace(tif)
            zpath.unlink(missing_ok=True)
        paths.append(tif)
    return paths


def build_vrt(tile_paths: list[Path], vrt_path: str | Path) -> Path:
    """Write a simple VRT mosaic XML (nearest-neighbor categorical source)."""
    vrt_path = Path(vrt_path)
    vrt_path.parent.mkdir(parents=True, exist_ok=True)
    infos = []
    for p in tile_paths:
        with rasterio.open(p) as ds:
            infos.append(
                {
                    "path": str(p.resolve()),
                    "width": ds.width,
                    "height": ds.height,
                    "transform": ds.transform,
                    "bounds": ds.bounds,
                    "crs": ds.crs.to_wkt() if ds.crs else "",
                }
            )
    if not infos:
        raise ValueError("No tiles for VRT")

    minx = min(i["bounds"].left for i in infos)
    maxx = max(i["bounds"].right for i in infos)
    miny = min(i["bounds"].bottom for i in infos)
    maxy = max(i["bounds"].top for i in infos)
    px = abs(infos[0]["transform"].a)
    py = abs(infos[0]["transform"].e)
    out_w = int(round((maxx - minx) / px))
    out_h = int(round((maxy - miny) / py))
    gt = f"{minx}, {px}, 0.0, {maxy}, 0.0, {-py}"

    parts = [
        f'<VRTDataset rasterXSize="{out_w}" rasterYSize="{out_h}">',
        f"  <SRS>{escape(infos[0]['crs'])}</SRS>",
        f"  <GeoTransform>{gt}</GeoTransform>",
        '  <VRTRasterBand dataType="Byte" band="1">',
        "    <NoDataValue>255</NoDataValue>",
    ]
    for i in infos:
        xoff = int(round((i["bounds"].left - minx) / px))
        yoff = int(round((maxy - i["bounds"].top) / py))
        parts += [
            "    <ComplexSource>",
            f'      <SourceFilename relativeToVRT="0">{escape(i["path"])}</SourceFilename>',
            "      <SourceBand>1</SourceBand>",
            f'      <SourceProperties RasterXSize="{i["width"]}" RasterYSize="{i["height"]}" '
            f'DataType="Byte" BlockXSize="256" BlockYSize="256"/>',
            f'      <SrcRect xOff="0" yOff="0" xSize="{i["width"]}" ySize="{i["height"]}"/>',
            f'      <DstRect xOff="{xoff}" yOff="{yoff}" xSize="{i["width"]}" ySize="{i["height"]}"/>',
            "      <NODATA>255</NODATA>",
            "    </ComplexSource>",
        ]
    parts += ["  </VRTRasterBand>", "</VRTDataset>"]
    vrt_path.write_text("\n".join(parts))
    return vrt_path


@dataclass
class GhslMosaic:
    """Open per-tile datasets and route reads to covering tiles (avoids huge VRT I/O)."""

    cfg: dict[str, Any]
    tiles: list[Path] = field(default_factory=list)
    _datasets: list[DatasetReader] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        tiles_dir = Path(self.cfg["paths"]["ghsl_tiles_dir"])
        self.tiles = sorted(tiles_dir.glob("GHS_BUILT_C_FUN_*.tif"))
        if not self.tiles:
            raise FileNotFoundError(f"No GHSL tiles in {tiles_dir}")
        self._datasets = [rasterio.open(p) for p in self.tiles]

    @property
    def crs(self):
        return self._datasets[0].crs

    def close(self) -> None:
        for ds in self._datasets:
            ds.close()
        self._datasets = []

    def datasets_for_bounds(self, minx, miny, maxx, maxy) -> list[DatasetReader]:
        out = []
        for ds in self._datasets:
            b = ds.bounds
            if not (maxx < b.left or minx > b.right or maxy < b.bottom or miny > b.top):
                out.append(ds)
        return out

    def read_window(self, minx, miny, maxx, maxy, pad_px: int = 2):
        """Return (data, window, transform, dataset) for the primary covering tile."""
        dsets = self.datasets_for_bounds(minx, miny, maxx, maxy)
        if not dsets:
            raise ValueError(f"No GHSL tile covers bounds {(minx, miny, maxx, maxy)}")
        # Prefer the tile that contains the centroid
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        ds = next((d for d in dsets if (d.bounds.left <= cx <= d.bounds.right and d.bounds.bottom <= cy <= d.bounds.top)), dsets[0])
        window = from_bounds(minx, miny, maxx, maxy, transform=ds.transform)
        window = Window(
            max(0, int(window.col_off) - pad_px),
            max(0, int(window.row_off) - pad_px),
            int(np.ceil(window.width)) + 2 * pad_px,
            int(np.ceil(window.height)) + 2 * pad_px,
        ).intersection(Window(0, 0, ds.width, ds.height))
        data = ds.read(1, window=window, resampling=Resampling.nearest)
        transform = ds.window_transform(window)
        return data, window, transform, ds


def open_ghsl(cfg: dict[str, Any]) -> GhslMosaic:
    return GhslMosaic(cfg)


def sample_window(ds_or_mosaic, minx, miny, maxx, maxy, pad_px: int = 2):
    if isinstance(ds_or_mosaic, GhslMosaic):
        data, window, transform, _ = ds_or_mosaic.read_window(minx, miny, maxx, maxy, pad_px=pad_px)
        return data, window, transform
    ds: DatasetReader = ds_or_mosaic
    window = from_bounds(minx, miny, maxx, maxy, transform=ds.transform)
    window = Window(
        max(0, int(window.col_off) - pad_px),
        max(0, int(window.row_off) - pad_px),
        int(np.ceil(window.width)) + 2 * pad_px,
        int(np.ceil(window.height)) + 2 * pad_px,
    ).intersection(Window(0, 0, ds.width, ds.height))
    data = ds.read(1, window=window, resampling=Resampling.nearest)
    transform = ds.window_transform(window)
    return data, window, transform


def parse_tile_rc(path: Path) -> tuple[int, int] | None:
    m = re.search(r"_R(\d+)_C(\d+)\.tif$", path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))
