#!/usr/bin/env python3
"""Build Section 7 canvas board: AE rings (50/100/200 m) + Sentinel-2 NRES vs baselines."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANVAS = ROOT / "notebooks" / "ghsl_built_s_vs_ae" / "AlphaEarth Weeks 1–5 Canvas.html"
SECTION6_ID = "c9e6a4d3-7f0b-4c13-9e5a-4b8d0f2c3d67"
SECTION7_ID = "d0f7b5e4-8a1c-4d24-af6b-5c9e1a3d4e78"

# BUILT-S, Ridge, XGB, Ring, AE+S2, Rings+S2
COLORS = ["#C2691A", "#1F6FA8", "#8E44AD", "#2E7D32", "#0F766E", "#9B2226"]
NAMES = ["GHS-BUILT-S", "AE Ridge", "AE XGB", "AE Ring", "AE+S2", "Rings+S2"]


def f3(x: float) -> str:
    return f"{x:.3f}"


def bar_row(name: str, val: float, color: str, scale_max: float) -> str:
    w = max(2, int(round(320 * (val / scale_max)))) if scale_max > 0 else 2
    return (
        f'<div style="display:flex;align-items:center;gap:10px" title="{name}: MAE {f3(val)}">'
        f'<p style="margin:0;width:120px;font-size:14px">{name}</p>'
        f'<div style="width:{w}px;height:18px;background:{color};border-radius:0 4px 4px 0"></div>'
        f'<p style="margin:0;font-size:14px;font-weight:600;font-variant-numeric:tabular-nums">{f3(val)}</p>'
        f"</div>"
    )


def dumbbell_row(label: str, bg: str, vals: list[float], vmax: float) -> str:
    width = 380.0
    xs = [max(0.0, min(width, (v / vmax) * width)) if v == v else 0.0 for v in vals]
    finite_xs = [x for v, x in zip(vals, xs) if v == v]
    lo, hi = (min(finite_xs), max(finite_xs)) if finite_xs else (0.0, 0.0)
    ticks = "".join(
        f'<div style="position:absolute;left:{i * width / 5:.1f}px;top:0;bottom:0;width:1px;background:#E5E1D6"></div>'
        for i in range(1, 5)
    )
    dots = "".join(
        f'<div style="position:absolute;left:{x - 5:.1f}px;top:8px;width:10px;height:10px;'
        f'border-radius:50%;background:{c}"></div>'
        for x, c, v in zip(xs, COLORS, vals)
        if v == v
    )
    line = (
        f'<div style="position:absolute;left:{lo:.1f}px;top:12px;width:{max(2.0, hi - lo):.1f}px;'
        f'height:2px;background:#B0ADA3"></div>'
    )
    return (
        f'<div style="display:flex;align-items:center;height:26px;background:{bg}">'
        f'<p style="margin:0 0 0 6px;width:90px;font-size:14px">{label}</p>'
        f'<div style="position:relative;width:{width}px;height:26px">{ticks}{line}{dots}</div></div>'
    )


def decorate_metro(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        bs = r["ghsl_built_s"]
        rg = r["ae_ridge"]
        xg = r.get("ae_xgb", {"mean": float("nan"), "res": float("nan"), "nres": float("nan"), "n": 0})
        rk = r.get("ae_ring", {"mean": float("nan"), "res": float("nan"), "nres": float("nan"), "n": 0})
        s2 = r.get("ae_s2", {"mean": float("nan"), "res": float("nan"), "nres": float("nan"), "n": 0})
        rs = r["ae_ring_s2"]
        out.append(
            {
                "region": r["region"],
                "label": r.get("label", r["region"]),
                "bg": r.get("bg", "#F6F4EE"),
                "n": r["n"],
                "bs": bs,
                "rg": rg,
                "xg": xg,
                "rk": rk,
                "s2": s2,
                "rs": rs,
                "margin": float(rs["mean"] - bs["mean"]),
            }
        )
    return out


def metro_table(rows: list[dict]) -> str:
    models = ("bs", "rg", "xg", "rk", "s2", "rs")
    labels = ("BUILT-S", "Ridge", "XGB", "Ring", "AE+S2", "Rings+S2")
    head = (
        '<table style="width:100%;border-collapse:collapse;margin-top:8px">'
        '<tr><th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:left;font-weight:600;font-size:15px">Metro</th>'
        '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right;font-weight:600;font-size:15px">n</th>'
        + "".join(
            f'<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right;font-weight:600;'
            f'font-size:15px;border-left:1px solid #3a4a45" colspan="3">{lab}</th>'
            for lab in labels
        )
        + '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right;font-weight:600;'
        'font-size:15px;border-left:1px solid #3a4a45">Margin</th></tr>'
        '<tr style="background:#1b2a26;color:#EDEBE4"><td></td><td></td>'
        + "".join(
            '<td style="padding:4px 10px;text-align:right;font-size:12px;border-left:1px solid #3a4a45">res</td>'
            '<td style="padding:4px 10px;text-align:right;font-size:12px">nres</td>'
            '<td style="padding:4px 10px;text-align:right;font-size:12px">mean</td>'
            for _ in models
        )
        + '<td style="padding:4px 10px;text-align:right;font-size:12px;border-left:1px solid #3a4a45">RS−BS</td></tr>'
    )
    body = []
    for r in rows:
        objs = [r[k] for k in models]
        finite_means = [o["mean"] for o in objs if o["mean"] == o["mean"]]
        finite_res = [o["res"] for o in objs if o["res"] == o["res"]]
        finite_nres = [o["nres"] for o in objs if o["nres"] == o["nres"]]
        best_mean = min(finite_means) if finite_means else 0.0
        best_res = min(finite_res) if finite_res else 0.0
        best_nres = min(finite_nres) if finite_nres else 0.0

        def cell(v: float, best: float, border: bool = False) -> str:
            b = "border-left:1px solid #D9D5C8;" if border else ""
            if v != v:
                return (
                    f'<td style="padding:7px 10px;font-size:14px;text-align:right;'
                    f'font-variant-numeric:tabular-nums;{b}">—</td>'
                )
            bold = "<b>" if abs(v - best) < 1e-12 else ""
            bold_e = "</b>" if bold else ""
            return (
                f'<td style="padding:7px 10px;font-size:14px;text-align:right;'
                f'font-variant-numeric:tabular-nums;{b}">{bold}{f3(v)}{bold_e}</td>'
            )

        margin = r["margin"]
        mcol = "#C2691A" if margin >= 0 else "#9B2226"
        cells = "".join(
            f'{cell(o["res"], best_res, True)}{cell(o["nres"], best_nres)}{cell(o["mean"], best_mean)}'
            for o in objs
        )
        body.append(
            f'<tr style="background:{r["bg"]}">'
            f'<td style="padding:7px 10px;font-size:14px;font-weight:600">{r["label"]}</td>'
            f'<td style="padding:7px 10px;font-size:14px;text-align:right;font-variant-numeric:tabular-nums">{r["n"]:,}</td>'
            f"{cells}"
            f'<td style="padding:7px 10px;font-size:14px;text-align:right;font-variant-numeric:tabular-nums;'
            f'border-left:1px solid #D9D5C8;color:{mcol};font-weight:600">{margin:+.3f}</td></tr>'
        )
    return head + "".join(body) + "</table>"


def legend_block() -> str:
    dots = "".join(
        f'<div style="display:flex;align-items:center;gap:6px">'
        f'<div style="width:12px;height:12px;border-radius:50%;background:{c}"></div>{n}</div>'
        for c, n in zip(COLORS, NAMES)
    )
    return (
        '<div style="display:flex;flex-direction:column;gap:10px;margin:6px 0 10px">'
        f'<div style="display:flex;gap:16px;align-items:center;font-size:14px;color:#14231F;flex-wrap:wrap">{dots}</div>'
        "</div>"
    )


def metro_dumbbells(rows: list[dict], which: str) -> str:
    key = "res" if which == "res" else "nres"
    vals_all = []
    for r in rows:
        for k in ("bs", "rg", "xg", "rk", "s2", "rs"):
            v = r[k][key]
            if v == v:
                vals_all.append(v)
    vmax = max(vals_all) * 1.05 if vals_all else 1.0
    title = "Residential MAE" if which == "res" else "Non-residential MAE"
    body = "".join(
        dumbbell_row(
            r["label"],
            r["bg"],
            [r["bs"][key], r["rg"][key], r["xg"][key], r["rk"][key], r["s2"][key], r["rs"][key]],
            vmax,
        )
        for r in rows
    )
    return (
        f'<div style="flex:1;display:flex;flex-direction:column">'
        f'<p style="margin:0 0 6px;font-size:15px;font-weight:600">{title}</p>{body}</div>'
    )


def metro_card(title_html: str, blurb: str, note: str, rows: list[dict]) -> str:
    return (
        '<div style="display:flex;flex-direction:column;gap:10px;background:#FCFBF7;'
        'border:1px solid #D9D5C8;border-radius:12px;padding:24px">'
        f'<p style="margin:0;font-size:17px;font-weight:600">{title_html}</p>'
        f'<p style="margin:0 0 6px;font-size:15px;line-height:1.4;color:#4A5562">{blurb}</p>'
        f'<p style="margin:0;font-size:15px;line-height:1.45">{note}</p>'
        f"{legend_block()}"
        f'<div style="display:flex;gap:28px">{metro_dumbbells(rows, "res")}{metro_dumbbells(rows, "nres")}</div>'
        f"{metro_table(rows)}"
        "</div>"
    )


def _g(d: dict, key: str, field: str) -> float:
    if key not in d:
        return float("nan")
    return float(d[key][field])


def build_board(metrics: dict, run_stamp: str) -> str:
    building = metrics["buildings"]
    tile = metrics["tiles"]
    b = building["soft_pooled"]
    t = tile["soft_pooled"]
    n_b = int(metrics.get("eval_n_buildings") or building["n_shared"])
    n_t = int(metrics.get("eval_n_tiles") or tile["n_shared"])
    tile_cov = float(tile.get("coverage") or 0.0)
    tile_n_scored = int(t["ae_ring_s2"]["n"])
    n_feat = int(metrics.get("n_features") or len(metrics.get("features") or []))
    rings = metrics.get("rings_m") or [50, 100, 200]

    b_rows = decorate_metro(building["metro_soft"])
    t_rows = decorate_metro(tile["metro_soft"])
    b_beats_ghsl = sum(1 for r in b_rows if r["rs"]["mean"] + 1e-12 < r["bs"]["mean"])
    b_beats_s2 = sum(
        1
        for r in b_rows
        if r["rs"]["mean"] == r["rs"]["mean"]
        and r["s2"]["mean"] == r["s2"]["mean"]
        and r["rs"]["mean"] + 1e-12 < r["s2"]["mean"]
    )
    t_beats_ghsl = sum(
        1
        for r in t_rows
        if r["rs"]["mean"] == r["rs"]["mean"]
        and r["bs"]["mean"] == r["bs"]["mean"]
        and r["rs"]["mean"] + 1e-12 < r["bs"]["mean"]
    )

    intro = (
        "Same locked Section 3/4 Overture cohorts and soft GT. "
        "<b>Rings+S2</b> = AE inside-polygon mean (64-d) + mean/std in "
        f"{'/'.join(str(x) for x in rings)} m rings (448-d) + Sentinel-2 NRES-rule features "
        f"({n_feat} total); XGB LORO on soft fractions "
        f"(train n={int(metrics.get('train_n_buildings') or 0):,}). "
        "Targets industrial-park / RES context that cell-alone misses. "
        "Margin = <b>Rings+S2 mean − GHS-BUILT-S mean</b> (wine = Rings+S2 better)."
    )

    keys = ["ghsl_built_s", "ae_ridge", "ae_xgb", "ae_ring", "ae_s2", "ae_ring_s2"]
    scale_vals = []
    for pool in (b, t):
        for k in keys:
            if k in pool:
                scale_vals.extend([_g(pool, k, "res"), _g(pool, k, "nres")])
    scale = max([v for v in scale_vals if v == v] + [0.01])

    def panel_pair(title: str, pool: dict) -> str:
        rows_res = [
            ("GHS-BUILT-S", _g(pool, "ghsl_built_s", "res"), COLORS[0]),
            ("AE Ridge", _g(pool, "ae_ridge", "res"), COLORS[1]),
            ("AE XGB", _g(pool, "ae_xgb", "res"), COLORS[2]),
            ("AE Ring", _g(pool, "ae_ring", "res"), COLORS[3]),
            ("AE+S2", _g(pool, "ae_s2", "res"), COLORS[4]),
            ("Rings+S2", _g(pool, "ae_ring_s2", "res"), COLORS[5]),
        ]
        rows_nres = [
            ("GHS-BUILT-S", _g(pool, "ghsl_built_s", "nres"), COLORS[0]),
            ("AE Ridge", _g(pool, "ae_ridge", "nres"), COLORS[1]),
            ("AE XGB", _g(pool, "ae_xgb", "nres"), COLORS[2]),
            ("AE Ring", _g(pool, "ae_ring", "nres"), COLORS[3]),
            ("AE+S2", _g(pool, "ae_s2", "nres"), COLORS[4]),
            ("Rings+S2", _g(pool, "ae_ring_s2", "nres"), COLORS[5]),
        ]
        return (
            f'<p style="margin:4px 0 0;font-size:13px;font-weight:600;letter-spacing:1.5px;'
            f'text-transform:uppercase;color:#4A5562">{title}</p>'
            f'<div style="display:flex;gap:40px">'
            f'<div style="flex:1;display:flex;flex-direction:column;gap:6px">'
            f'<p style="margin:0 0 2px;font-size:15px;font-weight:600">Residential MAE</p>'
            + "".join(bar_row(n, v, c, scale) for n, v, c in rows_res if v == v)
            + "</div>"
            f'<div style="flex:1;display:flex;flex-direction:column;gap:6px">'
            f'<p style="margin:0 0 2px;font-size:15px;font-weight:600">Non-residential MAE</p>'
            + "".join(bar_row(n, v, c, scale) for n, v, c in rows_nres if v == v)
            + "</div></div>"
        )

    gap = (
        '<div style="display:flex;flex-direction:column;gap:10px;background:#FCFBF7;'
        'border:1px solid #D9D5C8;border-radius:12px;padding:24px">'
        '<p style="margin:0;font-size:17px;font-weight:600">Where the gap comes from: residential vs non-residential</p>'
        '<p style="margin:0 0 6px;font-size:15px;line-height:1.4;color:#4A5562">'
        "MAE on each fraction, locked Section 3/4 IDs (lower is better). Shared scale.</p>"
        + panel_pair(f"Building · {n_b:,} buildings", b)
        + panel_pair(
            f"100 m · {n_t:,} tiles (Rings+S2 scored n={tile_n_scored:,} · {100*tile_cov:.0f}% coverage)",
            t,
        )
        + "</div>"
    )

    b_card = metro_card(
        f'Per metro, building · <span style="color:#9B2226">Rings+S2 beats GHSL in {b_beats_ghsl} of {len(b_rows)}</span>'
        f' · beats AE+S2 in {b_beats_s2}/{len(b_rows)}',
        "Same Section 3/4 building IDs. Bold = best among available models. Margin = Rings+S2 − BUILT-S.",
        f"Pooled mean MAE: Rings+S2 <b>{f3(_g(b,'ae_ring_s2','mean'))}</b> · AE+S2 {f3(_g(b,'ae_s2','mean'))} · "
        f"Ring {f3(_g(b,'ae_ring','mean'))} · BUILT-S {f3(_g(b,'ghsl_built_s','mean'))}.",
        b_rows,
    )
    t_card = metro_card(
        f'Per metro, 100 m · <span style="color:#9B2226">Rings+S2 beats GHSL in {t_beats_ghsl} of {len(t_rows)}</span>'
        f' <span style="color:#4A5562;font-weight:400">(tile coverage {100*tile_cov:.0f}%)</span>',
        "Same Section 3/4 tile IDs. Rings+S2 tile preds = mean of building OOF in tile.",
        f"Pooled mean MAE on scored tiles: Rings+S2 <b>{f3(_g(t,'ae_ring_s2','mean'))}</b> (n={tile_n_scored:,}) · "
        f"BUILT-S {f3(_g(t,'ghsl_built_s','mean'))} (n={n_t:,}).",
        t_rows,
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>AE rings + S2 · Fraction MAE in Overture polygon space</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&amp;family=Source+Serif+4:wght@400;600&amp;display=swap">
<style>body{{margin:0}} table{{border-collapse:collapse;width:100%}} td{{border-bottom:1px solid #D9D5C8}} code{{font-size:13px}}</style></head><body>
<div style="width:1200px;min-height:0;box-sizing:border-box;padding:56px;background:#F6F4EE;color:#14231F;font-family:'IBM Plex Sans',Arial,sans-serif;display:flex;flex-direction:column;gap:28px">
<div style="display:flex;flex-direction:column;gap:8px">
<p style="margin:0;font-size:15px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#9B2226">Section 7 · Week 10</p>
<h1 style="margin:0;font-family:'Source Serif 4',Georgia,serif;font-size:48px;font-weight:600;line-height:1.1">Multi-scale AE rings + Sentinel-2</h1>
<p style="margin:0;font-size:18px;line-height:1.45;color:#4A5562">{intro}</p>
</div>
{gap}
{b_card}
{t_card}
<p style="margin:12px 0 0;font-size:15px;color:#4A5562">AlphaEarth · 50/100/200 m rings · Sentinel-2 2018 L2A · XGB LORO · locked Section 3/4 IDs · {run_stamp}</p>
</div></body></html>
"""


def inject_canvas(board: str, height: int) -> None:
    html = CANVAS.read_text()
    compressed = base64.b64encode(gzip.compress(board.encode("utf-8"), compresslevel=9)).decode("ascii")

    man_m = re.search(r'(<script type="__bundler/manifest">)(.*?)(</script>)', html, re.S)
    man = json.loads(man_m.group(2))
    man[SECTION7_ID] = {"mime": "text/html", "compressed": True, "data": compressed}
    html = html[: man_m.start(2)] + json.dumps(man, separators=(",", ":")) + html[man_m.end(2) :]

    po_m = re.search(r'(<script type="__bundler/page_order">)(.*?)(</script>)', html, re.S)
    order = json.loads(po_m.group(2))
    if SECTION7_ID in order:
        order = [x for x in order if x != SECTION7_ID]
    if SECTION6_ID not in order:
        raise SystemExit("Section 6 id missing from page_order — inject Section 6 first")
    i = order.index(SECTION6_ID) + 1
    order.insert(i, SECTION7_ID)
    html = html[: po_m.start(2)] + json.dumps(order, separators=(",", ":")) + html[po_m.end(2) :]

    tpl_m = re.search(r'(<script type="__bundler/template">)(.*?)(</script>)', html, re.S)
    template = json.loads(tpl_m.group(2))
    template = re.sub(
        rf'<section class="board"><h2>7 .*?</h2><iframe src="about:blank#{SECTION7_ID}".*?</iframe></section>\n?',
        "",
        template,
        count=1,
        flags=re.S,
    )
    if "7 Generalisation performance" in template:
        mapping = [
            ("11 GHSL BUILT-S vs AlphaEarth (AE cell centres)", "12 GHSL BUILT-S vs AlphaEarth (AE cell centres)"),
            ("10 Training dynamics and loss function", "11 Training dynamics and loss function"),
            ("9 Country-wise LORO and cross-country generalisation", "10 Country-wise LORO and cross-country generalisation"),
            ("8 Embedding dynamics", "9 Embedding dynamics"),
            ("7 Generalisation performance", "8 Generalisation performance"),
        ]
        for old, new in mapping:
            if old in template:
                template = template.replace(old, new, 1)

    iframe = (
        f'<section class="board"><h2>7 Multi-scale AE rings + Sentinel-2</h2>'
        f'<iframe src="about:blank#{SECTION7_ID}" title="7 Multi-scale AE rings + Sentinel-2" '
        f'width="1200" height="{height}" loading="eager" '
        f'style="width:1200px;height:{height}px"></iframe></section>\n'
    )
    m = re.search(
        rf'<section class="board"><h2>6 .*?</h2><iframe src="about:blank#{SECTION6_ID}".*?</iframe></section>\n?',
        template,
        flags=re.S,
    )
    if not m:
        raise SystemExit("could not locate Section 6 block")
    template = template[: m.end()] + iframe + template[m.end() :]

    html = html[: tpl_m.start(2)] + json.dumps(template, separators=(",", ":")) + html[tpl_m.end(2) :]
    for old, new in (
        ("AlphaEarth Weeks 1–9 Canvas", "AlphaEarth Weeks 1–10 Canvas"),
        ("Loading AlphaEarth Weeks 1–9 Canvas...", "Loading AlphaEarth Weeks 1–10 Canvas..."),
    ):
        html = html.replace(old, new, 1)
    CANVAS.write_text(html)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics",
        type=Path,
        default=ROOT
        / "outputs"
        / "week10_ae_rings_s2_loro"
        / "latest"
        / "artifacts"
        / "section7_rings_s2_metrics.json",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--height", type=int, default=4000)
    ap.add_argument("--no-inject", action="store_true")
    args = ap.parse_args()

    metrics_path = args.metrics.resolve()
    metrics = json.loads(metrics_path.read_text())
    run_stamp = metrics_path.parent.parent.name
    board = build_board(metrics, run_stamp)
    out = args.out or (metrics_path.parent / "section7_rings_s2_board.html")
    out.write_text(board)
    print(f"board → {out}")
    if not args.no_inject:
        inject_canvas(board, args.height)
        print(f"canvas injected id={SECTION7_ID} → {CANVAS}")
    b = metrics["buildings"]["soft_pooled"]
    print(
        f"buildings: rs2={b['ae_ring_s2']['mean']:.3f} s2={b.get('ae_s2',{}).get('mean')} "
        f"ring={b.get('ae_ring',{}).get('mean')} ghsl={b['ghsl_built_s']['mean']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
