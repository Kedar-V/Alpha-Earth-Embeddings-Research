#!/usr/bin/env python3
"""Rebuild Section 3 board: building panel from balanced run + new 100 m metrics."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANVAS = ROOT / "notebooks" / "ghsl_built_s_vs_ae" / "AlphaEarth Weeks 1–5 Canvas.html"
SECTION3_ID = "6498697d-cc82-4e8e-8068-85e76b691923"
BUILDING_METRICS_DEFAULT = (
    ROOT
    / "outputs"
    / "week6_overture_polygon_mae_100m"
    / "20261110_044005"
    / "artifacts"
    / "section3_combined_metrics.json"
)


def f3(x: float) -> str:
    return f"{x:.3f}"


def bar_row(name: str, val: float, color: str, scale_max: float) -> str:
    w = max(2, int(round(320 * (val / scale_max)))) if scale_max > 0 else 2
    return (
        f'<div style="display:flex;align-items:center;gap:10px" title="{name}: MAE {f3(val)}">'
        f'<p style="margin:0;width:100px;font-size:14px">{name}</p>'
        f'<div style="width:{w}px;height:18px;background:{color};border-radius:0 4px 4px 0"></div>'
        f'<p style="margin:0;font-size:14px;font-weight:600;font-variant-numeric:tabular-nums">{f3(val)}</p>'
        f"</div>"
    )


def panel_pair(title: str, res: dict, nres: dict) -> str:
    scale = max(res["built_s"], res["ridge"], res["pls"], nres["built_s"], nres["ridge"], nres["pls"], 0.01)
    return (
        f'<p style="margin:4px 0 0;font-size:13px;font-weight:600;letter-spacing:1.5px;'
        f'text-transform:uppercase;color:#4A5562">{title}</p>'
        f'<div style="display:flex;gap:40px">'
        f'<div style="flex:1;display:flex;flex-direction:column;gap:6px">'
        f'<p style="margin:0 0 2px;font-size:15px;font-weight:600">Residential MAE</p>'
        f'{bar_row("GHS-BUILT-S", res["built_s"], "#C2691A", scale)}'
        f'{bar_row("AE Ridge", res["ridge"], "#1F6FA8", scale)}'
        f'{bar_row("AE PLS", res["pls"], "#4AA0DB", scale)}'
        f"</div>"
        f'<div style="flex:1;display:flex;flex-direction:column;gap:6px">'
        f'<p style="margin:0 0 2px;font-size:15px;font-weight:600">Non-residential MAE</p>'
        f'{bar_row("GHS-BUILT-S", nres["built_s"], "#C2691A", scale)}'
        f'{bar_row("AE Ridge", nres["ridge"], "#1F6FA8", scale)}'
        f'{bar_row("AE PLS", nres["pls"], "#4AA0DB", scale)}'
        f"</div></div>"
    )


def dumbbell_row(label: str, bg: str, vals: list[float], vmax: float) -> str:
    # vals: [bs, ridge, pls]
    width = 380.0
    colors = ["#C2691A", "#1F6FA8", "#4AA0DB"]
    xs = [max(0.0, min(width, (v / vmax) * width)) for v in vals]
    lo, hi = min(xs), max(xs)
    ticks = "".join(
        f'<div style="position:absolute;left:{i * width / 5:.1f}px;top:0;bottom:0;width:1px;background:#E5E1D6"></div>'
        for i in range(1, 5)
    )
    dots = "".join(
        f'<div style="position:absolute;left:{x - 5:.1f}px;top:8px;width:10px;height:10px;'
        f'border-radius:50%;background:{c}"></div>'
        for x, c in zip(xs, colors)
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


def metro_table(rows: list[dict]) -> str:
    head = (
        '<table style="width:100%;border-collapse:collapse;margin-top:8px">'
        '<tr><th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:left;font-weight:600;font-size:15px">Metro</th>'
        '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right;font-weight:600;font-size:15px">n</th>'
        '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right;font-weight:600;font-size:15px;border-left:1px solid #3a4a45" colspan="3">BUILT-S</th>'
        '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right;font-weight:600;font-size:15px;border-left:1px solid #3a4a45" colspan="3">Ridge</th>'
        '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right;font-weight:600;font-size:15px;border-left:1px solid #3a4a45" colspan="3">PLS</th>'
        '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right;font-weight:600;font-size:15px;border-left:1px solid #3a4a45">Margin</th></tr>'
        '<tr style="background:#1b2a26;color:#EDEBE4"><td></td><td></td>'
        '<td style="padding:4px 10px;text-align:right;font-size:12px;border-left:1px solid #3a4a45">res</td>'
        '<td style="padding:4px 10px;text-align:right;font-size:12px">nres</td>'
        '<td style="padding:4px 10px;text-align:right;font-size:12px">mean</td>'
        '<td style="padding:4px 10px;text-align:right;font-size:12px;border-left:1px solid #3a4a45">res</td>'
        '<td style="padding:4px 10px;text-align:right;font-size:12px">nres</td>'
        '<td style="padding:4px 10px;text-align:right;font-size:12px">mean</td>'
        '<td style="padding:4px 10px;text-align:right;font-size:12px;border-left:1px solid #3a4a45">res</td>'
        '<td style="padding:4px 10px;text-align:right;font-size:12px">nres</td>'
        '<td style="padding:4px 10px;text-align:right;font-size:12px">mean</td>'
        '<td style="padding:4px 10px;text-align:right;font-size:12px;border-left:1px solid #3a4a45">AE−BS</td></tr>'
    )
    body = []
    for r in rows:
        bs, rg, pl = r["bs"], r["rg"], r["pl"]
        best_mean = min(bs["mean"], rg["mean"], pl["mean"])
        best_res = min(bs["res"], rg["res"], pl["res"])
        best_nres = min(bs["nres"], rg["nres"], pl["nres"])

        def cell(v: float, best: float, border: bool = False) -> str:
            b = "border-left:1px solid #D9D5C8;" if border else ""
            bold = "<b>" if abs(v - best) < 1e-12 else ""
            bold_e = "</b>" if bold else ""
            return (
                f'<td style="padding:7px 10px;font-size:15px;text-align:right;'
                f'font-variant-numeric:tabular-nums;{b}">{bold}{f3(v)}{bold_e}</td>'
            )

        margin = r["margin"]
        mcol = "#C2691A" if margin >= 0 else "#1F6FA8"
        body.append(
            f'<tr style="background:{r["bg"]}">'
            f'<td style="padding:7px 10px;font-size:15px;font-weight:600">{r["label"]}</td>'
            f'<td style="padding:7px 10px;font-size:15px;text-align:right;font-variant-numeric:tabular-nums">{r["n"]:,}</td>'
            f'{cell(bs["res"], best_res, True)}{cell(bs["nres"], best_nres)}{cell(bs["mean"], best_mean)}'
            f'{cell(rg["res"], best_res, True)}{cell(rg["nres"], best_nres)}{cell(rg["mean"], best_mean)}'
            f'{cell(pl["res"], best_res, True)}{cell(pl["nres"], best_nres)}{cell(pl["mean"], best_mean)}'
            f'<td style="padding:7px 10px;font-size:15px;text-align:right;font-variant-numeric:tabular-nums;'
            f'border-left:1px solid #D9D5C8;color:{mcol};font-weight:600">{margin:+.3f}</td></tr>'
        )
    return head + "".join(body) + "</table>"


def legend_block() -> str:
    return (
        '<div style="display:flex;flex-direction:column;gap:10px;margin:6px 0 10px">'
        '<div style="display:flex;gap:20px;align-items:center;font-size:14px;color:#14231F">'
        '<div style="display:flex;align-items:center;gap:6px"><div style="width:12px;height:12px;border-radius:50%;background:#C2691A"></div>GHS-BUILT-S</div>'
        '<div style="display:flex;align-items:center;gap:6px"><div style="width:12px;height:12px;border-radius:50%;background:#1F6FA8"></div>AE Ridge</div>'
        '<div style="display:flex;align-items:center;gap:6px"><div style="width:12px;height:12px;border-radius:50%;background:#4AA0DB"></div>AE PLS</div>'
        "</div>"
        '<div style="display:flex;gap:14px;flex-wrap:wrap;align-items:center;font-size:13px;color:#14231F">'
        '<span style="color:#4A5562">Rows grouped by region:</span>'
        '<div style="display:flex;align-items:center;gap:5px"><div style="width:14px;height:14px;border-radius:3px;background:#ECE8F4;border:1px solid #D9D5C8"></div>Northeast (NE)</div>'
        '<div style="display:flex;align-items:center;gap:5px"><div style="width:14px;height:14px;border-radius:3px;background:#E2EFE6;border:1px solid #D9D5C8"></div>Southeast (SE)</div>'
        '<div style="display:flex;align-items:center;gap:5px"><div style="width:14px;height:14px;border-radius:3px;background:#F3EDD5;border:1px solid #D9D5C8"></div>South (S)</div>'
        '<div style="display:flex;align-items:center;gap:5px"><div style="width:14px;height:14px;border-radius:3px;background:#F3E3E7;border:1px solid #D9D5C8"></div>Midwest (MW)</div>'
        '<div style="display:flex;align-items:center;gap:5px"><div style="width:14px;height:14px;border-radius:3px;background:#EFE4D9;border:1px solid #D9D5C8"></div>Southwest (SW)</div>'
        '<div style="display:flex;align-items:center;gap:5px"><div style="width:14px;height:14px;border-radius:3px;background:#E1ECEE;border:1px solid #D9D5C8"></div>West (W)</div>'
        '<div style="display:flex;align-items:center;gap:5px"><div style="width:14px;height:14px;border-radius:3px;background:#E9EDDC;border:1px solid #D9D5C8"></div>Northwest (NW)</div>'
        "</div></div>"
    )


def metro_dumbbells(rows: list[dict], which: str) -> str:
    key = "res" if which == "res" else "nres"
    vals_all = [r["bs"][key] for r in rows] + [r["rg"][key] for r in rows] + [r["pl"][key] for r in rows]
    vmax = max(vals_all) * 1.05 if vals_all else 1.0
    title = "Residential MAE" if which == "res" else "Non-residential MAE"
    body = "".join(
        dumbbell_row(r["label"], r["bg"], [r["bs"][key], r["rg"][key], r["pl"][key]], vmax) for r in rows
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


def win_note(rows: list[dict], scale: str) -> str:
    wins_res = sum(1 for r in rows if r["bs"]["res"] <= min(r["rg"]["res"], r["pl"]["res"]) + 1e-12)
    wins_nres = sum(1 for r in rows if r["bs"]["nres"] <= min(r["rg"]["nres"], r["pl"]["nres"]) + 1e-12)
    ae_wins = []
    for r in rows:
        best_ae = min(r["rg"]["mean"], r["pl"]["mean"])
        if best_ae + 1e-12 < r["bs"]["mean"]:
            who = "AE Ridge" if r["rg"]["mean"] <= r["pl"]["mean"] else "AE PLS"
            ae_wins.append(f'{r["label"]} ({who}, {f3(best_ae)} vs {f3(r["bs"]["mean"])}, {r["n"]:,} {scale})')
    ae_txt = "; ".join(ae_wins) if ae_wins else "none"
    return (
        f"At {scale} BUILT-S is best on residential in <b>{wins_res} of {len(rows)}</b> metros "
        f"and on non-residential in <b>{wins_nres} of {len(rows)}</b>. AE wins {ae_txt}."
    )


def build_board(building: dict, tile: dict, run_stamp: str) -> str:
    b = building["soft_pooled"]
    t = tile["soft_pooled"]
    n_b = building["n_shared"]
    n_t = tile["n_tiles"]
    b_wins = building["soft_wins"]
    t_wins = tile["soft_wins"]

    intro = (
        "Residential and non-residential coverage fractions after aggregating AE Ridge, AE PLS, and "
        "GHS-BUILT-S into each Overture footprint. Building GT is the mean of week-4 cell fractions "
        "inside the polygon (buildings balanced 1:1 on <code>y_res</code>). For 100 m: start from the "
        "unbalanced soft-covered building universe, soft-argmax 1:1 <b>cells</b> per metro "
        "(<code>frac_res ≥ frac_nres</code> → RES), mean-pool into UTM 100 m tiles (min ≥ 10; "
        "min ≥ 20 leaves Atlanta/Phoenix/Seattle with 0 RES soft-argmax tiles), then "
        "soft-argmax 1:1 <b>tiles</b> per metro. BUILT-S is the building-zonal mean (polygon path). "
        f"Cohort: building n = {n_b:,} · tiles n = {n_t:,}."
    )

    gap = (
        '<div style="display:flex;flex-direction:column;gap:10px;background:#FCFBF7;'
        'border:1px solid #D9D5C8;border-radius:12px;padding:24px">'
        '<p style="margin:0;font-size:17px;font-weight:600">Where the gap comes from: residential vs non-residential</p>'
        '<p style="margin:0 0 6px;font-size:15px;line-height:1.4;color:#4A5562">'
        "MAE on each fraction, pooled over each balanced shared set (lower is better). "
        "All four panels share one scale.</p>"
        + panel_pair(
            f"Building · {n_b:,} buildings",
            {"built_s": b["built_s"]["res"], "ridge": b["ridge"]["res"], "pls": b["pls"]["res"]},
            {"built_s": b["built_s"]["nres"], "ridge": b["ridge"]["nres"], "pls": b["pls"]["nres"]},
        )
        + panel_pair(
            f"100 m cell+tile soft-argmax 1:1 · {n_t:,} tiles",
            {"built_s": t["built_s"]["res"], "ridge": t["ridge"]["res"], "pls": t["pls"]["res"]},
            {"built_s": t["built_s"]["nres"], "ridge": t["ridge"]["nres"], "pls": t["pls"]["nres"]},
        )
        + "</div>"
    )

    # Fix shared scale across all four panels
    scale = max(
        b["built_s"]["res"],
        b["ridge"]["res"],
        b["pls"]["res"],
        b["built_s"]["nres"],
        b["ridge"]["nres"],
        b["pls"]["nres"],
        t["built_s"]["res"],
        t["ridge"]["res"],
        t["pls"]["res"],
        t["built_s"]["nres"],
        t["ridge"]["nres"],
        t["pls"]["nres"],
        0.01,
    )

    def panel_pair_fixed(title: str, res: dict, nres: dict) -> str:
        return (
            f'<p style="margin:4px 0 0;font-size:13px;font-weight:600;letter-spacing:1.5px;'
            f'text-transform:uppercase;color:#4A5562">{title}</p>'
            f'<div style="display:flex;gap:40px">'
            f'<div style="flex:1;display:flex;flex-direction:column;gap:6px">'
            f'<p style="margin:0 0 2px;font-size:15px;font-weight:600">Residential MAE</p>'
            f'{bar_row("GHS-BUILT-S", res["built_s"], "#C2691A", scale)}'
            f'{bar_row("AE Ridge", res["ridge"], "#1F6FA8", scale)}'
            f'{bar_row("AE PLS", res["pls"], "#4AA0DB", scale)}'
            f"</div>"
            f'<div style="flex:1;display:flex;flex-direction:column;gap:6px">'
            f'<p style="margin:0 0 2px;font-size:15px;font-weight:600">Non-residential MAE</p>'
            f'{bar_row("GHS-BUILT-S", nres["built_s"], "#C2691A", scale)}'
            f'{bar_row("AE Ridge", nres["ridge"], "#1F6FA8", scale)}'
            f'{bar_row("AE PLS", nres["pls"], "#4AA0DB", scale)}'
            f"</div></div>"
        )

    gap = (
        '<div style="display:flex;flex-direction:column;gap:10px;background:#FCFBF7;'
        'border:1px solid #D9D5C8;border-radius:12px;padding:24px">'
        '<p style="margin:0;font-size:17px;font-weight:600">Where the gap comes from: residential vs non-residential</p>'
        '<p style="margin:0 0 6px;font-size:15px;line-height:1.4;color:#4A5562">'
        "MAE on each fraction, pooled over each balanced shared set (lower is better). "
        "All four panels share one scale.</p>"
        + panel_pair_fixed(
            f"Building · {n_b:,} buildings",
            {"built_s": b["built_s"]["res"], "ridge": b["ridge"]["res"], "pls": b["pls"]["res"]},
            {"built_s": b["built_s"]["nres"], "ridge": b["ridge"]["nres"], "pls": b["pls"]["nres"]},
        )
        + panel_pair_fixed(
            f"100 m cell+tile soft-argmax 1:1 · {n_t:,} tiles",
            {"built_s": t["built_s"]["res"], "ridge": t["ridge"]["res"], "pls": t["pls"]["res"]},
            {"built_s": t["built_s"]["nres"], "ridge": t["ridge"]["nres"], "pls": t["pls"]["nres"]},
        )
        + "</div>"
    )

    b_note = (
        f"BUILT-S has the lowest mean MAE in <b>{b_wins} of {len(building['metro_soft'])}</b> metros. "
        "Buildings are 1:1 on <code>y_res</code> per metro."
    )
    dropped = [r for r, c in (tile.get("balance_regions") or {}).items() if c.get("n_per_class", 0) == 0]
    # fall back: infer from metro list vs canonical 10
    present = {r["region"] for r in tile["metro_soft"]}
    all10 = {"Boston_NE","NYC_NE","Atlanta_SE","Houston_S","Chicago_MW","Phoenix_SW","Denver_W","LA_W","SF_W","Seattle_NW"}
    missing = sorted(all10 - present)
    miss_txt = (
        f" Dropped metros with no soft-argmax RES tiles: {', '.join(missing)}."
        if missing else ""
    )
    tiny = [r["label"] for r in tile["metro_soft"] if r["n"] <= 4]
    tiny_txt = f" Tiny cohorts (n≤4): {', '.join(tiny)}." if tiny else ""
    t_note = (
        win_note(tile["metro_soft"], "tiles")
        + " Cells and tiles are both soft-argmax 1:1 per metro."
        + miss_txt
        + tiny_txt
    )

    b_card = metro_card(
        f'Per metro, building · <span style="color:#C2691A">BUILT-S wins {b_wins} of {len(building["metro_soft"])}</span>',
        "MAE on the residential (res) and non-residential (nres) fractions and their mean, which decides the winner. "
        "Bold = best of the three models in that column type. Margin = best AE mean − BUILT-S mean "
        "(orange = BUILT-S ahead, blue = AE ahead).",
        b_note,
        building["metro_soft"],
    )
    t_card = metro_card(
        f'Per metro, 100 m cell+tile soft-argmax 1:1 · <span style="color:#C2691A">BUILT-S wins {t_wins} of {len(tile["metro_soft"])}</span>',
        "Cells soft-argmax 1:1 per metro, then mean-pooled into UTM 100 m tiles (min ≥ 10), then tiles "
        "soft-argmax 1:1 per metro. Same columns as the building table.",
        t_note,
        tile["metro_soft"],
    )

    # Estimate height from content (~building card similar + tile card); keep tight.
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Fraction MAE in Overture polygon space</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&amp;family=Source+Serif+4:wght@400;600&amp;display=swap">
<style>body{{margin:0}} table{{border-collapse:collapse;width:100%}} td{{border-bottom:1px solid #D9D5C8}}</style></head><body>
<div style="width:1200px;min-height:0;box-sizing:border-box;padding:56px;background:#F6F4EE;color:#14231F;font-family:'IBM Plex Sans',Arial,sans-serif;display:flex;flex-direction:column;gap:28px">
<div style="display:flex;flex-direction:column;gap:8px">
<p style="margin:0;font-size:15px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#1F6FA8">Section 3 · Week 6</p>
<h1 style="margin:0;font-family:'Source Serif 4',Georgia,serif;font-size:48px;font-weight:600;line-height:1.1">Fraction MAE in Overture polygon space</h1>
<p style="margin:0;font-size:18px;line-height:1.45;color:#4A5562">{intro}</p>
</div>
{gap}
{b_card}
{t_card}
<p style="margin:12px 0 0;font-size:15px;color:#4A5562">AlphaEarth embeddings · week 6 Overture-polygon fraction MAE · building y_res 1:1 + 100 m cell+tile soft-argmax 1:1 · {run_stamp}</p>
</div></body></html>
"""


def inject_canvas(board: str, height: int) -> None:
    html = CANVAS.read_text()
    compressed = base64.b64encode(gzip.compress(board.encode("utf-8"), compresslevel=9)).decode("ascii")
    man_m = re.search(r'(<script type="__bundler/manifest">)(.*?)(</script>)', html, re.S)
    man = json.loads(man_m.group(2))
    man[SECTION3_ID] = {"mime": "text/html", "compressed": True, "data": compressed}
    html = html[: man_m.start(2)] + json.dumps(man, separators=(",", ":")) + html[man_m.end(2) :]

    # update iframe height in escaped template (any prior height)
    pat = re.compile(
        rf'(<iframe src=\\"about:blank#{SECTION3_ID}\\" title=\\"3 Fraction MAE in Overture polygon space\\" '
        rf'width=\\"1200\\" height=\\")(\d+)(\\" loading=\\"eager\\" style=\\"width:1200px;height:)(\d+)(px\\"></iframe>)'
    )
    html2, n = pat.subn(rf"\g<1>{height}\g<3>{height}\g<5>", html, count=1)
    if n != 1:
        raise SystemExit(f"iframe height replace failed (n={n})")
    CANVAS.write_text(html2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile-run", type=Path, required=True)
    ap.add_argument("--building-metrics", type=Path, default=BUILDING_METRICS_DEFAULT)
    ap.add_argument("--height", type=int, default=2860)
    args = ap.parse_args()

    tile = json.loads((args.tile_run / "artifacts" / "section3_100m_metrics.json").read_text())
    combined_src = json.loads(args.building_metrics.read_text())
    building = combined_src["building"] if "building" in combined_src else combined_src

    run_stamp = args.tile_run.name
    board = build_board(building, tile, run_stamp)
    out = args.tile_run / "artifacts" / "section3_board.html"
    out.write_text(board)

    combined = {
        "building": building,
        "tile_100m": tile,
        "protocol": {
            "building_balance": "y_res_1to1_per_metro",
            "cell_balance": "soft_argmax_1to1_per_metro",
            "tile_balance": "soft_argmax_1to1_per_metro",
            "tile_run": str(args.tile_run),
        },
    }
    (args.tile_run / "artifacts" / "section3_combined_metrics.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True)
    )

    inject_canvas(board, args.height)
    print(f"board → {out}")
    print(f"canvas injected id={SECTION3_ID} height={args.height}")
    print(f"tiles n={tile['n_tiles']} wins={tile['soft_wins']}/10 mean BS={tile['soft_pooled']['built_s']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
