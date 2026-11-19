#!/usr/bin/env python3
"""Build Section 4 canvas board: Section 3 layout + AE XGB vs Ridge/PLS/BUILT-S."""

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
# Stable id so re-runs replace the same page instead of stacking duplicates.
SECTION4_ID = "a7c4e2b1-5d8f-4a91-9c3e-2f6b8d0e1a45"


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


def dumbbell_row(label: str, bg: str, vals: list[float], vmax: float) -> str:
    # vals: [bs, ridge, pls, xgb]
    width = 380.0
    colors = ["#C2691A", "#1F6FA8", "#4AA0DB", "#8E44AD"]
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
        '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right;font-weight:600;font-size:15px;border-left:1px solid #3a4a45" colspan="3">XGB</th>'
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
        '<td style="padding:4px 10px;text-align:right;font-size:12px;border-left:1px solid #3a4a45">res</td>'
        '<td style="padding:4px 10px;text-align:right;font-size:12px">nres</td>'
        '<td style="padding:4px 10px;text-align:right;font-size:12px">mean</td>'
        '<td style="padding:4px 10px;text-align:right;font-size:12px;border-left:1px solid #3a4a45">XGB−Rg</td></tr>'
    )
    body = []
    for r in rows:
        bs, rg, pl, xg = r["bs"], r["rg"], r["pl"], r["xg"]
        best_mean = min(bs["mean"], rg["mean"], pl["mean"], xg["mean"])
        best_res = min(bs["res"], rg["res"], pl["res"], xg["res"])
        best_nres = min(bs["nres"], rg["nres"], pl["nres"], xg["nres"])

        def cell(v: float, best: float, border: bool = False) -> str:
            b = "border-left:1px solid #D9D5C8;" if border else ""
            bold = "<b>" if abs(v - best) < 1e-12 else ""
            bold_e = "</b>" if bold else ""
            return (
                f'<td style="padding:7px 10px;font-size:14px;text-align:right;'
                f'font-variant-numeric:tabular-nums;{b}">{bold}{f3(v)}{bold_e}</td>'
            )

        margin = r["margin"]
        mcol = "#1F6FA8" if margin >= 0 else "#8E44AD"  # + = Ridge better; − = XGB better
        body.append(
            f'<tr style="background:{r["bg"]}">'
            f'<td style="padding:7px 10px;font-size:14px;font-weight:600">{r["label"]}</td>'
            f'<td style="padding:7px 10px;font-size:14px;text-align:right;font-variant-numeric:tabular-nums">{r["n"]:,}</td>'
            f'{cell(bs["res"], best_res, True)}{cell(bs["nres"], best_nres)}{cell(bs["mean"], best_mean)}'
            f'{cell(rg["res"], best_res, True)}{cell(rg["nres"], best_nres)}{cell(rg["mean"], best_mean)}'
            f'{cell(pl["res"], best_res, True)}{cell(pl["nres"], best_nres)}{cell(pl["mean"], best_mean)}'
            f'{cell(xg["res"], best_res, True)}{cell(xg["nres"], best_nres)}{cell(xg["mean"], best_mean)}'
            f'<td style="padding:7px 10px;font-size:14px;text-align:right;font-variant-numeric:tabular-nums;'
            f'border-left:1px solid #D9D5C8;color:{mcol};font-weight:600">{margin:+.3f}</td></tr>'
        )
    return head + "".join(body) + "</table>"


def legend_block() -> str:
    return (
        '<div style="display:flex;flex-direction:column;gap:10px;margin:6px 0 10px">'
        '<div style="display:flex;gap:20px;align-items:center;font-size:14px;color:#14231F;flex-wrap:wrap">'
        '<div style="display:flex;align-items:center;gap:6px"><div style="width:12px;height:12px;border-radius:50%;background:#C2691A"></div>GHS-BUILT-S</div>'
        '<div style="display:flex;align-items:center;gap:6px"><div style="width:12px;height:12px;border-radius:50%;background:#1F6FA8"></div>AE Ridge</div>'
        '<div style="display:flex;align-items:center;gap:6px"><div style="width:12px;height:12px;border-radius:50%;background:#4AA0DB"></div>AE PLS</div>'
        '<div style="display:flex;align-items:center;gap:6px"><div style="width:12px;height:12px;border-radius:50%;background:#8E44AD"></div>AE XGB</div>'
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
    vals_all = []
    for r in rows:
        vals_all.extend([r["bs"][key], r["rg"][key], r["pl"][key], r["xg"][key]])
    vmax = max(vals_all) * 1.05 if vals_all else 1.0
    title = "Residential MAE" if which == "res" else "Non-residential MAE"
    body = "".join(
        dumbbell_row(
            r["label"],
            r["bg"],
            [r["bs"][key], r["rg"][key], r["pl"][key], r["xg"][key]],
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


def win_note_xgb(rows: list[dict], scale: str) -> str:
    xgb_beats = sum(1 for r in rows if r["xg"]["mean"] + 1e-12 < r["rg"]["mean"])
    ridge_beats = sum(1 for r in rows if r["rg"]["mean"] + 1e-12 < r["xg"]["mean"])
    ties = len(rows) - xgb_beats - ridge_beats
    bs_wins = sum(
        1
        for r in rows
        if r["bs"]["mean"] <= min(r["rg"]["mean"], r["pl"]["mean"], r["xg"]["mean"]) + 1e-12
    )
    return (
        f"At {scale} XGB beats Ridge on mean MAE in <b>{xgb_beats} of {len(rows)}</b> metros "
        f"(Ridge ahead in {ridge_beats}; ties {ties}). "
        f"BUILT-S still has the lowest mean in <b>{bs_wins} of {len(rows)}</b>."
    )


def build_board(metrics: dict, run_stamp: str) -> str:
    building = metrics["building"]
    tile = metrics["tile_100m"]
    b = building["soft_pooled"]
    t = tile["soft_pooled"]
    n_b = building["n_shared"]
    n_t = tile["n_tiles"]
    b_xgb_wins = building["xgb_beats_ridge"]
    t_xgb_wins = tile["xgb_beats_ridge"]

    intro = (
        "Same Section 3 Overture cohorts and soft GT, with AE XGBoost LORO OOF added beside AE Ridge, "
        "AE PLS, and GHS-BUILT-S. Buildings are the frozen y_res 1:1 set (n = "
        f"{n_b:,}); 100 m tiles are the frozen cell+tile soft-argmax 1:1 set (n = {n_t:,}). "
        "Margin in the tables is <b>XGB mean − Ridge mean</b> (purple = XGB better, blue = Ridge better)."
    )

    scale = max(
        b["built_s"]["res"],
        b["ridge"]["res"],
        b["pls"]["res"],
        b["xgb"]["res"],
        b["built_s"]["nres"],
        b["ridge"]["nres"],
        b["pls"]["nres"],
        b["xgb"]["nres"],
        t["built_s"]["res"],
        t["ridge"]["res"],
        t["pls"]["res"],
        t["xgb"]["res"],
        t["built_s"]["nres"],
        t["ridge"]["nres"],
        t["pls"]["nres"],
        t["xgb"]["nres"],
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
            f'{bar_row("AE XGB", res["xgb"], "#8E44AD", scale)}'
            f"</div>"
            f'<div style="flex:1;display:flex;flex-direction:column;gap:6px">'
            f'<p style="margin:0 0 2px;font-size:15px;font-weight:600">Non-residential MAE</p>'
            f'{bar_row("GHS-BUILT-S", nres["built_s"], "#C2691A", scale)}'
            f'{bar_row("AE Ridge", nres["ridge"], "#1F6FA8", scale)}'
            f'{bar_row("AE PLS", nres["pls"], "#4AA0DB", scale)}'
            f'{bar_row("AE XGB", nres["xgb"], "#8E44AD", scale)}'
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
            {
                "built_s": b["built_s"]["res"],
                "ridge": b["ridge"]["res"],
                "pls": b["pls"]["res"],
                "xgb": b["xgb"]["res"],
            },
            {
                "built_s": b["built_s"]["nres"],
                "ridge": b["ridge"]["nres"],
                "pls": b["pls"]["nres"],
                "xgb": b["xgb"]["nres"],
            },
        )
        + panel_pair_fixed(
            f"100 m cell+tile soft-argmax 1:1 · {n_t:,} tiles",
            {
                "built_s": t["built_s"]["res"],
                "ridge": t["ridge"]["res"],
                "pls": t["pls"]["res"],
                "xgb": t["xgb"]["res"],
            },
            {
                "built_s": t["built_s"]["nres"],
                "ridge": t["ridge"]["nres"],
                "pls": t["pls"]["nres"],
                "xgb": t["xgb"]["nres"],
            },
        )
        + "</div>"
    )

    b_note = (
        f"XGB beats Ridge on mean MAE in <b>{b_xgb_wins} of {len(building['metro_soft'])}</b> metros. "
        "Same building IDs as Section 3 (y_res 1:1 per metro)."
    )
    t_note = win_note_xgb(tile["metro_soft"], "tiles") + " Same tile IDs as Section 3."

    b_card = metro_card(
        f'Per metro, building · <span style="color:#8E44AD">XGB beats Ridge in {b_xgb_wins} of {len(building["metro_soft"])}</span>',
        "MAE on the residential (res) and non-residential (nres) fractions and their mean. "
        "Bold = best of the four models in that column type. Margin = XGB mean − Ridge mean "
        "(purple = XGB ahead, blue = Ridge ahead).",
        b_note,
        building["metro_soft"],
    )
    t_card = metro_card(
        f'Per metro, 100 m · <span style="color:#8E44AD">XGB beats Ridge in {t_xgb_wins} of {len(tile["metro_soft"])}</span>',
        "Same Section 3 100 m protocol and tile IDs. Columns match the building table, with XGB added.",
        t_note,
        tile["metro_soft"],
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>XGB vs Ridge · Fraction MAE in Overture polygon space</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&amp;family=Source+Serif+4:wght@400;600&amp;display=swap">
<style>body{{margin:0}} table{{border-collapse:collapse;width:100%}} td{{border-bottom:1px solid #D9D5C8}}</style></head><body>
<div style="width:1200px;min-height:0;box-sizing:border-box;padding:56px;background:#F6F4EE;color:#14231F;font-family:'IBM Plex Sans',Arial,sans-serif;display:flex;flex-direction:column;gap:28px">
<div style="display:flex;flex-direction:column;gap:8px">
<p style="margin:0;font-size:15px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#8E44AD">Section 4 · Week 7</p>
<h1 style="margin:0;font-family:'Source Serif 4',Georgia,serif;font-size:48px;font-weight:600;line-height:1.1">XGBoost vs Ridge in Overture polygon space</h1>
<p style="margin:0;font-size:18px;line-height:1.45;color:#4A5562">{intro}</p>
</div>
{gap}
{b_card}
{t_card}
<p style="margin:12px 0 0;font-size:15px;color:#4A5562">AlphaEarth embeddings · week 7 XGB LORO on Section 3 eval sets · {run_stamp}</p>
</div></body></html>
"""


def _renumber_later_sections(template: str) -> str:
    """Shift old sections 4–8 → 5–9 in the escaped template HTML."""
    # Work on the unescaped form inside the JSON string carefully via regex on escaped quotes.
    mapping = [
        ("8 GHSL BUILT-S vs AlphaEarth (AE cell centres)", "9 GHSL BUILT-S vs AlphaEarth (AE cell centres)"),
        ("7 Training dynamics and loss function", "8 Training dynamics and loss function"),
        ("6 Country-wise LORO and cross-country generalisation", "7 Country-wise LORO and cross-country generalisation"),
        ("5 Embedding dynamics", "6 Embedding dynamics"),
        ("4 Generalisation performance", "5 Generalisation performance"),
    ]
    out = template
    for old, new in mapping:
        out2, n = re.subn(re.escape(old), new, out)
        if n < 1:
            raise SystemExit(f"failed to renumber section title: {old!r} (n={n})")
        out = out2
    return out


def inject_canvas(board: str, height: int) -> None:
    html = CANVAS.read_text()
    compressed = base64.b64encode(gzip.compress(board.encode("utf-8"), compresslevel=9)).decode("ascii")

    man_m = re.search(r'(<script type="__bundler/manifest">)(.*?)(</script>)', html, re.S)
    man = json.loads(man_m.group(2))
    man[SECTION4_ID] = {"mime": "text/html", "compressed": True, "data": compressed}
    html = html[: man_m.start(2)] + json.dumps(man, separators=(",", ":")) + html[man_m.end(2) :]

    po_m = re.search(r'(<script type="__bundler/page_order">)(.*?)(</script>)', html, re.S)
    order = json.loads(po_m.group(2))
    if SECTION4_ID in order:
        order = [x for x in order if x != SECTION4_ID]
    if SECTION3_ID not in order:
        raise SystemExit("Section 3 id missing from page_order")
    i = order.index(SECTION3_ID) + 1
    order.insert(i, SECTION4_ID)
    html = html[: po_m.start(2)] + json.dumps(order, separators=(",", ":")) + html[po_m.end(2) :]

    tpl_m = re.search(r'(<script type="__bundler/template">)(.*?)(</script>)', html, re.S)
    template = json.loads(tpl_m.group(2))
    if SECTION4_ID in template:
        # replace existing section 4 iframe block
        template = re.sub(
            rf'<section class="board"><h2>4 .*?</h2><iframe src="about:blank#{SECTION4_ID}".*?</iframe></section>\n?',
            "",
            template,
            count=1,
            flags=re.S,
        )
        # also undo prior renumber if re-running: leave as-is and re-apply carefully
    if "5 Generalisation performance" not in template and "4 Generalisation performance" in template:
        template = _renumber_later_sections(template)
    elif "5 Generalisation performance" not in template:
        raise SystemExit("unexpected template section titles")

    iframe = (
        f'<section class="board"><h2>4 XGBoost vs Ridge in Overture polygon space</h2>'
        f'<iframe src="about:blank#{SECTION4_ID}" title="4 XGBoost vs Ridge in Overture polygon space" '
        f'width="1200" height="{height}" loading="eager" '
        f'style="width:1200px;height:{height}px"></iframe></section>\n'
    )
    needle = f'<section class="board"><h2>5 Generalisation performance</h2>'
    if needle not in template:
        raise SystemExit("renumbered section 5 not found for insert point")
    if SECTION4_ID not in template:
        template = template.replace(needle, iframe + needle, 1)
    else:
        # update height only
        template = re.sub(
            rf'(<iframe src="about:blank#{SECTION4_ID}" title="4 XGBoost vs Ridge in Overture polygon space" '
            rf'width="1200" height=")(\d+)(" loading="eager" style="width:1200px;height:)(\d+)(px"></iframe>)',
            rf"\g<1>{height}\g<3>{height}\g<5>",
            template,
            count=1,
        )

    html = html[: tpl_m.start(2)] + json.dumps(template, separators=(",", ":")) + html[tpl_m.end(2) :]

    # Keep page title in sync
    html = html.replace(
        "<title>AlphaEarth Weeks 1–6 Canvas</title>",
        "<title>AlphaEarth Weeks 1–7 Canvas</title>",
        1,
    )
    html = html.replace(
        "Loading AlphaEarth Weeks 1–6 Canvas...",
        "Loading AlphaEarth Weeks 1–7 Canvas...",
        1,
    )
    CANVAS.write_text(html)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True, help="section4_xgb_metrics.json")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--height", type=int, default=3180)
    ap.add_argument("--no-inject", action="store_true")
    args = ap.parse_args()

    metrics = json.loads(args.metrics.read_text())
    run_stamp = args.metrics.parent.parent.name
    board = build_board(metrics, run_stamp)
    out = args.out or (args.metrics.parent / "section4_xgb_board.html")
    out.write_text(board)
    print(f"board → {out} ({len(board):,} bytes)")

    if not args.no_inject:
        inject_canvas(board, args.height)
        print(f"canvas injected id={SECTION4_ID} height={args.height}")

    b = metrics["building"]["soft_pooled"]
    t = metrics["tile_100m"]["soft_pooled"]
    print(
        f"buildings: ridge={b['ridge']['mean']:.3f} xgb={b['xgb']['mean']:.3f} "
        f"(Δ={b['xgb']['mean']-b['ridge']['mean']:+.3f}) · "
        f"tiles: ridge={t['ridge']['mean']:.3f} xgb={t['xgb']['mean']:.3f} "
        f"(Δ={t['xgb']['mean']-t['ridge']['mean']:+.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
