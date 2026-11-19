#!/usr/bin/env python3
"""Build Section 5 canvas board: AE rings (train-where-you-score) vs Section 3/4 baselines."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANVAS = ROOT / "notebooks" / "ghsl_built_s_vs_ae" / "AlphaEarth Weeks 1–5 Canvas.html"
SECTION4_ID = "a7c4e2b1-5d8f-4a91-9c3e-2f6b8d0e1a45"
SECTION5_ID = "b8d5f3c2-6e9a-4b02-8d4f-3a7c9e1b2c56"


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
    # vals: [bs, ridge, pls, xgb, ring]
    width = 380.0
    colors = ["#C2691A", "#1F6FA8", "#4AA0DB", "#8E44AD", "#2E7D32"]
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
        '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right;font-weight:600;font-size:15px;border-left:1px solid #3a4a45" colspan="3">Ring</th>'
        '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right;font-weight:600;font-size:15px;border-left:1px solid #3a4a45">Margin</th></tr>'
        '<tr style="background:#1b2a26;color:#EDEBE4"><td></td><td></td>'
        + "".join(
            '<td style="padding:4px 10px;text-align:right;font-size:12px;border-left:1px solid #3a4a45">res</td>'
            '<td style="padding:4px 10px;text-align:right;font-size:12px">nres</td>'
            '<td style="padding:4px 10px;text-align:right;font-size:12px">mean</td>'
            for _ in range(5)
        )
        + '<td style="padding:4px 10px;text-align:right;font-size:12px;border-left:1px solid #3a4a45">Rk−Rg</td></tr>'
    )
    body = []
    for r in rows:
        bs, rg, pl, xg, rk = r["bs"], r["rg"], r["pl"], r["xg"], r["rk"]
        best_mean = min(bs["mean"], rg["mean"], pl["mean"], xg["mean"], rk["mean"])
        best_res = min(bs["res"], rg["res"], pl["res"], xg["res"], rk["res"])
        best_nres = min(bs["nres"], rg["nres"], pl["nres"], xg["nres"], rk["nres"])

        def cell(v: float, best: float, border: bool = False) -> str:
            b = "border-left:1px solid #D9D5C8;" if border else ""
            bold = "<b>" if abs(v - best) < 1e-12 else ""
            bold_e = "</b>" if bold else ""
            return (
                f'<td style="padding:7px 10px;font-size:14px;text-align:right;'
                f'font-variant-numeric:tabular-nums;{b}">{bold}{f3(v)}{bold_e}</td>'
            )

        margin = r["margin"]
        mcol = "#1F6FA8" if margin >= 0 else "#2E7D32"
        body.append(
            f'<tr style="background:{r["bg"]}">'
            f'<td style="padding:7px 10px;font-size:14px;font-weight:600">{r["label"]}</td>'
            f'<td style="padding:7px 10px;font-size:14px;text-align:right;font-variant-numeric:tabular-nums">{r["n"]:,}</td>'
            f'{cell(bs["res"], best_res, True)}{cell(bs["nres"], best_nres)}{cell(bs["mean"], best_mean)}'
            f'{cell(rg["res"], best_res, True)}{cell(rg["nres"], best_nres)}{cell(rg["mean"], best_mean)}'
            f'{cell(pl["res"], best_res, True)}{cell(pl["nres"], best_nres)}{cell(pl["mean"], best_mean)}'
            f'{cell(xg["res"], best_res, True)}{cell(xg["nres"], best_nres)}{cell(xg["mean"], best_mean)}'
            f'{cell(rk["res"], best_res, True)}{cell(rk["nres"], best_nres)}{cell(rk["mean"], best_mean)}'
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
        '<div style="display:flex;align-items:center;gap:6px"><div style="width:12px;height:12px;border-radius:50%;background:#2E7D32"></div>AE Ring</div>'
        "</div></div>"
    )


def metro_dumbbells(rows: list[dict], which: str) -> str:
    key = "res" if which == "res" else "nres"
    vals_all = []
    for r in rows:
        vals_all.extend([r["bs"][key], r["rg"][key], r["pl"][key], r["xg"][key], r["rk"][key]])
    vmax = max(vals_all) * 1.05 if vals_all else 1.0
    title = "Residential MAE" if which == "res" else "Non-residential MAE"
    body = "".join(
        dumbbell_row(
            r["label"],
            r["bg"],
            [r["bs"][key], r["rg"][key], r["pl"][key], r["xg"][key], r["rk"][key]],
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


def build_board(metrics: dict, run_stamp: str) -> str:
    building = metrics["building"]
    tile = metrics["tile_100m"]
    b = building["soft_pooled"]
    t = tile["soft_pooled"]
    n_b = building["n_shared"]
    n_t = tile["n_tiles"]
    b_wins = building["ring_beats_ridge"]
    t_wins = tile["ring_beats_ridge"]

    intro = (
        "Same locked Section 3/4 Overture cohorts and soft GT. "
        "<b>AE Ring</b> = mean AE embedding plus mean/std in 50 / 100 / 200 m rings, "
        "Ridge LORO trained on soft fractions (train-where-you-score; training may use extra "
        "soft-covered buildings). Eval IDs match Section 3/4 exactly. "
        "Margin = <b>Ring mean − Ridge mean</b> (green = Ring better)."
    )

    def g(d: dict, key: str, field: str) -> float:
        return float(d[key][field])

    scale = max(
        g(b, "ghsl_built_s", "res"),
        g(b, "ae_ridge", "res"),
        g(b, "ae_pls", "res"),
        g(b, "ae_xgb", "res"),
        g(b, "ae_ring", "res"),
        g(b, "ghsl_built_s", "nres"),
        g(b, "ae_ridge", "nres"),
        g(b, "ae_pls", "nres"),
        g(b, "ae_xgb", "nres"),
        g(b, "ae_ring", "nres"),
        g(t, "ghsl_built_s", "res"),
        g(t, "ae_ridge", "res"),
        g(t, "ae_pls", "res"),
        g(t, "ae_xgb", "res"),
        g(t, "ae_ring", "res"),
        g(t, "ghsl_built_s", "nres"),
        g(t, "ae_ridge", "nres"),
        g(t, "ae_pls", "nres"),
        g(t, "ae_xgb", "nres"),
        g(t, "ae_ring", "nres"),
        0.01,
    )

    def panel_pair(title: str, res: dict, nres: dict) -> str:
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
            f'{bar_row("AE Ring", res["ring"], "#2E7D32", scale)}'
            f"</div>"
            f'<div style="flex:1;display:flex;flex-direction:column;gap:6px">'
            f'<p style="margin:0 0 2px;font-size:15px;font-weight:600">Non-residential MAE</p>'
            f'{bar_row("GHS-BUILT-S", nres["built_s"], "#C2691A", scale)}'
            f'{bar_row("AE Ridge", nres["ridge"], "#1F6FA8", scale)}'
            f'{bar_row("AE PLS", nres["pls"], "#4AA0DB", scale)}'
            f'{bar_row("AE XGB", nres["xgb"], "#8E44AD", scale)}'
            f'{bar_row("AE Ring", nres["ring"], "#2E7D32", scale)}'
            f"</div></div>"
        )

    gap = (
        '<div style="display:flex;flex-direction:column;gap:10px;background:#FCFBF7;'
        'border:1px solid #D9D5C8;border-radius:12px;padding:24px">'
        '<p style="margin:0;font-size:17px;font-weight:600">Where the gap comes from: residential vs non-residential</p>'
        '<p style="margin:0 0 6px;font-size:15px;line-height:1.4;color:#4A5562">'
        "MAE on each fraction, pooled over the locked Section 3/4 sets (lower is better). "
        "All panels share one scale.</p>"
        + panel_pair(
            f"Building · {n_b:,} buildings",
            {
                "built_s": g(b, "ghsl_built_s", "res"),
                "ridge": g(b, "ae_ridge", "res"),
                "pls": g(b, "ae_pls", "res"),
                "xgb": g(b, "ae_xgb", "res"),
                "ring": g(b, "ae_ring", "res"),
            },
            {
                "built_s": g(b, "ghsl_built_s", "nres"),
                "ridge": g(b, "ae_ridge", "nres"),
                "pls": g(b, "ae_pls", "nres"),
                "xgb": g(b, "ae_xgb", "nres"),
                "ring": g(b, "ae_ring", "nres"),
            },
        )
        + panel_pair(
            f"100 m · {n_t:,} tiles",
            {
                "built_s": g(t, "ghsl_built_s", "res"),
                "ridge": g(t, "ae_ridge", "res"),
                "pls": g(t, "ae_pls", "res"),
                "xgb": g(t, "ae_xgb", "res"),
                "ring": g(t, "ae_ring", "res"),
            },
            {
                "built_s": g(t, "ghsl_built_s", "nres"),
                "ridge": g(t, "ae_ridge", "nres"),
                "pls": g(t, "ae_pls", "nres"),
                "xgb": g(t, "ae_xgb", "nres"),
                "ring": g(t, "ae_ring", "nres"),
            },
        )
        + "</div>"
    )

    b_card = metro_card(
        f'Per metro, building · <span style="color:#2E7D32">Ring beats Ridge in {b_wins} of {len(building["metro_soft"])}</span>',
        "Same Section 3 building IDs (y_res 1:1). Bold = best of five. Margin = Ring − Ridge.",
        f"Ring beats Ridge on mean MAE in <b>{b_wins} of {len(building['metro_soft'])}</b> metros.",
        building["metro_soft"],
    )
    t_card = metro_card(
        f'Per metro, 100 m · <span style="color:#2E7D32">Ring beats Ridge in {t_wins} of {len(tile["metro_soft"])}</span>',
        "Same Section 3 100 m tile IDs.",
        f"Ring beats Ridge on mean MAE in <b>{t_wins} of {len(tile['metro_soft'])}</b> metros.",
        tile["metro_soft"],
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>AE Rings · Fraction MAE in Overture polygon space</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&amp;family=Source+Serif+4:wght@400;600&amp;display=swap">
<style>body{{margin:0}} table{{border-collapse:collapse;width:100%}} td{{border-bottom:1px solid #D9D5C8}}</style></head><body>
<div style="width:1200px;min-height:0;box-sizing:border-box;padding:56px;background:#F6F4EE;color:#14231F;font-family:'IBM Plex Sans',Arial,sans-serif;display:flex;flex-direction:column;gap:28px">
<div style="display:flex;flex-direction:column;gap:8px">
<p style="margin:0;font-size:15px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#2E7D32">Section 5 · Week 8</p>
<h1 style="margin:0;font-family:'Source Serif 4',Georgia,serif;font-size:48px;font-weight:600;line-height:1.1">Multi-scale AE rings vs GHS-BUILT-S</h1>
<p style="margin:0;font-size:18px;line-height:1.45;color:#4A5562">{intro}</p>
</div>
{gap}
{b_card}
{t_card}
<p style="margin:12px 0 0;font-size:15px;color:#4A5562">AlphaEarth · 50/100/200 m rings · train-where-you-score LORO · {run_stamp}</p>
</div></body></html>
"""


def inject_canvas(board: str, height: int) -> None:
    html = CANVAS.read_text()
    compressed = base64.b64encode(gzip.compress(board.encode("utf-8"), compresslevel=9)).decode("ascii")

    man_m = re.search(r'(<script type="__bundler/manifest">)(.*?)(</script>)', html, re.S)
    man = json.loads(man_m.group(2))
    man[SECTION5_ID] = {"mime": "text/html", "compressed": True, "data": compressed}
    html = html[: man_m.start(2)] + json.dumps(man, separators=(",", ":")) + html[man_m.end(2) :]

    po_m = re.search(r'(<script type="__bundler/page_order">)(.*?)(</script>)', html, re.S)
    order = json.loads(po_m.group(2))
    if SECTION5_ID in order:
        order = [x for x in order if x != SECTION5_ID]
    if SECTION4_ID not in order:
        raise SystemExit("Section 4 id missing from page_order — inject Section 4 first")
    i = order.index(SECTION4_ID) + 1
    order.insert(i, SECTION5_ID)
    html = html[: po_m.start(2)] + json.dumps(order, separators=(",", ":")) + html[po_m.end(2) :]

    tpl_m = re.search(r'(<script type="__bundler/template">)(.*?)(</script>)', html, re.S)
    template = json.loads(tpl_m.group(2))
    # drop prior section 5 board if re-running
    template = re.sub(
        rf'<section class="board"><h2>5 .*?</h2><iframe src="about:blank#{SECTION5_ID}".*?</iframe></section>\n?',
        "",
        template,
        count=1,
        flags=re.S,
    )
    # renumber old "5 Generalisation..." if still present after section4 inject
    if "5 Generalisation performance" in template:
        mapping = [
            ("9 GHSL BUILT-S vs AlphaEarth (AE cell centres)", "10 GHSL BUILT-S vs AlphaEarth (AE cell centres)"),
            ("8 Training dynamics and loss function", "9 Training dynamics and loss function"),
            ("7 Country-wise LORO and cross-country generalisation", "8 Country-wise LORO and cross-country generalisation"),
            ("6 Embedding dynamics", "7 Embedding dynamics"),
            ("5 Generalisation performance", "6 Generalisation performance"),
        ]
        for old, new in mapping:
            if old in template:
                template = template.replace(old, new, 1)

    iframe = (
        f'<section class="board"><h2>5 Multi-scale AE rings vs GHS-BUILT-S</h2>'
        f'<iframe src="about:blank#{SECTION5_ID}" title="5 Multi-scale AE rings vs GHS-BUILT-S" '
        f'width="1200" height="{height}" loading="eager" '
        f'style="width:1200px;height:{height}px"></iframe></section>\n'
    )
    # insert after section 4 iframe
    needle4 = f'about:blank#{SECTION4_ID}'
    if needle4 not in template:
        raise SystemExit("Section 4 iframe missing in template")
    # find end of section 4 </section>
    m = re.search(
        rf'<section class="board"><h2>4 .*?</h2><iframe src="about:blank#{SECTION4_ID}".*?</iframe></section>\n?',
        template,
        flags=re.S,
    )
    if not m:
        raise SystemExit("could not locate Section 4 block")
    template = template[: m.end()] + iframe + template[m.end() :]

    html = html[: tpl_m.start(2)] + json.dumps(template, separators=(",", ":")) + html[tpl_m.end(2) :]
    html = html.replace(
        "<title>AlphaEarth Weeks 1–7 Canvas</title>",
        "<title>AlphaEarth Weeks 1–8 Canvas</title>",
        1,
    )
    html = html.replace(
        "Loading AlphaEarth Weeks 1–7 Canvas...",
        "Loading AlphaEarth Weeks 1–8 Canvas...",
        1,
    )
    # also handle 1-6 title if section4 title update didn't stick
    html = html.replace(
        "<title>AlphaEarth Weeks 1–6 Canvas</title>",
        "<title>AlphaEarth Weeks 1–8 Canvas</title>",
        1,
    )
    CANVAS.write_text(html)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--height", type=int, default=3400)
    ap.add_argument("--no-inject", action="store_true")
    args = ap.parse_args()

    metrics = json.loads(args.metrics.read_text())
    run_stamp = args.metrics.parent.parent.name
    board = build_board(metrics, run_stamp)
    out = args.out or (args.metrics.parent / "section5_ring_board.html")
    out.write_text(board)
    print(f"board → {out}")
    if not args.no_inject:
        inject_canvas(board, args.height)
        print(f"canvas injected id={SECTION5_ID}")
    b = metrics["building"]["soft_pooled"]
    print(
        f"buildings: ridge={b['ae_ridge']['mean']:.3f} ring={b['ae_ring']['mean']:.3f} "
        f"ghsl={b['ghsl_built_s']['mean']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
