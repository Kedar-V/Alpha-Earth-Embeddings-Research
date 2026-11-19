#!/usr/bin/env python3
"""Build Section 1 canvas board: classification F1 with Section 2/4 plot set.

Plots (F1 ↑, same layout as regression MAE boards):
  1. Where the gap comes from: RES F1 vs NRES F1 at 10 m and 100 m
  2. Per metro, 10 m — dumbbells + table
  3. Per metro, 100 m — dumbbells + table
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANVAS = ROOT / "notebooks" / "ghsl_built_s_vs_ae" / "AlphaEarth Weeks 1–5 Canvas.html"
SECTION1_ID = "6746c10d-4e42-427c-995a-d2a3a7debdb0"

# GHS-BUILT-S soft-argmax / AE LR / AE Ridge soft-argmax
COLORS = {"ghsl_built_s": "#C2691A", "ae_lr": "#1F6FA8", "ae_ridge": "#4AA0DB"}
NAMES = {"ghsl_built_s": "GHS-BUILT-S", "ae_lr": "AE LR", "ae_ridge": "AE Ridge"}
KEYS = ("ghsl_built_s", "ae_lr", "ae_ridge")


def f3(x: float) -> str:
    return f"{x:.3f}"


def bar_row(name: str, val: float, color: str, scale_max: float) -> str:
    w = max(2, int(round(320 * (val / scale_max)))) if scale_max > 0 else 2
    return (
        f'<div style="display:flex;align-items:center;gap:10px" title="{name}: F1 {f3(val)}">'
        f'<p style="margin:0;width:110px;font-size:14px">{name}</p>'
        f'<div style="width:{w}px;height:18px;background:{color};border-radius:0 4px 4px 0"></div>'
        f'<p style="margin:0;font-size:14px;font-weight:600;font-variant-numeric:tabular-nums">{f3(val)}</p>'
        f"</div>"
    )


def dumbbell_row(label: str, bg: str, vals: list[float], vmin: float, vmax: float) -> str:
    width = 380.0
    colors = [COLORS[k] for k in KEYS]
    span = max(vmax - vmin, 1e-6)

    def x_of(v: float) -> float:
        return max(0.0, min(width, ((v - vmin) / span) * width))

    xs = [x_of(v) for v in vals]
    lo, hi = min(xs), max(xs)
    n_ticks = 5
    ticks = "".join(
        f'<div style="position:absolute;left:{i * width / n_ticks:.1f}px;top:0;bottom:0;'
        f'width:1px;background:#E5E1D6"></div>'
        for i in range(1, n_ticks)
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


def axis_row(vmin: float, vmax: float) -> str:
    width = 380.0
    labels = [vmin + i * (vmax - vmin) / 5 for i in range(6)]
    ticks = "".join(
        f'<p style="position:absolute;left:{i * width / 5 - 14:.1f}px;width:28px;margin:0;'
        f'text-align:center;font-size:12px;color:#4A5562">{v:.2f}</p>'
        for i, v in enumerate(labels)
    )
    return (
        f'<div style="display:flex"><div style="width:96px"></div>'
        f'<div style="position:relative;width:{width}px;height:18px">{ticks}</div></div>'
        f'<p style="margin:2px 0 0 96px;font-size:12px;color:#4A5562">F1 (higher is better)</p>'
    )


def metro_table(rows: list[dict]) -> str:
    head = (
        '<table style="width:100%;border-collapse:collapse;margin-top:8px">'
        '<tr><th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:left;font-weight:600;font-size:15px">Metro</th>'
        '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right;font-weight:600;font-size:15px">n</th>'
        '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right;font-weight:600;font-size:15px;border-left:1px solid #3a4a45" colspan="3">BUILT-S</th>'
        '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right;font-weight:600;font-size:15px;border-left:1px solid #3a4a45" colspan="3">AE LR</th>'
        '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right;font-weight:600;font-size:15px;border-left:1px solid #3a4a45" colspan="3">AE Ridge</th>'
        '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right;font-weight:600;font-size:15px;border-left:1px solid #3a4a45">Margin</th></tr>'
        '<tr style="background:#1b2a26;color:#EDEBE4"><td></td><td></td>'
        + "".join(
            '<td style="padding:4px 10px;text-align:right;font-size:12px;border-left:1px solid #3a4a45">RES</td>'
            '<td style="padding:4px 10px;text-align:right;font-size:12px">NRES</td>'
            '<td style="padding:4px 10px;text-align:right;font-size:12px">macro</td>'
            for _ in range(3)
        )
        + '<td style="padding:4px 10px;text-align:right;font-size:12px;border-left:1px solid #3a4a45">LR−GHSL</td></tr>'
    )
    body = []
    for r in rows:
        models = [r["ghsl_built_s"], r["ae_lr"], r["ae_ridge"]]
        best_res = max(m["res_f1"] for m in models)
        best_nres = max(m["nres_f1"] for m in models)
        best_macro = max(m["macro_f1"] for m in models)
        margin = r["ae_lr"]["macro_f1"] - r["ghsl_built_s"]["macro_f1"]
        mcol = "#1F6FA8" if margin >= 0 else "#C2691A"

        def cell(v: float, best: float, border: bool = False) -> str:
            b = "border-left:1px solid #D9D5C8;" if border else ""
            bold = "<b>" if abs(v - best) < 1e-12 else ""
            bold_e = "</b>" if bold else ""
            return (
                f'<td style="padding:7px 10px;font-size:14px;text-align:right;'
                f'font-variant-numeric:tabular-nums;{b}">{bold}{f3(v)}{bold_e}</td>'
            )

        body.append(
            f'<tr style="background:{r["bg"]}">'
            f'<td style="padding:7px 10px;font-size:14px;font-weight:600">{r["label"]}</td>'
            f'<td style="padding:7px 10px;font-size:14px;text-align:right;font-variant-numeric:tabular-nums">{r["n"]:,}</td>'
            f'{cell(r["ghsl_built_s"]["res_f1"], best_res, True)}'
            f'{cell(r["ghsl_built_s"]["nres_f1"], best_nres)}'
            f'{cell(r["ghsl_built_s"]["macro_f1"], best_macro)}'
            f'{cell(r["ae_lr"]["res_f1"], best_res, True)}'
            f'{cell(r["ae_lr"]["nres_f1"], best_nres)}'
            f'{cell(r["ae_lr"]["macro_f1"], best_macro)}'
            f'{cell(r["ae_ridge"]["res_f1"], best_res, True)}'
            f'{cell(r["ae_ridge"]["nres_f1"], best_nres)}'
            f'{cell(r["ae_ridge"]["macro_f1"], best_macro)}'
            f'<td style="padding:7px 10px;font-size:14px;text-align:right;font-variant-numeric:tabular-nums;'
            f'border-left:1px solid #D9D5C8;color:{mcol};font-weight:600">{margin:+.3f}</td></tr>'
        )
    return head + "".join(body) + "</table>"


def legend_block() -> str:
    return (
        '<div style="display:flex;flex-direction:column;gap:10px;margin:6px 0 10px">'
        '<div style="display:flex;gap:20px;align-items:center;font-size:14px;color:#14231F;flex-wrap:wrap">'
        '<div style="display:flex;align-items:center;gap:6px"><div style="width:12px;height:12px;border-radius:50%;background:#C2691A"></div>GHS-BUILT-S</div>'
        '<div style="display:flex;align-items:center;gap:6px"><div style="width:12px;height:12px;border-radius:50%;background:#1F6FA8"></div>AE LR</div>'
        '<div style="display:flex;align-items:center;gap:6px"><div style="width:12px;height:12px;border-radius:50%;background:#4AA0DB"></div>AE Ridge (soft-argmax)</div>'
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


def metro_dumbbells(rows: list[dict], which: str, vmin: float, vmax: float) -> str:
    key = "res_f1" if which == "res" else "nres_f1"
    title = "Residential F1" if which == "res" else "Non-residential F1"
    body = "".join(
        dumbbell_row(
            r["label"],
            r["bg"],
            [r[k][key] for k in KEYS],
            vmin,
            vmax,
        )
        for r in rows
    )
    return (
        f'<div style="flex:1;display:flex;flex-direction:column">'
        f'<p style="margin:0 0 6px;font-size:15px;font-weight:600">{title}</p>{body}'
        f"{axis_row(vmin, vmax)}</div>"
    )


def metro_card(title_html: str, blurb: str, note: str, rows: list[dict]) -> str:
    vals = []
    for r in rows:
        for k in KEYS:
            vals.extend([r[k]["res_f1"], r[k]["nres_f1"]])
    vmin = max(0.45, min(vals) - 0.03)
    vmax = min(1.0, max(vals) + 0.02)
    return (
        '<div style="display:flex;flex-direction:column;gap:10px;background:#FCFBF7;'
        'border:1px solid #D9D5C8;border-radius:12px;padding:24px">'
        f'<p style="margin:0;font-size:17px;font-weight:600">{title_html}</p>'
        f'<p style="margin:0 0 6px;font-size:15px;line-height:1.4;color:#4A5562">{blurb}</p>'
        f'<p style="margin:0;font-size:15px;line-height:1.45">{note}</p>'
        f"{legend_block()}"
        f'<div style="display:flex;gap:28px">{metro_dumbbells(rows, "res", vmin, vmax)}'
        f'{metro_dumbbells(rows, "nres", vmin, vmax)}</div>'
        f"{metro_table(rows)}"
        "</div>"
    )


def build_board(metrics: dict, run_stamp: str) -> str:
    c10 = metrics["cell_10m"]
    t100 = metrics["tile_100m"]
    p10 = c10["pooled"]
    p100 = t100["pooled"]
    n10 = c10["n"]
    n100 = t100["n"]
    beats10 = c10.get("ae_lr_beats_ghsl", sum(1 for r in c10["metro"] if r["ae_lr"]["macro_f1"] > r["ghsl_built_s"]["macro_f1"]))
    beats100 = t100.get("ae_lr_beats_ghsl", sum(1 for r in t100["metro"] if r["ae_lr"]["macro_f1"] > r["ghsl_built_s"]["macro_f1"]))

    intro = (
        "Binary residential vs non-residential on the Week-4 balanced pure set "
        f"(1:1, n = {n10:,} cells), scored region-LORO. "
        "<b>AE LR</b> is the 64-D logistic head; <b>AE Ridge</b> is soft-argmax of the fraction head; "
        "<b>GHS-BUILT-S</b> is soft-argmax of soft RES/NRES fractions at AE cell centres (no training). "
        f"100 m uses the shared tile set with ≥5 cells (n = {n100:,}). "
        "Margin in the tables is <b>AE LR macro − GHSL macro</b> (blue = AE ahead, orange = GHSL ahead)."
    )

    scale = 1.0

    def panel_pair(title: str, pooled: dict) -> str:
        return (
            f'<p style="margin:4px 0 0;font-size:13px;font-weight:600;letter-spacing:1.5px;'
            f'text-transform:uppercase;color:#4A5562">{title}</p>'
            f'<div style="display:flex;gap:40px">'
            f'<div style="flex:1;display:flex;flex-direction:column;gap:6px">'
            f'<p style="margin:0 0 2px;font-size:15px;font-weight:600">Residential F1</p>'
            + "".join(bar_row(NAMES[k], pooled[k]["res_f1"], COLORS[k], scale) for k in KEYS)
            + "</div>"
            f'<div style="flex:1;display:flex;flex-direction:column;gap:6px">'
            f'<p style="margin:0 0 2px;font-size:15px;font-weight:600">Non-residential F1</p>'
            + "".join(bar_row(NAMES[k], pooled[k]["nres_f1"], COLORS[k], scale) for k in KEYS)
            + "</div></div>"
        )

    gap = (
        '<div style="display:flex;flex-direction:column;gap:10px;background:#FCFBF7;'
        'border:1px solid #D9D5C8;border-radius:12px;padding:24px">'
        '<p style="margin:0;font-size:17px;font-weight:600">Where the gap comes from: residential vs non-residential</p>'
        '<p style="margin:0 0 6px;font-size:15px;line-height:1.4;color:#4A5562">'
        "Per-class F1 on each scale (higher is better). All four panels share a 0–1 scale.</p>"
        + panel_pair(f"10 m · {n10:,} cells", p10)
        + panel_pair(f"100 m · {n100:,} shared tiles (≥5 cells)", p100)
        + "</div>"
    )

    note10 = (
        f"AE LR has higher macro-F1 than GHS-BUILT-S in <b>{beats10} of {len(c10['metro'])}</b> metros. "
        f"Pooled OOF: AE LR macro <b>{f3(p10['ae_lr']['macro_f1'])}</b> vs BUILT-S <b>{f3(p10['ghsl_built_s']['macro_f1'])}</b> "
        f"(Δ = {p10['ae_lr']['macro_f1'] - p10['ghsl_built_s']['macro_f1']:+.3f}). "
        "Most of the AE lead is on non-residential F1."
    )
    note100 = (
        f"On shared 100 m tiles AE LR beats GHS-BUILT-S on macro-F1 in <b>{beats100} of {len(t100['metro'])}</b> metros. "
        f"Pooled: AE LR <b>{f3(p100['ae_lr']['macro_f1'])}</b> vs BUILT-S <b>{f3(p100['ghsl_built_s']['macro_f1'])}</b>."
    )
    color10 = "#1F6FA8" if beats10 >= len(c10["metro"]) / 2 else "#C2691A"
    color100 = "#1F6FA8" if beats100 >= len(t100["metro"]) / 2 else "#C2691A"
    title10 = (
        f'Per metro, 10 m · <span style="color:{color10}">AE LR beats BUILT-S in {beats10} of {len(c10["metro"])}</span>'
    )
    title100 = (
        f'Per metro, 100 m · <span style="color:{color100}">AE LR beats BUILT-S in {beats100} of {len(t100["metro"])}</span>'
        if beats100 >= len(t100["metro"]) / 2
        else f'Per metro, 100 m · <span style="color:{color100}">BUILT-S beats AE LR in {len(t100["metro"]) - beats100} of {len(t100["metro"])}</span>'
    )

    c_card = metro_card(
        title10,
        "RES F1, NRES F1, and macro-F1. Bold = best of the three models in that column. "
        "Margin = AE LR macro − BUILT-S macro. Grey line spans the three models.",
        note10,
        c10["metro"],
    )
    t_card = metro_card(
        title100,
        "Same three models on the shared ≥5-cell tile set. Columns match the 10 m table.",
        note100,
        t100["metro"],
    )

    leg = metrics.get("legacy") or {}
    d100 = p100["ae_lr"]["macro_f1"] - p100["ghsl_built_s"]["macro_f1"]
    pool100 = (
        f"Pooling to 100 m flips the ranking: GHS-BUILT-S macro <b>{f3(p100['ghsl_built_s']['macro_f1'])}</b> "
        f"vs AE LR <b>{f3(p100['ae_lr']['macro_f1'])}</b> (Δ = {d100:+.3f} for AE) on the shared tile set."
        if d100 < 0
        else f"Pooling to 100 m keeps AE LR ahead on macro "
        f"({f3(p100['ae_lr']['macro_f1'])} vs {f3(p100['ghsl_built_s']['macro_f1'])} on the shared tile set)."
    )
    means = (
        '<div style="display:flex;flex-direction:column;gap:10px;background:#FCFBF7;'
        'border:1px solid #D9D5C8;border-radius:12px;padding:24px">'
        '<h2 style="margin:0;font-family:\'Source Serif 4\',Georgia,serif;font-size:26px;font-weight:600">What this means</h2>'
        f'<p style="margin:0;font-size:16px;line-height:1.45">At 10 m, AE LR leads soft GHS-BUILT-S. '
        f'Pooled OOF macro-F1 is <b>{f3(p10["ae_lr"]["macro_f1"])}</b> for AE LR vs '
        f'<b>{f3(p10["ghsl_built_s"]["macro_f1"])}</b> for BUILT-S; '
        f'leave-one-metro-out mean is {leg.get("loro_macro_f1_mean", 0.779):.3f}. '
        f'The gap is mostly non-residential (AE LR NRES F1 {f3(p10["ae_lr"]["nres_f1"])} vs '
        f'BUILT-S {f3(p10["ghsl_built_s"]["nres_f1"])}).</p>'
        f'<p style="margin:0;font-size:16px;line-height:1.45">{pool100} '
        f'3D ViT at 100 m still trails a linear head '
        f'({leg.get("vit_v4_macro_f1", 0.738):.3f} vs {leg.get("linear_100m_macro_f1", 0.761):.3f}).</p>'
        "</div>"
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Classification results · RES vs NRES F1</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&amp;family=Source+Serif+4:wght@400;600&amp;display=swap">
<style>body{{margin:0}} table{{border-collapse:collapse;width:100%}} td{{border-bottom:1px solid #D9D5C8}}</style></head><body>
<div style="width:1200px;min-height:0;box-sizing:border-box;padding:56px;background:#F6F4EE;color:#14231F;font-family:'IBM Plex Sans',Arial,sans-serif;display:flex;flex-direction:column;gap:28px">
<div style="display:flex;flex-direction:column;gap:8px">
<p style="margin:0;font-size:15px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#1F6FA8">Section 1 · Classification</p>
<h1 style="margin:0;font-family:'Source Serif 4',Georgia,serif;font-size:48px;font-weight:600;line-height:1.1">Classification results</h1>
<p style="margin:0;font-size:18px;line-height:1.45;color:#4A5562">{intro}</p>
</div>
{gap}
{c_card}
{t_card}
{means}
<p style="margin:12px 0 0;font-size:15px;color:#4A5562">AlphaEarth embeddings · Week-4 balanced LORO classification · {run_stamp}</p>
</div></body></html>
"""


def inject_canvas(board: str, height: int) -> None:
    html = CANVAS.read_text()
    compressed = base64.b64encode(gzip.compress(board.encode("utf-8"), compresslevel=9)).decode("ascii")

    man_m = re.search(r'(<script type="__bundler/manifest">)(.*?)(</script>)', html, re.S)
    man = json.loads(man_m.group(2))
    man[SECTION1_ID] = {"mime": "text/html", "compressed": True, "data": compressed}
    html = html[: man_m.start(2)] + json.dumps(man, separators=(",", ":")) + html[man_m.end(2) :]

    tpl_m = re.search(r'(<script type="__bundler/template">)(.*?)(</script>)', html, re.S)
    template = json.loads(tpl_m.group(2))
    template = re.sub(
        rf'(<section class="board"><h2>1 Classification results</h2><iframe src="about:blank#{SECTION1_ID}" '
        rf'title="1 Classification results" width="1200" height=")(\d+)(" loading="eager" '
        rf'style="width:1200px;height:)(\d+)(px"></iframe></section>)',
        rf"\g<1>{height}\g<3>{height}\g<5>",
        template,
        count=1,
    )
    html = html[: tpl_m.start(2)] + json.dumps(template, separators=(",", ":")) + html[tpl_m.end(2) :]
    CANVAS.write_text(html)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics",
        type=Path,
        default=ROOT
        / "outputs"
        / "week4_pixel_balanced_train"
        / "section1_classification_metrics.json",
    )
    ap.add_argument("--height", type=int, default=3600)
    ap.add_argument("--no-inject", action="store_true")
    args = ap.parse_args()

    metrics_path = args.metrics.resolve()
    metrics = json.loads(metrics_path.read_text())
    board = build_board(metrics, metrics_path.parent.name)
    out = metrics_path.parent / "section1_classification_board.html"
    out.write_text(board)
    print(f"board → {out} ({len(board):,} bytes)")

    if not args.no_inject:
        inject_canvas(board, args.height)
        print(f"canvas injected id={SECTION1_ID} height={args.height}")

    p = metrics["cell_10m"]["pooled"]
    t = metrics["tile_100m"]["pooled"]
    print(
        f"10m macro: LR={p['ae_lr']['macro_f1']:.3f} Ridge={p['ae_ridge']['macro_f1']:.3f} "
        f"GHSL={p['ghsl_built_s']['macro_f1']:.3f} | "
        f"100m macro: LR={t['ae_lr']['macro_f1']:.3f} Ridge={t['ae_ridge']['macro_f1']:.3f} "
        f"GHSL={t['ghsl_built_s']['macro_f1']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
