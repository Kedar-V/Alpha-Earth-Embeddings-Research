#!/usr/bin/env python3
"""Build Section 11 canvas board: AE-grid ViT + S2 late fusion vs CNN+S2 / AE+S2 / GHSL."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANVAS = ROOT / "notebooks" / "ghsl_built_s_vs_ae" / "AlphaEarth Weeks 1–5 Canvas.html"
SECTION10_ID = "a3c0e8b7-1d4f-4057-d29e-8f2b4d6a7b01"
SECTION11_ID = "b4d1f9c8-2e5a-4168-e30f-9a3c5d7e8c12"

COLORS = ["#C2691A", "#1F6FA8", "#0F766E", "#2E7D32", "#8E44AD", "#0B525B"]
KEYS = ["ghsl_built_s", "ae_ridge", "ae_s2", "ae_cnn_s2", "ae_vit_s2", "ae_ring"]
NAMES = ["GHS-BUILT-S", "AE Ridge", "AE+S2", "CNN+S2", "ViT+S2", "AE Ring"]


def f3(x: float) -> str:
    return f"{x:.3f}"


def bar_row(name: str, val: float, color: str, scale_max: float) -> str:
    w = max(2, int(round(320 * (val / scale_max)))) if scale_max > 0 and val == val else 2
    return (
        f'<div style="display:flex;align-items:center;gap:10px" title="{name}: MAE {f3(val)}">'
        f'<p style="margin:0;width:120px;font-size:14px">{name}</p>'
        f'<div style="width:{w}px;height:18px;background:{color};border-radius:0 4px 4px 0"></div>'
        f'<p style="margin:0;font-size:14px;font-weight:600;font-variant-numeric:tabular-nums">{f3(val)}</p>'
        f"</div>"
    )


def _g(d: dict, key: str, field: str) -> float:
    if key not in d:
        return float("nan")
    return float(d[key][field])


def metro_table(rows: list[dict]) -> str:
    labs = ("BUILT-S", "AE+S2", "CNN+S2", "ViT+S2")
    keys = ("ghsl_built_s", "ae_s2", "ae_cnn_s2", "ae_vit_s2")
    head = (
        '<table style="width:100%;border-collapse:collapse;margin-top:8px">'
        '<tr><th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:left">Metro</th>'
        '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right">n</th>'
        + "".join(
            f'<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right;'
            f'border-left:1px solid #3a4a45" colspan="3">{lab}</th>'
            for lab in labs
        )
        + '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right;border-left:1px solid #3a4a45">Δ</th></tr>'
        '<tr style="background:#1b2a26;color:#EDEBE4"><td></td><td></td>'
        + "".join(
            '<td style="padding:4px 10px;text-align:right;font-size:12px;border-left:1px solid #3a4a45">res</td>'
            '<td style="padding:4px 10px;text-align:right;font-size:12px">nres</td>'
            '<td style="padding:4px 10px;text-align:right;font-size:12px">mean</td>'
            for _ in labs
        )
        + '<td style="padding:4px 10px;text-align:right;font-size:12px;border-left:1px solid #3a4a45">ViT−CNN</td></tr>'
    )
    body = []
    for r in rows:
        objs = [r.get(k, {}) for k in keys]
        means = [o.get("mean", float("nan")) for o in objs]
        best = min([x for x in means if x == x], default=0.0)

        def cell(v, best_v, border=False):
            b = "border-left:1px solid #D9D5C8;" if border else ""
            if v != v:
                return f'<td style="padding:7px 10px;text-align:right;{b}">—</td>'
            bold = "<b>" if abs(v - best_v) < 1e-12 else ""
            be = "</b>" if bold else ""
            return (
                f'<td style="padding:7px 10px;text-align:right;font-variant-numeric:tabular-nums;{b}">'
                f"{bold}{f3(v)}{be}</td>"
            )

        cells = ""
        for o in objs:
            cells += (
                f'{cell(o.get("res", float("nan")), min([x.get("res", float("nan")) for x in objs if x.get("res", float("nan")) == x.get("res", float("nan"))], default=0), True)}'
                f'{cell(o.get("nres", float("nan")), min([x.get("nres", float("nan")) for x in objs if x.get("nres", float("nan")) == x.get("nres", float("nan"))], default=0))}'
                f'{cell(o.get("mean", float("nan")), best)}'
            )
        d = (r.get("ae_vit_s2", {}).get("mean", float("nan"))) - (r.get("ae_cnn_s2", {}).get("mean", float("nan")))
        mcol = "#C2691A" if (d == d and d >= 0) else "#0B525B"
        body.append(
            f'<tr style="background:{r.get("bg", "#F6F4EE")}">'
            f'<td style="padding:7px 10px;font-weight:600">{r.get("label", r["region"])}</td>'
            f'<td style="padding:7px 10px;text-align:right">{r["n"]:,}</td>'
            f"{cells}"
            f'<td style="padding:7px 10px;text-align:right;border-left:1px solid #D9D5C8;color:{mcol};font-weight:600">{d:+.3f}</td></tr>'
        )
    return head + "".join(body) + "</table>"


def build_board(metrics: dict, run_stamp: str) -> str:
    b = metrics["buildings"]
    pool = b["soft_pooled"]
    n_b = int(metrics.get("eval_n_buildings") or b["n_shared"])
    pad = metrics.get("pad", 16)
    patch = metrics.get("patch", 4)
    depth = metrics.get("vit_depth", 4)
    epochs = metrics.get("epochs", "?")
    delta_cnn = _g(pool, "ae_vit_s2", "mean") - _g(pool, "ae_cnn_s2", "mean")
    delta_s2 = _g(pool, "ae_vit_s2", "mean") - _g(pool, "ae_s2", "mean")
    beats_cnn = b.get("vit_beats_cnn", 0)
    beats_s2 = b.get("vit_beats_s2", 0)

    intro = (
        "Same locked Section 3/4 building IDs as Week 13. "
        f"<b>ViT+S2</b> = AE cells on a {pad}×{pad} footprint grid → tiny ViT "
        f"(patch={patch}, depth={depth}, d=128, CLS) late-fused with AE mean 64-d + "
        f"Sentinel-2 NRES features → MLP; LORO MSE. Epochs={epochs}. "
        f"Headline <b>0.5·(MAE_res+MAE_nres)</b>. "
        f"Δ vs CNN {delta_cnn:+.3f} · vs AE+S2 {delta_s2:+.3f}."
    )

    scale_vals = []
    for k in KEYS:
        if k in pool:
            scale_vals.extend([_g(pool, k, "res"), _g(pool, k, "nres")])
    scale = max([v for v in scale_vals if v == v] + [0.01])

    def panel(title, field):
        rows = [(NAMES[i], _g(pool, k, field), COLORS[i]) for i, k in enumerate(KEYS)]
        return (
            f'<div style="flex:1;display:flex;flex-direction:column;gap:6px">'
            f'<p style="margin:0 0 2px;font-size:15px;font-weight:600">{title}</p>'
            + "".join(bar_row(n, v, c, scale) for n, v, c in rows if v == v)
            + "</div>"
        )

    gap = (
        '<div style="display:flex;flex-direction:column;gap:10px;background:#FCFBF7;'
        'border:1px solid #D9D5C8;border-radius:12px;padding:24px">'
        f'<p style="margin:0;font-size:17px;font-weight:600">Building soft MAE · {n_b:,}</p>'
        '<p style="margin:0 0 6px;font-size:15px;color:#4A5562">Lower is better. Shared scale.</p>'
        f'<div style="display:flex;gap:40px">{panel("Residential MAE", "res")}{panel("Non-residential MAE", "nres")}</div>'
        "</div>"
    )

    card = (
        '<div style="display:flex;flex-direction:column;gap:10px;background:#FCFBF7;'
        'border:1px solid #D9D5C8;border-radius:12px;padding:24px">'
        f'<p style="margin:0;font-size:17px;font-weight:600">Per metro · '
        f'<span style="color:#0B525B">ViT beats CNN in {beats_cnn}/10 · AE+S2 in {beats_s2}/10</span></p>'
        f'<p style="margin:0;font-size:15px">Pooled: ViT+S2 <b>{f3(_g(pool, "ae_vit_s2", "mean"))}</b> · '
        f'CNN+S2 {f3(_g(pool, "ae_cnn_s2", "mean"))} · AE+S2 {f3(_g(pool, "ae_s2", "mean"))} · '
        f'BUILT-S {f3(_g(pool, "ghsl_built_s", "mean"))}</p>'
        f'{metro_table(b.get("metro_soft") or [])}'
        "</div>"
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>AE-grid ViT + S2 · Fraction MAE</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&amp;family=Source+Serif+4:wght@400;600&amp;display=swap">
<style>body{{margin:0}} table{{border-collapse:collapse;width:100%}} td{{border-bottom:1px solid #D9D5C8}} code{{font-size:13px}}</style></head><body>
<div style="width:1200px;box-sizing:border-box;padding:56px;background:#F6F4EE;color:#14231F;font-family:'IBM Plex Sans',Arial,sans-serif;display:flex;flex-direction:column;gap:28px">
<div style="display:flex;flex-direction:column;gap:8px">
<p style="margin:0;font-size:15px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#0B525B">Section 11 · Week 14</p>
<h1 style="margin:0;font-family:'Source Serif 4',Georgia,serif;font-size:48px;font-weight:600;line-height:1.1">AE-grid ViT + Sentinel-2 late fusion</h1>
<p style="margin:0;font-size:18px;line-height:1.45;color:#4A5562">{intro}</p>
</div>
{gap}
{card}
<p style="margin:12px 0 0;font-size:15px;color:#4A5562">AlphaEarth · footprint ViT · AE+S2 late fusion · LORO · locked Section 3/4 · {run_stamp}</p>
</div></body></html>
"""


def inject_canvas(board: str, height: int) -> None:
    html = CANVAS.read_text()
    compressed = base64.b64encode(gzip.compress(board.encode("utf-8"), compresslevel=9)).decode("ascii")
    man_m = re.search(r'(<script type="__bundler/manifest">)(.*?)(</script>)', html, re.S)
    man = json.loads(man_m.group(2))
    man[SECTION11_ID] = {"mime": "text/html", "compressed": True, "data": compressed}
    html = html[: man_m.start(2)] + json.dumps(man, separators=(",", ":")) + html[man_m.end(2) :]

    po_m = re.search(r'(<script type="__bundler/page_order">)(.*?)(</script>)', html, re.S)
    order = json.loads(po_m.group(2))
    if SECTION11_ID in order:
        order = [x for x in order if x != SECTION11_ID]
    if SECTION10_ID not in order:
        raise SystemExit("Section 10 missing — inject Week 13 board first")
    order.insert(order.index(SECTION10_ID) + 1, SECTION11_ID)
    html = html[: po_m.start(2)] + json.dumps(order, separators=(",", ":")) + html[po_m.end(2) :]

    tpl_m = re.search(r'(<script type="__bundler/template">)(.*?)(</script>)', html, re.S)
    template = json.loads(tpl_m.group(2))
    template = re.sub(
        rf'<section class="board"><h2>11 .*?</h2><iframe src="about:blank#{SECTION11_ID}".*?</iframe></section>\n?',
        "",
        template,
        count=1,
        flags=re.S,
    )
    # Renumber legacy sections that still sit after 10
    if "11 Embedding dynamics" in template or "11 Generalisation performance" in template:
        mapping = [
            ("15 GHSL BUILT-S vs AlphaEarth (AE cell centres)", "16 GHSL BUILT-S vs AlphaEarth (AE cell centres)"),
            ("14 Training dynamics and loss function", "15 Training dynamics and loss function"),
            ("13 Country-wise LORO and cross-country generalisation", "14 Country-wise LORO and cross-country generalisation"),
            ("12 Embedding dynamics", "13 Embedding dynamics"),
            ("11 Embedding dynamics", "12 Embedding dynamics"),
            ("11 Generalisation performance", "12 Generalisation performance"),
        ]
        for old, new in mapping:
            if old in template:
                template = template.replace(old, new, 1)

    iframe = (
        f'<section class="board"><h2>11 AE-grid ViT + Sentinel-2 late fusion</h2>'
        f'<iframe src="about:blank#{SECTION11_ID}" title="11 AE-grid ViT + S2" '
        f'width="1200" height="{height}" loading="eager" '
        f'style="width:1200px;height:{height}px"></iframe></section>\n'
    )
    m = re.search(
        rf'<section class="board"><h2>10 .*?</h2><iframe src="about:blank#{SECTION10_ID}".*?</iframe></section>\n?',
        template,
        flags=re.S,
    )
    if not m:
        raise SystemExit("could not locate Section 10")
    template = template[: m.end()] + iframe + template[m.end() :]
    html = html[: tpl_m.start(2)] + json.dumps(template, separators=(",", ":")) + html[tpl_m.end(2) :]
    for old, new in (
        ("AlphaEarth Weeks 1–13 Canvas", "AlphaEarth Weeks 1–14 Canvas"),
        ("Loading AlphaEarth Weeks 1–13 Canvas...", "Loading AlphaEarth Weeks 1–14 Canvas..."),
        ("AlphaEarth Weeks 1–12 Canvas", "AlphaEarth Weeks 1–14 Canvas"),
        ("Loading AlphaEarth Weeks 1–12 Canvas...", "Loading AlphaEarth Weeks 1–14 Canvas..."),
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
        / "week14_ae_grid_vit_s2_loro"
        / "latest"
        / "artifacts"
        / "section11_ae_grid_vit_s2_metrics.json",
    )
    ap.add_argument("--height", type=int, default=3400)
    ap.add_argument("--no-inject", action="store_true")
    args = ap.parse_args()
    metrics_path = args.metrics.resolve()
    metrics = json.loads(metrics_path.read_text())
    board = build_board(metrics, metrics_path.parent.parent.name)
    out = metrics_path.parent / "section11_ae_grid_vit_s2_board.html"
    out.write_text(board)
    print(f"board → {out}")
    if not args.no_inject:
        inject_canvas(board, args.height)
        print(f"canvas injected id={SECTION11_ID}")
    p = metrics["buildings"]["soft_pooled"]
    print(
        f"vit={p['ae_vit_s2']['mean']:.3f} "
        f"cnn={p.get('ae_cnn_s2', {}).get('mean')} "
        f"s2={p.get('ae_s2', {}).get('mean')} "
        f"ghsl={p['ghsl_built_s']['mean']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
