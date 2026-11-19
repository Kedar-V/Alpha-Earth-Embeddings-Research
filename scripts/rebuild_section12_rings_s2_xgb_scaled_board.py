#!/usr/bin/env python3
"""Section 12: Rings+S2 with XGB scaled for ~6× feature width vs thin recipe / AE+S2."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANVAS = ROOT / "notebooks" / "ghsl_built_s_vs_ae" / "AlphaEarth Weeks 1–5 Canvas.html"
SECTION11_ID = "b4d1f9c8-2e5a-4168-e30f-9a3c5d7e8c12"
SECTION12_ID = "c5e2a0d9-3f6b-4279-f41a-0b4d6e8f9d23"

COLORS = ["#C2691A", "#0F766E", "#9B2226", "#2E7D32", "#8E44AD"]
KEYS = ["ghsl_built_s", "ae_s2", "ae_ring_s2_thin", "ae_ring_s2_scaled", "ae_cnn_s2"]
NAMES = ["GHS-BUILT-S", "AE+S2", "Rings+S2 thin", "Rings+S2 scaled", "CNN+S2"]


def f3(x: float) -> str:
    return f"{x:.3f}"


def bar_row(name: str, val: float, color: str, scale_max: float) -> str:
    w = max(2, int(round(320 * (val / scale_max)))) if scale_max > 0 and val == val else 2
    return (
        f'<div style="display:flex;align-items:center;gap:10px" title="{name}: MAE {f3(val)}">'
        f'<p style="margin:0;width:140px;font-size:14px">{name}</p>'
        f'<div style="width:{w}px;height:18px;background:{color};border-radius:0 4px 4px 0"></div>'
        f'<p style="margin:0;font-size:14px;font-weight:600;font-variant-numeric:tabular-nums">{f3(val)}</p>'
        f"</div>"
    )


def _g(d: dict, key: str, field: str) -> float:
    if key not in d:
        return float("nan")
    return float(d[key][field])


def metro_table(rows: list[dict]) -> str:
    labs = ("AE+S2", "Thin", "Scaled")
    keys = ("ae_s2", "ae_ring_s2_thin", "ae_ring_s2_scaled")
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
        + '<td style="padding:4px 10px;text-align:right;font-size:12px;border-left:1px solid #3a4a45">sc−thin</td></tr>'
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
        d = (r.get("ae_ring_s2_scaled", {}).get("mean", float("nan"))) - (
            r.get("ae_ring_s2_thin", {}).get("mean", float("nan"))
        )
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
    xgb = metrics.get("xgb", {})
    est = xgb.get("n_estimators", 1800)
    cs = xgb.get("colsample_bytree", 0.172)
    lr = xgb.get("learning_rate", 0.025)
    n_feat = metrics.get("n_features", 464)
    d_thin = _g(pool, "ae_ring_s2_scaled", "mean") - _g(pool, "ae_ring_s2_thin", "mean")
    d_s2 = _g(pool, "ae_ring_s2_scaled", "mean") - _g(pool, "ae_s2", "mean")
    beats_thin = b.get("scaled_beats_thin", 0)
    beats_s2 = b.get("scaled_beats_s2", 0)

    intro = (
        f"Same locked IDs and same <b>{n_feat}-d</b> Rings+S2 features as Week 10. "
        f"Week 10 used the thin AE+S2 XGB recipe (300 trees, colsample 0.8) on ~6× more dims and diluted S2. "
        f"This run scales capacity: <b>n_estimators={est}</b>, <b>colsample={cs}</b> (~80 feats/tree), "
        f"<b>lr={lr}</b>. Headline <b>0.5·(MAE_res+MAE_nres)</b>. "
        f"Δ vs thin {d_thin:+.3f} · vs AE+S2 {d_s2:+.3f}."
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
        f'<span style="color:#0B525B">scaled beats thin in {beats_thin}/10 · AE+S2 in {beats_s2}/10</span></p>'
        f'<p style="margin:0;font-size:15px">Pooled: scaled <b>{f3(_g(pool, "ae_ring_s2_scaled", "mean"))}</b> · '
        f'thin {f3(_g(pool, "ae_ring_s2_thin", "mean"))} · AE+S2 {f3(_g(pool, "ae_s2", "mean"))} · '
        f'CNN+S2 {f3(_g(pool, "ae_cnn_s2", "mean"))}</p>'
        f'{metro_table(b.get("metro_soft") or [])}'
        "</div>"
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Rings+S2 scaled XGB · Fraction MAE</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&amp;family=Source+Serif+4:wght@400;600&amp;display=swap">
<style>body{{margin:0}} table{{border-collapse:collapse;width:100%}} td{{border-bottom:1px solid #D9D5C8}}</style></head><body>
<div style="width:1200px;box-sizing:border-box;padding:56px;background:#F6F4EE;color:#14231F;font-family:'IBM Plex Sans',Arial,sans-serif;display:flex;flex-direction:column;gap:28px">
<div style="display:flex;flex-direction:column;gap:8px">
<p style="margin:0;font-size:15px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#0B525B">Section 12 · Week 15</p>
<h1 style="margin:0;font-family:'Source Serif 4',Georgia,serif;font-size:44px;font-weight:600;line-height:1.1">Rings+S2 · XGB scaled for 6× features</h1>
<p style="margin:0;font-size:18px;line-height:1.45;color:#4A5562">{intro}</p>
</div>
{gap}
{card}
<p style="margin:12px 0 0;font-size:15px;color:#4A5562">AlphaEarth · rings+S2 · scaled XGB · LORO · locked Section 3/4 · {run_stamp}</p>
</div></body></html>
"""


def inject_canvas(board: str, height: int) -> None:
    html = CANVAS.read_text()
    compressed = base64.b64encode(gzip.compress(board.encode("utf-8"), compresslevel=9)).decode("ascii")
    man_m = re.search(r'(<script type="__bundler/manifest">)(.*?)(</script>)', html, re.S)
    man = json.loads(man_m.group(2))
    man[SECTION12_ID] = {"mime": "text/html", "compressed": True, "data": compressed}
    html = html[: man_m.start(2)] + json.dumps(man, separators=(",", ":")) + html[man_m.end(2) :]

    po_m = re.search(r'(<script type="__bundler/page_order">)(.*?)(</script>)', html, re.S)
    order = json.loads(po_m.group(2))
    if SECTION12_ID in order:
        order = [x for x in order if x != SECTION12_ID]
    if SECTION11_ID not in order:
        raise SystemExit("Section 11 missing — inject Week 14 board first")
    order.insert(order.index(SECTION11_ID) + 1, SECTION12_ID)
    html = html[: po_m.start(2)] + json.dumps(order, separators=(",", ":")) + html[po_m.end(2) :]

    tpl_m = re.search(r'(<script type="__bundler/template">)(.*?)(</script>)', html, re.S)
    template = json.loads(tpl_m.group(2))
    template = re.sub(
        rf'<section class="board"><h2>12 .*?</h2><iframe src="about:blank#{SECTION12_ID}".*?</iframe></section>\n?',
        "",
        template,
        count=1,
        flags=re.S,
    )
    iframe = (
        f'<section class="board"><h2>12 Rings+S2 · XGB scaled for 6× features</h2>'
        f'<iframe src="about:blank#{SECTION12_ID}" title="12 Rings+S2 scaled XGB" '
        f'width="1200" height="{height}" loading="eager" '
        f'style="width:1200px;height:{height}px"></iframe></section>\n'
    )
    m = re.search(
        rf'<section class="board"><h2>11 .*?</h2><iframe src="about:blank#{SECTION11_ID}".*?</iframe></section>\n?',
        template,
        flags=re.S,
    )
    if not m:
        raise SystemExit("could not locate Section 11")
    template = template[: m.end()] + iframe + template[m.end() :]
    html = html[: tpl_m.start(2)] + json.dumps(template, separators=(",", ":")) + html[tpl_m.end(2) :]
    for old, new in (
        ("AlphaEarth Weeks 1–14 Canvas", "AlphaEarth Weeks 1–15 Canvas"),
        ("Loading AlphaEarth Weeks 1–14 Canvas...", "Loading AlphaEarth Weeks 1–15 Canvas..."),
        ("AlphaEarth Weeks 1–13 Canvas", "AlphaEarth Weeks 1–15 Canvas"),
        ("Loading AlphaEarth Weeks 1–13 Canvas...", "Loading AlphaEarth Weeks 1–15 Canvas..."),
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
        / "week15_rings_s2_xgb_scaled_loro"
        / "latest"
        / "artifacts"
        / "section12_rings_s2_xgb_scaled_metrics.json",
    )
    ap.add_argument("--height", type=int, default=3400)
    ap.add_argument("--no-inject", action="store_true")
    args = ap.parse_args()
    metrics_path = args.metrics.resolve()
    metrics = json.loads(metrics_path.read_text())
    board = build_board(metrics, metrics_path.parent.parent.name)
    out = metrics_path.parent / "section12_rings_s2_xgb_scaled_board.html"
    out.write_text(board)
    print(f"board → {out}")
    if not args.no_inject:
        inject_canvas(board, args.height)
        print(f"canvas injected id={SECTION12_ID}")
    p = metrics["buildings"]["soft_pooled"]
    print(
        f"scaled={p['ae_ring_s2_scaled']['mean']:.3f} "
        f"thin={p.get('ae_ring_s2_thin', {}).get('mean')} "
        f"s2={p.get('ae_s2', {}).get('mean')} "
        f"cnn={p.get('ae_cnn_s2', {}).get('mean')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
