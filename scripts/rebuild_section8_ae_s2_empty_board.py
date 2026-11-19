#!/usr/bin/env python3
"""Build Section 8 canvas board: AE+S2 with frac_empty head vs AE+S2 2-head."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANVAS = ROOT / "notebooks" / "ghsl_built_s_vs_ae" / "AlphaEarth Weeks 1–5 Canvas.html"
SECTION7_ID = "d0f7b5e4-8a1c-4d24-af6b-5c9e1a3d4e78"
SECTION8_ID = "e1a8c6f5-9b2d-4e35-b07c-6d0f2b4e5f89"

COLORS = ["#C2691A", "#1F6FA8", "#8E44AD", "#0F766E", "#9B2226", "#5C4B8A"]
NAMES = ["GHS-BUILT-S", "AE Ridge", "AE XGB", "AE+S2", "AE+S2+empty", "AE+S2+empty renorm"]


def f3(x: float) -> str:
    return f"{x:.3f}"


def bar_row(name: str, val: float, color: str, scale_max: float) -> str:
    w = max(2, int(round(320 * (val / scale_max)))) if scale_max > 0 else 2
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


def decorate_metro(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        bs = r["ghsl_built_s"]
        s2 = r.get("ae_s2", {"mean": float("nan"), "res": float("nan"), "nres": float("nan")})
        s2e = r["ae_s2_empty"]
        out.append(
            {
                "region": r["region"],
                "label": r.get("label", r["region"]),
                "bg": r.get("bg", "#F6F4EE"),
                "n": r["n"],
                "bs": bs,
                "s2": s2,
                "s2e": s2e,
                "margin": float(s2e["mean"] - s2["mean"]) if s2["mean"] == s2["mean"] else float("nan"),
                "empty_mae": r.get("empty_mae", {}).get("mae", float("nan")),
            }
        )
    return out


def metro_table(rows: list[dict]) -> str:
    head = (
        '<table style="width:100%;border-collapse:collapse;margin-top:8px">'
        '<tr><th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:left">Metro</th>'
        '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right">n</th>'
        '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right" colspan="3">BUILT-S</th>'
        '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right;border-left:1px solid #3a4a45" colspan="3">AE+S2</th>'
        '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right;border-left:1px solid #3a4a45" colspan="3">AE+S2+empty</th>'
        '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right;border-left:1px solid #3a4a45">Δ mean</th>'
        '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right">empty MAE</th></tr>'
        '<tr style="background:#1b2a26;color:#EDEBE4"><td></td><td></td>'
        + "".join(
            '<td style="padding:4px 10px;text-align:right;font-size:12px;border-left:1px solid #3a4a45">res</td>'
            '<td style="padding:4px 10px;text-align:right;font-size:12px">nres</td>'
            '<td style="padding:4px 10px;text-align:right;font-size:12px">mean</td>'
            for _ in range(3)
        )
        + '<td style="padding:4px 10px;text-align:right;font-size:12px;border-left:1px solid #3a4a45">S2e−S2</td>'
        '<td style="padding:4px 10px;text-align:right;font-size:12px">MAE</td></tr>'
    )
    body = []
    for r in rows:
        objs = [r["bs"], r["s2"], r["s2e"]]
        finite = [o["mean"] for o in objs if o["mean"] == o["mean"]]
        best = min(finite) if finite else 0.0

        def cell(v, best_v, border=False):
            b = "border-left:1px solid #D9D5C8;" if border else ""
            if v != v:
                return f'<td style="padding:7px 10px;text-align:right;{b}">—</td>'
            bold = "<b>" if abs(v - best_v) < 1e-12 else ""
            be = "</b>" if bold else ""
            return f'<td style="padding:7px 10px;text-align:right;font-variant-numeric:tabular-nums;{b}">{bold}{f3(v)}{be}</td>'

        margin = r["margin"]
        mcol = "#C2691A" if (margin == margin and margin >= 0) else "#5C4B8A"
        cells = "".join(
            f'{cell(o["res"], min(x["res"] for x in objs if x["res"]==x["res"]), True)}'
            f'{cell(o["nres"], min(x["nres"] for x in objs if x["nres"]==x["nres"]))}'
            f'{cell(o["mean"], best)}'
            for o in objs
        )
        em = r["empty_mae"]
        body.append(
            f'<tr style="background:{r["bg"]}">'
            f'<td style="padding:7px 10px;font-weight:600">{r["label"]}</td>'
            f'<td style="padding:7px 10px;text-align:right">{r["n"]:,}</td>'
            f"{cells}"
            f'<td style="padding:7px 10px;text-align:right;border-left:1px solid #D9D5C8;color:{mcol};font-weight:600">'
            f'{margin:+.3f}</td>'
            f'<td style="padding:7px 10px;text-align:right">{f3(em) if em==em else "—"}</td></tr>'
        )
    return head + "".join(body) + "</table>"


def build_board(metrics: dict, run_stamp: str) -> str:
    building = metrics["buildings"]
    tile = metrics["tiles"]
    b = building["soft_pooled"]
    t = tile["soft_pooled"]
    n_b = int(metrics.get("eval_n_buildings") or building["n_shared"])
    n_t = int(metrics.get("eval_n_tiles") or tile["n_shared"])
    tile_cov = float(tile.get("coverage") or 0.0)
    empty_mae = building.get("empty_mae", {}).get("ae_s2_empty", {}).get("mae", float("nan"))
    b_rows = decorate_metro(building["metro_soft"])
    beats = building.get("s2e_beats_s2", sum(1 for r in b_rows if r["margin"] == r["margin"] and r["margin"] < 0))

    delta = _g(b, "ae_s2_empty", "mean") - _g(b, "ae_s2", "mean")
    intro = (
        "Same locked Section 3/4 cohorts. "
        "<b>AE+S2+empty</b> = Week 9 AE 64-d + Sentinel-2 NRES features, but XGB LORO predicts "
        "<code>(soft_gt_frac_res, soft_gt_frac_nres, soft_gt_frac_empty)</code> jointly. "
        "Empty GT = mean week4 <code>frac_empty_full</code> over inside-polygon AE cells "
        "(<code>clip(1−res−nres)</code> at 10 m). "
        "Headline metric still <b>0.5·(MAE_res+MAE_nres)</b> — empty is auxiliary. "
        f"Δ mean = AE+S2+empty − AE+S2 ({delta:+.3f})."
    )

    keys = ["ghsl_built_s", "ae_ridge", "ae_xgb", "ae_s2", "ae_s2_empty", "ae_s2_empty_renorm"]
    scale_vals = []
    for pool in (b, t):
        for k in keys:
            if k in pool:
                scale_vals.extend([_g(pool, k, "res"), _g(pool, k, "nres")])
    scale = max([v for v in scale_vals if v == v] + [0.01])

    def panel(title, pool):
        rows_res = [(NAMES[i], _g(pool, k, "res"), COLORS[i]) for i, k in enumerate(keys)]
        rows_nres = [(NAMES[i], _g(pool, k, "nres"), COLORS[i]) for i, k in enumerate(keys)]
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
        '<p style="margin:0;font-size:17px;font-weight:600">Does an empty head help RES / NRES?</p>'
        '<p style="margin:0 0 6px;font-size:15px;line-height:1.4;color:#4A5562">'
        f"Pooled empty MAE = <b>{f3(empty_mae)}</b>. Bars: locked-cohort soft MAE (lower better).</p>"
        + panel(f"Building · {n_b:,}", b)
        + panel(f"100 m · {n_t:,} (coverage {100*tile_cov:.0f}%)", t)
        + "</div>"
    )

    card = (
        '<div style="display:flex;flex-direction:column;gap:10px;background:#FCFBF7;'
        'border:1px solid #D9D5C8;border-radius:12px;padding:24px">'
        f'<p style="margin:0;font-size:17px;font-weight:600">Per metro, building · '
        f'<span style="color:#5C4B8A">AE+S2+empty beats AE+S2 in {beats} of {len(b_rows)}</span></p>'
        f'<p style="margin:0;font-size:15px">Pooled mean: AE+S2+empty <b>{f3(_g(b,"ae_s2_empty","mean"))}</b> · '
        f'AE+S2 {f3(_g(b,"ae_s2","mean"))} · BUILT-S {f3(_g(b,"ghsl_built_s","mean"))}.</p>'
        f"{metro_table(b_rows)}"
        "</div>"
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>AE+S2 empty head · Fraction MAE</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&amp;family=Source+Serif+4:wght@400;600&amp;display=swap">
<style>body{{margin:0}} table{{border-collapse:collapse;width:100%}} td{{border-bottom:1px solid #D9D5C8}} code{{font-size:13px}}</style></head><body>
<div style="width:1200px;box-sizing:border-box;padding:56px;background:#F6F4EE;color:#14231F;font-family:'IBM Plex Sans',Arial,sans-serif;display:flex;flex-direction:column;gap:28px">
<div style="display:flex;flex-direction:column;gap:8px">
<p style="margin:0;font-size:15px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#5C4B8A">Section 8 · Week 11</p>
<h1 style="margin:0;font-family:'Source Serif 4',Georgia,serif;font-size:48px;font-weight:600;line-height:1.1">AE+S2 with frac_empty head</h1>
<p style="margin:0;font-size:18px;line-height:1.45;color:#4A5562">{intro}</p>
</div>
{gap}
{card}
<p style="margin:12px 0 0;font-size:15px;color:#4A5562">AlphaEarth · AE+S2 features · 3-head XGB LORO · locked Section 3/4 IDs · {run_stamp}</p>
</div></body></html>
"""


def inject_canvas(board: str, height: int) -> None:
    html = CANVAS.read_text()
    compressed = base64.b64encode(gzip.compress(board.encode("utf-8"), compresslevel=9)).decode("ascii")

    man_m = re.search(r'(<script type="__bundler/manifest">)(.*?)(</script>)', html, re.S)
    man = json.loads(man_m.group(2))
    man[SECTION8_ID] = {"mime": "text/html", "compressed": True, "data": compressed}
    html = html[: man_m.start(2)] + json.dumps(man, separators=(",", ":")) + html[man_m.end(2) :]

    po_m = re.search(r'(<script type="__bundler/page_order">)(.*?)(</script>)', html, re.S)
    order = json.loads(po_m.group(2))
    if SECTION8_ID in order:
        order = [x for x in order if x != SECTION8_ID]
    if SECTION7_ID not in order:
        raise SystemExit("Section 7 missing — inject Section 7 first")
    order.insert(order.index(SECTION7_ID) + 1, SECTION8_ID)
    html = html[: po_m.start(2)] + json.dumps(order, separators=(",", ":")) + html[po_m.end(2) :]

    tpl_m = re.search(r'(<script type="__bundler/template">)(.*?)(</script>)', html, re.S)
    template = json.loads(tpl_m.group(2))
    template = re.sub(
        rf'<section class="board"><h2>8 .*?</h2><iframe src="about:blank#{SECTION8_ID}".*?</iframe></section>\n?',
        "",
        template,
        count=1,
        flags=re.S,
    )
    if "8 Generalisation performance" in template:
        mapping = [
            ("12 GHSL BUILT-S vs AlphaEarth (AE cell centres)", "13 GHSL BUILT-S vs AlphaEarth (AE cell centres)"),
            ("11 Training dynamics and loss function", "12 Training dynamics and loss function"),
            ("10 Country-wise LORO and cross-country generalisation", "11 Country-wise LORO and cross-country generalisation"),
            ("9 Embedding dynamics", "10 Embedding dynamics"),
            ("8 Generalisation performance", "9 Generalisation performance"),
        ]
        for old, new in mapping:
            if old in template:
                template = template.replace(old, new, 1)

    iframe = (
        f'<section class="board"><h2>8 AE+S2 with frac_empty head</h2>'
        f'<iframe src="about:blank#{SECTION8_ID}" title="8 AE+S2 with frac_empty head" '
        f'width="1200" height="{height}" loading="eager" '
        f'style="width:1200px;height:{height}px"></iframe></section>\n'
    )
    m = re.search(
        rf'<section class="board"><h2>7 .*?</h2><iframe src="about:blank#{SECTION7_ID}".*?</iframe></section>\n?',
        template,
        flags=re.S,
    )
    if not m:
        raise SystemExit("could not locate Section 7 block")
    template = template[: m.end()] + iframe + template[m.end() :]
    html = html[: tpl_m.start(2)] + json.dumps(template, separators=(",", ":")) + html[tpl_m.end(2) :]
    for old, new in (
        ("AlphaEarth Weeks 1–10 Canvas", "AlphaEarth Weeks 1–11 Canvas"),
        ("Loading AlphaEarth Weeks 1–10 Canvas...", "Loading AlphaEarth Weeks 1–11 Canvas..."),
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
        / "week11_ae_s2_empty_head_loro"
        / "latest"
        / "artifacts"
        / "section8_ae_s2_empty_metrics.json",
    )
    ap.add_argument("--height", type=int, default=3200)
    ap.add_argument("--no-inject", action="store_true")
    args = ap.parse_args()
    metrics_path = args.metrics.resolve()
    metrics = json.loads(metrics_path.read_text())
    board = build_board(metrics, metrics_path.parent.parent.name)
    out = metrics_path.parent / "section8_ae_s2_empty_board.html"
    out.write_text(board)
    print(f"board → {out}")
    if not args.no_inject:
        inject_canvas(board, args.height)
        print(f"canvas injected id={SECTION8_ID}")
    b = metrics["buildings"]["soft_pooled"]
    print(
        f"buildings: s2e={b['ae_s2_empty']['mean']:.3f} s2={b.get('ae_s2',{}).get('mean')} "
        f"ghsl={b['ghsl_built_s']['mean']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
