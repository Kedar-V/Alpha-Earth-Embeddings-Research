#!/usr/bin/env python3
"""Build Section 9 canvas board: AE+S2 XGBClassifier vs GHS-BUILT-S (RES/NRES F1)."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANVAS = ROOT / "notebooks" / "ghsl_built_s_vs_ae" / "AlphaEarth Weeks 1–5 Canvas.html"
SECTION8_ID = "e1a8c6f5-9b2d-4e35-b07c-6d0f2b4e5f89"
SECTION9_ID = "f2b9d7a6-0c3e-4f46-c18d-7e1a3c5f6a90"

COLORS = ["#C2691A", "#1F6FA8", "#8E44AD", "#0F766E", "#5C4B8A", "#1B4332"]
# order in bars
KEYS = ["ghsl_built_s", "ae_ridge", "ae_xgb", "ae_s2_reg", "ae_s2_empty_reg", "ae_s2_clf"]
NAMES = ["GHS-BUILT-S", "AE Ridge", "AE XGB", "AE+S2 reg", "AE+S2+empty reg", "AE+S2 clf"]


def f3(x: float) -> str:
    return f"{x:.3f}"


def bar_row(name: str, val: float, color: str, scale_max: float = 1.0) -> str:
    w = max(2, int(round(320 * (val / scale_max)))) if scale_max > 0 and val == val else 2
    return (
        f'<div style="display:flex;align-items:center;gap:10px" title="{name}: {f3(val)}">'
        f'<p style="margin:0;width:150px;font-size:14px">{name}</p>'
        f'<div style="width:{w}px;height:18px;background:{color};border-radius:0 4px 4px 0"></div>'
        f'<p style="margin:0;font-size:14px;font-weight:600;font-variant-numeric:tabular-nums">{f3(val)}</p>'
        f"</div>"
    )


def _g(d: dict, key: str, field: str) -> float:
    if key not in d:
        return float("nan")
    return float(d[key][field])


def metro_table(rows: list[dict]) -> str:
    head = (
        '<table style="width:100%;border-collapse:collapse;margin-top:8px">'
        '<tr><th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:left">Metro</th>'
        '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right">n</th>'
        '<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right">RES%</th>'
        + "".join(
            f'<th style="padding:8px 10px;background:#14231F;color:#EDEBE4;text-align:right;'
            f'border-left:1px solid #3a4a45" colspan="3">{lab}</th>'
            for lab in ("BUILT-S", "AE+S2 reg", "AE+S2 clf")
        )
        + "</tr>"
        '<tr style="background:#1b2a26;color:#EDEBE4"><td></td><td></td><td></td>'
        + "".join(
            '<td style="padding:4px 10px;text-align:right;font-size:12px;border-left:1px solid #3a4a45">RES</td>'
            '<td style="padding:4px 10px;text-align:right;font-size:12px">NRES</td>'
            '<td style="padding:4px 10px;text-align:right;font-size:12px">macro</td>'
            for _ in range(3)
        )
        + "</tr>"
    )
    body = []
    for r in rows:
        models = [
            r.get("ghsl_built_s", {}),
            r.get("ae_s2_reg", {}),
            r.get("ae_s2_clf", {}),
        ]
        macros = [m.get("macro_f1", float("nan")) for m in models]
        best = max([x for x in macros if x == x], default=0.0)

        def cell(v, best_v, border=False):
            b = "border-left:1px solid #D9D5C8;" if border else ""
            if v != v:
                return f'<td style="padding:7px 10px;text-align:right;{b}">—</td>'
            bold = "<b>" if abs(v - best_v) < 1e-12 else ""
            be = "</b>" if bold else ""
            return f'<td style="padding:7px 10px;text-align:right;font-variant-numeric:tabular-nums;{b}">{bold}{f3(v)}{be}</td>'

        cells = ""
        for m in models:
            cells += (
                f'{cell(m.get("res_f1", float("nan")), max([x.get("res_f1", float("nan")) for x in models if x.get("res_f1", float("nan"))==x.get("res_f1", float("nan"))], default=0), True)}'
                f'{cell(m.get("nres_f1", float("nan")), max([x.get("nres_f1", float("nan")) for x in models if x.get("nres_f1", float("nan"))==x.get("nres_f1", float("nan"))], default=0))}'
                f'{cell(m.get("macro_f1", float("nan")), best)}'
            )
        body.append(
            f'<tr style="background:{r.get("bg","#F6F4EE")}">'
            f'<td style="padding:7px 10px;font-weight:600">{r.get("label", r["region"])}</td>'
            f'<td style="padding:7px 10px;text-align:right">{r["n"]:,}</td>'
            f'<td style="padding:7px 10px;text-align:right">{100*r.get("res_rate",0):.0f}%</td>'
            f"{cells}</tr>"
        )
    return head + "".join(body) + "</table>"


def build_board(metrics: dict, run_stamp: str) -> str:
    b = metrics["buildings"]
    pooled = b["pooled"]
    n_b = int(metrics.get("eval_n_buildings") or b["n_shared"])
    res_rate = float(b.get("res_rate") or 0)
    mixed = float(b.get("mixed_both_rate") or 0)

    intro = (
        "Binary soft-argmax labels on locked Section 3/4 buildings: "
        "<code>RES</code> iff <code>soft_gt_frac_res ≥ soft_gt_frac_nres</code> "
        f"(RES rate {100*res_rate:.0f}%; both&gt;0 mix {100*mixed:.1f}%). "
        "<b>AE+S2 clf</b> = XGBClassifier LORO on the same AE 64-d + Sentinel-2 NRES features as Week 9. "
        "Baselines = soft-argmax of continuous fraction heads (GHSL, Ridge, XGB, AE+S2 reg). "
        "Metrics: RES F1, NRES F1, macro F1."
    )

    def panel(title: str, field: str) -> str:
        rows = [(NAMES[i], _g(pooled, k, field), COLORS[i]) for i, k in enumerate(KEYS)]
        return (
            f'<div style="flex:1;display:flex;flex-direction:column;gap:6px">'
            f'<p style="margin:0 0 2px;font-size:15px;font-weight:600">{title}</p>'
            + "".join(bar_row(n, v, c) for n, v, c in rows if v == v)
            + "</div>"
        )

    gap = (
        '<div style="display:flex;flex-direction:column;gap:10px;background:#FCFBF7;'
        'border:1px solid #D9D5C8;border-radius:12px;padding:24px">'
        f'<p style="margin:0;font-size:17px;font-weight:600">Building classification · {n_b:,} locked IDs</p>'
        '<p style="margin:0 0 6px;font-size:15px;color:#4A5562">Higher is better. Shared 0–1 scale.</p>'
        f'<div style="display:flex;gap:28px">{panel("RES F1","res_f1")}{panel("NRES F1","nres_f1")}{panel("Macro F1","macro_f1")}</div>'
        "</div>"
    )

    clf = pooled["ae_s2_clf"]
    ghsl = pooled["ghsl_built_s"]
    s2 = pooled.get("ae_s2_reg", {})
    card = (
        '<div style="display:flex;flex-direction:column;gap:10px;background:#FCFBF7;'
        'border:1px solid #D9D5C8;border-radius:12px;padding:24px">'
        f'<p style="margin:0;font-size:17px;font-weight:600">Pooled · '
        f'clf macro <b>{f3(clf["macro_f1"])}</b> · GHSL {f3(ghsl["macro_f1"])} · '
        f'AE+S2 reg {f3(s2.get("macro_f1", float("nan")))}</p>'
        f'<p style="margin:0;font-size:15px">clf RES F1 <b>{f3(clf["res_f1"])}</b> / NRES F1 <b>{f3(clf["nres_f1"])}</b> · '
        f'GHSL RES {f3(ghsl["res_f1"])} / NRES {f3(ghsl["nres_f1"])}</p>'
        f'{metro_table(b.get("metro") or [])}'
        "</div>"
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>AE+S2 XGBClassifier · RES vs NRES F1</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&amp;family=Source+Serif+4:wght@400;600&amp;display=swap">
<style>body{{margin:0}} table{{border-collapse:collapse;width:100%}} td{{border-bottom:1px solid #D9D5C8}} code{{font-size:13px}}</style></head><body>
<div style="width:1200px;box-sizing:border-box;padding:56px;background:#F6F4EE;color:#14231F;font-family:'IBM Plex Sans',Arial,sans-serif;display:flex;flex-direction:column;gap:28px">
<div style="display:flex;flex-direction:column;gap:8px">
<p style="margin:0;font-size:15px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#1B4332">Section 9 · Week 12</p>
<h1 style="margin:0;font-family:'Source Serif 4',Georgia,serif;font-size:48px;font-weight:600;line-height:1.1">RES vs NRES as classification</h1>
<p style="margin:0;font-size:18px;line-height:1.45;color:#4A5562">{intro}</p>
</div>
{gap}
{card}
<p style="margin:12px 0 0;font-size:15px;color:#4A5562">AlphaEarth · AE+S2 XGBClassifier LORO · soft-argmax labels · locked Section 3/4 · {run_stamp}</p>
</div></body></html>
"""


def inject_canvas(board: str, height: int) -> None:
    html = CANVAS.read_text()
    compressed = base64.b64encode(gzip.compress(board.encode("utf-8"), compresslevel=9)).decode("ascii")

    man_m = re.search(r'(<script type="__bundler/manifest">)(.*?)(</script>)', html, re.S)
    man = json.loads(man_m.group(2))
    man[SECTION9_ID] = {"mime": "text/html", "compressed": True, "data": compressed}
    html = html[: man_m.start(2)] + json.dumps(man, separators=(",", ":")) + html[man_m.end(2) :]

    po_m = re.search(r'(<script type="__bundler/page_order">)(.*?)(</script>)', html, re.S)
    order = json.loads(po_m.group(2))
    if SECTION9_ID in order:
        order = [x for x in order if x != SECTION9_ID]
    if SECTION8_ID not in order:
        raise SystemExit("Section 8 missing")
    order.insert(order.index(SECTION8_ID) + 1, SECTION9_ID)
    html = html[: po_m.start(2)] + json.dumps(order, separators=(",", ":")) + html[po_m.end(2) :]

    tpl_m = re.search(r'(<script type="__bundler/template">)(.*?)(</script>)', html, re.S)
    template = json.loads(tpl_m.group(2))
    template = re.sub(
        rf'<section class="board"><h2>9 .*?</h2><iframe src="about:blank#{SECTION9_ID}".*?</iframe></section>\n?',
        "",
        template,
        count=1,
        flags=re.S,
    )
    if "9 Generalisation performance" in template:
        mapping = [
            ("13 GHSL BUILT-S vs AlphaEarth (AE cell centres)", "14 GHSL BUILT-S vs AlphaEarth (AE cell centres)"),
            ("12 Training dynamics and loss function", "13 Training dynamics and loss function"),
            ("11 Country-wise LORO and cross-country generalisation", "12 Country-wise LORO and cross-country generalisation"),
            ("10 Embedding dynamics", "11 Embedding dynamics"),
            ("9 Generalisation performance", "10 Generalisation performance"),
        ]
        for old, new in mapping:
            if old in template:
                template = template.replace(old, new, 1)

    iframe = (
        f'<section class="board"><h2>9 RES vs NRES classification (AE+S2 XGB)</h2>'
        f'<iframe src="about:blank#{SECTION9_ID}" title="9 RES vs NRES classification" '
        f'width="1200" height="{height}" loading="eager" '
        f'style="width:1200px;height:{height}px"></iframe></section>\n'
    )
    m = re.search(
        rf'<section class="board"><h2>8 .*?</h2><iframe src="about:blank#{SECTION8_ID}".*?</iframe></section>\n?',
        template,
        flags=re.S,
    )
    if not m:
        raise SystemExit("could not locate Section 8")
    template = template[: m.end()] + iframe + template[m.end() :]
    html = html[: tpl_m.start(2)] + json.dumps(template, separators=(",", ":")) + html[tpl_m.end(2) :]
    for old, new in (
        ("AlphaEarth Weeks 1–11 Canvas", "AlphaEarth Weeks 1–12 Canvas"),
        ("Loading AlphaEarth Weeks 1–11 Canvas...", "Loading AlphaEarth Weeks 1–12 Canvas..."),
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
        / "week12_ae_s2_xgb_clf_loro"
        / "latest"
        / "artifacts"
        / "section9_ae_s2_clf_metrics.json",
    )
    ap.add_argument("--height", type=int, default=2800)
    ap.add_argument("--no-inject", action="store_true")
    args = ap.parse_args()
    metrics_path = args.metrics.resolve()
    metrics = json.loads(metrics_path.read_text())
    board = build_board(metrics, metrics_path.parent.parent.name)
    out = metrics_path.parent / "section9_ae_s2_clf_board.html"
    out.write_text(board)
    print(f"board → {out}")
    if not args.no_inject:
        inject_canvas(board, args.height)
        print(f"canvas injected id={SECTION9_ID}")
    p = metrics["buildings"]["pooled"]
    print(
        f"clf macro={p['ae_s2_clf']['macro_f1']:.3f} RES={p['ae_s2_clf']['res_f1']:.3f} NRES={p['ae_s2_clf']['nres_f1']:.3f} | "
        f"GHSL macro={p['ghsl_built_s']['macro_f1']:.3f} RES={p['ghsl_built_s']['res_f1']:.3f} NRES={p['ghsl_built_s']['nres_f1']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
