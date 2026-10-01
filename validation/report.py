"""Write validation summary artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _flatten_functional(level: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for system in ("model", "ghsl"):
        metrics = summary.get("functional", {}).get(system, {})
        for metric, value in metrics.items():
            rows.append({"level": level, "benchmark": "functional", "system": system, "metric": metric, "value": value})
    for system in ("model", "ghsl"):
        metrics = summary.get("end_to_end", {}).get(system, {})
        for metric, value in metrics.items():
            rows.append({"level": level, "benchmark": "end_to_end", "system": system, "metric": metric, "value": value})
    return rows


def write_comparison_table(rows: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    df = pd.DataFrame(rows)
    csv_path = out_dir / "comparison_table.csv"
    md_path = out_dir / "comparison_table.md"
    df.to_csv(csv_path, index=False)

    # Compact markdown: macro-F1 + RES-as-positive class metrics
    focus_metrics = [
        "n", "macro_f1", "precision", "recall", "residential_f1", "residential_iou", "accuracy",
    ]
    focus = df[(df["benchmark"] == "functional") & (df["metric"].isin(focus_metrics))]
    lines = [
        "# GHSL vs model comparison",
        "",
        "Headline: **macro-F1**. Class metrics use **RES = positive** (precision / recall / residential_f1 / IoU).",
        "",
    ]
    for level in focus["level"].unique():
        lines.append(f"## {level}-level (functional classification)")
        lines.append("")
        lines.append("| metric | model | GHSL |")
        lines.append("|---|---:|---:|")
        sub = focus[focus["level"] == level]
        for metric in focus_metrics:
            m = sub[(sub["system"] == "model") & (sub["metric"] == metric)]
            g = sub[(sub["system"] == "ghsl") & (sub["metric"] == metric)]
            mv = f"{m['value'].iloc[0]:.4f}" if len(m) else "—"
            gv = f"{g['value'].iloc[0]:.4f}" if len(g) else "—"
            if metric == "n":
                mv = f"{int(m['value'].iloc[0])}" if len(m) else "—"
                gv = f"{int(g['value'].iloc[0])}" if len(g) else "—"
            lines.append(f"| {metric} | {mv} | {gv} |")
        lines.append("")
    # End-to-end coverage
    e2e = df[df["benchmark"] == "end_to_end"]
    if len(e2e):
        lines.append("## End-to-end (abstentions counted)")
        lines.append("")
        lines.append("| level | system | coverage | selective_macro_f1 | effective_accuracy |")
        lines.append("|---|---|---:|---:|---:|")
        for level in e2e["level"].unique():
            for system in ("model", "ghsl"):
                s = e2e[(e2e["level"] == level) & (e2e["system"] == system)]
                def get(m):
                    r = s[s["metric"] == m]
                    return f"{r['value'].iloc[0]:.4f}" if len(r) else "—"
                lines.append(
                    f"| {level} | {system} | {get('coverage')} | {get('selective_macro_f1')} | {get('effective_accuracy')} |"
                )
        lines.append("")
    md_path.write_text("\n".join(lines))
    return csv_path, md_path


def write_summary_json(payload: dict[str, Any], out_dir: Path) -> Path:
    path = out_dir / "summary.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def disagreement_buildings(per_building: pd.DataFrame, gt_gdf, out_path: Path) -> Path | None:
    if per_building.empty:
        return None
    # Buildings where model and GHSL disagree on functional subset
    d = per_building[
        per_building["ghsl_covered"]
        & (per_building["model_prediction"] != per_building["ghsl_prediction"])
    ].copy()
    if d.empty:
        return None
    g = gt_gdf.merge(d, on="building_id", how="inner")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        g.to_file(out_path, driver="GPKG")
        return out_path
    except Exception:
        # fallback parquet
        alt = out_path.with_suffix(".parquet")
        g.drop(columns="geometry", errors="ignore").to_parquet(alt, index=False)
        return alt
