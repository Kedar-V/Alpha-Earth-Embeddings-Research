#!/usr/bin/env python3
"""Build dashboard_metrics.json for the embedding globe viewer.

Combines:
  - per-city classification + probability metrics on the viz sample (points.json)
  - official leave-one-city-out holdout metrics (India/Japan LORO CSVs, USA week4 folds)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "viz" / "usa_embedding_globe" / "data"
GLOBE_POINTS = DATA / "points.json"
DASH_POINTS = DATA / "dashboard_points.json"
OUT = DATA / "dashboard_metrics.json"


def default_points_path() -> Path:
    return DASH_POINTS if DASH_POINTS.exists() else GLOBE_POINTS

CITY_TO_WEEK4 = {
    "USA_ATLANTA": "Atlanta_SE",
    "USA_BOSTON": "Boston_NE",
    "USA_CHICAGO": "Chicago_MW",
    "USA_DENVER": "Denver_W",
    "USA_HOUSTON": "Houston_S",
    "USA_LOS_ANGELES": "LA_W",
    "USA_NEW_YORK": "NYC_NE",
    "USA_PHOENIX": "Phoenix_SW",
    "USA_SAN_FRANCISCO": "SF_W",
    "USA_SEATTLE": "Seattle_NW",
}
WEEK4_TO_CITY = {v: k for k, v in CITY_TO_WEEK4.items()}

# GHS-SMOD study rollup (peri_urban ≈ suburban / peri-urban).
DOU_ORDER = ("urban", "peri_urban", "rural")
DOU_UI = {
    "urban": "urban",
    "peri_urban": "suburban",
    "rural": "rural",
}


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _f1(y: np.ndarray, pred: np.ndarray, lab: int) -> float:
    tp = int(((pred == lab) & (y == lab)).sum())
    fp = int(((pred == lab) & (y != lab)).sum())
    fn = int(((pred != lab) & (y == lab)).sum())
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    return _safe_div(2 * prec * rec, prec + rec)


def _auroc(y: np.ndarray, score: np.ndarray) -> float | None:
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None
    order = np.argsort(score)
    y_sorted = y[order]
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos < 1 or n_neg < 1:
        return None
    ranks = np.empty(len(y_sorted), dtype=np.float64)
    # average ranks for ties
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and score[order[j + 1]] == score[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    sum_ranks_pos = float(ranks[y == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _metrics_block(arr: list[dict]) -> dict | None:
    if not arr:
        return None
    y = np.asarray([int(p["gt"]) for p in arr], dtype=np.int64)
    pred = np.asarray([int(p["pred"]) for p in arr], dtype=np.int64)
    proba = np.asarray(
        [
            float(p["proba_res"]) if p.get("proba_res") is not None else np.nan
            for p in arr
        ],
        dtype=np.float64,
    )
    ok = np.isfinite(proba)
    f1_res = _f1(y, pred, 1)
    f1_nres = _f1(y, pred, 0)
    mean_p_res = float(np.nanmean(proba[y == 1])) if (y == 1).any() else None
    mean_p_nres = float(np.nanmean(proba[y == 0])) if (y == 0).any() else None
    delta_p = (
        None
        if mean_p_res is None or mean_p_nres is None
        else float(mean_p_res - mean_p_nres)
    )
    return {
        "n": int(len(arr)),
        "n_res": int((y == 1).sum()),
        "n_nres": int((y == 0).sum()),
        "accuracy": float((pred == y).mean()),
        "macro_f1": 0.5 * (f1_res + f1_nres),
        "f1_res": f1_res,
        "f1_nres": f1_nres,
        "auroc": _auroc(y, proba),
        "brier": float(np.mean((proba[ok] - y[ok]) ** 2)) if ok.any() else None,
        "mae": float(np.mean(np.abs(proba[ok] - y[ok]))) if ok.any() else None,
        "rmse": float(math.sqrt(np.mean((proba[ok] - y[ok]) ** 2))) if ok.any() else None,
        "mean_p_res": mean_p_res,
        "mean_p_nres": mean_p_nres,
        "delta_p": delta_p,
    }


def load_dou_for_cells(cell_ids: list[int]) -> dict[int, dict]:
    """Map cell_id → {dou, smod_code, smod_label} from Postgres."""
    if not cell_ids:
        return {}
    import sys

    sys.path.insert(0, str(ROOT))
    from lib.db import connect  # noqa: WPS433

    out: dict[int, dict] = {}
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT cell_id, dou, smod_code, smod_label
            FROM cell_degree_of_urbanisation
            WHERE cell_id = ANY(%s)
            """,
            (cell_ids,),
        ).fetchall()
    for r in rows:
        out[int(r["cell_id"])] = {
            "dou": r["dou"],
            "smod_code": int(r["smod_code"]) if r["smod_code"] is not None else None,
            "smod_label": r["smod_label"],
        }
    return out


def attach_dou(points: list[dict], dou_map: dict[int, dict]) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = {k: 0 for k in DOU_ORDER}
    counts["unknown"] = 0
    n_ok = 0
    for p in points:
        info = dou_map.get(int(p["id"]))
        if not info or not info.get("dou"):
            p["dou"] = None
            p["smod_label"] = info.get("smod_label") if info else None
            counts["unknown"] += 1
            continue
        p["dou"] = info["dou"]
        p["smod_code"] = info.get("smod_code")
        p["smod_label"] = info.get("smod_label")
        n_ok += 1
        counts[info["dou"]] = counts.get(info["dou"], 0) + 1
    return n_ok, counts


def city_metrics_from_points(points: list[dict]) -> list[dict]:
    by: dict[str, list[dict]] = {}
    for p in points:
        by.setdefault(str(p["city_id"]), []).append(p)
    rows = []
    for city_id, arr in sorted(by.items()):
        sample = _metrics_block(arr)
        assert sample is not None
        by_dou: dict[str, dict] = {}
        for dou in DOU_ORDER:
            sub = [p for p in arr if p.get("dou") == dou]
            block = _metrics_block(sub)
            if block is not None:
                by_dou[dou] = block
        rows.append(
            {
                "city_id": city_id,
                "city_name": arr[0].get("city_name") or city_id,
                "country": arr[0].get("country") or city_id.split("_", 1)[0],
                "n": int(sample["n"]),
                "n_res": int(sample["n_res"]),
                "n_nres": int(sample["n_nres"]),
                "sample": {k: v for k, v in sample.items() if k not in ("n", "n_res", "n_nres")},
                "n_by_dou": {d: by_dou[d]["n"] for d in by_dou},
                "by_dou": by_dou,
            }
        )
    return rows


def aggregate_dou_global(points: list[dict]) -> dict[str, dict]:
    """Overall + per-country sample metrics sliced by DoU."""
    out: dict[str, dict] = {}
    for dou in DOU_ORDER:
        sub = [p for p in points if p.get("dou") == dou]
        block = _metrics_block(sub)
        if block is None:
            continue
        by_country: dict[str, dict] = {}
        countries = sorted({str(p.get("country") or "") for p in sub if p.get("country")})
        for c in countries:
            cb = _metrics_block([p for p in sub if p.get("country") == c])
            if cb is not None:
                by_country[c] = cb
        out[dou] = {
            "label": DOU_UI[dou],
            "overall": block,
            "by_country": by_country,
        }
    return out


def load_india_japan_loro(path: Path, country: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df = df[(df["model"] == "lr64") & (df["country"] == country)]
    out = {}
    for _, r in df.iterrows():
        out[str(r["held_out"])] = {
            "macro_f1": float(r["macro_f1"]),
            "accuracy": float(r["accuracy"]),
            "auroc": float(r["auroc"]) if pd.notna(r.get("auroc")) else None,
            "f1_res": float(r["f1_residential"]) if "f1_residential" in r else None,
            "f1_nres": float(r["f1_nonresidential"]) if "f1_nonresidential" in r else None,
            "n": int(r["n"]),
            "model": "lr64",
            "source": "loro_holdout",
        }
    return out


def load_usa_week4(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    out = {}
    for fold in payload.get("folds") or []:
        region = fold.get("held_out_region")
        city = WEEK4_TO_CITY.get(region)
        if not city:
            continue
        out[city] = {
            "macro_f1": float(fold["macro_f1"]),
            "accuracy": float(fold["accuracy"]),
            "auroc": None,
            "f1_res": None,
            "f1_nres": None,
            "n": int(fold.get("n_test") or 0),
            "model": "week4_baseline_lr",
            "source": "loro_holdout",
            "week4_region": region,
        }
    return out


def aggregate_country(cities: list[dict], key: str = "sample") -> list[dict]:
    by: dict[str, list[dict]] = {}
    for c in cities:
        by.setdefault(c["country"], []).append(c)
    rows = []
    for country, arr in sorted(by.items()):
        def mean_of(field: str):
            vals = [
                c[key].get(field)
                for c in arr
                if c.get(key) and c[key].get(field) is not None
            ]
            return float(np.mean(vals)) if vals else None

        rows.append(
            {
                "country": country,
                "n_cities": len(arr),
                "n": int(sum(c["n"] for c in arr)),
                key: {
                    "accuracy": mean_of("accuracy"),
                    "macro_f1": mean_of("macro_f1"),
                    "auroc": mean_of("auroc"),
                    "brier": mean_of("brier"),
                    "mae": mean_of("mae"),
                    "rmse": mean_of("rmse"),
                    "delta_p": mean_of("delta_p"),
                },
            }
        )
    return rows


def _model_key(model: str, k: int | float | None) -> str:
    if model == "lr64":
        return "lr64"
    if k is None or (isinstance(k, float) and math.isnan(k)):
        return str(model)
    return f"{model}{int(k)}"


def load_cross_country(country_csv: Path, city_csv: Path | None = None) -> dict | None:
    """Load train-country → test-country (+ optional city→city) transfer metrics."""
    if not country_csv.exists():
        return None
    df = pd.read_csv(country_csv)
    needed = {"source_country", "target_country", "macro_f1", "model"}
    if not needed.issubset(df.columns):
        return None

    df = df.copy()
    df["model_key"] = [
        _model_key(str(m), k) for m, k in zip(df["model"], df["k"] if "k" in df.columns else [None] * len(df))
    ]

    # Prefer a stable model order
    preferred = ["lr64", "pls9", "pls2", "pca9", "pca2"]
    models = [m for m in preferred if m in set(df["model_key"])]
    models += sorted(set(df["model_key"]) - set(models))

    countries = sorted(set(df["source_country"].astype(str)) | set(df["target_country"].astype(str)))
    # Prefer USA / JPN / IND order when present
    order = [c for c in ["USA", "JPN", "IND"] if c in countries] + [
        c for c in countries if c not in {"USA", "JPN", "IND"}
    ]

    by_model: dict[str, dict] = {}
    for mk in models:
        sub = df[df["model_key"] == mk]
        matrix: dict[str, dict[str, float | None]] = {s: {t: None for t in order} for s in order}
        pairs: list[dict] = []
        within: dict[str, float] = {}
        for _, r in sub.iterrows():
            s = str(r["source_country"])
            t = str(r["target_country"])
            f1 = float(r["macro_f1"]) if pd.notna(r["macro_f1"]) else None
            if s in matrix and t in matrix[s]:
                matrix[s][t] = f1
            if s == t and f1 is not None:
                within[s] = f1
            row = {
                "source": s,
                "target": t,
                "macro_f1": f1,
                "accuracy": float(r["accuracy"]) if "accuracy" in r and pd.notna(r["accuracy"]) else None,
                "auroc": float(r["auroc"]) if "auroc" in r and pd.notna(r["auroc"]) else None,
                "f1_res": float(r["f1_residential"])
                if "f1_residential" in r and pd.notna(r["f1_residential"])
                else None,
                "f1_nres": float(r["f1_nonresidential"])
                if "f1_nonresidential" in r and pd.notna(r["f1_nonresidential"])
                else None,
                "n": int(r["n"]) if "n" in r and pd.notna(r["n"]) else None,
                "protocol": str(r["protocol"]) if "protocol" in r and pd.notna(r["protocol"]) else None,
                "cross": s != t,
            }
            pairs.append(row)
        for p in pairs:
            w = within.get(p["source"])
            p["retention"] = (
                None
                if p["macro_f1"] is None or not w
                else float(p["macro_f1"] / w)
            )
        by_model[mk] = {
            "countries": order,
            "matrix_f1": matrix,
            "pairs": pairs,
            "within_f1": within,
        }

    city_cross: list[dict] = []
    if city_csv is not None and city_csv.exists():
        cdf = pd.read_csv(city_csv)
        if {"source", "target", "macro_f1", "model", "source_country", "target_country"}.issubset(cdf.columns):
            cdf = cdf.copy()
            cdf["model_key"] = [
                _model_key(str(m), k)
                for m, k in zip(cdf["model"], cdf["k"] if "k" in cdf.columns else [None] * len(cdf))
            ]
            # Keep lr64 cross-country city pairs only (compact for the dashboard)
            sub = cdf[(cdf["model_key"] == "lr64") & (cdf["source_country"] != cdf["target_country"])]
            for _, r in sub.iterrows():
                city_cross.append(
                    {
                        "source": str(r["source"]),
                        "target": str(r["target"]),
                        "source_country": str(r["source_country"]),
                        "target_country": str(r["target_country"]),
                        "macro_f1": float(r["macro_f1"]) if pd.notna(r["macro_f1"]) else None,
                        "accuracy": float(r["accuracy"]) if "accuracy" in r and pd.notna(r["accuracy"]) else None,
                        "auroc": float(r["auroc"]) if "auroc" in r and pd.notna(r["auroc"]) else None,
                        "n": int(r["n"]) if "n" in r and pd.notna(r["n"]) else None,
                    }
                )

    # Directional country means from city pairs (lr64) for richer city-filter views
    city_dir_means: list[dict] = []
    if city_cross:
        from collections import defaultdict

        buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
        for r in city_cross:
            if r["macro_f1"] is None:
                continue
            buckets[(r["source_country"], r["target_country"])].append(r["macro_f1"])
        for (s, t), vals in sorted(buckets.items()):
            city_dir_means.append(
                {
                    "source": s,
                    "target": t,
                    "macro_f1": float(np.mean(vals)),
                    "n_pairs": len(vals),
                }
            )

    return {
        "default_model": "lr64" if "lr64" in by_model else (models[0] if models else None),
        "models": models,
        "by_model": by_model,
        "city_cross_lr64": city_cross,
        "city_direction_means_lr64": city_dir_means,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=Path, default=None,
                    help="Default: dashboard_points.json if present, else points.json")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument(
        "--india-loro",
        type=Path,
        default=ROOT
        / "runs"
        / "india_loro_20261005_202600"
        / "artifacts"
        / "loro_IND_metrics.csv",
    )
    ap.add_argument(
        "--japan-loro",
        type=Path,
        default=ROOT
        / "runs"
        / "japan_loro_20261005_225832"
        / "artifacts"
        / "loro_JPN_metrics.csv",
    )
    ap.add_argument(
        "--usa-metrics",
        type=Path,
        default=DATA / "week4_baseline_lr_metrics.json",
    )
    ap.add_argument(
        "--intl-transfer-dir",
        type=Path,
        default=ROOT / "runs" / "intl_transfer_20261005_142709" / "artifacts",
        help="intl_transfer artifacts dir with country_transfer_metrics.csv",
    )
    args = ap.parse_args()
    if args.points is None:
        args.points = default_points_path()

    payload = json.loads(args.points.read_text())
    points = payload["points"]
    # Prefer DoU already on the sample (dashboard_points.json); else join from DB.
    if points and all(p.get("dou") in DOU_ORDER for p in points):
        n_dou = len(points)
        dou_counts = {k: 0 for k in DOU_ORDER}
        dou_counts["unknown"] = 0
        for p in points:
            dou_counts[p["dou"]] = dou_counts.get(p["dou"], 0) + 1
        print(f"using prelabeled DoU on points: {dou_counts}", flush=True)
    else:
        cell_ids = [int(p["id"]) for p in points]
        dou_map = load_dou_for_cells(cell_ids)
        n_dou, dou_counts = attach_dou(points, dou_map)

    cities = city_metrics_from_points(points)

    holdout = {}
    holdout.update(load_india_japan_loro(args.india_loro, "IND"))
    holdout.update(load_india_japan_loro(args.japan_loro, "JPN"))
    holdout.update(load_usa_week4(args.usa_metrics))

    for c in cities:
        h = holdout.get(c["city_id"])
        if h:
            c["holdout"] = h

    # country aggregates for sample + holdout (holdout only where present)
    countries_sample = aggregate_country(cities, "sample")
    holdout_cities = [
        {
            **c,
            "holdout_as_sample": c["holdout"],
            "sample": c["holdout"],
        }
        for c in cities
        if c.get("holdout")
    ]
    countries_holdout = aggregate_country(holdout_cities, "sample")
    for row in countries_holdout:
        row["holdout"] = row.pop("sample")

    # Country aggregates per DoU (viz sample only — holdout CSVs are not DoU-stratified)
    countries_by_dou: dict[str, list] = {}
    for dou in DOU_ORDER:
        sliced = []
        for c in cities:
            block = (c.get("by_dou") or {}).get(dou)
            if not block:
                continue
            metrics = {k: v for k, v in block.items() if k not in ("n", "n_res", "n_nres")}
            sliced.append(
                {
                    "city_id": c["city_id"],
                    "city_name": c["city_name"],
                    "country": c["country"],
                    "n": block["n"],
                    "sample": metrics,
                }
            )
        countries_by_dou[dou] = aggregate_country(sliced, "sample") if sliced else []

    dou_summary = aggregate_dou_global(points)

    xfer_country = args.intl_transfer_dir / "country_transfer_metrics.csv"
    xfer_city = args.intl_transfer_dir / "city_transfer_metrics.csv"
    cross_country = load_cross_country(xfer_country, xfer_city)

    out = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "n_cities": len(cities),
        "countries": sorted({c["country"] for c in cities}),
        "dou": {
            "order": list(DOU_ORDER),
            "labels": dict(DOU_UI),
            "note": (
                "GHS-SMOD Degree of Urbanisation rollup on viz sample cells; "
                "peri_urban shown as suburban in the UI. Holdout metrics are not DoU-sliced."
            ),
            "n_labeled": n_dou,
            "n_points": len(points),
            "counts": dou_counts,
        },
        "sources": {
            "sample": "viz points.json LORO-scored cells (threshold 0.5)",
            "holdout": "official leave-one-city/metro-out metrics where available",
            "dou": "cell_degree_of_urbanisation (GHS-SMOD R2023A) joined by cell_id",
            "cross_country": (
                "train all source country → test target country "
                "(intl_transfer run; diagonal = within-country 20% holdout)"
            ),
            "india_loro": str(args.india_loro) if args.india_loro.exists() else None,
            "japan_loro": str(args.japan_loro) if args.japan_loro.exists() else None,
            "usa_metrics": str(args.usa_metrics) if args.usa_metrics.exists() else None,
            "intl_transfer": str(args.intl_transfer_dir) if args.intl_transfer_dir.exists() else None,
        },
        "cities": cities,
        "countries_summary": {
            "sample": countries_sample,
            "holdout": countries_holdout,
            "by_dou": countries_by_dou,
        },
        "dou_summary": dou_summary,
        "cross_country": cross_country,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    n_xfer = 0
    if cross_country and cross_country.get("by_model"):
        mk = cross_country.get("default_model")
        n_xfer = len(cross_country["by_model"].get(mk, {}).get("pairs", []))
    print(
        f"wrote {args.out} cities={len(cities)} "
        f"holdout={sum(1 for c in cities if c.get('holdout'))} "
        f"xfer_pairs={n_xfer} dou={dou_counts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
