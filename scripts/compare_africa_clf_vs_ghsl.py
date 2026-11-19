#!/usr/bin/env python3
"""Africa classification: AE vs GHS-BUILT-S soft-argmax F1 on city-LORO cells.

Default input: scored.parquet from africa_loro_vs_ghsl (EGY+NGA city-LORO),
which already has AE XGB clf / Ridge / XGB-reg soft preds and BUILT-S fractions.

Metrics: RES F1, NRES F1, macro F1 (higher is better).
Also optionally scores the 5-city intl panel (Ridge/PLS soft-argmax vs BUILT-S).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    os.replace(tmp, path)


def write_progress(run_dir: Path, **kwargs: Any) -> None:
    path = run_dir / "progress.json"
    prev: dict[str, Any] = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text())
        except Exception:
            prev = {}
    prev.update(kwargs)
    prev["updated_at"] = utc_now()
    prev["last_progress_at"] = utc_now()
    if "_t0" in prev:
        elapsed = max(1e-6, time.time() - float(prev["_t0"]))
        prev["elapsed_seconds"] = round(elapsed, 1)
        if "completed" in prev and "total" in prev and int(prev["total"]) > 0:
            done, total = int(prev["completed"]), int(prev["total"])
            rate = done / elapsed
            prev["rate_per_second"] = round(rate, 4)
            prev["percent_complete"] = round(100.0 * done / total, 2)
            prev["eta_seconds"] = round(max(0, total - done) / rate, 1) if rate > 0 else None
    atomic_json(path, prev)


def log(run_dir: Path, msg: str) -> None:
    line = f"[{utc_now()}] {msg}"
    print(line, flush=True)
    with (run_dir / "job.log").open("a") as f:
        f.write(line + "\n")


def clf_metrics(yt: np.ndarray, yp: np.ndarray) -> dict[str, float]:
    m = np.isfinite(yt.astype(float)) & np.isfinite(yp.astype(float))
    if m.sum() == 0:
        return {
            "n": 0,
            "res_f1": float("nan"),
            "nres_f1": float("nan"),
            "macro_f1": float("nan"),
            "accuracy": float("nan"),
            "res_rate": float("nan"),
        }
    y_t = yt[m].astype(np.int64)
    y_p = yp[m].astype(np.int64)
    return {
        "n": int(m.sum()),
        "res_f1": float(f1_score(y_t, y_p, pos_label=1, zero_division=0)),
        "nres_f1": float(f1_score(y_t, y_p, pos_label=0, zero_division=0)),
        "macro_f1": float(f1_score(y_t, y_p, average="macro", zero_division=0)),
        "accuracy": float((y_t == y_p).mean()),
        "res_rate": float(y_t.mean()),
    }


def soft_argmax_res(frac_res: np.ndarray, frac_nres: np.ndarray) -> np.ndarray:
    return (frac_res >= frac_nres).astype(np.int64)


def _score_group(payload: tuple) -> dict[str, Any]:
    level, keys, key_vals, yt, preds = payload
    recs = []
    for model, yp in preds.items():
        m = clf_metrics(yt, yp)
        rec = {"level": level, "model": model, **m}
        for k, v in zip(keys, key_vals):
            rec[k] = v
        recs.append(rec)
    return {"recs": recs}


def score_frame(
    df: pd.DataFrame,
    models: dict[str, tuple[str, str] | str],
    group_specs: list[tuple[str, list[str]]],
    n_workers: int,
) -> pd.DataFrame:
    """models: name -> ('col_res','col_nres') for soft-argmax, or 'y_pred' col for hard."""
    tasks = []
    for level, keys in group_specs:
        for key, sub in df.groupby(keys, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            yt = sub["y"].to_numpy()
            preds: dict[str, np.ndarray] = {}
            for model, spec in models.items():
                if isinstance(spec, str):
                    preds[model] = sub[spec].to_numpy().astype(np.int64)
                else:
                    c_res, c_nres = spec
                    preds[model] = soft_argmax_res(
                        sub[c_res].to_numpy(dtype=float),
                        sub[c_nres].to_numpy(dtype=float),
                    )
            tasks.append((level, keys, key, yt, preds))

    rows: list[dict[str, Any]] = []
    workers = max(1, min(n_workers, len(tasks)))
    if workers == 1 or len(tasks) < 4:
        for t in tasks:
            rows.extend(_score_group(t)["recs"])
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_score_group, t) for t in tasks]
            for fut in as_completed(futs):
                rows.extend(fut.result()["recs"])
    return pd.DataFrame(rows)


def prepare_loro_scored(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "city_id" not in df.columns and "region" in df.columns:
        df = df.rename(columns={"region": "city_id"})
    if "y" not in df.columns:
        df["y"] = (df["frac_res_full"] > 0).astype(np.int64)  # pure cells
    # selective BUILT-S
    if "built_s_valid" in df.columns:
        n0 = len(df)
        df = df.loc[df["built_s_valid"]].copy()
        if len(df) < n0:
            pass
    need = [
        "y",
        "y_pred",
        "pred_frac_res",
        "pred_frac_nres",
        "ridge_frac_res",
        "ridge_frac_nres",
        "bs_frac_res",
        "bs_frac_nres",
        "country",
        "city_id",
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"scored missing columns: {missing}")
    return df


def prepare_intl_scored(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "y" not in df.columns:
        # pure 1:1 balanced: stratum or binary from fracs
        if "stratum" in df.columns:
            df["y"] = (df["stratum"].astype(str).str.lower().isin(["res", "residential"])).astype(np.int64)
        else:
            df["y"] = (df["frac_res_full"] >= df["frac_nres_full"]).astype(np.int64)
    if "built_s_valid" in df.columns:
        df = df.loc[df["built_s_valid"]].copy()
    need = [
        "y",
        "ridge_frac_res",
        "ridge_frac_nres",
        "pls_frac_res",
        "pls_frac_nres",
        "bs_frac_res",
        "bs_frac_nres",
        "country",
        "city_id",
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"intl scored missing: {missing}")
    return df


def write_figure_country(metrics: pd.DataFrame, out: Path, title: str, model_order: list[tuple[str, str]]) -> None:
    country = metrics[metrics.level == "country"].copy()
    if country.empty:
        return
    countries = sorted(country.country.unique())
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), sharey=True)
    fields = [("res_f1", "RES F1"), ("nres_f1", "NRES F1"), ("macro_f1", "Macro F1")]
    x = np.arange(len(countries))
    w = 0.8 / max(len(model_order), 1)
    colors = ["#C2691A", "#1F6FA8", "#8E44AD", "#4AA0DB", "#0F766E"]
    for ax, (field, lab) in zip(axes, fields):
        for i, (model, name) in enumerate(model_order):
            vals = []
            for c in countries:
                s = country[(country.country == c) & (country.model == model)]
                vals.append(float(s[field].iloc[0]) if len(s) else np.nan)
            ax.bar(x + (i - (len(model_order) - 1) / 2) * w, vals, w, label=name, color=colors[i % len(colors)])
        ax.set_xticks(x)
        ax.set_xticklabels(countries)
        ax.set_ylim(0, 1.05)
        ax.set_title(lab)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("F1 (higher is better)")
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_results_md(
    run_dir: Path,
    loro_metrics: pd.DataFrame | None,
    intl_metrics: pd.DataFrame | None,
    loro_src: str,
    intl_src: str | None,
) -> None:
    lines = [
        "# Africa classification · AE vs GHS-BUILT-S",
        "",
        "Binary RES vs NRES. GHSL = soft-argmax of GHS-BUILT-S fractions at AE centres.",
        "",
    ]
    if loro_metrics is not None:
        lines += [
            "## City-LORO panel (EGY + NGA)",
            "",
            f"Source: `{loro_src}`",
            "",
            "| Country | n | AE XGB clf | AE XGB soft | AE Ridge soft | GHS-BUILT-S | winner (macro) |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        c = loro_metrics[loro_metrics.level == "country"]
        for country in sorted(c.country.unique()):
            g = c[c.country == country]
            def v(model: str) -> float:
                s = g[g.model == model]
                return float(s.macro_f1.iloc[0]) if len(s) else float("nan")

            def n_of(model: str) -> int:
                s = g[g.model == model]
                return int(s.n.iloc[0]) if len(s) else 0

            scores = {
                "AE XGB clf": v("ae_xgb_clf"),
                "AE XGB soft": v("ae_xgb_soft"),
                "AE Ridge soft": v("ae_ridge_soft"),
                "GHS-BUILT-S": v("ghsl_built_s"),
            }
            win = max(scores.items(), key=lambda t: (t[1] if t[1] == t[1] else -1))[0]
            lines.append(
                f"| {country} | {n_of('ae_xgb_clf'):,} | {scores['AE XGB clf']:.3f} | "
                f"{scores['AE XGB soft']:.3f} | {scores['AE Ridge soft']:.3f} | "
                f"{scores['GHS-BUILT-S']:.3f} | {win} |"
            )
        city = loro_metrics[loro_metrics.level == "city"]
        lines += [
            "",
            "### Per-city macro-F1",
            "",
            "| Country | City | AE XGB clf | AE Ridge soft | GHS-BUILT-S |",
            "|---|---|---:|---:|---:|",
        ]
        for (ctry, cid), g in city.groupby(["country", "city_id"]):
            def v(model: str) -> float:
                s = g[g.model == model]
                return float(s.macro_f1.iloc[0]) if len(s) else float("nan")

            lines.append(
                f"| {ctry} | {cid} | {v('ae_xgb_clf'):.3f} | {v('ae_ridge_soft'):.3f} | {v('ghsl_built_s'):.3f} |"
            )
        lines += ["", "![loro](figures/loro_country_f1.png)", ""]

    if intl_metrics is not None:
        lines += [
            "## Five-city intl panel (one city / country)",
            "",
            f"Source: `{intl_src}`",
            "",
            "| Country | City | n | AE Ridge soft | AE PLS soft | GHS-BUILT-S | winner |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        city = intl_metrics[intl_metrics.level == "city"]
        for _, r in city.sort_values(["country", "city_id"]).iterrows():
            if r.model != "ae_ridge_soft":
                continue
            cid = r.city_id
            g = city[city.city_id == cid]

            def v(model: str) -> float:
                s = g[g.model == model]
                return float(s.macro_f1.iloc[0]) if len(s) else float("nan")

            scores = {
                "AE Ridge": v("ae_ridge_soft"),
                "AE PLS": v("ae_pls_soft"),
                "GHS-BUILT-S": v("ghsl_built_s"),
            }
            win = max(scores.items(), key=lambda t: (t[1] if t[1] == t[1] else -1))[0]
            n = int(g[g.model == "ae_ridge_soft"].n.iloc[0])
            lines.append(
                f"| {r.country} | {cid} | {n:,} | {scores['AE Ridge']:.3f} | "
                f"{scores['AE PLS']:.3f} | {scores['GHS-BUILT-S']:.3f} | {win} |"
            )
        lines += ["", "![intl](figures/intl_country_f1.png)", ""]

    (run_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scored",
        type=Path,
        default=ROOT / "outputs" / "africa_loro_vs_ghsl" / "20261113_125810" / "artifacts" / "scored.parquet",
        help="EGY+NGA city-LORO scored parquet with AE OOF + BUILT-S",
    )
    ap.add_argument(
        "--intl-scored",
        type=Path,
        default=ROOT
        / "outputs"
        / "ghsl_built_s_vs_ae_intl"
        / "africa_20261113_020114"
        / "artifacts"
        / "scored.parquet",
        help="5-city intl scored parquet (Ridge/PLS + BUILT-S); empty to skip",
    )
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--skip-intl", action="store_true")
    ap.add_argument("--workers", type=int, default=0, help="CPU workers (0 = all cores)")
    args = ap.parse_args()

    n_workers = args.workers or (os.cpu_count() or 4)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (ROOT / "outputs" / "africa_clf_vs_ghsl" / stamp)
    for sub in ("artifacts", "figures", "checkpoints", "failures", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    (run_dir / "job.pid").write_text(str(os.getpid()) + "\n")
    (run_dir / "environment.txt").write_text(
        f"python={sys.executable}\ncpus={os.cpu_count()}\nworkers={n_workers}\n"
        f"hostname={os.uname().nodename}\n"
    )

    t0 = time.time()
    total_stages = 4 if not args.skip_intl else 3
    write_progress(
        run_dir,
        status="running",
        stage="load_loro",
        started_at=utc_now(),
        _t0=t0,
        completed=0,
        total=total_stages,
        failed=0,
        output_location=str(run_dir),
        resource_allocation={"cpus": os.cpu_count(), "workers": n_workers, "gpus": 0},
    )
    log(run_dir, f"Africa clf vs GHS-BUILT-S · run_dir={run_dir}")
    log(run_dir, f"scored={args.scored} workers={n_workers}")

    if not args.scored.exists():
        raise SystemExit(f"missing scored parquet: {args.scored}")

    loro_df = prepare_loro_scored(args.scored)
    log(
        run_dir,
        f"LORO panel n={len(loro_df):,} countries={sorted(loro_df.country.unique())} "
        f"cities={loro_df.city_id.nunique()}",
    )
    write_progress(run_dir, stage="score_loro", completed=1)

    loro_models = {
        "ae_xgb_clf": "y_pred",
        "ae_xgb_soft": ("pred_frac_res", "pred_frac_nres"),
        "ae_ridge_soft": ("ridge_frac_res", "ridge_frac_nres"),
        "ghsl_built_s": ("bs_frac_res", "bs_frac_nres"),
    }
    # pooled + country + city
    pooled = []
    for model, spec in loro_models.items():
        if isinstance(spec, str):
            yp = loro_df[spec].to_numpy().astype(np.int64)
        else:
            yp = soft_argmax_res(
                loro_df[spec[0]].to_numpy(dtype=float),
                loro_df[spec[1]].to_numpy(dtype=float),
            )
        m = clf_metrics(loro_df["y"].to_numpy(), yp)
        pooled.append({"level": "pooled", "model": model, "country": "ALL", **m})

    loro_metrics = score_frame(
        loro_df,
        loro_models,
        [("country", ["country"]), ("city", ["country", "city_id"])],
        n_workers=n_workers,
    )
    loro_metrics = pd.concat([pd.DataFrame(pooled), loro_metrics], ignore_index=True)
    loro_path = run_dir / "artifacts" / "loro_clf_metrics.csv"
    loro_metrics.to_csv(loro_path, index=False)
    log(run_dir, f"wrote {loro_path}")

    # checkpoint intermediate
    atomic_json(
        run_dir / "checkpoints" / "loro_pooled.json",
        {r["model"]: r for r in pooled},
    )
    write_progress(run_dir, stage="figure_loro", completed=2)

    write_figure_country(
        loro_metrics,
        run_dir / "figures" / "loro_country_f1.png",
        "Africa city-LORO · classification F1 vs GHS-BUILT-S",
        [
            ("ghsl_built_s", "GHS-BUILT-S"),
            ("ae_ridge_soft", "AE Ridge soft"),
            ("ae_xgb_soft", "AE XGB soft"),
            ("ae_xgb_clf", "AE XGB clf"),
        ],
    )

    intl_metrics = None
    intl_src = None
    if not args.skip_intl and args.intl_scored and args.intl_scored.exists():
        write_progress(run_dir, stage="score_intl", completed=2)
        intl_src = str(args.intl_scored)
        intl_df = prepare_intl_scored(args.intl_scored)
        log(
            run_dir,
            f"intl panel n={len(intl_df):,} countries={sorted(intl_df.country.unique())}",
        )
        intl_models = {
            "ae_ridge_soft": ("ridge_frac_res", "ridge_frac_nres"),
            "ae_pls_soft": ("pls_frac_res", "pls_frac_nres"),
            "ghsl_built_s": ("bs_frac_res", "bs_frac_nres"),
        }
        intl_pooled = []
        for model, spec in intl_models.items():
            yp = soft_argmax_res(
                intl_df[spec[0]].to_numpy(dtype=float),
                intl_df[spec[1]].to_numpy(dtype=float),
            )
            m = clf_metrics(intl_df["y"].to_numpy(), yp)
            intl_pooled.append({"level": "pooled", "model": model, "country": "ALL", **m})
        intl_metrics = score_frame(
            intl_df,
            intl_models,
            [("country", ["country"]), ("city", ["country", "city_id"])],
            n_workers=n_workers,
        )
        intl_metrics = pd.concat([pd.DataFrame(intl_pooled), intl_metrics], ignore_index=True)
        intl_path = run_dir / "artifacts" / "intl_clf_metrics.csv"
        intl_metrics.to_csv(intl_path, index=False)
        atomic_json(
            run_dir / "checkpoints" / "intl_pooled.json",
            {r["model"]: r for r in intl_pooled},
        )
        write_figure_country(
            intl_metrics,
            run_dir / "figures" / "intl_country_f1.png",
            "Africa 5-city · soft-argmax F1 vs GHS-BUILT-S",
            [
                ("ghsl_built_s", "GHS-BUILT-S"),
                ("ae_ridge_soft", "AE Ridge soft"),
                ("ae_pls_soft", "AE PLS soft"),
            ],
        )
        log(run_dir, f"wrote {intl_path}")
        write_progress(run_dir, stage="write_results", completed=3)
    else:
        write_progress(run_dir, stage="write_results", completed=2)
        if not args.skip_intl:
            log(run_dir, f"skip intl (missing {args.intl_scored})")

    write_results_md(run_dir, loro_metrics, intl_metrics, str(args.scored), intl_src)

    # summary
    def pooled_map(df: pd.DataFrame) -> dict[str, Any]:
        out = {}
        for _, r in df[df.level == "pooled"].iterrows():
            out[r.model] = {
                "n": int(r.n),
                "macro_f1": float(r.macro_f1),
                "res_f1": float(r.res_f1),
                "nres_f1": float(r.nres_f1),
            }
        return out

    country_winners = {}
    for country, g in loro_metrics[loro_metrics.level == "country"].groupby("country"):
        best = g.loc[g.macro_f1.idxmax()]
        country_winners[country] = {
            "winner": str(best.model),
            "macro_f1": float(best.macro_f1),
            "ae_xgb_clf": float(g.loc[g.model == "ae_xgb_clf", "macro_f1"].iloc[0]),
            "ghsl_built_s": float(g.loc[g.model == "ghsl_built_s", "macro_f1"].iloc[0]),
        }

    summary = {
        "status": "completed",
        "task": "africa_classification_ae_vs_ghs_built_s",
        "protocol": "soft-argmax RES vs NRES; AE XGB clf uses dedicated binary head",
        "ghsl_product": "GHS-BUILT-S (soft)",
        "loro_scored": str(args.scored),
        "intl_scored": intl_src,
        "n_loro": int(len(loro_df)),
        "loro_pooled": pooled_map(loro_metrics),
        "loro_country_winners": country_winners,
        "intl_pooled": pooled_map(intl_metrics) if intl_metrics is not None else None,
        "runtime_seconds": round(time.time() - t0, 1),
        "workers": n_workers,
        "run_dir": str(run_dir),
        "results_md": str(run_dir / "RESULTS.md"),
        "figures": {
            "loro": str(run_dir / "figures" / "loro_country_f1.png"),
            "intl": str(run_dir / "figures" / "intl_country_f1.png")
            if intl_metrics is not None
            else None,
        },
    }
    atomic_json(run_dir / "summary.json", summary)
    write_progress(
        run_dir,
        status="completed",
        stage="done",
        completed=total_stages,
        total=total_stages,
        percent_complete=100.0,
        success_count=1,
        failure_count=0,
    )
    (run_dir / "DONE").write_text(utc_now() + "\n")
    log(run_dir, f"DONE · {run_dir}")

    # print headline
    lp = summary["loro_pooled"]
    log(
        run_dir,
        "LORO pooled macro-F1: "
        + " · ".join(f"{k}={lp[k]['macro_f1']:.3f}" for k in lp),
    )
    if summary["intl_pooled"]:
        ip = summary["intl_pooled"]
        log(
            run_dir,
            "intl pooled macro-F1: "
            + " · ".join(f"{k}={ip[k]['macro_f1']:.3f}" for k in ip),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
