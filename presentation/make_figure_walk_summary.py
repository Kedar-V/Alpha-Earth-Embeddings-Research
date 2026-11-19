#!/usr/bin/env python3
"""Country- and city-level summary of the embedding-globe boundary walks (85 completed walks).

Per walk (12 steps, nearest real 10 m cell at each step):
  agreement = share of steps where the cell's true label matches the walk model's call (P(res) >= 0.5)
  clean     = the true label flips exactly once along the walk
  strict    = endpoints met the 0.9 / 0.1 confidence cut-offs (otherwise the app relaxed them)
Source: week4/international_geo_generalization/viz/usa_embedding_globe/data/boundary_walks.json
Run:  .venv/bin/python presentation/make_figure_walk_summary.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "week4/international_geo_generalization/viz/usa_embedding_globe/data/boundary_walks.json"
OUT = Path(__file__).resolve().parent / "figures" / "walks_country_city.png"
NAVY, SLATE, BLUE, LBLUE, GREY, LIGHT = "#0B2238", "#4B5A6E", "#1F6FA8", "#4AA0DB", "#8A93A0", "#E4E7EB"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": NAVY, "axes.edgecolor": "#B8BEC6",
                     "xtick.color": SLATE, "ytick.color": NAVY})

rows = []
for w in json.load(open(SRC))["cities"]:
    base = dict(country=w["country"], city=w["city_name"], dou=w["dou"], status=w["status"])
    if w["status"] != "ok":
        rows.append(base); continue
    fr = w["frames"]
    gt = [f["gt"] for f in fr]; p = [f["proba_res"] for f in fr]
    base.update(agree=np.mean([(q >= 0.5) == (g == 1) for g, q in zip(gt, p)]),
                clean=sum(a != b for a, b in zip(gt, gt[1:])) == 1,
                strict=w["endpoint_selection"]["endpoint_mode"].startswith("correct_hi"))
    rows.append(base)
t = pd.DataFrame(rows)
ok = t[t.status == "ok"]
NAMES = {"USA": "USA", "JPN": "Japan", "IND": "India"}
CCOL = {"USA": NAVY, "JPN": GREY, "IND": BLUE}
countries = ["USA", "JPN", "IND"]
summ = ok.groupby("country")[["agree", "clean", "strict"]].mean()

W, H = 11.9, 4.15
fig = plt.figure(figsize=(W, H), dpi=220)
fig.text(0.008, 0.975, "Walks cross cleanly in the US and Japan; in India they rarely do",
         fontsize=13, fontweight="bold", va="top")
fig.text(0.008, 0.905, f"{len(ok)} walks in {ok.city.nunique()} cities · match = % of the 12 steps where the nearest real cell's label agrees with the model",
         fontsize=10, color=SLATE, va="top")

# ---- left: country level, three metrics as grouped horizontal bars (0-100%)
ax = fig.add_axes([0.125, 0.1, 0.2, 0.6])
metrics = [("agree", "Steps where label\nmatches the model"), ("clean", "Walks with one\nclean label flip"),
           ("strict", "Walks reaching the\n0.9 / 0.1 cut-offs")]
bh = 0.24
for m, (col, lab) in enumerate(metrics):
    for k, c in enumerate(countries):
        y = m + (k - 1) * bh
        v = summ.loc[c, col]
        ax.barh(y, v, height=bh * 0.9, color=CCOL[c])
        ax.text(v + 0.02, y, f"{v:.0%}", va="center", fontsize=9.5, color=CCOL[c], fontweight="bold")
ax.set_yticks(range(3)); ax.set_yticklabels([m[1] for m in metrics], fontsize=9)
ax.set_ylim(2.6, -0.6); ax.set_xlim(0, 1.15); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "50%", "100%"], fontsize=9)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.text(0.008, 0.8, "By country", fontsize=11.5, fontweight="bold", va="center")
for k, c in enumerate(countries):
    fig.text(0.125 + k * 0.07, 0.8, "■ " + NAMES[c], color=CCOL[c], fontsize=10, fontweight="bold", va="center")

# ---- right: city level, one grid per country (rows = cities sorted by mean match, cols = urbanisation)
cmap = LinearSegmentedColormap.from_list("m", ["#F3F6F8", "#CFE5E8", LBLUE, BLUE, NAVY])
dous = [("urban", "Urban"), ("peri_urban", "Suburban"), ("rural", "Rural")]
for k, c in enumerate(countries):
    sub = t[t.country == c]
    order = sub[sub.status == "ok"].groupby("city").agree.mean().sort_values(ascending=False).index.tolist()
    order += [x for x in sub.city.unique() if x not in order]
    M = np.full((len(order), 3), np.nan)
    for i, city in enumerate(order):
        for j, (d, _) in enumerate(dous):
            r = sub[(sub.city == city) & (sub.dou == d)]
            if len(r) and r.status.iloc[0] == "ok":
                M[i, j] = r.agree.iloc[0]
    axh = fig.add_axes([0.45 + k * 0.195, 0.06, 0.115, 0.66])
    axh.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0.4, vmax=1.0, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(3):
            v = M[i, j]
            if np.isnan(v):
                axh.text(j, i, "–", ha="center", va="center", fontsize=9, color=SLATE)
            else:
                axh.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=8.5,
                         color="white" if v >= 0.8 else NAVY, fontweight="bold")
    axh.set_xticks(range(3)); axh.set_xticklabels(["Urban", "Suburb", "Rural"], fontsize=7.5)
    axh.xaxis.set_ticks_position("top")
    axh.set_yticks(range(len(order))); axh.set_yticklabels(order, fontsize=9)
    axh.tick_params(length=0)
    for s in axh.spines.values():
        s.set_visible(False)
    axh.set_title(f"{NAMES[c]}: match %", loc="left", fontsize=11, fontweight="bold",
                  color=CCOL[c] if c != "JPN" else NAVY, pad=16)
fig.text(0.45, 0.005, "– = walk skipped (too few confident cells)", fontsize=9, color=SLATE, va="bottom")
fig.savefig(OUT, facecolor="white")
print("wrote", OUT)
print(summ.round(2))
