#!/usr/bin/env python3
"""Boundary-walk figure for the deck: Phoenix vs New York, urban walks.

Uses the embedding-globe app's walk data and image chips unchanged
(week4/international_geo_generalization/viz/usa_embedding_globe/data/):
  - boundary_walks.json: 12-step straight line in LORO PLS space from the residential centroid
    (P(res) > 0.9) to the non-residential centroid (P(res) < 0.1); proba_res from a PLS-3 logistic model
  - each frame's nearest real 10 m cell (PLS-32 cosine NN) and its Esri context chip
Run:  .venv/bin/python presentation/make_figure_walks.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "week4/international_geo_generalization/viz/usa_embedding_globe/data"
OUT = Path(__file__).resolve().parent / "figures" / "walks_phoenix_nyc.png"
NAVY, SLATE, TEAL, AMBER = "#0B2238", "#4B5A6E", "#2B9DB0", "#C27A2C"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": NAVY})

walks = {w["walk_key"]: w for w in json.load(open(APP / "boundary_walks.json"))["cities"]}
rows = [("USA_PHOENIX::urban", "Phoenix, urban: the true label flips once, right after the model crosses 50%"),
        ("USA_NEW_YORK::urban", "New York, urban: the label flips early and back again; the boundary is blurred")]

W, H = 12.1, 4.6
fig = plt.figure(figsize=(W, H), dpi=220)
n = 12
left, right, gap = 0.012, 0.988, 0.004
cw = (right - left - gap * (n - 1)) / n
chip_h = cw * W / H                       # square chips in figure fraction
for r, (key, title) in enumerate(rows):
    fr = walks[key]["frames"]
    assert walks[key]["endpoint_selection"]["endpoint_mode"] == "correct_hi_0.9_0.1"
    top = 0.94 - r * 0.49
    fig.text(left, top, title, fontsize=12, fontweight="bold", va="bottom", color=NAVY)
    y0 = top - 0.03 - chip_h
    cross = next(i for i in range(1, n) if fr[i - 1]["proba_res"] >= 0.5 > fr[i]["proba_res"])
    for i, f in enumerate(fr):
        x0 = left + i * (cw + gap)
        ax = fig.add_axes([x0, y0, cw, chip_h])
        ax.imshow(imread(APP / f["context"]))
        ax.set_xticks([]); ax.set_yticks([])
        col = TEAL if f["gt"] == 1 else AMBER
        for s in ax.spines.values():
            s.set_edgecolor(col); s.set_linewidth(4)
        fig.text(x0 + cw / 2, y0 - 0.012, f"{f['proba_res']:.2f}", ha="center", va="top", fontsize=10.5,
                 fontweight="bold", color=NAVY)
    xb = left + cross * (cw + gap) - gap / 2
    fig.add_artist(plt.Line2D([xb, xb], [y0 - 0.06, y0 + chip_h + 0.012], color=NAVY, lw=2, ls="--",
                              transform=fig.transFigure))
fig.text(left, 0.012, "Number = model's P(residential) at that step  ·  border = true label of the nearest real cell:",
         fontsize=10, color=SLATE, va="bottom")
fig.text(0.585, 0.012, "■ residential", color=TEAL, fontsize=10.5, fontweight="bold", va="bottom")
fig.text(0.685, 0.012, "■ non-residential", color=AMBER, fontsize=10.5, fontweight="bold", va="bottom")
fig.text(0.825, 0.012, "- - model's 50% boundary", color=NAVY, fontsize=10.5, va="bottom")
fig.savefig(OUT, facecolor="white")
print("wrote", OUT)
for key, _ in rows:
    fr = walks[key]["frames"]
    print(key, [f["gt"] for f in fr], [round(f["proba_res"], 2) for f in fr])
