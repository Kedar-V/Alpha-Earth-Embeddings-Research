#!/usr/bin/env python3
"""Build a barebones 4-section Tuesday meeting deck."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml import parse_xml
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "tuesday_meeting_deck.pptx"
PLOTS = ROOT / "outputs"

BLACK = RGBColor(0x00, 0x00, 0x00)
GRAY = RGBColor(0x44, 0x44, 0x44)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"


def set_run(run, text, size=16, bold=False, color=BLACK):
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = FONT
    run.font.italic = False


def add_text_box(slide, l, t, w, h, text, size=16, bold=False, color=BLACK, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    set_run(p.add_run(), text, size=size, bold=bold, color=color)
    return box


def footer(slide, n):
    add_text_box(
        slide,
        Inches(12.4),
        Inches(7.15),
        Inches(0.5),
        Inches(0.25),
        str(n),
        size=12,
        color=GRAY,
        align=PP_ALIGN.RIGHT,
    )


def header(slide, title, answer, n, section=""):
    if section:
        add_text_box(slide, Inches(0.5), Inches(0.08), Inches(12.3), Inches(0.22), section, size=11, color=GRAY)
        add_text_box(slide, Inches(0.5), Inches(0.28), Inches(12.3), Inches(0.48), title, size=24, bold=True)
        add_text_box(slide, Inches(0.5), Inches(0.78), Inches(12.3), Inches(0.38), answer, size=16, color=GRAY)
    else:
        add_text_box(slide, Inches(0.5), Inches(0.25), Inches(12.3), Inches(0.55), title, size=24, bold=True)
        add_text_box(slide, Inches(0.5), Inches(0.82), Inches(12.3), Inches(0.4), answer, size=16, color=GRAY)
    footer(slide, n)


def bullets(slide, l, t, w, h, items, size=16):
    box = slide.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(8)
        set_run(p.add_run(), "•  " + item, size=size)
    return box


def _set_cell_border(cell):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    ln = "a:lnL a:lnR a:lnT a:lnB"
    xml = (
        "<a:tcBorders xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\">"
        "<a:lnL w=\"6350\" cap=\"flat\" cmpd=\"sng\" algn=\"ctr\">"
        "<a:solidFill><a:srgbClr val=\"000000\"/></a:solidFill><a:prstDash val=\"solid\"/></a:lnL>"
        "<a:lnR w=\"6350\" cap=\"flat\" cmpd=\"sng\" algn=\"ctr\">"
        "<a:solidFill><a:srgbClr val=\"000000\"/></a:solidFill><a:prstDash val=\"solid\"/></a:lnR>"
        "<a:lnT w=\"6350\" cap=\"flat\" cmpd=\"sng\" algn=\"ctr\">"
        "<a:solidFill><a:srgbClr val=\"000000\"/></a:solidFill><a:prstDash val=\"solid\"/></a:lnT>"
        "<a:lnB w=\"6350\" cap=\"flat\" cmpd=\"sng\" algn=\"ctr\">"
        "<a:solidFill><a:srgbClr val=\"000000\"/></a:solidFill><a:prstDash val=\"solid\"/></a:lnB>"
        "</a:tcBorders>"
    )
    # python-pptx uses lnL on tcPr, not tcBorders. Set each edge:
    for edge in ("lnL", "lnR", "lnT", "lnB"):
        el = parse_xml(
            f'<a:{edge} xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            f'w="6350" cap="flat" cmpd="sng" algn="ctr">'
            f"<a:solidFill><a:srgbClr val=\"000000\"/></a:solidFill>"
            f'<a:prstDash val="solid"/>'
            f"</a:{edge}>"
        )
        tcPr.append(el)


def table(slide, l, t, w, h, rows, col_w=None, font=14):
    n_rows, n_cols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_rows, n_cols, l, t, w, h)
    tbl = shape.table
    if col_w:
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = cw
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT
            set_run(p.add_run(), str(val), size=font, bold=(r == 0))
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE
            _set_cell_border(cell)
    return shape


def add_plot(slide, path, l, t, w, h):
    if path.exists():
        slide.shapes.add_picture(str(path), l, t, w, h)
    else:
        add_text_box(slide, l, t, w, h, f"[missing] {path.name}", size=12, color=GRAY)


def make_us_all_cities_plot() -> Path:
    """15 downtowns vs every US city, plus state tag rates."""
    out = PLOTS / "overture_eda" / "us_all_cities_vs_downtowns.png"
    region_all = pd.read_csv(PLOTS / "overture_eda" / "us_all_region_fill_rates.csv")
    state_all = pd.read_csv(PLOTS / "overture_eda" / "us_all_state_fill_rates.csv")
    order = ["Northeast", "Midwest", "South", "West"]
    downtown = {"Northeast": 11.8, "Midwest": 24.0, "South": 23.9, "West": 62.7}
    rall = region_all.set_index("region").reindex(order)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.15), gridspec_kw={"width_ratios": [1.05, 1.55]})

    x = np.arange(len(order))
    axes[0].bar(x - 0.18, [downtown[r] for r in order], width=0.36, color="#777777", label="15 downtowns")
    axes[0].bar(x + 0.18, rall["pct_subtype"].to_numpy(), width=0.36, color="black", label="Every US city")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order)
    axes[0].set_ylabel("% of buildings with a use tag")
    axes[0].set_ylim(0, 80)
    axes[0].set_title("Downtown sample vs every US city")
    axes[0].legend(frameon=False, loc="upper left")

    s = state_all.copy()
    s["region"] = pd.Categorical(s["region"], order, ordered=True)
    s = s.sort_values(["region", "pct_subtype"], ascending=[True, True])
    colors = s["region"].map(
        {"Northeast": "#4C78A8", "Midwest": "#54A24B", "South": "#E45756", "West": "#B279A2"}
    )
    y = np.arange(len(s))
    axes[1].barh(y, s["pct_subtype"], color=colors, height=0.85)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(s["state"], fontsize=7)
    axes[1].set_xlabel("% of buildings with a use tag")
    axes[1].set_title("Every city in each state")
    axes[1].set_xlim(0, 70)
    for lab, col in [
        ("Northeast", "#4C78A8"),
        ("Midwest", "#54A24B"),
        ("South", "#E45756"),
        ("West", "#B279A2"),
    ]:
        axes[1].plot([], [], color=col, linewidth=8, label=lab)
    axes[1].legend(frameon=False, fontsize=9, loc="lower right")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def build():
    make_us_all_cities_plot()
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # 1
    s = prs.slides.add_slide(blank)
    header(
        s,
        "Bottom line",
        "We already have world-wide embeddings. Building-use labels are clear only when a tag exists.",
        1,
        "1 · AlphaEarth embeddings",
    )
    table(
        s,
        Inches(0.5),
        Inches(1.4),
        Inches(12.3),
        Inches(3.2),
        [
            ["Question", "Answer"],
            [
                "Do we have embeddings for the whole world?",
                "Yes. Public AlphaEarth V1 covers land and shallow coast. 10 m. Years 2017–2024. We get them from Earth Engine. Google did not share the model.",
            ],
            [
                "Can Overture tell homes from other buildings?",
                "Yes, if the building has a use tag. Most buildings have no tag. In every US city, 23% have a tag (98M buildings). In 15 US downtowns, 34%. In 8 world downtowns, 17%.",
            ],
            [
                "So what do we do?",
                "Train on tagged buildings. Use shop and school points to add rough labels. Never treat an untagged building as “not a home.”",
            ],
        ],
        col_w=[Inches(4.0), Inches(8.3)],
        font=14,
    )
    add_text_box(s, Inches(0.5), Inches(4.85), Inches(12.3), Inches(0.3), "Decide today", size=16, bold=True)
    add_text_box(
        s,
        Inches(0.5),
        Inches(5.2),
        Inches(12.3),
        Inches(1.5),
        "1. Use the public yearly AlphaEarth layers in Earth Engine.\n"
        "2. Use Overture use tags as clean home vs not-home labels.\n"
        "3. Try Places points next to add coverage. Leave untagged buildings out of training.",
        size=16,
    )

    # 2
    s = prs.slides.add_slide(blank)
    header(
        s,
        "Do we have embeddings for the whole world?",
        "Yes. Use the public yearly product.",
        2,
        "1 · AlphaEarth embeddings",
    )
    table(
        s,
        Inches(0.5),
        Inches(1.4),
        Inches(12.3),
        Inches(2.2),
        [
            ["Option", "Where it covers", "What we do"],
            ["Yearly Satellite Embedding V1", "Land and shallow coast, world-wide", "Use this now"],
            ["Custom embeddings", "A place and time we choose", "Later, if we need months not years"],
            ["The AlphaEarth model itself", "—", "Not shared. We only use the finished numbers."],
        ],
        col_w=[Inches(4.0), Inches(4.2), Inches(4.1)],
        font=15,
    )
    table(
        s,
        Inches(0.5),
        Inches(3.8),
        Inches(12.3),
        Inches(1.8),
        [
            ["View", "What this means"],
            [
                "World",
                "One public 10 m layer for each year 2017–2024. Berlin, Lagos, and Boston use the same numbers.",
            ],
            [
                "US: 4 regions",
                "The same layer covers Northeast, Midwest, South, and West. There is no separate US product.",
            ],
        ],
        col_w=[Inches(2.2), Inches(10.1)],
        font=14,
    )
    add_text_box(
        s,
        Inches(0.5),
        Inches(5.8),
        Inches(12.3),
        Inches(1.0),
        "Get it in Earth Engine: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL. License is CC-BY 4.0. Each cell is 64 numbers. Google did not share the model.",
        size=14,
        color=GRAY,
    )

    # 3 — invite checklist: sources, imagery, sensors, resolution, years, model
    s = prs.slides.add_slide(blank)
    header(
        s,
        "What went into AlphaEarth embeddings?",
        "Google already made the numbers. We use the public 10 m yearly layers. We do not start from raw satellite photos.",
        3,
        "1 · AlphaEarth embeddings",
    )
    table(
        s,
        Inches(0.5),
        Inches(1.35),
        Inches(12.3),
        Inches(5.15),
        [
            ["Question", "Answer"],
            ["What product?", "Satellite Embedding V1, from AlphaEarth Foundations v2.1."],
            [
                "What data sources?",
                "To make the public layers: Sentinel-2, Landsat 8/9, and Sentinel-1. Height, climate, and text were used only while training.",
            ],
            [
                "What types of pictures?",
                "Color and infrared photos, heat, and radar. Not close-up commercial photos of building fronts.",
            ],
            ["What sensors?", "Sentinel-2, Landsat 8/9, and Sentinel-1 radar."],
            [
                "What resolution?",
                "Each embedding cell is 10 m across. A house is about 1 to 4 cells.",
            ],
            ["What years?", "2017 to 2024. One world-wide layer per year. Eight years in total."],
            [
                "What model made them?",
                "AlphaEarth Foundations STP (about 480 million parameters). Google did not share the model.",
            ],
        ],
        col_w=[Inches(2.8), Inches(9.5)],
        font=14,
    )
    add_text_box(
        s,
        Inches(0.5),
        Inches(6.6),
        Inches(12.3),
        Inches(0.5),
        "Sources: DeepMind blog · Brown et al. 2025, arXiv:2507.22291 · Earth Engine catalog GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
        size=12,
        color=GRAY,
    )

    # 4
    s = prs.slides.add_slide(blank)
    header(
        s,
        "How are the embeddings made?",
        "From Sentinel-2, Landsat, and Sentinel-1. Not from close-up commercial photos.",
        4,
        "1 · AlphaEarth embeddings",
    )
    table(
        s,
        Inches(0.5),
        Inches(1.4),
        Inches(12.3),
        Inches(2.1),
        [
            ["Type of picture", "Sensor", "Pixel size", "When used"],
            ["Color / infrared", "Sentinel-2", "10 / 20 / 60 m", "To make the public layers"],
            ["Color / heat", "Landsat 8/9", "15 / 30 / 100 m", "To make the public layers"],
            ["Radar", "Sentinel-1", "10 m", "To make the public layers"],
        ],
        col_w=[Inches(2.4), Inches(5.2), Inches(2.4), Inches(2.3)],
        font=14,
    )
    bullets(
        s,
        Inches(0.5),
        Inches(3.75),
        Inches(12.3),
        Inches(3.0),
        [
            "The model is AlphaEarth Foundations STP (about 480 million parameters). It learned from millions of year-long clips at sites around the world.",
            "Height, climate, and land-cover maps were used only in training. Building height in the embeddings is weak.",
            "A house is about 1 to 4 pixels. Nearby streets and blocks are already in the numbers. We will sample 2024 and average inside each building later.",
        ],
    )

    # ----- 2 · Overture EDA (notebook 06) -----
    # 5
    s = prs.slides.add_slide(blank)
    header(
        s,
        "Can Overture tell homes from other buildings?",
        "Yes, if the building has a use tag. Same rule everywhere. Most tagged buildings are homes.",
        5,
        "2 · Overture labels · notebook 06",
    )
    table(
        s,
        Inches(0.5),
        Inches(1.25),
        Inches(5.9),
        Inches(1.55),
        [
            ["If Overture says…", "We treat it as…"],
            ["residential", "Home"],
            ["any other use tag", "Not a home"],
            ["no tag", "Skip. Do not guess."],
        ],
        col_w=[Inches(2.6), Inches(3.3)],
        font=13,
    )
    bullets(
        s,
        Inches(0.5),
        Inches(2.95),
        Inches(5.9),
        Inches(3.8),
        [
            "Of tagged buildings in the 8 world downtowns, 82% are homes.",
            "Of tagged buildings in the 15 US downtowns, 86% are homes.",
            "The rest of the tagged set is shops, schools, offices, and other uses.",
            "Never treat a missing tag as “not a home.”",
        ],
        size=15,
    )
    add_plot(s, PLOTS / "overture_eda" / "subtype_mix_labeled.png", Inches(6.6), Inches(1.25), Inches(6.3), Inches(5.6))

    # 6
    s = prs.slides.add_slide(blank)
    header(
        s,
        "Do enough buildings have a use tag?",
        "No. 17% tagged in all 8 world downtowns (883k buildings). Europe is high. Lagos and Nairobi are not.",
        6,
        "2 · Overture labels · notebook 06",
    )
    add_plot(s, PLOTS / "overture_eda" / "fill_rates_by_city.png", Inches(0.4), Inches(1.25), Inches(7.0), Inches(5.6))
    table(
        s,
        Inches(7.5),
        Inches(1.25),
        Inches(5.4),
        Inches(5.5),
        [
            ["City", "% with a tag", "Of tagged: homes"],
            ["Berlin", "75%", "76%"],
            ["Amsterdam", "70%", "91%"],
            ["Chicago", "24%", "75%"],
            ["Boston", "22%", "80%"],
            ["Houston", "16%", "83%"],
            ["SF", "9%", "67%"],
            ["Nairobi", "2%", "67%"],
            ["Lagos", "1%", "82%"],
            ["All 8", "17%", "82%"],
        ],
        col_w=[Inches(1.8), Inches(1.8), Inches(1.8)],
        font=13,
    )

    # 7
    s = prs.slides.add_slide(blank)
    header(
        s,
        "How do US downtowns compare?",
        "34% tagged across all 15 US downtowns (2.18M buildings). The West is high because of LA and Denver.",
        7,
        "2 · Overture labels · notebook 06",
    )
    add_plot(
        s,
        PLOTS / "overture_eda" / "us_fill_rate_region_state_city.png",
        Inches(0.35),
        Inches(1.2),
        Inches(8.3),
        Inches(5.7),
    )
    table(
        s,
        Inches(8.75),
        Inches(1.2),
        Inches(4.15),
        Inches(2.6),
        [
            ["US region", "% tagged", "Homes"],
            ["West", "63%", "87%"],
            ["Midwest", "24%", "84%"],
            ["South", "24%", "88%"],
            ["Northeast", "12%", "78%"],
            ["All 15", "34%", "86%"],
        ],
        col_w=[Inches(1.55), Inches(1.3), Inches(1.3)],
        font=12,
    )
    add_plot(
        s,
        PLOTS / "overture_eda" / "us_city_fill_heatmap.png",
        Inches(8.75),
        Inches(4.0),
        Inches(4.15),
        Inches(2.9),
    )

    # 8 — every US city
    s = prs.slides.add_slide(blank)
    header(
        s,
        "Does the whole US look like those 15 downtowns?",
        "No. In every US city, 23% of 98 million buildings have a use tag. The 15 downtowns were 34%.",
        8,
        "2 · Overture labels · notebook 06",
    )
    add_plot(
        s,
        PLOTS / "overture_eda" / "us_all_cities_vs_downtowns.png",
        Inches(0.3),
        Inches(1.15),
        Inches(8.55),
        Inches(5.75),
    )
    table(
        s,
        Inches(8.95),
        Inches(1.2),
        Inches(3.95),
        Inches(3.35),
        [
            ["US region", "Cities", "% tagged"],
            ["West", "6,071", "32%"],
            ["Northeast", "4,929", "21%"],
            ["Midwest", "10,502", "21%"],
            ["South", "10,527", "19%"],
            ["All US", "32,029", "23%"],
        ],
        col_w=[Inches(1.45), Inches(1.25), Inches(1.25)],
        font=12,
    )
    add_text_box(
        s,
        Inches(8.95),
        Inches(4.7),
        Inches(3.95),
        Inches(2.15),
        "This is every city and town in the 50 states and DC. Of tagged buildings, 89% are homes. Buildings outside a city are left out. The West downtown number was high because of LA and Denver.",
        size=13,
    )

    # 9
    s = prs.slides.add_slide(blank)
    header(
        s,
        "Why do so few buildings have a tag?",
        "Use tags come from OpenStreetMap only. Extra Microsoft and Google outlines have no use tag.",
        9,
        "2 · Overture labels · notebook 06",
    )
    add_plot(s, PLOTS / "overture_eda" / "label_rate_by_source.png", Inches(0.4), Inches(1.2), Inches(6.8), Inches(5.6))
    table(
        s,
        Inches(7.4),
        Inches(1.2),
        Inches(5.5),
        Inches(3.7),
        [
            ["City", "OSM % tagged", "Extra % tagged"],
            ["Berlin", "79%", "0%"],
            ["Amsterdam", "70%", "0%"],
            ["Boston", "22%", "0%"],
            ["Houston", "16%", "0%"],
            ["Nairobi", "3%", "0%"],
            ["Lagos", "1%", "0%"],
        ],
        col_w=[Inches(1.9), Inches(1.8), Inches(1.8)],
        font=13,
    )
    add_text_box(
        s,
        Inches(7.4),
        Inches(5.1),
        Inches(5.5),
        Inches(1.7),
        "Every use tag we saw comes from OpenStreetMap. Microsoft, Google, and Esri outlines have zero use tags. Do not treat those outlines as “not a home.”",
        size=14,
    )

    # 9
    s = prs.slides.add_slide(blank)
    header(
        s,
        "Can extra map points fill the gaps?",
        "Yes. Shop and school points are more common than building tags. Treat them as rough labels, not as truth.",
        10,
        "2 · Overture labels · notebook 06",
    )
    add_plot(s, PLOTS / "overture_eda" / "buildings_vs_places.png", Inches(0.35), Inches(1.2), Inches(6.5), Inches(3.55))
    add_plot(s, PLOTS / "overture_eda" / "taxonomy_coverage.png", Inches(6.95), Inches(1.2), Inches(5.95), Inches(3.55))
    table(
        s,
        Inches(0.5),
        Inches(4.85),
        Inches(12.3),
        Inches(2.1),
        [
            ["City", "School buildings", "School points", "Hospital buildings", "Hospital points"],
            ["Berlin", "536", "540", "148", "99"],
            ["Boston", "199", "502", "92", "399"],
            ["Nairobi", "182", "677", "60", "363"],
            ["Lagos", "5", "447", "2", "95"],
            ["All 8 downtowns", "1,468", "3,295", "409", "1,358"],
        ],
        col_w=[Inches(2.5), Inches(2.45), Inches(2.45), Inches(2.5), Inches(2.4)],
        font=12,
    )

    # ----- 3 · Embedding EDA (notebook 08) -----
    # 10
    s = prs.slides.add_slide(blank)
    header(
        s,
        "Do building types show up in the embeddings?",
        "A little. 11,280 tagged buildings. Types sit in one cloud, not in clean groups.",
        11,
        "3 · Embedding space · notebook 08",
    )
    add_plot(s, PLOTS / "embedding_eda" / "pca_overview.png", Inches(0.35), Inches(1.2), Inches(6.4), Inches(5.6))
    add_plot(s, PLOTS / "embedding_eda" / "umap_by_label.png", Inches(6.85), Inches(1.2), Inches(6.1), Inches(4.4))
    add_text_box(
        s,
        Inches(6.85),
        Inches(5.7),
        Inches(6.1),
        Inches(1.1),
        "First two axes hold 33% of the variance. Ten building types overlap. Houses stand out more than churches or civic buildings.",
        size=14,
    )

    # 11
    s = prs.slides.add_slide(blank)
    header(
        s,
        "Is the embedding showing place or building type?",
        "Place first. Cities pull apart. Types inside a city do not.",
        12,
        "3 · Embedding space · notebook 08",
    )
    add_plot(s, PLOTS / "embedding_eda" / "umap_by_region.png", Inches(0.4), Inches(1.2), Inches(7.8), Inches(5.6))
    bullets(
        s,
        Inches(8.4),
        Inches(1.4),
        Inches(4.5),
        Inches(5.2),
        [
            "Boston, LA, and Houston sit in different parts of the plot.",
            "A house in Boston looks more like a shop in Boston than like a house in LA.",
            "A model trained in one city will lose some skill in another city.",
            "We still have type signal, but we must test transfer across cities.",
        ],
        size=15,
    )

    # 12
    s = prs.slides.add_slide(blank)
    header(
        s,
        "Do nearby buildings share a type?",
        "Yes, more than chance. Neighbors help. Automatic grouping into 10 piles does not.",
        13,
        "3 · Embedding space · notebook 08",
    )
    add_plot(s, PLOTS / "embedding_eda" / "nn_precision.png", Inches(0.35), Inches(1.2), Inches(6.4), Inches(5.6))
    add_plot(s, PLOTS / "embedding_eda" / "clustering_overview.png", Inches(6.85), Inches(1.2), Inches(6.1), Inches(4.2))
    table(
        s,
        Inches(6.85),
        Inches(5.5),
        Inches(6.1),
        Inches(1.35),
        [
            ["Check", "Score", "Chance"],
            ["Nearest neighbor is the same type", "41%", "10%"],
            ["Auto 10 groups match true types", "2%", "near 0"],
        ],
        col_w=[Inches(3.3), Inches(1.4), Inches(1.4)],
        font=12,
    )

    # 13
    s = prs.slides.add_slide(blank)
    header(
        s,
        "Is there enough signal for a first model?",
        "Yes for a supervised home vs not-home test. No for finding 10 types with no labels.",
        14,
        "3 · Embedding space · notebook 08",
    )
    add_plot(s, PLOTS / "embedding_eda" / "distance_analysis.png", Inches(0.35), Inches(1.2), Inches(6.5), Inches(5.6))
    add_plot(s, PLOTS / "embedding_eda" / "similarity_and_nn1.png", Inches(6.95), Inches(1.2), Inches(5.95), Inches(3.7))
    add_text_box(
        s,
        Inches(6.95),
        Inches(5.05),
        Inches(5.95),
        Inches(1.8),
        "Same-type buildings are only a little closer than different-type buildings (gap 0.03). Houses match their neighbors 76% of the time. Churches 17%. Enough to train a simple model. Not enough to skip labels.",
        size=14,
    )

    # ----- 4 · First classifier (notebook 02) -----
    # 14
    s = prs.slides.add_slide(blank)
    header(
        s,
        "Does a simple model work on tagged buildings?",
        "Yes. Logistic regression on 2024 embeddings. 3,600 tagged buildings. 83% on a mixed holdout.",
        15,
        "4 · First classifier · notebook 02",
    )
    table(
        s,
        Inches(0.5),
        Inches(1.25),
        Inches(12.3),
        Inches(2.55),
        [
            ["Setup", "Choice"],
            ["Task", "Home vs not-home"],
            ["Labels", "Overture use tags only"],
            ["Features", "64 numbers at the building center, 2024"],
            ["Model", "Logistic regression"],
            ["Cities", "9 US metros, 200 home + 200 not-home each"],
        ],
        col_w=[Inches(2.6), Inches(9.7)],
        font=14,
    )
    table(
        s,
        Inches(0.5),
        Inches(4.05),
        Inches(12.3),
        Inches(2.5),
        [
            ["Split", "Accuracy", "Macro F1", "ROC-AUC"],
            ["Mixed holdout", "83%", "83%", "0.90"],
            ["Leave one city out (mean)", "77%", "77%", "0.84"],
            ["Best city (Atlanta)", "87%", "87%", "0.92"],
            ["Worst city (NYC)", "58%", "57%", "0.67"],
        ],
        col_w=[Inches(4.5), Inches(2.6), Inches(2.6), Inches(2.6)],
        font=14,
    )

    # 15
    s = prs.slides.add_slide(blank)
    header(
        s,
        "Does the model transfer to a new city?",
        "Mostly. Mean leave-one-city-out is 77%. Atlanta, Seattle, and LA hold up. NYC and Denver drop.",
        16,
        "4 · First classifier · notebook 02",
    )
    add_plot(
        s,
        PLOTS / "trial_residential" / "us_region_performance_map.png",
        Inches(0.3),
        Inches(1.2),
        Inches(6.6),
        Inches(5.6),
    )
    add_plot(
        s,
        PLOTS / "trial_residential" / "us_region_performance_bars.png",
        Inches(6.95),
        Inches(1.2),
        Inches(6.0),
        Inches(5.6),
    )

    # 16
    s = prs.slides.add_slide(blank)
    header(
        s,
        "What does this first model buy us?",
        "A working home vs not-home baseline on tagged buildings. Not a finished 10-type product.",
        17,
        "4 · First classifier · notebook 02",
    )
    add_plot(
        s,
        PLOTS / "trial_residential" / "roc_pr_pooled_vs_loro.png",
        Inches(0.35),
        Inches(1.2),
        Inches(7.3),
        Inches(5.6),
    )
    add_text_box(s, Inches(7.8), Inches(1.25), Inches(5.1), Inches(0.35), "Keep", size=16, bold=True)
    bullets(
        s,
        Inches(7.8),
        Inches(1.6),
        Inches(5.1),
        Inches(2.3),
        [
            "Public yearly AlphaEarth layers.",
            "Train only on tagged buildings.",
            "This logistic regression as the first bar to beat.",
        ],
        size=14,
    )
    add_text_box(s, Inches(7.8), Inches(4.05), Inches(5.1), Inches(0.35), "Next", size=16, bold=True)
    bullets(
        s,
        Inches(7.8),
        Inches(4.4),
        Inches(5.1),
        Inches(2.3),
        [
            "Average the embedding inside the building, not just the center.",
            "Link shop and school points where tags are rare.",
            "Test Europe and a tagged African downtown.",
        ],
        size=14,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print("Wrote", OUT)


if __name__ == "__main__":
    build()
