// Build the AlphaEarth research deck. Run: node presentation/build_deck.js
// Visual theme follows the team's weekly client-update deck (grey ground, navy + teal, bold sans).
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/home/kv133/.claude/skills/synced/13d5e1f2-0768-4c46-aa8a-d16c0e142588_4dbf7dd0-96ad-4a93-a635-70ed48a8e3ab/pptx/scripts/apply_theme.js");

const THEME = {
  name: "Weekly Update",
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  colors: {
    dk1: "0B2238",   // navy: titles, primary text
    lt1: "FFFFFF",   // cards
    dk2: "4B5A6E",   // slate: body / subtitles
    lt2: "EEEEEE",   // slide background
    accent1: "2B9DB0", // teal: headline accent
    accent2: "0057A0", // mid blue
    accent3: "0B1F5C", // deep navy blue
    accent4: "CFE5E8", // light teal tint (chips, highlight cards)
    accent5: "C27A2C", // amber: warnings / escalation
    accent6: "6B7280", // grey
    hlink: "2B9DB0", folHlink: "0057A0",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "AlphaEarth Embeddings Research";
const C = pres.SchemeColor;

// ---------- Layouts ----------
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: C.background2 },
  objects: [
    { placeholder: { options: { name: "wordmark", type: "body", x: 0.85, y: 1.3, w: 11.5, h: 1.45,
        fontSize: 96, bold: true, color: C.text1, valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.85, y: 2.8, w: 11.8, h: 1.0,
        fontSize: 48, bold: true, color: C.accent1, valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.94, y: 4.45, w: 10.2, h: 0.8,
        fontSize: 20, color: C.text2, valign: "top", margin: 0 }, text: "" } },
    { line: { x: 0.94, y: 5.45, w: 0.95, h: 0, line: { color: C.accent1, width: 3 } } },
    { placeholder: { options: { name: "team", type: "body", x: 0.94, y: 5.75, w: 10.0, h: 0.35,
        fontSize: 14, bold: true, color: C.text1, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "meta", type: "body", x: 0.94, y: 6.1, w: 10.0, h: 0.3,
        fontSize: 12, color: C.text2, margin: 0 }, text: "" } },
  ],
});

// Content slide: title top-left, everything else composed per slide.
pres.defineSlideMaster({
  title: "TITLE_ONLY",
  background: { color: C.background2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.55, w: 12.1, h: 0.75,
        fontSize: 28, bold: true, color: C.text1, valign: "middle", margin: 0 }, text: "" } },
  ],
});

// Section divider: big teal number, navy title, slate question (same family as the title slide)
pres.defineSlideMaster({
  title: "SECTION",
  background: { color: C.background2 },
  objects: [
    { placeholder: { options: { name: "num", type: "body", x: 0.85, y: 1.9, w: 3, h: 0.7,
        fontSize: 32, bold: true, color: C.accent1, margin: 0, valign: "bottom" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.85, y: 2.7, w: 11.6, h: 1.3,
        fontSize: 48, bold: true, color: C.text1, valign: "top", margin: 0 }, text: "" } },
    { line: { x: 0.94, y: 4.25, w: 0.95, h: 0, line: { color: C.accent1, width: 3 } } },
    { placeholder: { options: { name: "sub", type: "body", x: 0.94, y: 4.5, w: 10.5, h: 1.2,
        fontSize: 20, color: C.text2, valign: "top", margin: 0 }, text: "" } },
  ],
});

// ---------- Slide 1: Introduction ----------
pres.addSection({ title: "Introduction" });
const s1 = pres.addSlide({ masterName: "TITLE", sectionTitle: "Introduction" });
s1.addText("AlphaEarth", { placeholder: "wordmark" });
s1.addText("Building-Use Intelligence", { placeholder: "title" });
s1.addText(
  "Research update: residential vs non-residential mapping, generalisation across metros and countries, and the GHSL baseline",
  { placeholder: "subtitle" });
s1.addText("AlphaEarth Embeddings Research", { placeholder: "team" });
s1.addText("November 2026", { placeholder: "meta" });
s1.addNotes(
  "Introduce the question: we want building-use labels (residential vs non-residential, and finer types) " +
  "at global scale to feed carbon-emissions estimates. AlphaEarth gives us a ready-made global 10 m, 64-D " +
  "embedding (annual layers 2017-2024), so the open question is how far these embeddings alone can go, and where they break.");

// ---------- Icons (react-icons -> PNG) ----------
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
async function icon(IconComp, hex) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(IconComp, { color: "#" + hex, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
const { FaSatellite, FaBuilding, FaCity } = require("react-icons/fa");

async function buildSources() {
  // ---------- Slide 2: Our sources ----------
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Data" });
  s.addText("Our sources: features, labels and a baseline", { placeholder: "title" });

  const cards = [
    {
      role: "INPUT FEATURES", icon: FaSatellite,
      name: "AlphaEarth embeddings", by: "Google DeepMind · Satellite Embedding V1",
      bullets: [
        "64 numbers per 10 m pixel, learned from many satellite sensors",
        "Global land coverage, annual layers 2017–2024",
        "Precomputed vectors from Earth Engine; model weights not released",
      ],
    },
    {
      role: "GROUND TRUTH", icon: FaBuilding,
      name: "Overture Maps", by: "Overture Maps Foundation · Buildings + Places",
      bullets: [
        "Building footprints with use tags, mostly from OpenStreetMap",
        "Tag coverage varies: ~75% in Berlin, ~1% in Lagos",
        "Places (points of interest) kept as a separate weak-label track",
      ],
    },
    {
      role: "CONTEXT & BASELINE", icon: FaCity,
      name: "GHSL", by: "EU Joint Research Centre · Global Human Settlement Layer",
      bullets: [
        "GHS-SMOD: urban, peri-urban or rural for each building",
        "GHS-BUILT-S: built-up surface fraction, our baseline",
        "GHS-BUILT-C: GHSL's own residential vs non-residential map",
      ],
    },
  ];

  const x0 = 0.6, cw = 3.83, gap = 0.3, y0 = 1.6, ch = 4.4;
  for (let i = 0; i < cards.length; i++) {
    const c = cards[i];
    const x = x0 + i * (cw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: y0, w: cw, h: ch, rectRadius: 0.08,
      fill: { color: C.background1 }, line: { color: "DDD6C8", width: 0.75 },
      objectName: `source-card-${i + 1}`,
    });
    // header chip: icon in navy circle + role label
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: x + 0.15, y: y0 + 0.15, w: cw - 0.3, h: 0.55, rectRadius: 0.08,
      fill: { color: C.accent4 }, line: { type: "none" }, objectName: `source-chip-${i + 1}`,
    });
    s.addShape(pres.shapes.OVAL, {
      x: x + 0.27, y: y0 + 0.24, w: 0.37, h: 0.37,
      fill: { color: C.text1 }, line: { type: "none" }, objectName: `source-icon-bg-${i + 1}`,
    });
    s.addImage({
      data: await icon(c.icon, "FFFFFF"), x: x + 0.355, y: y0 + 0.325, w: 0.2, h: 0.2,
      altText: c.role.toLowerCase() + " icon", objectName: `source-icon-${i + 1}`,
    });
    s.addText(c.role, {
      x: x + 0.78, y: y0 + 0.15, w: cw - 1.0, h: 0.55, fontSize: 16, bold: true,
      color: C.text1, valign: "middle", margin: 0, isTextBox: true, objectName: `source-role-${i + 1}`,
    });
    // name + provider
    s.addText(c.name, {
      x: x + 0.3, y: y0 + 0.95, w: cw - 0.6, h: 0.4, fontSize: 20, bold: true,
      color: C.accent1, valign: "top", margin: 0, isTextBox: true, objectName: `source-name-${i + 1}`,
    });
    s.addText(c.by, {
      x: x + 0.3, y: y0 + 1.38, w: cw - 0.6, h: 0.48, fontSize: 12,
      color: C.text2, valign: "top", margin: 0, isTextBox: true, objectName: `source-by-${i + 1}`,
    });
    s.addText(
      c.bullets.map((b, j) => ({
        text: b,
        options: { bullet: true, paraSpaceAfter: 10, breakLine: j < c.bullets.length - 1 },
      })),
      {
        x: x + 0.3, y: y0 + 1.95, w: cw - 0.55, h: ch - 2.1, fontSize: 14,
        color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: `source-bullets-${i + 1}`,
      });
  }

  // bottom rule bar (mirrors the "To confirm with BNY" banner)
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y: 6.3, w: 12.1, h: 0.65, rectRadius: 0.08,
    fill: { color: C.text1 }, line: { type: "none" }, objectName: "rule-banner",
  });
  s.addText([
    { text: "Labelling rule", options: { bold: true, color: C.accent1 } },
    { text: "     Buildings without a use tag are left out, never counted as non-residential.", options: { color: C.background1 } },
  ], {
    x: 0.9, y: 6.3, w: 11.6, h: 0.65, fontSize: 16, valign: "middle", margin: 0,
    isTextBox: true, objectName: "rule-text",
  });

  s.addNotes(
    "Three sources. AlphaEarth gives the features: a 64-D vector per 10 m pixel, global and annual, read from Earth Engine. " +
    "Overture gives the labels: building footprints whose use tags come mostly from OpenStreetMap. The catch is coverage, " +
    "about 75% of buildings tagged in Berlin but about 1% in Lagos, so labels, not embeddings, limit how far we can go. " +
    "Places points of interest are a weaker, separate label track and are never pooled into the main scores. " +
    "GHSL from the EU JRC gives context and a baseline: SMOD to split urban, peri-urban and rural; BUILT-S and BUILT-C as " +
    "the existing products our model has to beat. Untagged buildings are excluded, never treated as non-residential.");
}

// ---------- Shared helpers ----------
const FIG = (f) => path.join(__dirname, "figures", f);
function card(s, x, y, w, h, name, fill) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.08, fill: { color: fill || C.background1 },
    line: { color: "DDD6C8", width: 0.75 }, objectName: name,
  });
}
function banner(s, label, text, y, h) {
  h = h || 0.65;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y, w: 12.1, h, rectRadius: 0.08, fill: { color: C.text1 }, line: { type: "none" },
    objectName: "banner",
  });
  s.addText([
    { text: label, options: { bold: true, color: C.accent1 } },
    { text: "     " + text, options: { color: C.background1 } },
  ], { x: 0.9, y, w: 11.6, h, fontSize: 16, valign: "middle", margin: 0, isTextBox: true, objectName: "banner-text" });
}
function step(s, n, x, y, w, head, body) {
  s.addShape(pres.shapes.OVAL, {
    x, y, w: 0.42, h: 0.42, fill: { color: C.text1 }, line: { type: "none" }, objectName: `step-${n}-dot`,
  });
  s.addText(String(n), {
    x, y, w: 0.42, h: 0.42, fontSize: 15, bold: true, color: C.background1, align: "center",
    valign: "middle", margin: 0, isTextBox: true, objectName: `step-${n}-num`,
  });
  s.addText([
    { text: head, options: { bold: true, fontSize: 16, color: C.text1, breakLine: true } },
    { text: body, options: { fontSize: 14, color: C.text2 } },
  ], { x: x + 0.6, y: y - 0.02, w: w - 0.6, h: 1.25, valign: "top", margin: 0, paraSpaceAfter: 4,
       isTextBox: true, objectName: `step-${n}-text` });
}
function caption(s, text, x, y, w, name) {
  s.addText(text, { x, y, w, h: 0.32, fontSize: 12, color: C.text2, italic: true, margin: 0,
                    valign: "top", isTextBox: true, objectName: name || "caption" });
}


// chart rendered by presentation/make_charts.py (data-visualization-principles skill), placed on a white card
function chartImage(s, file, x, y, w, h, alt, name) {
  card(s, x, y, w, h, `${name}-card`);
  s.addImage({ path: FIG(file), x: x + 0.1, y: y + 0.1, w: w - 0.2, h: h - 0.2, altText: alt, objectName: name });
}

// ---------- Slide 3: Ground truth ----------
function buildGroundTruth() {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Data" });
  s.addText("Ground truth: from tagged buildings to 10 m cells", { placeholder: "title" });

  card(s, 0.6, 1.55, 8.1, 4.0, "gt-map-card");
  s.addImage({ path: FIG("gt_footprints_to_cells.png"), x: 0.72, y: 1.66, w: 7.86, h: 3.74,
               altText: "Cambridge MA: Overture buildings coloured residential or non-residential, and the AlphaEarth 10 m cells coloured by residential share",
               objectName: "gt-map" });
  caption(s, "Cambridge, MA (Boston metro) · Overture buildings with a use tag · AlphaEarth 10 m grid", 0.6, 5.65, 8.1, "gt-caption");

  step(s, 1, 9.05, 1.65, 3.65, "Label the building",
       "Overture class: house, apartments → residential; school, office, retail, hospital … → non-residential");
  step(s, 2, 9.05, 3.05, 3.65, "Share per 10 m cell",
       "Each building counts by how much of the cell it covers");
  step(s, 3, 9.05, 4.3, 3.65, "Keep pure cells",
       "Classification trains on cells with one building type. Under 1% of cells are mixed");

  banner(s, "Rule", "Buildings without a use tag are left out, never counted as non-residential.", 6.1);
  s.addNotes(
    "Ground truth comes from Overture building footprints that carry a use tag. The Overture class maps to " +
    "residential (house, detached, semi, apartments, dormitory) or non-residential (education, medical, retail, office, " +
    "hotel, industrial, religious, civic). Classes like 'residential' with no detail, 'terrace' and null are excluded. " +
    "Each AlphaEarth 10 m cell gets a residential share, weighted by how much of the cell each tagged building covers. " +
    "For classification we keep pure cells, where only one building type is present: 3,128,049 of 3,151,005 cells, so " +
    "fewer than 1% are mixed. Regression uses the share directly. Untagged buildings are never treated as negatives.");
}

// ---------- Slide 4: Balancing ----------
function buildBalancing() {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Data" });
  s.addText("Balancing: equal classes in every metro", { placeholder: "title" });

  card(s, 0.6, 1.55, 7.1, 3.85, "bal-map-card");
  s.addImage({ path: FIG("balancing_before_after_map.png"), x: 0.72, y: 1.66, w: 6.86, h: 3.6,
               altText: "Phoenix: before balancing 851 residential and 231 non-residential cells; after, 178 residential and 231 non-residential",
               objectName: "bal-map" });
  caption(s, "Phoenix, AZ · residential cells are thinned at random across the metro; every non-residential cell is kept",
          0.6, 5.48, 7.1, "bal-caption");

  // per-metro counts before balancing (thousands), from datasets/week4_pixel_gt_balanced
  const metros = ["Atlanta", "Boston", "Chicago", "Denver", "Houston", "LA", "NYC", "Phoenix", "SF", "Seattle"];
  const res = [320.9, 107.7, 73.5, 240.8, 122.1, 566.9, 116.7, 396.1, 17.5, 200.3];
  const nres = [56.4, 93.3, 157.2, 130.1, 84.0, 228.4, 80.0, 74.3, 23.3, 38.5];
  chartImage(s, "chart_balance.png", 7.95, 1.55, 4.75, 3.85,
    "Residential share of pure cells per metro before balancing: from 85% in Atlanta to 32% in Chicago", "bal-chart");

  // stat strip
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y: 5.95, w: 12.1, h: 1.0, rectRadius: 0.08, fill: { color: C.text1 }, line: { type: "none" },
    objectName: "bal-stats",
  });
  const stats = [
    ["3.13M → 1.75M", "pure cells, before → after"],
    ["69/31 → 50/50", "residential / non-residential split"],
    ["71,497 · 25,220", "frozen 1:1 test buildings · 100 m tiles"],
  ];
  stats.forEach(([big, small], i) => {
    const x = 0.9 + i * 4.0;
    s.addText([
      { text: big, options: { fontSize: 22, bold: true, color: C.accent1, breakLine: true } },
      { text: small, options: { fontSize: 12, color: C.background1 } },
    ], { x, y: 5.95, w: 3.7, h: 1.0, valign: "middle", margin: 0, isTextBox: true, objectName: `bal-stat-${i + 1}` });
  });

  s.addNotes(
    "Without balancing, a model can score well by mostly predicting the majority class. Across the 10 metros there are " +
    "2,162,534 pure residential cells and 965,515 pure non-residential cells (69/31). In each metro we keep the smaller " +
    "class whole and randomly drop cells from the larger one (seed 42, no synthetic samples). Most metros drop residential " +
    "cells; Chicago and SF have more non-residential cells, so there the non-residential side is thinned. Result: 875,907 " +
    "cells per class. The evaluation set is balanced 1:1 per metro the same way and then frozen: 71,497 buildings and " +
    "25,220 100 m tiles, so every model is scored on exactly the same rows.");
}

// ---------- Slide 5: Fair comparison ----------
function buildFairTest() {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Data" });
  s.addText("A fair test: each map read in its own grid", { placeholder: "title" });

  card(s, 0.6, 1.55, 4.45, 4.25, "grid-card");
  s.addImage({ path: FIG("ae_vs_ghsl_shifted_grids.png"), x: 0.95, y: 1.65, w: 3.75, h: 3.55,
               altText: "Houston: AlphaEarth 10 m cells in blue and GHSL 10 m cells in orange, offset by 5.3 m",
               objectName: "grid-map" });
  s.addText("Houston: the two 10 m grids sit ~4 m apart (median overlap 0.36). Resampling either one would change its answers.", {
    x: 0.8, y: 5.22, w: 4.05, h: 0.55, fontSize: 12, color: C.text2, italic: true, margin: 0, valign: "top",
    isTextBox: true, objectName: "grid-caption",
  });

  const how = [
    ["AlphaEarth → building", "Average of the AlphaEarth cells inside the outline, in its own UTM grid"],
    ["GHSL → building", "Average of GHS-BUILT-S (2018) under the same outline, in GHSL's own Mollweide grid"],
  ];
  how.forEach(([head, body], i) => {
    const x = 5.3 + i * 3.8;
    card(s, x, 1.55, 3.55, 1.5, `how-card-${i + 1}`, THEME.colors.accent4);
    s.addText([
      { text: head, options: { bold: true, fontSize: 16, color: C.text1, breakLine: true } },
      { text: body, options: { fontSize: 14, color: C.text1 } },
    ], { x: x + 0.2, y: 1.65, w: 3.15, h: 1.3, valign: "top", margin: 0, paraSpaceAfter: 4,
         isTextBox: true, objectName: `how-text-${i + 1}` });
  });

  card(s, 5.3, 3.3, 7.35, 2.5, "ghsl-card");
  s.addImage({ path: FIG("ghsl_codes_under_footprint.png"), x: 5.4, y: 3.38, w: 4.75, h: 2.3,
               altText: "GHSL 10 m cells around one NYC footprint, and only the cells under the footprint",
               objectName: "ghsl-mask" });
  s.addText([
    { text: "Example: one NYC building", options: { bold: true, fontSize: 14, color: C.text1, breakLine: true } },
    { text: "Left: GHSL cells around the outline. Right: only the cells under it are averaged.", options: { fontSize: 14, color: C.text2 } },
  ], { x: 10.3, y: 3.5, w: 2.2, h: 2.2, valign: "top", margin: 0, paraSpaceAfter: 4, isTextBox: true, objectName: "ghsl-text" });

  banner(s, "Same for every model", "71,497 frozen buildings · one metro held out · error = ½ (res + non-res)", 6.1);
  s.addNotes(
    "AlphaEarth and GHSL both come on 10 m grids, but not the same grid: AlphaEarth is in a per-metro UTM projection, " +
    "GHSL in Mollweide. The cell centres are a median 4.1 m apart and a median cell overlap (IoU) is only 0.36. Warping " +
    "either grid onto the other would change its predictions, so we don't. Instead each product is summarised inside the " +
    "Overture building outline in its own native grid: AlphaEarth by averaging the cells inside the polygon, GHSL by " +
    "averaging the GHS-BUILT-S 2018 fractions, split into residential and non-residential with the GHSL NRES layer, under " +
    "the footprint mask. Every model is scored on the same frozen 71,497 buildings; our models are trained with one metro " +
    "held out at a time (GHSL is a global product and needs no training). The metric is the average of the residential " +
    "and non-residential absolute errors.");
}

// ---------- Research-path helpers ----------
const HEX = THEME.colors;
// width / height of each figure, so images are never stretched
const ASPECT = { "ae_rgb_cambridge.png": 1063 / 1147, "transfer_64d_vs_pls2.png": 2103 / 1047,
  "input_cnn.png": 985 / 1006, "input_vit.png": 991 / 1006, "neighbourhood_rings.png": 1,
  "s2_heuristics_denver.png": 2373 / 1102, "map_lagos.png": 3025 / 1150,
  "emb_umap_wrong_side.png": 2784 / 1153, "emb_gap_f1_india.png": 2063 / 762, "walks_phoenix_nyc.png": 2662 / 1011, "walks_country_city.png": 2618 / 913, "flow_examples_denver.png": 2618 / 913 };
function chartStyle(extra) {
  return Object.assign({
    titleFontSize: 13, titleColor: HEX.dk1, titleFontFace: "+mn-lt",
    catAxisLabelFontSize: 11, catAxisLabelColor: HEX.dk2, catAxisLabelFontFace: "+mn-lt",
    valAxisLabelFontSize: 10, valAxisLabelColor: HEX.dk2, valAxisLabelFontFace: "+mn-lt",
    dataLabelFontSize: 10, dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt",
    legendFontSize: 11, legendFontFace: "+mn-lt", legendColor: HEX.dk2,
    valGridLine: { color: "E2E2E2", size: 0.5 }, catGridLine: { style: "none" },
  }, extra);
}
// Column of stat callouts on a white card: [[big, label], ...]
function statStack(s, x, y, w, h, items, prefix) {
  card(s, x, y, w, h, `${prefix}-card`);
  const rowH = h / items.length;
  items.forEach(([big, label], i) => {
    s.addText([
      { text: big, options: { fontSize: 22, bold: true, color: i === 0 ? C.accent1 : C.text1, breakLine: true } },
      { text: label, options: { fontSize: 13, color: C.text2 } },
    ], { x: x + 0.25, y: y + i * rowH + 0.1, w: w - 0.45, h: rowH - 0.2, valign: "middle", margin: 0,
         paraSpaceAfter: 2, isTextBox: true, objectName: `${prefix}-stat-${i + 1}` });
    if (i > 0) {
      s.addShape(pres.shapes.LINE, { x: x + 0.25, y: y + i * rowH, w: w - 0.5, h: 0,
        line: { color: "E2E2E2", width: 0.75 }, objectName: `${prefix}-rule-${i}` });
    }
  });
}
function imageCard(s, file, x, y, w, h, alt, name, cap) {
  const capH = cap ? 0.55 : 0;
  card(s, x, y, w, h, `${name}-card`);
  const bw = w - 0.24, bh = h - 0.2 - capH, ar = ASPECT[file];
  let iw = bw, ih = bw / ar;
  if (ih > bh) { ih = bh; iw = bh * ar; }
  s.addImage({ path: FIG(file), x: x + 0.12 + (bw - iw) / 2, y: y + 0.1 + (bh - ih) / 2, w: iw, h: ih,
               altText: alt, objectName: name });
  if (cap) {
    s.addText(cap, { x: x + 0.2, y: y + h - capH - 0.05, w: w - 0.4, h: capH, fontSize: 12, italic: true,
                     color: C.text2, valign: "middle", margin: 0, isTextBox: true, objectName: `${name}-caption` });
  }
}

// ---------- Slide 6: Research path overview ----------
async function buildPathOverview() {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Research path" });
  s.addText("Research path: from 64 numbers to beating GHSL", { placeholder: "title" });
  const { FaTh, FaCompressArrowsAlt, FaBorderAll, FaProjectDiagram, FaBullseye, FaSatelliteDish } = require("react-icons/fa");
  const steps = [
    ["AlphaEarth", FaTh, HEX.accent1, "Is the signal there?", "0.78", "F1 on a held-out metro"],
    ["PLS", FaCompressArrowsAlt, HEX.accent1, "Which dimensions matter?", "0.70", "F1 on other metros (64-D: 0.59)"],
    ["CNN", FaBorderAll, HEX.accent2, "Do the building's surroundings help?", "0.75", "F1 vs 0.70 for one cell"],
    ["ViT", FaProjectDiagram, HEX.accent2, "Does a bigger model help?", "0.76", "F1, same as a simple model"],
    ["Rings", FaBullseye, HEX.accent3, "Does the neighbourhood help?", "0.241", "error, down from 0.288"],
    ["+ S2 & shape", FaSatelliteDish, HEX.accent3, "Do imagery and footprint shape help?", "0.185", "error; GHSL is 0.211"],
  ];
  const x0 = 0.6, gap = 0.15, w = (12.1 - 5 * gap) / 6, yH = 1.6, hH = 0.95, yB = 2.7, hB = 3.15;
  for (let i = 0; i < steps.length; i++) {
    const [name, Ico, col, q, big, lab] = steps[i];
    const x = x0 + i * (w + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: yH, w, h: hH, rectRadius: 0.08, fill: { color: col },
      line: { type: "none" }, objectName: `path-head-${i + 1}` });
    s.addImage({ data: await icon(Ico, "FFFFFF"), x: x + 0.15, y: yH + 0.14, w: 0.3, h: 0.3,
                 altText: name + " icon", objectName: `path-icon-${i + 1}` });
    s.addText(`${i + 1}`, { x: x + w - 0.5, y: yH + 0.1, w: 0.35, h: 0.35, fontSize: 14, bold: true,
      color: C.background1, align: "right", margin: 0, isTextBox: true, objectName: `path-num-${i + 1}` });
    s.addText(name, { x: x + 0.15, y: yH + 0.5, w: w - 0.3, h: 0.38, fontSize: 14, bold: true,
      color: C.background1, valign: "middle", margin: 0, isTextBox: true, objectName: `path-name-${i + 1}` });
    card(s, x, yB, w, hB, `path-body-${i + 1}`);
    s.addText(q, { x: x + 0.15, y: yB + 0.15, w: w - 0.3, h: 1.05, fontSize: 14, color: C.text1,
      valign: "top", margin: 0, isTextBox: true, objectName: `path-q-${i + 1}` });
    s.addText([
      { text: big, options: { fontSize: 28, bold: true, color: i === 5 ? C.accent1 : C.text1, breakLine: true } },
      { text: lab, options: { fontSize: 12, color: C.text2 } },
    ], { x: x + 0.15, y: yB + 1.35, w: w - 0.3, h: 1.65, valign: "top", margin: 0, isTextBox: true,
         objectName: `path-result-${i + 1}` });
    if (i < steps.length - 1) {
      s.addShape(pres.shapes.ISOSCELES_TRIANGLE, { x: x + w + 0.005, y: yH + hH / 2 - 0.07, w: 0.14, h: 0.12,
        rotate: 90, fill: { color: "9AA3AE" }, line: { type: "none" }, objectName: `path-arrow-${i + 1}` });
    }
  }
  banner(s, "Two scores", "Steps 1–4: F1, higher is better · Steps 5–6: error, lower is better", 6.1);
  s.addNotes(
    "This is the map for the next six slides. Each step asks one question. Steps 1 to 4 are classification " +
    "(residential vs non-residential), scored by macro-F1 with one metro held out at a time. Steps 5 and 6 switch to " +
    "the regression task (the residential and non-residential share of each building), scored by mean absolute error on " +
    "the frozen 71,497 buildings, where GHSL scores 0.211. The CNN number is from week 2 on a smaller, unbalanced " +
    "building set; the rest are from the balanced, locked evaluation sets.");
}

// ---------- Slide 7: Step 1, AlphaEarth 64-D ----------
function buildStepAE() {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Research path" });
  s.addText("Step 1: AlphaEarth 64-D finds the signal", { placeholder: "title" });
  imageCard(s, "ae_rgb_cambridge.png", 0.6, 1.55, 3.95, 4.35,
    "AlphaEarth embeddings for Cambridge MA shown as colour: first three principal components as RGB",
    "ae-rgb", "64 numbers per cell; the 3 strongest patterns shown as colour");
  const metros = ["Phoenix", "Atlanta", "Denver", "LA", "Houston", "Seattle", "Boston", "Chicago", "NYC", "SF"];
  const f1 = [0.906, 0.868, 0.867, 0.857, 0.768, 0.765, 0.719, 0.711, 0.673, 0.657];
  chartImage(s, "chart_ae_metro.png", 4.75, 1.55, 4.95, 4.35,
    "Macro-F1 per held-out metro, from 0.91 in Phoenix to 0.66 in SF, mean 0.78", "ae-chart");
  statStack(s, 9.95, 1.55, 2.75, 4.35, [
    ["0.78", "mean F1, one metro held out at a time"],
    ["0.91 → 0.66", "best (Phoenix) to worst (SF)"],
    ["1.75M", "balanced cells, logistic regression"],
  ], "ae");
  banner(s, "Takeaway", "The signal is real, but how well it works depends on the city.", 6.1);
  s.addNotes(
    "Each 10 m cell has 64 AlphaEarth numbers; the map shows the three strongest patterns as colour, so cells with " +
    "similar colours look alike to the model. A plain logistic regression on the 64 numbers, trained on nine metros and " +
    "tested on the tenth, reaches a mean macro-F1 of 0.78 (plus or minus 0.085) on 1.75M balanced pure cells. Phoenix, " +
    "Atlanta and Denver are easy (0.87 to 0.91); dense, mixed cities like NYC and SF are hard (0.66 to 0.67). Pooling all " +
    "held-out predictions gives 0.806, but the per-metro mean is the number to quote.");
}

// ---------- Slide 8: Step 2, PLS ----------
function buildStepPLS() {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Research path" });
  s.addText("Step 2: PLS keeps what travels between cities", { placeholder: "title" });
  imageCard(s, "transfer_64d_vs_pls2.png", 0.6, 1.55, 8.0, 4.35,
    "Two 10 by 10 grids of macro-F1 when training on one metro and testing on another: full 64-D (mean 0.59) and PLS with 2 dimensions (mean 0.70)",
    "pls-transfer", "Train on one metro (row), test on another (column). Darker = better.");
  statStack(s, 8.85, 1.55, 3.85, 4.35, [
    ["0.59 → 0.70", "F1 on other metros: 64-D vs 2 PLS dimensions"],
    ["0.735", "F1 from a single PLS dimension (64-D: 0.78)"],
    ["99.9% → 30%", "how often the metro can be guessed from the features"],
  ], "pls");
  banner(s, "Takeaway", "A few label-focused dimensions forget the city and keep the building.", 6.1);
  s.addNotes(
    "PLS (partial least squares) compresses the 64 numbers into a few directions chosen to predict the label. " +
    "Trained on one metro and tested on each of the other nine (90 pairs), two PLS dimensions average 0.70 macro-F1 " +
    "against 0.59 for the full 64-D model. One PLS dimension alone already scores 0.735 when one metro is held out. " +
    "Why it travels: from the raw 64 numbers a simple probe can tell which metro a cell is from 99.9% of the time; from " +
    "the two PLS dimensions only 30% (10 metros, so chance is 10%), while building F1 stays at 0.775. Caveat: the metro " +
    "probe uses a random pixel split inside metros, not a held-out metro.");
}

// ---------- Slide 9: Step 3, CNN ----------
function buildStepCNN() {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Research path" });
  s.addText("Step 3: a CNN reads the building's surroundings", { placeholder: "title" });
  imageCard(s, "input_cnn.png", 0.6, 1.55, 3.95, 4.35,
    "One Cambridge building as a 16 by 16 grid of AlphaEarth cells, footprint outlined in white, with a 3 by 3 filter",
    "cnn-input", "One building: its grid of AlphaEarth cells; white = footprint");
  const metros = ["Denver", "Boston", "NYC", "Chicago", "Phoenix", "Seattle", "SF", "LA", "Houston", "Atlanta"];
  const lr = [0.531, 0.648, 0.654, 0.754, 0.737, 0.777, 0.795, 0.697, 0.581, 0.786];
  const cnn = [0.754, 0.814, 0.735, 0.803, 0.777, 0.813, 0.798, 0.697, 0.576, 0.696];
  chartImage(s, "chart_cnn_metro.png", 4.75, 1.55, 4.95, 4.35,
    "Building F1 per metro, CNN vs one centre cell: CNN ahead in 6 metros, tied in 3, behind in Atlanta", "cnn-chart");
  statStack(s, 9.95, 1.55, 2.75, 4.35, [
    ["0.75 vs 0.70", "CNN vs one centre cell"],
    ["6 of 10", "metros clearly better; 3 tied, Atlanta worse"],
    ["0.186", "error with Sentinel-2 and footprint shape (week 13)"],
  ], "cnn");
  banner(s, "Takeaway", "Context around the building helps; the big jump came with footprint shape.", 6.1);
  s.addNotes(
    "Instead of averaging the cells, the CNN looks at the whole grid of AlphaEarth cells around each building (the " +
    "bounding box, padded to 16 by 16, with the footprint as a mask) and slides small filters across it. In week 2, on " +
    "about 11,000 buildings (unbalanced, one metro held out), it averaged 0.75 macro-F1 against 0.70 for a logistic " +
    "regression on the building's centre cell. Counting differences within 0.01 as ties, it was clearly better in 6 metros (biggest gains Denver and Boston), tied in 3 (Houston, LA, SF), and worse in Atlanta. " +
    "In week 13 the same idea, combined with the Sentinel-2 and footprint-shape features, reached a share error of 0.186 on " +
    "the locked regression set.");
}

// ---------- Slide 10: Step 4, ViT ----------
function buildStepViT() {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Research path" });
  s.addText("Step 4: a ViT adds capacity, not accuracy", { placeholder: "title" });
  imageCard(s, "input_vit.png", 0.6, 1.55, 3.95, 4.35,
    "The same building grid cut into 16 numbered patches, as a vision transformer sees it",
    "vit-input", "Same building, cut into 16 patches (tokens)");
  const models = ["Logistic regression", "PLS", "ViT", "3D ViT", "LR on full 10×10 grid"];
  const f1 = [0.761, 0.762, 0.756, 0.738, 0.700];
  chartImage(s, "chart_vit.png", 4.75, 1.55, 4.95, 4.35,
    "Macro-F1 on 100 m tiles: PLS 0.76, logistic regression 0.76, ViT 0.76, 3D ViT 0.74, grid LR 0.70", "vit-chart");
  statStack(s, 9.95, 1.55, 2.75, 4.35, [
    ["0.76 = 0.76", "ViT vs simple logistic regression"],
    ["0.55–0.59", "3D ViT on Chicago, SF, NYC (Atlanta 0.90)"],
    ["0.185", "error with S2 and footprint shape (week 14)"],
  ], "vit");
  banner(s, "Takeaway", "Bigger models learn what the embeddings already hold; hard cities stay hard.", 6.1);
  s.addNotes(
    "A vision transformer cuts the grid into patches and lets every patch attend to every other. On 100 m tiles " +
    "(52,492 balanced tiles, one metro held out) it scored 0.756 macro-F1, the same as a logistic regression on the " +
    "averaged cells (0.761) or PLS (0.762). The 3D ViT variants reached 0.73 to 0.74 and failed on the hard cities: " +
    "Chicago 0.55, SF 0.58, NYC 0.59, against Atlanta 0.90. In NYC the residential and non-residential tiles form a " +
    "single cloud in the model's own space, so capacity cannot separate them. Paired with the Sentinel-2 and footprint-shape features in week 14, a small " +
    "ViT on the building grid gave the best regression error so far, 0.185.");
}

// ---------- Slide 11: Steps 5-6, context + Sentinel-2 ----------
function buildStepContext() {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Research path" });
  s.addText("Steps 5–6: neighbourhood, then S2 and shape", { placeholder: "title" });
  imageCard(s, "neighbourhood_rings.png", 0.6, 1.55, 3.95, 4.35,
    "AlphaEarth cells around one building with rings at 50, 100 and 200 metres",
    "rings-map", "Rings at 50, 100 and 200 m around one building");
  const cards = [
    ["5  Neighbourhood rings", "For each ring, the average and spread of the AlphaEarth cells around the building. Tells the model what kind of block it sits in."],
    ["6  Sentinel-2 + footprint shape", "Summer 2018 imagery (brightness, blue, vegetation, texture) plus the Overture footprint's size and shape. Modelled on GHSL's own non-residential rules."],
  ];
  cards.forEach(([head, body], i) => {
    const y = 1.55 + i * 2.25;
    card(s, 4.75, y, 4.95, 2.1, `ctx-card-${i + 1}`, HEX.accent4);
    s.addText([
      { text: head, options: { bold: true, fontSize: 16, color: C.text1, breakLine: true } },
      { text: body, options: { fontSize: 14, color: C.text1 } },
    ], { x: 4.95, y: y + 0.15, w: 4.55, h: 1.8, valign: "top", margin: 0, paraSpaceAfter: 6,
         isTextBox: true, objectName: `ctx-text-${i + 1}` });
  });
  statStack(s, 9.95, 1.55, 2.75, 4.35, [
    ["0.288", "error, AlphaEarth alone"],
    ["0.241", "with neighbourhood rings"],
    ["0.195", "with S2 and shape: first result below GHSL (0.211)"],
  ], "ctx");
  banner(s, "Takeaway", "New information moved the error; swapping the model head did not.", 6.1);
  s.addNotes(
    "Step 5 adds context: for rings of 0 to 50, 50 to 100 and 100 to 200 metres around each building we add the mean " +
    "and spread of the AlphaEarth cells. That cut the share error from 0.288 to 0.241 and helped most on non-residential " +
    "buildings. Step 6 adds a second sensor: Sentinel-2 summer 2018 features (brightness, blue dominance, NDVI, " +
    "texture, footprint size and compactness) modelled on the rules GHSL itself uses to flag non-residential buildings. " +
    "AlphaEarth plus these 16 features reached 0.195, the first result below GHSL's 0.211. In part 3 we show that " +
    "most of that gain comes from the footprint size, not the imagery. Note also that from week 8 the models are trained " +
    "on buildings directly, while weeks 6-7 trained on pixels and averaged them, which is part of the drop from 0.288.");
}

// ---------- Slide 12: progression ----------
function buildProgression() {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Beating GHSL" });
  s.addText("The road past GHSL: error at each step", { placeholder: "title" });
  const labels = ["AE Ridge", "AE PLS", "AE XGBoost", "+ Rings", "+ S2 & shape", "CNN + S2 & shape", "ViT + S2 & shape"];
  const err = [0.288, 0.290, 0.287, 0.241, 0.195, 0.186, 0.185];
  chartImage(s, "chart_progression.png", 0.6, 1.55, 8.6, 4.35,
    "Share error per building at each step, falling below GHSL 0.211 only with Sentinel-2 and footprint shape", "prog-chart");
  statStack(s, 9.45, 1.55, 3.25, 4.35, [
    ["−12%", "error vs GHSL (0.185 vs 0.211)"],
    ["7 of 10", "metros where the ViT clearly beats GHSL; LA is the one loss"],
    ["0.200", "scaled-up XGBoost (week 15): bigger didn't help"],
  ], "prog");
  banner(s, "Measured on", "71,497 frozen buildings · one metro held out · error = ½ (res + non-res)", 6.1);
  s.addNotes(
    "All bars are scored on the same frozen 71,497 buildings with one metro held out at a time. AlphaEarth alone sits " +
    "at 0.287 to 0.290 whatever the model head (Ridge, PLS, XGBoost), well above GHSL's 0.211. Neighbourhood rings bring " +
    "it to 0.241. Sentinel-2 and footprint-shape features take it below GHSL to 0.195, and putting the building's AlphaEarth grid through a " +
    "CNN or ViT alongside those features gives 0.186 and 0.185, about 12% lower error than GHSL; the next slide shows it metro by metro. The texture-fixed rerun of week 9 scores 0.193 (was 0.195). " +
    "Scaling XGBoost up on rings plus Sentinel-2 (week 15) gave 0.200, so size alone did not help. These are building-level " +
    "results; at 100 m tiles GHSL still wins once both are scored on the same tiles.");
}

// ---------- Slide 13: the GHSL cues we borrowed for Sentinel-2 ----------
function buildHeuristics() {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Beating GHSL" });
  s.addText("The Sentinel-2 cues we borrowed from GHSL", { placeholder: "title" });
  imageCard(s, "s2_heuristics_denver.png", 0.6, 1.55, 7.55, 4.35,
    "Denver: Sentinel-2 true colour with residential and non-residential outlines, and each building coloured by the GHSL-style non-residential score",
    "s2-map", "Denver: the bright commercial strip scores high, but so do big apartment blocks");
  // separation of each cue on the frozen 71,497 buildings (AUC, residential vs non-residential)
  const cues = ["Footprint area", "Vegetation (NDVI)", "Blue", "Brightness", "Fine texture",
                "Combined score", "Blue share", "Footprint compactness", "Texture ratio"];
  const auc = [0.81, 0.80, 0.74, 0.74, 0.73, 0.73, 0.63, 0.51, 0.50];
  const cols = [HEX.accent5, HEX.accent1, HEX.accent1, HEX.accent1, HEX.accent1, HEX.dk1, HEX.accent1, HEX.accent5, HEX.accent1];
  chartImage(s, "chart_cues.png", 8.4, 1.55, 4.3, 4.35,
    "AUC of each cue alone: footprint area 0.81 highest, compactness 0.51 lowest", "cue-chart");
  banner(s, "The score", "Six 0–1 cues averaged: bright, bluish, smooth, coarse texture, large, compact", 6.1);
  s.addNotes(
    "GHSL flags non-residential buildings with rules about how they look from space (Pesaresi et al. 2024). We copied " +
    "the idea with Sentinel-2 summer 2018 imagery (bands blue, green, red, near-infrared; scenes under 20% cloud, " +
    "mosaicked per metro). For each building we read the Sentinel-2 pixels under its bounding box (at least 8 by 8 " +
    "pixels, so small houses also see their surroundings) and compute: mean of each band, brightness, blue share, NDVI, " +
    "fine texture (spread of brightness), a coarse texture and their ratio, plus three numbers from the Overture " +
    "footprint itself: area, equivalent diameter and compactness. A hand-built score averages six cues, each scaled 0 to " +
    "1: bright roofs, bluish roofs, smooth roofs, coarse-over-fine texture, large footprint, compact footprint. " +
    "The chart shows how well each cue alone separates residential from non-residential on the frozen 71,497 buildings " +
    "(AUC; 0.5 is a coin flip), using the texture-fixed features. Blue dots come from the footprint, not the imagery. " +
    "Footprint area is the strongest single cue (0.81); vegetation is next (0.80, residential is greener). The combined " +
    "score (0.78) is weaker than area alone. Compactness (0.51) carries no signal. The texture ratio was broken in the " +
    "first run (coarse texture came from every other pixel, so it scored 0.50); fixed to 2x2 block means it scores 0.70. " +
    "On the map, large apartment blocks " +
    "score like warehouses because size and compactness push them up.");
}

// ---------- Slide 14: ablation ----------
function buildAblation() {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Beating GHSL" });
  s.addText("Look closer: footprint shape did most of the work", { placeholder: "title" });
  const rows = [
    ["AlphaEarth only", 0.231, "9AA3AE"],
    ["Sentinel-2 imagery only", 0.266, "9AA3AE"],
    ["GHSL-style score only", 0.308, "9AA3AE"],
    ["Footprint shape only (3 numbers)", 0.228, HEX.accent5],
    ["AlphaEarth + imagery", 0.224, "9AA3AE"],
    ["AlphaEarth + footprint shape", 0.201, HEX.accent1],
    ["AlphaEarth + all 16 (week 9)", 0.195, HEX.accent1],
    ["GHSL (baseline)", 0.211, HEX.dk1],
  ];
  chartImage(s, "chart_ablation.png", 0.6, 1.55, 7.9, 4.35,
    "Error by feature group: AlphaEarth + footprint shape 0.201 and all 16 features 0.193 beat GHSL 0.211; AlphaEarth + imagery 0.221 does not", "abl-chart");
  statStack(s, 8.75, 1.55, 3.95, 4.35, [
    ["0.231 → 0.201", "adding 3 footprint numbers to AlphaEarth"],
    ["0.201 → 0.193", "adding the 10 imagery features on top"],
    ["0.221", "AlphaEarth + imagery, no footprint: still worse than GHSL"],
  ], "abl");
  banner(s, "So what", "The win over GHSL rests on building footprints; the imagery adds little.", 6.1);
  s.addNotes(
    "We re-ran the week 9 model with different feature groups: same XGBoost settings, same 86,158 training buildings, " +
    "one metro held out at a time, scored on the frozen 71,497 buildings. These numbers use the texture-fixed week 9 features (Nov 17 run); the full set scores 0.193 (0.195 before the fix). " +
    "AlphaEarth alone, trained per building, scores 0.231, better than the 0.287 from weeks 6-7, which trained on pixels " +
    "and averaged them. Three footprint numbers alone (area, diameter, compactness) score 0.228, slightly better than all " +
    "64 AlphaEarth numbers. AlphaEarth plus footprint shape reaches 0.201, already below GHSL's 0.211. The ten imagery " +
    "features add only 0.008 on top, and AlphaEarth plus imagery without the footprint (0.221) stays above GHSL. " +
    "Implications: the win needs a building footprint for every building, which Overture provides in many places but " +
    "with uneven quality, and GHSL does not use per-building footprints, so the comparison is not like for like. Open " +
    "question: week 8's rings (0.241, Ridge) should also be compared against this per-building AlphaEarth baseline. " +
    "Source: presentation/analysis_s2_ablation.py, results in presentation/figures/texture_fix/s2_ablation.csv.");
}

// ---------- Slide: XGBoost ----------
function compareCard(s, x, y, w, h, head, a, b, note, name) {
  card(s, x, y, w, h, name);
  s.addText(head, { x: x + 0.25, y: y + 0.15, w: w - 0.5, h: 0.35, fontSize: 14, bold: true, color: C.accent1,
                    margin: 0, valign: "middle", isTextBox: true, objectName: `${name}-head` });
  [a, b].forEach(([lab, val], i) => {
    const cx = x + 0.25 + i * (w - 0.5) / 2;
    s.addText([
      { text: val, options: { fontSize: 28, bold: true, color: i === 1 ? C.accent2 : C.text1, breakLine: true } },
      { text: lab, options: { fontSize: 12, color: C.text2 } },
    ], { x: cx, y: y + 0.5, w: (w - 0.5) / 2 - 0.1, h: h - 0.6, valign: "top", margin: 0, isTextBox: true,
         objectName: `${name}-val-${i + 1}` });
  });
  if (note) s.addText(note, { x: x + 0.25, y: y + h - 0.45, w: w - 0.5, h: 0.35, fontSize: 12, italic: true, color: C.text2,
                    margin: 0, valign: "middle", isTextBox: true, objectName: `${name}-note` });
}

function buildStepXGB() {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Research path" });
  s.addText("XGBoost: a stronger head alone changed nothing", { placeholder: "title" });
  const w = 3.75, ch = 1.38, g = 0.105;
  compareCard(s, 0.6, 1.55, w, ch, "Wk 7 · cells · F1, higher better",
    ["Logistic regression", "0.78"], ["XGBoost", "0.75"], null, "xgb-c1");
  compareCard(s, 0.6, 1.55 + ch + g, w, ch, "Wk 7 · buildings · error, lower better",
    ["Ridge", "0.288"], ["XGBoost", "0.287"], null, "xgb-c2");
  compareCard(s, 0.6, 1.55 + 2 * (ch + g), w, ch, "Wk 15 · 6× bigger · error",
    ["300 trees", "0.201"], ["1,800 trees", "0.200"], null, "xgb-c3");
  chartImage(s, "chart_xgb_metro.png", 4.6, 1.55, 8.1, 4.35,
    "Macro-F1 per metro, week 12 XGBoost classifier vs GHSL: ahead in 5 metros, tied in 2, behind in 3", "xgb-metro-chart");
  banner(s, "Takeaway", "XGBoost is a good workhorse, but every gain came from new inputs, not the head.", 6.1);
  s.addNotes(
    "Week 7 swapped the Ridge head for XGBoost (300 trees, depth 6, learning rate 0.05, on GPU) with everything else " +
    "fixed: trained on the 1.75M balanced cells, one metro held out. As a classifier it scored 0.751 mean macro-F1, " +
    "below logistic regression's 0.779 (SF fell to 0.53). As a share regressor, averaged over each building's cells, " +
    "it scored 0.287 against Ridge's 0.288 on the frozen 71,497 buildings, and 0.267 vs 0.266 on the 100 m tiles, still " +
    "far behind GHSL. Week 15 scaled XGBoost about 6 times (1,800 trees, learning rate 0.025, a sample of 17% of features " +
    "per tree) for the 464 rings plus Sentinel-2 and shape features: 0.200 against 0.201 for the standard recipe, and " +
    "worse than the simpler 80-feature AE plus S2 and shape model (0.195). The one XGBoost win is week 12: a classifier " +
    "on AlphaEarth plus the 16 Sentinel-2 and footprint-shape features scores 0.838 macro-F1 against GHSL's 0.822 on " +
    "the locked buildings (label = whichever soft share is larger; about 4% of buildings are mixed). Counting differences within 0.01 as ties, it is clearly ahead " +
    "in 5 metros (biggest in Seattle and Boston), tied in Houston and LA, and behind in Denver, NYC and SF. As part 3 shows, much of " +
    "that comes from footprint shape, not the model.");
}

// ---------- Slide: per-metro regression vs GHSL ----------
function buildMetroRegression() {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Beating GHSL" });
  s.addText("Metro by metro: the ViT wins clearly in seven", { placeholder: "title" });
  chartImage(s, "chart_vit_metro.png", 0.6, 1.55, 12.1, 4.35,
    "Residential and non-residential share error per metro, ViT + S2 & shape vs GHSL, with the margin per metro", "vit-metro-chart");
  banner(s, "Takeaway", "Biggest gains in Phoenix, Boston, Atlanta and Houston; NYC and Denver tie, LA loses.", 6.1);
  s.addNotes(
    "Each row is a held-out metro, shaded by US region as in the project canvas. Blue dots are the week 14 ViT on the " +
    "AlphaEarth grid plus Sentinel-2 and footprint-shape features, orange dots are GHS-BUILT-S, for the residential " +
    "share (left) and non-residential share (right). The margin is the ViT's mean error minus GHSL's: negative means " +
    "the ViT is better. Counting differences within 0.01 as ties, the ViT is clearly better in 7 metros (largest in " +
    "Phoenix, -0.099, then Boston, Atlanta and Houston at about -0.08), ties NYC (+0.002) and Denver (-0.004), and " +
    "loses LA (+0.020). Denver is GHSL's best metro; there GHSL still has the lower non-residential error.");
}

// ================= Generalisation section (charts from make_charts.py; numbers from week4/5 runs, as in the
// weeks 1-5 stakeholder canvas, notebooks/ghsl_built_s_vs_ae/"AlphaEarth Weeks 1–5 Canvas.html")
function genSlide(title, chart, alt, name, stats, bannerLabel, bannerText, notes, first) {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Generalisation" });
  s.addText(title, { placeholder: "title" });
  chartImage(s, chart, 0.6, 1.55, 8.1, 4.35, alt, name);
  statStack(s, 8.95, 1.55, 3.75, 4.35, stats, name);
  banner(s, bannerLabel, bannerText, 6.1);
  s.addNotes(notes);
  return s;
}

function buildGeneralisation() {
  genSlide("Generalisation: farther from home, lower scores", "chart_gen_ladder.png",
    "Macro-F1 for USA, Japan and India at three distances: random split, held-out city, city to city. India falls from 0.71 to 0.45",
    "gen-ladder", [
      ["0.80 → 0.77", "Japan: random split vs a held-out city"],
      ["0.71 → 0.52", "India: the same step, close to chance"],
      ["0.44 → 0.52", "India with 5 vs 10 training cities; the gap stays"],
    ], "Takeaway", "The US and Japan share one residential signal across cities; India does not.",
    "Three tests of the same 64-D logistic regression, 10 cities per country, balanced cells. A random 80/20 split " +
    "inside a country is the easy case: USA 0.89, Japan 0.80, India 0.71. Holding out a whole city is the realistic " +
    "within-country test: USA 0.86, Japan 0.77, India 0.52. Training on a single city and testing on each other city is " +
    "hardest: USA 0.67, Japan 0.73, India 0.45. Japan barely drops, so its residential signal is shared across cities; " +
    "India falls to near chance. Going from 5 to 10 Indian cities lifted the held-out-city score from 0.44 to 0.52 but " +
    "left the gap. The same lesson holds inside the US: a pooled random split overstates quality by 0.03 to 0.06 " +
    "(week 4: 0.806 pooled vs 0.779 per held-out metro). Note the US 10-city panel uses 30 km disks, not the downtown " +
    "boxes of the earlier slides, so 0.86 is not comparable with 0.78.", true);

  genSlide("Across countries, transfer runs one way", "chart_gen_transfer.png",
    "Three-by-three grids of macro-F1 when training in one country and testing in another, for the full 64-D head and for 2 PLS dimensions",
    "gen-transfer", [
      ["0.72 vs 0.65", "Japan → USA vs USA → Japan"],
      ["0.37 · 0.33", "USA → India · India → Japan"],
      ["0.37 → 0.55", "USA → India with 2 PLS dimensions"],
    ], "Takeaway", "No single decision line works in both directions; low rank softens the worst pairs.",
    "Train on all cities of one country, test on another (rows train, columns test; the outlined diagonal is an " +
    "80/20 split within the country). Transfer is asymmetric: a Japan-trained head keeps 0.72 on the US, while a " +
    "US-trained head drops to 0.65 on Japan and 0.37 on India, and an India-trained head scores 0.33 on Japan. With two " +
    "PLS dimensions the worst pairs improve (USA to India 0.37 to 0.55, India to USA 0.45 to 0.76, India to Japan 0.33 " +
    "to 0.60) at a small cost on the diagonal. This run used 10 US, 5 Japanese and 5 Indian cities.");

  genSlide("When transfer fails, the cut-off fails first", "chart_gen_threshold.png",
    "AUROC and macro-F1 at a 0.5 cut-off for six train-test country pairs; AUROC stays 0.61 to 0.81 while F1 falls as low as 0.33",
    "gen-threshold", [
      ["0.69 vs 0.33", "India → Japan: ranking vs decision"],
      ["0.00", "non-residential F1: everything is called residential"],
      ["Not yet tested", "a target-country cut-off or prior shift"],
    ], "Next step", "Recalibrate the cut-off on the target country before retraining anything.",
    "AUROC asks whether the model still ranks residential above non-residential; F1 at 0.5 asks whether the default " +
    "cut-off gives the right decision. Across countries the ranking mostly survives (AUROC 0.61 to 0.81) but the decision " +
    "collapses: India to Japan has AUROC 0.69 yet F1 0.33, because non-residential F1 is 0.00 and every building is " +
    "called residential. USA to India shows the same pattern (AUROC 0.61, non-residential F1 0.07). That points to a " +
    "shift in the base rate or calibration rather than a missing signal, so the cheap first fix is to set the cut-off, " +
    "or the class prior, on a small labelled sample from the target country. This has not been tested yet.");

  genSlide("India: two PLS dimensions rescue the worst cities", "chart_gen_india.png",
    "Held-out Indian city macro-F1, full 64-D vs 2 PLS dimensions; Kolkata +0.25, Chennai +0.24, Bengaluru +0.13, Jaipur -0.01",
    "gen-india", [
      ["0.52 → 0.60", "held-out Indian city F1, 64-D vs PLS-2"],
      ["0.45 → 0.54", "Indian city-to-city transfer"],
      ["8 of 10", "Japanese cities where the full 64-D is still better"],
    ], "Takeaway", "Low rank repairs India and is neutral or worse elsewhere: choose the head per country.",
    "Leave-one-city-out across 10 Indian cities. Compressing the 64 AlphaEarth numbers to two PLS dimensions fixes the " +
    "cities where the full head fails: Kolkata 0.37 to 0.62, Chennai 0.39 to 0.63, Bengaluru 0.45 to 0.59. Eight cities " +
    "improve by more than 0.01, Lucknow is level and Jaipur drops by 0.01. City-to-city transfer inside India rises from " +
    "0.45 to 0.54. The opposite happens in Japan: the full head beats PLS-2 in 8 of 10 cities and matches PLS-9 overall " +
    "(0.77); Sapporo is the outlier (0.59, and 0.67 with PLS-9).");

  genSlide("Unseen countries: a shared low-rank head helps", "chart_gen_loco.png",
    "Leave-one-country-out macro-F1 for four heads: full 64-D 0.56 mean, task-invariant subspace 0.71 mean; Japan +0.22, India +0.25, USA -0.04",
    "gen-loco", [
      ["0.56 → 0.71", "mean F1 on a country never seen in training"],
      ["+0.22 · +0.25", "Japan · India; the USA loses 0.04"],
      ["0.007", "gain over a label-shuffle baseline"],
    ], "Caveat", "The lift is real, but we can't yet claim one shared 'residential' direction.",
    "The strictest test: the target country is never seen in training. 2.0M cells balanced by degree of urbanisation " +
    "(1:1:1 urban, peri-urban, rural, and 1:1 residential within each), 5 seeds, subspace rank 3. Mean macro-F1 rises from " +
    "0.56 for the full 64-D head to 0.66 for pooled PLS, 0.68 for a geometric consensus of per-country subspaces and " +
    "0.71 for a task-invariant subspace. The gains come on Japan (+0.22) and India (+0.25); the USA loses 0.04. But the " +
    "shared-subspace spectrum does not beat a label-shuffle null (excess task invariance 0.007), and a matched local " +
    "residual beats a mismatched one by only 0.006 to 0.020. So the improvement may be generic low-rank regularisation " +
    "rather than a shared definition of 'residential'.");

  // ---- recommendations
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Generalisation" });
  s.addText("Generalisation: what we recommend", { placeholder: "title" });
  const recs = [
    ["Judge by the worst metro, not the mean", "A 0.78 mean hides a 0.66 metro (SF). Set a per-metro floor as the acceptance test."],
    ["Test dense mixed-use cores on their own", "NYC, SF, Chicago and Boston are the hardest metros for every model we tried."],
    ["Recalibrate per country before retraining", "Ranking transfers (AUROC 0.61–0.81); the 0.5 cut-off does not."],
    ["Use a low-rank head for a new country", "PLS-2 or a shared subspace: +0.25 in India, neutral to slightly worse in the USA."],
  ];
  const cw = (12.1 - 0.3) / 2, chh = 2.05;
  recs.forEach(([head, body], i) => {
    const x = 0.6 + (i % 2) * (cw + 0.3), y = 1.55 + Math.floor(i / 2) * (chh + 0.25);
    card(s, x, y, cw, chh, `rec-card-${i + 1}`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.3, w: 0.5, h: 0.5, fill: { color: C.text1 },
      line: { type: "none" }, objectName: `rec-dot-${i + 1}` });
    s.addText(String(i + 1), { x: x + 0.3, y: y + 0.3, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: C.background1,
      align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `rec-num-${i + 1}` });
    s.addText([
      { text: head, options: { fontSize: 18, bold: true, color: C.text1, breakLine: true } },
      { text: body, options: { fontSize: 14, color: C.text2 } },
    ], { x: x + 1.0, y: y + 0.28, w: cw - 1.3, h: chh - 0.5, valign: "top", margin: 0, paraSpaceAfter: 6,
         isTextBox: true, objectName: `rec-text-${i + 1}` });
  });
  banner(s, "Bottom line", "Results inside one country don't predict a new one: validate locally before use.", 6.1);
  s.addNotes(
    "Four actions from the generalisation results. One: report the worst metro, not just the mean, because the 0.78 " +
    "mean in the US hides 0.66 in SF and 0.67 in NYC. Two: dense mixed-use cores (NYC, SF, Chicago, Boston) are the " +
    "hardest for the linear head, the CNN and the ViT alike, so validate them as their own slice. Three: across " +
    "countries the model still ranks buildings well but the default cut-off fails, so recalibrate on a small target " +
    "sample before retraining; this is untested. Four: for a new country, start from a low-rank head (PLS-2 or the " +
    "task-invariant subspace), which repaired India and cost little in the USA, while keeping the full head where it " +
    "already works, as in Japan.");
}

// ================= Africa section (outputs/ghsl_built_s_vs_ae_intl/africa_*, africa_loro_vs_ghsl, africa_clf_vs_ghsl)
function africaSlide(title, chart, alt, stats, prefix, bannerLabel, bannerText, notes, first) {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Africa" });
  s.addText(title, { placeholder: "title" });
  chartImage(s, chart, 0.6, 1.55, 8.1, 4.35, alt, `${prefix}-chart`);
  statStack(s, 8.95, 1.55, 3.75, 4.35, stats, prefix);
  banner(s, bannerLabel, bannerText, 6.1);
  s.addNotes(notes);
}

// four numbered recommendation cards, same style as the generalisation recommendations slide
function recCards(section, title, items, bannerLabel, bannerText, notes, prefix) {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  const cw = (12.1 - 0.3) / 2, chh = 2.05;
  items.forEach(([head, body], i) => {
    const x = 0.6 + (i % 2) * (cw + 0.3), y = 1.55 + Math.floor(i / 2) * (chh + 0.25);
    card(s, x, y, cw, chh, `${prefix}-card-${i + 1}`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.3, w: 0.5, h: 0.5, fill: { color: C.text1 },
      line: { type: "none" }, objectName: `${prefix}-dot-${i + 1}` });
    s.addText(String(i + 1), { x: x + 0.3, y: y + 0.3, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: C.background1,
      align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `${prefix}-num-${i + 1}` });
    s.addText([
      { text: head, options: { fontSize: 18, bold: true, color: C.text1, breakLine: true } },
      { text: body, options: { fontSize: 14, color: C.text2 } },
    ], { x: x + 1.0, y: y + 0.28, w: cw - 1.3, h: chh - 0.5, valign: "top", margin: 0, paraSpaceAfter: 6,
         isTextBox: true, objectName: `${prefix}-text-${i + 1}` });
  });
  banner(s, bannerLabel, bannerText, 6.1);
  s.addNotes(notes);
}

async function buildAfrica() {
  africaSlide("Africa: GHSL misses most non-residential buildings", "chart_africa_bias.png",
    "Share of non-residential cells caught per African city: GHSL 37% in Lagos to 53% in Johannesburg; AlphaEarth 78% to 94%",
    [["37%", "of Lagos non-residential cells caught by GHSL (AlphaEarth 94%)"],
     ["73–81%", "of cells GHSL calls residential; the truth is 50%"],
     ["0.65 vs 0.74", "GHSL F1 in Lagos vs on US cells"]],
    "afr1", "Why it matters", "GHSL is a weak baseline, and a poor stand-in for labels, in African cities.",
    "Five African cities (Cairo, Accra, Nairobi, Lagos, Johannesburg), 365,506 balanced pure 10 m cells labelled " +
    "from Overture tags, half residential and half non-residential. GHSL is read at each AlphaEarth cell centre and " +
    "turned into a class by whichever of its residential or non-residential shares is larger. GHSL calls 73% to 81% " +
    "of these cells residential, so it catches almost every residential cell but only 37% (Lagos) to 53% " +
    "(Johannesburg) of the non-residential ones. The same GHSL method scores 0.74 macro-F1 on our US cells; in Lagos " +
    "it scores 0.65 and in Cairo 0.69. AlphaEarth here is trained inside each city (next slides), catching 78% to 94%.", true);

  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Africa" });
  s.addText("Lagos: GHSL calls the commercial strip residential", { placeholder: "title" });
  imageCard(s, "map_lagos.png", 0.6, 1.55, 12.1, 4.35,
    "Lagos, 900 m window: true labels, GHSL and AlphaEarth predictions on 616 labelled 10 m cells; wrong cells outlined. GHSL wrong on 26%, AlphaEarth 15%.",
    "lagos-map", null);
  banner(s, "Read it", "Outlined cells are wrong. GHSL misses whole non-residential blocks; AlphaEarth's errors are scattered.", 6.1);
  s.addNotes(
    "A 900 by 900 metre window in Lagos with 616 labelled cells, 64% residential, chosen because it has both classes " +
    "and its error rates are close to the city-wide ones (not the most extreme window). Left: the Overture labels. " +
    "Middle: GHSL, wrong on 26% of cells, almost all of them a non-residential block it calls residential. Right: " +
    "AlphaEarth trained on other Lagos cells (5-fold), wrong on 15%, mostly single residential cells called " +
    "non-residential. Grey areas have no labelled buildings; only about 1% of Lagos buildings carry an Overture use tag.");

  africaSlide("Trained in the same city, AlphaEarth wins all five", "chart_africa_within.png",
    "Macro-F1 per African city, AlphaEarth trained within the city vs GHSL: Lagos 0.88 vs 0.65, Cairo 0.85 vs 0.69",
    [["+0.24", "Lagos: 0.88 vs 0.65"],
     ["5 of 5", "cities where AlphaEarth is clearly ahead"],
     ["Same city", "5-fold splits inside each city: optimistic"]],
    "afr3", "Caveat", "This test uses labels from the same city; the next slide holds a whole city out.",
    "AlphaEarth (64-D Ridge on the cell fractions) is trained and tested with 5-fold splits inside each city, so it " +
    "has seen labels from the same city. It beats GHSL everywhere: Lagos +0.24, Cairo +0.16, Johannesburg +0.12, " +
    "Nairobi +0.05, Accra +0.02. The dashed line is GHSL on our US cells (0.74), so GHSL is below its US level in " +
    "Lagos, Cairo and Nairobi. Adding the week 9 Sentinel-2 features changes the share error by less than 0.004, so " +
    "the result does not depend on them. Because the split is within a city, this is an upper bound for a new city.");

  africaSlide("On a new city, GHSL still wins", "chart_africa_loro.png",
    "Macro-F1 per held-out city in Egypt and Nigeria, AlphaEarth XGBoost vs GHSL: AlphaEarth ahead in 4, GHSL in 5, 1 tie",
    [["0.59 vs 0.68", "Egypt: AlphaEarth vs GHSL"],
     ["0.60 vs 0.67", "Nigeria: AlphaEarth vs GHSL"],
     ["~1%", "of Lagos buildings carry an Overture use tag"]],
    "afr4", "Takeaway", "In Africa, AlphaEarth beats GHSL only when it has local labels.",
    "Leave-one-city-out within Egypt (Alexandria, Aswan, Cairo, Luxor, Port Said) and Nigeria (Abuja, Ibadan, Kano, " +
    "Lagos, Port Harcourt): the model never sees the test city. Country-level macro-F1: Egypt 0.59 vs GHSL 0.68, " +
    "Nigeria 0.60 vs 0.67, and the share error tells the same story (Egypt 0.295 vs 0.238, Nigeria 0.244 vs 0.229). " +
    "Per city it is mixed: AlphaEarth wins big in Port Said (+0.35) and slightly in Luxor, Lagos and Abuja; GHSL wins " +
    "big in Kano (+0.31) and Alexandria (+0.16). Small cities make some rows noisy (Aswan has 312 test cells, Luxor " +
    "842). The gap with the previous slide is the value of local labels, and labels are scarce: about 1% of Lagos " +
    "buildings carry an Overture use tag.");

  recCards("Africa", "Africa: what we recommend", [
    ["Don't use GHSL as truth for non-residential", "It misses 47–63% of non-residential cells in the five cities."],
    ["Collect a small local label set per city", "With in-city labels AlphaEarth wins in all five cities, by up to +0.24."],
    ["Close the label gap first", "Only ~1% of Lagos buildings have an Overture use tag; add Places and local surveys."],
    ["Hold out whole cities in more countries", "Only Egypt and Nigeria so far; add Ghana, Kenya and South Africa."],
  ], "Bottom line", "In Africa the bottleneck is labels, not embeddings.",
  "Four actions from the Africa results. GHSL's non-residential layer is unreliable in these cities, so it should " +
  "not be used as a label source or as the only baseline. AlphaEarth wins clearly once it has labels from the city, " +
  "so a modest local labelling effort per city is likely the highest-value step. Labels are the bottleneck: Overture " +
  "use tags are almost absent in Lagos, so other sources (Places, local surveys) are needed. Finally, the held-out-" +
  "city test covers only Egypt and Nigeria; Ghana, Kenya and South Africa should get the same test.", "afr-rec");
}

// ================= Embedding structure (figures taken as-is from the weeks 1-5 canvas report, "Embedding dynamics")
function buildEmbedding() {
  let s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Inside the embedding" });
  s.addText("Embedding: NYC's classes overlap, Phoenix's separate", { placeholder: "title" });
  imageCard(s, "emb_umap_wrong_side.png", 0.6, 1.55, 12.1, 4.35,
    "UMAP of balanced AlphaEarth cells for each of 10 US metros with a 2D logistic cut; wrong-side cells coloured. NYC 43% wrong, Phoenix 20%.",
    "emb-umap", null);
  banner(s, "Takeaway", "20% (Phoenix) to 43% (NYC) of cells sit on the wrong side of a simple cut; mean 31%.", 6.1);
  s.addNotes(
    "Each panel is a 2D UMAP of the balanced AlphaEarth cells for one metro (week 4). The background shows a 2D " +
    "logistic regression cut; grey points are on the correct side, coloured points are on the wrong side, coloured by " +
    "their true class. The wrong-side share runs from 20% in Phoenix to 43% in NYC, mean 31%: NYC 43, Boston 34, " +
    "Denver 33, Houston 33, Atlanta 31, Chicago 30, SF 30, Seattle 29, LA 25, Phoenix 20. The dense, mixed-use metros " +
    "are where residential and non-residential cells sit on top of each other in the embedding, which matches the " +
    "metros where every model scores lowest. UMAP distorts distances, so read this as a picture of overlap, not a " +
    "measurement.");

  s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Inside the embedding" });
  s.addText("One axis carries the signal; the rest encode place", { placeholder: "title" });
  imageCard(s, "emb_gap_f1_india.png", 0.6, 1.55, 8.4, 4.35,
    "Left: held-out cluster gap vs macro-F1 for 10 US metros, Pearson r 0.96. Right: class separation on PLS axis 1 for five Indian cities, fit inside the city vs projected through the other four cities' fit.",
    "emb-gap", null);
  statStack(s, 9.25, 1.55, 3.45, 4.35, [
    ["0.735 vs 0.779", "one PLS direction vs all 64, held-out metro F1"],
    ["r = 0.96", "cluster gap vs F1 across 10 metros"],
    ["30–63%", "of India's in-city separation lost in another city's fit"],
  ], "emb");
  banner(s, "Takeaway", "The first direction is shared; later ones mostly encode where a cell is.", 6.1);
  s.addNotes(
    "A single supervised PLS direction gives a held-out-metro F1 of 0.735 against 0.779 for all 64 dimensions, and an " +
    "L1 model on 32 PLS scores keeps about 1.2 of them, always the first. Left panel: for the ViT v4 held-out tiles, " +
    "the gap between the class centres divided by the class spread tracks F1 across the 10 metros (r = 0.96, n = 10): " +
    "NYC, SF and Chicago sit near 'one cloud', Atlanta, Phoenix and Denver past a clear gap. Right panel: Indian " +
    "cities separate well on PLS axis 1 fitted inside the city, but lose 30% to 63% of that separation when projected " +
    "through the other four cities' fit. Later components encode geography: from 1 to 20 PLS components India gains " +
    "0.08 within the country and loses 0.20 across countries (USA +0.03 / -0.14, Japan +0.05 / +0.02). At k = 2, " +
    "label-shuffled subspaces sit further apart than real ones, so aligning cities needs real task signal. Figures are " +
    "taken unchanged from the weeks 1-5 canvas report.");
}

// ---------- Boundary walks (data and image chips from the embedding-globe app, week4 viz/usa_embedding_globe)
function buildWalks() {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Inside the embedding" });
  s.addText("Boundary walks: residential to non-residential", { placeholder: "title" });
  imageCard(s, "walks_phoenix_nyc.png", 0.6, 1.55, 12.1, 4.35,
    "Twelve-step walks from the residential to the non-residential centre of the embedding in Phoenix and New York, each step shown as the nearest real 10 m cell's aerial image with its true label and the model's P(residential).",
    "walks", null);
  banner(s, "Takeaway", "Phoenix crosses cleanly; in NYC real non-residential cells sit inside the residential zone.", 6.1);
  s.addNotes(
    "These come from the boundary-walk view of our embedding globe app (week4/international_geo_generalization/" +
    "viz/usa_embedding_globe, served locally on port 8766). For each city and urbanisation level we take the centre " +
    "of cells the model calls residential with high confidence and the centre of confidently non-residential cells, " +
    "in PLS space fitted with that city held out, and walk a straight line between them in 12 steps. Each frame is " +
    "the nearest real 10 m cell to that point (cosine distance on 32 PLS dimensions), shown as its Esri aerial image " +
    "with surroundings. The border is that cell's true label; the number is the model's P(residential) at the step " +
    "(logistic regression on 3 PLS dimensions); the dashed line is where it crosses 50%. Phoenix: suburban houses " +
    "with pools give way to flat commercial roofs, and the label flips once, one step after the 50% crossing. New " +
    "York: non-residential blocks appear from step 5 while the model still says 0.77, and step 8 is a residential " +
    "cell at 0.36; the classes overlap, as on the UMAP slide. Some frames repeat because the nearest real cell is " +
    "the same for neighbouring steps (9 distinct cells in Phoenix, 8 in New York). The app has 85 walks across 30 " +
    "US, Indian and Japanese cities; both walks here met the strict 0.9 / 0.1 confidence cut-offs. The aerial " +
    "images are Esri basemap tiles shown for illustration; they are not the imagery the embeddings were computed from.");
}

// ---------- Boundary walks across countries and cities (summary of all 85 walks in the globe app)
function buildWalkSummary() {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Inside the embedding" });
  s.addText("Boundary walks by country and city", { placeholder: "title" });
  imageCard(s, "walks_country_city.png", 0.6, 1.55, 12.1, 4.35,
    "Left: by country, share of walk steps whose nearest cell label matches the model (USA 88%, Japan 87%, India 75%), walks with one clean label flip (71%, 75%, 31%) and walks reaching the 0.9/0.1 cut-offs (96%, 86%, 7%). Right: match % for each city and urbanisation level in the three countries.",
    "walk-sum", null);
  banner(s, "Takeaway", "India's classes blur even within a city: only 7% of its walks reach the strict cut-offs.", 6.1);
  s.addNotes(
    "All 85 completed walks in the globe app: 30 cities, each walked separately for urban, suburban and rural cells " +
    "(5 walks were skipped for too few confident cells: Lucknow rural, Tokyo suburban and rural, Seattle suburban and " +
    "rural). For each walk, 'match' is the share of its 12 steps where the nearest real cell's true label agrees with " +
    "the model's call at that step. The US and Japan look alike: about 87% to 88% of steps match, roughly three " +
    "quarters of walks flip label exactly once, and nearly all reach the strict 0.9 / 0.1 confidence cut-offs. India " +
    "is different: 75% of steps match, only 31% of walks flip once, and just 7% reach the strict cut-offs, so most " +
    "Indian walks start and end at less certain points. At city level the weakest walks are Kolkata (rural 25%, mean " +
    "53%), Lucknow suburban (42%) and, in the US, Los Angeles suburban (50%); Sapporo urban and Kobe rural (50%) are " +
    "the weak spots in Japan. Osaka and Nagoya are the cleanest (97%). With 12 steps per walk one step is 8 " +
    "percentage points, so single cells are noisy; read the pattern, not individual numbers. This matches the " +
    "held-out-city results: India is where AlphaEarth's residential signal is least shared across places.");
}

// ================= Story structure: agenda, section dividers, closing
const SECTIONS = [
  ["01", "Data and a fair test", "Data", "Where the labels come from, and how we compare AlphaEarth with GHSL"],
  ["02", "The research path", "Research path", "Six steps, from 64 numbers to neighbourhood, imagery and footprint shape"],
  ["03", "Beating GHSL, and why", "Beating GHSL", "Where we win on buildings, and which inputs did the work"],
  ["04", "Inside the embedding", "Inside the embedding", "Why some cities are easy and others are hard"],
  ["05", "Generalisation", "Generalisation", "What happens in cities and countries the model has never seen"],
  ["06", "Africa", "Africa", "Where GHSL is weakest, and what AlphaEarth needs to help"],
];

function divider(i) {
  const [num, title, section, sub] = SECTIONS[i];
  pres.addSection({ title: section });
  const s = pres.addSlide({ masterName: "SECTION", sectionTitle: section });
  s.addText(num, { placeholder: "num" });
  s.addText(title, { placeholder: "title" });
  s.addText(sub, { placeholder: "sub" });
  s.addNotes(`Section ${num}: ${title}. ${sub}.`);
}

function buildAgenda() {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Introduction" });
  s.addText("The story in six parts", { placeholder: "title" });
  const rowH = 0.68, y0 = 1.6;
  SECTIONS.forEach(([num, title, , sub], i) => {
    const y = y0 + i * rowH;
    card(s, 0.6, y, 12.1, rowH - 0.1, `agenda-row-${i + 1}`);
    s.addText(num, { x: 0.85, y, w: 0.8, h: rowH - 0.1, fontSize: 22, bold: true, color: C.accent1,
                     valign: "middle", margin: 0, isTextBox: true, objectName: `agenda-num-${i + 1}` });
    s.addText(title, { x: 1.7, y, w: 3.9, h: rowH - 0.1, fontSize: 18, bold: true, color: C.text1,
                       valign: "middle", margin: 0, isTextBox: true, objectName: `agenda-title-${i + 1}` });
    s.addText(sub, { x: 5.7, y, w: 6.8, h: rowH - 0.1, fontSize: 14, color: C.text2,
                     valign: "middle", margin: 0, isTextBox: true, objectName: `agenda-sub-${i + 1}` });
  });
  s.addNotes(
    "The deck follows one question: can AlphaEarth embeddings tell residential from non-residential buildings well " +
    "enough to replace or improve on GHSL? Part 1 sets up the data and a fair test. Part 2 walks through the models " +
    "we tried. Part 3 shows where we beat GHSL and what actually did the work. Part 4 looks inside the embedding to " +
    "explain why some cities are hard. Part 5 tests other cities and countries, and part 6 focuses on Africa, where " +
    "GHSL is weakest. We close with four findings and next steps.");
}

function buildClosing() {
  pres.addSection({ title: "Summary" });
  recCards("Summary", "What we found", [
    ["AlphaEarth carries a real, city-dependent signal", "Mean F1 0.78 across held-out US metros, from 0.91 in Phoenix to 0.66 in SF."],
    ["We beat GHSL on buildings, mostly via footprint shape", "Error 0.185 vs 0.211. Footprint shape takes AlphaEarth alone from 0.231 to 0.201."],
    ["It doesn't travel on its own", "Held-out Indian cities score 0.52; a low-rank head lifts unseen countries from 0.56 to 0.71."],
    ["In Africa, labels are the bottleneck", "GHSL misses 47–63% of non-residential cells; AlphaEarth wins only with local labels."],
  ], "Next steps", "Per-country recalibration · local label sets in Africa · held-out-city tests in more countries",
  "Four findings. One: AlphaEarth separates residential from non-residential, but how well depends on the city; " +
  "dense mixed-use metros are hardest. Two: on the frozen US buildings our best model beats GHSL by about 12%, and " +
  "most of that gain comes from the Overture footprint's size and shape rather than the Sentinel-2 imagery. Three: " +
  "the signal does not transfer by itself; India breaks on held-out cities, and low-rank heads help unseen countries. " +
  "Four: in African cities GHSL misses most non-residential buildings, and AlphaEarth beats it only when it has local " +
  "labels, which are scarce. Next steps: recalibrate the cut-off per country, collect small local label sets in " +
  "African cities, and run held-out-city tests in Ghana, Kenya and South Africa.", "sum");

  const q = pres.addSlide({ masterName: "TITLE", sectionTitle: "Summary" });
  q.addText("Questions?", { placeholder: "wordmark" });
  q.addText("Thank you", { placeholder: "title" });
  q.addText("AlphaEarth Embeddings Research", { placeholder: "team" });
  q.addText("November 2026", { placeholder: "meta" });
}

// ---------- Flow diagram: how the GHSL-style Sentinel-2 features are built (scripts/run_week9_ae_s2_nres_loro.py)
function buildHeuristicsFlow() {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Beating GHSL" });
  s.addText("How the GHSL-style features are built", { placeholder: "title" });
  const widths = [1.75, 1.75, 2.4, 3.0, 1.75], gap = 0.3, yH = 1.55, hH = 0.75, yB = 2.4, hB = 3.5;
  const heads = [["1", "Inputs"], ["2", "Read a chip"], ["3", "Measure"], ["4", "Score like GHSL"], ["5", "Model"]];
  const cols = [HEX.accent1, HEX.accent1, HEX.accent2, HEX.accent2, HEX.accent3];
  const T = (text, o) => ({ text, options: Object.assign({ fontSize: 14, color: C.text1 }, o || {}) });
  const bodies = [
    [T("Sentinel-2", { bold: true, breakLine: true }), T("Summer 2018; blue, green, red, near-infrared; 10 m", { color: C.text2, breakLine: true }),
     T(" ", { fontSize: 8, breakLine: true }),
     T("Overture footprint", { bold: true, breakLine: true }), T("One outline per building", { color: C.text2 })],
    [T("Pixels under the building's box", { bold: true, breakLine: true }),
     T("At least 8 × 8 pixels (80 m), so a small house also sees its block", { color: C.text2 })],
    [T("Colour", { bold: true, breakLine: true }), T("Band means, brightness, blue share, vegetation (NDVI)", { color: C.text2, breakLine: true }),
     T(" ", { fontSize: 8, breakLine: true }),
     T("Texture", { bold: true, breakLine: true }), T("Fine and coarse spread, and their ratio", { color: C.text2, breakLine: true }),
     T(" ", { fontSize: 8, breakLine: true }),
     T("Shape (footprint)", { bold: true, breakLine: true }), T("Area, diameter, compactness", { color: C.text2 })],
    [T("Bright", { bold: true }), T("  800 → 3,300", { color: C.text2, breakLine: true }),
     T("Bluish", { bold: true }), T("  0.30 → 0.55", { color: C.text2, breakLine: true }),
     T("Smooth", { bold: true }), T("  400 → 0", { color: C.text2, breakLine: true }),
     T("Coarse", { bold: true }), T("  0.77 → 0.89", { color: C.text2, breakLine: true }),
     T("Large", { bold: true }), T("  20 → 80 m across", { color: C.text2, breakLine: true }),
     T("Compact", { bold: true }), T("  0.4 → 0.9", { color: C.text2, breakLine: true }),
     T(" ", { fontSize: 8, breakLine: true }),
     T("Each cue scaled 0 to 1, then averaged into one non-residential score", { bold: true, color: C.accent2 })],
    [T("AlphaEarth 64 + these 16 features", { bold: true, breakLine: true }),
     T("XGBoost, one metro held out", { color: C.text2, breakLine: true }),
     T(" ", { fontSize: 8, breakLine: true }),
     T("Output", { bold: true, breakLine: true }),
     T("Residential and non-residential share of each building", { color: C.text2 })],
  ];
  let x = 0.6;
  for (let i = 0; i < 5; i++) {
    const w = widths[i];
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: yH, w, h: hH, rectRadius: 0.08, fill: { color: cols[i] },
      line: { type: "none" }, objectName: `flow-head-${i + 1}` });
    s.addText([
      { text: heads[i][0] + "  ", options: { fontSize: 16, bold: true, color: C.background1 } },
      { text: heads[i][1], options: { fontSize: 16, bold: true, color: C.background1 } },
    ], { x: x + 0.15, y: yH, w: w - 0.3, h: hH, valign: "middle", margin: 0, isTextBox: true, objectName: `flow-head-text-${i + 1}` });
    card(s, x, yB, w, hB, `flow-body-${i + 1}`, i === 3 ? HEX.accent4 : undefined);
    s.addText(bodies[i], { x: x + 0.15, y: yB + 0.15, w: w - 0.3, h: hB - 0.3, valign: "top", margin: 0,
      paraSpaceAfter: i === 3 ? 4 : 2, isTextBox: true, objectName: `flow-body-text-${i + 1}` });
    if (i < 4) {
      s.addShape(pres.shapes.ISOSCELES_TRIANGLE, { x: x + w + 0.08, y: yH + hH / 2 - 0.07, w: 0.14, h: 0.12,
        rotate: 90, fill: { color: "9AA3AE" }, line: { type: "none" }, objectName: `flow-arrow-${i + 1}` });
    }
    x += w + gap;
  }
  banner(s, "Result", "AlphaEarth 0.231 → 0.193 with all 16 features, and → 0.201 with just the 3 shape numbers.", 6.1);
  s.addNotes(
    "How the week 9 features are built, following GHSL's own rules for spotting non-residential buildings (Pesaresi " +
    "et al. 2024). 1: inputs are a Sentinel-2 summer 2018 mosaic per metro (blue, green, red, near-infrared bands at " +
    "10 m, scenes under 20% cloud) and each building's Overture footprint. 2: for each building we read the Sentinel-2 " +
    "pixels under its bounding box, at least 8 by 8 pixels, so even a small house is seen with its block. 3: we " +
    "measure colour (band means, brightness, blue share, NDVI), texture (spread of brightness at pixel level and over " +
    "2x2 blocks, and their ratio) and shape from the footprint (area, equivalent diameter, compactness). 4: six cues " +
    "(bright: brightness; bluish: blue share; smooth: fine texture; coarse: coarse-over-fine texture ratio; large: " +
    "footprint diameter; compact: compactness) are each scaled from 0 to 1 over the ranges shown and averaged into one GHSL-style non-residential score; NDVI is " +
    "a feature but not part of the score, and the texture-ratio range was calibrated on training buildings after the " +
    "texture fix. 5: all 16 values (the measures, the score, a pixel count and a valid flag) are added to the 64 " +
    "AlphaEarth numbers and fed to XGBoost, trained with one metro held out, to predict each building's residential " +
    "and non-residential share. The banner numbers are share errors on the frozen 71,497 buildings; the next two " +
    "slides show which cues matter and that the footprint shape does most of the work.");
}

// ---------- Real-data companion to the flow diagram (make_figure_flow_examples.py)
function buildFlowExamples() {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Beating GHSL" });
  s.addText("The same steps on two real Denver buildings", { placeholder: "title" });
  imageCard(s, "flow_examples_denver.png", 0.6, 1.55, 12.1, 4.35,
    "A Denver house and shop taken through the five steps: Sentinel-2 image with Overture footprints, the 16 by 16 pixel chip, measured values, six cue scores (house 0.11, shop 0.43) and the model's predicted shares against the truth.",
    "flow-ex", null);
  banner(s, "Read it", "The shop scores 0.43 and is called 72% non-residential; the house scores 0.11 and is called 71% residential.", 6.1);
  s.addNotes(
    "Two buildings from the frozen test set in Denver, taken through the steps on the previous slide with the real " +
    "data. 1: the Sentinel-2 summer 2018 mosaic used in week 9, about 300 m across, with Overture footprints on top " +
    "(white is the building, teal and amber are other residential and non-residential footprints). 2: the chip the " +
    "code reads; both buildings are smaller than 8 pixels across, so the window is padded to 16 by 16 pixels around " +
    "them. 3: the measured values from the texture-fixed feature table. 4: the six cues computed with the same ranges " +
    "as the code (the scores match the stored values). The house is dark, green and small, so only Smooth and " +
    "Compact light up and the score is 0.11. The shop is brighter, coarser in texture, larger and more compact, so " +
    "the score is 0.43. 5: the model's held-out prediction (one metro held out) against the truth tick. The house is " +
    "predicted 71% residential (truth 54%, the rest of its cells are empty ground) and the shop 72% non-residential " +
    "(truth 92%). At 10 m the Sentinel-2 pixels are coarse: the house covers only a few of them, which is why the " +
    "footprint shape carries so much of the signal.");
}

// ---------- Pure vs mixed cells: a case for classification (outputs/cell_purity_by_city)
function buildPurity() {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Beating GHSL" });
  s.addText("Do we need regression? Mixed cells are rare", { placeholder: "title" });
  chartImage(s, "chart_purity.png", 0.6, 1.55, 8.1, 4.35,
    "Share of labelled 10 m cells containing both residential and non-residential buildings, per city and country: 1.2% overall, Japan 3.0% (Nagoya 6.7%), every other country under 0.4%.",
    "purity-chart");
  statStack(s, 8.95, 1.55, 3.75, 4.35, [
    ["1.2%", "of 11.9M labelled cells are mixed (146,300)"],
    ["0.86%", "highest city outside Japan (Aswan); Nagoya 6.7%"],
    ["0.838 vs 0.832", "macro-F1: trained as a classifier vs regression's call (week 12)"],
  ], "purity");
  banner(s, "Proposal", "Treat it as classification; keep shares only for the rare mixed cells.", 6.1);
  s.addNotes(
    "A cell is mixed when both its residential and non-residential share are above zero. Across 47 cities and 11.9M " +
    "labelled 10 m cells, 146,300 are mixed (1.2%). Africa is 0.21%, North America 0.30%, Asia 2.35%. Japan drives " +
    "Asia's rate (3.0% overall; Nagoya 6.7%, Fukuoka 4.7%, Osaka 3.9%); India is 0.28%. Outside Japan no city passes " +
    "0.86% (Aswan), and the densest US cities are LA 0.76% and NYC 0.74%. With almost every cell holding one kind of " +
    "building, predicting a continuous residential share mostly re-learns a yes/no answer, so a classifier is the " +
    "simpler target. The week 12 test points the same way: on the frozen 71,497 US buildings an XGBoost classifier " +
    "scores 0.838 macro-F1, against 0.832 for the regression model's larger share, with the same features. Caveats: " +
    "purity counts only tagged buildings (untagged ones are excluded, so some mixing is hidden); the regression also " +
    "predicts how much of a cell is built at all, which a classifier would need as a separate built / not-built step " +
    "(GHS-BUILT-S already provides that); and the classifier evidence is one comparison on the US set. Source: " +
    "outputs/cell_purity_by_city (all selected cities in the Postgres label set; this differs from the filtered US " +
    "cohort of 3.15M cells on the ground-truth slide).");
}

(async () => {
  buildAgenda();
  divider(0);
  await buildSources();
  buildGroundTruth();
  buildBalancing();
  buildFairTest();
  divider(1);
  await buildPathOverview();
  buildStepAE();
  buildStepPLS();
  buildStepCNN();
  buildStepViT();
  buildStepXGB();
  buildStepContext();
  divider(2);
  buildProgression();
  buildMetroRegression();
  buildHeuristicsFlow();
  buildFlowExamples();
  buildHeuristics();
  buildAblation();
  buildPurity();
  divider(3);
  buildEmbedding();
  buildWalks();
  buildWalkSummary();
  divider(4);
  buildGeneralisation();
  divider(5);
  await buildAfrica();
  buildClosing();
  const out = path.join(__dirname, "alphaearth_research_deck.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})().catch((e) => { console.error(e); process.exit(1); });
