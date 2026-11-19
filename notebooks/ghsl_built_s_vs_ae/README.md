# GHS-BUILT-S vs AE fraction MAE

Soft GHSL built-up surface vs Week-4 AlphaEarth fraction heads.
Primary metric: **MAE mean(res, nres)** (↓). Target = Overture soft fractions on AE cells.

**Do not mix cohorts.** Numbers from different sections are not comparable unless the cohort matches.

Append new runs to the tables below; link the run dir.

---

## Master summary (2026-11-03)

Metric: **MAE mean(res, nres)** ↓. Soft BUILT-S at AE centers vs AE fraction heads. Cohorts differ — see notes.

### Head-to-head by country (canonical within-country protocols)

| Country | Protocol | Scale | n | BUILT-S | Ridge | PLS | Winner |
|---|---|---|---:|---:|---:|---:|---|
| **USA** | region-LORO (W4 balanced pure) | 10 m | 1,751,814 | **0.197** | 0.237 | 0.240 | BUILT-S |
| **USA** | same → 100 m tiles min≥20 | 100 m | 35,173 tiles | **0.124** | 0.152 | 0.158 | BUILT-S |
| **IND** | city-LORO | 10 m | 1,136,114 | **0.258** | 0.295 | 0.291 | BUILT-S (8/10 cities) |
| **JPN** | city-LORO | 10 m | 2,612,910 | 0.231 | **0.211** | 0.212 | AE (7/10 cities) |
| **IND** | leave-one-country-out | 10 m | 1,136,114 | **0.258** | 0.297 | 0.285 | BUILT-S |
| **JPN** | leave-one-country-out | 10 m | 2,612,910 | **0.231** | 0.273 | 0.235 | BUILT-S |

Pooled IND+JPN city-LORO: AE ≈ BUILT-S (Ridge 0.236 · PLS 0.236 · BUILT-S 0.239).

### Pattern

- **USA & India:** soft BUILT-S beats AE Ridge/PLS on Overture fraction MAE.
- **Japan city-LORO:** AE wins (Ridge slightly ahead of PLS); under LOCO, BUILT-S wins again.
- FUN one-hot (US only) is much worse (~0.34 @ 10 m balanced).

### City / metro MAE (canonical) — mean and res / nres split

Each model cell is **mean** then **(res / nres)**. Winner by mean(res, nres).

CSVs (with `*_res`, `*_nres`, `*_mean` columns):
[`mae_by_city_USA_balanced.csv`](mae_by_city_USA_balanced.csv) ·
[`mae_by_city_IND_JPN_city_loro.csv`](mae_by_city_IND_JPN_city_loro.csv) ·
[`mae_by_city_IND_JPN_loco.csv`](mae_by_city_IND_JPN_loco.csv)

#### USA — balanced pure, region-LORO @ 10 m (BUILT-S wins 10/10)

| Metro | n | BUILT-S mean (res/nres) | Ridge mean (res/nres) | PLS mean (res/nres) | Winner |
|---|---:|---|---|---|---|
| Atlanta_SE | 113k | **0.197** (0.209/0.184) | 0.238 (0.214/0.262) | 0.233 (0.237/0.228) | BUILT-S |
| Boston_NE | 187k | **0.229** (0.251/0.207) | 0.253 (0.248/0.257) | 0.259 (0.261/0.257) | BUILT-S |
| Chicago_MW | 147k | **0.223** (0.245/0.201) | 0.279 (0.251/0.306) | 0.274 (0.250/0.298) | BUILT-S |
| Denver_W | 260k | **0.157** (0.195/0.119) | 0.198 (0.193/0.204) | 0.200 (0.201/0.199) | BUILT-S |
| Houston_S | 168k | **0.211** (0.244/0.178) | 0.217 (0.201/0.233) | 0.232 (0.202/0.262) | BUILT-S |
| LA_W | 457k | **0.185** (0.227/0.142) | 0.216 (0.214/0.218) | 0.227 (0.242/0.212) | BUILT-S |
| NYC_NE | 160k | **0.212** (0.253/0.172) | 0.305 (0.283/0.328) | 0.307 (0.277/0.337) | BUILT-S |
| Phoenix_SW | 149k | **0.193** (0.229/0.157) | 0.244 (0.196/0.292) | 0.232 (0.194/0.270) | BUILT-S |
| SF_W | 35k | **0.238** (0.265/0.210) | 0.314 (0.264/0.364) | 0.311 (0.261/0.362) | BUILT-S |
| Seattle_NW | 77k | **0.196** (0.224/0.168) | 0.227 (0.241/0.214) | 0.221 (0.225/0.216) | BUILT-S |

#### India — city-LORO (BUILT-S 8/10)

| City | BUILT-S mean (res/nres) | Ridge mean (res/nres) | PLS mean (res/nres) | Winner |
|---|---|---|---|---|
| Ahmedabad | 0.291 (0.339/0.243) | **0.277** (0.272/0.281) | 0.280 (0.287/0.272) | Ridge |
| Bengaluru | **0.253** (0.271/0.234) | 0.304 (0.343/0.264) | 0.292 (0.315/0.269) | BUILT-S |
| Chennai | 0.352 (0.387/0.317) | 0.324 (0.253/0.394) | **0.317** (0.261/0.373) | PLS |
| Delhi | **0.252** (0.293/0.210) | 0.303 (0.257/0.348) | 0.307 (0.238/0.375) | BUILT-S |
| Hyderabad | **0.248** (0.275/0.221) | 0.284 (0.232/0.336) | 0.283 (0.244/0.323) | BUILT-S |
| Jaipur | **0.273** (0.305/0.240) | 0.299 (0.263/0.335) | 0.292 (0.268/0.316) | BUILT-S |
| Kolkata | **0.221** (0.270/0.171) | 0.289 (0.221/0.358) | 0.282 (0.232/0.332) | BUILT-S |
| Lucknow | **0.298** (0.318/0.277) | 0.308 (0.285/0.331) | 0.301 (0.267/0.335) | BUILT-S |
| Mumbai | **0.227** (0.245/0.208) | 0.288 (0.287/0.289) | 0.284 (0.271/0.297) | BUILT-S |
| Pune | **0.267** (0.274/0.260) | 0.275 (0.233/0.317) | 0.285 (0.233/0.336) | BUILT-S |

#### Japan — city-LORO (AE 7/10)

| City | BUILT-S mean (res/nres) | Ridge mean (res/nres) | PLS mean (res/nres) | Winner |
|---|---|---|---|---|
| Fukuoka | 0.254 (0.303/0.205) | **0.194** (0.187/0.200) | 0.198 (0.198/0.198) | Ridge |
| Hiroshima | 0.233 (0.273/0.194) | **0.211** (0.172/0.249) | 0.212 (0.172/0.252) | Ridge |
| Kobe | **0.209** (0.252/0.166) | 0.217 (0.188/0.246) | 0.221 (0.186/0.257) | BUILT-S |
| Kyoto | 0.226 (0.264/0.187) | **0.218** (0.200/0.235) | 0.220 (0.197/0.244) | Ridge |
| Nagoya | 0.256 (0.300/0.211) | **0.201** (0.182/0.219) | 0.206 (0.190/0.222) | Ridge |
| Osaka | **0.214** (0.254/0.175) | 0.225 (0.198/0.252) | 0.228 (0.196/0.261) | BUILT-S |
| Sapporo | **0.242** (0.281/0.202) | 0.284 (0.193/0.375) | 0.262 (0.216/0.309) | BUILT-S |
| Sendai | 0.249 (0.283/0.214) | **0.194** (0.184/0.205) | 0.196 (0.188/0.205) | Ridge |
| Tokyo | 0.219 (0.256/0.183) | 0.209 (0.188/0.229) | **0.208** (0.192/0.225) | PLS |
| Yokohama | 0.214 (0.244/0.184) | 0.198 (0.173/0.223) | **0.198** (0.177/0.219) | PLS |

#### USA — balanced pure → 100 m tiles min≥20 (same cells as 10 m table)

CSV: [`mae_by_city_USA_balanced_100m.csv`](mae_by_city_USA_balanced_100m.csv)

| Metro | n tiles | BUILT-S mean (res/nres) | Ridge mean (res/nres) | PLS mean (res/nres) | Winner |
|---|---:|---|---|---|---|
| Atlanta_SE | 1,189 | 0.194 (0.144/0.245) | **0.153** (0.151/0.155) | 0.193 (0.197/0.189) | Ridge |
| Boston_NE | 3,684 | **0.157** (0.150/0.163) | 0.167 (0.151/0.183) | 0.171 (0.163/0.179) | BUILT-S |
| Chicago_MW | 2,897 | **0.147** (0.146/0.148) | 0.202 (0.180/0.224) | 0.201 (0.181/0.220) | BUILT-S |
| Denver_W | 6,472 | **0.075** (0.092/0.057) | 0.111 (0.100/0.122) | 0.114 (0.116/0.111) | BUILT-S |
| Houston_S | 3,345 | **0.155** (0.167/0.142) | 0.172 (0.173/0.172) | 0.182 (0.179/0.185) | BUILT-S |
| LA_W | 11,117 | **0.097** (0.120/0.074) | 0.129 (0.123/0.136) | 0.141 (0.152/0.130) | BUILT-S |
| NYC_NE | 3,270 | **0.144** (0.153/0.135) | 0.208 (0.190/0.225) | 0.210 (0.190/0.231) | BUILT-S |
| Phoenix_SW | 1,614 | 0.186 (0.167/0.204) | 0.133 (0.120/0.147) | **0.117** (0.111/0.123) | PLS |
| SF_W | 712 | **0.167** (0.171/0.164) | 0.230 (0.196/0.264) | 0.233 (0.208/0.257) | BUILT-S |
| Seattle_NW | 873 | **0.188** (0.154/0.223) | 0.211 (0.183/0.239) | 0.204 (0.166/0.241) | BUILT-S |

At 100 m, BUILT-S still wins most metros; Atlanta (Ridge) and Phoenix (PLS) flip. Broader AE-path 100 m (Ridge only): [`mae_by_city_USA_100m_broad.csv`](mae_by_city_USA_100m_broad.csv).

**IND / JPN @ 100 m:** not run yet (10 m city-LORO / LOCO only).

LOCO city-level (also with res/nres): [`mae_by_city_IND_JPN_loco.csv`](mae_by_city_IND_JPN_loco.csv) · figures: [`figures/mae_city_city_loro_IND.png`](figures/mae_city_city_loro_IND.png) · [`figures/mae_city_city_loro_JPN.png`](figures/mae_city_city_loro_JPN.png)

### History (US, other cohorts — not matched to IND/JPN)

| Cohort | n | BUILT-S | Ridge | FUN |
|---|---:|---:|---:|---:|
| shared 59k @ 10 m | 58,893 | **0.236** | 0.271 | 0.380 |
| full_building @ 10 m | 3,042,818 | **0.179** | 0.229 | 0.323 |
| 100 m broader AE path | 61,477 tiles | **0.100** | 0.152 | 0.289 |

---

## Method

- **GHSL:** `GHS-BUILT-S` E2018 10 m BUFRAC (0–100) + `GHS-BUILT-S NRES` E2018 10 m boolean
- **Sample:** nearest GHSL value at each AE cell center (same protocol as Week-4 FUN)
- **Encoding:** `bu = BUFRAC/100`; if NRES=1 → `(frac_res, frac_nres)=(0, bu)` else `(bu, 0)`; `empty=1-bu`
- **Also report:** FUN one-hot for contrast

Scripts:

| Script | Notes |
|---|---|
| `scripts/eval_ghsl_built_s_vs_ae.py` | 10 m; shared / full-building |
| `scripts/eval_ghsl_built_s_vs_ae_100m.py` | 100 m tiles (broader AE path) |
| `scripts/eval_ghsl_built_s_vs_ae_pls.py` | balanced PLS + Ridge + BUILT-S + FUN |
| `scripts/eval_ghsl_built_s_vs_ae_intl.py` | IND/JPN via Postgres `pgdata/`; city-LORO Ridge + PLS |

---

## Cohorts

| ID | Universe | AE preds |
|---|---|---|
| `balanced_pls` | W4 balanced pure PLS OOF (1,751,814) | PLS LORO + Ridge on same ids |
| `100m_from_balanced` | UTM 100 m tiles from those cells, min≥20 (35,173) | PLS + Ridge |
| `shared_59k` | AE ∩ FUN shared set (~59k) | W4 Ridge full-GT OOF |
| `full_building` | AE cells with building signal (~3.0M) | W4 Ridge full-GT OOF |
| `100m_from_shared_path` | UTM 100 m tiles, min≥20 (~61k) | W4 Ridge |
| `intl_city_loro` | IND/JPN pure clear 1:1 per city (Postgres) | city-LORO Ridge + PLS |

---

## Results — canonical: balanced

Same pixel universe for every model. 100 m = mean-aggregate those cells (min ≥ 20 / tile).

| Date | Scale | n | BUILT-S | PLS | Ridge | FUN | Run |
|---|---|---:|---:|---:|---:|---:|---|
| 2026-11-03 | 10 m | 1,751,814 | **0.197** | 0.240 | 0.237 | 0.340 | [`20261103_014700`](../../outputs/ghsl_built_s_vs_ae_pls/20261103_014700/) |
| 2026-11-03 | 100 m | 35,173 tiles | **0.124** | 0.158 | 0.152 | 0.293 | same |

On balanced cells, soft BUILT-S beats both AE heads; Ridge slightly ahead of PLS; gap holds at 100 m.

Preds: PLS `outputs/week4_pls_loro/oof_predictions.parquet` · Ridge `outputs/week4_pixel_balanced_train/oof_predictions_full_gt.parquet` · BUILT-S cache `outputs/ghsl_built_s_vs_ae/20261103_010752/checkpoints/built_s_at_ae_centers.parquet`

Notebook: [`03_built_s_vs_ae_pls.ipynb`](03_built_s_vs_ae_pls.ipynb)

---

## Results — earlier cohorts (history)

### shared 59k @ 10 m

| Date | n | BUILT-S | Ridge | FUN | Run |
|---|---:|---:|---:|---:|---|
| 2026-11-03 | 58,893 | **0.236** | 0.271 | 0.380 | [`20261103_010752`](../../outputs/ghsl_built_s_vs_ae/20261103_010752/) |

`full_building` (n=3,042,818): BUILT-S **0.179** · Ridge 0.229 · FUN 0.323 · same run.

Notebook: [`01_built_s_vs_ae_mae.ipynb`](01_built_s_vs_ae_mae.ipynb)

### 100 m from broader AE path (not balanced)

| Date | n tiles | BUILT-S | Ridge | FUN | Run |
|---|---:|---:|---:|---:|---|
| 2026-11-03 | 61,477 | **0.100** | 0.152 | 0.289 | [`20261103_013904`](../../outputs/ghsl_built_s_vs_ae_100m/20261103_013904/) |

Notebook: [`02_built_s_vs_ae_100m.ipynb`](02_built_s_vs_ae_100m.ipynb)

---

## Results — India / Japan (`intl_city_loro`)

Not comparable to US `balanced_pls` (different cities, city-LORO vs region-LORO, Postgres labels). See **Master summary** and the city/country × LORO/LOCO section below for the full IND+JPN tables.

Artifacts: `outputs/ghsl_built_s_vs_ae_intl/20261103_pgdata/` · protocols: `outputs/ghsl_built_s_vs_ae_intl_protocols/20261103_city_country_loro_loco/`

---

## Grid / alignment

| Check | Result |
|---|---|
| AE grid | UTM ~10 m |
| GHSL | Mollweide ESRI:54009 10 m; sampled at AE centers |
| Origin-shift IoU (~4.1 m, same CRS) | ~0.34 @ 10 m → ~0.89 @ 100 m |
| Cross-CRS cell IoU | ~0.36 |

### Figures — shift + step-by-step construction

| What | Path |
|---|---|
| AE vs GHSL grids shifted (map) | [`figures/ghsl_ae_shifted_grids_map.png`](figures/ghsl_ae_shifted_grids_map.png) · source [`outputs/week4_ghsl_pixel/figures/ghsl_ae_shifted_grids_map.png`](../../outputs/week4_ghsl_pixel/figures/ghsl_ae_shifted_grids_map.png) |
| Alignment audit strip | [`figures/ghsl_alignment_audit.png`](figures/ghsl_alignment_audit.png) · source [`outputs/week4_ghsl_pixel/figures/ghsl_alignment_audit.png`](../../outputs/week4_ghsl_pixel/figures/ghsl_alignment_audit.png) |
| **10 m** step-by-step on map (GT → AE → BUILT-S → residuals) | [`figures/comparison_steps_on_map.png`](figures/comparison_steps_on_map.png) · run [`outputs/ghsl_built_s_vs_ae/20261103_010752/figures/`](../../outputs/ghsl_built_s_vs_ae/20261103_010752/figures/) |
| **100 m** step-by-step on map | [`figures/comparison_steps_100m_on_map.png`](figures/comparison_steps_100m_on_map.png) · run [`outputs/ghsl_built_s_vs_ae_100m/20261103_013904/figures/`](../../outputs/ghsl_built_s_vs_ae_100m/20261103_013904/figures/) |
| Origin-shift IoU **10 m vs 100 m** | [`figures/iou_10m_vs_100m.png`](figures/iou_10m_vs_100m.png) · same 100 m run |
| Shift impact JSON (median IoU ~0.36, ~4.1 m) | [`outputs/ghsl_built_s_vs_ae/20261103_010752/shift_iou_impact.json`](../../outputs/ghsl_built_s_vs_ae/20261103_010752/shift_iou_impact.json) |

Notebooks that embed these: [`01_built_s_vs_ae_mae.ipynb`](01_built_s_vs_ae_mae.ipynb) (10 m steps) · [`02_built_s_vs_ae_100m.ipynb`](02_built_s_vs_ae_100m.ipynb) (100 m steps + IoU).

---


---

## Results — city & country × city-LORO / leave-one-country-out

Cohort: `intl_city_loro` balanced pure (Postgres). Protocols:

- **city_loro** — leave-one-city-out within country
- **loco** — leave-one-country-out (train on other countries, test on held-out)

BUILT-S is protocol-independent (same soft surface at AE centers).

### Country-level MAE mean(res, nres)

| Date | Protocol | Country | n | BUILT-S | Ridge | PLS | Run |
|---|---|---|---:|---:|---:|---:|---|
| 2026-11-03 | city_loro | IND | 1,136,114 | **0.258** | 0.295 | 0.291 | [`20261103_city_country_loro_loco`](../../outputs/ghsl_built_s_vs_ae_intl_protocols/20261103_city_country_loro_loco) |
| 2026-11-03 | city_loro | JPN | 2,612,910 | **0.231** | 0.211 | 0.212 | [`20261103_city_country_loro_loco`](../../outputs/ghsl_built_s_vs_ae_intl_protocols/20261103_city_country_loro_loco) |
| 2026-11-03 | loco | IND | 1,136,114 | **0.258** | 0.297 | 0.285 | [`20261103_city_country_loro_loco`](../../outputs/ghsl_built_s_vs_ae_intl_protocols/20261103_city_country_loro_loco) |
| 2026-11-03 | loco | JPN | 2,612,910 | **0.231** | 0.273 | 0.235 | [`20261103_city_country_loro_loco`](../../outputs/ghsl_built_s_vs_ae_intl_protocols/20261103_city_country_loro_loco) |

City-level CSV: [`mae_by_city_protocol.csv`](../../outputs/ghsl_built_s_vs_ae_intl_protocols/20261103_city_country_loro_loco/mae_by_city_protocol.csv)
Notebook: [`05_intl_city_country_loro_loco.ipynb`](05_intl_city_country_loro_loco.ipynb)

Takeaway: city_loro/IND: BUILT-S=0.258 Ridge=0.295 PLS=0.291 · city_loro/JPN: BUILT-S=0.231 Ridge=0.211 PLS=0.212 · loco/IND: BUILT-S=0.258 Ridge=0.297 PLS=0.285 · loco/JPN: BUILT-S=0.231 Ridge=0.273 PLS=0.235


---

## Results — India & Japan (balanced pure, city-LORO, Postgres labels)

Labels: `pgdata/` → `aef_international` (`resi_def_v1_pixel_frac_v1` pure clear). AE heads: city-LORO OOF Ridge + PLS. GHSL: BUILT-S sampled at AE centers.

| Date | Country | n | BUILT-S | Ridge | PLS | Run |
|---|---|---:|---:|---:|---:|---|
| 2026-11-13 | EGY | 70,516 | **0.236** | 0.228 | 0.235 | [`africa_20261113_020114`](../..//home/kv133/Alpha-Earth-Embeddings-Research/outputs/ghsl_built_s_vs_ae_intl/africa_20261113_020114) |
| 2026-11-13 | GHA | 63,094 | **0.207** | 0.210 | 0.215 | [`africa_20261113_020114`](../..//home/kv133/Alpha-Earth-Embeddings-Research/outputs/ghsl_built_s_vs_ae_intl/africa_20261113_020114) |
| 2026-11-13 | KEN | 75,582 | **0.203** | 0.238 | 0.242 | [`africa_20261113_020114`](../..//home/kv133/Alpha-Earth-Embeddings-Research/outputs/ghsl_built_s_vs_ae_intl/africa_20261113_020114) |
| 2026-11-13 | NGA | 35,320 | **0.253** | 0.186 | 0.190 | [`africa_20261113_020114`](../..//home/kv133/Alpha-Earth-Embeddings-Research/outputs/ghsl_built_s_vs_ae_intl/africa_20261113_020114) |
| 2026-11-13 | ZAF | 120,994 | **0.207** | 0.184 | 0.187 | [`africa_20261113_020114`](../..//home/kv133/Alpha-Earth-Embeddings-Research/outputs/ghsl_built_s_vs_ae_intl/africa_20261113_020114) |

Notebook: [`04_built_s_vs_ae_intl.ipynb`](04_built_s_vs_ae_intl.ipynb)

Takeaway: EGY n=70,516: BUILT-S=0.236 Ridge=0.228 PLS=0.235 | GHA n=63,094: BUILT-S=0.207 Ridge=0.210 PLS=0.215 | KEN n=75,582: BUILT-S=0.203 Ridge=0.238 PLS=0.242 | NGA n=35,320: BUILT-S=0.253 Ridge=0.186 PLS=0.190 | ZAF n=120,994: BUILT-S=0.207 Ridge=0.184 PLS=0.187

## Changelog

| Date | Change |
|---|---|
| 2026-11-13 | India & Japan city-LORO via Postgres pgdata |
| 2026-11-03 | City/metro MAE with res/nres split |
| 2026-11-03 | City/metro MAE tables (USA + IND + JPN) |
| 2026-11-03 | Master summary: USA + IND + JPN |
| 2026-11-03 | city/country LORO+LOCO fraction MAE tables |
| 2026-11-03 | India early city-LORO MAE (BUILT-S 0.258 vs PLS 0.291 / Ridge 0.295) |
| 2026-11-03 | Seeded shared / 100 m-broad / balanced PLS+Ridge results |
