# Residential definition — `resi_def_v1`

**Locked** 2026-09-14 · Kyle Meeting 1 P0 #1 · JSON: [`residential_definition_v1.json`](residential_definition_v1.json)

Unlabeled footprints are **excluded** (never negatives). DoU stratifies location only — it does not define these labels.

---

## A. Building GT (`y_res`)

Source: Overture Buildings `class` via `CLASS_TO_LABEL` in `scripts/build_building_type_dataset.py`.

| `y_res` | Study labels | Overture `class` |
|---|---|---|
| **1** | `single_family` | `house`, `detached`, `bungalow`, `semi`, `semidetached_house` |
| **1** | `multi_family` | `apartments`, `dormitory` |
| **0** | `education` | `school`, `kindergarten`, `college`, `university` |
| **0** | `medical` | `hospital`, `clinic` |
| **0** | `retail` | `retail`, `commercial`, `supermarket` |
| **0** | `office` | `office` |
| **0** | `hotel` | `hotel` |
| **0** | `industrial` | `industrial`, `warehouse`, `manufacture` |
| **0** | `religious` | `church`, `cathedral`, `mosque`, `temple`, `synagogue` |
| **0** | `civic` | `civic`, `government`, `public`, `fire_station`, `police` |

**Exclude from GT:** null `class`; `residential`; `terrace`; any other class; subtype-only tags.

```text
y_res = 1 if label in {single_family, multi_family} else 0
```

---

## B. 10 m cell (`f_res`)

Only GT buildings (Track A). Weight `a_i` = intersection area of cell ∩ building (fallback: centroid-in-cell × footprint area).

```text
f_res(c) = sum(a_i * y_res_i) / sum(a_i)   over GT buildings in cell c
```

Hard labels (τ = 0.7):

| Condition | Label |
|---|---|
| `f_res >= 0.7` | residential |
| `f_res <= 0.3` | non-residential |
| `0.3 < f_res < 0.7` | **mixed** (exclude from binary metrics; report rate) |
| no GT buildings | undefined (exclude) |

Prefer fraction over majority class. Mixed cells are expected, not errors.

---

## C. Weak (not GT)

`subtype=residential`, Places, GHS-OBAT, cadastral — separate fields only; never pool into `resi_def_v1` F1.

Stamp metrics/parquets with `"label_definition": "resi_def_v1"`.
