# Claude prompt — keep residential vs non-residential only

Attach: `AlphaEarth_Weeks_1_5_Canvas.pdf`

Then paste:

---

```text
Update this AlphaEarth Weeks 1–5 canvas so it covers residential vs non-residential analysis only. Drop every empty-class / vacant / 3-class story.

Scope
- Keep: binary RES vs NRES classification, metro LORO, ViT vs linear generalisation, embedding/UMAP dynamics, country LORO / transfer, training/loss notes that are about the binary task.
- Remove or rewrite anything about the empty fraction, empty MAE/R², “remaining empty share,” binary heads putting zero mass on empty, or the 3-class soft head as an empty-class fix.
- Figures/tables that include an Empty column or a 3-class row: either delete that column/row, or replace the panel with RES/NRES-only metrics if those numbers are already in the document. Do not invent new numbers.
- Recommendations: delete “model the empty class explicitly” / “add an explicit empty-class term.” Reframe area-fraction advice as RES and NRES only (e.g. report those two fractions at 100 m if that claim remains supported without empty).

Editing rules
- Preserve exact RES/NRES numbers that stay in scope.
- If a sentence mixes RES/NRES with empty, rewrite it to RES/NRES only or delete the empty clause.
- If a whole bullet or chart exists only to justify empty/3-class modeling, delete it.
- Do not add new experiments, cities, or scores.
- Blunt stakeholder tone. No AI filler.

Output
1. Section-by-section change list (what you cut / rewrote).
2. Full revised canvas copy, section order unchanged where content remains.
3. Short “open gaps” list: any claim that depended on empty metrics and can no longer be supported without new numbers.

Known empty/3-class spots to scrub (from the current canvas)
- Classification table/chart rows: “3-class soft head (W3)”
- Regression intro: “remaining empty share”
- Empty R² bar / “MAE res / nres / empty” / “R² res / nres / empty”
- Regression “What this means”: R² for empty 0.10; binary heads zero mass on empty (empty MAE = 0.37); 3-class head 0.37 → 0.29
- Any recommendation to model empty explicitly
```
