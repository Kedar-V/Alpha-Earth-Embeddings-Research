#!/usr/bin/env python3
"""Assemble gt_builder.html = template + window data (build_gt_canvas_data.py) + metro audit (build_gt_audit_data.py)."""
from pathlib import Path
H = Path(__file__).resolve().parent
t = (H / "gt_builder_template.html").read_text()
t = t.replace("__DATA__", (H / "data/gt_canvas_data.json").read_text()).replace("__AUDIT__", (H / "data/gt_audit.json").read_text()).replace("__BAL__", (H / "data/balance.json").read_text())
(H / "gt_builder.html").write_text(t)
print("gt_builder.html MB", round((H / "gt_builder.html").stat().st_size / 1e6, 2))
