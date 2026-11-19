#!/usr/bin/env python3
"""gt_flow.html = flow_template.html + data/flow_data.json (build_flow_data.py)."""
from pathlib import Path
H = Path(__file__).resolve().parent
t = (H / "flow_template.html").read_text().replace("__FLOW__", (H / "data/flow_data.json").read_text())
(H / "gt_flow.html").write_text(t)
print("gt_flow.html MB", round((H / "gt_flow.html").stat().st_size / 1e6, 2))
