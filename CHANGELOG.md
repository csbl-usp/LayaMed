# Changelog

## 0.1.0

First public version.

### Added
- Step-by-step guide (`docs/`) from data to interface, and an end-to-end Colab notebook trained on synthetic cases.
- Synthetic case generator for appendicitis, cholecystitis, pancreatitis and diverticulitis (complete and partial
  cases, no patient data).
- Out-of-scope detector: text that does not look like a clinical case gets no probability.
- Recalibration by the amount of text written, chosen on validation only.
- Local FastAPI server and single-page interface with live evidence bars, light and dark themes.
- Warnings for disease or surgery names, imaging, post-operative findings, organ names and misplaced section titles.
- Test suite without GPU and GitHub Actions workflow; `tools/capture_demo.py` for screenshots and a demo GIF.

### Security
- Linear post-operative clause removal (a crafted text could freeze the server).
- Host header check against DNS rebinding; private-network-only clients.
