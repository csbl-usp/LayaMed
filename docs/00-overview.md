# 0. Overview

[Guide index](../README.md#step-by-step-guide) · [1. Environment →](01-environment.md)

## What LayaMed shows

While the case is being written, the interface shows:

- **evidence bars** for the 4 diseases: probability × case understanding (the bars start low and grow as history, exam and labs are filled in);
- a **result card**: the most likely disease with its probability, the management decision (surgical × conservative), the tokens read and the inference time;
- **warnings** when the text has terms the model never saw in training (disease name, imaging, post-operative finding) and what was done with them;
- **states without a diagnosis**: "waiting for minimum context" (text too short) and "not a clinical case" (scope detector).

## The pieces

| Piece | File | Role |
|---|---|---|
| Input format | `layamed/inference.py` | Builds the text exactly as in training: 3 sections with fixed titles, cleaning and masking |
| Model | `checkpoint/` | Fine-tuned Laya (mmBERT encoder + question head), loaded with `laya.Agent` |
| Questions | `inference.QUESTIONS` | Diagnosis (`choice`, 4 options) and management (`noul`, yes/no) |
| Scope detector | `layamed/scope.py` + `data/scope.pt` | Non-clinical text gets no probability |
| Recalibration | `layamed/recalibration.py` + `data/recalibration.json` | Optional: moderates the output while the case is incomplete |
| Service | `layamed/service.py` | Puts it all together: `analyze(hpi, pe, lab)` returns a plain-data dictionary |
| Server | `server.py` | FastAPI: serves the page and answers `POST /api/analyze` |
| Interface | `web/index.html` | Plain HTML/CSS/JS: single field, bars, card, example cases |

## The path of a text

1. **Interface:** on each pause in typing (120 ms), the page sends the field's text to `/api/analyze`. Late responses to older requests are discarded.
2. **Sections:** `inference.split_free_text` separates history, exam and labs by the titles the user types (`Physical examination:`, `Labs:`...). Without titles, everything is history.
3. **Cleaning:** `inference.build_text` applies the training-time cleaning: drops sentences that mention imaging, drops post-operative findings from the exam and replaces disease and surgery names with `[redacted]`. An empty section becomes `Not documented.`.
4. **Minimum context:** below 100 record tokens (~50 words), the model does not run.
5. **Scope:** the text vector in the Laya encoder must be close to the training cases, and the text must have clinical vocabulary. Otherwise: "not a clinical case".
6. **Laya:** one sequence per question (question + options + record, up to 2048 tokens) and one probability per option, with the calibrated temperature.
7. **Recalibration (if present):** mixes the output with the class frequencies, with a weight that grows with the text written.
8. **Interface:** draws the bars, the card and the warnings.

## Why an encoder and not an LLM

Laya **generates 0 tokens**: it reads everything at once and gives one probability per option. This allows:

- latency of tens of milliseconds, which makes updating while typing feasible;
- calibrated probabilities (the training loss is a proper scoring rule and a temperature is fitted on validation);
- running on a laptop GPU (the model takes ~1.2 GB of VRAM).

The price: the model is **closed on its training options**. It cannot say "another disease"; that is why the scope detector exists, and why cases of other diseases still get probabilities (see [09-limitations.md](09-limitations.md)).

---

[Guide index](../README.md#step-by-step-guide) · [1. Environment →](01-environment.md)
