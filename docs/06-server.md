# 6. Service and server

[← 5. Scope and recalibration](05-scope-and-recalibration.md) · [Guide index](../README.md#step-by-step-guide) · [7. Interface →](07-interface.md)

## `layamed/service.py`

All the logic lives in one class, separate from the interface. Any front end (the HTML page, a Gradio app, a script) can use the same analysis.

```python
from layamed.service import Service
s = Service(Path("."))                 # loads checkpoint/, data/scope.pt and data/recalibration.json (if present)
r = s.analyze(hpi, pe, lab)            # plain-data dictionary
```

`Service.__init__` loads the model once, runs a warm-up inference and **refuses to start without `data/scope.pt`**: without the detector, any text would get a probability.

`analyze` follows this order:

| Step | Cost | State returned |
|---|---|---|
| empty text | — | `empty` |
| record tokens < 100 | tokenizer only | `waiting` (+ `record_tokens`, `min_tokens`) |
| scope detector | encoder | `out_of_scope` (+ `scope`: distance, limit, terms, reason) |
| Laya + recalibration | encoder + head | `result` |

A `threading.Lock` ensures one analysis at a time on the GPU, even with several browsers open.

Fields of a `result`:

| Field | Content |
|---|---|
| `probabilities` | per disease, recalibrated (or original, without recalibration) |
| `original` | Laya output without recalibration |
| `leader`, `original_leader` | most likely disease in each |
| `understanding` | 1% to 100%: text size × sections filled in (a display scale, not a probability) |
| `evidence` | probability × understanding (what the bars show) |
| `p_surgical` | P(surgical management) |
| `budget` | head tokens, record tokens, tokens dropped, whether it was truncated |
| `warnings` | terms to check: never seen in training (disease or surgery name, imaging, post-operative), organ names that can give the answer away (kept), and section titles in the middle of a line (`inline_title`, ignored; added by `server.py`) |
| `model_text` | the exact text Laya read, after cleaning |
| `ms` | detector + Laya time |

**Case understanding:** `size = (tokens − 100) / (260 − 100)` clipped to [0, 1]; `sections = (1 + exam + labs) / 3`, each section full at 30 tokens; `understanding = max(1%, size × sections)`. It makes the bars start near zero and grow as the case is written, without hiding the probability (shown on the card and in the tooltip).

## `server.py` (FastAPI)

| Route | What |
|---|---|
| `GET /` | `web/index.html` (no cache) |
| `GET /logo.svg` | logo |
| `GET /api/config` | classes, labels, limits and the example cases (already in the single-field format) |
| `POST /api/analyze` | `{"text": "..."}` (up to 60,000 characters) → `split_free_text` → `Service.analyze` + `sections_read` |

```powershell
python server.py
```

- Listens on `127.0.0.1:8000` only. Another port: `$env:PORT = "8001"` before the command.
- `/docs`, `/redoc` and `/openapi.json` are disabled.
- `analyze` is synchronous on purpose: FastAPI runs it in a worker thread, and the service `Lock` serializes the GPU.
- A middleware refuses (403) clients outside private or loopback IPs. See `--lan` mode in [08-run-locally.md](08-run-locally.md).
- The `Host` header must name this machine (`127.0.0.1`, `localhost` and, with `--lan`, its network IPs and name); otherwise 400. This blocks DNS rebinding (a web page that makes the browser talk to the local server). `LAYAMED_ALLOWED_HOSTS` replaces the list (comma-separated; `*` accepts any, as the Colab notebook does behind its proxy).

Test without the interface (titles must start a line; in PowerShell, `` `n `` is a line break):

```powershell
$body = @{ text = "History: 19-year-old woman with 18 hours of periumbilical pain moving to the right lower quadrant, nausea and fever.`nExam: T 38.0 C, HR 102, focal right lower quadrant tenderness with rebound and guarding.`nLabs: White Blood Cells: 14.6 K/uL, Neutrophils: 84.0 %" } | ConvertTo-Json
```

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/analyze -Method Post -ContentType "application/json" -Body $body
```

---

[← 5. Scope and recalibration](05-scope-and-recalibration.md) · [Guide index](../README.md#step-by-step-guide) · [7. Interface →](07-interface.md)
