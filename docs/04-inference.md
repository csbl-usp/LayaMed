# 4. Inference in Python

[← 3. Fine-tuning](03-fine-tuning.md) · [Guide index](../README.md#step-by-step-guide) · [5. Scope and recalibration →](05-scope-and-recalibration.md)

```python
import torch                      # before numpy (see the OpenMP note in 01-environment.md)
from pathlib import Path
from layamed import inference as inf

model = inf.LayaInference(Path("checkpoint"), batch_size=8)

text = inf.build_text(
    hpi="19-year-old woman with 18 hours of periumbilical pain that moved to the right lower quadrant...",
    pe="T 38.0 C, HR 102. Focal tenderness in the right lower quadrant with rebound...",
    lab="White Blood Cells: 14.6 K/uL, Neutrophils: 84.0 %",
)
P, p_surgical = model.predict([text])
print(dict(zip(inf.CLASSES, P[0].round(3))), "P(surgical) =", round(p_surgical[0], 3))
```

## The `LayaInference` class

| Method | Returns |
|---|---|
| `predict(texts)` | `P` [n, 4] in `CLASSES` order and `p_surgical` [n] |
| `count_tokens(snippet)` | tokens of the snippet in the Laya tokenizer |
| `budget(text)` | per question: head tokens, record tokens, tokens used, tokens dropped, whether it was truncated |
| `embeddings(texts)` | record vector in the encoder [n, 768], normalized (used by the scope detector) |

Under the hood, `predict` calls:

```python
agent.predict_batch(texts, inf.QUESTIONS, batch_size=8, sort_by_length=True, max_len=2048, head_max_len=256)
# r["answers"]["diagnosis"]["probabilities"][disease]  and  r["answers"]["management"]["noul"]
```

## Free text → sections

The interface has a single field. `inf.split_free_text(text)` separates the sections by titles at the start of a line, followed by a colon (any case):

| Section | Accepted titles |
|---|---|
| history | `History of present illness:`, `History:`, `HPI:` |
| exam | `Physical examination:`, `Physical exam:`, `Exam:`, `On exam:`, `PE:` |
| labs | `Labs:`, `Lab:`, `Laboratory:`, `Pertinent / abnormal labs:` |

```python
s = inf.split_free_text("Pain since yesterday.\nExam: T 38.5, RLQ tenderness\nLabs: WBC 15")
# {'hpi': 'Pain since yesterday.', 'pe': 'T 38.5, RLQ tenderness', 'lab': 'WBC 15'}
text = inf.build_text(s["hpi"], s["pe"], s["lab"])
```

Without titles, everything goes to the history.

## Performance

On an RTX 4060 Laptop (bf16): ~40 ms per case, ~50 ms per full interface analysis (detector + Laya), 1.2 GB of VRAM. On Colab's T4, about 2× slower.

---

[← 3. Fine-tuning](03-fine-tuning.md) · [Guide index](../README.md#step-by-step-guide) · [5. Scope and recalibration →](05-scope-and-recalibration.md)
