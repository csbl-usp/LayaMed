# 2. Data

[← 1. Environment](01-environment.md) · [Guide index](../README.md#-step-by-step-guide) · [3. Fine-tuning →](03-fine-tuning.md)

## Input format

Laya reads a single text with 3 sections, always in this order and with these titles (`layamed.inference.build_text`):

```
HISTORY OF PRESENT ILLNESS:
52-year-old woman with known gallstones presents with 10 hours of constant epigastric pain...

PHYSICAL EXAMINATION:
T 37.6 C, HR 108, BP 128/80. Tender in the epigastrium and right upper quadrant...

PERTINENT / ABNORMAL LABS:
White Blood Cells: 13.9 K/uL, Lipase: 2280.0 IU/L, ...
```

- Empty section: `Not documented.` (history and exam) or `No abnormal labs reported.` (labs).
- Labs as a comma-separated `Name: value unit` list.
- English. mmBERT is multilingual, but the cleaning (regex) only knows English terms.

**Training and inference need the same format.** That is why the training notebook and the server import the same `layamed/inference.py`.

## Cleaning (`inference.clean_section`)

| Section | What is done | Why |
|---|---|---|
| history | drops sentences that mention imaging; replaces disease/surgery names with `[redacted]` | the question is the clinical diagnosis, without imaging; the disease name in the text would give the answer away |
| exam | drops post-operative clauses (incision, drain, dressing...) and imaging sentences; masks | a post-operative finding reveals that surgery happened |
| labs | masks only | |

The terms are in `LEAK_TERMS`, `_RE_IMAGING` and `_RE_POSTOP`. Adapt them if you change the diseases. `inference.audit_text` lists what was found, and the interface shows it as a warning. It also lists organ names of the 4 diseases (`appendix`, `gallbladder`, `pancreatic`, `diverticulosis`...) in `_RE_ORGAN`: they are **not** masked, because they are legitimate findings ("tender gallbladder"), but in real records they can give the answer away; check whether your training data has them and decide whether to mask them in both training and inference. The synthetic generator fails if a case has a term that is masked or dropped.

## Synthetic cases (this repository's example)

```powershell
python synthetic/generate_cases.py
```

Writes `data/synthetic/partial_cases_<disease>.csv`: 250 base cases per disease × 4 stages = 1,000 rows per disease.

| Column | Content |
|---|---|
| `case_id`, `row_id` | base case and row (`<case>_<stage>`) |
| `disease`, `label_id` | disease and its index in `inference.CLASSES` |
| `stage` | `hpi_partial` (start of the history), `hpi`, `hpi_pe`, `complete` |
| `split` | `train` / `val` / `test` (70/15/15), drawn **per base case** |
| `hpi`, `pe`, `lab` | the sections as free text |
| `model_text` | the text already in Laya's format |
| `mimics` | disease whose sentences the case "borrows" (so it is not too easy) |

The cases come from typical textbook clinical ranges, with sentences shared by the 4 diseases, ~50% generic openings, ~35% cases that mimic the most similar disease and ~25% atypical presentations. Even so, **they are much easier than real records**: they are for learning the path, not for measuring performance.

## Using your own data

1. Build a DataFrame with `hpi`, `pe`, `lab`, `label_id` and `split` (split by patient/admission, never by row).
2. Create `model_text` with `inference.build_text(hpi, pe, lab)`: the cleaning is applied there.
3. (Recommended) Create partial versions of each training case with the same label: history only, history + exam. `layamed.recalibration.prefixes(text)` cuts a case at fractions of its words, in the order it is written.
4. In the notebook, replace cell 2 with reading your file, keeping the columns `model_text`, `label_id`, `split`, `stage` and `surgical`.

If you change the diseases, edit `CLASSES`, `DISEASE_NAME`, `DISEASE_DESC` and the masking terms in `inference.py`. The description of each option goes into the model's sequence together with the text.

### Restricted-access data

If the data come from a database with a usage agreement (for example, MIMIC under the PhysioNet DUA):

- the **trained weights** and derived files (`data/scope.pt`, which stores the mean and covariance of the training case vectors) are as restricted as the data: no public repository, Hugging Face or open link;
- do not send case text to external LLM APIs;
- demo cases in the interface: hand-written or synthetic;
- check `.gitignore` before each commit: `checkpoint/` and `data/` are already excluded.

---

[← 1. Environment](01-environment.md) · [Guide index](../README.md#-step-by-step-guide) · [3. Fine-tuning →](03-fine-tuning.md)
