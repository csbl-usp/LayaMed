# 9. Limitations

[← 8. Run on your computer](08-run-locally.md) · [Guide index](../README.md#step-by-step-guide)

## What LayaMed does not do

- **It is not a medical device** and must not guide clinical decisions.
- **Closed on 4 diseases.** A case of myocardial infarction, pneumonia or cystitis passes the detector (it is a clinical case) and gets probabilities for the 4 diseases. Fixing this needs a 5th option trained on other diseases.
- **Probabilities are calibrated only for the training format.** Text with another structure, language or writing style may come out poorly calibrated. The cleaning regex only knows English.
- **No imaging.** Sentences that mention imaging are removed before the model.
- **The Colab example uses rule-based cases.** High accuracy on the synthetic test says nothing about real records; the example's management label is a rule per disease.
- **A simple detector.** An ordinary text containing a sentence about abdominal pain may pass (it is, after all, a report of abdominal pain).

## Data and privacy

- Everything runs locally: the typed text only goes from the browser to your server.
- Trained on restricted data? Then the weights (`checkpoint/`) and `data/scope.pt` are restricted too. This repository's `.gitignore` already keeps `checkpoint/` and `data/` out of git; check before each `git push`.
- Do not send real cases to external LLM APIs (to generate data, translate or review) if the usage agreement forbids it.
- Demo cases: hand-written or synthetic.

## Licenses

- Base model: [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya). Check its license before redistributing derived weights.
- `laya` package (PyPI): see the package license.
- This repository: code and guide; the example cases are fictional.

---

[← 8. Run on your computer](08-run-locally.md) · [Guide index](../README.md#step-by-step-guide)
