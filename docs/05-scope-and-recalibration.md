# 5. Scope detector and recalibration

## The problem

Laya is closed on the 4 diseases: **any** text comes out as one of them. A shopping list can get "cholecystitis 85%", and the model's confidence does not help to tell them apart (non-clinical text often comes out with high probability). That is why the interface only shows a probability if the text passes the detector.

## Detector (`layamed/scope.py`)

The text must pass **both** signals:

1. **Distance in the encoder.** `LayaInference.embeddings` gives the record vector (mean of the encoder's last layer over the record tokens, with the diagnosis question in front, as in training). The Mahalanobis distance to the training cases is

   `d(v) = √((v − μ)ᵀ Σ⁻¹ (v − μ))`, with `Σ = covariance + 0.1 · (trace/768) · I`.

   The shrinkage (0.1) keeps the matrix invertible with few cases.
2. **Clinical vocabulary.** How many of the ~115 `CLINICAL_TERMS` groups appear in the written text (symptoms, exam, vital signs, labs, units, medications). Each group counts once.

The thresholds come **only from valid texts** that were not used in the fit:

```python
scope = ScopeDetector.fit(model.embeddings(train_texts))                      # μ and Σ⁻¹
scope.calibrate(scope.distance(model.embeddings(valid_texts)),                 # threshold = 1.10 × largest distance
                [count_clinical_terms(t) for t in valid_texts])                # minimum = 0.1% quantile of terms
scope.save("data/scope.pt")
```

In the notebook, the valid texts are the validation set (all stages) and prefixes of the interface's example cases. The non-medical texts in `synthetic/out_of_scope.py` are **only used to check** (cell 5). If you choose the thresholds by looking at the negatives, you need a second set of negatives, written afterwards, to really measure.

Expected result: ~0% of valid texts blocked, ~100% of non-medical texts blocked. In the interface: a "Not a clinical case / No diagnosis" card, with the reason (too few clinical terms or too far from the training cases).

**Known limitation:** clinical cases of **other** diseases (myocardial infarction, pneumonia, stroke) pass the detector and get probabilities for the 4 diseases. The detector separates "clinical case" from "any text". To separate "another disease", train a 5th option in the diagnosis question (for example `"other": "none of these four"`) with examples of other diseases and of non-clinical texts.

## Minimum context

`service.MIN_CONTEXT_TOKENS = 100`. Below it Laya does not run: the interface shows how many tokens are missing. The empty case already has ~32 tokens (titles and `Not documented.`), so 100 tokens ≈ 50 typed words. With less text, the detector does not separate well.

## Recalibration (`layamed/recalibration.py`, optional)

A model trained only on complete cases is overconfident with little text (it falls on the most frequent class with ~90%). Recalibration does not touch the model: it mixes the output with the training class frequencies, with a weight that grows as the case is written:

```
w        = sigmoid(a + b·log(1 + typed tokens) + e·[exam written] + g·[labs written])
logit_k  = w · log P_k + (1 − w) · log freq_k
P'       = softmax(logit)
```

The 4 parameters are fitted by maximum likelihood on **prefixes** (0% to 100% of the words, in the order history → exam → labs) of the validation cases, with the true label:

```python
texts = [p for t in val_complete for p in rc.prefixes(t)]
recal = rc.Recalibrator.fit(P_laya, tokens, texts, y, freq, n_empty)
recal.save("data/recalibration.json")
```

The notebook only saves the file if the NLL of the **test** prefixes improves. The example already trains on partial cases, so the gain is small; with a model trained only on complete cases it is large. Without the file, the interface uses the Laya output as is.

Next: [6. Service and server](06-server.md).
