# 3. Fine-tuning

[← 2. Data](02-data.md) · [Guide index](../README.md#step-by-step-guide) · [4. Inference →](04-inference.md)

Everything is in cells 3a–3c of `notebooks/layamed_colab.ipynb`.

## The model

[Laya Multilingual](https://huggingface.co/convaiinnovations/laya) (`convaiinnovations/laya`, `multilingual` folder) is an mmBERT encoder (322 M parameters) with a head that reads **question + options + text** in a single sequence and returns one logit per option. The same network answers different questions, because the question is part of the input.

```
[CLS] question instruction [option 1] description ... [option k] description [SEP] record [SEP]
       └──────────────── head: up to 256 tokens ────────────────┘              └─ up to 2048 − head ─┘
```

`laya.common.build_sequence(tok, text, question, max_len=2048, head_max_len=256)` builds the sequence and the option markers. Inference (`laya.Agent`) uses the same function, so training and the app read the text the same way.

## The questions (`inference.QUESTIONS`)

```python
"diagnosis":  {"type": "choice",
               "criteria": {"appendicitis": "acute appendicitis (right lower quadrant pain, ...)", ...},
               "instructions": "Which is the primary acute abdominal diagnosis of this patient?"}
"management": {"type": "noul",
               "criteria": {"false": "No. Conservative management ...", "true": "Yes. ... surgical ..."},
               "instructions": "Does this patient require surgical intervention rather than conservative management?"}
```

- `choice`: one option out of k; probabilities sum to 1.
- `noul`: yes/no; `laya.Agent` returns P(yes) in `answers["management"]["noul"]`.
- Each training case becomes **one item per question**.

In the example, the management label is a teaching rule per disease (appendicitis and cholecystitis → surgical). With real data, use the recorded outcome (whether surgery or a procedure happened during the admission).

## Training items

```python
ids, markers = build_sequence(tok, text, inf.Q_INTERNAL[question], inf.MAX_LEN, inf.HEAD_MAX_LEN)
item = {"ids": ids, "markers": markers, "qtype": QTYPES[qtype],
        "target": [0.95 if i == correct else 0.05 / (k − 1)], "label": correct}
```

The target is smoothed (0.95 on the right answer) so the model does not learn to say 100%.

**Partial cases in training.** The example trains on the 4 stages of each case (start of the history, history, + exam, + labs), all with the true label. With a proper scoring rule, this teaches the model to give **moderate** probabilities when there is little information, which is what the interface shows while the case is being written. A model trained only on complete cases is overconfident after 2 sentences (see recalibration in [05](05-scope-and-recalibration.md)).

## Training loop

| Item | Value |
|---|---|
| Model | `build_model(cfg, encoder_dir, pretrained=False)` + `load_state_dict` of the base weights |
| Loss | `−proper_reward(softmax(logits), target, qtype, marker_mask)`: laya's strictly proper scoring rule |
| Optimizer | AdamW, lr 3e-5 for the encoder and 1e-4 for the head, weight decay 0.01, gradient clipping 1.0 |
| Batch | 8 (or 4 × accumulation 2 on a 16 GB GPU, with gradient checkpointing) |
| Epochs | 3; keeps the epoch with the **lowest diagnosis NLL on validation** |
| Precision | bf16 autocast (Ampere+) or fp16 with `GradScaler` (T4) |

Choosing the epoch by NLL instead of accuracy favors calibrated probabilities, which is what the interface shows.

## Temperature

After training, one temperature per question type (`choice`, `noul`) is chosen by **grid search** on validation: the T that minimizes the NLL of `softmax(logits / T)`. It goes into `rl_agent_config.json["temperature"]` at position `QTYPES[type]`, and `laya.Agent` applies it automatically at inference.

## Checkpoint

```
checkpoint/
├── model.safetensors        weights (~1.2 GB)
├── rl_agent_config.json     max_len, head_max_len, temperatures, amp_dtype
├── encoder/config.json
├── tokenizer/               tokenizer.json, tokenizer_config.json
└── training_meta.json       data, seed, epoch, history
```

It is the same format as the base model, so `laya.Agent("checkpoint")` loads it directly.

## Check

Cell 4 measures, on the test set, the accuracy and the mean probability of the right disease **by stage**. Expected: both grow from `hpi_partial` to `complete`. If complete-case accuracy is not well above chance (25%), check the GPU, the number of epochs and whether `QUICK` mode was left on.

---

[← 2. Data](02-data.md) · [Guide index](../README.md#step-by-step-guide) · [4. Inference →](04-inference.md)
