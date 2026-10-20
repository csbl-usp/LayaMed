"""Text preparation and inference for a Laya model fine-tuned on 4 acute abdominal diseases (input without imaging).

Training and inference must use exactly the same input format: same sections, same titles, same masking and the
same questions (QUESTIONS). Any difference makes the model read a text unlike the ones it saw in training. That is
why the training notebook imports this same module.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ==============================================================================
# INPUT FORMAT
# ==============================================================================
CLASSES = ["appendicitis", "cholecystitis", "pancreatitis", "diverticulitis"]

MAX_LEN = 2048          # token budget used in training and inference
HEAD_MAX_LEN = 256      # budget of the "head" (question + options)
MASK_TOKEN = "[redacted]"

# Input sections, in text order (no imaging section)
SECTION_TITLES = {"hpi": "HISTORY OF PRESENT ILLNESS", "pe": "PHYSICAL EXAMINATION", "lab": "PERTINENT / ABNORMAL LABS"}
SECTIONS = ["hpi", "pe", "lab"]
EMPTY_SECTION = {"hpi": "Not documented.", "pe": "Not documented.", "lab": "No abnormal labs reported."}

# ==============================================================================
# LAYA QUESTIONS
# ==============================================================================
DISEASE_NAME = {
    "appendicitis": "acute appendicitis",
    "cholecystitis": "acute cholecystitis",
    "pancreatitis": "acute pancreatitis",
    "diverticulitis": "acute colonic diverticulitis",
}
DISEASE_DESC = {
    "appendicitis": "right lower quadrant pain, periumbilical pain radiating to RLQ, localized tenderness at McBurney's point",
    "cholecystitis": "right upper quadrant or epigastric pain, positive Murphy's sign, fever, or elevated biliary enzymes",
    "pancreatitis": "acute epigastric pain radiating to the back, serum lipase or amylase elevated > 3x upper limit of normal",
    "diverticulitis": "left lower quadrant pain, fever, leukocytosis, or change in bowel habits",
}
QUESTIONS = {
    "diagnosis": {
        "type": "choice",
        "criteria": {c: f"{DISEASE_NAME[c]} ({DISEASE_DESC[c]})" for c in CLASSES},
        "instructions": "Which is the primary acute abdominal diagnosis of this patient?",
    },
    "management": {
        "type": "noul",
        "criteria": {
            "false": "No. Conservative management (antibiotics, fluids, bowel rest, observation) is appropriate.",
            "true": "Yes. The patient requires surgical or procedural intervention.",
        },
        "instructions": "Does this patient require surgical intervention rather than conservative management?",
    },
}
# Internal format built by laya.Agent (Agent._to_internal); used to build sequences and count tokens
Q_INTERNAL = {
    k: {"t": q["type"], "ins": q["instructions"], "crit": {str(c).lower(): v for c, v in q["criteria"].items()}}
    for k, q in QUESTIONS.items()
}

# ==============================================================================
# TEXT CLEANING
# ==============================================================================
# Disease and surgery names: they would give the answer away, so they are masked
LEAK_TERMS = [
    r"appendicit\w*", r"appendectom\w*", r"\bappy\b",
    r"cholecystit\w*", r"cholecystectom\w*", r"\bchole\b",
    r"pancreatit\w*",
    r"diverticulit\w*", r"sigmoidectom\w*", r"colectom\w*", r"\bhartmann'?s?\b",
]
_RE_DIAG = re.compile("|".join(LEAK_TERMS), re.IGNORECASE)
_RE_IMAGING = re.compile(
    r"\b(?:CTA?|CAT\s+scan|computed\s+tomography|ultrasound\w*|ultrasonograph\w*|sonogra\w*|u/s|RUQ\s*US|RUQUS|"
    r"MRI|MRCP|HIDA|ERCP|imaging|scans?|radiograph\w*|x-?rays?|KUB)\b", re.IGNORECASE)
_RE_US = re.compile(r"\bUS\b")
_RE_POSTOP = (r"incision\w*|\bc/d/i\b|\bcdi\b|staple\w*|steri-?strip\w*|dressing\w*|\bJP\b|\bdrains?\b|"
              r"\bport\s+sites?\b|post-?op\w*|\bs/p\b|surgical\s+site\w*|\bwound\w*")
_RE_POSTOP_TEST = re.compile(_RE_POSTOP, re.IGNORECASE)
_RE_CLAUSE_END = re.compile(r"([,;.\n])")
# Organ names of the 4 diseases: not masked (they are legitimate findings, e.g. "tender gallbladder"), but they can
# give the answer away, so the interface warns about them
_RE_ORGAN = re.compile(r"\b(?:appendi(?:x|ces|ceal)|gall\s?bladder|pancrea(?:s|tic)|diverticul(?:a|um|ar|osis))\b",
                       re.IGNORECASE)
_RE_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")


def _has_imaging(t: str) -> bool:
    return bool(_RE_IMAGING.search(t) or _RE_US.search(t))


def _tidy(t: str) -> str:
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"[ \t]+([,;.])", r"\1", t)
    t = re.sub(r"([,;])[ \t]*(?=[,;.]|$)", "", t, flags=re.MULTILINE)
    return t.strip()


def _drop_imaging_sentences(t: str) -> str:
    return " ".join(s for s in _RE_SENTENCES.split(t) if s.strip() and not _has_imaging(s))


def _drop_postop_clauses(t: str) -> str:
    """Replaces each clause (text between , ; . or line break) that has a post-operative term, together with the
    punctuation that closes it, by a space; line breaks are kept. Linear in the text length (a single regex with
    [^,;.\\n]* around the terms was quadratic and froze the server on long texts without punctuation)."""
    pieces = _RE_CLAUSE_END.split(t)  # [clause, separator, clause, separator, ..., clause]
    out = []
    for i in range(0, len(pieces), 2):
        clause, sep = pieces[i], (pieces[i + 1] if i + 1 < len(pieces) else "")
        if _RE_POSTOP_TEST.search(clause):
            out.append(" \n" if sep == "\n" else " ")
        else:
            out.append(clause + sep)
    return "".join(out)


def _mask(t: str) -> str:
    return _RE_DIAG.sub(MASK_TOKEN, t)


def clean_section(text: Optional[str], section: str) -> str:
    """Applies the training-time cleaning to one section ('hpi', 'pe' or 'lab').

    hpi: drops sentences that mention imaging and masks disease/surgery names.
    pe:  drops post-operative clauses and imaging sentences, and masks.
    lab: masks only (so a disease name typed in the labs never reaches the model).
    """
    t = text if isinstance(text, str) else ""
    if section == "lab":
        return _mask(t.strip()) or EMPTY_SECTION["lab"]
    if section == "pe":
        t = _drop_postop_clauses(t)
    t = _drop_imaging_sentences(t)
    t = _mask(t)
    return _tidy(t) or EMPTY_SECTION[section]


def build_text(hpi: str, pe: str, lab: str, clean: bool = True) -> str:
    """Builds the model input in the training format: 'TITLE:\\ncontent' blocks separated by a blank line."""
    parts = {"hpi": hpi, "pe": pe, "lab": lab}
    content = {s: (clean_section(parts[s], s) if clean else (parts[s] or "").strip() or EMPTY_SECTION[s])
               for s in SECTIONS}
    return "\n\n".join(f"{SECTION_TITLES[s]}:\n{content[s]}" for s in SECTIONS)


def split_sections(model_text: str) -> Dict[str, str]:
    """Inverse of build_text: the content of each section of a model input."""
    pattern = "|".join(re.escape(SECTION_TITLES[s]) for s in SECTIONS)
    pieces = re.split(rf"(?m)^({pattern}):\n", model_text)
    inv = {v: k for k, v in SECTION_TITLES.items()}
    out = {s: "" for s in SECTIONS}
    for title, body in zip(pieces[1::2], pieces[2::2]):
        out[inv[title]] = body.strip()
    return out


# Titles the user may type in the single text field (start of a line, followed by a colon; any case).
# The content may follow on the same line ("Exam: T 38.2 C...") or on the next lines.
_RE_FREE_TITLE = re.compile(
    r"(?im)^[ \t]*(?:(?P<hpi>history of present illness|history|hpi)"
    r"|(?P<pe>physical examination|physical exam|examination|exam|on examination|on exam|pe)"
    r"|(?P<lab>pertinent\s*/\s*abnormal labs|laboratory results|laboratory|labs|lab))[ \t]*:[ \t]*")


def split_free_text(text: str) -> Dict[str, str]:
    """Splits the single text field into history, exam and labs using the titles the user typed.

    Without titles, everything is history. Text before the first title is history too. A repeated title appends
    to the same section.
    """
    out = {s: [] for s in SECTIONS}
    current, start = "hpi", 0
    for m in _RE_FREE_TITLE.finditer(text or ""):
        out[current].append(text[start:m.start()])
        current, start = m.lastgroup, m.end()
    out[current].append((text or "")[start:])
    return {s: "\n".join(t.strip() for t in parts if t.strip()) for s, parts in out.items()}


def audit_text(text: str) -> Dict[str, List[str]]:
    """Terms the model never saw in training (shown as warnings in the interface, before cleaning)."""
    return {
        "disease_or_surgery_name": sorted({m.group(0) for m in _RE_DIAG.finditer(text)}),
        "imaging": sorted({m.group(0) for m in _RE_IMAGING.finditer(text)} | {m.group(0) for m in _RE_US.finditer(text)}),
        "postop_finding": sorted({m.group(0) for m in _RE_POSTOP_TEST.finditer(text)}),
        "organ_name": sorted({m.group(0) for m in _RE_ORGAN.finditer(text)}),
    }


# A title after the end of a sentence ("... pain. Exam: soft") does not start a section: split_free_text only reads
# titles at the start of a line. The interface warns so the text is not silently read as history.
_RE_INLINE_TITLE = re.compile(
    r"(?i)(?<=[.;])[ \t]+(history of present illness|hpi|physical examination|physical exam|exam|"
    r"laboratory results|labs|lab)[ \t]*:")


def inline_titles(text: str) -> List[str]:
    """Section titles typed in the middle of a line (ignored by split_free_text)."""
    return sorted({m.group(1) for m in _RE_INLINE_TITLE.finditer(text or "")})


# ==============================================================================
# MODEL
# ==============================================================================
def find_checkpoint(root: Path) -> Path:
    """Finds the fine-tuned checkpoint folder (the one that contains rl_agent_config.json).

    Priority: the LAYAMED_CHECKPOINT environment variable, then `checkpoint/`, then the only folder with
    rl_agent_config.json under `root`.
    """
    if os.environ.get("LAYAMED_CHECKPOINT"):
        return Path(os.environ["LAYAMED_CHECKPOINT"])
    root = Path(root)
    if (root / "rl_agent_config.json").exists():
        return root
    if (root / "checkpoint" / "rl_agent_config.json").exists():
        return root / "checkpoint"
    candidates = sorted(p.parent for p in root.rglob("rl_agent_config.json"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Checkpoint not found (or ambiguous) under {root}: {candidates}")


class LayaInference:
    """Loads the checkpoint once and exposes batch prediction, token counts and the token budget."""

    def __init__(self, checkpoint_dir: Path, device: Optional[str] = None, batch_size: int = 8):
        import laya  # imported here so the text functions work without the package

        self.checkpoint_dir = Path(checkpoint_dir)
        for f in ["model.safetensors", "rl_agent_config.json", "encoder/config.json",
                  "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]:
            if not (self.checkpoint_dir / f).exists():
                raise FileNotFoundError(f"Missing {f} in {self.checkpoint_dir}")
        self.agent = laya.Agent(str(self.checkpoint_dir), device=device)
        self.tok = self.agent.tok
        self.batch_size = batch_size

    @property
    def device(self) -> str:
        return str(self.agent.device)

    def predict(self, texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (P_diag [n, 4] in CLASSES order, p_surgical [n])."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, len(CLASSES))), np.zeros(0)
        res = self.agent.predict_batch(
            texts, QUESTIONS, batch_size=self.batch_size,
            sort_by_length=len(texts) > self.batch_size,
            max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN,
        )
        P, ps = [], []
        for r in res:
            d = r["answers"]["diagnosis"]
            probs = d.get("probabilities", d.get("probs"))
            if not isinstance(probs, dict):
                raise KeyError(f"'choice' answer without probabilities; keys: {list(d)}")
            P.append([probs[c] for c in CLASSES])
            ps.append(r["answers"]["management"]["noul"])
        return np.asarray(P, dtype=float), np.asarray(ps, dtype=float)

    def count_tokens(self, text: str) -> int:
        """Tokens the snippet takes in the Laya tokenizer (without special tokens)."""
        if not text:
            return 0
        return len(self.tok(text.replace(self.tok.mask_token, " "), add_special_tokens=False)["input_ids"])

    def embeddings(self, texts: Sequence[str], batch: int = 16):
        """Record vector in the Laya encoder: mean of the last layer over the record tokens, with the diagnosis
        question head in front (the same sequence as in training), L2-normalized. Returns a torch tensor [n, 768]
        (float32, on the model device); vector math stays in torch (see the OpenMP note in docs/01-environment.md)."""
        import torch
        from laya.common import build_sequence

        seqs, spans = [], []
        for t in texts:
            ids, _, st = build_sequence(self.tok, t, Q_INTERNAL["diagnosis"], MAX_LEN, HEAD_MAX_LEN,
                                        return_truncation_stats=True)
            start = len(ids) - st["state_tokens_used"] - 1  # head | record | [SEP]
            seqs.append(ids)
            spans.append((start, start + st["state_tokens_used"]))
        encoder, dev = self.agent.model.encoder, self.agent.device
        # bf16 on GPUs that support it (Ampere or newer: A100, L4, RTX 30/40); fp16 on the others (Colab T4)
        dt = torch.bfloat16 if dev.type == "cuda" and torch.cuda.get_device_capability(dev)[0] >= 8 else torch.float16
        out = torch.zeros(len(seqs), encoder.config.hidden_size, device=dev)
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))  # batches of similar length
        for k in range(0, len(order), batch):
            idx = order[k:k + batch]
            n_max = max(len(seqs[i]) for i in idx)
            ids = torch.full((len(idx), n_max), self.tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(idx), n_max), dtype=torch.long)
            weights = torch.zeros((len(idx), n_max))
            for j, i in enumerate(idx):
                ids[j, :len(seqs[i])] = torch.tensor(seqs[i])
                mask[j, :len(seqs[i])] = 1
                a, b = spans[i]
                weights[j, a:max(b, a + 1)] = 1  # empty record: use the [SEP]
            with torch.inference_mode(), torch.autocast(dev.type, dtype=dt, enabled=dev.type == "cuda"):
                h = encoder(input_ids=ids.to(dev), attention_mask=mask.to(dev)).last_hidden_state.float()
            weights = weights.to(dev)
            v = (h * weights[:, :, None]).sum(1) / weights.sum(1, keepdim=True)
            out[idx] = torch.nn.functional.normalize(v, dim=1)
        return out

    def budget(self, text: str) -> Dict[str, Dict[str, int]]:
        """Per question: head tokens, record tokens, tokens used, tokens dropped and the full sequence length."""
        from laya.common import build_sequence

        out = {}
        for k, q in Q_INTERNAL.items():
            ids, _, st = build_sequence(self.tok, text, q, MAX_LEN, HEAD_MAX_LEN, return_truncation_stats=True)
            out[k] = {
                "sequence_tokens": len(ids),
                "head_tokens": len(ids) - st["state_tokens_used"] - 1,
                "record_tokens": st["state_tokens"],
                "record_tokens_used": st["state_tokens_used"],
                "tokens_dropped": st["state_tokens_dropped"],
                "truncated": bool(st["truncated"]),
            }
        return out
