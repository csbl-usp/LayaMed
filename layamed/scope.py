"""Out-of-scope detector: does the text look like a clinical case like the training ones?

Laya is a closed 4-class classifier: any text (a recipe, a news story) comes out as one of the 4 diseases, often
with 75%+ probability, and the model confidence does not separate them. Two independent signals; a text must pass
both:

1. Where the text falls in the Laya encoder space:
       vector   = mean of the encoder last layer over the record tokens (LayaInference.embeddings)
       distance = Mahalanobis distance from the vector to the training cases:  √((v − μ)ᵀ Σ⁻¹ (v − μ))
       Σ = training covariance + shrinkage · (trace/768) · I  (768 dimensions; shrinkage keeps Σ invertible)
   Alone, it lets a few long non-medical texts through.
2. Clinical vocabulary: how many CLINICAL_TERMS groups (symptoms, diseases, procedures, exam, vital signs, labs,
   units) appear in what was written, without section titles and "Not documented.".

Both thresholds come only from valid texts (validation and the interface examples), never from the test set or the
non-medical texts. Out of scope, the interface shows no probabilities.
Clinical cases of other diseases (myocardial infarction, pneumonia) sit close to the training cases and are NOT
blocked: the detector separates "clinical case" from "any text", not "one of the 4 diseases" from "another
disease" (that needs an "other" option in training).

The math uses torch (never numpy @ / np.linalg after torch; see the OpenMP note in docs/01-environment.md). The saved
file holds the mean and precision (Σ⁻¹) of the training vectors: if training used restricted data, the file is as
restricted as the weights.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch

from . import inference as inf

VERSION = 1
SHRINKAGE = 0.1
MARGIN = 1.10           # distance threshold = 1.10 × the largest distance among calibration texts
TERMS_QUANTILE = 0.001  # minimum clinical terms = 0.1% quantile of calibration (rounded down, ≥ 1)

# Clinical vocabulary in English (the training language): only terms that rarely appear outside medical texts
# (no "history", "patient", "reports", "soft"...). Each group counts once, however many times it appears.
_WORD_TERMS = [
    # symptoms and findings
    r"pain\w*|ache|aching", r"abdom\w*", r"epigastr\w*", r"quadrants?", r"periumbilic\w*|umbilic\w*", r"flank",
    r"suprapubic", r"tender\w*", r"guarding", r"rebound", r"distend\w*|distension|distention",
    r"nause\w*", r"vomit\w*|emesis", r"n/v\w*", r"diarrh\w*", r"constipat\w*", r"bowels?", r"stools?",
    r"fevers?|febrile|afebrile|pyrexi\w*", r"chills|rigors", r"anorexi\w*", r"jaundic\w*|icter\w*",
    r"murphy'?s?", r"rovsing'?s?", r"mcburney'?s?", r"psoas", r"obturator", r"periton\w*", r"radiat\w*",
    r"colick?y", r"dyspne\w*", r"hypox\w*", r"cough\w*", r"dysuria", r"hematuria", r"melena", r"hematochezia",
    r"hematemesis", r"syncop\w*", r"malaise", r"letharg\w*", r"diaphore\w*", r"tachycard\w*|bradycard\w*",
    r"tachypne\w*", r"hypotens\w*", r"hypertens\w*|htn", r"diabet\w*|t2dm|iddm|niddm", r"hyperlipid\w*|hld",
    r"ckd|esrd|dialysis", r"chf|copd|dvt|afib|a-fib|cva", r"obes\w*", r"alcohol\w*|etoh", r"tobacco|smok\w*",
    r"gallstones?|cholelithiasis|biliary|bile", r"colonoscop\w*|endoscop\w*",
    # procedures, diseases and treatment
    r"surg\w*|laparoscop\w*", r"\w+(?:ectomy|otomy|ostomy|plasty|scopy)", r"\w+(?:itis|emia|osis|algia|uria)",
    r"stent\w*", r"intubat\w*", r"transplant\w*", r"chemo\w*|oncolog\w*|cancer|carcinoma|adenoca\w*|metasta\w*",
    r"infect\w*|sepsis|septic", r"antibiotic\w*|abx|analgesi\w*|morphine|opioid\w*|dilaudid|ondansetron|zofran",
    r"tylenol|acetaminophen|ibuprofen|nsaids?|omeprazole|ppi", r"medications?|meds", r"hospitali\w*|discharg\w*",
    r"flagyl|metronidazole|cipro\w*|ceftriaxone|zosyn|piperacillin|levofloxacin|vancomycin|cefazolin|augmentin",
    r"fibrillation|arrhythm\w*|anticoag\w*|warfarin|coumadin|heparin|apixaban|eliquis",
    r"osh|pcp",
    r"pmh|psh|h/o|s/p|y/?o|yof|yom",
    # physical exam and vital signs
    r"palpat\w*", r"auscult\w*", r"percuss\w*", r"vital\s+signs?|vitals", r"mmhg", r"bpm", r"spo2|o2\s*sat\w*",
    # labs and units
    r"lipase", r"amylase", r"bilirubin|t\.?\s?bili", r"alkaline\s+phosphatase|alk\s*phos", r"aminotransferase\w*",
    r"white\s+blood\s+cells?|leukocyt\w*|wbc", r"neutrophil\w*|pmns?", r"hemoglobin|hgb", r"hematocrit|hct",
    r"platelets?|plt", r"creatinine", r"urea|bun", r"lactate|lactic", r"c-?reactive|crp", r"sodium", r"potassium",
    r"glucose", r"calcium", r"albumin", r"triglycerides?", r"urinalysis", r"inr|ptt", r"troponin",
    r"iu/l|u/l", r"k/ul|x\s?10\^?9", r"mg/dl", r"g/dl", r"meq/l", r"mmol/l", r"mg/l", r"ng/ml",
]
_ACRONYM_TERMS = [r"RUQ|RLQ|LUQ|LLQ", r"ALT|AST|GGT|ALP", r"HR|BP|RR", r"ED|ER|ICU", r"CBC|BMP|CMP|LFTs?",
                  r"T\s?\d{2}(?:[.,]\d)?\s?C"]
# Word boundary made of letters and digits only: "nausea/vomiting" counts both
_BEFORE, _AFTER = r"(?<![A-Za-z0-9])", r"(?![A-Za-z0-9])"
CLINICAL_TERMS = ([re.compile(rf"{_BEFORE}(?:{t}){_AFTER}", re.IGNORECASE) for t in _WORD_TERMS]
                  + [re.compile(rf"{_BEFORE}(?:{t}){_AFTER}") for t in _ACRONYM_TERMS])


def count_clinical_terms(model_text: str) -> int:
    """How many CLINICAL_TERMS groups appear in the written text (without section titles and "Not documented.")."""
    sections = inf.split_sections(model_text)
    written = " \n ".join(c for s, c in sections.items() if c != inf.EMPTY_SECTION[s])
    return sum(1 for r in CLINICAL_TERMS if r.search(written))


class ScopeDetector:
    def __init__(self, mean: torch.Tensor, precision: torch.Tensor, limit: float = float("inf"),
                 min_terms: int = 0, meta: dict | None = None):
        self.mean, self.precision, self.limit, self.meta = mean, precision, float(limit), meta or {}
        self.min_terms = int(min_terms)

    @classmethod
    def fit(cls, R_train: torch.Tensor, shrinkage: float = SHRINKAGE) -> "ScopeDetector":
        """Mean and precision of the training vectors (R_train [n, d], output of LayaInference.embeddings)."""
        R = R_train.float()
        mean = R.mean(0)
        X = R - mean
        cov = X.T @ X / (len(X) - 1)
        d = cov.shape[0]
        cov = cov + shrinkage * torch.trace(cov) / d * torch.eye(d, device=cov.device)
        return cls(mean, torch.linalg.inv(cov), meta={"n_train": len(R), "shrinkage": shrinkage})

    def distance(self, R: torch.Tensor) -> torch.Tensor:
        """Mahalanobis distance of each vector [n, d] to the training set; tensor [n] on the CPU."""
        diff = R.float().to(self.mean.device) - self.mean
        return ((diff @ self.precision) * diff).sum(1).clamp_min(0).sqrt().cpu()

    def calibrate(self, D_valid: torch.Tensor, terms_valid: list[int]) -> None:
        """Both thresholds from valid (in-scope) texts that were not used in fit."""
        self.limit = float(D_valid.max()) * MARGIN
        terms = torch.tensor(terms_valid, dtype=torch.float)
        self.min_terms = max(1, int(torch.quantile(terms, TERMS_QUANTILE)))
        self.meta.update({"margin": MARGIN, "max_calibration_distance": float(D_valid.max()),
                          "n_calibration": len(D_valid), "terms_quantile": TERMS_QUANTILE,
                          "min_terms_calibration": int(terms.min())})

    def evaluate(self, R: torch.Tensor, texts: list[str]) -> list[dict]:
        """Per text: distance, clinical terms and the out-of-scope reason (None = in scope)."""
        out = []
        for d, t in zip(self.distance(R).tolist(), texts):
            n = count_clinical_terms(t)
            reason = "few_terms" if n < self.min_terms else ("distance" if d > self.limit else None)
            out.append({"distance": d, "limit": self.limit, "terms": n, "min_terms": self.min_terms,
                        "out_of_scope": reason is not None, "reason": reason})
        return out

    def save(self, path: Path) -> None:
        torch.save({"version": VERSION, "mean": self.mean.cpu(), "precision": self.precision.cpu(),
                    "limit": self.limit, "min_terms": self.min_terms, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: Path, device=None) -> "ScopeDetector":
        d = torch.load(path, map_location=device or "cpu", weights_only=True)
        if d.get("version") != VERSION:
            raise ValueError(f"{path} is version {d.get('version')}; build it again (notebook, part 5)")
        return cls(d["mean"], d["precision"], d["limit"], d["min_terms"], d["meta"])
